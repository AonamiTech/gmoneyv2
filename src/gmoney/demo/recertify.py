from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.demo.store import JobStore, JobTransactionError
from gmoney.extraction.validation import validate_extraction_result

app = typer.Typer(no_args_is_help=False)
PLAN_VERSION = "gmoney_recertification_plan_v1"
REPORT_VERSION = "gmoney_recertification_report_v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def plan_recertification(
    data_root: Path,
    job_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Inspect v5 workspaces without recovery, migration, or publication writes."""

    store = JobStore(data_root)
    entries: list[dict[str, Any]] = []
    directories = sorted(
        directory
        for directory in store.jobs_root.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    for directory in directories:
        job_id = directory.name
        if job_ids is not None and job_id not in job_ids:
            continue
        entry: dict[str, Any] = {"job_id": job_id, "eligibility": "ineligible"}
        try:
            with store.job_lock(job_id, exclusive=False):
                store._require_stable_workspace(job_id)
                state = store._read_state_unlocked(job_id)
                result_bytes = store._regular_file_bytes(
                    directory / "result.json", "invalid_live_result"
                )
                result = json.loads(result_bytes)
                if not isinstance(result, dict):
                    raise JobTransactionError("invalid_live_result")
                entry.update(
                    status=state.get("status"),
                    output_version=result.get("output_version"),
                    contract_revision=result.get("contract_revision"),
                    result_sha256=_sha256(result_bytes),
                    approval_will_be_cleared=bool(
                        store._read_review_unlocked(job_id).get("approval")
                    ),
                )
                if result.get("output_version") != "offline_accuracy_spine_v5":
                    entry["reason"] = "not_v5"
                elif result.get("contract_revision") == 2 and bool(
                    (result.get("recovery") or {}).get("attempted")
                ):
                    entry["eligibility"] = "reprocess_required"
                    entry["reason"] = "revision_2_recovery_audit_incomplete"
                elif result.get("contract_revision") not in {2, 3}:
                    entry["reason"] = "unsupported_contract_revision"
                else:
                    candidate = json.loads(json.dumps(result))
                    candidate.pop("semantic_validation", None)
                    report = validate_extraction_result(
                        directory / "source.pdf",
                        candidate,
                        directory / "artifacts",
                    )
                    entry["validation_status"] = report.status.value
                    entry["validation_issue_codes"] = list(
                        dict.fromkeys(issue.code for issue in report.issues)
                    )
                    if report.fatal:
                        entry["reason"] = "fatal_validation_failure"
                    else:
                        report_json = report.model_dump(mode="json")
                        candidate["semantic_validation"] = report_json
                        serialized = (
                            json.dumps(candidate, indent=2, sort_keys=True) + "\n"
                        ).encode()
                        report_payload = json.dumps(
                            report_json, sort_keys=True, separators=(",", ":")
                        ).encode()
                        certification = store._certification_v2_unlocked(
                            job_id,
                            candidate,
                            serialized,
                            report_json,
                            report_payload,
                        )
                        entry.update(
                            eligibility="eligible",
                            target_certification_sha256=certification["certification_sha256"],
                        )
        except (
            JobTransactionError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            entry["reason"] = str(error) or type(error).__name__
        entries.append(entry)
    if job_ids is not None:
        found = {entry["job_id"] for entry in entries}
        entries.extend(
            {
                "job_id": job_id,
                "eligibility": "ineligible",
                "reason": "job_not_found",
            }
            for job_id in sorted(job_ids - found)
        )
    payload: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "created_at": _now(),
        "data_root": str(data_root.resolve()),
        "jobs": entries,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def apply_recertification(data_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("unsupported recertification plan")
    supplied_digest = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if supplied_digest != _canonical_sha256(unsigned):
        raise ValueError("recertification plan digest mismatch")
    if Path(str(plan.get("data_root"))).resolve() != data_root.resolve():
        raise ValueError("recertification plan targets another data root")

    store = JobStore(data_root)
    outcomes: list[dict[str, Any]] = []
    for entry in plan.get("jobs") or ():
        if entry.get("eligibility") != "eligible":
            outcomes.append(
                {
                    "job_id": entry.get("job_id"),
                    "status": "skipped",
                    "reason": entry.get("reason") or entry.get("eligibility"),
                }
            )
            continue
        job_id = str(entry["job_id"])
        result_path = store.job_dir(job_id) / "result.json"
        live_bytes = store._regular_file_bytes(result_path, "invalid_live_result")
        live_digest = _sha256(live_bytes)
        if live_digest != entry.get("result_sha256"):
            outcomes.append({"job_id": job_id, "status": "conflict", "reason": "result_changed"})
            continue
        candidate = json.loads(live_bytes)
        candidate.pop("semantic_validation", None)
        published = store.publish_maintenance_result(
            job_id,
            candidate,
            expected_result_sha256=live_digest,
            expected_certification_sha256=str(
                entry["target_certification_sha256"]
            ),
        )
        state = store.read(job_id)
        outcomes.append(
            {
                "job_id": job_id,
                "status": "recertified" if published else "skipped",
                "certification_sha256": (
                    (state.get("certification") or {}).get("certification_sha256")
                    if published
                    else None
                ),
            }
        )
    return {
        "report_version": REPORT_VERSION,
        "applied_at": _now(),
        "plan_sha256": supplied_digest,
        "jobs": outcomes,
    }


@app.command()
def main(
    data_root: Annotated[Path, typer.Option("--data-root")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    job_id: Annotated[list[str] | None, typer.Option("--job-id")] = None,
    apply_from: Annotated[Path | None, typer.Option("--apply-from")] = None,
) -> None:
    if apply_from is None:
        if output is None:
            raise typer.BadParameter("--output is required for a dry-run plan")
        payload = plan_recertification(data_root, set(job_id) if job_id else None)
    else:
        if job_id:
            raise typer.BadParameter("--job-id cannot be combined with --apply-from")
        payload = apply_recertification(
            data_root,
            json.loads(apply_from.read_text()),
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(encoded, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)


if __name__ == "__main__":
    app()
