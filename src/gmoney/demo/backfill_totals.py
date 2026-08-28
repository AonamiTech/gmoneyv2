from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.contracts.extraction import RawTotalCandidate
from gmoney.demo.store import JobStore, JobTransactionError
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
    DocumentTotalCandidate,
    select_document_total,
    select_document_totals,
)

app = typer.Typer(add_completion=False, invoke_without_command=True)


def backfill_totals(*, root: Path, apply: bool = False) -> dict[str, Any]:
    store = JobStore(root)
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "updated": 0,
        "staged": 0,
        "would_update": 0,
        "already_current": 0,
        "not_found": 0,
        "failed": 0,
        "failures": [],
    }
    for state in store.states():
        if state.get("status") not in {"complete", "needs_review"}:
            continue
        job_id = str(state["id"])
        try:
            with store.job_lock(job_id, exclusive=True):
                store._recover_publication_unlocked(job_id)
                store._require_stable_workspace(job_id)
                locked_state = store._read_state_unlocked(job_id)
                if locked_state.get("status") not in {"complete", "needs_review"}:
                    continue
                result_path = store.job_dir(job_id) / "result.json"
                store._require_regular_file(result_path, "invalid_live_result")
                result_bytes = result_path.read_bytes()
                result_digest = hashlib.sha256(result_bytes).hexdigest()
                result = json.loads(result_bytes)
            if (
                result.get("document_total_version") == DOCUMENT_TOTAL_VERSION
                and result.get("document_totals_version") == DOCUMENT_TOTALS_VERSION
            ):
                summary["already_current"] += 1
                continue
            if (
                result.get("output_version") != "offline_accuracy_spine_v5"
                or result.get("contract_revision") != 2
                or not isinstance(result.get("raw_total_candidates"), list)
            ):
                summary["failed"] += 1
                summary["failures"].append(
                    {"job_id": job_id, "error": "requires_full_reprocess"}
                )
                continue
            if not apply:
                summary["would_update"] += 1
                continue
            candidates = []
            for payload in result["raw_total_candidates"]:
                raw = RawTotalCandidate.model_validate(payload)
                candidates.append(
                    DocumentTotalCandidate(
                        total=raw.total,
                        label_priority=raw.label_priority,
                        vertical_position=raw.vertical_position,
                        local_context=raw.local_context,
                    )
                )
            totals = select_document_totals(candidates)
            primary = select_document_total(candidates)
            result["document_total_version"] = DOCUMENT_TOTAL_VERSION
            result["document_totals_version"] = DOCUMENT_TOTALS_VERSION
            result["document_totals"] = [item.model_dump(mode="json") for item in totals]
            result["document_total"] = (
                primary.model_dump(mode="json") if primary is not None else None
            )
            published = store.publish_maintenance_result(
                job_id,
                result,
                expected_result_sha256=result_digest,
            )
            if not published:
                raise ValueError("job state changed during totals reprojection")
            summary["updated"] += 1
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            JobTransactionError,
        ) as error:
            summary["failed"] += 1
            summary["failures"].append({"job_id": job_id, "error": str(error)})
    return summary


@app.callback()
def run(
    root: Annotated[Path, typer.Option(file_okay=False, resolve_path=True)],
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Publish a validated totals-only reprojection."),
    ] = False,
) -> None:
    summary = backfill_totals(root=root, apply=apply)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
