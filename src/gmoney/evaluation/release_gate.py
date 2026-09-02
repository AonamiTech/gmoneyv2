from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gmoney.demo.store import JobStore, JobTransactionError

app = typer.Typer(no_args_is_help=True)
MANIFEST_VERSION = "gmoney_release_corpus_v2"
REPORT_VERSION = "gmoney_release_corpus_report_v2"
RELEASE_COHORT_COUNTS = {"production14": 14, "passing36": 36, "staging159": 159}
SHA256_PATTERN = r"^[0-9a-f]{64}$"
IMAGE_PATTERN = r"^sha256:[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GoldSnapshot(StrictModel):
    status: Literal["complete", "needs_review", "failed"]
    validation_status: Literal["passed", "needs_review", "failed"]
    canonical_rows_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tables_sha256: str = Field(pattern=SHA256_PATTERN)
    service_dates_sha256: str = Field(pattern=SHA256_PATTERN)
    receipts_sha256: str = Field(pattern=SHA256_PATTERN)
    totals_sha256: str = Field(pattern=SHA256_PATTERN)
    issues_sha256: str = Field(pattern=SHA256_PATTERN)
    recovery_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_usage_sha256: str = Field(pattern=SHA256_PATTERN)
    # Optional V6 certification metadata keeps sealed V5 manifests readable.
    semantic_v5_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    artifact_inventory_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    artifact_graph_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    mapping_error_summary_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidence_contract_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class CorpusDocument(StrictModel):
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    gold: GoldSnapshot
    baseline: GoldSnapshot | None = None


class CorpusCohort(StrictModel):
    required_count: int = Field(gt=0)
    documents: list[CorpusDocument]

    @model_validator(mode="after")
    def unique_and_complete(self) -> CorpusCohort:
        hashes = [document.source_sha256 for document in self.documents]
        if len(hashes) != self.required_count:
            raise ValueError("cohort document count does not match required_count")
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate source hashes are not allowed in a cohort")
        return self


class ReleaseAttestation(StrictModel):
    release_revision: str = Field(pattern=REVISION_PATTERN)
    api_image_digest: str = Field(pattern=IMAGE_PATTERN)
    frontend_image_digest: str = Field(pattern=IMAGE_PATTERN)
    worker_image_digest: str = Field(pattern=IMAGE_PATTERN)


class CorpusManifest(StrictModel):
    manifest_version: Literal["gmoney_release_corpus_v2"] = MANIFEST_VERSION
    created_at: str
    attestation: ReleaseAttestation
    cohorts: dict[str, CorpusCohort]

    @model_validator(mode="after")
    def complete_fixed_gate(self) -> CorpusManifest:
        if set(self.cohorts) != set(RELEASE_COHORT_COUNTS):
            raise ValueError(
                "release manifest must contain exactly production14, passing36, and staging159"
            )
        for name, count in RELEASE_COHORT_COUNTS.items():
            if self.cohorts[name].required_count != count:
                raise ValueError(f"{name} must contain exactly {count} documents")
            if name == "passing36" and any(
                document.baseline is None for document in self.cohorts[name].documents
            ):
                raise ValueError("every passing36 document requires a baseline snapshot")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _strip_v6_evidence(value: object) -> object:
    if isinstance(value, list):
        return [_strip_v6_evidence(item) for item in value]
    if isinstance(value, dict):
        if "source_page_polygon" in value:
            return {
                "page_number": value.get("source_page_number"),
                "polygon": _strip_v6_evidence(value.get("source_page_polygon")),
                "token_ids": sorted(value.get("ocr_token_ids") or []),
            }
        return {key: _strip_v6_evidence(item) for key, item in value.items()}
    return value


def _pdf_inventory(root: Path) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    for path in sorted(root.rglob("*.pdf")):
        digest = _sha256(path)
        if digest in inventory:
            raise ValueError(f"duplicate PDF content in cohort: {digest}")
        inventory[digest] = path
    return inventory


def _gold_snapshot(
    state: dict[str, Any], result: dict[str, Any], report: dict[str, Any]
) -> GoldSnapshot:
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    service_dates = [
        {
            "id": row.get("id"),
            "page_number": row.get("page_number"),
            "table_id": row.get("table_id"),
            "service_date_raw": row.get("service_date_raw"),
            "service_date_iso": row.get("service_date_iso"),
            "field_evidence": (row.get("field_evidence") or {}).get("service_date"),
        }
        for row in rows
        if isinstance(row, dict)
    ]
    receipts = {
        "rows": [
            row
            for row in rows
            if isinstance(row, dict)
            and (
                row.get("role") == "supporting_charge"
                or "receipt" in (row.get("validation_flags") or ())
                or row.get("receipt_metadata") is not None
            )
        ],
        "duplicate_pairs": result.get("receipt_duplicate_pairs") or [],
        "printed": [
            table
            for table in result.get("source_tables") or []
            if isinstance(table, dict)
            and (
                table.get("table_type") in {"receipt", "payment"}
                or any(
                    isinstance(row, dict) and row.get("receipt_metadata")
                    for row in table.get("rows") or []
                )
            )
        ],
    }
    totals = {
        "primary": result.get("document_total"),
        "selected": result.get("document_totals") or [],
        "raw": result.get("raw_total_candidates") or [],
    }
    manifest = result.get("artifact_manifest") or {}
    certification = state.get("certification") or {}
    v6 = result.get("output_version") == "offline_accuracy_spine_v6"
    semantic_v5 = _payload_sha256(
        {
            "rows": _strip_v6_evidence(rows),
            "source_tables": _strip_v6_evidence(result.get("source_tables") or []),
        }
    )
    mapping_summary = [
        {
            "artifact_id": item.get("artifact_id"),
            "mapping_sha256": (item.get("child_to_parent_mapping") or {}).get("mapping_sha256"),
        }
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    ]
    evidence_contract = {
        "evidence": result.get("evidence") or [],
        "tokens": result.get("token_manifest") or [],
        "adapters": result.get("adapter_inputs") or [],
    }
    return GoldSnapshot(
        status=state.get("status"),
        validation_status=report.get("status"),
        canonical_rows_sha256=_payload_sha256(rows),
        source_tables_sha256=_payload_sha256(result.get("source_tables") or []),
        service_dates_sha256=_payload_sha256(service_dates),
        receipts_sha256=_payload_sha256(receipts),
        totals_sha256=_payload_sha256(totals),
        issues_sha256=_payload_sha256(report.get("issues") or []),
        recovery_sha256=_payload_sha256(result.get("recovery") or {}),
        provider_usage_sha256=_payload_sha256(result.get("provider_usage") or {}),
        semantic_v5_sha256=semantic_v5 if v6 else None,
        artifact_inventory_sha256=certification.get("artifact_inventory_sha256") if v6 else None,
        artifact_graph_sha256=manifest.get("manifest_sha256") if v6 else None,
        mapping_error_summary_sha256=_payload_sha256(mapping_summary) if v6 else None,
        evidence_contract_sha256=_payload_sha256(evidence_contract) if v6 else None,
    )


def _workspace_inventory(data_root: Path, release_revision: str) -> dict[str, list[dict[str, Any]]]:
    store = JobStore(data_root)
    inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in store.states():
        job_id = str(state["id"])
        source = store.job_dir(job_id) / "source.pdf"
        if not source.is_file() or source.is_symlink():
            continue
        try:
            result = store.read_result(job_id)
            report = store.read_validation(job_id)
            certified_state = store.read(job_id)
        except (JobTransactionError, KeyError):
            continue
        certification = certified_state.get("certification") or {}
        inventory[_sha256(source)].append(
            {
                "job_id": job_id,
                "certified": certified_state.get("_certification_valid") is True,
                "release_matches": certification.get("worker_release_revision") == release_revision,
                "gold": _gold_snapshot(certified_state, result, report),
            }
        )
    return inventory


def seal_manifest(
    cohorts: dict[str, tuple[Path, int]],
    *,
    gold: dict[str, dict[str, Any]],
    release_revision: str,
    image_digests: dict[str, str],
    baseline_data_root: Path,
) -> dict[str, Any]:
    if set(cohorts) != set(RELEASE_COHORT_COUNTS):
        raise ValueError("all fixed release cohorts are required")
    baselines = _workspace_inventory(baseline_data_root, release_revision)
    sealed: dict[str, Any] = {}
    for name, (root, required_count) in sorted(cohorts.items()):
        documents = _pdf_inventory(root)
        if required_count != RELEASE_COHORT_COUNTS[name] or len(documents) != required_count:
            raise ValueError(f"{name} requires {RELEASE_COHORT_COUNTS[name]} unique PDFs")
        entries: list[dict[str, Any]] = []
        for digest in sorted(documents):
            if digest not in gold:
                raise ValueError(f"missing audited gold snapshot for {digest}")
            entry: dict[str, Any] = {
                "source_sha256": digest,
                "gold": GoldSnapshot.model_validate(gold[digest]).model_dump(mode="json"),
            }
            if name == "passing36":
                matching = baselines.get(digest, [])
                if len(matching) != 1:
                    raise ValueError(f"passing36 requires one baseline workspace for {digest}")
                if not matching[0]["certified"]:
                    raise ValueError(f"passing36 baseline is not certified for {digest}")
                entry["baseline"] = matching[0]["gold"].model_dump(mode="json")
            entries.append(entry)
        sealed[name] = {"required_count": required_count, "documents": entries}
    manifest = CorpusManifest.model_validate(
        {
            "manifest_version": MANIFEST_VERSION,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "attestation": {
                "release_revision": release_revision,
                "api_image_digest": image_digests.get("api"),
                "frontend_image_digest": image_digests.get("frontend"),
                "worker_image_digest": image_digests.get("worker"),
            },
            "cohorts": sealed,
        }
    )
    return manifest.model_dump(mode="json")


def evaluate_manifest(
    manifest: dict[str, Any],
    source_roots: dict[str, Path],
    data_root: Path,
    *,
    release_revision: str,
    image_digests: dict[str, str],
) -> dict[str, Any]:
    try:
        parsed = CorpusManifest.model_validate(manifest)
    except ValidationError as error:
        raise ValueError(f"invalid release corpus manifest: {error}") from error
    if parsed.attestation.release_revision != release_revision:
        raise ValueError("release revision does not match sealed manifest")
    expected_images = {
        "api": parsed.attestation.api_image_digest,
        "frontend": parsed.attestation.frontend_image_digest,
        "worker": parsed.attestation.worker_image_digest,
    }
    if expected_images != image_digests:
        raise ValueError("image digests do not match sealed manifest")
    if set(source_roots) != set(RELEASE_COHORT_COUNTS):
        raise ValueError("all fixed source roots are required")
    workspaces = _workspace_inventory(data_root, release_revision)
    results: dict[str, Any] = {}
    passed = True
    for name, cohort in parsed.cohorts.items():
        actual_sources = set(_pdf_inventory(source_roots[name]))
        expected_sources = {document.source_sha256 for document in cohort.documents}
        if actual_sources != expected_sources:
            raise ValueError(f"source hashes do not match sealed cohort {name}")
        documents: list[dict[str, Any]] = []
        for document in cohort.documents:
            matches = workspaces.get(document.source_sha256, [])
            failures: list[str] = []
            actual: GoldSnapshot | None = None
            if len(matches) != 1:
                failures.append(
                    "candidate_workspace_missing"
                    if not matches
                    else "multiple_candidate_workspaces"
                )
            else:
                match = matches[0]
                actual = match["gold"]
                if not match["certified"]:
                    failures.append("candidate_not_certified")
                if not match["release_matches"]:
                    failures.append("candidate_release_revision_mismatch")
                for field in GoldSnapshot.model_fields:
                    if getattr(actual, field) != getattr(document.gold, field):
                        failures.append(f"{field}_mismatch")
            passed = passed and not failures
            documents.append(
                {
                    "source_sha256": document.source_sha256,
                    "actual": actual.model_dump(mode="json") if actual else None,
                    "failures": failures,
                }
            )
        results[name] = {
            "required_count": cohort.required_count,
            "passed": all(not item["failures"] for item in documents),
            "documents": documents,
        }
    return {
        "report_version": REPORT_VERSION,
        "evaluated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "attestation": parsed.attestation.model_dump(mode="json"),
        "passed": passed,
        "cohorts": results,
    }


def require_release_cohorts(manifest: dict[str, Any]) -> None:
    try:
        CorpusManifest.model_validate(manifest)
    except ValidationError as error:
        raise ValueError(
            "release manifest must contain exactly production14, passing36, "
            "and staging159 with strict audited gold"
        ) from error


def _cohort_values(values: list[str]) -> dict[str, tuple[Path, int]]:
    output: dict[str, tuple[Path, int]] = {}
    for value in values:
        name, separator, remainder = value.partition("=")
        path_value, count_separator, count_value = remainder.rpartition(":")
        if not separator or not count_separator or not name or not path_value:
            raise typer.BadParameter("cohort must be NAME=PATH:COUNT")
        output[name] = (Path(path_value), int(count_value))
    return output


def _root_values(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise typer.BadParameter("source root must be NAME=PATH")
        output[name] = Path(path)
    return output


@app.command("seal")
def seal(
    cohort: Annotated[list[str], typer.Option("--cohort")],
    output: Annotated[Path, typer.Option("--output")],
    gold: Annotated[Path, typer.Option("--gold")],
    baseline_data_root: Annotated[Path, typer.Option("--baseline-data-root")],
    release_revision: Annotated[str, typer.Option("--release-revision")],
    api_image_digest: Annotated[str, typer.Option("--api-image-digest")],
    frontend_image_digest: Annotated[str, typer.Option("--frontend-image-digest")],
    worker_image_digest: Annotated[str, typer.Option("--worker-image-digest")],
) -> None:
    manifest = seal_manifest(
        _cohort_values(cohort),
        gold=json.loads(gold.read_text()),
        release_revision=release_revision,
        image_digests={
            "api": api_image_digest,
            "frontend": frontend_image_digest,
            "worker": worker_image_digest,
        },
        baseline_data_root=baseline_data_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@app.command("evaluate")
def evaluate(
    manifest: Annotated[Path, typer.Option("--manifest")],
    source_root: Annotated[list[str], typer.Option("--source-root")],
    data_root: Annotated[Path, typer.Option("--data-root")],
    output: Annotated[Path, typer.Option("--output")],
    release_revision: Annotated[str, typer.Option("--release-revision")],
    api_image_digest: Annotated[str, typer.Option("--api-image-digest")],
    frontend_image_digest: Annotated[str, typer.Option("--frontend-image-digest")],
    worker_image_digest: Annotated[str, typer.Option("--worker-image-digest")],
) -> None:
    report = evaluate_manifest(
        json.loads(manifest.read_text()),
        _root_values(source_root),
        data_root,
        release_revision=release_revision,
        image_digests={
            "api": api_image_digest,
            "frontend": frontend_image_digest,
            "worker": worker_image_digest,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
