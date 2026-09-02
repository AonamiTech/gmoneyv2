from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import fitz
from fastapi import FastAPI, File, Header, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from gmoney.contracts.extraction import SourceTable
from gmoney.contracts.v6 import SourceTableV2
from gmoney.demo.alias_transactions import AliasTransactionCoordinator
from gmoney.demo.review import (
    ReviewValidationError,
    approval_blockers,
    create_evidence_bundle,
    evidence_for_page,
    export_csv,
    export_payload,
    normalize_changes,
    project_hospital,
    project_legacy_evidence_tree,
    project_rows,
    public_page_assets,
    review_summary,
    reviewer_row,
    structural_issues,
    totals_summary,
)
from gmoney.demo.store import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    JobStore,
    JobTransactionError,
    ReviewRevisionConflict,
    utc_now,
    utc_text,
)
from gmoney.extraction.typed_values import parse_alias_field_value
from gmoney.profiles.aliases import (
    ALIAS_CANONICAL_FIELDS,
    CANONICAL_TO_SOURCE_FIELD,
    AliasRegistryRevisionConflict,
    AliasRegistryUnavailable,
    JsonAliasRepository,
    normalize_header,
    normalize_hospital_name,
)
from gmoney.profiles.repository import (
    HospitalIdentityConflict as PersistedHospitalIdentityConflict,
)
from gmoney.profiles.repository import (
    JsonProfileRepository,
    ProfileRegistryRevisionConflict,
    ProfileRegistryUnavailable,
    active_hospital_identities,
    combined_hospital_name_owners,
    validate_combined_hospital_identities,
)
from gmoney.release import build_revision

MAX_UPLOAD_BYTES = int(os.environ.get("GMONEY_MAX_UPLOAD_BYTES", "0"))
MAX_ACTIVE_JOBS = int(os.environ.get("GMONEY_MAX_ACTIVE_JOBS", "20"))
MAX_PDF_PAGES = int(os.environ.get("GMONEY_MAX_PDF_PAGES", "200"))
WORKER_CAPACITY = int(os.environ.get("GMONEY_WORKER_CAPACITY", "2"))
RETENTION_HOURS = int(os.environ.get("GMONEY_RETENTION_HOURS", "720"))
MIN_FREE_BYTES = int(os.environ.get("GMONEY_MIN_FREE_BYTES", "0"))
RELEASE_REVISION = build_revision()
WORKER_STATUS_MAX_AGE_SECONDS = int(os.environ.get("GMONEY_WORKER_STATUS_MAX_AGE_SECONDS", "30"))
WORKER_STATUS_FUTURE_SKEW_SECONDS = int(
    os.environ.get("GMONEY_WORKER_STATUS_FUTURE_SKEW_SECONDS", "5")
)
if WORKER_STATUS_MAX_AGE_SECONDS <= 0:
    raise RuntimeError("GMONEY_WORKER_STATUS_MAX_AGE_SECONDS must be positive")
if WORKER_STATUS_FUTURE_SKEW_SECONDS < 0:
    raise RuntimeError("GMONEY_WORKER_STATUS_FUTURE_SKEW_SECONDS cannot be negative")
DEMO_ROOT = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
PROFILE_REGISTRY = Path(
    os.environ.get("GMONEY_PROFILE_REGISTRY", str(DEMO_ROOT / "profile-registry.json"))
)
PROFILE_REGISTRY_LOCK = Path(
    os.environ.get(
        "GMONEY_PROFILE_REGISTRY_LOCK",
        str(PROFILE_REGISTRY.with_suffix(f"{PROFILE_REGISTRY.suffix}.lock")),
    )
)
ALIAS_REGISTRY = Path(
    os.environ.get("GMONEY_ALIAS_REGISTRY", str(DEMO_ROOT / "alias-registry.json"))
)
WORKER_STATUS_PATH = Path(value) if (value := os.environ.get("GMONEY_WORKER_STATUS_PATH")) else None
store = JobStore(DEMO_ROOT)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GMoney V2 Evidence Demo",
    version="0.4.0",
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
)


@app.on_event("startup")
def recover_alias_transactions_at_startup() -> None:
    try:
        _alias_coordinator().recover_all()
    except AliasRegistryUnavailable:
        logger.exception("hospital alias registry recovery failed during API startup")


def _byte_limit_label(size: int) -> str:
    mebibyte = 1024 * 1024
    kibibyte = 1024
    if size >= mebibyte and size % mebibyte == 0:
        return f"{size // mebibyte} MiB"
    if size >= kibibyte and size % kibibyte == 0:
        return f"{size // kibibyte} KiB"
    return f"{size} bytes"


class ReviewPoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class ReviewPolygon(BaseModel):
    points: list[ReviewPoint] = Field(min_length=3, max_length=20)


class RowPatch(BaseModel):
    changes: dict[str, Any]
    reason: str = Field(min_length=3, max_length=500)
    page_number: int | None = Field(default=None, ge=1)
    polygon: ReviewPolygon | None = None


class RowCreate(BaseModel):
    values: dict[str, Any]
    page_number: int = Field(ge=1)
    polygon: ReviewPolygon
    reason: str = Field(min_length=3, max_length=500)


class BulkRowsPatch(BaseModel):
    row_ids: list[str] = Field(min_length=1, max_length=500)
    action: Literal["reject", "restore"]
    reason: str = Field(min_length=3, max_length=500)


class IssuePatch(BaseModel):
    status: Literal["open", "resolved"]
    reason: str = Field(min_length=3, max_length=500)


class HospitalPatch(BaseModel):
    hospital_name: str = Field(min_length=2, max_length=200)
    page_number: int = Field(ge=1)
    polygon: ReviewPolygon
    reason: str = Field(min_length=3, max_length=500)


class HospitalLinkPatch(BaseModel):
    hospital_id: str | None = Field(default=None, min_length=1, max_length=200)
    create: bool = False
    registry_revision: int = Field(ge=0)
    profile_revision: int = Field(ge=0)
    reason: str = Field(min_length=3, max_length=500)


class ColumnAliasPreviewRequest(BaseModel):
    hospital_id: str = Field(min_length=1, max_length=200)
    source_label: str = Field(min_length=1, max_length=200)
    canonical_field: str = Field(min_length=1, max_length=100)


class ColumnAliasApplyRequest(ColumnAliasPreviewRequest):
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    registry_revision: int = Field(ge=0)
    selected_candidate_ids: list[str] = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=3, max_length=500)


class ColumnAliasPatch(BaseModel):
    registry_revision: int = Field(ge=0)
    canonical_field: str | None = Field(default=None, min_length=1, max_length=100)
    active: bool | None = None
    reason: str = Field(min_length=3, max_length=500)


class AliasRegistryItemNotFound(LookupError):
    pass


class HospitalSelectionNotCandidate(ReviewValidationError):
    def __init__(self, candidate_ids: set[str] | list[str]) -> None:
        super().__init__("Selected hospital is not a normalized-name candidate")
        self.candidate_ids = sorted(candidate_ids)


class HospitalIdentityConflict(ReviewValidationError):
    def __init__(self, code: str, candidate_ids: set[str] | list[str]) -> None:
        super().__init__(code)
        self.code = code
        self.candidate_ids = sorted(candidate_ids)


class ApprovalBlocked(ReviewValidationError):
    def __init__(self, blockers: list[str]) -> None:
        self.blockers = sorted(set(blockers))
        if "legacy_uncertified" in self.blockers:
            self.code = "legacy_uncertified"
            self.status_code = 409
        elif "certification_invalid" in self.blockers:
            self.code = "certification_invalid"
            self.status_code = 409
        else:
            self.code = "approval_blocked"
            self.status_code = 422
        super().__init__(self.code)


@app.middleware("http")
async def private_demo_responses(request: Any, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: state.get(key)
        for key in (
            "id",
            "status",
            "original_name",
            "created_at",
            "updated_at",
            "page",
            "pages",
            "row_count",
            "hospital_name",
            "hospital_confidence",
            "validation_status",
            "validation_issue_count",
            "validation_issue_codes",
            "error",
            "output_version",
            "contract_revision",
            "evidence_contract_status",
            "reprocess_recommended",
        )
    }
    public["hospital_name_source"] = "machine" if state.get("hospital_name") else None
    if public.get("evidence_contract_status") is None and state.get("status") in {
        "complete",
        "needs_review",
    }:
        public["evidence_contract_status"] = "legacy_v5"
        public["reprocess_recommended"] = True
    certified = state.get("_certification_valid") is True
    if state.get("status") in {"complete", "needs_review"} and not certified:
        public["validation_status"] = None
        public["validation_issue_count"] = None
        public["validation_issue_codes"] = None
    public["certification_status"] = (
        state.get("validation_status")
        if certified
        else (
            state.get("_certification_status", "legacy_uncertified")
            if state.get("status") in {"complete", "needs_review"}
            else None
        )
    )
    if state.get("status") in {"complete", "needs_review"}:
        try:
            hospital_override = (
                _alias_coordinator()
                .run_job_operation(state["id"], lambda: store.read_review(state["id"]))
                .get("document_overrides", {})
                .get("hospital")
            )
        except KeyError:
            hospital_override = None
        if hospital_override:
            public["hospital_name"] = hospital_override["name"]
            public["hospital_name_source"] = "reviewer"
    public["last_activity_at"] = utc_text(store.last_activity(state["id"], state))
    public["expires_at"] = store.expires_at(state["id"], RETENTION_HOURS, state)
    return public


def _state_or_404(job_id: str) -> dict[str, Any]:
    try:
        return store.read(job_id)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="Document not found") from error


def _complete_workspace(
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        state, result, review = _alias_coordinator().run_job_operation(
            job_id, lambda: store.read_workspace(job_id)
        )
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail="Document workspace is temporarily unavailable",
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "alias_registry_unavailable"},
        ) from error
    if state.get("status") not in {"complete", "needs_review"}:
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    return state, result, review


def _complete_result(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _state, result, review = _complete_workspace(job_id)
    return result, review


def _expected_revision(value: str | None) -> int:
    if value is None:
        raise HTTPException(status_code=428, detail="If-Match review revision is required")
    normalized = value.strip().removeprefix("W/").strip('"')
    try:
        revision = int(normalized)
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail="If-Match must be an integer revision"
        ) from error
    if revision < 0:
        raise HTTPException(status_code=400, detail="If-Match must be an integer revision")
    return revision


def _mutate(
    job_id: str,
    expected: int,
    mutation: Any,
) -> dict[str, Any]:
    try:
        return _alias_coordinator().mutate_review(job_id, expected, mutation)
    except ReviewRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except HospitalSelectionNotCandidate as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "hospital_selection_not_candidate",
                "candidate_ids": error.candidate_ids,
            },
        ) from error
    except HospitalIdentityConflict as error:
        raise HTTPException(
            status_code=409,
            detail={"code": error.code, "candidate_ids": error.candidate_ids},
        ) from error
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error


def _mutate_with_workspace(
    job_id: str,
    expected: int,
    mutation: Any,
) -> dict[str, Any]:
    try:
        return _alias_coordinator().mutate_review_with_workspace(
            job_id,
            expected,
            mutation,
        )
    except ReviewRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except ApprovalBlocked as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "blockers": error.blockers},
        ) from error
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail={"code": str(error)},
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "alias_registry_unavailable"},
        ) from error


def _event(
    revision: int,
    action: str,
    target_id: str,
    reason: str,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "revision": revision,
        "action": action,
        "target_id": target_id,
        "reviewer": "demo-reviewer",
        "reason": reason.strip(),
        "changes": changes or {},
        "created_at": utc_now(),
    }


def _registry_event(action: str, reason: str, **details: Any) -> dict[str, Any]:
    return {
        "action": action,
        **details,
        "reviewer": "demo-reviewer",
        "reason": reason.strip(),
        "created_at": utc_now(),
    }


def _merge_row_override(
    previous: dict[str, Any],
    changes: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Update row values without discarding disposition provenance metadata."""
    updated = dict(previous)
    updated["changes"] = {**previous.get("changes", {}), **changes}
    updated["reason"] = reason.strip()
    updated["updated_at"] = utc_now()
    return updated


def _alias_coordinator() -> AliasTransactionCoordinator:
    return AliasTransactionCoordinator(store, ALIAS_REGISTRY)


def _profile_repository() -> JsonProfileRepository:
    return JsonProfileRepository(PROFILE_REGISTRY, PROFILE_REGISTRY_LOCK)


def _alias_snapshot() -> dict[str, Any]:
    try:
        return _alias_coordinator().registry_snapshot()
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "alias_registry_unavailable"},
        ) from error


def _identity_snapshots() -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        return _alias_coordinator().identity_snapshots(_profile_repository())
    except ProfileRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "profile_registry_unavailable"},
        ) from error
    except PersistedHospitalIdentityConflict as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "hospital_identity_conflict",
                "candidate_ids": error.owner_ids,
            },
        ) from error


def _mutate_review_and_alias_registry(
    job_id: str,
    expected_review_revision: int,
    expected_registry_revision: int,
    mutation: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _alias_coordinator().mutate_review_and_registry(
            job_id,
            expected_review_revision,
            expected_registry_revision,
            mutation,
        )
    except AliasRegistryRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except ReviewRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except HospitalSelectionNotCandidate as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "hospital_selection_not_candidate",
                "candidate_ids": error.candidate_ids,
            },
        ) from error
    except HospitalIdentityConflict as error:
        raise HTTPException(
            status_code=409,
            detail={"code": error.code, "candidate_ids": error.candidate_ids},
        ) from error
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error


def _mutate_hospital_link(
    job_id: str,
    expected_review_revision: int,
    expected_registry_revision: int,
    expected_profile_revision: int,
    mutation: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _alias_coordinator().mutate_review_and_registry_with_profiles(
            job_id,
            expected_review_revision,
            expected_registry_revision,
            _profile_repository(),
            expected_profile_revision,
            mutation,
        )
    except ProfileRegistryRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "profile_registry_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except ProfileRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "profile_registry_unavailable"},
        ) from error
    except AliasRegistryRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except ReviewRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except HospitalSelectionNotCandidate as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "hospital_selection_not_candidate",
                "candidate_ids": error.candidate_ids,
            },
        ) from error
    except HospitalIdentityConflict as error:
        raise HTTPException(
            status_code=409,
            detail={"code": error.code, "candidate_ids": error.candidate_ids},
        ) from error
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "alias_registry_unavailable"},
        ) from error


def _validate_alias_field(value: str) -> str:
    field = value.strip()
    if field not in ALIAS_CANONICAL_FIELDS:
        raise HTTPException(status_code=422, detail="Unsupported normalized field")
    return field


def _hospital_link(review: dict[str, Any]) -> str | None:
    value = review.get("document_overrides", {}).get("hospital_link")
    return str(value.get("hospital_id")) if isinstance(value, dict) else None


def _alias_value(
    canonical_field: str,
    raw_value: str,
) -> tuple[dict[str, Any], str, str] | None:
    return parse_alias_field_value(canonical_field, raw_value)


def _alias_preview_payload(
    result: dict[str, Any],
    review: dict[str, Any],
    registry: dict[str, Any],
    request: ColumnAliasPreviewRequest,
) -> dict[str, Any]:
    canonical_field = _validate_alias_field(request.canonical_field)
    if _hospital_link(review) != request.hospital_id:
        raise HTTPException(
            status_code=422,
            detail="Link this bill to the selected hospital before teaching aliases",
        )
    source_payload = result.get("source_tables")
    if not source_payload:
        raise HTTPException(
            status_code=422,
            detail="Grounded printed columns are unavailable; reprocess this bill first",
        )
    try:
        tables = tuple(SourceTable.model_validate(item) for item in source_payload)
    except ValidationError as error:
        raise HTTPException(
            status_code=409,
            detail="Printed extraction failed grounding validation; reprocess this bill",
        ) from error
    normalized_label = normalize_header(request.source_label)
    if not normalized_label:
        raise HTTPException(status_code=422, detail="Alias label is empty after normalization")
    rows = {str(row["id"]): row for row in project_rows(result, review)}
    candidates: list[dict[str, Any]] = []
    source_columns: list[dict[str, str]] = []
    for table in tables:
        matching_columns = [
            column for column in table.columns if normalize_header(column.label) == normalized_label
        ]
        for column in matching_columns:
            source_ref = {
                "source_table_id": table.id,
                "source_column_id": column.id,
            }
            source_columns.append(source_ref)
            for source_row in table.rows:
                cell = next(item for item in source_row.cells if item.column_id == column.id)
                base = {
                    "source_table_id": table.id,
                    "source_table_name": table.table_id,
                    "source_page": table.page_number,
                    "source_row_id": source_row.id,
                    "source_column_id": column.id,
                    "source_column_label": column.label,
                    "source_column_evidence": [
                        item.model_dump(mode="json") for item in column.evidence
                    ],
                    "row_id": source_row.canonical_row_id,
                    "source_value": cell.raw_value,
                    "evidence": [item.model_dump(mode="json") for item in cell.evidence],
                }
                candidate_material = "\0".join(
                    (
                        str(result["document_id"]),
                        table.id,
                        column.id,
                        source_row.id,
                        str(source_row.canonical_row_id or ""),
                        canonical_field,
                    )
                )
                base["candidate_id"] = hashlib.sha256(candidate_material.encode()).hexdigest()
                if source_row.canonical_row_id is None or source_row.canonical_row_id not in rows:
                    candidates.append({**base, "classification": "unlinked"})
                    continue
                parsed = _alias_value(canonical_field, cell.raw_value or "")
                if parsed is None or not cell.evidence:
                    candidates.append({**base, "classification": "invalid"})
                    continue
                changes, comparison_field, evidence_field = parsed
                current = rows[source_row.canonical_row_id].get(comparison_field)
                proposed = changes.get(comparison_field)
                if current is None or str(current).strip() == "":
                    classification = "fillable"
                elif normalize_header(str(current)) == normalize_header(str(proposed)):
                    classification = "unchanged"
                else:
                    classification = "conflicting"
                candidates.append(
                    {
                        **base,
                        "classification": classification,
                        "current_value": current,
                        "proposed_value": proposed,
                        "changes": changes,
                        "evidence_field": evidence_field,
                    }
                )
    if not source_columns:
        raise HTTPException(status_code=422, detail="Printed header was not found")
    digest_payload = {
        "document_id": result["document_id"],
        "review_revision": review["revision"],
        "registry_revision": registry["revision"],
        "hospital_id": request.hospital_id,
        "source_label": normalized_label,
        "canonical_field": canonical_field,
        "columns": sorted(
            source_columns,
            key=lambda item: (item["source_table_id"], item["source_column_id"]),
        ),
        "candidates": sorted(candidates, key=lambda item: item["candidate_id"]),
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    counts = {
        classification: sum(item["classification"] == classification for item in candidates)
        for classification in (
            "fillable",
            "unchanged",
            "conflicting",
            "invalid",
            "unlinked",
        )
    }
    return {
        "document_id": result["document_id"],
        "hospital_id": request.hospital_id,
        "source_label": request.source_label.strip(),
        "normalized_label": normalized_label,
        "canonical_field": canonical_field,
        "review_revision": review["revision"],
        "registry_revision": registry["revision"],
        "source_digest": digest,
        "source_columns": sorted(
            source_columns,
            key=lambda item: (item["source_table_id"], item["source_column_id"]),
        ),
        "counts": counts,
        "candidates": candidates,
    }


def _download_stem(original_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip("-._")
    return normalized[:100] or "bill"


@app.get("/api/v2/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v2/health/ready", tags=["health"])
def ready(response: Response) -> dict[str, Any]:
    profiles, aliases, _ = _identity_snapshots()
    storage = shutil.disk_usage(store.jobs_root)
    worker_release_revision: str | None = None
    worker_status_updated_at: str | None = None
    release_consistent = True
    worker_ready = True
    if WORKER_STATUS_PATH is not None:
        try:
            worker_status = json.loads(WORKER_STATUS_PATH.read_text())
            worker_release_revision = str(worker_status["release_revision"])
            worker_status_updated_at = str(worker_status["updated_at"])
            updated_at = datetime.fromisoformat(worker_status_updated_at.replace("Z", "+00:00"))
            age = (datetime.now(UTC) - updated_at).total_seconds()
            worker_ready = (
                worker_status.get("status") == "running"
                and updated_at.tzinfo is not None
                and -WORKER_STATUS_FUTURE_SKEW_SECONDS <= age <= WORKER_STATUS_MAX_AGE_SECONDS
            )
            release_consistent = worker_release_revision == RELEASE_REVISION
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            worker_ready = False
            release_consistent = False
    status_value = "ready" if worker_ready and release_consistent else "unready"
    if status_value != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": status_value,
        "release_revision": RELEASE_REVISION,
        "worker_release_revision": worker_release_revision,
        "worker_status_updated_at": worker_status_updated_at,
        "worker_status_max_age_seconds": WORKER_STATUS_MAX_AGE_SECONDS,
        "worker_status_future_skew_seconds": WORKER_STATUS_FUTURE_SKEW_SECONDS,
        "release_consistent": release_consistent,
        "profile_revision": profiles.revision,
        "alias_registry_revision": aliases["revision"],
        "active_jobs": store.active_count(),
        "worker_capacity": WORKER_CAPACITY,
        "queue_capacity": MAX_ACTIVE_JOBS,
        "retention_hours": RETENTION_HOURS,
        "storage_total_bytes": storage.total,
        "storage_free_bytes": storage.free,
        "storage_min_free_bytes": MIN_FREE_BYTES,
    }


@app.get("/api/v2/hospitals/trained")
def list_trained_hospitals() -> dict[str, Any]:
    profiles, aliases, identities = _identity_snapshots()
    hospitals: dict[str, dict[str, Any]] = {}
    for hospital_id, identity in identities.items():
        if not identity["hospital_name"]:
            continue
        hospitals[hospital_id] = {
            "hospital_id": hospital_id,
            "hospital_name": identity["hospital_name"],
            "active_profile_count": identity["active_profile_count"],
            "alias_count": 0,
            "training_sources": ["profile"],
        }
    for hospital in aliases["hospitals"]:
        hospital_id = str(hospital["hospital_id"])
        profile_identity = identities.get(hospital_id)
        item = hospitals.setdefault(
            hospital_id,
            {
                "hospital_id": hospital_id,
                "hospital_name": hospital["hospital_name"],
                "active_profile_count": (
                    profile_identity["active_profile_count"] if profile_identity else 0
                ),
                "alias_count": 0,
                "training_sources": ["profile"] if profile_identity else [],
            },
        )
        profile_name = (profile_identity or {}).get("hospital_name")
        if not profile_name:
            item["hospital_name"] = hospital["hospital_name"]
        for origin in hospital.get("origins", []):
            if origin not in item["training_sources"]:
                item["training_sources"].append(origin)
        item["alias_count"] = sum(
            alias["hospital_id"] == hospital_id and alias.get("active", True)
            for alias in aliases["aliases"]
        )
    items = sorted(
        hospitals.values(),
        key=lambda item: (str(item["hospital_name"]).casefold(), item["hospital_id"]),
    )
    return {
        "registry_revision": aliases["revision"],
        "profile_revision": profiles.revision,
        "total": len(items),
        "hospitals": items,
    }


@app.post("/api/v2/documents", status_code=status.HTTP_202_ACCEPTED)
async def create_document(
    file: Annotated[UploadFile, File(description="Hospital bill PDF")],
) -> dict[str, Any]:
    if store.active_count() >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=429, detail="Demo queue is full")
    if MIN_FREE_BYTES and shutil.disk_usage(store.jobs_root).free < MIN_FREE_BYTES:
        raise HTTPException(
            status_code=507,
            detail="Evidence storage is below its safe free-space threshold",
        )
    if not (file.filename or "").casefold().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted")

    state = store.create(file.filename or "bill.pdf")
    target = store.job_dir(state["id"]) / "source.pdf"
    size = 0
    signature = b""
    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                if not signature:
                    signature = chunk[:5]
                size += len(chunk)
                if MAX_UPLOAD_BYTES > 0 and size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {_byte_limit_label(MAX_UPLOAD_BYTES)} limit",
                    )
                output.write(chunk)
        if signature != b"%PDF-":
            raise HTTPException(status_code=415, detail="File does not have a valid PDF signature")
        try:
            with fitz.open(target) as document:
                if document.needs_pass:
                    raise HTTPException(status_code=415, detail="Encrypted PDFs are not accepted")
                if document.page_count < 1:
                    raise HTTPException(status_code=415, detail="PDF contains no pages")
                if document.page_count > MAX_PDF_PAGES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_PDF_PAGES}-page demo limit",
                    )
        except fitz.FileDataError as error:
            raise HTTPException(status_code=415, detail="PDF could not be parsed") from error
        state = store.update(state["id"], status="queued", size_bytes=size)
        return _public_state(state)
    except Exception:
        target.unlink(missing_ok=True)
        with suppress(KeyError):
            store.update(state["id"], status="failed", error="Upload rejected")
        raise
    finally:
        await file.close()


@app.get("/api/v2/documents")
def list_documents(
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    scope: Annotated[Literal["active", "history", "all"], Query()] = "all",
    query: Annotated[str | None, Query(max_length=200)] = None,
    document_status: Annotated[
        Literal[
            "uploading",
            "queued",
            "processing",
            "complete",
            "needs_review",
            "failed",
        ]
        | None,
        Query(alias="status"),
    ] = None,
) -> dict[str, Any]:
    states = store.states()
    if scope == "active":
        states = [state for state in states if state.get("status") in ACTIVE_STATUSES]
    elif scope == "history":
        states = [state for state in states if state.get("status") in TERMINAL_STATUSES]
    if document_status is not None:
        states = [state for state in states if state.get("status") == document_status]
    documents: list[dict[str, Any]] = []
    for state in states:
        try:
            documents.append(_public_state(state))
        except (KeyError, JobTransactionError, AliasRegistryUnavailable):
            continue
    if query:
        needle = query.casefold().strip()
        documents = [
            document
            for document in documents
            if needle in str(document.get("original_name") or "").casefold()
            or needle in str(document.get("hospital_name") or "").casefold()
        ]
    documents.sort(
        key=lambda document: str(
            document.get("last_activity_at")
            or document.get("updated_at")
            or document.get("created_at")
            or ""
        ),
        reverse=True,
    )
    total = len(documents)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "documents": documents[offset : offset + limit],
    }


@app.get("/api/v2/documents/{job_id}")
def get_document(job_id: str) -> dict[str, Any]:
    try:
        return _public_state(_state_or_404(job_id))
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail="Document workspace is temporarily unavailable",
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error


@app.get("/api/v2/documents/{job_id}/validation")
def get_document_validation(job_id: str) -> dict[str, Any]:
    try:
        return _alias_coordinator().run_job_operation(
            job_id,
            lambda: store.read_validation(job_id),
        )
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail={"code": str(error)},
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error


@app.get("/api/v2/documents/{job_id}/rows")
def get_rows(
    job_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    query: Annotated[str | None, Query(max_length=200)] = None,
    disposition: Annotated[str | None, Query()] = None,
    source_page: Annotated[int | None, Query(ge=1)] = None,
    anchor_row_id: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    rows = project_rows(result, review)
    totals = totals_summary(result, review, rows)
    if result.get("output_version") == "offline_accuracy_spine_v6":
        totals = project_legacy_evidence_tree(result, totals)
    if query:
        needle = query.casefold().strip()
        rows = [
            row
            for row in rows
            if needle in str(row.get("description") or "").casefold()
            or needle in str(row.get("service_code") or "").casefold()
            or needle in str(row.get("section") or "").casefold()
            or needle in str(row.get("service_date_raw") or "").casefold()
            or needle in str(row.get("service_date_iso") or "").casefold()
        ]
    if disposition == "active":
        rows = [row for row in rows if row.get("review_disposition") != "rejected"]
    elif disposition:
        rows = [row for row in rows if row.get("review_disposition") == disposition]
    if source_page:
        rows = [row for row in rows if row.get("page_number") == source_page]
    total = len(rows)
    if anchor_row_id is not None:
        anchor_index = next(
            (index for index, row in enumerate(rows) if str(row.get("id")) == anchor_row_id),
            None,
        )
        if anchor_index is None:
            raise HTTPException(status_code=404, detail="Canonical issue row not found")
        offset = (anchor_index // limit) * limit
    populated_fields = [
        field
        for field in (
            "section",
            "request_no",
            "service_code",
            "hsn_code",
            "discount",
        )
        if any(row.get(field) not in (None, "") for row in rows)
    ]
    return {
        "document_id": result["document_id"],
        "pages": result["pages"],
        "page_assets": public_page_assets(result),
        "hospital": project_hospital(result, review),
        "review_revision": review["revision"],
        "totals": totals,
        "total": total,
        "populated_fields": populated_fields,
        "offset": offset,
        "limit": limit,
        "rows": rows[offset : offset + limit],
    }


@app.get("/api/v2/documents/{job_id}/source-tables")
def get_source_tables(
    job_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    query: Annotated[str | None, Query(max_length=200)] = None,
    source_page: Annotated[int | None, Query(ge=1)] = None,
    anchor_row_id: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    source_payload = result.get("source_tables")
    source_present = "source_tables" in result
    available = bool(source_payload)
    unavailable_reason = (
        None if available else ("no_source_tables" if source_present else "legacy_result")
    )
    try:
        table_model = (
            SourceTableV2
            if result.get("output_version") == "offline_accuracy_spine_v6"
            else SourceTable
        )
        tables = [table_model.model_validate(table) for table in source_payload or []]
    except ValidationError as error:
        raise HTTPException(
            status_code=409,
            detail="Printed extraction failed grounding validation; reprocess this bill",
        ) from error
    column_mappings = review.get("column_mappings", {})
    tables = [
        table.model_copy(
            update={
                "columns": tuple(
                    column.model_copy(
                        update={
                            "canonical_field": CANONICAL_TO_SOURCE_FIELD.get(
                                str(column_mappings[table.id][column.id]["canonical_field"]),
                                str(column_mappings[table.id][column.id]["canonical_field"]),
                            )
                        }
                    )
                    if column.id in column_mappings.get(table.id, {})
                    else column
                    for column in table.columns
                )
            }
        )
        for table in tables
    ]
    if source_page is not None:
        tables = [table for table in tables if table.page_number == source_page]

    needle = query.casefold().strip() if query else ""
    selected_rows: list[tuple[int, Any]] = []
    for table_index, table in enumerate(tables):
        for row in table.rows:
            if needle and not any(
                needle in str(cell.raw_value or "").casefold() for cell in row.cells
            ):
                continue
            selected_rows.append((table_index, row))

    total = len(selected_rows)
    if anchor_row_id is not None:
        anchor_index = next(
            (index for index, (_, row) in enumerate(selected_rows) if str(row.id) == anchor_row_id),
            None,
        )
        if anchor_index is None:
            raise HTTPException(status_code=404, detail="Printed issue row not found")
        offset = (anchor_index // limit) * limit
    window = list(enumerate(selected_rows, start=1))[offset : offset + limit]
    rows_by_table: dict[int, list[tuple[Any, int]]] = {}
    for ordinal, (table_index, row) in window:
        rows_by_table.setdefault(table_index, []).append((row, ordinal))
    public_tables: list[dict[str, Any]] = []
    for index, table in enumerate(tables):
        if index not in rows_by_table:
            continue
        payload = table.model_dump(mode="json")
        if result.get("output_version") == "offline_accuracy_spine_v6":
            payload = project_legacy_evidence_tree(result, payload)
        payload["rows"] = []
        for row, ordinal in rows_by_table[index]:
            row_payload = row.model_dump(mode="json")
            if result.get("output_version") == "offline_accuracy_spine_v6":
                row_payload = project_legacy_evidence_tree(result, row_payload)
            row_payload["ordinal"] = ordinal
            payload["rows"].append(row_payload)
        public_tables.append(payload)
    return {
        "document_id": result["document_id"],
        "available": available,
        "unavailable_reason": unavailable_reason,
        "total": total,
        "offset": offset,
        "limit": limit,
        "hospital_id": _hospital_link(review),
        "tables": public_tables,
    }


@app.get("/api/v2/documents/{job_id}/review")
def get_review(job_id: str) -> dict[str, Any]:
    state, result, review = _complete_workspace(job_id)
    blockers = approval_blockers(
        store,
        job_id,
        result,
        review,
        state=state,
    )
    summary = review_summary(result, review)
    summary["approval_blockers"] = blockers
    summary["approval_effective"] = bool(review.get("approval")) and not blockers
    summary["export_eligible"] = summary["approval_effective"]
    return summary


@app.patch("/api/v2/documents/{job_id}/metadata")
def update_document_metadata(
    job_id: str,
    payload: HospitalPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)
    evidence = evidence_for_page(
        result,
        payload.page_number,
        [point.model_dump() for point in payload.polygon.points],
    )
    hospital_name = re.sub(r"\s+", " ", payload.hospital_name).strip()

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        review.setdefault("document_overrides", {})["hospital"] = {
            "name": hospital_name,
            "reason": payload.reason.strip(),
            "evidence": evidence,
            "updated_at": utc_now(),
        }
        review["approval"] = None
        review["events"].append(
            _event(
                expected + 1,
                "hospital_updated",
                job_id,
                payload.reason,
                {"hospital_name": hospital_name, "page_number": payload.page_number},
            )
        )
        return review

    review = _mutate(job_id, expected, mutation)
    return {
        "review_revision": review["revision"],
        "hospital": project_hospital(result, review),
    }


@app.post("/api/v2/documents/{job_id}/hospital-link")
def link_document_hospital(
    job_id: str,
    payload: HospitalLinkPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, current_review = _complete_result(job_id)
    expected = _expected_revision(if_match)
    hospital = project_hospital(result, current_review)
    if not hospital or not hospital.get("name") or not hospital.get("evidence"):
        raise HTTPException(
            status_code=422,
            detail="Confirm a grounded hospital identity before linking aliases",
        )
    if payload.create == (payload.hospital_id is not None):
        raise HTTPException(
            status_code=422,
            detail="Select one existing hospital or create one from this bill",
        )

    profile_snapshot, snapshot, profile_identities = _identity_snapshots()
    if snapshot["revision"] != payload.registry_revision:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": snapshot["revision"],
            },
        )
    if profile_snapshot.revision != payload.profile_revision:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "profile_registry_revision_conflict",
                "current_revision": profile_snapshot.revision,
            },
        )
    selected_name = str(hospital["name"])
    normalized_name = normalize_hospital_name(selected_name)
    registry_matches = combined_hospital_name_owners(
        profile_identities,
        snapshot,
        selected_name,
    )
    candidates = sorted(registry_matches)
    if payload.create and candidates:
        raise HTTPException(
            status_code=409,
            detail={
                "code": (
                    "hospital_identity_ambiguous"
                    if len(candidates) > 1
                    else "hospital_already_exists"
                ),
                "candidate_ids": candidates,
            },
        )
    hospital_id = (
        JsonAliasRepository.hospital_id_for_name(selected_name)
        if payload.create
        else str(payload.hospital_id)
    )
    if not payload.create and hospital_id not in candidates:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "hospital_selection_not_candidate",
                "candidate_ids": candidates,
            },
        )
    existing = next(
        (item for item in snapshot["hospitals"] if item["hospital_id"] == hospital_id),
        None,
    )
    if not payload.create and existing is None and hospital_id not in profile_identities:
        raise HTTPException(status_code=422, detail="Selected hospital was not found")
    conflicting_owner = next(
        (owner for owner in registry_matches if owner != hospital_id),
        None,
    )
    if conflicting_owner is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "hospital_name_conflict",
                "candidate_ids": sorted(registry_matches),
            },
        )
    canonical_name_holder: dict[str, str] = {}

    def coordinated_mutation(
        review: dict[str, Any],
        registry: dict[str, Any],
        locked_result: dict[str, Any],
        locked_profile_snapshot: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if locked_result["document_id"] != result["document_id"]:
            raise ReviewValidationError("Extraction result changed; reload the document")
        record = next(
            (item for item in registry["hospitals"] if item["hospital_id"] == hospital_id),
            None,
        )
        locked_identities = active_hospital_identities(locked_profile_snapshot)
        try:
            validate_combined_hospital_identities(locked_identities, registry)
        except PersistedHospitalIdentityConflict as error:
            raise HospitalIdentityConflict(
                "hospital_identity_ambiguous",
                set(error.owner_ids),
            ) from error
        owners = combined_hospital_name_owners(
            locked_identities,
            registry,
            selected_name,
        )
        locked_candidates = owners
        if payload.create and locked_candidates:
            raise HospitalIdentityConflict(
                (
                    "hospital_identity_ambiguous"
                    if len(locked_candidates) > 1
                    else "hospital_already_exists"
                ),
                locked_candidates,
            )
        if not payload.create and hospital_id not in locked_candidates:
            raise HospitalSelectionNotCandidate(locked_candidates)
        if owners - {hospital_id}:
            raise HospitalIdentityConflict("hospital_name_conflict", owners)
        profile_identity = locked_identities.get(hospital_id)
        locked_profile_name = (
            str(profile_identity["hospital_name"])
            if profile_identity and profile_identity.get("hospital_name")
            else None
        )
        locked_canonical_name = locked_profile_name or (
            str(record["hospital_name"]) if record is not None else selected_name
        )
        canonical_owners = combined_hospital_name_owners(
            locked_identities,
            registry,
            locked_canonical_name,
        )
        conflicting_canonical_owners = canonical_owners - {hospital_id}
        if conflicting_canonical_owners:
            raise HospitalIdentityConflict(
                "hospital_name_conflict",
                conflicting_canonical_owners,
            )
        old_canonical_name = str(record["hospital_name"]) if record is not None else None
        canonical_name_source = (
            "active_profile"
            if locked_profile_name
            else "registry"
            if record is not None
            else "reviewer_alias"
        )
        canonical_name_holder["value"] = locked_canonical_name
        if record is None:
            record = {
                "hospital_id": hospital_id,
                "hospital_name": locked_canonical_name,
                "origins": [],
                "name_variants": [],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            registry["hospitals"].append(record)
        else:
            record["hospital_name"] = locked_canonical_name
        for origin in (
            "profile" if hospital_id in locked_identities else None,
            "reviewer_alias",
        ):
            if origin and origin not in record["origins"]:
                record["origins"].append(origin)
        if not any(
            variant.get("normalized_name") == normalized_name for variant in record["name_variants"]
        ):
            record["name_variants"].append(
                {
                    "display_name": selected_name,
                    "normalized_name": normalized_name,
                    "verified": True,
                    "source_document_id": result["document_id"],
                    "reviewer": "demo-reviewer",
                    "reason": payload.reason.strip(),
                    "created_at": utc_now(),
                }
            )
        record["updated_at"] = utc_now()
        registry["events"].append(
            _registry_event(
                "hospital_linked",
                payload.reason,
                hospital_id=hospital_id,
                document_id=result["document_id"],
                old_hospital_name=old_canonical_name,
                new_hospital_name=locked_canonical_name,
                canonical_name_source=canonical_name_source,
                profile_revision=locked_profile_snapshot.revision,
            )
        )
        review.setdefault("document_overrides", {})["hospital_link"] = {
            "hospital_id": hospital_id,
            "hospital_name": locked_canonical_name,
            "reason": payload.reason.strip(),
            "updated_at": utc_now(),
        }
        review["approval"] = None
        review["events"].append(
            _event(
                expected + 1,
                "hospital_linked",
                hospital_id,
                payload.reason,
                {
                    "hospital_name": locked_canonical_name,
                    "old_hospital_name": old_canonical_name,
                    "new_hospital_name": locked_canonical_name,
                    "canonical_name_source": canonical_name_source,
                    "profile_revision": locked_profile_snapshot.revision,
                },
            )
        )
        return review, registry

    review, updated_registry = _mutate_hospital_link(
        job_id,
        expected,
        payload.registry_revision,
        payload.profile_revision,
        coordinated_mutation,
    )
    return {
        "review_revision": review["revision"],
        "registry_revision": updated_registry["revision"],
        "profile_revision": payload.profile_revision,
        "hospital_id": hospital_id,
        "hospital_name": canonical_name_holder["value"],
    }


@app.get("/api/v2/hospitals/{hospital_id}/aliases")
def list_hospital_aliases(hospital_id: str) -> dict[str, Any]:
    snapshot = _alias_snapshot()
    hospital = next(
        (item for item in snapshot["hospitals"] if item["hospital_id"] == hospital_id),
        None,
    )
    if hospital is None:
        raise HTTPException(status_code=404, detail="Hospital was not found")
    aliases = [item for item in snapshot["aliases"] if item["hospital_id"] == hospital_id]
    aliases.sort(key=lambda item: (item["normalized_label"], item["alias_id"]))
    return {
        "registry_revision": snapshot["revision"],
        "hospital": hospital,
        "aliases": aliases,
    }


@app.post("/api/v2/documents/{job_id}/column-aliases/preview")
def preview_column_alias(
    job_id: str,
    payload: ColumnAliasPreviewRequest,
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    return _alias_preview_payload(result, review, _alias_snapshot(), payload)


@app.post("/api/v2/documents/{job_id}/column-aliases/apply")
def apply_column_alias(
    job_id: str,
    payload: ColumnAliasApplyRequest,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    expected = _expected_revision(if_match)
    if review["revision"] != expected:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": review["revision"],
            },
        )
    registry = _alias_snapshot()
    if registry["revision"] != payload.registry_revision:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": registry["revision"],
            },
        )
    selected_ids = tuple(
        dict.fromkeys(item.strip() for item in payload.selected_candidate_ids if item.strip())
    )
    if len(selected_ids) != len(payload.selected_candidate_ids):
        raise HTTPException(
            status_code=422,
            detail="Selected candidate IDs must be unique and non-empty",
        )
    alias_id_holder: dict[str, str] = {}
    applied_holder: dict[str, Any] = {}

    def coordinated_mutation(
        current: dict[str, Any],
        snapshot: dict[str, Any],
        locked_result: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        locked_preview = _alias_preview_payload(locked_result, current, snapshot, payload)
        if locked_preview["source_digest"] != payload.source_digest:
            raise HTTPException(
                status_code=409,
                detail={"code": "alias_source_digest_conflict"},
            )
        candidates_by_id = {
            str(item["candidate_id"]): item for item in locked_preview["candidates"]
        }
        unknown = set(selected_ids) - set(candidates_by_id)
        if unknown:
            raise ReviewValidationError("Selected alias candidates are stale or unknown")
        selected = [candidates_by_id[candidate_id] for candidate_id in selected_ids]
        invalid = [
            item["candidate_id"]
            for item in selected
            if item["classification"] not in {"fillable", "unchanged", "conflicting"}
        ]
        if invalid:
            raise ReviewValidationError("Invalid or unlinked candidates cannot be selected")
        targets = [(str(item["row_id"]), str(item["evidence_field"])) for item in selected]
        if len(set(targets)) != len(targets):
            raise ReviewValidationError(
                "Only one source candidate may update each normalized row field"
            )
        if not any(item["hospital_id"] == payload.hospital_id for item in snapshot["hospitals"]):
            raise ReviewValidationError("Hospital was removed")
        alias_id = next(
            (
                item["alias_id"]
                for item in snapshot["aliases"]
                if item["hospital_id"] == payload.hospital_id
                and item["normalized_label"] == locked_preview["normalized_label"]
            ),
            JsonAliasRepository.new_alias_id(),
        )
        alias_id_holder["value"] = alias_id
        alias = next(
            (item for item in snapshot["aliases"] if item["alias_id"] == alias_id),
            None,
        )
        if alias is None:
            alias = {
                "alias_id": alias_id,
                "hospital_id": payload.hospital_id,
                "source_label": locked_preview["source_label"],
                "normalized_label": locked_preview["normalized_label"],
                "canonical_field": locked_preview["canonical_field"],
                "active": True,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "reason": payload.reason.strip(),
            }
            snapshot["aliases"].append(alias)
        else:
            alias.update(
                source_label=locked_preview["source_label"],
                canonical_field=locked_preview["canonical_field"],
                active=True,
                updated_at=utc_now(),
                reason=payload.reason.strip(),
            )
        snapshot["events"].append(
            _registry_event(
                "column_alias_applied",
                payload.reason,
                alias_id=alias_id,
                hospital_id=payload.hospital_id,
                canonical_field=locked_preview["canonical_field"],
                document_id=locked_result["document_id"],
            )
        )
        projected = {str(row["id"]): row for row in project_rows(locked_result, current)}
        current.setdefault("column_mappings", {})
        for column_ref in locked_preview["source_columns"]:
            table_mappings = current["column_mappings"].setdefault(
                column_ref["source_table_id"], {}
            )
            table_mappings[column_ref["source_column_id"]] = {
                "alias_id": alias_id,
                "hospital_id": payload.hospital_id,
                "source_label": locked_preview["source_label"],
                "canonical_field": locked_preview["canonical_field"],
                "reason": payload.reason.strip(),
                "updated_at": utc_now(),
            }
        for candidate in selected:
            row_id = str(candidate["row_id"])
            row = projected[row_id]
            changes = (
                {} if candidate["classification"] == "unchanged" else dict(candidate["changes"])
            )
            field_evidence = dict(row.get("field_evidence") or {})
            field_evidence[candidate["evidence_field"]] = candidate["evidence"]
            changes["field_evidence"] = field_evidence
            routes = list(row.get("source_routes") or [])
            if "reviewer_hospital_alias" not in routes:
                routes.append("reviewer_hospital_alias")
            changes["source_routes"] = routes
            flags = list(row.get("validation_flags") or [])
            if "reviewer_column_alias" not in flags:
                flags.append("reviewer_column_alias")
            changes["validation_flags"] = flags
            if row_id in current["added_rows"]:
                current["added_rows"][row_id].update(changes)
                current["added_rows"][row_id]["review_reason"] = payload.reason.strip()
            else:
                previous = current["row_overrides"].get(row_id, {})
                current["row_overrides"][row_id] = _merge_row_override(
                    previous,
                    changes,
                    payload.reason,
                )
        current["approval"] = None
        current["events"].append(
            _event(
                expected + 1,
                "column_alias_applied",
                alias_id,
                payload.reason,
                {
                    "source_label": locked_preview["source_label"],
                    "canonical_field": locked_preview["canonical_field"],
                    "updated_row_ids": [item["row_id"] for item in selected],
                    "selected_candidate_ids": list(selected_ids),
                },
            )
        )
        applied_holder["count"] = len(selected)
        applied_holder["counts"] = locked_preview["counts"]
        return current, snapshot

    updated_review, updated_registry = _mutate_review_and_alias_registry(
        job_id,
        expected,
        payload.registry_revision,
        coordinated_mutation,
    )
    return {
        "review_revision": updated_review["revision"],
        "registry_revision": updated_registry["revision"],
        "alias_id": alias_id_holder["value"],
        "updated_count": applied_holder["count"],
        "skipped": applied_holder["counts"],
    }


@app.patch("/api/v2/hospitals/{hospital_id}/aliases/{alias_id}")
def update_column_alias(
    hospital_id: str,
    alias_id: str,
    payload: ColumnAliasPatch,
) -> dict[str, Any]:
    if payload.canonical_field is None and payload.active is None:
        raise HTTPException(status_code=422, detail="Alias update is empty")
    canonical_field = (
        _validate_alias_field(payload.canonical_field)
        if payload.canonical_field is not None
        else None
    )

    def mutation(current: dict[str, Any]) -> dict[str, Any]:
        alias = next(
            (
                item
                for item in current["aliases"]
                if item["alias_id"] == alias_id and item["hospital_id"] == hospital_id
            ),
            None,
        )
        if alias is None:
            raise AliasRegistryItemNotFound(alias_id)
        old_canonical_field = alias["canonical_field"]
        old_active = alias["active"]
        if canonical_field is not None:
            alias["canonical_field"] = canonical_field
        if payload.active is not None:
            alias["active"] = payload.active
        alias["reason"] = payload.reason.strip()
        alias["updated_at"] = utc_now()
        current["events"].append(
            _registry_event(
                "column_alias_updated",
                payload.reason,
                alias_id=alias_id,
                hospital_id=hospital_id,
                old_canonical_field=old_canonical_field,
                new_canonical_field=alias["canonical_field"],
                old_active=old_active,
                new_active=alias["active"],
            )
        )
        return current

    try:
        updated = _alias_coordinator().mutate_registry(
            payload.registry_revision,
            mutation,
        )
    except AliasRegistryItemNotFound as error:
        raise HTTPException(status_code=404, detail="Column alias was not found") from error
    except AliasRegistryRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "alias_registry_unavailable"},
        ) from error
    alias = next(item for item in updated["aliases"] if item["alias_id"] == alias_id)
    return {"registry_revision": updated["revision"], "alias": alias}


@app.patch("/api/v2/documents/{job_id}/rows/bulk")
def bulk_update_rows_route(
    job_id: str,
    payload: BulkRowsPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    return _bulk_update_rows(job_id, payload, if_match)


def _apply_reviewer_disposition(
    result: dict[str, Any],
    review: dict[str, Any],
    row_ids: tuple[str, ...],
    action: Literal["reject", "restore"],
    reason: str,
) -> dict[str, dict[str, str]]:
    rows = {str(row["id"]): row for row in project_rows(result, review)}
    missing = [row_id for row_id in row_ids if row_id not in rows]
    if missing:
        raise ReviewValidationError(f"Rows not found: {', '.join(missing)}")
    ineligible = [row_id for row_id in row_ids if rows[row_id].get("bulk_action") != action]
    if ineligible:
        verb = "rejectable" if action == "reject" else "restorable"
        raise ReviewValidationError(f"Rows are not {verb}: {', '.join(ineligible)}")

    changed: dict[str, dict[str, str]] = {}
    machine_rows = {str(row["id"]): row for row in result.get("rows", [])}
    for row_id in row_ids:
        row = rows[row_id]
        if row_id in review["added_rows"]:
            added = review["added_rows"][row_id]
            old_disposition = str(row.get("review_disposition") or "accepted")
            if action == "reject":
                added["rejection"] = {
                    "source": "reviewer",
                    "previous_disposition": old_disposition,
                    "rejected_at": utc_now(),
                    "reason": reason.strip(),
                }
                added["review_disposition"] = "rejected"
                new_disposition = "rejected"
            else:
                rejection = added.pop("rejection", None)
                legacy = added.pop("pre_rejection_disposition", None)
                new_disposition = str(
                    rejection.get("previous_disposition", "accepted")
                    if isinstance(rejection, dict)
                    else legacy or "accepted"
                )
                added["review_disposition"] = new_disposition
            added["review_reason"] = reason.strip()
        else:
            machine = machine_rows[row_id]
            machine_disposition = str(machine.get("review_disposition") or "pending")
            previous_override = review["row_overrides"].get(row_id, {})
            previous_changes = dict(previous_override.get("changes", {}))
            old_disposition = str(row.get("review_disposition") or machine_disposition)
            if action == "reject":
                previous_changes["review_disposition"] = "rejected"
                rejection = {
                    "source": "reviewer",
                    "previous_disposition": old_disposition,
                    "rejected_at": utc_now(),
                    "reason": reason.strip(),
                }
                new_disposition = "rejected"
            else:
                saved = previous_override.get("rejection")
                legacy = previous_override.get("pre_rejection_disposition")
                new_disposition = str(
                    saved.get("previous_disposition", machine_disposition)
                    if isinstance(saved, dict)
                    else legacy or machine_disposition
                )
                if new_disposition == machine_disposition:
                    previous_changes.pop("review_disposition", None)
                else:
                    previous_changes["review_disposition"] = new_disposition
                rejection = None
            updated_override = {
                "changes": previous_changes,
                "reason": reason.strip(),
                "updated_at": utc_now(),
            }
            if rejection is not None:
                updated_override["rejection"] = rejection
            review["row_overrides"][row_id] = updated_override
        changed[row_id] = {
            "old_disposition": old_disposition,
            "new_disposition": new_disposition,
        }
    return changed


def _apply_direct_disposition(
    result: dict[str, Any],
    review: dict[str, Any],
    row_id: str,
    desired_disposition: str,
    reason: str,
) -> dict[str, str]:
    """Apply an explicit editor correction without invoking reviewer Restore."""
    rows = {str(row["id"]): row for row in project_rows(result, review)}
    if row_id not in rows:
        raise ReviewValidationError("Row not found")
    old_disposition = str(rows[row_id].get("review_disposition") or "pending")
    if row_id in review["added_rows"]:
        added = review["added_rows"][row_id]
        added["review_disposition"] = desired_disposition
        added.pop("rejection", None)
        added.pop("pre_rejection_disposition", None)
        added["review_reason"] = reason.strip()
    else:
        machine = next(row for row in result.get("rows", []) if str(row["id"]) == row_id)
        machine_disposition = str(machine.get("review_disposition") or "pending")
        previous = review["row_overrides"].get(row_id, {})
        direct_changes = dict(previous.get("changes", {}))
        if desired_disposition == machine_disposition:
            direct_changes.pop("review_disposition", None)
        else:
            direct_changes["review_disposition"] = desired_disposition
        review["row_overrides"][row_id] = {
            "changes": direct_changes,
            "reason": reason.strip(),
            "updated_at": utc_now(),
        }
    return {
        "old_disposition": old_disposition,
        "new_disposition": desired_disposition,
    }


@app.patch("/api/v2/documents/{job_id}/rows/{row_id}")
def update_row(
    job_id: str,
    row_id: str,
    payload: RowPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)
    try:
        changes = normalize_changes(payload.changes) if payload.changes else {}
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if (payload.page_number is None) != (payload.polygon is None):
        raise HTTPException(
            status_code=422,
            detail="page_number and polygon must be supplied together",
        )
    if payload.page_number is not None and payload.polygon is not None:
        evidence = evidence_for_page(
            result,
            payload.page_number,
            [point.model_dump() for point in payload.polygon.points],
        )
        changes.update(
            evidence=[evidence],
            field_evidence={"description": [evidence], "amount": [evidence]},
            page_number=payload.page_number,
        )
    if not changes:
        raise HTTPException(status_code=422, detail="At least one row value must change")

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        rows = {str(row["id"]): row for row in project_rows(result, review)}
        if row_id not in rows:
            raise ReviewValidationError("Row not found")
        row_changes = dict(changes)
        audit_changes = dict(changes)
        desired_disposition = row_changes.get("review_disposition")
        current_disposition = str(rows[row_id].get("review_disposition") or "pending")
        if desired_disposition == current_disposition:
            row_changes.pop("review_disposition", None)
            audit_changes.pop("review_disposition", None)
        elif desired_disposition == "rejected":
            transition = _apply_reviewer_disposition(
                result, review, (row_id,), "reject", payload.reason
            )[row_id]
            audit_changes["review_disposition"] = transition
            row_changes.pop("review_disposition", None)
        elif desired_disposition is not None:
            transition = _apply_direct_disposition(
                result,
                review,
                row_id,
                str(desired_disposition),
                payload.reason,
            )
            audit_changes["review_disposition"] = transition
            row_changes.pop("review_disposition", None)
        if isinstance(row_changes.get("field_evidence"), dict):
            row_changes["field_evidence"] = {
                **(rows[row_id].get("field_evidence") or {}),
                **row_changes["field_evidence"],
            }
        if not row_changes and not audit_changes:
            raise ReviewValidationError("At least one row value must change")
        if row_id in review["added_rows"]:
            review["added_rows"][row_id].update(row_changes)
            review["added_rows"][row_id]["review_reason"] = payload.reason.strip()
        else:
            previous = review["row_overrides"].get(row_id, {})
            review["row_overrides"][row_id] = _merge_row_override(
                previous,
                row_changes,
                payload.reason,
            )
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_updated", row_id, payload.reason, audit_changes)
        )
        return review

    review = _mutate(job_id, expected, mutation)
    row = next(item for item in project_rows(result, review) if str(item["id"]) == row_id)
    return {"review_revision": review["revision"], "row": row}


@app.post("/api/v2/documents/{job_id}/rows", status_code=status.HTTP_201_CREATED)
def add_row(
    job_id: str,
    payload: RowCreate,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        row = reviewer_row(
            result,
            project_rows(result, review),
            payload.values,
            payload.page_number,
            [point.model_dump() for point in payload.polygon.points],
            payload.reason.strip(),
        )
        review["added_rows"][row["id"]] = row
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_added", row["id"], payload.reason, payload.values)
        )
        return review

    review = _mutate(job_id, expected, mutation)
    added_id = review["events"][-1]["target_id"]
    added = review["added_rows"][added_id]
    return {"review_revision": review["revision"], "row": added}


@app.delete("/api/v2/documents/{job_id}/rows/{row_id}")
def reject_row(
    job_id: str,
    row_id: str,
    reason: Annotated[str, Query(min_length=3, max_length=500)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        changes = _apply_reviewer_disposition(result, review, (row_id,), "reject", reason)
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_rejected", row_id, reason, changes[row_id])
        )
        return review

    review = _mutate(job_id, expected, mutation)
    return {"review_revision": review["revision"], "rejected_row_id": row_id}


def _bulk_update_rows(
    job_id: str,
    payload: BulkRowsPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)
    row_ids = tuple(dict.fromkeys(item.strip() for item in payload.row_ids if item.strip()))
    if len(row_ids) != len(payload.row_ids):
        raise HTTPException(status_code=422, detail="Row IDs must be unique and non-empty")

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        changes_by_row = _apply_reviewer_disposition(
            result, review, row_ids, payload.action, payload.reason
        )

        review["approval"] = None
        review["events"].append(
            _event(
                expected + 1,
                f"rows_{'rejected' if payload.action == 'reject' else 'restored'}",
                job_id,
                payload.reason,
                {
                    "row_ids": list(row_ids),
                    "count": len(row_ids),
                    "rows": changes_by_row,
                },
            )
        )
        return review

    review = _mutate(job_id, expected, mutation)
    return {
        "review_revision": review["revision"],
        "action": payload.action,
        "row_ids": list(row_ids),
        "updated_count": len(row_ids),
    }


@app.patch("/api/v2/documents/{job_id}/issues/{issue_id}")
def update_issue(
    job_id: str,
    issue_id: str,
    payload: IssuePatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        if issue_id not in {issue["id"] for issue in structural_issues(result, review)}:
            raise ReviewValidationError("Review issue not found")
        review["issue_overrides"][issue_id] = {
            "status": payload.status,
            "reason": payload.reason.strip(),
            "updated_at": utc_now(),
        }
        review["approval"] = None
        review["events"].append(
            _event(
                expected + 1,
                f"issue_{payload.status}",
                issue_id,
                payload.reason,
                {"status": payload.status},
            )
        )
        return review

    review = _mutate(job_id, expected, mutation)
    issue = next(item for item in structural_issues(result, review) if item["id"] == issue_id)
    return {"review_revision": review["revision"], "issue": issue}


@app.post("/api/v2/documents/{job_id}/approval")
def approve_document(
    job_id: str,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    _complete_result(job_id)
    expected = _expected_revision(if_match)

    def mutation(
        review: dict[str, Any],
        state: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        blockers = approval_blockers(
            store,
            job_id,
            result,
            review,
            state=state,
        )
        if blockers:
            raise ApprovalBlocked(blockers)
        certification = state["certification"]
        approval = {
            "status": "approved",
            "reviewer": "demo-reviewer",
            "approved_at": utc_now(),
            "review_revision": expected + 1,
            **{
                field: certification[field]
                for field in (
                    "certification_sha256",
                    "result_sha256",
                    "report_sha256",
                    "source_sha256",
                    "artifact_inventory_sha256",
                )
            },
        }
        if certification.get("artifact_graph_sha256") is not None:
            approval["artifact_graph_sha256"] = certification["artifact_graph_sha256"]
        review["approval"] = approval
        review["events"].append(
            _event(expected + 1, "document_approved", job_id, "Review complete")
        )
        return review

    review = _mutate_with_workspace(job_id, expected, mutation)
    return {"review_revision": review["revision"], "approval": review["approval"]}


def _require_effective_approval(
    job_id: str,
    state: dict[str, Any],
    result: dict[str, Any],
    review: dict[str, Any],
) -> None:
    blockers = approval_blockers(
        store,
        job_id,
        result,
        review,
        state=state,
    )
    if "legacy_uncertified" in blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "legacy_uncertified",
                "blockers": blockers,
            },
        )
    if "certification_invalid" in blockers:
        raise HTTPException(
            status_code=409,
            detail={"code": "certification_invalid", "blockers": blockers},
        )
    if review.get("approval") is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "document_not_approved"},
        )
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={"code": "approval_no_longer_effective", "blockers": blockers},
        )


@app.get("/api/v2/documents/{job_id}/exports/{export_format}")
def export_document(job_id: str, export_format: Literal["csv", "json", "evidence.zip"]) -> Response:
    if export_format == "evidence.zip":

        def build_evidence_export() -> tuple[str, Path]:
            with store.locked_workspace(job_id) as (state, result, review):
                if state.get("status") not in {"complete", "needs_review"}:
                    raise HTTPException(
                        status_code=409,
                        detail="Extraction is not complete",
                    )
                _require_effective_approval(job_id, state, result, review)
                stem = _download_stem(state["original_name"])
                return stem, create_evidence_bundle(store, job_id, state, result, review)

        try:
            stem, bundle = _alias_coordinator().run_job_operation(job_id, build_evidence_export)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        except JobTransactionError as error:
            raise HTTPException(
                status_code=409,
                detail="Document workspace is temporarily unavailable",
            ) from error
        except AliasRegistryUnavailable as error:
            raise HTTPException(
                status_code=503, detail={"code": "alias_registry_unavailable"}
            ) from error
        except ReviewValidationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return FileResponse(
            bundle,
            media_type="application/zip",
            filename=f"{stem}-evidence.zip",
        )

    def build_regular_export() -> tuple[str, str]:
        with store.locked_workspace(job_id) as (state, result, review):
            if state.get("status") not in {"complete", "needs_review"}:
                raise HTTPException(
                    status_code=409,
                    detail="Extraction is not complete",
                )
            _require_effective_approval(job_id, state, result, review)
            stem = _download_stem(state["original_name"])
            if export_format == "csv":
                return stem, export_csv(result, review)
            return (
                stem,
                json.dumps(export_payload(result, review), indent=2, sort_keys=True) + "\n",
            )

    try:
        stem, content = _alias_coordinator().run_job_operation(
            job_id,
            build_regular_export,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail="Document workspace is temporarily unavailable",
        ) from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error
    if export_format == "csv":
        return Response(
            content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}-reviewed.csv"'},
        )
    if export_format == "json":
        return Response(
            content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}-reviewed.json"'},
        )
    raise HTTPException(status_code=400, detail="Unsupported export format")


@app.get("/api/v2/documents/{job_id}/pages/{page_number}")
def get_page(job_id: str, page_number: int) -> Response:
    try:
        state, page = _alias_coordinator().run_job_operation(
            job_id, lambda: store.read_page_bytes(job_id, page_number)
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        if str(error) == "page_not_found":
            raise HTTPException(status_code=404, detail="Page not found") from error
        raise HTTPException(status_code=409, detail="Page is not available") from error
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error
    if state.get("status") not in {"processing", "complete", "needs_review"}:
        raise HTTPException(status_code=409, detail="Page is not available")
    return Response(content=page, media_type="image/png")


@app.delete("/api/v2/documents/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(job_id: str) -> None:
    _state_or_404(job_id)
    try:
        _alias_coordinator().delete_job(job_id)
    except AliasRegistryUnavailable as error:
        raise HTTPException(
            status_code=503, detail={"code": "alias_registry_unavailable"}
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=409,
            detail="Active extraction cannot be deleted",
        ) from error


@app.post("/api/v2/documents/{job_id}/abort", status_code=status.HTTP_202_ACCEPTED)
def abort_document(job_id: str) -> dict[str, str]:
    _state_or_404(job_id)
    try:
        state = store.request_abort(job_id)
    except RuntimeError as error:
        raise HTTPException(
            status_code=409,
            detail="Only queued or processing documents can be aborted",
        ) from error
    return {"id": state["id"], "status": "cancelling"}
