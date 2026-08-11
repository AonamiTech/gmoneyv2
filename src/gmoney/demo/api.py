from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, Literal

import fitz
from fastapi import FastAPI, File, Header, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from gmoney.contracts.extraction import SourceTable
from gmoney.contracts.phase3 import ProfileLifecycle
from gmoney.demo.review import (
    ReviewValidationError,
    approval_blockers,
    create_evidence_bundle,
    evidence_for_page,
    export_csv,
    export_payload,
    normalize_changes,
    project_hospital,
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
from gmoney.extraction.canonicalize import parse_service_date
from gmoney.extraction.typed_values import parse_decimal, parse_quantity
from gmoney.profiles.aliases import (
    ALIAS_CANONICAL_FIELDS,
    CANONICAL_TO_SOURCE_FIELD,
    AliasRegistryRevisionConflict,
    JsonAliasRepository,
    normalize_alias,
)
from gmoney.profiles.repository import JsonProfileRepository

MAX_UPLOAD_BYTES = int(os.environ.get("GMONEY_MAX_UPLOAD_BYTES", "0"))
MAX_ACTIVE_JOBS = int(os.environ.get("GMONEY_MAX_ACTIVE_JOBS", "20"))
MAX_PDF_PAGES = int(os.environ.get("GMONEY_MAX_PDF_PAGES", "200"))
WORKER_CAPACITY = int(os.environ.get("GMONEY_WORKER_CAPACITY", "2"))
RETENTION_HOURS = int(os.environ.get("GMONEY_RETENTION_HOURS", "720"))
MIN_FREE_BYTES = int(os.environ.get("GMONEY_MIN_FREE_BYTES", "0"))
DEMO_ROOT = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
PROFILE_REGISTRY = Path(
    os.environ.get("GMONEY_PROFILE_REGISTRY", str(DEMO_ROOT / "profile-registry.json"))
)
ALIAS_REGISTRY = Path(
    os.environ.get("GMONEY_ALIAS_REGISTRY", str(DEMO_ROOT / "alias-registry.json"))
)
store = JobStore(DEMO_ROOT)

app = FastAPI(
    title="GMoney V2 Evidence Demo",
    version="0.4.0",
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
)


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
    reason: str = Field(min_length=3, max_length=500)


class ColumnAliasPreviewRequest(BaseModel):
    hospital_id: str = Field(min_length=1, max_length=200)
    source_label: str = Field(min_length=1, max_length=200)
    canonical_field: str = Field(min_length=1, max_length=100)


class ColumnAliasApplyRequest(ColumnAliasPreviewRequest):
    preview_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    registry_revision: int = Field(ge=0)
    overwrite_row_ids: list[str] = Field(default_factory=list, max_length=500)
    reason: str = Field(min_length=3, max_length=500)


class ColumnAliasPatch(BaseModel):
    canonical_field: str | None = Field(default=None, min_length=1, max_length=100)
    active: bool | None = None
    reason: str = Field(min_length=3, max_length=500)


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
            "error",
        )
    }
    public["hospital_name_source"] = "machine" if state.get("hospital_name") else None
    if state.get("status") == "complete":
        try:
            hospital_override = (
                store.read_review(state["id"]).get("document_overrides", {}).get("hospital")
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
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error


def _complete_result(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        _recover_alias_operation(job_id)
        state, result, review = store.read_workspace(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        raise HTTPException(
            status_code=409,
            detail="Document workspace is temporarily unavailable",
        ) from error
    if state.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Extraction is not complete")
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
        return store.mutate_review(job_id, expected, mutation)
    except ReviewRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "review_revision_conflict",
                "current_revision": error.current_revision,
            },
        ) from error
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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


def _alias_repository() -> JsonAliasRepository:
    return JsonAliasRepository(ALIAS_REGISTRY)


def _alias_journal_path(job_id: str) -> Path:
    return store.job_dir(job_id) / ".alias-operation.json"


def _write_alias_journal(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _recover_alias_operation(job_id: str) -> None:
    journal_path = _alias_journal_path(job_id)
    if not journal_path.is_file():
        return
    repo = _alias_repository()
    with repo.locked(exclusive=True), store.job_lock(job_id, exclusive=True):
        if not journal_path.is_file():
            return
        journal = json.loads(journal_path.read_text())
        if (
            journal.get("version") != "alias_operation_v1"
            or journal.get("job_id") != job_id
        ):
            raise JobTransactionError("unsupported_alias_operation_journal")
        review = journal.get("review")
        registry = journal.get("registry")
        if not isinstance(review, dict) or not isinstance(registry, dict):
            raise JobTransactionError("invalid_alias_operation_journal")
        repo._validate(registry)
        store._restore_payload(store.job_dir(job_id) / "review.json", review)
        repo._write_unlocked(registry)
        journal_path.unlink()


def _recover_alias_operations() -> None:
    for journal in store.jobs_root.glob("*/.alias-operation.json"):
        _recover_alias_operation(journal.parent.name)


def _alias_snapshot() -> dict[str, Any]:
    try:
        _recover_alias_operations()
        return _alias_repository().read()
    except (JobTransactionError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=503,
            detail="The hospital alias registry is unavailable",
        ) from error


def _mutate_review_and_alias_registry(
    job_id: str,
    expected_review_revision: int,
    expected_registry_revision: int,
    mutation: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = _alias_repository()
    try:
        with repo.locked(exclusive=True) as registry:
            if registry["revision"] != expected_registry_revision:
                raise AliasRegistryRevisionConflict(registry["revision"])
            with store.job_lock(job_id, exclusive=True):
                store._require_stable_workspace(job_id)
                review = store._read_review_unlocked(job_id)
                if review["revision"] != expected_review_revision:
                    raise ReviewRevisionConflict(review["revision"])
                updated_review, updated_registry = mutation(
                    json.loads(json.dumps(review)),
                    json.loads(json.dumps(registry)),
                )
                updated_review["revision"] = review["revision"] + 1
                updated_review["updated_at"] = utc_now()
                updated_registry["revision"] = registry["revision"] + 1
                repo._validate(updated_registry)
                journal_path = _alias_journal_path(job_id)
                _write_alias_journal(
                    journal_path,
                    {
                        "version": "alias_operation_v1",
                        "job_id": job_id,
                        "review": updated_review,
                        "registry": updated_registry,
                    },
                )
                repo._write_unlocked(updated_registry)
                store._restore_payload(
                    store.job_dir(job_id) / "review.json", updated_review
                )
                journal_path.unlink()
                return updated_review, updated_registry
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
    except ReviewValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
    value = raw_value.strip()
    if not value:
        return None
    if canonical_field in {
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }:
        parsed = parse_quantity(value) if canonical_field == "quantity" else parse_decimal(value)
        if parsed is None:
            return None
        changes = normalize_changes({canonical_field: str(parsed)})
        evidence_field = {
            "unit_price": "rate",
            "net_amount": "amount",
        }.get(canonical_field, canonical_field)
        return changes, canonical_field, evidence_field
    if canonical_field == "service_date":
        parsed_date = parse_service_date(value)
        if parsed_date is None:
            return None
        return (
            {"service_date_raw": value, "service_date_iso": parsed_date},
            "service_date_iso",
            "service_date",
        )
    changes = normalize_changes({canonical_field: value})
    return changes, canonical_field, canonical_field


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
    normalized_label = normalize_alias(request.source_label)
    if not normalized_label:
        raise HTTPException(status_code=422, detail="Alias label is empty after normalization")
    rows = {str(row["id"]): row for row in project_rows(result, review)}
    candidates: list[dict[str, Any]] = []
    source_column_ids: set[str] = set()
    for table in tables:
        matching_columns = [
            column
            for column in table.columns
            if normalize_alias(column.label) == normalized_label
        ]
        for column in matching_columns:
            source_column_ids.add(column.id)
            for source_row in table.rows:
                cell = next(
                    item for item in source_row.cells if item.column_id == column.id
                )
                base = {
                    "source_row_id": source_row.id,
                    "source_column_id": column.id,
                    "row_id": source_row.canonical_row_id,
                    "source_value": cell.raw_value,
                    "evidence": [item.model_dump(mode="json") for item in cell.evidence],
                }
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
                elif normalize_alias(str(current)) == normalize_alias(str(proposed)):
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
    if not source_column_ids:
        raise HTTPException(status_code=422, detail="Printed header was not found")
    digest_payload = {
        "document_id": result["document_id"],
        "review_revision": review["revision"],
        "registry_revision": registry["revision"],
        "hospital_id": request.hospital_id,
        "source_label": normalized_label,
        "canonical_field": canonical_field,
        "columns": sorted(source_column_ids),
        "rows": [
            {
                key: item.get(key)
                for key in (
                    "source_row_id",
                    "row_id",
                    "source_value",
                    "classification",
                    "current_value",
                    "proposed_value",
                )
            }
            for item in candidates
        ],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    counts = {
        classification: sum(
            item["classification"] == classification for item in candidates
        )
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
        "preview_digest": digest,
        "source_column_ids": sorted(source_column_ids),
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
def ready() -> dict[str, Any]:
    storage = shutil.disk_usage(store.jobs_root)
    return {
        "status": "ready",
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
    try:
        profiles = JsonProfileRepository(PROFILE_REGISTRY).list_profiles()
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=503,
            detail="The trained hospital registry is unavailable",
        ) from error

    aliases = _alias_snapshot()
    hospitals: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if profile.lifecycle is not ProfileLifecycle.ACTIVE or not profile.hospital_id:
            continue
        item = hospitals.setdefault(
            profile.hospital_id,
            {
                "hospital_id": profile.hospital_id,
                "hospital_name": profile.hospital_name or profile.hospital_id,
                "active_profile_count": 0,
                "alias_count": 0,
                "training_sources": ["profile"],
            },
        )
        if profile.hospital_name:
            item["hospital_name"] = profile.hospital_name
        if "profile" not in item["training_sources"]:
            item["training_sources"].append("profile")
        item["active_profile_count"] += 1
    for hospital in aliases["hospitals"]:
        hospital_id = str(hospital["hospital_id"])
        item = hospitals.setdefault(
            hospital_id,
            {
                "hospital_id": hospital_id,
                "hospital_name": hospital["hospital_name"],
                "active_profile_count": 0,
                "alias_count": 0,
                "training_sources": [],
            },
        )
        item["hospital_name"] = hospital["hospital_name"]
        if "reviewer_alias" not in item["training_sources"]:
            item["training_sources"].append("reviewer_alias")
        item["alias_count"] = sum(
            alias["hospital_id"] == hospital_id and alias.get("active", True)
            for alias in aliases["aliases"]
        )
    items = sorted(
        hospitals.values(),
        key=lambda item: (str(item["hospital_name"]).casefold(), item["hospital_id"]),
    )
    return {"total": len(items), "hospitals": items}


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
        Literal["uploading", "queued", "processing", "complete", "failed"] | None,
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
        except (KeyError, JobTransactionError):
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


@app.get("/api/v2/documents/{job_id}/rows")
def get_rows(
    job_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    query: Annotated[str | None, Query(max_length=200)] = None,
    disposition: Annotated[str | None, Query()] = None,
    source_page: Annotated[int | None, Query(ge=1)] = None,
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    rows = project_rows(result, review)
    totals = totals_summary(result, review, rows)
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
) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    source_payload = result.get("source_tables")
    source_present = "source_tables" in result
    available = bool(source_payload)
    unavailable_reason = (
        None
        if available
        else ("no_source_tables" if source_present else "legacy_result")
    )
    try:
        tables = [SourceTable.model_validate(table) for table in source_payload or []]
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
                                str(column_mappings[column.id]["canonical_field"]),
                                str(column_mappings[column.id]["canonical_field"]),
                            )
                        }
                    )
                    if column.id in column_mappings
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
    window = list(enumerate(selected_rows, start=1))[offset : offset + limit]
    rows_by_table: dict[int, list[tuple[Any, int]]] = {}
    for ordinal, (table_index, row) in window:
        rows_by_table.setdefault(table_index, []).append((row, ordinal))
    public_tables: list[dict[str, Any]] = []
    for index, table in enumerate(tables):
        if index not in rows_by_table:
            continue
        payload = table.model_dump(mode="json")
        payload["rows"] = []
        for row, ordinal in rows_by_table[index]:
            row_payload = row.model_dump(mode="json")
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
    result, review = _complete_result(job_id)
    return review_summary(result, review)


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

    repo = _alias_repository()
    snapshot = _alias_snapshot()
    selected_name = str(hospital["name"])
    hospital_id = (
        repo.hospital_id_for_name(selected_name)
        if payload.create
        else str(payload.hospital_id)
    )
    profile_names = {
        str(profile.hospital_id): str(profile.hospital_name or profile.hospital_id)
        for profile in JsonProfileRepository(PROFILE_REGISTRY).list_profiles()
        if profile.lifecycle is ProfileLifecycle.ACTIVE and profile.hospital_id
    }
    existing = next(
        (
            item
            for item in snapshot["hospitals"]
            if item["hospital_id"] == hospital_id
        ),
        None,
    )
    if not payload.create and existing is None and hospital_id not in profile_names:
        raise HTTPException(status_code=422, detail="Selected hospital was not found")
    canonical_name = (
        str(existing["hospital_name"])
        if existing
        else profile_names.get(hospital_id, selected_name)
    )

    def coordinated_mutation(
        review: dict[str, Any], registry: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record = next(
            (
                item
                for item in registry["hospitals"]
                if item["hospital_id"] == hospital_id
            ),
            None,
        )
        normalized_name = normalize_alias(selected_name)
        if record is None:
            record = {
                "hospital_id": hospital_id,
                "hospital_name": canonical_name,
                "normalized_names": [normalize_alias(canonical_name)],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            registry["hospitals"].append(record)
        if normalized_name not in record["normalized_names"]:
            record["normalized_names"].append(normalized_name)
        record["updated_at"] = utc_now()
        registry["events"].append(
            {
                "action": "hospital_linked",
                "hospital_id": hospital_id,
                "document_id": result["document_id"],
                "reason": payload.reason.strip(),
                "created_at": utc_now(),
            }
        )
        review.setdefault("document_overrides", {})["hospital_link"] = {
            "hospital_id": hospital_id,
            "hospital_name": canonical_name,
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
                {"hospital_name": canonical_name},
            )
        )
        return review, registry

    review, updated_registry = _mutate_review_and_alias_registry(
        job_id,
        expected,
        snapshot["revision"],
        coordinated_mutation,
    )
    return {
        "review_revision": review["revision"],
        "registry_revision": updated_registry["revision"],
        "hospital_id": hospital_id,
        "hospital_name": canonical_name,
    }


@app.get("/api/v2/hospitals/{hospital_id}/aliases")
def list_hospital_aliases(hospital_id: str) -> dict[str, Any]:
    snapshot = _alias_snapshot()
    hospital = next(
        (
            item
            for item in snapshot["hospitals"]
            if item["hospital_id"] == hospital_id
        ),
        None,
    )
    if hospital is None:
        raise HTTPException(status_code=404, detail="Hospital was not found")
    aliases = [
        item for item in snapshot["aliases"] if item["hospital_id"] == hospital_id
    ]
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
    preview = _alias_preview_payload(result, review, registry, payload)
    if preview["preview_digest"] != payload.preview_digest:
        raise HTTPException(status_code=409, detail="Alias preview is stale")
    overwrite_ids = tuple(
        dict.fromkeys(item.strip() for item in payload.overwrite_row_ids if item.strip())
    )
    if len(overwrite_ids) != len(payload.overwrite_row_ids):
        raise HTTPException(
            status_code=422,
            detail="Overwrite row IDs must be unique and non-empty",
        )
    conflicting_ids = {
        str(item["row_id"])
        for item in preview["candidates"]
        if item["classification"] == "conflicting" and item.get("row_id")
    }
    unknown_overwrites = set(overwrite_ids) - conflicting_ids
    if unknown_overwrites:
        raise HTTPException(
            status_code=422,
            detail="Only previewed conflicting rows can be overwritten",
        )
    applicable = [
        item
        for item in preview["candidates"]
        if item["classification"] == "fillable"
        or (
            item["classification"] == "conflicting"
            and item.get("row_id") in overwrite_ids
        )
    ]
    unchanged = [
        item
        for item in preview["candidates"]
        if item["classification"] == "unchanged"
    ]
    if not applicable and not unchanged:
        raise HTTPException(
            status_code=422,
            detail="Alias has no valid grounded values to confirm",
        )

    alias_id = next(
        (
            item["alias_id"]
            for item in registry["aliases"]
            if item["hospital_id"] == payload.hospital_id
            and item["normalized_label"] == preview["normalized_label"]
        ),
        JsonAliasRepository.new_alias_id(),
    )

    def coordinated_mutation(
        current: dict[str, Any], snapshot: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not any(
            item["hospital_id"] == payload.hospital_id
            for item in snapshot["hospitals"]
        ):
            raise ValueError("hospital was removed")
        alias = next(
            (item for item in snapshot["aliases"] if item["alias_id"] == alias_id),
            None,
        )
        if alias is None:
            alias = {
                "alias_id": alias_id,
                "hospital_id": payload.hospital_id,
                "source_label": preview["source_label"],
                "normalized_label": preview["normalized_label"],
                "canonical_field": preview["canonical_field"],
                "active": True,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "reason": payload.reason.strip(),
            }
            snapshot["aliases"].append(alias)
        else:
            alias.update(
                source_label=preview["source_label"],
                canonical_field=preview["canonical_field"],
                active=True,
                updated_at=utc_now(),
                reason=payload.reason.strip(),
            )
        snapshot["events"].append(
            {
                "action": "column_alias_applied",
                "alias_id": alias_id,
                "hospital_id": payload.hospital_id,
                "canonical_field": preview["canonical_field"],
                "document_id": result["document_id"],
                "reason": payload.reason.strip(),
                "created_at": utc_now(),
            }
        )
        projected = {str(row["id"]): row for row in project_rows(result, current)}
        current.setdefault("column_mappings", {})
        for column_id in preview["source_column_ids"]:
            current["column_mappings"][column_id] = {
                "alias_id": alias_id,
                "hospital_id": payload.hospital_id,
                "source_label": preview["source_label"],
                "canonical_field": preview["canonical_field"],
                "reason": payload.reason.strip(),
                "updated_at": utc_now(),
            }
        for candidate in applicable:
            row_id = str(candidate["row_id"])
            row = projected[row_id]
            changes = dict(candidate["changes"])
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
                current["row_overrides"][row_id] = {
                    "changes": {**previous.get("changes", {}), **changes},
                    "reason": payload.reason.strip(),
                    "updated_at": utc_now(),
                }
        current["approval"] = None
        current["events"].append(
            _event(
                expected + 1,
                "column_alias_applied",
                alias_id,
                payload.reason,
                {
                    "source_label": preview["source_label"],
                    "canonical_field": preview["canonical_field"],
                    "updated_row_ids": [item["row_id"] for item in applicable],
                },
            )
        )
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
        "alias_id": alias_id,
        "updated_count": len(applicable),
        "skipped": preview["counts"],
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
    repo = _alias_repository()
    snapshot = _alias_snapshot()

    def mutation(current: dict[str, Any]) -> dict[str, Any]:
        alias = next(
            (
                item
                for item in current["aliases"]
                if item["alias_id"] == alias_id
                and item["hospital_id"] == hospital_id
            ),
            None,
        )
        if alias is None:
            raise KeyError(alias_id)
        if canonical_field is not None:
            alias["canonical_field"] = canonical_field
        if payload.active is not None:
            alias["active"] = payload.active
        alias["reason"] = payload.reason.strip()
        alias["updated_at"] = utc_now()
        current["events"].append(
            {
                "action": "column_alias_updated",
                "alias_id": alias_id,
                "hospital_id": hospital_id,
                "canonical_field": alias["canonical_field"],
                "active": alias["active"],
                "reason": payload.reason.strip(),
                "created_at": utc_now(),
            }
        )
        return current

    try:
        updated = repo.mutate(snapshot["revision"], mutation)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Column alias was not found") from error
    except AliasRegistryRevisionConflict as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "alias_registry_revision_conflict",
                "current_revision": error.current_revision,
            },
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


@app.patch("/api/v2/documents/{job_id}/rows/{row_id}")
def update_row(
    job_id: str,
    row_id: str,
    payload: RowPatch,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> dict[str, Any]:
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)
    changes = normalize_changes(payload.changes)
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

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        rows = {str(row["id"]): row for row in project_rows(result, review)}
        if row_id not in rows:
            raise ReviewValidationError("Row not found")
        if row_id in review["added_rows"]:
            review["added_rows"][row_id].update(changes)
            review["added_rows"][row_id]["review_reason"] = payload.reason.strip()
        else:
            previous = review["row_overrides"].get(row_id, {})
            review["row_overrides"][row_id] = {
                "changes": {**previous.get("changes", {}), **changes},
                "reason": payload.reason.strip(),
                "updated_at": utc_now(),
            }
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_updated", row_id, payload.reason, changes)
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
        rows = {str(row["id"]): row for row in project_rows(result, review)}
        if row_id not in rows:
            raise ReviewValidationError("Row not found")
        if row_id in review["added_rows"]:
            review["added_rows"][row_id]["pre_rejection_disposition"] = str(
                review["added_rows"][row_id].get("review_disposition") or "accepted"
            )
            review["added_rows"][row_id]["review_disposition"] = "rejected"
            review["added_rows"][row_id]["review_reason"] = reason.strip()
        else:
            previous = review["row_overrides"].get(row_id, {})
            review["row_overrides"][row_id] = {
                "changes": {**previous.get("changes", {}), "review_disposition": "rejected"},
                "reason": reason.strip(),
                "updated_at": utc_now(),
                "pre_rejection_disposition": str(
                    rows[row_id].get("review_disposition") or "accepted"
                ),
            }
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_rejected", row_id, reason, {"review_disposition": "rejected"})
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
        rows = {str(row["id"]): row for row in project_rows(result, review)}
        missing = [row_id for row_id in row_ids if row_id not in rows]
        if missing:
            raise ReviewValidationError(f"Rows not found: {', '.join(missing)}")
        if payload.action == "restore":
            not_rejected = [
                row_id
                for row_id in row_ids
                if rows[row_id].get("review_disposition") != "rejected"
            ]
            if not_rejected:
                raise ReviewValidationError(
                    f"Rows are not rejected: {', '.join(not_rejected)}"
                )

        changes_by_row: dict[str, dict[str, Any]] = {}
        for row_id in row_ids:
            row = rows[row_id]
            if row_id in review["added_rows"]:
                added = review["added_rows"][row_id]
                if payload.action == "reject":
                    previous = str(added.get("review_disposition") or "accepted")
                    added["pre_rejection_disposition"] = previous
                    added["review_disposition"] = "rejected"
                else:
                    previous = str(added.pop("pre_rejection_disposition", "accepted"))
                    added["review_disposition"] = previous
                added["review_reason"] = payload.reason.strip()
                changes_by_row[row_id] = {
                    "review_disposition": added["review_disposition"]
                }
                continue

            previous_override = review["row_overrides"].get(row_id, {})
            previous_changes = dict(previous_override.get("changes", {}))
            if payload.action == "reject":
                original_disposition = str(
                    row.get("review_disposition") or "accepted"
                )
                previous_changes["review_disposition"] = "rejected"
                pre_rejection = original_disposition
            else:
                machine = next(
                    item for item in result.get("rows", []) if str(item["id"]) == row_id
                )
                pre_rejection = str(
                    previous_override.get("pre_rejection_disposition")
                    or machine.get("review_disposition")
                    or "accepted"
                )
                if pre_rejection == str(machine.get("review_disposition") or "accepted"):
                    previous_changes.pop("review_disposition", None)
                else:
                    previous_changes["review_disposition"] = pre_rejection
            review["row_overrides"][row_id] = {
                "changes": previous_changes,
                "reason": payload.reason.strip(),
                "updated_at": utc_now(),
                **(
                    {"pre_rejection_disposition": pre_rejection}
                    if payload.action == "reject"
                    else {}
                ),
            }
            changes_by_row[row_id] = {
                "review_disposition": (
                    "rejected" if payload.action == "reject" else pre_rejection
                )
            }

        review["approval"] = None
        review["events"].append(
            _event(
                expected + 1,
                f"rows_{'rejected' if payload.action == 'reject' else 'restored'}",
                job_id,
                payload.reason,
                {"row_ids": list(row_ids), "rows": changes_by_row},
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
    result, _ = _complete_result(job_id)
    expected = _expected_revision(if_match)

    def mutation(review: dict[str, Any]) -> dict[str, Any]:
        blockers = approval_blockers(store, job_id, result, review)
        if blockers:
            raise ReviewValidationError(f"Approval blocked: {', '.join(blockers)}")
        approval = {
            "status": "approved",
            "reviewer": "demo-reviewer",
            "approved_at": utc_now(),
            "review_revision": expected + 1,
        }
        review["approval"] = approval
        review["events"].append(
            _event(expected + 1, "document_approved", job_id, "Review complete")
        )
        return review

    review = _mutate(job_id, expected, mutation)
    return {"review_revision": review["revision"], "approval": review["approval"]}


@app.get("/api/v2/documents/{job_id}/exports/{export_format}")
def export_document(job_id: str, export_format: Literal["csv", "json", "evidence.zip"]) -> Response:
    if export_format == "evidence.zip":
        try:
            with store.locked_workspace(job_id) as (state, result, review):
                if state.get("status") != "complete":
                    raise HTTPException(
                        status_code=409,
                        detail="Extraction is not complete",
                    )
                if review.get("approval") is None:
                    raise HTTPException(
                        status_code=409,
                        detail="Document must be approved before export",
                    )
                stem = _download_stem(state["original_name"])
                bundle = create_evidence_bundle(store, job_id, result, review)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        except JobTransactionError as error:
            raise HTTPException(
                status_code=409,
                detail="Document workspace is temporarily unavailable",
            ) from error
        except ReviewValidationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return FileResponse(
            bundle,
            media_type="application/zip",
            filename=f"{stem}-evidence.zip",
        )

    result, review = _complete_result(job_id)
    if review.get("approval") is None:
        raise HTTPException(status_code=409, detail="Document must be approved before export")
    state = _state_or_404(job_id)
    stem = _download_stem(state["original_name"])
    if export_format == "csv":
        return Response(
            export_csv(result, review),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}-reviewed.csv"'},
        )
    if export_format == "json":
        return Response(
            json.dumps(export_payload(result, review), indent=2, sort_keys=True) + "\n",
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}-reviewed.json"'},
        )
    raise HTTPException(status_code=400, detail="Unsupported export format")


@app.get("/api/v2/documents/{job_id}/pages/{page_number}")
def get_page(job_id: str, page_number: int) -> Response:
    try:
        state, page = store.read_page_bytes(job_id, page_number)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except JobTransactionError as error:
        if str(error) == "page_not_found":
            raise HTTPException(status_code=404, detail="Page not found") from error
        raise HTTPException(status_code=409, detail="Page is not available") from error
    if state.get("status") not in {"processing", "complete"}:
        raise HTTPException(status_code=409, detail="Page is not available")
    return Response(content=page, media_type="image/png")


@app.delete("/api/v2/documents/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(job_id: str) -> None:
    _state_or_404(job_id)
    try:
        store.delete(job_id)
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
