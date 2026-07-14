from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, Literal

import fitz
from fastapi import FastAPI, File, Header, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from gmoney.demo.review import (
    ReviewValidationError,
    approval_blockers,
    create_evidence_bundle,
    evidence_for_page,
    export_csv,
    export_payload,
    load_result,
    normalize_changes,
    project_rows,
    public_page_assets,
    review_summary,
    reviewer_row,
    structural_issues,
)
from gmoney.demo.store import JobStore, ReviewRevisionConflict, utc_now

MAX_UPLOAD_BYTES = int(os.environ.get("GMONEY_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_ACTIVE_JOBS = int(os.environ.get("GMONEY_MAX_ACTIVE_JOBS", "20"))
MAX_PDF_PAGES = int(os.environ.get("GMONEY_MAX_PDF_PAGES", "200"))
WORKER_CAPACITY = int(os.environ.get("GMONEY_WORKER_CAPACITY", "2"))
DEMO_ROOT = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
store = JobStore(DEMO_ROOT)

app = FastAPI(
    title="GMoney V2 Evidence Demo",
    version="0.3.0",
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
)


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


class IssuePatch(BaseModel):
    status: Literal["open", "resolved"]
    reason: str = Field(min_length=3, max_length=500)


@app.middleware("http")
async def private_demo_responses(request: Any, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
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
            "error",
        )
    }


def _state_or_404(job_id: str) -> dict[str, Any]:
    try:
        return store.read(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error


def _complete_result(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _state_or_404(job_id)
    if state.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    try:
        return load_result(store, job_id), store.read_review(job_id)
    except ReviewValidationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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


def _download_stem(original_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip("-._")
    return normalized[:100] or "bill"


@app.get("/api/v2/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v2/health/ready", tags=["health"])
def ready() -> dict[str, Any]:
    return {
        "status": "ready",
        "active_jobs": store.active_count(),
        "worker_capacity": WORKER_CAPACITY,
        "queue_capacity": MAX_ACTIVE_JOBS,
    }


@app.post("/api/v2/documents", status_code=status.HTTP_202_ACCEPTED)
async def create_document(
    file: Annotated[UploadFile, File(description="Hospital bill PDF")],
) -> dict[str, Any]:
    if store.active_count() >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=429, detail="Demo queue is full")
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
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 25 MiB limit")
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
    document_status: Annotated[
        Literal["uploading", "queued", "processing", "complete", "failed"] | None,
        Query(alias="status"),
    ] = None,
) -> dict[str, Any]:
    states = store.states()
    if document_status is not None:
        states = [state for state in states if state.get("status") == document_status]
    states.sort(key=lambda state: str(state.get("created_at") or ""), reverse=True)
    return {
        "total": len(states),
        "documents": [_public_state(state) for state in states[:limit]],
    }


@app.get("/api/v2/documents/{job_id}")
def get_document(job_id: str) -> dict[str, Any]:
    return _public_state(_state_or_404(job_id))


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
    if query:
        needle = query.casefold().strip()
        rows = [
            row
            for row in rows
            if needle in str(row.get("description") or "").casefold()
            or needle in str(row.get("service_code") or "").casefold()
            or needle in str(row.get("section") or "").casefold()
        ]
    if disposition:
        rows = [row for row in rows if row.get("review_disposition") == disposition]
    if source_page:
        rows = [row for row in rows if row.get("page_number") == source_page]
    total = len(rows)
    return {
        "document_id": result["document_id"],
        "pages": result["pages"],
        "page_assets": public_page_assets(result),
        "review_revision": review["revision"],
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": rows[offset : offset + limit],
    }


@app.get("/api/v2/documents/{job_id}/review")
def get_review(job_id: str) -> dict[str, Any]:
    result, review = _complete_result(job_id)
    return review_summary(result, review)


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
            review["added_rows"][row_id]["review_disposition"] = "rejected"
            review["added_rows"][row_id]["review_reason"] = reason.strip()
        else:
            previous = review["row_overrides"].get(row_id, {})
            review["row_overrides"][row_id] = {
                "changes": {**previous.get("changes", {}), "review_disposition": "rejected"},
                "reason": reason.strip(),
                "updated_at": utc_now(),
            }
        review["approval"] = None
        review["events"].append(
            _event(expected + 1, "row_rejected", row_id, reason, {"review_disposition": "rejected"})
        )
        return review

    review = _mutate(job_id, expected, mutation)
    return {"review_revision": review["revision"], "rejected_row_id": row_id}


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
    try:
        bundle = create_evidence_bundle(store, job_id, result, review)
    except ReviewValidationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return FileResponse(
        bundle,
        media_type="application/zip",
        filename=f"{stem}-evidence.zip",
    )


@app.get("/api/v2/documents/{job_id}/pages/{page_number}")
def get_page(job_id: str, page_number: int) -> FileResponse:
    state = _state_or_404(job_id)
    if state.get("status") not in {"processing", "complete"}:
        raise HTTPException(status_code=409, detail="Page is not available")
    try:
        result = load_result(store, job_id)
    except ReviewValidationError as error:
        raise HTTPException(status_code=409, detail="Page is not available") from error
    asset = next(
        (item for item in result.get("page_assets", []) if item["page_number"] == page_number),
        None,
    )
    if not asset:
        raise HTTPException(status_code=404, detail="Page not found")
    artifact_root = (store.job_dir(job_id) / "artifacts").resolve()
    path = (artifact_root / asset["relative_path"]).resolve()
    if artifact_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(path, media_type="image/png")


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
