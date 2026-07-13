from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from gmoney.demo.store import JobStore

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ACTIVE_JOBS = 20
DEMO_ROOT = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
store = JobStore(DEMO_ROOT)

app = FastAPI(
    title="GMoney V2 Evidence Demo",
    version="0.2.0",
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
)


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


@app.get("/api/v2/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v2/health/ready", tags=["health"])
def ready() -> dict[str, Any]:
    return {"status": "ready", "active_jobs": store.active_count(), "capacity": MAX_ACTIVE_JOBS}


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
        state = store.update(state["id"], status="queued", size_bytes=size)
        return _public_state(state)
    except Exception:
        target.unlink(missing_ok=True)
        with suppress(KeyError):
            store.update(state["id"], status="failed", error="Upload rejected")
        raise
    finally:
        await file.close()


@app.get("/api/v2/documents/{job_id}")
def get_document(job_id: str) -> dict[str, Any]:
    return _public_state(_state_or_404(job_id))


@app.get("/api/v2/documents/{job_id}/rows")
def get_rows(job_id: str) -> dict[str, Any]:
    state = _state_or_404(job_id)
    if state.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Extraction is not complete")
    result_path = store.job_dir(job_id) / "result.json"
    result = json.loads(result_path.read_text())
    return {
        "document_id": result["document_id"],
        "pages": result["pages"],
        "page_assets": [
            {
                key: asset[key]
                for key in ("page_number", "artifact_sha256", "width", "height")
            }
            for asset in result.get("page_assets", [])
        ],
        "rows": result["rows"],
    }


@app.get("/api/v2/documents/{job_id}/pages/{page_number}")
def get_page(job_id: str, page_number: int) -> FileResponse:
    state = _state_or_404(job_id)
    if state.get("status") not in {"processing", "complete"}:
        raise HTTPException(status_code=409, detail="Page is not available")
    result_path = store.job_dir(job_id) / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=409, detail="Page is not available")
    result = json.loads(result_path.read_text())
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
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


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
