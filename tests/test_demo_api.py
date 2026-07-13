from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from gmoney.demo import api
from gmoney.demo.store import JobStore


def client_for(tmp_path: Path, monkeypatch) -> tuple[TestClient, JobStore]:
    store = JobStore(tmp_path)
    monkeypatch.setattr(api, "store", store)
    return TestClient(api.app), store


def test_pdf_upload_status_and_no_listing_endpoint(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    response = client.post(
        "/api/v2/documents",
        files={"file": ("bill.pdf", b"%PDF-1.7\nfixture", "application/pdf")},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert (store.job_dir(payload["id"]) / "source.pdf").read_bytes().startswith(b"%PDF-")
    assert client.get(f"/api/v2/documents/{payload['id']}").status_code == 200
    assert client.get("/api/v2/documents").status_code == 405


def test_upload_rejects_extension_signature_and_oversize(tmp_path: Path, monkeypatch) -> None:
    client, _ = client_for(tmp_path, monkeypatch)
    assert (
        client.post(
            "/api/v2/documents",
            files={"file": ("bill.txt", b"%PDF-fixture", "text/plain")},
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/v2/documents",
            files={"file": ("bill.pdf", b"not-a-pdf", "application/pdf")},
        ).status_code
        == 415
    )
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 5)
    assert (
        client.post(
            "/api/v2/documents",
            files={"file": ("bill.pdf", b"%PDF-too-large", "application/pdf")},
        ).status_code
        == 413
    )


def test_completed_rows_and_page_are_scoped_to_uuid(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    state = store.create("bill.pdf")
    job_id = state["id"]
    artifact = store.job_dir(job_id) / "artifacts" / "pages" / "page-1.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"PNG fixture")
    result = {
        "document_id": "d" * 64,
        "pages": 1,
        "page_assets": [
            {
                "page_number": 1,
                "artifact_sha256": "a" * 64,
                "width": 100,
                "height": 200,
                "relative_path": "pages/page-1.png",
            }
        ],
        "rows": [{"id": "row", "description": "Test", "evidence": []}],
        "diagnostics": [{"content": "must never leave the API"}],
    }
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    store.update(job_id, status="complete", row_count=1)

    rows = client.get(f"/api/v2/documents/{job_id}/rows")
    assert rows.status_code == 200
    assert rows.json()["rows"][0]["description"] == "Test"
    assert "diagnostics" not in rows.json()
    assert "relative_path" not in rows.json()["page_assets"][0]
    assert client.get(f"/api/v2/documents/{job_id}/pages/1").status_code == 200
    assert client.get("/api/v2/documents/not-a-uuid").status_code == 404


def test_active_job_cannot_be_deleted(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    state = store.create("bill.pdf")
    store.update(state["id"], status="queued")
    assert client.delete(f"/api/v2/documents/{state['id']}").status_code == 409
    store.update(state["id"], status="failed")
    assert client.delete(f"/api/v2/documents/{state['id']}").status_code == 204


def test_worker_restart_requeues_interrupted_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    state = store.create("bill.pdf")
    store.update(state["id"], status="processing", page=2, pages=4)
    store.recover()
    recovered = store.read(state["id"])
    assert recovered["status"] == "queued"
    assert recovered["page"] == 0
