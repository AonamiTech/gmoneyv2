from __future__ import annotations

import json
from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from gmoney.demo import api
from gmoney.demo.store import JobStore


def client_for(tmp_path: Path, monkeypatch) -> tuple[TestClient, JobStore]:
    store = JobStore(tmp_path)
    monkeypatch.setattr(api, "store", store)
    return TestClient(api.app), store


def pdf_bytes(pages: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=300, height=400)
    payload = document.tobytes()
    document.close()
    return payload


def test_pdf_upload_status_and_shared_listing_endpoint(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    response = client.post(
        "/api/v2/documents",
        files={"file": ("bill.pdf", pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert (store.job_dir(payload["id"]) / "source.pdf").read_bytes().startswith(b"%PDF-")
    assert client.get(f"/api/v2/documents/{payload['id']}").status_code == 200
    listing = client.get("/api/v2/documents")
    assert listing.status_code == 200
    assert listing.json()["documents"][0]["id"] == payload["id"]


def test_upload_rejects_extension_signature_and_oversize(tmp_path: Path, monkeypatch) -> None:
    client, _ = client_for(tmp_path, monkeypatch)
    assert (
        client.post(
            "/api/v2/documents",
            files={"file": ("bill.txt", pdf_bytes(), "text/plain")},
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
            files={"file": ("bill.pdf", pdf_bytes(), "application/pdf")},
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


def completed_job(
    store: JobStore,
    *,
    description: str = "Consultation",
    diagnostics: list[dict[str, object]] | None = None,
) -> tuple[str, dict[str, object]]:
    state = store.create("client-bill.pdf")
    job_id = state["id"]
    artifact = store.job_dir(job_id) / "artifacts" / "pages" / "page-1.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"PNG fixture")
    evidence = {
        "page_number": 1,
        "table_id": "p1-t1",
        "polygon": {
            "points": [
                {"x": 10, "y": 10},
                {"x": 90, "y": 10},
                {"x": 90, "y": 30},
                {"x": 10, "y": 30},
            ]
        },
        "artifact_sha256": "a" * 64,
        "token_ids": ["one"],
    }
    row = {
        "id": "machine-row",
        "contract_version": "canonical_row_v1",
        "created_at": "2026-07-14T00:00:00Z",
        "document_id": "d" * 64,
        "page_number": 1,
        "table_id": "p1-t1",
        "page_type": "itemized_charges",
        "table_type": "item_ledger",
        "row_order": 0,
        "role": "detail",
        "review_disposition": "accepted",
        "section": None,
        "description": description,
        "service_date_raw": None,
        "service_date_iso": None,
        "request_no": None,
        "service_code": None,
        "hsn_code": None,
        "quantity_raw": "1",
        "quantity": "1",
        "unit_price_raw": "100.00",
        "unit_price": "100.00",
        "gross_amount_raw": "100.00",
        "gross_amount": "100.00",
        "discount_raw": "0.00",
        "discount": "0.00",
        "net_amount_raw": "100.00",
        "net_amount": "100.00",
        "evidence": [evidence],
        "field_evidence": {"description": [evidence], "amount": [evidence]},
        "candidate_ids": [],
        "source_routes": ["ocr_spatial_graph"],
        "validation_flags": [],
    }
    result: dict[str, object] = {
        "output_version": "offline_accuracy_spine_v3",
        "document_id": "d" * 64,
        "source_sha256": "e" * 64,
        "source_name": "client-bill.pdf",
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
        "rows": [row],
        "diagnostics": diagnostics or [],
        "provider_usage": {"gemini_calls": 0},
    }
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    store.update(job_id, status="complete", row_count=1, pages=1, page=1)
    return job_id, result


def test_review_updates_are_revisioned_and_machine_output_is_immutable(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, machine = completed_job(store)

    response = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {"description": "Specialist consultation", "net_amount": "125.50"},
            "reason": "Corrected from source",
        },
    )
    assert response.status_code == 200
    assert response.json()["review_revision"] == 1
    assert response.json()["row"]["net_amount"] == "125.50"
    assert response.json()["row"]["review"]["machine_values"]["description"] == "Consultation"
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == machine

    stale = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={"changes": {"description": "Stale"}, "reason": "Stale browser edit"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["current_revision"] == 1


def test_reviewer_can_add_and_soft_reject_a_grounded_row(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    response = client.post(
        f"/api/v2/documents/{job_id}/rows",
        headers={"If-Match": "0"},
        json={
            "values": {"description": "Added medicine", "net_amount": "42.00"},
            "page_number": 1,
            "polygon": {
                "points": [
                    {"x": 10, "y": 40},
                    {"x": 90, "y": 40},
                    {"x": 90, "y": 60},
                    {"x": 10, "y": 60},
                ]
            },
            "reason": "Missing visible ledger row",
        },
    )
    assert response.status_code == 201
    row_id = response.json()["row"]["id"]
    rejected = client.delete(
        f"/api/v2/documents/{job_id}/rows/{row_id}",
        headers={"If-Match": "1"},
        params={"reason": "Added in error"},
    )
    assert rejected.status_code == 200
    rows = client.get(f"/api/v2/documents/{job_id}/rows", params={"disposition": "rejected"})
    assert rows.json()["total"] == 1
    assert rows.json()["rows"][0]["review"]["source"] == "reviewer"


def test_structural_issue_blocks_approval_then_exports_are_available(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    diagnostic: dict[str, object] = {
        "page_number": 1,
        "table_id": "p1-t1",
        "table_type": "item_ledger",
        "phase3_route": {"reasons": ["low_yield"]},
        "recovery_attempts": [
            {"stage": "review", "status": "pending", "reason": "low_yield"}
        ],
    }
    job_id, _ = completed_job(store, description="=FORMULA", diagnostics=[diagnostic])
    review = client.get(f"/api/v2/documents/{job_id}/review").json()
    assert review["issues_open"] == 1
    issue_id = review["issues"][0]["id"]
    blocked = client.post(
        f"/api/v2/documents/{job_id}/approval", headers={"If-Match": "0"}
    )
    assert blocked.status_code == 422

    resolved = client.patch(
        f"/api/v2/documents/{job_id}/issues/{issue_id}",
        headers={"If-Match": "0"},
        json={"status": "resolved", "reason": "Compared against the visible page"},
    )
    assert resolved.status_code == 200
    approved = client.post(
        f"/api/v2/documents/{job_id}/approval", headers={"If-Match": "1"}
    )
    assert approved.status_code == 200
    assert approved.json()["approval"]["status"] == "approved"

    csv_export = client.get(f"/api/v2/documents/{job_id}/exports/csv")
    assert csv_export.status_code == 200
    assert "'=FORMULA" in csv_export.text
    assert client.get(f"/api/v2/documents/{job_id}/exports/json").status_code == 200
    bundle = client.get(f"/api/v2/documents/{job_id}/exports/evidence.zip")
    assert bundle.status_code == 200
    assert bundle.content.startswith(b"PK")


def test_active_job_cannot_be_deleted(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    state = store.create("bill.pdf")
    store.update(state["id"], status="queued")
    assert client.delete(f"/api/v2/documents/{state['id']}").status_code == 409
    store.update(state["id"], status="failed")
    assert client.delete(f"/api/v2/documents/{state['id']}").status_code == 204


def test_documents_can_be_discovered_in_newest_first_shared_queue(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    first = store.create("first.pdf")
    second = store.create("second.pdf")
    store.update(first["id"], status="complete", row_count=3)
    store.update(second["id"], status="queued")

    response = client.get("/api/v2/documents", params={"limit": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["documents"] == [
        {
            key: store.read(second["id"]).get(key)
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
    ]

    queued = client.get("/api/v2/documents", params={"status": "queued"}).json()
    assert queued["total"] == 1
    assert queued["documents"][0]["id"] == second["id"]


def test_worker_restart_requeues_interrupted_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    state = store.create("bill.pdf")
    store.update(state["id"], status="processing", page=2, pages=4)
    store.recover()
    recovered = store.read(state["id"])
    assert recovered["status"] == "queued"
    assert recovered["page"] == 0
