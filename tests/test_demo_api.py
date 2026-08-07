from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import LayoutProfile, ProfileLifecycle
from gmoney.demo import api
from gmoney.demo.store import JobStore
from gmoney.profiles.repository import JsonProfileRepository


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
    oversized = client.post(
        "/api/v2/documents",
        files={"file": ("bill.pdf", pdf_bytes(), "application/pdf")},
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "PDF exceeds the 5 bytes limit"


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


def test_source_tables_endpoint_returns_dynamic_columns_and_raw_cells(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    evidence = result["rows"][0]["evidence"]
    result["source_tables"] = [
        {
            "id": "p1-t1-s1",
            "page_number": 1,
            "table_id": "p1-t1",
            "table_type": "item_ledger",
            "columns": [
                {
                    "id": "particular",
                    "label": "Particular",
                    "order": 0,
                    "canonical_field": "description",
                    "evidence": evidence,
                },
                {
                    "id": "co-pay",
                    "label": "Co-pay %",
                    "order": 1,
                    "canonical_field": None,
                    "evidence": evidence,
                },
            ],
            "rows": [
                {
                    "id": "p1-t1-s1-r1",
                    "order": 0,
                    "canonical_row_id": "machine-row",
                    "cells": [
                        {
                            "column_id": "particular",
                            "raw_value": "Consultation",
                            "evidence": evidence,
                            "validation_flags": [],
                        },
                        {
                            "column_id": "co-pay",
                            "raw_value": "10",
                            "evidence": evidence,
                            "validation_flags": [],
                        },
                    ],
                    "validation_flags": [],
                }
            ],
            "validation_flags": [],
        }
    ]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    response = client.get(f"/api/v2/documents/{job_id}/source-tables")

    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["unavailable_reason"] is None
    assert response.json()["tables"][0]["rows"][0]["ordinal"] == 1
    assert response.json()["tables"][0]["columns"][1]["label"] == "Co-pay %"
    assert response.json()["tables"][0]["rows"][0]["cells"][1]["raw_value"] == "10"
    assert (
        client.get(f"/api/v2/documents/{job_id}/source-tables?query=missing").json()["total"] == 0
    )

    del result["source_tables"]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    legacy = client.get(f"/api/v2/documents/{job_id}/source-tables")
    assert legacy.status_code == 200
    assert legacy.json()["available"] is False
    assert legacy.json()["unavailable_reason"] == "legacy_result"
    assert legacy.json()["tables"] == []

    result["source_tables"] = []
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    empty = client.get(f"/api/v2/documents/{job_id}/source-tables")
    assert empty.status_code == 200
    assert empty.json()["available"] is False
    assert empty.json()["unavailable_reason"] == "no_source_tables"


def test_source_table_pagination_returns_global_ordinals_across_tables(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    grounded = result["rows"][0]["evidence"]

    def table(table_number: int) -> dict[str, object]:
        return {
            "id": f"p1-t{table_number}-s1",
            "page_number": 1,
            "table_id": f"p1-t{table_number}",
            "table_type": "item_ledger",
            "columns": [
                {
                    "id": "particular",
                    "label": "Particular",
                    "order": 0,
                    "canonical_field": "description",
                    "evidence": grounded,
                }
            ],
            "rows": [
                {
                    "id": f"p1-t{table_number}-s1-r{row_number}",
                    "order": row_number - 1,
                    "canonical_row_id": None,
                    "cells": [
                        {
                            "column_id": "particular",
                            "raw_value": f"Table {table_number} row {row_number}",
                            "evidence": grounded,
                            "validation_flags": [],
                        }
                    ],
                    "validation_flags": [],
                }
                for row_number in (1, 2)
            ],
            "validation_flags": [],
        }

    result["source_tables"] = [table(1), table(2)]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    response = client.get(
        f"/api/v2/documents/{job_id}/source-tables?offset=1&limit=2"
    )

    assert response.status_code == 200
    rows = [
        row_payload
        for table_payload in response.json()["tables"]
        for row_payload in table_payload["rows"]
    ]
    assert [row_payload["ordinal"] for row_payload in rows] == [2, 3]
    assert [row_payload["id"] for row_payload in rows] == [
        "p1-t1-s1-r2",
        "p1-t2-s1-r1",
    ]


def test_source_table_endpoint_rejects_ungrounded_stored_values(
    tmp_path: Path, monkeypatch
) -> None:
    _, store = client_for(tmp_path, monkeypatch)
    client = TestClient(api.app, raise_server_exceptions=False)
    job_id, result = completed_job(store)
    grounded = result["rows"][0]["evidence"]
    result["source_tables"] = [
        {
            "id": "p1-t1-s1",
            "page_number": 1,
            "table_id": "p1-t1",
            "columns": [
                {
                    "id": "c1",
                    "label": "Particular",
                    "order": 0,
                    "evidence": grounded,
                }
            ],
            "rows": [
                {
                    "id": "r1",
                    "order": 0,
                    "cells": [
                        {
                            "column_id": "c1",
                            "raw_value": "invented",
                            "evidence": [],
                        }
                    ],
                }
            ],
        }
    ]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    response = client.get(f"/api/v2/documents/{job_id}/source-tables")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Printed extraction failed grounding validation; reprocess this bill"
    )


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
        "document_total_version": "document_total_v1",
        "document_total": {
            "total_version": "document_total_v1",
            "amount_raw": "120.00",
            "amount": "120.00",
            "label": "Net Bill Amount",
            "page_number": 1,
            "evidence": evidence,
            "confidence": 0.98,
            "source_route": "page_ocr_final_total",
        },
        "document_id": "d" * 64,
        "source_sha256": "e" * 64,
        "source_name": "client-bill.pdf",
        "hospital": {
            "name": "Machine Hospital",
            "confidence": 0.94,
            "source": "machine",
            "page_number": 1,
            "evidence": evidence,
        },
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
    store.update(
        job_id,
        status="complete",
        row_count=1,
        pages=1,
        page=1,
        hospital_name="Machine Hospital",
        hospital_confidence=0.94,
    )
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
    totals = client.get(f"/api/v2/documents/{job_id}/rows").json()["totals"]
    assert totals == {
        "items_total": "125.50",
        "bill_total": {
            "amount": "120.00",
            "label": "Net Bill Amount",
            "kind": None,
            "scope": None,
            "page_number": 1,
            "evidence": machine["document_total"]["evidence"],
        },
        "printed_totals": [
            {
                "amount": "120.00",
                "label": "Net Bill Amount",
                "kind": "bill_total",
                "scope": "document",
                "page_number": 1,
                "evidence": machine["document_total"]["evidence"],
                "is_primary": True,
            }
        ],
        "difference": "5.50",
        "comparison": "mismatch",
        "missing_item_amounts": 0,
    }

    stale = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={"changes": {"description": "Stale"}, "reason": "Stale browser edit"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["current_revision"] == 1


def test_conflicting_explicit_document_totals_are_exposed_without_false_difference(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, machine = completed_job(store)
    primary = {
        **machine["document_total"],
        "kind": "bill_total",
        "scope": "document",
    }
    alternate = {
        **primary,
        "amount_raw": "36,956.00",
        "amount": "36956.00",
        "label": "Net Amount",
    }
    machine["document_total"] = primary
    machine["document_totals"] = [primary, alternate]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(machine))

    totals = client.get(f"/api/v2/documents/{job_id}/rows").json()["totals"]
    assert totals["comparison"] == "multiple_printed_totals"
    assert totals["difference"] is None
    assert len(totals["printed_totals"]) == 2
    assert totals["printed_totals"][0]["is_primary"] is True


def test_informational_rows_are_visible_but_excluded_from_totals_and_approval(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    detail_row = result["rows"][0]
    date_evidence = {
        **detail_row["evidence"][0],
        "token_ids": ["date-token"],
    }
    informational = {
        **detail_row,
        "id": "information-row",
        "row_order": 1,
        "role": "informational",
        "description": "Included package pathology",
        "service_date_raw": "20/01/2026",
        "service_date_iso": "2026-01-20",
        "quantity_raw": None,
        "quantity": None,
        "unit_price_raw": None,
        "unit_price": None,
        "gross_amount_raw": None,
        "gross_amount": None,
        "discount_raw": None,
        "discount": None,
        "net_amount_raw": None,
        "net_amount": None,
        "field_evidence": {
            "description": detail_row["field_evidence"]["description"],
            "service_date": [date_evidence],
        },
    }
    result["rows"].append(informational)
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    store.update(job_id, row_count=2)

    rows = client.get(f"/api/v2/documents/{job_id}/rows").json()
    assert rows["total"] == 2
    assert rows["rows"][1]["role"] == "informational"
    assert rows["rows"][1]["net_amount"] is None
    assert rows["totals"]["items_total"] == "100.00"
    assert rows["totals"]["difference"] == "-20.00"
    assert rows["totals"]["missing_item_amounts"] == 0

    approved = client.post(
        f"/api/v2/documents/{job_id}/approval",
        headers={"If-Match": "0"},
    )
    assert approved.status_code == 200


def test_repeated_package_rollup_matching_bill_total_is_not_double_counted(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    first = result["rows"][0]
    first["role"] = "category_rollup"
    first["net_amount_raw"] = "120.00"
    first["net_amount"] = "120.00"
    repeated = {
        **first,
        "id": "repeated-package-rollup",
        "row_order": 1,
        "page_number": 2,
        "description": "Package Name: Consultation",
    }
    result["rows"].append(repeated)
    result["pages"] = 2
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    store.update(job_id, row_count=2, pages=2, page=2)

    rows = client.get(f"/api/v2/documents/{job_id}/rows").json()

    assert rows["total"] == 2
    assert rows["totals"]["items_total"] == "120.00"
    assert rows["totals"]["difference"] == "0.00"
    assert rows["totals"]["comparison"] == "match"


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
    added_totals = client.get(
        f"/api/v2/documents/{job_id}/rows", params={"query": "Added medicine"}
    ).json()["totals"]
    assert added_totals["items_total"] == "142.00"
    assert added_totals["difference"] == "22.00"
    rejected = client.delete(
        f"/api/v2/documents/{job_id}/rows/{row_id}",
        headers={"If-Match": "1"},
        params={"reason": "Added in error"},
    )
    assert rejected.status_code == 200
    rows = client.get(f"/api/v2/documents/{job_id}/rows", params={"disposition": "rejected"})
    assert rows.json()["total"] == 1
    assert rows.json()["rows"][0]["review"]["source"] == "reviewer"
    assert rows.json()["totals"]["items_total"] == "100.00"
    assert rows.json()["totals"]["difference"] == "-20.00"


def test_hospital_name_correction_is_revisioned_grounded_and_exported(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, machine = completed_job(store)
    payload = {
        "hospital_name": "Reviewer Verified Hospital",
        "page_number": 1,
        "polygon": {
            "points": [
                {"x": 10, "y": 10},
                {"x": 90, "y": 10},
                {"x": 90, "y": 30},
                {"x": 10, "y": 30},
            ]
        },
        "reason": "Verified against the hospital header",
    }
    corrected = client.patch(
        f"/api/v2/documents/{job_id}/metadata",
        headers={"If-Match": "0"},
        json=payload,
    )
    assert corrected.status_code == 200
    assert corrected.json()["review_revision"] == 1
    assert corrected.json()["hospital"]["source"] == "reviewer"
    assert corrected.json()["hospital"]["machine_name"] == "Machine Hospital"
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == machine

    state = client.get(f"/api/v2/documents/{job_id}").json()
    assert state["hospital_name"] == "Reviewer Verified Hospital"
    assert state["hospital_name_source"] == "reviewer"
    rows = client.get(f"/api/v2/documents/{job_id}/rows").json()
    assert rows["hospital"]["name"] == "Reviewer Verified Hospital"

    assert (
        client.post(f"/api/v2/documents/{job_id}/approval", headers={"If-Match": "1"}).status_code
        == 200
    )
    exported = client.get(f"/api/v2/documents/{job_id}/exports/json").json()
    assert exported["hospital"]["name"] == "Reviewer Verified Hospital"
    csv_export = client.get(f"/api/v2/documents/{job_id}/exports/csv").text
    assert "Reviewer Verified Hospital" in csv_export


def test_structural_issue_blocks_approval_then_exports_are_available(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    diagnostic: dict[str, object] = {
        "page_number": 1,
        "table_id": "p1-t1",
        "table_type": "item_ledger",
        "phase3_route": {"reasons": ["low_yield"]},
        "recovery_attempts": [{"stage": "review", "status": "pending", "reason": "low_yield"}],
    }
    job_id, _ = completed_job(store, description="=FORMULA", diagnostics=[diagnostic])
    review = client.get(f"/api/v2/documents/{job_id}/review").json()
    assert review["issues_open"] == 1
    issue_id = review["issues"][0]["id"]
    blocked = client.post(f"/api/v2/documents/{job_id}/approval", headers={"If-Match": "0"})
    assert blocked.status_code == 422

    resolved = client.patch(
        f"/api/v2/documents/{job_id}/issues/{issue_id}",
        headers={"If-Match": "0"},
        json={"status": "resolved", "reason": "Compared against the visible page"},
    )
    assert resolved.status_code == 200
    approved = client.post(f"/api/v2/documents/{job_id}/approval", headers={"If-Match": "1"})
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


def test_active_job_can_be_aborted_idempotently_and_removed(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    state = store.create("bill.pdf")
    store.update(state["id"], status="queued")

    first = client.post(f"/api/v2/documents/{state['id']}/abort")
    assert first.status_code == 202
    assert first.json() == {"id": state["id"], "status": "cancelling"}
    assert store.abort_requested(state["id"])
    assert client.get("/api/v2/documents", params={"scope": "active"}).json()["total"] == 0

    repeated = client.post(f"/api/v2/documents/{state['id']}/abort")
    assert repeated.status_code == 202
    assert store.finalize_abort(state["id"])
    assert not store.job_dir(state["id"]).exists()


def test_completed_job_cannot_be_aborted(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    state = store.create("complete.pdf")
    store.update(state["id"], status="complete")

    response = client.post(f"/api/v2/documents/{state['id']}/abort")

    assert response.status_code == 409
    assert store.read(state["id"])["status"] == "complete"


def test_trained_hospitals_lists_only_active_hospital_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    client, _ = client_for(tmp_path / "jobs", monkeypatch)
    registry = tmp_path / "profiles" / "registry.json"
    repo = JsonProfileRepository(registry)

    def profile(
        key: str,
        hospital_id: str | None,
        hospital_name: str | None,
        lifecycle: ProfileLifecycle,
        *,
        global_family: str | None = None,
    ) -> LayoutProfile:
        return LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key=key,
            profile_version=1,
            lifecycle=lifecycle,
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            global_family=global_family,
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )

    repo.add_profile(
        profile("vijaya-items", "vijaya", "Vijaya Group of Hospitals", ProfileLifecycle.ACTIVE)
    )
    repo.add_profile(
        profile("vijaya-pharmacy", "vijaya", "Vijaya Group of Hospitals", ProfileLifecycle.ACTIVE)
    )
    repo.add_profile(
        profile("candidate", "candidate-hospital", "Candidate Hospital", ProfileLifecycle.CANDIDATE)
    )
    repo.add_profile(
        profile("global", None, None, ProfileLifecycle.ACTIVE, global_family="generic")
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", registry)

    response = client.get("/api/v2/hospitals/trained")

    assert response.status_code == 200
    assert response.json() == {
        "total": 1,
        "hospitals": [
            {
                "hospital_id": "vijaya",
                "hospital_name": "Vijaya Group of Hospitals",
                "active_profile_count": 2,
            }
        ],
    }


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
    listed = response.json()["documents"]
    assert len(listed) == 1
    assert listed[0]["id"] == second["id"]
    assert listed[0]["original_name"] == "second.pdf"
    assert listed[0]["hospital_name"] is None
    assert listed[0]["hospital_name_source"] is None

    queued = client.get("/api/v2/documents", params={"status": "queued"}).json()
    assert queued["total"] == 1
    assert queued["documents"][0]["id"] == second["id"]


def test_history_withholds_only_the_document_requiring_transaction_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    healthy_id, _ = completed_job(store)
    recovering_id, _ = completed_job(store)
    (store.job_dir(recovering_id) / ".cutover.json").write_text(
        json.dumps(
            {
                "version": "job_cutover_v1",
                "job_id": recovering_id,
            }
        )
    )

    response = client.get("/api/v2/documents", params={"scope": "history"})

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["id"] for item in response.json()["documents"]] == [healthy_id]
    assert client.get(f"/api/v2/documents/{recovering_id}").status_code == 409


def test_evidence_export_holds_workspace_lock_through_page_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    review = store.empty_review()
    review["approval"] = {
        "status": "approved",
        "reviewer": "test-reviewer",
        "approved_at": "2026-07-23T00:00:00Z",
        "review_revision": 0,
    }
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(review))
    artifact = store.job_dir(job_id) / "artifacts" / "pages" / "page-1.png"
    page_read_started = threading.Event()
    allow_page_read = threading.Event()
    writer_acquired = threading.Event()
    response_holder: list[object] = []
    original_read_bytes = Path.read_bytes

    def paused_read_bytes(path: Path) -> bytes:
        if path == artifact:
            page_read_started.set()
            assert allow_page_read.wait(timeout=2)
        return original_read_bytes(path)

    def request_export() -> None:
        response_holder.append(
            client.get(f"/api/v2/documents/{job_id}/exports/evidence.zip")
        )

    def acquire_writer() -> None:
        with store.job_lock(job_id, exclusive=True):
            writer_acquired.set()

    monkeypatch.setattr(Path, "read_bytes", paused_read_bytes)
    export_thread = threading.Thread(target=request_export)
    export_thread.start()
    assert page_read_started.wait(timeout=2)
    writer_thread = threading.Thread(target=acquire_writer)
    writer_thread.start()
    assert not writer_acquired.wait(timeout=0.1)

    allow_page_read.set()
    export_thread.join(timeout=2)
    writer_thread.join(timeout=2)

    assert not export_thread.is_alive()
    assert not writer_thread.is_alive()
    assert writer_acquired.is_set()
    response = response_holder[0]
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        page = archive.read("pages/page-1.png")
        manifest = archive.read("manifest.sha256").decode()
    assert page == b"PNG fixture"
    assert f"{hashlib.sha256(page).hexdigest()}  pages/page-1.png" in manifest


def test_worker_restart_requeues_interrupted_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    state = store.create("bill.pdf")
    store.update(state["id"], status="processing", page=2, pages=4)
    store.recover()
    recovered = store.read(state["id"])
    assert recovered["status"] == "queued"
    assert recovered["page"] == 0
