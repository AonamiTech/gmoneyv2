from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import LayoutProfile, ProfileLifecycle
from gmoney.demo import alias_transactions as alias_transactions_module
from gmoney.demo import api
from gmoney.demo.alias_transactions import (
    ALIAS_JOURNAL_VERSION,
    AliasTransactionCoordinator,
    _digest,
)
from gmoney.demo.store import JobStore
from gmoney.profiles.aliases import (
    ALIAS_REGISTRY_VERSION,
    LEGACY_ALIAS_REGISTRY_VERSION,
    AliasRegistryFormatError,
    AliasRegistryUnavailable,
    JsonAliasRepository,
)
from gmoney.profiles.repository import JsonProfileRepository


def client_for(tmp_path: Path, monkeypatch) -> tuple[TestClient, JobStore]:
    store = JobStore(tmp_path)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "ALIAS_REGISTRY", tmp_path / "alias-registry.json")
    return TestClient(api.app), store


def pdf_bytes(pages: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=300, height=400)
    payload = document.tobytes()
    document.close()
    return payload


def registry_event(action: str) -> dict[str, str]:
    return {
        "action": action,
        "reviewer": "test-reviewer",
        "reason": "Testing durable alias registry behavior",
        "created_at": "2026-08-12T00:00:00Z",
    }


def registry_with_alias(revision: int = 1) -> dict[str, object]:
    return {
        "registry_version": ALIAS_REGISTRY_VERSION,
        "revision": revision,
        "header_normalizer_version": "header_normalizer_v1",
        "hospital_normalizer_version": "hospital_name_normalizer_v1",
        "hospitals": [
            {
                "hospital_id": "hospital-1",
                "hospital_name": "Machine Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Machine Hospital",
                        "normalized_name": "machine hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "test-reviewer",
                        "reason": "Verified from grounded header",
                        "created_at": "2026-08-12T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        "aliases": [
            {
                "alias_id": "alias-1",
                "hospital_id": "hospital-1",
                "source_label": "Item #",
                "normalized_label": "item",
                "canonical_field": "service_code",
                "active": True,
                "reason": "Verified printed service code",
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        "events": [registry_event("column_alias_applied")],
    }


def legacy_registry_with_alias(revision: int = 1) -> dict[str, object]:
    registry = registry_with_alias(revision)
    registry["registry_version"] = LEGACY_ALIAS_REGISTRY_VERSION
    for event in registry["events"]:
        event.pop("reviewer", None)
    return registry


def _run_alias_cutover_to_boundary(
    root_value: str,
    registry_value: str,
    job_id: str,
    boundary: str,
    sender,
) -> None:
    store = JobStore(Path(root_value))
    coordinator = AliasTransactionCoordinator(store, Path(registry_value))
    original_replace = alias_transactions_module.durable_json_replace
    original_write = coordinator.repository._write_unlocked
    original_unlink = alias_transactions_module.durable_unlink

    def reached() -> None:
        sender.send(boundary)
        while True:
            time.sleep(1)

    if boundary in {"journal", "review"}:

        def pausing_replace(path, payload, *, suffix="tmp"):
            original_replace(path, payload, suffix=suffix)
            if suffix == f"{boundary}.tmp":
                reached()

        alias_transactions_module.durable_json_replace = pausing_replace
    elif boundary == "registry":

        def pausing_write(payload):
            original_write(payload)
            reached()

        coordinator.repository._write_unlocked = pausing_write
    else:

        def pausing_unlink(path):
            original_unlink(path)
            reached()

        alias_transactions_module.durable_unlink = pausing_unlink

    def mutation(review, registry, result):
        review["events"] = [*review["events"], {"action": "target_review"}]
        registry["events"] = [*registry["events"], registry_event("target_registry")]
        return review, registry

    coordinator.mutate_review_and_registry(job_id, 0, 0, mutation)


def _concurrent_registry_snapshot(
    root_value: str,
    registry_value: str,
    start_event,
    sender,
) -> None:
    start_event.wait()
    snapshot = AliasTransactionCoordinator(
        JobStore(Path(root_value)), Path(registry_value)
    ).registry_snapshot()
    sender.send(("snapshot", snapshot["revision"]))


def _concurrent_alias_patch(
    root_value: str,
    registry_value: str,
    start_event,
    sender,
) -> None:
    api.store = JobStore(Path(root_value))
    api.ALIAS_REGISTRY = Path(registry_value)
    start_event.wait()
    response = TestClient(api.app).patch(
        "/api/v2/hospitals/hospital-1/aliases/alias-1",
        json={
            "registry_revision": 1,
            "active": False,
            "reason": "Concurrent patch after pending recovery",
        },
    )
    sender.send(("patch", response.status_code, response.json()))


def _run_alias_cutover_paused_after_registry_write(
    root_value: str,
    registry_value: str,
    job_id: str,
    registry_written,
    release,
) -> None:
    store = JobStore(Path(root_value))
    coordinator = AliasTransactionCoordinator(store, Path(registry_value))
    original_write = coordinator.repository._write_unlocked

    def pausing_write(payload):
        original_write(payload)
        registry_written.set()
        release.wait()

    coordinator.repository._write_unlocked = pausing_write

    def mutation(review, registry, result):
        review["events"] = [*review["events"], {"action": "target_review"}]
        registry["events"] = [*registry["events"], registry_event("target_registry")]
        return review, registry

    coordinator.mutate_review_and_registry(job_id, 0, 0, mutation)


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


def test_hospital_alias_preview_and_apply_preserve_grounded_source_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    evidence = result["rows"][0]["evidence"]
    unchanged_row = json.loads(json.dumps(result["rows"][0]))
    unchanged_row["id"] = "unchanged-row"
    unchanged_row["row_order"] = 1
    unchanged_row["service_code"] = "PROC-44"
    result["rows"].append(unchanged_row)
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
                    "id": "procedure-ref",
                    "label": "Procedure Ref.",
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
                            "column_id": "procedure-ref",
                            "raw_value": "PROC-44",
                            "evidence": evidence,
                            "validation_flags": [],
                        },
                    ],
                    "validation_flags": [],
                },
                {
                    "id": "p1-t1-s1-r2",
                    "order": 1,
                    "canonical_row_id": "unchanged-row",
                    "cells": [
                        {
                            "column_id": "particular",
                            "raw_value": "Follow-up",
                            "evidence": evidence,
                            "validation_flags": [],
                        },
                        {
                            "column_id": "procedure-ref",
                            "raw_value": "PROC/44",
                            "evidence": evidence,
                            "validation_flags": [],
                        },
                    ],
                    "validation_flags": [],
                },
            ],
            "validation_flags": ["unmapped_columns"],
        }
    ]
    result["source_tables"].append(
        {
            "id": "p1-t2-s1",
            "page_number": 1,
            "table_id": "p1-t2",
            "table_type": "item_ledger",
            "columns": [
                {
                    "id": "procedure-ref",
                    "label": "Unrelated Ref.",
                    "order": 0,
                    "canonical_field": None,
                    "evidence": evidence,
                }
            ],
            "rows": [
                {
                    "id": "p1-t2-s1-r1",
                    "order": 0,
                    "canonical_row_id": None,
                    "cells": [
                        {
                            "column_id": "procedure-ref",
                            "raw_value": "UNRELATED-9",
                            "evidence": evidence,
                            "validation_flags": [],
                        }
                    ],
                    "validation_flags": [],
                }
            ],
            "validation_flags": ["unmapped_columns"],
        }
    )
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    linked = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": True,
            "registry_revision": 0,
            "reason": "Verified hospital for alias training",
        },
    )
    assert linked.status_code == 200
    hospital_id = linked.json()["hospital_id"]

    preview = client.post(
        f"/api/v2/documents/{job_id}/column-aliases/preview",
        json={
            "hospital_id": hospital_id,
            "source_label": "Procedure Ref.",
            "canonical_field": "service_code",
        },
    )
    assert preview.status_code == 200
    proposal = preview.json()
    assert proposal["counts"]["fillable"] == 1
    assert proposal["counts"]["unchanged"] == 1

    apply_payload = {
        "hospital_id": hospital_id,
        "source_label": "Procedure Ref.",
        "canonical_field": "service_code",
        "source_digest": proposal["source_digest"],
        "registry_revision": proposal["registry_revision"],
        "selected_candidate_ids": [
            candidate["candidate_id"]
            for candidate in proposal["candidates"]
            if candidate["classification"] in {"fillable", "unchanged"}
        ],
        "reason": "Procedure Ref is this hospital's service code",
    }
    changed_result = json.loads(json.dumps(result))
    changed_result["source_tables"][0]["rows"][0]["cells"][1]["evidence"][0]["token_ids"] = [
        "changed-token"
    ]
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(changed_result))
    stale_source = client.post(
        f"/api/v2/documents/{job_id}/column-aliases/apply",
        headers={"If-Match": "1"},
        json=apply_payload,
    )
    assert stale_source.status_code == 409
    assert stale_source.json()["detail"]["code"] == "alias_source_digest_conflict"
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    unknown_candidate = client.post(
        f"/api/v2/documents/{job_id}/column-aliases/apply",
        headers={"If-Match": "1"},
        json={**apply_payload, "selected_candidate_ids": ["unknown-candidate"]},
    )
    assert unknown_candidate.status_code == 422
    assert unknown_candidate.json()["detail"] == (
        "Selected alias candidates are stale or unknown"
    )
    assert store.read_review(job_id)["revision"] == 1
    assert JsonAliasRepository(api.ALIAS_REGISTRY).read()["revision"] == 1

    applied = client.post(
        f"/api/v2/documents/{job_id}/column-aliases/apply",
        headers={"If-Match": "1"},
        json=apply_payload,
    )

    assert applied.status_code == 200
    rows_payload = client.get(f"/api/v2/documents/{job_id}/rows").json()
    row = rows_payload["rows"][0]
    assert row["service_code"] == "PROC-44"
    assert row["field_evidence"]["service_code"] == evidence
    unchanged = next(item for item in rows_payload["rows"] if item["id"] == "unchanged-row")
    assert unchanged["service_code"] == "PROC-44"
    assert unchanged["field_evidence"]["service_code"] == evidence
    assert "service_code" in rows_payload["populated_fields"]
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == result
    printed = client.get(f"/api/v2/documents/{job_id}/source-tables").json()
    assert printed["tables"][0]["columns"][1]["canonical_field"] == "service_code"
    assert printed["tables"][1]["columns"][0]["canonical_field"] is None
    aliases = client.get(f"/api/v2/hospitals/{hospital_id}/aliases").json()
    assert aliases["aliases"][0]["canonical_field"] == "service_code"
    assert aliases["aliases"][0]["active"] is True
    deactivated = client.patch(
        f"/api/v2/hospitals/{hospital_id}/aliases/{aliases['aliases'][0]['alias_id']}",
        json={
            "registry_revision": aliases["registry_revision"],
            "active": False,
            "reason": "Retiring an incorrect hospital convention",
        },
    )
    assert deactivated.status_code == 200
    assert deactivated.json()["alias"]["active"] is False
    registry_event = JsonAliasRepository(api.ALIAS_REGISTRY).read()["events"][-1]
    assert registry_event["reviewer"] == "demo-reviewer"
    assert registry_event["old_active"] is True
    assert registry_event["new_active"] is False
    stale = client.patch(
        f"/api/v2/hospitals/{hospital_id}/aliases/{aliases['aliases'][0]['alias_id']}",
        json={
            "registry_revision": aliases["registry_revision"],
            "active": True,
            "reason": "A stale browser must not overwrite the newer decision",
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "alias_registry_revision_conflict"


def test_interrupted_alias_operation_is_recovered_as_one_cutover(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    review = json.loads(json.dumps(base_review))
    review["revision"] = 1
    review["events"].append({"action": "recovered_alias_review"})
    base_registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    registry = json.loads(json.dumps(base_registry))
    registry["revision"] = 1
    registry["events"].append(registry_event("recovered_alias_registry"))
    journal = store.job_dir(job_id) / ".alias-operation.json"
    journal.write_text(
        json.dumps(
            {
                "version": ALIAS_JOURNAL_VERSION,
                "job_id": job_id,
                "base_review": base_review,
                "target_review": review,
                "base_registry": base_registry,
                "target_registry": registry,
                "base_review_sha256": _digest(base_review),
                "target_review_sha256": _digest(review),
                "base_registry_sha256": _digest(base_registry),
                "target_registry_sha256": _digest(registry),
            }
        )
    )
    JsonAliasRepository(api.ALIAS_REGISTRY)._write_unlocked(registry)

    response = client.get("/api/v2/hospitals/trained")

    assert response.status_code == 200
    assert store.read_review(job_id)["revision"] == 1
    assert JsonAliasRepository(api.ALIAS_REGISTRY).read()["revision"] == 1
    assert not journal.exists()


def test_registry_snapshot_holds_one_lock_through_recovery_and_read(
    tmp_path: Path, monkeypatch
) -> None:
    store = JobStore(tmp_path)
    coordinator = AliasTransactionCoordinator(store, tmp_path / "alias-registry.json")
    snapshot_in_recovery = threading.Event()
    release_snapshot = threading.Event()
    mutation_finished = threading.Event()
    original_recover = coordinator._recover_all_unlocked
    snapshots: list[dict[str, object]] = []

    def pausing_recovery() -> None:
        original_recover()
        if threading.current_thread().name == "snapshot-reader":
            snapshot_in_recovery.set()
            assert release_snapshot.wait(2)

    monkeypatch.setattr(coordinator, "_recover_all_unlocked", pausing_recovery)

    snapshot_thread = threading.Thread(
        target=lambda: snapshots.append(coordinator.registry_snapshot()),
        name="snapshot-reader",
    )

    def mutate() -> None:
        coordinator.mutate_registry(0, lambda registry: registry)
        mutation_finished.set()

    mutation_thread = threading.Thread(target=mutate, name="registry-writer")
    snapshot_thread.start()
    assert snapshot_in_recovery.wait(2)
    mutation_thread.start()
    assert not mutation_finished.wait(0.2)
    release_snapshot.set()
    snapshot_thread.join(timeout=2)
    mutation_thread.join(timeout=2)

    assert not snapshot_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert snapshots[0]["revision"] == 0
    assert mutation_finished.is_set()
    assert coordinator.registry_snapshot()["revision"] == 1


def test_registry_mutation_recovers_pending_cutover_before_incrementing(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    coordinator = AliasTransactionCoordinator(store, tmp_path / "alias-registry.json")
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_registry = coordinator.registry_snapshot()
    target_registry = json.loads(json.dumps(base_registry))
    target_registry["revision"] = 1
    target_registry["events"].append(registry_event("pending_alias_cutover"))
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))
    coordinator.repository._write_unlocked(target_registry)

    updated = coordinator.mutate_registry(
        1,
        lambda registry: {
            **registry,
            "events": [*registry["events"], registry_event("following_patch")],
        },
    )

    assert updated["revision"] == 2
    assert store.read_review(job_id)["revision"] == 1
    assert not (store.job_dir(job_id) / ".alias-operation.json").exists()
    assert updated["events"][-1]["action"] == "following_patch"


@pytest.mark.parametrize(
    ("failure_phase", "journal_survives"),
    (
        ("journal", False),
        ("registry", True),
        ("review", True),
        ("unlink", True),
    ),
)
def test_alias_cutover_failure_boundaries_are_recoverable(
    tmp_path: Path,
    monkeypatch,
    failure_phase: str,
    journal_survives: bool,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(base_review))
    coordinator = AliasTransactionCoordinator(store, tmp_path / "alias-registry.json")
    original_replace = alias_transactions_module.durable_json_replace
    original_write = coordinator.repository._write_unlocked
    original_unlink = alias_transactions_module.durable_unlink

    if failure_phase in {"journal", "review"}:

        def failing_replace(
            path: Path,
            payload: dict[str, object],
            *,
            suffix: str = "tmp",
        ) -> None:
            if suffix == f"{failure_phase}.tmp":
                raise OSError(f"simulated {failure_phase} write failure")
            original_replace(path, payload, suffix=suffix)

        monkeypatch.setattr(alias_transactions_module, "durable_json_replace", failing_replace)
    elif failure_phase == "registry":

        def failing_registry_write(payload: dict[str, object]) -> None:
            raise OSError("simulated registry write failure")

        monkeypatch.setattr(coordinator.repository, "_write_unlocked", failing_registry_write)
    else:

        def failing_unlink(path: Path) -> None:
            raise OSError("simulated journal unlink failure")

        monkeypatch.setattr(alias_transactions_module, "durable_unlink", failing_unlink)

    def mutation(
        review: dict[str, object],
        registry: dict[str, object],
        result: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert result["document_id"] == "d" * 64
        review["events"] = [*review["events"], {"action": "target_review"}]
        registry["events"] = [*registry["events"], registry_event("target_registry")]
        return review, registry

    with pytest.raises(AliasRegistryUnavailable):
        coordinator.mutate_review_and_registry(job_id, 0, 0, mutation)

    journal = store.job_dir(job_id) / ".alias-operation.json"
    assert journal.exists() is journal_survives
    monkeypatch.setattr(alias_transactions_module, "durable_json_replace", original_replace)
    monkeypatch.setattr(coordinator.repository, "_write_unlocked", original_write)
    monkeypatch.setattr(alias_transactions_module, "durable_unlink", original_unlink)

    snapshot = coordinator.registry_snapshot()
    expected_revision = 1 if journal_survives else 0
    assert snapshot["revision"] == expected_revision
    assert store.read_review(job_id)["revision"] == expected_revision
    assert not journal.exists()


@pytest.mark.parametrize("boundary", ("journal", "registry", "review", "unlink"))
def test_alias_cutover_recovers_after_process_termination_at_each_boundary(
    tmp_path: Path,
    boundary: str,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(base_review))
    registry_path = tmp_path / "alias-registry.json"
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_alias_cutover_to_boundary,
        args=(str(tmp_path), str(registry_path), job_id, boundary, sender),
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(5)
        assert receiver.recv() == boundary
    finally:
        receiver.close()
        process.terminate()
        process.join(timeout=5)
    assert not process.is_alive()

    snapshot = AliasTransactionCoordinator(store, registry_path).registry_snapshot()

    assert snapshot["revision"] == 1
    assert store.read_review(job_id)["revision"] == 1
    assert not (store.job_dir(job_id) / ".alias-operation.json").exists()


def test_pending_journal_serializes_concurrent_snapshot_and_alias_patch(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    target_review["events"].append({"action": "target_review"})
    coordinator = AliasTransactionCoordinator(store, tmp_path / "alias-registry.json")
    base_registry = coordinator.registry_snapshot()
    target_registry = registry_with_alias()
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))
    coordinator.repository._write_unlocked(target_registry)

    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    snapshot_process = context.Process(
        target=_concurrent_registry_snapshot,
        args=(str(tmp_path), str(coordinator.repository.path), start_event, sender),
    )
    patch_process = context.Process(
        target=_concurrent_alias_patch,
        args=(str(tmp_path), str(coordinator.repository.path), start_event, sender),
    )
    snapshot_process.start()
    patch_process.start()
    sender.close()
    start_event.set()
    messages = []
    try:
        assert receiver.poll(8)
        messages.append(receiver.recv())
        assert receiver.poll(8)
        messages.append(receiver.recv())
    finally:
        receiver.close()
        for process in (snapshot_process, patch_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert snapshot_process.exitcode == 0
    assert patch_process.exitcode == 0
    snapshot_message = next(item for item in messages if item[0] == "snapshot")
    patch_message = next(item for item in messages if item[0] == "patch")
    assert snapshot_message[1] in {1, 2}
    assert patch_message[1] == 200
    assert patch_message[2]["registry_revision"] == 2
    final_registry = coordinator.registry_snapshot()
    assert final_registry["revision"] == 2
    assert final_registry["aliases"][0]["active"] is False
    assert store.read_review(job_id)["revision"] == 1
    assert not (store.job_dir(job_id) / ".alias-operation.json").exists()


def test_snapshot_cannot_observe_future_registry_before_review_is_durable(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(base_review))
    registry_path = tmp_path / "alias-registry.json"
    context = multiprocessing.get_context("spawn")
    registry_written = context.Event()
    release = context.Event()
    cutover = context.Process(
        target=_run_alias_cutover_paused_after_registry_write,
        args=(
            str(tmp_path),
            str(registry_path),
            job_id,
            registry_written,
            release,
        ),
    )
    receiver, sender = context.Pipe(duplex=False)
    snapshot = context.Process(
        target=_concurrent_registry_snapshot,
        args=(str(tmp_path), str(registry_path), registry_written, sender),
    )
    cutover.start()
    snapshot.start()
    sender.close()
    try:
        assert registry_written.wait(8)
        assert json.loads(registry_path.read_text())["revision"] == 1
        assert json.loads(
            (store.job_dir(job_id) / "review.json").read_text()
        )["revision"] == 0
        assert not receiver.poll(0.5)
        release.set()
        assert receiver.poll(8)
        assert receiver.recv() == ("snapshot", 1)
    finally:
        release.set()
        receiver.close()
        for process in (cutover, snapshot):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert cutover.exitcode == 0
    assert snapshot.exitcode == 0
    assert store.read_review(job_id)["revision"] == 1


@pytest.mark.parametrize(
    "error",
    (AttributeError("callback bug"), KeyError("callback bug"), TypeError("callback bug")),
)
def test_registry_callback_programming_errors_are_not_reported_as_outages(
    tmp_path: Path,
    error: Exception,
) -> None:
    coordinator = AliasTransactionCoordinator(
        JobStore(tmp_path), tmp_path / "alias-registry.json"
    )

    def broken_callback(registry):
        raise error

    with pytest.raises(type(error), match="callback bug"):
        coordinator.mutate_registry(0, broken_callback)


def test_alias_patch_returns_registry_unavailable_for_corrupt_json(
    tmp_path: Path, monkeypatch
) -> None:
    client, _ = client_for(tmp_path, monkeypatch)
    api.ALIAS_REGISTRY.write_text("{")

    response = client.patch(
        "/api/v2/hospitals/hospital/aliases/alias",
        json={
            "registry_revision": 0,
            "active": False,
            "reason": "Registry integrity failure should be explicit",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {"code": "alias_registry_unavailable"}


def test_alias_patch_preserves_missing_alias_response(tmp_path: Path, monkeypatch) -> None:
    client, _ = client_for(tmp_path, monkeypatch)

    response = client.patch(
        "/api/v2/hospitals/hospital/aliases/missing",
        json={
            "registry_revision": 0,
            "active": False,
            "reason": "Unknown aliases remain a not-found response",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Column alias was not found"


def test_callback_registry_validation_failure_is_not_reported_as_outage(
    tmp_path: Path,
) -> None:
    coordinator = AliasTransactionCoordinator(
        JobStore(tmp_path),
        tmp_path / "alias-registry.json",
    )

    with pytest.raises(ValueError, match="unsupported hospital alias registry"):
        coordinator.mutate_registry(
            0,
            lambda registry: {**registry, "registry_version": "unsupported"},
        )


def test_generated_v2_registry_is_migrated_once_to_strict_v3(tmp_path: Path) -> None:
    path = tmp_path / "alias-registry.json"
    legacy = legacy_registry_with_alias(revision=7)
    path.write_text(json.dumps(legacy))
    coordinator = AliasTransactionCoordinator(JobStore(tmp_path), path)

    migrated = coordinator.registry_snapshot()

    assert migrated["registry_version"] == ALIAS_REGISTRY_VERSION
    assert migrated["revision"] == 8
    assert migrated["events"][0]["reviewer"] == "legacy-unknown"
    assert migrated["events"][-1]["action"] == "alias_registry_migrated"
    assert migrated["events"][-1]["reviewer"] == "system:migration"
    assert json.loads(path.read_text()) == migrated
    assert coordinator.registry_snapshot() == migrated


def test_concurrent_processes_migrate_v2_registry_idempotently(tmp_path: Path) -> None:
    registry_path = tmp_path / "alias-registry.json"
    registry_path.write_text(json.dumps(legacy_registry_with_alias(revision=4)))
    JobStore(tmp_path)
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    processes = [
        context.Process(
            target=_concurrent_registry_snapshot,
            args=(str(tmp_path), str(registry_path), start_event, sender),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    sender.close()
    start_event.set()
    try:
        assert receiver.poll(8)
        first = receiver.recv()
        assert receiver.poll(8)
        second = receiver.recv()
    finally:
        receiver.close()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    assert first == ("snapshot", 5)
    assert second == ("snapshot", 5)
    final = JsonAliasRepository(registry_path).read()
    assert final["revision"] == 5
    assert sum(
        event["action"] == "alias_registry_migrated" for event in final["events"]
    ) == 1


def test_pending_v2_journal_recovers_before_registry_migration(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    target_review["events"].append({"action": "legacy_target_review"})
    base_registry = legacy_registry_with_alias(revision=0)
    target_registry = json.loads(json.dumps(base_registry))
    target_registry["revision"] = 1
    target_registry["aliases"][0]["active"] = False
    target_registry["events"].append(
        {
            "action": "legacy_target_registry",
            "reason": "Legacy cutover pending during upgrade",
            "created_at": "2026-08-12T00:01:00Z",
        }
    )
    registry_path = tmp_path / "alias-registry.json"
    registry_path.write_text(json.dumps(target_registry))
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))

    migrated = AliasTransactionCoordinator(store, registry_path).registry_snapshot()

    assert store.read_review(job_id)["revision"] == 1
    assert migrated["registry_version"] == ALIAS_REGISTRY_VERSION
    assert migrated["revision"] == 2
    assert migrated["aliases"][0]["active"] is False
    assert migrated["events"][-1]["action"] == "alias_registry_migrated"
    assert not (store.job_dir(job_id) / ".alias-operation.json").exists()


@pytest.mark.parametrize("stage", ("journal", "registry", "review"))
def test_first_v2_journal_recovers_at_every_upgrade_boundary(
    tmp_path: Path,
    stage: str,
) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    target_review["events"].append({"action": "first_legacy_review"})
    base_registry = legacy_registry_with_alias(revision=0)
    base_registry["hospitals"] = []
    base_registry["aliases"] = []
    base_registry["events"] = []
    target_registry = legacy_registry_with_alias(revision=1)
    registry_path = tmp_path / "alias-registry.json"
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))
    if stage in {"registry", "review"}:
        registry_path.write_text(json.dumps(target_registry))
    if stage == "review":
        (store.job_dir(job_id) / "review.json").write_text(json.dumps(target_review))

    migrated = AliasTransactionCoordinator(store, registry_path).registry_snapshot()

    assert store.read_review(job_id)["revision"] == 1
    assert migrated["registry_version"] == ALIAS_REGISTRY_VERSION
    assert migrated["revision"] == 2
    assert migrated["aliases"][0]["alias_id"] == "alias-1"
    assert migrated["events"][-1]["action"] == "alias_registry_migrated"
    assert not (store.job_dir(job_id) / ".alias-operation.json").exists()


def test_missing_registry_refuses_non_empty_journal_base(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_registry = legacy_registry_with_alias(revision=0)
    target_registry = legacy_registry_with_alias(revision=1)
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))

    with pytest.raises(
        AliasRegistryUnavailable,
        match="missing registry cannot recover a non-empty alias base",
    ):
        AliasTransactionCoordinator(store, tmp_path / "alias-registry.json").recover_all()

    assert not (tmp_path / "alias-registry.json").exists()
    review_path = store.job_dir(job_id) / "review.json"
    assert not review_path.exists() or json.loads(review_path.read_text())["revision"] == 0


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda registry: registry.update(revision=True),
        lambda registry: registry["hospitals"][0].update(origins="profile"),
        lambda registry: registry["hospitals"][0]["name_variants"][0].update(
            verified="true"
        ),
        lambda registry: registry["aliases"][0].update(active="false"),
        lambda registry: registry["events"][0].pop("reviewer"),
        lambda registry: registry["events"][0].update(created_at="yesterday"),
    ),
)
def test_alias_registry_rejects_malformed_typed_fields(
    tmp_path: Path,
    corrupt,
) -> None:
    registry = {
        "registry_version": ALIAS_REGISTRY_VERSION,
        "revision": 1,
        "header_normalizer_version": "header_normalizer_v1",
        "hospital_normalizer_version": "hospital_name_normalizer_v1",
        "hospitals": [
            {
                "hospital_id": "hospital-1",
                "hospital_name": "Machine Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Machine Hospital",
                        "normalized_name": "machine hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "test-reviewer",
                        "reason": "Verified from grounded header",
                        "created_at": "2026-08-12T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        "aliases": [
            {
                "alias_id": "alias-1",
                "hospital_id": "hospital-1",
                "source_label": "Item #",
                "normalized_label": "item",
                "canonical_field": "service_code",
                "active": True,
                "reason": "Verified printed service code",
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        "events": [registry_event("column_alias_applied")],
    }
    corrupt(registry)
    path = tmp_path / "alias-registry.json"
    path.write_text(json.dumps(registry))

    with pytest.raises(AliasRegistryUnavailable):
        AliasTransactionCoordinator(JobStore(tmp_path), path).registry_snapshot()


def test_cleanup_recovers_alias_journals_once_for_many_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = JobStore(tmp_path)
    for name in ("one.pdf", "two.pdf", "three.pdf"):
        state = store.create(name)
        store.update(state["id"], status="complete")
    coordinator = AliasTransactionCoordinator(store, tmp_path / "alias-registry.json")
    recover_calls = 0
    lock_calls = 0
    original_recover = coordinator._recover_all_unlocked
    original_lock = coordinator.repository.lock

    def tracking_recover() -> None:
        nonlocal recover_calls
        recover_calls += 1
        original_recover()

    @contextmanager
    def tracking_lock(*, exclusive: bool):
        nonlocal lock_calls
        lock_calls += 1
        with original_lock(exclusive=exclusive):
            yield

    monkeypatch.setattr(coordinator, "_recover_all_unlocked", tracking_recover)
    monkeypatch.setattr(coordinator.repository, "lock", tracking_lock)

    assert coordinator.cleanup(720) == 0
    assert recover_calls == 1
    assert lock_calls == 1


def test_alias_recovery_refuses_to_overwrite_newer_review_state(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    target_registry = json.loads(json.dumps(base_registry))
    target_registry["revision"] = 1
    journal = store.job_dir(job_id) / ".alias-operation.json"
    journal.write_text(
        json.dumps(
            {
                "version": ALIAS_JOURNAL_VERSION,
                "job_id": job_id,
                "base_review": base_review,
                "target_review": target_review,
                "base_registry": base_registry,
                "target_registry": target_registry,
                "base_review_sha256": _digest(base_review),
                "target_review_sha256": _digest(target_review),
                "base_registry_sha256": _digest(base_registry),
                "target_registry_sha256": _digest(target_registry),
            }
        )
    )
    newer_review = json.loads(json.dumps(target_review))
    newer_review["revision"] = 2
    newer_review["events"].append({"action": "newer_concurrent_review"})
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(newer_review))

    response = client.get(f"/api/v2/documents/{job_id}/rows")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "alias_registry_unavailable"
    assert journal.is_file()
    assert json.loads((store.job_dir(job_id) / "review.json").read_text()) == newer_review
    assert JsonAliasRepository(api.ALIAS_REGISTRY).read() == base_registry


def test_document_deletion_recovers_pending_alias_transaction_before_removal(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    base_review = store.read_review(job_id)
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    target_registry = json.loads(json.dumps(base_registry))
    target_registry["revision"] = 1
    target_registry["events"].append(registry_event("committed_before_delete"))
    journal = {
        "version": ALIAS_JOURNAL_VERSION,
        "job_id": job_id,
        "base_review": base_review,
        "target_review": target_review,
        "base_registry": base_registry,
        "target_registry": target_registry,
        "base_review_sha256": _digest(base_review),
        "target_review_sha256": _digest(target_review),
        "base_registry_sha256": _digest(base_registry),
        "target_registry_sha256": _digest(target_registry),
    }
    (store.job_dir(job_id) / ".alias-operation.json").write_text(json.dumps(journal))

    response = client.delete(f"/api/v2/documents/{job_id}")

    assert response.status_code == 204
    assert not store.job_dir(job_id).exists()
    registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    assert registry["revision"] == 1
    assert registry["events"][-1]["action"] == "committed_before_delete"


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

    response = client.get(f"/api/v2/documents/{job_id}/source-tables?offset=1&limit=2")

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


def test_bulk_reject_and_restore_are_atomic_and_revisioned(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    second = {**result["rows"][0], "id": "second-row", "row_order": 1}
    result["rows"].append(second)
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    rejected = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "0"},
        json={
            "row_ids": ["machine-row", "second-row"],
            "action": "reject",
            "reason": "Duplicate summary lines",
        },
    )

    assert rejected.status_code == 200
    assert rejected.json()["updated_count"] == 2
    active = client.get(f"/api/v2/documents/{job_id}/rows?disposition=active").json()
    assert active["total"] == 0
    assert active["totals"]["items_total"] == "0"
    rejected_rows = client.get(f"/api/v2/documents/{job_id}/rows?disposition=rejected").json()
    assert {row["id"] for row in rejected_rows["rows"]} == {
        "machine-row",
        "second-row",
    }

    restored = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "1"},
        json={
            "row_ids": ["machine-row", "second-row"],
            "action": "restore",
            "reason": "Confirmed as valid charges",
        },
    )
    assert restored.status_code == 200
    assert client.get(f"/api/v2/documents/{job_id}/rows?disposition=active").json()["total"] == 2
    review = store.read_review(job_id)
    assert review["revision"] == 2
    assert [event["action"] for event in review["events"]] == [
        "rows_rejected",
        "rows_restored",
    ]


def test_bulk_reject_validates_every_row_before_mutating(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)

    response = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "0"},
        json={
            "row_ids": ["machine-row", "missing-row"],
            "action": "reject",
            "reason": "Invalid mixed selection",
        },
    )

    assert response.status_code == 422
    assert store.read_review(job_id)["revision"] == 0
    assert (
        client.get(f"/api/v2/documents/{job_id}/rows").json()["rows"][0]["review_disposition"]
        == "accepted"
    )


def test_repeated_reject_is_atomic_and_preserves_restoration_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)

    first = client.delete(
        f"/api/v2/documents/{job_id}/rows/machine-row?reason=Duplicate+charge",
        headers={"If-Match": "0"},
    )
    assert first.status_code == 200
    rejected = client.get(f"/api/v2/documents/{job_id}/rows?disposition=rejected").json()["rows"][0]
    assert rejected["bulk_action"] == "restore"
    assert rejected["rejection_provenance"]["previous_disposition"] == "accepted"

    repeated = client.delete(
        f"/api/v2/documents/{job_id}/rows/machine-row?reason=Duplicate+again",
        headers={"If-Match": "1"},
    )
    assert repeated.status_code == 422
    assert store.read_review(job_id)["revision"] == 1

    restored = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "1"},
        json={
            "row_ids": ["machine-row"],
            "action": "restore",
            "reason": "Confirmed legitimate charge",
        },
    )
    assert restored.status_code == 200
    active = client.get(f"/api/v2/documents/{job_id}/rows").json()["rows"][0]
    assert active["review_disposition"] == "accepted"
    assert active["bulk_action"] == "reject"


def test_legacy_rejection_provenance_survives_field_edits_and_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)
    review = store.read_review(job_id)
    review["row_overrides"]["machine-row"] = {
        "changes": {"review_disposition": "rejected"},
        "pre_rejection_disposition": "unreadable",
        "reason": "Legacy reviewer rejection",
        "updated_at": "2026-08-01T00:00:00Z",
    }
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(review))

    edited = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {"description": "Corrected legacy rejected charge"},
            "reason": "Corrected text without changing legacy rejection",
        },
    )

    assert edited.status_code == 200
    row = edited.json()["row"]
    assert row["bulk_action"] == "restore"
    assert row["rejection_provenance"] == {
        "source": "reviewer",
        "previous_disposition": "unreadable",
        "legacy_fallback": True,
    }
    saved = store.read_review(job_id)["row_overrides"]["machine-row"]
    assert saved["pre_rejection_disposition"] == "unreadable"

    restored = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "1"},
        json={
            "row_ids": ["machine-row"],
            "action": "restore",
            "reason": "Restored to the saved legacy disposition",
        },
    )

    assert restored.status_code == 200
    restored_row = client.get(f"/api/v2/documents/{job_id}/rows").json()["rows"][0]
    assert restored_row["review_disposition"] == "unreadable"


def test_machine_rejected_rows_are_not_reviewer_restorable(tmp_path: Path, monkeypatch) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    result["rows"][0]["review_disposition"] = "rejected"
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    row = client.get(f"/api/v2/documents/{job_id}/rows").json()["rows"][0]
    assert row["bulk_action"] is None
    response = client.patch(
        f"/api/v2/documents/{job_id}/rows/bulk",
        headers={"If-Match": "0"},
        json={
            "row_ids": ["machine-row"],
            "action": "restore",
            "reason": "Attempted invalid restoration",
        },
    )
    assert response.status_code == 422
    assert store.read_review(job_id)["revision"] == 0

    description_only = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {
                "description": "Corrected machine-rejected charge",
                "review_disposition": "rejected",
            },
            "reason": "Corrected text while preserving machine rejection",
        },
    )
    assert description_only.status_code == 200
    assert description_only.json()["row"]["description"] == (
        "Corrected machine-rejected charge"
    )
    assert description_only.json()["row"]["review_disposition"] == "rejected"
    assert description_only.json()["row"]["bulk_action"] is None

    corrected = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "1"},
        json={
            "changes": {"review_disposition": "accepted"},
            "reason": "Reviewer directly confirmed the machine-rejected charge",
        },
    )
    assert corrected.status_code == 200
    assert corrected.json()["row"]["review_disposition"] == "accepted"
    assert corrected.json()["row"]["bulk_action"] == "reject"
    event = store.read_review(job_id)["events"][-1]
    assert event["changes"]["review_disposition"] == {
        "old_disposition": "rejected",
        "new_disposition": "accepted",
    }


def test_single_row_editor_uses_the_same_reject_restore_state_machine(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, _ = completed_job(store)

    rejected = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {"review_disposition": "rejected"},
            "reason": "Rejected through the row editor",
        },
    )
    assert rejected.status_code == 200
    assert rejected.json()["row"]["bulk_action"] == "restore"
    assert rejected.json()["row"]["rejection_provenance"]["previous_disposition"] == "accepted"
    rejection_event = store.read_review(job_id)["events"][-1]
    assert rejection_event["changes"]["review_disposition"] == {
        "old_disposition": "accepted",
        "new_disposition": "rejected",
    }

    corrected_text = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "1"},
        json={
            "changes": {
                "description": "Corrected reviewer-rejected charge",
                "review_disposition": "rejected",
            },
            "reason": "Corrected text without restoring the rejected row",
        },
    )
    assert corrected_text.status_code == 200
    corrected_row = corrected_text.json()["row"]
    assert corrected_row["description"] == "Corrected reviewer-rejected charge"
    assert corrected_row["review_disposition"] == "rejected"
    assert corrected_row["bulk_action"] == "restore"
    assert corrected_row["rejection_provenance"]["previous_disposition"] == "accepted"

    repeated = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "2"},
        json={
            "changes": {"review_disposition": "rejected"},
            "reason": "Repeated editor rejection",
        },
    )
    assert repeated.status_code == 422
    assert store.read_review(job_id)["revision"] == 2

    restored = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "2"},
        json={
            "changes": {"review_disposition": "accepted"},
            "reason": "Restored through the row editor",
        },
    )
    assert restored.status_code == 200
    assert restored.json()["row"]["review_disposition"] == "accepted"
    assert restored.json()["row"]["bulk_action"] == "reject"


def test_service_date_corrections_update_raw_projection_and_clear_authoritatively(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    result["rows"][0]["service_date_raw"] = "20/01/2026"
    result["rows"][0]["service_date_iso"] = "2026-01-20"
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))

    raw_only = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {"service_date_raw": "99/99/2026"},
            "reason": "Attempted a raw-only date correction",
        },
    )
    assert raw_only.status_code == 422
    assert raw_only.json()["detail"] == (
        "service_date_raw requires authoritative service_date_iso"
    )
    assert store.read_review(job_id)["revision"] == 0

    changed = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {
                "service_date_iso": "2026-02-21",
                "service_date_raw": "99/99/2026",
            },
            "reason": "Corrected the printed service date",
        },
    )

    assert changed.status_code == 200
    assert changed.json()["row"]["service_date_iso"] == "2026-02-21"
    assert changed.json()["row"]["service_date_raw"] == "2026-02-21"
    assert changed.json()["row"]["review"]["machine_values"] == {
        "service_date_iso": "2026-01-20",
        "service_date_raw": "20/01/2026",
    }

    cleared = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "1"},
        json={
            "changes": {"service_date_iso": ""},
            "reason": "Confirmed that no service date is printed",
        },
    )

    assert cleared.status_code == 200
    assert cleared.json()["row"]["service_date_iso"] is None
    assert cleared.json()["row"]["service_date_raw"] is None

    invalid = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "2"},
        json={
            "changes": {"service_date_iso": "2026-02-30"},
            "reason": "Attempted an impossible date",
        },
    )
    assert invalid.status_code == 422
    assert store.read_review(job_id)["revision"] == 2


def test_evidence_relink_preserves_unrelated_field_evidence_and_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    source_evidence = result["rows"][0]["evidence"][0]
    date_evidence = {**source_evidence, "token_ids": ["alias-date-token"]}
    review = store.read_review(job_id)
    review["row_overrides"]["machine-row"] = {
        "changes": {
            "review_disposition": "rejected",
            "field_evidence": {"service_date": [date_evidence]},
            "source_routes": ["ocr_spatial_graph", "reviewer_hospital_alias"],
        },
        "rejection": {
            "source": "reviewer",
            "previous_disposition": "unreadable",
            "rejected_at": "2026-08-12T00:00:00Z",
            "reason": "Rejected before evidence relink",
        },
        "reason": "Alias-trained date before relink",
        "updated_at": "2026-08-12T00:00:00Z",
    }
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(review))

    relinked = client.patch(
        f"/api/v2/documents/{job_id}/rows/machine-row",
        headers={"If-Match": "0"},
        json={
            "changes": {"description": "Relinked rejected charge"},
            "page_number": 1,
            "polygon": {
                "points": [
                    {"x": 12, "y": 12},
                    {"x": 88, "y": 12},
                    {"x": 88, "y": 28},
                    {"x": 12, "y": 28},
                ]
            },
            "reason": "Relinked description evidence only",
        },
    )

    assert relinked.status_code == 200
    row = relinked.json()["row"]
    assert row["field_evidence"]["service_date"] == [date_evidence]
    assert set(row["field_evidence"]) >= {"service_date", "description", "amount"}
    assert row["bulk_action"] == "restore"
    assert row["rejection_provenance"]["previous_disposition"] == "unreadable"
    saved = store.read_review(job_id)["row_overrides"]["machine-row"]
    assert saved["rejection"]["previous_disposition"] == "unreadable"
    assert saved["changes"]["field_evidence"]["service_date"] == [date_evidence]


def test_historical_iso_only_date_clear_hides_machine_raw_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path, monkeypatch)
    job_id, result = completed_job(store)
    result["rows"][0]["service_date_raw"] = "20/01/2026"
    result["rows"][0]["service_date_iso"] = "2026-01-20"
    (store.job_dir(job_id) / "result.json").write_text(json.dumps(result))
    review = store.read_review(job_id)
    review["row_overrides"]["machine-row"] = {
        "changes": {"service_date_iso": None},
        "reason": "Historical clear before paired date overrides",
        "updated_at": "2026-08-01T00:00:00Z",
    }
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(review))

    row = client.get(f"/api/v2/documents/{job_id}/rows").json()["rows"][0]

    assert row["service_date_iso"] is None
    assert row["service_date_raw"] is None
    assert row["review"]["machine_values"]["service_date_raw"] == "20/01/2026"


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


def test_trained_hospitals_lists_only_active_hospital_profiles(tmp_path: Path, monkeypatch) -> None:
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
        "registry_revision": 0,
        "total": 1,
        "hospitals": [
            {
                "hospital_id": "vijaya",
                "hospital_name": "Vijaya Group of Hospitals",
                "active_profile_count": 2,
                "alias_count": 0,
                "training_sources": ["profile"],
            }
        ],
    }


def test_trained_hospital_directory_prefers_active_profile_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _ = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="current-hospital-profile",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="hospital-1",
            hospital_name="Current Profile Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    stale_registry = registry_with_alias(revision=1)
    stale_registry["hospitals"][0]["hospital_name"] = "Stale Alias Hospital"
    JsonAliasRepository(api.ALIAS_REGISTRY)._write_unlocked(stale_registry)

    response = client.get("/api/v2/hospitals/trained")

    assert response.status_code == 200
    assert response.json()["hospitals"] == [
        {
            "hospital_id": "hospital-1",
            "hospital_name": "Current Profile Hospital",
            "active_profile_count": 1,
            "alias_count": 1,
            "training_sources": ["profile", "reviewer_alias"],
        }
    ]


def test_hospital_link_reuses_profile_identity_and_rejects_duplicate_create(
    tmp_path: Path, monkeypatch
) -> None:
    client, store = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="machine-hospital-items",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="profile-machine-hospital",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="other-hospital-items",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="profile-other-hospital",
            hospital_name="Other Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    job_id, _ = completed_job(store)

    duplicate = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": True,
            "registry_revision": 0,
            "reason": "Verified hospital name from the document header",
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == {
        "code": "hospital_already_exists",
        "candidate_ids": ["profile-machine-hospital"],
    }

    unrelated = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "profile-other-hospital",
            "registry_revision": 0,
            "reason": "Attempted unrelated trained hospital selection",
        },
    )
    assert unrelated.status_code == 422
    assert unrelated.json()["detail"] == {
        "code": "hospital_selection_not_candidate",
        "candidate_ids": ["profile-machine-hospital"],
    }

    linked = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "profile-machine-hospital",
            "registry_revision": 0,
            "reason": "Selected the matching trained hospital identity",
        },
    )
    assert linked.status_code == 200
    hospital = JsonAliasRepository(api.ALIAS_REGISTRY).read()["hospitals"][0]
    assert hospital["hospital_id"] == "profile-machine-hospital"
    assert hospital["origins"] == ["profile", "reviewer_alias"]
    event = JsonAliasRepository(api.ALIAS_REGISTRY).read()["events"][-1]
    assert event["reviewer"] == "demo-reviewer"


def test_hospital_link_locked_revalidation_returns_fresh_candidate_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="machine-hospital-items",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="profile-machine-hospital",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    job_id, _ = completed_job(store)
    original_list = JsonProfileRepository.list_profiles
    calls = 0

    def changing_profiles(repository):
        nonlocal calls
        calls += 1
        return original_list(repository) if calls == 1 else []

    monkeypatch.setattr(JsonProfileRepository, "list_profiles", changing_profiles)

    response = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "profile-machine-hospital",
            "registry_revision": 0,
            "reason": "Candidate changed while waiting for coordinated locks",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "hospital_selection_not_candidate",
        "candidate_ids": [],
    }
    assert store.read_review(job_id)["revision"] == 0
    assert JsonAliasRepository(api.ALIAS_REGISTRY).read()["revision"] == 0


def test_hospital_link_persists_profile_metadata_from_locked_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="machine-hospital-current-name",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="profile-machine-hospital",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    job_id, _ = completed_job(store)
    original_list = JsonProfileRepository.list_profiles
    calls = 0

    def changing_profile_name(repository):
        nonlocal calls
        calls += 1
        profiles = original_list(repository)
        if calls == 1:
            return profiles
        return [
            profile.model_copy(update={"hospital_name": "MACHINE HOSPITAL"})
            for profile in profiles
        ]

    monkeypatch.setattr(JsonProfileRepository, "list_profiles", changing_profile_name)

    response = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "profile-machine-hospital",
            "registry_revision": 0,
            "reason": "Use the current locked profile identity",
        },
    )

    assert response.status_code == 200
    assert response.json()["hospital_name"] == "MACHINE HOSPITAL"
    registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    assert registry["hospitals"][0]["hospital_name"] == "MACHINE HOSPITAL"
    assert registry["hospitals"][0]["origins"] == ["profile", "reviewer_alias"]
    review = store.read_review(job_id)
    assert review["document_overrides"]["hospital_link"]["hospital_name"] == (
        "MACHINE HOSPITAL"
    )


def test_hospital_link_refreshes_existing_record_from_locked_profile_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="machine-hospital-existing-alias",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="profile-machine-hospital",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    existing_registry = registry_with_alias(revision=1)
    existing_registry["hospitals"][0]["hospital_id"] = "profile-machine-hospital"
    existing_registry["aliases"][0]["hospital_id"] = "profile-machine-hospital"
    JsonAliasRepository(api.ALIAS_REGISTRY)._write_unlocked(existing_registry)
    job_id, _ = completed_job(store)
    original_list = JsonProfileRepository.list_profiles
    calls = 0

    def changing_profile_name(repository):
        nonlocal calls
        calls += 1
        profiles = original_list(repository)
        if calls == 1:
            return profiles
        return [
            profile.model_copy(
                update={"hospital_name": "New Machine Medical Center"}
            )
            for profile in profiles
        ]

    monkeypatch.setattr(JsonProfileRepository, "list_profiles", changing_profile_name)

    response = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "profile-machine-hospital",
            "registry_revision": 1,
            "reason": "Refresh the existing alias record from the locked profile",
        },
    )

    assert response.status_code == 200
    assert response.json()["hospital_name"] == "New Machine Medical Center"
    registry = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    assert registry["hospitals"][0]["hospital_name"] == "New Machine Medical Center"
    assert registry["hospitals"][0]["name_variants"][0]["display_name"] == (
        "Machine Hospital"
    )
    assert (
        JsonAliasRepository.resolve_hospital(
            registry,
            "New Machine Medical Center",
        )
        == "profile-machine-hospital"
    )
    assert store.read_review(job_id)["document_overrides"]["hospital_link"][
        "hospital_name"
    ] == "New Machine Medical Center"
    registry_event = registry["events"][-1]
    assert registry_event["old_hospital_name"] == "Machine Hospital"
    assert registry_event["new_hospital_name"] == "New Machine Medical Center"
    assert registry_event["canonical_name_source"] == "active_profile"
    review_event = store.read_review(job_id)["events"][-1]
    assert review_event["changes"]["old_hospital_name"] == "Machine Hospital"
    assert review_event["changes"]["new_hospital_name"] == (
        "New Machine Medical Center"
    )
    assert review_event["changes"]["canonical_name_source"] == "active_profile"


def test_hospital_link_rejects_locked_profile_name_owned_by_another_hospital(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store = client_for(tmp_path / "jobs", monkeypatch)
    profile_registry = tmp_path / "profiles" / "registry.json"
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="machine-hospital-conflicting-name",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="hospital-1",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    JsonProfileRepository(profile_registry).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="other-hospital-owning-new-name",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="hospital-2",
            hospital_name="New Machine Medical Center",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    monkeypatch.setattr(api, "PROFILE_REGISTRY", profile_registry)
    registry = registry_with_alias(revision=1)
    JsonAliasRepository(api.ALIAS_REGISTRY)._write_unlocked(registry)
    job_id, _ = completed_job(store)
    original_list = JsonProfileRepository.list_profiles
    calls = 0

    def changing_profile_name(repository):
        nonlocal calls
        calls += 1
        profiles = original_list(repository)
        if calls == 1:
            return profiles
        return [
            profile.model_copy(
                update={"hospital_name": "New Machine Medical Center"}
            )
            if profile.hospital_id == "hospital-1"
            else profile
            for profile in profiles
        ]

    monkeypatch.setattr(JsonProfileRepository, "list_profiles", changing_profile_name)

    response = client.post(
        f"/api/v2/documents/{job_id}/hospital-link",
        headers={"If-Match": "0"},
        json={
            "create": False,
            "hospital_id": "hospital-1",
            "registry_revision": 1,
            "reason": "Reject a locked canonical name owned by another hospital",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "hospital_name_conflict",
        "candidate_ids": ["hospital-2"],
    }
    assert store.read_review(job_id)["revision"] == 0
    unchanged = JsonAliasRepository(api.ALIAS_REGISTRY).read()
    assert unchanged["revision"] == 1
    assert unchanged["hospitals"][0]["hospital_name"] == "Machine Hospital"


def test_hospital_registry_rejects_ambiguous_canonical_name_ownership() -> None:
    registry = registry_with_alias(revision=1)
    registry["hospitals"].append(
        {
            "hospital_id": "hospital-2",
            "hospital_name": "Other Hospital",
            "origins": ["reviewer_alias"],
            "name_variants": [
                {
                    "display_name": "Machine Hospital",
                    "normalized_name": "machine hospital",
                    "verified": True,
                    "source_document_id": "f" * 64,
                    "reviewer": "test-reviewer",
                    "reason": "Conflicting historical ownership",
                    "created_at": "2026-08-12T00:00:00Z",
                }
            ],
            "created_at": "2026-08-12T00:00:00Z",
            "updated_at": "2026-08-12T00:00:00Z",
        }
    )

    assert JsonAliasRepository.resolve_hospital(registry, "Machine Hospital") is None
    with pytest.raises(AliasRegistryFormatError, match="belongs to multiple hospitals"):
        JsonAliasRepository(Path("unused.json"))._validate(registry)


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
        response_holder.append(client.get(f"/api/v2/documents/{job_id}/exports/evidence.zip"))

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
