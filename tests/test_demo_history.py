from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from gmoney.demo import api
from gmoney.demo.import_history import import_history
from gmoney.demo.store import JobStore


def _write_completed(store: JobStore, name: str, hospital: str) -> str:
    state = store.create(name)
    result = {
        "document_id": hashlib.sha256(name.encode()).hexdigest(),
        "source_name": name,
        "pages": 0,
        "page_assets": [],
        "hospital": {"name": hospital, "confidence": 0.9},
        "rows": [],
    }
    (store.job_dir(state["id"]) / "result.json").write_text(json.dumps(result))
    store.update(
        state["id"],
        status="complete",
        pages=0,
        page=0,
        row_count=0,
        hospital_name=hospital,
        hospital_confidence=0.9,
    )
    return state["id"]


def test_history_listing_supports_scope_search_pagination_and_expiry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = JobStore(tmp_path)
    first = _write_completed(store, "Bill 10.pdf", "Vijaya Group of Hospitals")
    _write_completed(store, "Bill 11.pdf", "Dr.Kamakshi Memorial Hospitals")
    queued = store.create("new.pdf")
    store.update(queued["id"], status="queued")
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "RETENTION_HOURS", 720)
    client = TestClient(api.app)

    active = client.get("/api/v2/documents?scope=active")
    assert active.status_code == 200
    assert [item["id"] for item in active.json()["documents"]] == [queued["id"]]

    history = client.get("/api/v2/documents?scope=history&query=vijaya&offset=0&limit=1")
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["has_more"] is False
    assert history.json()["documents"][0]["id"] == first
    assert history.json()["documents"][0]["expires_at"] is not None


def test_cleanup_uses_latest_review_activity_and_never_removes_active_jobs(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    expired = _write_completed(store, "expired.pdf", "Expired Hospital")
    reviewed = _write_completed(store, "reviewed.pdf", "Reviewed Hospital")
    active = store.create("active.pdf")
    store.update(active["id"], status="processing")
    old = (datetime.now(UTC) - timedelta(days=31)).isoformat().replace("+00:00", "Z")
    for job_id in (expired, reviewed, active["id"]):
        state_path = store.job_dir(job_id) / "state.json"
        state = json.loads(state_path.read_text())
        state["updated_at"] = old
        state_path.write_text(json.dumps(state))
    review = store.empty_review()
    (store.job_dir(reviewed) / "review.json").write_text(json.dumps(review))

    assert store.cleanup(720) == 1
    assert not store.job_dir(expired).exists()
    assert store.job_dir(reviewed).exists()
    assert store.job_dir(active["id"]).exists()


def test_history_import_is_hash_checked_hardlinked_and_tombstone_idempotent(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    root = data / "runtime"
    results = data / "results"
    artifacts = data / "artifacts"
    sources = data / "sources"
    for directory in (results, artifacts / "bill_11", sources):
        directory.mkdir(parents=True)
    source = sources / "Bill 11.pdf"
    source.write_bytes(b"%PDF-history-fixture")
    page = artifacts / "bill_11" / "pages" / "page-1.png"
    page.parent.mkdir(parents=True)
    page.write_bytes(b"PNG-history-fixture")
    document_id = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "document_id": document_id,
        "source_sha256": document_id,
        "source_name": source.name,
        "pages": 1,
        "page_assets": [
            {
                "page_number": 1,
                "relative_path": "pages/page-1.png",
                "artifact_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                "width": 100,
                "height": 200,
            }
        ],
        "hospital": {"name": "Kamakshi", "confidence": 0.9},
        "rows": [],
    }
    (results / "bill_11.json").write_text(json.dumps(result))

    first = import_history(
        root=root,
        results=results,
        artifacts=artifacts,
        sources=sources,
        batch_id="sample-11",
    )
    assert first["imported"] == 1
    store = JobStore(root)
    state = store.states()[0]
    imported_page = store.job_dir(state["id"]) / "artifacts" / "pages" / "page-1.png"
    assert imported_page.stat().st_ino == page.stat().st_ino

    # A crash after the atomic payload rename but before state/manifest writes
    # is recoverable without copying the evidence tree again.
    (store.job_dir(state["id"]) / "state.json").unlink()
    Path(first["manifest"]).unlink()
    resumed = import_history(
        root=root,
        results=results,
        artifacts=artifacts,
        sources=sources,
        batch_id="sample-11",
    )
    assert resumed["imported"] == 1
    assert store.read(state["id"])["status"] == "complete"

    store.delete(state["id"])
    second = import_history(
        root=root,
        results=results,
        artifacts=artifacts,
        sources=sources,
        batch_id="sample-11",
    )
    assert second["imported"] == 0
    assert second["skipped"] == 1
    assert second["documents"] == 1
    assert not store.job_dir(state["id"]).exists()
    assert page.read_bytes() == b"PNG-history-fixture"
