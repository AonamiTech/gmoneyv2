from __future__ import annotations

import json
from pathlib import Path

from gmoney.demo.backfill_totals import backfill_totals
from gmoney.demo.store import JobStore
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
)


def completed_cached_job(root: Path) -> tuple[JobStore, str, Path]:
    store = JobStore(root)
    state = store.create("historic-bill.pdf")
    job_id = state["id"]
    result = {
        "output_version": "offline_accuracy_spine_v3",
        "document_id": "d" * 64,
        "pages": 1,
        "page_assets": [
            {
                "page_number": 1,
                "artifact_sha256": "a" * 64,
                "width": 1000,
                "height": 1400,
                "relative_path": "pages/page-1.png",
            }
        ],
        "rows": [],
    }
    result_path = store.job_dir(job_id) / "result.json"
    result_path.write_text(json.dumps(result))
    cache_path = store.job_dir(job_id) / "artifacts" / "inference" / "page-1.ocr.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "artifact_sha256": "a" * 64,
                "response": {
                    "output": {
                        "pages": [
                            {
                                "res": {
                                    "rec_polys": [
                                        [[100, 100], [340, 100], [340, 125], [100, 125]],
                                        [[800, 100], [940, 100], [940, 125], [800, 125]],
                                    ],
                                    "rec_texts": ["Net Bill Amount", "1,234.50"],
                                    "rec_scores": [0.99, 0.98],
                                }
                            }
                        ]
                    }
                },
            }
        )
    )
    store.update(job_id, status="complete", pages=1, page=1, row_count=0)
    return store, job_id, result_path


def test_backfill_is_dry_by_default_atomic_and_idempotent(tmp_path: Path) -> None:
    _, _, result_path = completed_cached_job(tmp_path)
    original = result_path.read_bytes()

    dry_run = backfill_totals(root=tmp_path)
    assert dry_run["would_update"] == 1
    assert dry_run["updated"] == 0
    assert result_path.read_bytes() == original

    applied = backfill_totals(root=tmp_path, apply=True)
    assert applied["updated"] == 1
    result = json.loads(result_path.read_text())
    assert result["document_total_version"] == DOCUMENT_TOTAL_VERSION
    assert result["document_totals_version"] == DOCUMENT_TOTALS_VERSION
    assert result["document_total"]["amount"] == "1234.50"
    assert result["document_totals"][0]["amount"] == "1234.50"

    repeated = backfill_totals(root=tmp_path, apply=True)
    assert repeated["already_current"] == 1
    assert repeated["updated"] == 0


def test_backfill_leaves_result_unchanged_when_cache_hash_differs(tmp_path: Path) -> None:
    store, job_id, result_path = completed_cached_job(tmp_path)
    original = result_path.read_bytes()
    cache_path = store.job_dir(job_id) / "artifacts" / "inference" / "page-1.ocr.json"
    envelope = json.loads(cache_path.read_text())
    envelope["artifact_sha256"] = "b" * 64
    cache_path.write_text(json.dumps(envelope))

    summary = backfill_totals(root=tmp_path, apply=True)
    assert summary["failed"] == 1
    assert "hash differs" in summary["failures"][0]["error"]
    assert result_path.read_bytes() == original


def test_backfill_includes_needs_review_results(tmp_path: Path) -> None:
    store, job_id, result_path = completed_cached_job(tmp_path)
    store.update(job_id, status="needs_review")

    summary = backfill_totals(root=tmp_path, apply=True)

    assert summary["updated"] == 1
    assert json.loads(result_path.read_text())["document_total"]["amount"] == "1234.50"
