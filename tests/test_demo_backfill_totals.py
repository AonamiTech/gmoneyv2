from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from gmoney.demo.backfill_totals import backfill_totals
from gmoney.demo.store import JobStore
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
)


def completed_job(
    root: Path,
    *,
    output_version: str = "offline_accuracy_spine_v5",
    status: str = "complete",
) -> tuple[JobStore, str, Path]:
    store = JobStore(root)
    state = store.create("historic-bill.pdf")
    job_id = state["id"]
    source_path = store.job_dir(job_id) / "source.pdf"
    document = fitz.open()
    document.new_page(width=100, height=200)
    document.save(source_path)
    document.close()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    page_path = store.job_dir(job_id) / "artifacts" / "pages" / "page-1.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"page")
    page_sha = hashlib.sha256(page_path.read_bytes()).hexdigest()
    result_path = store.job_dir(job_id) / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "output_version": output_version,
                "document_total_version": "old-total-version",
                "document_totals_version": "old-totals-version",
                "document_total": None,
                "document_totals": [],
                "document_id": source_sha,
                "source_sha256": source_sha,
                "source_name": "historic-bill.pdf",
                "pages": 1,
                "page_assets": [
                    {
                        "document_sha256": source_sha,
                        "page_number": 1,
                        "artifact_sha256": page_sha,
                        "relative_path": "pages/page-1.png",
                        "width": 100,
                        "height": 200,
                        "dpi": 300,
                        "renderer": "test",
                        "renderer_version": "1",
                    }
                ],
                "source_tables": [],
                "token_manifest": [],
                "rows": [],
                "diagnostics": [
                    {
                        "diagnostic_id": "page-1",
                        "diagnostic_kind": "page",
                        "page_number": 1,
                        "page_classification": "blank",
                        "demonstrably_blank": True,
                    }
                ],
                "provider_usage": {
                    "initial": {"gemini_calls": 0, "gemini_measured_cost_usd": "0"},
                    "recovery": {"gemini_calls": 0, "gemini_measured_cost_usd": "0"},
                    "aggregate": {"gemini_calls": 0, "gemini_measured_cost_usd": "0"},
                },
                "recovery": {
                    "attempted": False,
                    "targets": [],
                    "untargeted_units_sha256": None,
                },
            }
        )
    )
    store.update(job_id, status=status, pages=1, page=1, row_count=0)
    return store, job_id, result_path


def test_backfill_is_dry_by_default_and_never_changes_live_results(
    tmp_path: Path,
) -> None:
    _, _, result_path = completed_job(tmp_path)
    original = result_path.read_bytes()

    summary = backfill_totals(root=tmp_path)

    assert summary["would_update"] == 1
    assert summary["updated"] == 0
    assert result_path.read_bytes() == original


def test_apply_publishes_validated_totals_only_reprojection(
    tmp_path: Path,
) -> None:
    store, job_id, result_path = completed_job(tmp_path)

    summary = backfill_totals(root=tmp_path, apply=True)

    assert summary["updated"] == 1
    result = json.loads(result_path.read_text())
    assert result["document_total_version"] == DOCUMENT_TOTAL_VERSION
    assert result["document_totals_version"] == DOCUMENT_TOTALS_VERSION
    assert result["semantic_validation"]["status"] == "passed"
    assert store.read(job_id)["status"] == "complete"


def test_legacy_results_require_full_reprocessing_and_remain_unchanged(
    tmp_path: Path,
) -> None:
    _, _, result_path = completed_job(
        tmp_path,
        output_version="offline_accuracy_spine_v3",
    )
    original = result_path.read_bytes()

    summary = backfill_totals(root=tmp_path, apply=True)

    assert summary["failed"] == 1
    assert summary["failures"][0]["error"] == "requires_full_reprocess"
    assert summary["staged"] == 0
    assert result_path.read_bytes() == original


def test_backfill_includes_needs_review_results(tmp_path: Path) -> None:
    _, _, _ = completed_job(tmp_path, status="needs_review")

    summary = backfill_totals(root=tmp_path, apply=True)

    assert summary["updated"] == 1
