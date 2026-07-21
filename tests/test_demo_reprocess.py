from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gmoney.demo.reprocess import (
    apply_staged_jobs,
    reprocess_jobs,
    rollback_jobs,
    stage_reprocess_jobs,
)
from gmoney.demo.store import JobStore


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evidence(artifact_sha256: str, token_id: str) -> dict[str, Any]:
    return {
        "page_number": 1,
        "table_id": "p1-t1",
        "polygon": {
            "points": [
                {"x": 1, "y": 1},
                {"x": 10, "y": 1},
                {"x": 10, "y": 10},
                {"x": 1, "y": 10},
            ]
        },
        "artifact_sha256": artifact_sha256,
        "token_ids": [token_id],
    }


def row(
    row_id: str,
    artifact_sha256: str,
    *,
    role: str = "detail",
    amount: str | None = "100.00",
) -> dict[str, Any]:
    description_evidence = evidence(artifact_sha256, "description-token")
    field_evidence = {"description": [description_evidence]}
    if role == "informational":
        field_evidence["service_date"] = [evidence(artifact_sha256, "date-token")]
    else:
        field_evidence["amount"] = [evidence(artifact_sha256, "amount-token")]
    return {
        "id": row_id,
        "contract_version": "canonical_row_v2",
        "created_at": "2026-07-20T00:00:00Z",
        "document_id": "d" * 64,
        "page_number": 1,
        "table_id": "p1-t1",
        "page_type": "itemized_charges",
        "table_type": "item_ledger",
        "row_order": 0 if role != "informational" else 1,
        "role": role,
        "review_disposition": "accepted",
        "section": "package",
        "description": "Package charge" if role != "informational" else "Included pathology",
        "service_date_raw": "20/01/2026" if role == "informational" else None,
        "service_date_iso": "2026-01-20" if role == "informational" else None,
        "request_no": None,
        "service_code": None,
        "hsn_code": None,
        "quantity_raw": None,
        "quantity": None,
        "unit_price_raw": None,
        "unit_price": None,
        "gross_amount_raw": None,
        "gross_amount": None,
        "discount_raw": None,
        "discount": None,
        "net_amount_raw": amount,
        "net_amount": amount,
        "evidence": list(field_evidence["description"]),
        "field_evidence": field_evidence,
        "candidate_ids": [],
        "source_routes": ["ocr_spatial_graph"],
        "validation_flags": [],
    }


def setup_job(tmp_path: Path) -> tuple[JobStore, str, dict[str, Any]]:
    store = JobStore(tmp_path)
    state = store.create("Bill 12.pdf")
    job_id = state["id"]
    job_dir = store.job_dir(job_id)
    source = b"%PDF-fixture"
    page = b"PNG fixture"
    (job_dir / "source.pdf").write_bytes(source)
    page_path = job_dir / "artifacts" / "pages" / "page-1.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(page)
    page_sha = digest(page)
    result = {
        "output_version": "offline_accuracy_spine_v3",
        "document_id": "d" * 64,
        "document_total": None,
        "source_sha256": digest(source),
        "source_name": "Bill 12.pdf",
        "hospital": None,
        "pages": 1,
        "page_assets": [
            {
                "page_number": 1,
                "artifact_sha256": page_sha,
                "width": 100,
                "height": 200,
                "relative_path": "pages/page-1.png",
            }
        ],
        "rows": [row("old-row", page_sha)],
        "diagnostics": [],
    }
    (job_dir / "result.json").write_text(json.dumps(result))
    review = store.empty_review()
    review.update(
        revision=1,
        row_overrides={
            "old-row": {
                "changes": {"description": "Reviewer package charge"},
                "reason": "Checked source",
            }
        },
        approval={"status": "approved"},
    )
    (job_dir / "review.json").write_text(json.dumps(review))
    store.update(job_id, status="complete", page=1, pages=1, row_count=1)
    return store, job_id, result


class FakeExtractor:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
        assert source.is_file()
        assert (artifact_root / "pages" / "page-1.png").is_file()
        return json.loads(json.dumps(self.result))


def test_reprocess_preserves_job_and_review_with_backup(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_result = {
        **old_result,
        "rows": [
            row("new-row", page_sha),
            row("information-row", page_sha, role="informational", amount=None),
        ],
    }
    dry_run = reprocess_jobs(root=tmp_path, job_ids=[job_id])
    assert dry_run["would_reprocess"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result

    summary = reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        apply=True,
        extractor=FakeExtractor(new_result),
    )
    assert summary["reprocessed"] == 1
    assert store.read(job_id)["row_count"] == 2
    current = json.loads((store.job_dir(job_id) / "result.json").read_text())
    assert [item["id"] for item in current["rows"]] == ["new-row", "information-row"]
    review = store.read_review(job_id)
    assert review["revision"] == 2
    assert review["approval"] is None
    assert set(review["row_overrides"]) == {"new-row"}
    assert review["events"][-1]["action"] == "document_reprocessed"
    backup = Path(summary["documents"][0]["backup"])
    assert json.loads((backup / "result.json").read_text()) == old_result
    assert (backup / "artifacts" / "pages" / "page-1.png").is_file()

    rolled_back = rollback_jobs(root=tmp_path, backup_batch=backup.parent)
    assert rolled_back["restored"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result
    assert store.read_review(job_id)["revision"] == 1


def test_reprocess_stops_if_review_changes_during_staging(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)

    class MutatingExtractor(FakeExtractor):
        def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
            review_path = store.job_dir(job_id) / "review.json"
            review = json.loads(review_path.read_text())
            review["revision"] = 2
            review_path.write_text(json.dumps(review))
            return super().extract(source, artifact_root)

    with pytest.raises(ValueError, match="review changed"):
        reprocess_jobs(
            root=tmp_path,
            job_ids=[job_id],
            apply=True,
            extractor=MutatingExtractor(old_result),
        )
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_stage_and_apply_are_separate_review_checked_phases(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_result = {**old_result, "rows": [row("new-row", page_sha)]}

    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        extractor=FakeExtractor(new_result),
    )
    assert staged["staged"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result

    applied = apply_staged_jobs(
        root=tmp_path,
        stage_batch=Path(staged["staging_root"]),
    )
    assert applied["reprocessed"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text())["rows"][0][
        "id"
    ] == "new-row"


def test_unmappable_reviewed_row_is_preserved_as_reviewer_row(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    replacement = row("unrelated-row", page_sha)
    replacement["description"] = "Unrelated upgraded row"
    replacement["field_evidence"]["description"][0]["token_ids"] = ["different-description"]
    replacement["field_evidence"]["amount"][0]["token_ids"] = ["different-amount"]

    summary = reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        apply=True,
        extractor=FakeExtractor({**old_result, "rows": [replacement]}),
    )
    assert summary["reprocessed"] == 1
    review = store.read_review(job_id)
    assert review["row_overrides"] == {}
    preserved = next(iter(review["added_rows"].values()))
    assert preserved["description"] == "Reviewer package charge"
    assert preserved["contract_version"] == "canonical_row_reviewer_v1"
    assert "reviewer_preserved" in preserved["validation_flags"]
    assert review["approval"] is None
