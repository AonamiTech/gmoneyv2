from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from gmoney.demo import reprocess as reprocess_module
from gmoney.demo.reprocess import (
    _validate_result,
    apply_staged_jobs,
    reprocess_jobs,
    rollback_jobs,
    stage_reprocess_jobs,
)
from gmoney.demo.store import JobStore, ReviewRevisionConflict


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


def source_tables(
    rows: list[dict[str, Any]],
    artifact_sha256: str,
    *,
    amount_in_description_cell: bool = False,
) -> list[dict[str, Any]]:
    header_description = evidence(artifact_sha256, "header-description")
    header_amount = evidence(artifact_sha256, "header-amount")
    printed_rows = []
    for index, canonical in enumerate(rows):
        description_evidence = canonical["field_evidence"]["description"]
        amount_evidence = canonical["field_evidence"].get("amount", [])
        description_cell_evidence = (
            [*description_evidence, *amount_evidence]
            if amount_in_description_cell
            else description_evidence
        )
        printed_rows.append(
            {
                "id": f"p1-t1-s1-r{index + 1}",
                "order": index,
                "canonical_row_id": canonical["id"],
                "cells": [
                    {
                        "column_id": "description",
                        "raw_value": canonical["description"],
                        "evidence": description_cell_evidence,
                        "validation_flags": [],
                    },
                    {
                        "column_id": "amount",
                        "raw_value": canonical["net_amount_raw"],
                        "evidence": (
                            description_evidence
                            if amount_in_description_cell
                            and canonical["net_amount_raw"] is not None
                            else amount_evidence
                        ),
                        "validation_flags": (
                            ["empty_cell"] if canonical["net_amount_raw"] is None else []
                        ),
                    },
                ],
                "validation_flags": [],
            }
        )
    return [
        {
            "id": "p1-t1-s1",
            "page_number": 1,
            "table_id": "p1-t1",
            "table_type": "item_ledger",
            "columns": [
                {
                    "id": "description",
                    "label": "Particular",
                    "order": 0,
                    "canonical_field": "description",
                    "evidence": [header_description],
                    "validation_flags": [],
                },
                {
                    "id": "amount",
                    "label": "Total",
                    "order": 1,
                    "canonical_field": "net_amount",
                    "evidence": [header_amount],
                    "validation_flags": [],
                },
            ],
            "rows": printed_rows,
            "validation_flags": [],
        }
    ]


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


def test_reprocess_validation_requires_printed_tables_for_canonical_rows(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)

    with pytest.raises(ValueError, match="source tables"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            old_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_rejects_mapped_field_in_the_wrong_source_cell(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": source_tables(
            new_rows,
            page_sha,
            amount_in_description_cell=True,
        ),
    }

    with pytest.raises(ValueError, match="net_amount.*source cell"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_rejects_printed_value_missing_from_canonical_row(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        1,
        {
            "id": "quantity",
            "label": "Qty",
            "order": 1,
            "canonical_field": "quantity",
            "evidence": [evidence(page_sha, "header-quantity")],
            "validation_flags": [],
        },
    )
    printed[0]["columns"][2]["order"] = 2
    printed[0]["rows"][0]["cells"].insert(
        1,
        {
            "column_id": "quantity",
            "raw_value": "2",
            "evidence": [evidence(page_sha, "quantity-token")],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    with pytest.raises(ValueError, match="quantity.*missing canonical value"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_preserves_job_and_review_with_backup(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row("new-row", page_sha),
        row("information-row", page_sha, role="informational", amount=None),
    ]
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": source_tables(new_rows, page_sha),
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
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    extracted_result = {
        **old_result,
        "source_tables": source_tables(old_result["rows"], page_sha),
    }

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
            extractor=MutatingExtractor(extracted_result),
        )
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_stage_and_apply_are_separate_review_checked_phases(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": source_tables(new_rows, page_sha),
    }

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
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": [replacement],
                "source_tables": source_tables([replacement], page_sha),
            }
        ),
    )
    assert summary["reprocessed"] == 1
    review = store.read_review(job_id)
    assert review["row_overrides"] == {}
    preserved = next(iter(review["added_rows"].values()))
    assert preserved["description"] == "Reviewer package charge"
    assert preserved["contract_version"] == "canonical_row_reviewer_v1"
    assert "reviewer_preserved" in preserved["validation_flags"]
    assert review["approval"] is None


def test_review_started_during_cutover_is_not_overwritten(
    tmp_path: Path, monkeypatch
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    marker_calls = 0
    mutation_started = threading.Event()
    mutation_outcome: list[str] = []
    mutation_thread: threading.Thread | None = None
    original_marker = reprocess_module._review_marker

    def mutate_review() -> None:
        mutation_started.set()
        try:
            store.mutate_review(
                job_id,
                1,
                lambda review: {
                    **review,
                    "events": [
                        *review["events"],
                        {"action": "concurrent-review"},
                    ],
                },
            )
        except ReviewRevisionConflict:
            mutation_outcome.append("conflict")
        else:
            mutation_outcome.append("saved")

    def racing_marker(job_dir: Path) -> tuple[bool, str | None]:
        nonlocal marker_calls, mutation_thread
        marker = original_marker(job_dir)
        marker_calls += 1
        if marker_calls == 2:
            mutation_thread = threading.Thread(target=mutate_review)
            mutation_thread.start()
            assert mutation_started.wait(timeout=1)
            mutation_thread.join(timeout=0.1)
        return marker

    monkeypatch.setattr(reprocess_module, "_review_marker", racing_marker)

    applied = apply_staged_jobs(
        root=tmp_path,
        stage_batch=Path(staged["staging_root"]),
    )
    assert applied["reprocessed"] == 1
    assert mutation_thread is not None
    mutation_thread.join(timeout=2)
    assert not mutation_thread.is_alive()
    assert mutation_outcome == ["conflict"]
    assert store.read_review(job_id)["revision"] == 2


def test_cutover_failure_restores_the_complete_old_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    stage_dir = Path(staged["staging_root"]) / job_id
    (stage_dir / "artifacts" / "new-only.txt").write_text("new")
    job_dir = store.job_dir(job_id)
    original_replace = Path.replace
    injected = False

    def failing_replace(path: Path, target: Path) -> Path:
        nonlocal injected
        if (
            not injected
            and path == stage_dir / "result.json"
            and target == job_dir / "result.json"
        ):
            injected = True
            raise OSError("injected cutover failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="injected cutover failure"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=Path(staged["staging_root"]),
        )

    assert json.loads((job_dir / "result.json").read_text()) == old_result
    assert not (job_dir / "artifacts" / "new-only.txt").exists()
    assert store.read_review(job_id)["revision"] == 1
    assert not (job_dir / ".cutover.json").exists()


def test_store_recovery_restores_an_interrupted_cutover(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    stage_dir = Path(staged["staging_root"]) / job_id
    (stage_dir / "artifacts" / "new-only.txt").write_text("new")
    job_dir = store.job_dir(job_id)
    backup_dir = store.jobs_root / ".reprocess-backups" / "interrupted" / job_id
    backup_dir.mkdir(parents=True)
    shutil.copy2(job_dir / "state.json", backup_dir / "state.json")
    shutil.copy2(job_dir / "review.json", backup_dir / "review.json")
    (job_dir / "artifacts").replace(backup_dir / "artifacts")
    (stage_dir / "artifacts").replace(job_dir / "artifacts")
    (job_dir / "result.json").replace(backup_dir / "result.json")
    (job_dir / ".cutover.json").write_text(
        json.dumps(
            {
                "version": "job_cutover_v1",
                "job_id": job_id,
                "stage_dir": str(stage_dir),
                "backup_dir": str(backup_dir),
            }
        )
    )

    store.recover()

    assert json.loads((job_dir / "result.json").read_text()) == old_result
    assert not (job_dir / "artifacts" / "new-only.txt").exists()
    assert store.read_review(job_id)["revision"] == 1
    assert not (job_dir / ".cutover.json").exists()


def test_batch_failure_rolls_back_jobs_already_cut_over(
    tmp_path: Path, monkeypatch
) -> None:
    first_store, first_id, first_old = setup_job(tmp_path)
    _, second_id, second_old = setup_job(tmp_path)
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row("new-row", page_sha)]
    new_result = {
        **first_old,
        "rows": new_rows,
        "source_tables": source_tables(new_rows, page_sha),
    }
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[first_id, second_id],
        extractor=FakeExtractor(new_result),
    )
    ordered_ids = sorted((first_id, second_id))
    failed_id = ordered_ids[1]
    failed_stage = Path(staged["staging_root"]) / failed_id / "result.json"
    failed_live = first_store.job_dir(failed_id) / "result.json"
    original_replace = Path.replace
    injected = False

    def failing_replace(path: Path, target: Path) -> Path:
        nonlocal injected
        if not injected and path == failed_stage and target == failed_live:
            injected = True
            raise OSError("second job failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="second job failed"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=Path(staged["staging_root"]),
        )

    assert json.loads(
        (first_store.job_dir(first_id) / "result.json").read_text()
    ) == first_old
    assert json.loads(
        (first_store.job_dir(second_id) / "result.json").read_text()
    ) == second_old
    assert first_store.read_review(first_id)["revision"] == 1
    assert first_store.read_review(second_id)["revision"] == 1
