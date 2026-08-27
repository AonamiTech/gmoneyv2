from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import fitz
import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.demo import reprocess as reprocess_module
from gmoney.demo.reprocess import (
    _validate_result,
    apply_staged_jobs,
    reprocess_jobs,
    rollback_jobs,
    stage_reprocess_jobs,
)
from gmoney.demo.store import JobStore, JobTransactionError, ReviewRevisionConflict
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_rows import reconstruct_ocr_rows
from gmoney.extraction.offline import _link_source_tables
from gmoney.extraction.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationStatus,
)

VISUAL_AUDIT_CHECKS = {
    "hospital_identity",
    "printed_columns",
    "row_order_and_count",
    "cell_values",
    "explicit_totals",
    "non_ledger_exclusion",
}


def fixture_row_id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"gmoney-reprocess-fixture:{label}"))


_FIXTURE_PDFS: dict[bytes, bytes] = {}


def fixture_pdf(seed: bytes) -> bytes:
    cached = _FIXTURE_PDFS.get(seed)
    if cached is not None:
        return cached
    document = fitz.open()
    page_document = document.new_page(width=100, height=200)
    page_document.insert_text((10, 20), seed.decode(errors="replace"))
    payload = document.tobytes()
    document.close()
    _FIXTURE_PDFS[seed] = payload
    return payload


def _run_default_gpu_stage_then_wait(
    root_value: str,
    job_id: str,
    old_result: dict[str, Any],
    fail: bool,
    sender: Any,
) -> None:
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    class ProcessGpuExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
        ) -> dict[str, Any]:
            if fail:
                raise RuntimeError("maintenance extraction failed")
            return {
                **deepcopy(old_result),
                "rows": deepcopy(new_rows),
                "source_tables": source_tables(new_rows, page_sha),
            }

    reprocess_module.OfflineExtractor = ProcessGpuExtractor
    try:
        stage_reprocess_jobs(
            root=Path(root_value),
            job_ids=[job_id],
            paddle_device="gpu:0",
            vl_device="cuda:0",
        )
    except RuntimeError as error:
        sender.send(("failed", str(error), os.getpid()))
    else:
        sender.send(("staged", None, os.getpid()))
    while True:
        time.sleep(1)


def _report_reprocess_lock_entry(root_value: str, sender: Any) -> None:
    store = JobStore(Path(root_value))
    sender.send(("ready", str(store.inference_lock_path)))
    with store.inference_lock():
        sender.send(("entered", str(store.inference_lock_path)))
    sender.close()


def _run_default_gpu_stage_then_fork_competing_stage(
    root_value: str,
    job_id: str,
    old_result: dict[str, Any],
    owner_sender: Any,
    child_sender: Any,
) -> None:
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    class ProcessGpuExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
        ) -> dict[str, Any]:
            return {
                **deepcopy(old_result),
                "rows": deepcopy(new_rows),
                "source_tables": source_tables(new_rows, page_sha),
            }

    reprocess_module.OfflineExtractor = ProcessGpuExtractor
    stage_reprocess_jobs(
        root=Path(root_value),
        job_ids=[job_id],
        stage_root=Path(root_value) / "owner-stage",
        paddle_device="gpu:0",
    )
    owner_sender.send(("owner_staged", os.getpid()))
    child_pid = os.fork()
    if child_pid == 0:
        owner_sender.close()
        try:
            stage_reprocess_jobs(
                root=Path(root_value),
                job_ids=[job_id],
                stage_root=Path(root_value) / "child-stage",
                paddle_device="gpu:0",
            )
            child_sender.send(("child_staged", os.getpid()))
        finally:
            child_sender.close()
        os._exit(0)
    child_sender.close()
    owner_sender.send(("child_started", child_pid))
    while True:
        time.sleep(1)


def _run_default_gpu_stage_then_fork_survivor(
    root_value: str,
    job_id: str,
    old_result: dict[str, Any],
    sender: Any,
) -> None:
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    class ProcessGpuExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
        ) -> dict[str, Any]:
            return {
                **deepcopy(old_result),
                "rows": deepcopy(new_rows),
                "source_tables": source_tables(new_rows, page_sha),
            }

    reprocess_module.OfflineExtractor = ProcessGpuExtractor
    stage_reprocess_jobs(
        root=Path(root_value),
        job_ids=[job_id],
        paddle_device="gpu:0",
    )
    survivor_pid = os.fork()
    if survivor_pid == 0:
        sender.close()
        while True:
            time.sleep(1)
    sender.send(("forked_survivor", os.getpid(), survivor_pid))
    sender.close()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evidence(
    artifact_sha256: str,
    token_id: str,
    *,
    left: int = 1,
    top: int = 1,
    right: int = 10,
    bottom: int = 10,
) -> dict[str, Any]:
    return {
        "page_number": 1,
        "table_id": "p1-t1",
        "polygon": {
            "points": [
                {"x": left, "y": top},
                {"x": right, "y": top},
                {"x": right, "y": bottom},
                {"x": left, "y": bottom},
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


def setup_job(
    tmp_path: Path,
    *,
    source_name: str = "Bill 12.pdf",
    source: bytes = b"%PDF-fixture",
) -> tuple[JobStore, str, dict[str, Any]]:
    try:
        with fitz.open(stream=source, filetype="pdf") as candidate:
            if candidate.page_count < 1:
                raise ValueError("empty fixture PDF")
    except (ValueError, RuntimeError, fitz.FileDataError):
        source = fixture_pdf(source)
    store = JobStore(tmp_path)
    state = store.create(source_name)
    job_id = state["id"]
    job_dir = store.job_dir(job_id)
    page = b"PNG fixture"
    (job_dir / "source.pdf").write_bytes(source)
    page_path = job_dir / "artifacts" / "pages" / "page-1.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(page)
    page_sha = digest(page)
    result = {
        "output_version": "offline_accuracy_spine_v3",
        "document_id": digest(source),
        "document_total": None,
        "source_sha256": digest(source),
        "source_name": source_name,
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
        "rows": [row(fixture_row_id("old-row"), page_sha)],
        "diagnostics": [],
    }
    (job_dir / "result.json").write_text(json.dumps(result))
    review = store.empty_review()
    review.update(
        revision=1,
        row_overrides={
            fixture_row_id("old-row"): {
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
        self.calls: list[Path] = []

    def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
        assert source.is_file()
        assert (artifact_root / "pages" / "page-1.png").is_file()
        self.calls.append(source)
        return json.loads(json.dumps(self.result))


def write_passing_visual_audit(staging_root: Path) -> dict[str, Any]:
    manifest = json.loads((staging_root / "manifest.json").read_text())
    audit = {
        "version": "reprocess_visual_audit_v1",
        "sources": [
            {
                "source_sha256": source["source_sha256"],
                "reviewer": "visual-auditor@example.com",
                "reviewed_at": manifest["sealed_at"],
                "pages": [
                    {"page_number": page_number, "status": "pass", "notes": ""}
                    for page_number in range(1, source["page_count"] + 1)
                ],
                "checks": {check: "pass" for check in sorted(VISUAL_AUDIT_CHECKS)},
            }
            for source in manifest["sources"]
        ],
    }
    (staging_root / "visual-audit.json").write_text(json.dumps(audit))
    return audit


def stage_and_apply(
    *,
    root: Path,
    job_ids: list[str],
    extractor: Any,
) -> dict[str, Any]:
    staged = stage_reprocess_jobs(
        root=root,
        job_ids=job_ids,
        extractor=extractor,
    )
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    return apply_staged_jobs(root=root, stage_batch=staging_root)


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


def test_reprocess_validation_requires_exact_page_asset_numbers(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_result = {
        **old_result,
        "page_assets": [{**old_result["page_assets"][0], "page_number": 2}],
        "rows": new_rows,
        "source_tables": source_tables(new_rows, page_sha),
    }

    with pytest.raises(ValueError, match="page inventory"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_rejects_mapped_field_in_the_wrong_source_cell(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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


def test_reprocess_validation_requires_all_description_tokens_in_printed_cell(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_rows[0]["field_evidence"]["description"].append(evidence(page_sha, "coverage-token"))
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"][0]["cells"][0]["evidence"] = [evidence(page_sha, "description-token")]
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    with pytest.raises(ValueError, match="description evidence.*source cell"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_requires_exact_serial_grounding_for_blank_particular(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]

    def printed_token(
        index: int,
        text: str,
        box: tuple[float, float, float, float],
    ) -> OcrToken:
        left, top, right, bottom = box
        return OcrToken(
            token_id=f"token-{index}",
            page_number=1,
            text=text,
            confidence=0.99,
            polygon=Polygon(
                points=(
                    Point(x=left, y=top),
                    Point(x=right, y=top),
                    Point(x=right, y=bottom),
                    Point(x=left, y=bottom),
                )
            ),
            artifact_sha256=page_sha,
            model_name="fixture",
            model_version="1",
        )

    reconstructed = reconstruct_ocr_rows(
        (
            printed_token(0, "Sr.N", (50, 30, 90, 45)),
            printed_token(1, "Particular", (100, 30, 420, 45)),
            printed_token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
            printed_token(3, "Total", (880, 30, 970, 45)),
            printed_token(4, "0.", (50, 70, 70, 85)),
            printed_token(5, "300.00", (620, 70, 700, 85)),
            printed_token(6, "1", (760, 70, 780, 85)),
            printed_token(7, "300.00", (890, 70, 960, 85)),
        ),
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        old_result["document_id"],
        1,
        "p1-t1",
        page_sha,
        reconstructed.rows,
    )
    linked_tables = _link_source_tables(reconstructed.source_tables, canonical)
    assert len(canonical) == 1
    assert linked_tables[0].rows[0].canonical_row_id == str(canonical[0].id)

    new_result = {
        **old_result,
        "rows": [item.model_dump(mode="json") for item in canonical],
        "source_tables": [item.model_dump(mode="json") for item in linked_tables],
    }
    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )

    mismatched_serial = deepcopy(new_result)
    serial_cell = mismatched_serial["source_tables"][0]["rows"][0]["cells"][0]
    assert serial_cell["raw_value"] == "0."
    serial_cell["raw_value"] = "0)"
    with pytest.raises(ValueError, match="description.*missing printed value"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            mismatched_serial,
            store.job_dir(job_id) / "artifacts",
        )


@pytest.mark.parametrize(
    ("printed_date", "accepted"),
    (
        ("15/07/2026 11:31:00", True),
        ("15/07/2026 11:31:00 - MNEIPI/265604", True),
        ("15/07/2026 11:31:00 - MNEIPI/265604 E", True),
        ("15/07/2026 11:31:00 - MNEIPI/265604 X", False),
        ("15/07/2026 99:99:99", False),
        ("15/07/2026 99:99:99 - MNEIPI/265604 E", False),
        ("16/07/2026 11:31:00", False),
    ),
)
def test_reprocess_validation_compares_grounded_printed_time_by_canonical_date(
    tmp_path: Path,
    printed_date: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_rows[0]["service_date_raw"] = "15/07/2026"
    new_rows[0]["service_date_iso"] = "2026-07-15"
    new_rows[0]["field_evidence"]["service_date"] = [evidence(page_sha, "service-date-token")]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        0,
        {
            "id": "service-date",
            "label": "Date",
            "order": 0,
            "canonical_field": "service_date_raw",
            "evidence": [evidence(page_sha, "header-service-date")],
            "validation_flags": [],
        },
    )
    for order, column in enumerate(printed[0]["columns"]):
        column["order"] = order
    printed[0]["rows"][0]["cells"].insert(
        0,
        {
            "column_id": "service-date",
            "raw_value": printed_date,
            "evidence": [evidence(page_sha, "service-date-token")],
            "validation_flags": ["split_from_merged_ocr_token"],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="service_date_raw.*source cell"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


def test_reprocess_validation_accepts_grounded_group_service_date(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_rows[0]["service_date_raw"] = "15/07/2026"
    new_rows[0]["service_date_iso"] = "2026-07-15"
    new_rows[0]["field_evidence"]["service_date"] = [evidence(page_sha, "service-date-token")]
    new_rows[0]["validation_flags"] = ["service_date_inherited_from_group"]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        0,
        {
            "id": "sale-no",
            "label": "Sale No",
            "order": 0,
            "canonical_field": None,
            "evidence": [evidence(page_sha, "header-sale-no")],
            "validation_flags": [],
        },
    )
    for order, column in enumerate(printed[0]["columns"]):
        column["order"] = order
    printed[0]["rows"][0]["cells"].insert(
        0,
        {
            "column_id": "sale-no",
            "raw_value": "S54281",
            "evidence": [evidence(page_sha, "sale-no-token")],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "sale-no",
                    "raw_value": "15/07/2026",
                    "evidence": [evidence(page_sha, "service-date-token")],
                    "validation_flags": [],
                },
                {
                    "column_id": "description",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )

    printed[0]["rows"][1]["cells"][0]["raw_value"] = "16/07/2026"
    with pytest.raises(ValueError, match="service_date_raw lacks matching"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_accepts_grounded_recovered_unmapped_service_date(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_rows[0]["service_date_raw"] = "15/07/2026"
    new_rows[0]["service_date_iso"] = "2026-07-15"
    new_rows[0]["field_evidence"]["service_date"] = [evidence(page_sha, "service-date-token")]
    new_rows[0]["validation_flags"] = ["service_date_recovered_from_source_cell"]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        0,
        {
            "id": "mapped-date",
            "label": "Date",
            "order": 0,
            "canonical_field": "service_date_raw",
            "evidence": [evidence(page_sha, "header-date")],
            "validation_flags": [],
        },
    )
    printed[0]["columns"].insert(
        1,
        {
            "id": "mixed-reference",
            "label": "Reference / Date",
            "order": 1,
            "canonical_field": None,
            "evidence": [evidence(page_sha, "header-reference-date")],
            "validation_flags": [],
        },
    )
    for order, column in enumerate(printed[0]["columns"]):
        column["order"] = order
    printed[0]["rows"][0]["cells"].insert(
        0,
        {
            "column_id": "mapped-date",
            "raw_value": "15/07/2026",
            "evidence": [evidence(page_sha, "service-date-token")],
            "validation_flags": ["recovered_from_source_fragment"],
        },
    )
    printed[0]["rows"][0]["cells"].insert(
        1,
        {
            "column_id": "mixed-reference",
            "raw_value": "15/07/2026",
            "evidence": [evidence(page_sha, "service-date-token")],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )

    printed[0]["rows"][0]["cells"][0]["raw_value"] = "16/07/2026"
    with pytest.raises(ValueError, match="service_date_raw"):
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
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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


def test_reprocess_validation_accepts_grounded_day_quantity(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    new_rows[0]["quantity_raw"] = "2"
    new_rows[0]["quantity"] = "2"
    new_rows[0]["field_evidence"]["quantity"] = [evidence(page_sha, "quantity-token")]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        1,
        {
            "id": "quantity",
            "label": "Qty.",
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
            "raw_value": "2 Days",
            "evidence": [evidence(page_sha, "quantity-token")],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )


@pytest.mark.parametrize("printed_quantity", ("NNNN", None))
@pytest.mark.parametrize(("amount", "accepted"), (("90.00", True), ("89.00", False)))
def test_reprocess_validation_accepts_only_proven_derived_quantity(
    tmp_path: Path,
    printed_quantity: str | None,
    amount: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha, amount=amount)]
    canonical = new_rows[0]
    canonical["quantity_raw"] = "2"
    canonical["quantity"] = "2"
    canonical["unit_price_raw"] = "45.00"
    canonical["unit_price"] = "45.00"
    canonical["field_evidence"]["quantity"] = (
        [evidence(page_sha, "unreadable-quantity-token")]
        if printed_quantity is not None
        else [
            evidence(page_sha, "rate-token"),
            evidence(page_sha, "amount-token"),
        ]
    )
    canonical["field_evidence"]["rate"] = [evidence(page_sha, "rate-token")]
    canonical["validation_flags"] = ["quantity_derived_from_rate_amount"]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        1,
        {
            "id": "rate",
            "label": "Rate",
            "order": 1,
            "canonical_field": "unit_price",
            "evidence": [evidence(page_sha, "header-rate")],
            "validation_flags": [],
        },
    )
    printed[0]["columns"].insert(
        2,
        {
            "id": "quantity",
            "label": "Qty",
            "order": 2,
            "canonical_field": "quantity",
            "evidence": [evidence(page_sha, "header-quantity")],
            "validation_flags": [],
        },
    )
    printed[0]["columns"][3]["order"] = 3
    printed[0]["rows"][0]["cells"].insert(
        1,
        {
            "column_id": "rate",
            "raw_value": "45.00",
            "evidence": [evidence(page_sha, "rate-token")],
            "validation_flags": [],
        },
    )
    printed[0]["rows"][0]["cells"].insert(
        2,
        {
            "column_id": "quantity",
            "raw_value": printed_quantity,
            "evidence": (
                [evidence(page_sha, "unreadable-quantity-token")]
                if printed_quantity is not None
                else []
            ),
            "validation_flags": ([] if printed_quantity is not None else ["empty_cell"]),
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(
            ValueError,
            match="quantity.*(?:source cell|printed value)",
        ):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize("canonical_field", ["net_amount", "gross_amount"])
def test_reprocess_validation_rejects_unlinked_printed_financial_total(
    tmp_path: Path,
    canonical_field: str,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    if canonical_field == "gross_amount":
        new_rows[0]["gross_amount_raw"] = "100.00"
        new_rows[0]["gross_amount"] = "100.00"
        new_rows[0]["field_evidence"]["gross_amount"] = [evidence(page_sha, "amount-token")]
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"][1]["canonical_field"] = canonical_field
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Grand Total",
                    "evidence": [evidence(page_sha, "footer-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "100.00",
                    "evidence": [evidence(page_sha, "footer-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": ["unlinked_canonical_row"],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    with pytest.raises(ValueError, match="unlinked source row.*financial"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


@pytest.mark.parametrize(
    ("label", "subtotal", "accepted"),
    (
        ("Sub Total", "100.00", True),
        ("Sub Total", "99.00", False),
        ("Sub Total : Registration", "100.00", True),
        ("Sub Total : Registration", "99.00", False),
        ("Sub Total : Pharmacy", "100.00", False),
    ),
)
def test_reprocess_validation_accepts_only_matching_section_subtotal(
    tmp_path: Path,
    label: str,
    subtotal: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha, role="category_rollup")]
    new_rows[0]["description"] = "Registration"
    new_rows[0]["section"] = "registration"
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "category_summary"
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": label,
                    "evidence": [evidence(page_sha, "subtotal-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": subtotal,
                    "evidence": [evidence(page_sha, "subtotal-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("subtotal", "accepted"),
    (("100.00", True), ("99.00", False)),
)
def test_labeled_subtotal_matches_grounded_section_heading(
    tmp_path: Path,
    subtotal: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("pharmacy-row"), page_sha)]
    new_rows[0]["description"] = "Pharmacy sales bill"
    new_rows[0]["section"] = "pharmacy"
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].insert(
        0,
        {
            "id": "p1-t1-s1-r1",
            "order": 0,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Medicines & Consumables",
                    "evidence": [evidence(page_sha, "section-heading")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"][1]["id"] = "p1-t1-s1-r2"
    printed[0]["rows"][1]["order"] = 1
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r3",
            "order": 2,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Sub Total : Medicines & Consumables",
                    "evidence": [evidence(page_sha, "subtotal-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": subtotal,
                    "evidence": [evidence(page_sha, "subtotal-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(("bill_total", "accepted"), (("100.00", True), ("99.00", False)))
def test_reprocess_validation_accepts_only_matching_internal_bill_total(
    tmp_path: Path,
    bill_total: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-row"), page_sha, amount="100.00"),
        row(fixture_row_id("following-row"), page_sha, amount="50.00"),
    ]
    first_description_evidence = evidence(
        page_sha,
        "description-token",
        left=100,
        top=100,
        right=220,
        bottom=120,
    )
    first_amount_evidence = evidence(
        page_sha,
        "amount-token",
        left=700,
        top=100,
        right=780,
        bottom=140,
    )
    new_rows[0]["field_evidence"]["description"] = [first_description_evidence]
    new_rows[0]["field_evidence"]["amount"] = [first_amount_evidence]
    new_rows[0]["evidence"] = [first_description_evidence]
    new_rows[1]["row_order"] = 1
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].insert(
        1,
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Syringe",
                    "evidence": [
                        evidence(
                            page_sha,
                            "continuation-description",
                            left=100,
                            top=125,
                            right=180,
                            bottom=145,
                        )
                    ],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].insert(
        2,
        {
            "id": "p1-t1-s1-r3",
            "order": 2,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [evidence(page_sha, "bill-total-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": bill_total,
                    "evidence": [evidence(page_sha, "bill-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"][3]["id"] = "p1-t1-s1-r4"
    printed[0]["rows"][3]["order"] = 3
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("bill_total", "accepted"),
    (("150.00", True), ("149.00", False)),
)
def test_internal_bill_total_can_continue_across_compatible_page_tables(
    tmp_path: Path,
    bill_total: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page(width=100, height=200)
        page.insert_text((10, 20), f"page {page_number + 1}")
    source_payload = document.tobytes()
    document.close()
    (store.job_dir(job_id) / "source.pdf").write_bytes(source_payload)
    source_sha = digest(source_payload)
    old_result["document_id"] = source_sha
    old_result["source_sha256"] = source_sha
    first_page_sha = old_result["page_assets"][0]["artifact_sha256"]
    second_page = b"Second PNG fixture"
    second_page_path = store.job_dir(job_id) / "artifacts" / "pages" / "page-2.png"
    second_page_path.write_bytes(second_page)
    second_page_sha = digest(second_page)
    old_result["pages"] = 2
    old_result["page_assets"].append(
        {
            "page_number": 2,
            "artifact_sha256": second_page_sha,
            "width": 100,
            "height": 200,
            "relative_path": "pages/page-2.png",
        }
    )

    preceding_rows = [
        row(fixture_row_id("preceding-first"), first_page_sha, amount="40.00"),
        row(fixture_row_id("preceding-second"), first_page_sha, amount="60.00"),
    ]
    current_rows = [
        row(fixture_row_id("current-first"), second_page_sha, amount="50.00"),
        row(fixture_row_id("following-row"), second_page_sha, amount="25.00"),
    ]
    for canonical in current_rows:
        canonical["document_id"] = source_sha
        canonical["page_number"] = 2
        canonical["table_id"] = "p2-t1"
        for evidence_items in canonical["field_evidence"].values():
            for item in evidence_items:
                item["page_number"] = 2
                item["table_id"] = "p2-t1"
        for item in canonical["evidence"]:
            item["page_number"] = 2
            item["table_id"] = "p2-t1"

    for canonical in preceding_rows:
        canonical["document_id"] = source_sha

    first_tables = [source_tables([canonical], first_page_sha)[0] for canonical in preceding_rows]
    first_tables[1]["id"] = "p1-t1-s2"
    first_tables[1]["rows"][0]["id"] = "p1-t1-s2-r1"
    second_table = source_tables(current_rows, second_page_sha)[0]
    second_table["id"] = "p2-t1-s1"
    second_table["page_number"] = 2
    second_table["table_id"] = "p2-t1"
    second_table["rows"][0]["id"] = "p2-t1-s1-r1"
    for column in second_table["columns"]:
        for item in column["evidence"]:
            item["page_number"] = 2
            item["table_id"] = "p2-t1"
    for source_row in second_table["rows"]:
        for cell in source_row["cells"]:
            for item in cell["evidence"]:
                item["page_number"] = 2
                item["table_id"] = "p2-t1"
    subtotal_description = evidence(
        second_page_sha,
        "cross-page-bill-total-description",
    )
    subtotal_amount = evidence(
        second_page_sha,
        "cross-page-bill-total-amount",
    )
    for item in (subtotal_description, subtotal_amount):
        item["page_number"] = 2
        item["table_id"] = "p2-t1"
    second_table["rows"].insert(
        1,
        {
            "id": "p2-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [subtotal_description],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": bill_total,
                    "evidence": [subtotal_amount],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    second_table["rows"][2]["id"] = "p2-t1-s1-r3"
    second_table["rows"][2]["order"] = 2
    new_result = {
        **old_result,
        "rows": [*preceding_rows, *current_rows],
        "source_tables": [*first_tables, second_table],
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("label", "summary_amount", "accepted"),
    (
        ("Total Amount", "300.00", True),
        ("Total Amount", "299.00", False),
        ("IP Pharmacy Return", "-50.00", True),
        ("IP Pharmacy Return", "-49.00", False),
    ),
)
def test_pharmacy_summary_requires_exact_positive_or_return_arithmetic(
    tmp_path: Path,
    label: str,
    summary_amount: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-charge"), page_sha, amount="100.00"),
        row(fixture_row_id("second-charge"), page_sha, amount="200.00"),
        row(fixture_row_id("returned-charge"), page_sha, amount="-50.00"),
    ]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
        canonical["table_type"] = "pharmacy"
        canonical["role"] = "refund" if canonical["net_amount"].startswith("-") else "detail"
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "pharmacy"
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r4",
            "order": 3,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": label,
                    "evidence": [evidence(page_sha, "pharmacy-summary-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": summary_amount,
                    "evidence": [evidence(page_sha, "pharmacy-summary-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


def test_separate_pharmacy_tables_validate_their_own_tail_summaries(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-pharmacy-charge"), page_sha, amount="100.00"),
        row(fixture_row_id("second-pharmacy-charge"), page_sha, amount="50.00"),
    ]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
        canonical["table_type"] = "pharmacy"
    new_rows[1]["table_id"] = "p1-t2"
    for evidence_items in new_rows[1]["field_evidence"].values():
        for item in evidence_items:
            item["table_id"] = "p1-t2"
    for item in new_rows[1]["evidence"]:
        item["table_id"] = "p1-t2"

    printed = [source_tables([canonical], page_sha)[0] for canonical in new_rows]
    printed[1]["id"] = "p1-t2-s1"
    printed[1]["table_id"] = "p1-t2"
    for column in printed[1]["columns"]:
        for item in column["evidence"]:
            item["table_id"] = "p1-t2"
    for source_row in printed[1]["rows"]:
        source_row["id"] = "p1-t2-s1-r1"
        for cell in source_row["cells"]:
            for item in cell["evidence"]:
                item["table_id"] = "p1-t2"

    for table_index, (table, amount) in enumerate(
        zip(printed, ("100.00", "50.00"), strict=True),
        start=1,
    ):
        table["table_type"] = "pharmacy"
        summary_label = evidence(page_sha, f"summary-{table_index}-label")
        summary_amount = evidence(page_sha, f"summary-{table_index}-amount")
        summary_label["table_id"] = table["table_id"]
        summary_amount["table_id"] = table["table_id"]
        table["rows"].append(
            {
                "id": f"{table['id']}-r2",
                "order": 1,
                "canonical_row_id": None,
                "cells": [
                    {
                        "column_id": "description",
                        "raw_value": "Total Amount",
                        "evidence": [summary_label],
                        "validation_flags": [],
                    },
                    {
                        "column_id": "amount",
                        "raw_value": amount,
                        "evidence": [summary_amount],
                        "validation_flags": [],
                    },
                ],
                "validation_flags": [],
            }
        )

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        {
            **old_result,
            "rows": new_rows,
            "source_tables": printed,
        },
        store.job_dir(job_id) / "artifacts",
    )


@pytest.mark.parametrize(
    ("bill_total", "accepted"),
    (("150.00", True), ("149.00", False)),
)
def test_internal_bill_total_spans_intervening_text_within_prior_row_envelope(
    tmp_path: Path,
    bill_total: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-row"), page_sha, amount="100.00"),
        row(fixture_row_id("second-row"), page_sha, amount="50.00"),
        row(fixture_row_id("following-row"), page_sha, amount="25.00"),
    ]
    first_description_evidence = evidence(
        page_sha,
        "description-token",
        left=100,
        top=100,
        right=220,
        bottom=120,
    )
    first_amount_evidence = evidence(
        page_sha,
        "amount-token",
        left=700,
        top=100,
        right=780,
        bottom=140,
    )
    new_rows[0]["field_evidence"]["description"] = [first_description_evidence]
    new_rows[0]["field_evidence"]["amount"] = [first_amount_evidence]
    new_rows[0]["evidence"] = [first_description_evidence]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].insert(
        1,
        {
            "id": "intervening-continuation",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Syringe",
                    "evidence": [
                        evidence(
                            page_sha,
                            "intervening-continuation",
                            left=100,
                            top=125,
                            right=180,
                            bottom=145,
                        )
                    ],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].insert(
        3,
        {
            "id": "intervening-bill-total",
            "order": 3,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [evidence(page_sha, "intervening-bill-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": bill_total,
                    "evidence": [evidence(page_sha, "intervening-bill-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    for order, source_row in enumerate(printed[0]["rows"]):
        source_row["id"] = f"p1-t1-s1-r{order + 1}"
        source_row["order"] = order
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("heading", "bill_total", "accepted"),
    (
        ("NEXT SECTION", "50.00", True),
        ("NEXT SECTION", "150.00", False),
        ("7.57.5", "150.00", True),
        ("7.57.5", "50.00", False),
    ),
)
def test_internal_bill_total_does_not_cross_an_unlinked_section_heading(
    tmp_path: Path,
    heading: str,
    bill_total: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-section-row"), page_sha, amount="100.00"),
        row(fixture_row_id("second-section-row"), page_sha, amount="50.00"),
        row(fixture_row_id("following-row"), page_sha, amount="25.00"),
    ]
    first_description_evidence = evidence(
        page_sha,
        "description-token",
        left=100,
        top=100,
        right=220,
        bottom=120,
    )
    first_amount_evidence = evidence(
        page_sha,
        "amount-token",
        left=700,
        top=100,
        right=780,
        bottom=120,
    )
    new_rows[0]["field_evidence"]["description"] = [first_description_evidence]
    new_rows[0]["field_evidence"]["amount"] = [first_amount_evidence]
    new_rows[0]["evidence"] = [first_description_evidence]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].insert(
        1,
        {
            "id": "section-heading",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": heading,
                    "evidence": [
                        evidence(
                            page_sha,
                            "section-heading",
                            left=100,
                            top=125,
                            right=220,
                            bottom=145,
                        )
                    ],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].insert(
        3,
        {
            "id": "section-bill-total",
            "order": 3,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [evidence(page_sha, "section-bill-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": bill_total,
                    "evidence": [evidence(page_sha, "section-bill-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    for order, source_row in enumerate(printed[0]["rows"]):
        source_row["id"] = f"p1-t1-s1-r{order + 1}"
        source_row["order"] = order
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("bill_total", "accepted"),
    (("150.00", True), ("50.00", False)),
)
@pytest.mark.parametrize(
    "preceding_overlay_flags",
    ([], ["all_text_rotated"]),
)
def test_internal_bill_total_ignores_non_section_structured_overlay_fragments(
    tmp_path: Path,
    bill_total: str,
    accepted: bool,
    preceding_overlay_flags: list[str],
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-row"), page_sha, amount="100.00"),
        row(fixture_row_id("second-row"), page_sha, amount="50.00"),
        row(fixture_row_id("following-row"), page_sha, amount="25.00"),
    ]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
    printed = source_tables(new_rows, page_sha)
    printed[0]["columns"].insert(
        1,
        {
            "id": "request-number",
            "label": "Bill Number",
            "order": 1,
            "canonical_field": "request_no",
            "evidence": [evidence(page_sha, "request-number-header")],
            "validation_flags": [],
        },
    )
    for order, column in enumerate(printed[0]["columns"]):
        column["order"] = order
    for source_row in printed[0]["rows"]:
        source_row["cells"].insert(
            1,
            {
                "column_id": "request-number",
                "raw_value": None,
                "evidence": [],
                "validation_flags": ["empty_cell"],
            },
        )
    printed[0]["rows"].insert(
        1,
        {
            "id": "rotated-overlay",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "7.57.5",
                    "evidence": [evidence(page_sha, "numeric-overlay-fragment")],
                    "validation_flags": [],
                },
                {
                    "column_id": "request-number",
                    "raw_value": "No: 10&10/1,",
                    "evidence": [evidence(page_sha, "rotated-overlay-fragment")],
                    "validation_flags": preceding_overlay_flags,
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].insert(
        3,
        {
            "id": "section-bill-total",
            "order": 3,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [evidence(page_sha, "section-bill-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "request-number",
                    "raw_value": "P:044-42649097/90052",
                    "evidence": [evidence(page_sha, "rotated-phone-overlay-fragment")],
                    "validation_flags": ["all_text_rotated"],
                },
                {
                    "column_id": "amount",
                    "raw_value": bill_total,
                    "evidence": [evidence(page_sha, "section-bill-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    printed[0]["rows"].append(
        {
            "id": "following-bill-total",
            "order": 5,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "BILL TOTAL",
                    "evidence": [evidence(page_sha, "following-bill-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "request-number",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
                {
                    "column_id": "amount",
                    "raw_value": "25.00",
                    "evidence": [evidence(page_sha, "following-bill-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    for order, source_row in enumerate(printed[0]["rows"]):
        source_row["id"] = f"p1-t1-s1-r{order + 1}"
        source_row["order"] = order
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


def test_reprocess_validation_accepts_matching_total_across_header_segments(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-segment-row"), page_sha, amount="16000.00"),
        row(fixture_row_id("second-segment-row"), page_sha, amount="52230.96"),
    ]
    new_rows[1]["row_order"] = 1
    printed = source_tables(new_rows, page_sha)[0]
    first_segment = deepcopy(printed)
    first_segment["id"] = "p1-t1-s1"
    first_segment["rows"] = [first_segment["rows"][0]]
    first_segment["rows"][0]["id"] = "p1-t1-s1-r1"
    second_segment = deepcopy(printed)
    second_segment["id"] = "p1-t1-s2"
    second_segment["rows"] = [second_segment["rows"][1]]
    second_segment["rows"][0]["id"] = "p1-t1-s2-r1"
    second_segment["rows"][0]["order"] = 0
    second_segment["rows"].append(
        {
            "id": "p1-t1-s2-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Total",
                    "evidence": [evidence(page_sha, "table-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "68230.96",
                    "evidence": [evidence(page_sha, "table-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": [first_segment, second_segment],
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )


@pytest.mark.parametrize(
    ("label", "total", "accepted"),
    (
        ("Total", "68230.96", True),
        ("Total", "68230.95", False),
        ("Continued", "68230.96", False),
    ),
)
def test_reprocess_validation_only_accepts_matching_total_in_structured_lane(
    tmp_path: Path,
    label: str,
    total: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-segment-row"), page_sha, amount="16000.00"),
        row(fixture_row_id("second-segment-row"), page_sha, amount="52230.96"),
    ]
    new_rows[1]["row_order"] = 1
    printed = source_tables(new_rows, page_sha)[0]
    printed["columns"].insert(
        0,
        {
            "id": "service-date",
            "label": "Date",
            "order": 0,
            "canonical_field": "service_date_raw",
            "evidence": [evidence(page_sha, "date-header")],
            "validation_flags": [],
        },
    )
    for order, column in enumerate(printed["columns"]):
        column["order"] = order
    for source_row in printed["rows"]:
        source_row["cells"].insert(
            0,
            {
                "column_id": "service-date",
                "raw_value": None,
                "evidence": [],
                "validation_flags": ["empty_cell"],
            },
        )
    printed["rows"].append(
        {
            "id": "p1-t1-s1-r3",
            "order": 2,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "service-date",
                    "raw_value": label,
                    "evidence": [evidence(page_sha, "structured-total-label")],
                    "validation_flags": [],
                },
                {
                    "column_id": "description",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
                {
                    "column_id": "amount",
                    "raw_value": total,
                    "evidence": [evidence(page_sha, "structured-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": [printed],
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("boundary_label", "boundary_metadata"),
    (
        ("Grand Total", None),
        ("Total Bill Amount", None),
        ("Advance Received", None),
        ("Advance Received", "09/07/26, 475"),
    ),
)
@pytest.mark.parametrize(("subtotal", "accepted"), (("60.00", True), ("100.00", False)))
def test_reprocess_validation_resets_subtotal_at_financial_boundary(
    tmp_path: Path,
    boundary_label: str,
    boundary_metadata: str | None,
    subtotal: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-row"), page_sha, role="category_rollup", amount="40.00"),
        row(fixture_row_id("second-row"), page_sha, role="category_rollup", amount="60.00"),
    ]
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "category_summary"
    first, second = printed[0]["rows"]
    first["order"] = 0
    second["id"] = "p1-t1-s1-r3"
    second["order"] = 2
    printed[0]["rows"] = [
        first,
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": boundary_label,
                    "evidence": [evidence(page_sha, "grand-total-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "40.00",
                    "evidence": [evidence(page_sha, "grand-total-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
        second,
        {
            "id": "p1-t1-s1-r4",
            "order": 3,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Sub Total",
                    "evidence": [evidence(page_sha, "subtotal-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": subtotal,
                    "evidence": [evidence(page_sha, "subtotal-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        },
    ]
    if boundary_metadata is not None:
        for column in printed[0]["columns"]:
            column["order"] += 1
        printed[0]["columns"].insert(
            0,
            {
                "id": "metadata",
                "label": "Metadata",
                "order": 0,
                "canonical_field": None,
                "evidence": [evidence(page_sha, "metadata-header")],
                "validation_flags": [],
            },
        )
        for printed_row in printed[0]["rows"]:
            printed_row["cells"].insert(
                0,
                {
                    "column_id": "metadata",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            )
        printed[0]["rows"][1]["cells"][0] = {
            "column_id": "metadata",
            "raw_value": boundary_metadata,
            "evidence": [evidence(page_sha, "metadata-value")],
            "validation_flags": [],
        }
    primary_total = {
        "total_version": "document_total_v3",
        "amount_raw": "40.00",
        "amount": "40.00",
        "label": "Grand Total",
        "kind": "bill_total",
        "scope": "document",
        "page_number": 1,
        "evidence": evidence(page_sha, "grand-total-amount"),
        "confidence": 0.99,
        "source_route": "page_ocr_final_total",
        "context_id": "p1:p1-t1:document_final:o1",
        "context_kind": "document_final",
    }
    new_result = {
        **old_result,
        "document_total": primary_total,
        "document_totals_version": "document_totals_v2",
        "document_totals": [primary_total],
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


def test_reprocess_validation_accepts_verified_total_and_settlement_source_rows(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].extend(
        [
            {
                "id": "p1-t1-s1-r2",
                "order": 1,
                "canonical_row_id": None,
                "cells": [
                    {
                        "column_id": "description",
                        "raw_value": "Total Bill Amount",
                        "evidence": [evidence(page_sha, "total-description")],
                        "validation_flags": [],
                    },
                    {
                        "column_id": "amount",
                        "raw_value": "100.00",
                        "evidence": [evidence(page_sha, "total-amount")],
                        "validation_flags": [],
                    },
                ],
                "validation_flags": [],
            },
            {
                "id": "p1-t1-s1-r3",
                "order": 2,
                "canonical_row_id": None,
                "cells": [
                    {
                        "column_id": "description",
                        "raw_value": "Advance Received",
                        "evidence": [evidence(page_sha, "advance-description")],
                        "validation_flags": [],
                    },
                    {
                        "column_id": "amount",
                        "raw_value": "0.00",
                        "evidence": [evidence(page_sha, "advance-amount")],
                        "validation_flags": [],
                    },
                ],
                "validation_flags": [],
            },
        ]
    )
    for order, label in enumerate(
        (
            "Total",
            "Totals",
            "Sub Total",
            "Subtotal",
            "Bill Amount",
            "Net Medical Amount",
            "Pre Authorization Amount",
            "Co-Payment",
            "Claim Amount",
        ),
        start=3,
    ):
        printed[0]["rows"].append(
            {
                "id": f"p1-t1-s1-r{order + 1}",
                "order": order,
                "canonical_row_id": None,
                "cells": [
                    {
                        "column_id": "description",
                        "raw_value": label,
                        "evidence": [evidence(page_sha, f"short-total-description-{order}")],
                        "validation_flags": [],
                    },
                    {
                        "column_id": "amount",
                        "raw_value": "100.00",
                        "evidence": [evidence(page_sha, f"short-total-amount-{order}")],
                        "validation_flags": [],
                    },
                ],
                "validation_flags": [],
            }
        )
    printed[0]["columns"].append(
        {
            "id": "quantity",
            "label": "Quantity",
            "order": 2,
            "canonical_field": "quantity",
            "evidence": [evidence(page_sha, "quantity-header")],
            "validation_flags": [],
        }
    )
    for printed_row in printed[0]["rows"]:
        printed_row["cells"].append(
            {
                "column_id": "quantity",
                "raw_value": None,
                "evidence": [],
                "validation_flags": ["empty_cell"],
            }
        )
    displaced_order = len(printed[0]["rows"])
    printed[0]["rows"].append(
        {
            "id": f"p1-t1-s1-r{displaced_order + 1}",
            "order": displaced_order,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
                {
                    "column_id": "amount",
                    "raw_value": "100.00",
                    "evidence": [evidence(page_sha, "displaced-settlement-amount")],
                    "validation_flags": [],
                },
                {
                    "column_id": "quantity",
                    "raw_value": "Pre Authorization Amount",
                    "evidence": [evidence(page_sha, "displaced-settlement-label")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    primary_total = {
        "total_version": "document_total_v3",
        "amount_raw": "100.00",
        "amount": "100.00",
        "label": "Total Bill Amount",
        "kind": "bill_total",
        "scope": "document",
        "page_number": 1,
        "evidence": {
            **evidence(page_sha, "total-description"),
            "token_ids": ["total-description", "total-amount"],
        },
        "confidence": 0.99,
        "source_route": "page_ocr_final_total",
        "context_id": "p1:p1-t1:document_final:o1",
        "context_kind": "document_final",
    }
    new_result = {
        **old_result,
        "document_total": primary_total,
        "document_totals": [primary_total],
        "rows": new_rows,
        "source_tables": printed,
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )


def test_reprocess_validation_accepts_unique_repeated_grounded_summary(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha, role="category_rollup")]
    new_rows[0]["description"] = "Package Name: Coronary Angiography (CAG)"
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "category_summary"
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Package (IPD) - Coronary",
                    "evidence": [evidence(page_sha, "duplicate-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "100.00",
                    "evidence": [evidence(page_sha, "duplicate-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    _validate_result(
        store.job_dir(job_id) / "source.pdf",
        old_result,
        new_result,
        store.job_dir(job_id) / "artifacts",
    )


@pytest.mark.parametrize(
    (
        "printed_description",
        "second_description",
        "printed_quantity",
        "printed_amount",
        "accepted",
    ),
    (
        ("CONSULTING CHARGES PAR DAY", None, "2", "100.00", True),
        ("CONSULTING CHARGES", None, "2", "100.00", False),
        ("CONSULTING CHARGES PAR DAY", None, "2", "99.00", False),
        ("CONSULTING CHARGES PAR DAY", None, "3", "100.00", False),
        ("CONSULTING CHARGES PAR DAY", None, "3 Days", "100.00", False),
        ("CONSULTING CHARGES PAR DAY", "MRI Service", "2", "100.00", False),
    ),
)
def test_reprocess_validation_accepts_only_exact_repeated_detail_summary(
    tmp_path: Path,
    printed_description: str,
    second_description: str | None,
    printed_quantity: str,
    printed_amount: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [
        row(fixture_row_id("first-detail"), page_sha),
        row(fixture_row_id("second-detail"), page_sha),
    ]
    for order, canonical in enumerate(new_rows):
        canonical["row_order"] = order
        canonical["description"] = "CONSULTING CHARGES PAR DAY"
        canonical["quantity_raw"] = "2"
        canonical["quantity"] = "2"
        canonical["field_evidence"]["quantity"] = [evidence(page_sha, f"quantity-{order}")]
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "category_summary"
    printed[0]["columns"][1]["order"] = 2
    printed[0]["columns"].insert(
        1,
        {
            "id": "quantity",
            "label": "Unit/Days",
            "order": 1,
            "canonical_field": "quantity",
            "evidence": [evidence(page_sha, "quantity-header")],
            "validation_flags": [],
        },
    )
    for order, printed_row in enumerate(printed[0]["rows"]):
        printed_row["cells"].insert(
            1,
            {
                "column_id": "quantity",
                "raw_value": "2",
                "evidence": [evidence(page_sha, f"quantity-{order}")],
                "validation_flags": [],
            },
        )
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r3",
            "order": 2,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": printed_description,
                    "evidence": [evidence(page_sha, "summary-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "quantity",
                    "raw_value": printed_quantity,
                    "evidence": [evidence(page_sha, "summary-quantity")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": printed_amount,
                    "evidence": [evidence(page_sha, "summary-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    if second_description is not None:
        printed[0]["columns"][2]["order"] = 3
        printed[0]["columns"].insert(
            2,
            {
                "id": "second-description",
                "label": "Service",
                "order": 2,
                "canonical_field": "description",
                "evidence": [evidence(page_sha, "second-description-header")],
                "validation_flags": [],
            },
        )
        for printed_row in printed[0]["rows"][:-1]:
            printed_row["cells"].insert(
                2,
                {
                    "column_id": "second-description",
                    "raw_value": "CONSULTING CHARGES PAR DAY",
                    "evidence": [evidence(page_sha, "description-token")],
                    "validation_flags": [],
                },
            )
        printed[0]["rows"][-1]["cells"].insert(
            2,
            {
                "column_id": "second-description",
                "raw_value": second_description,
                "evidence": [evidence(page_sha, "second-summary-description")],
                "validation_flags": [],
            },
        )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    "printed_description",
    (
        "Total Knee Replacement",
        "Deposit Implant Charge",
        "Advance Received Physiotherapy",
        "Deposit Amount Implant Charge",
        "Payment Details Consultation Fee",
    ),
)
def test_reprocess_validation_does_not_misclassify_billable_description_as_footer(
    tmp_path: Path,
    printed_description: str,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": printed_description,
                    "evidence": [evidence(page_sha, "unlinked-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "100.00",
                    "evidence": [evidence(page_sha, "unlinked-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "document_total": {
            "total_version": "document_total_v2",
            "amount_raw": "100.00",
            "amount": "100.00",
            "label": "Total Bill Amount",
            "kind": "bill_total",
            "scope": "document",
            "page_number": 1,
            "evidence": evidence(page_sha, "document-total"),
            "confidence": 0.99,
            "source_route": "page_ocr_final_total",
        },
        "rows": new_rows,
        "source_tables": printed,
    }

    with pytest.raises(ValueError, match="unlinked source row.*financial"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


@pytest.mark.parametrize(
    ("printed_label", "printed_description", "variant", "accepted"),
    (
        ("Discount (Rs.):", None, "live", True),
        ("Advance/Received Amount:", None, "live", True),
        ("Discount Service", None, "live", False),
        ("Discount", "MRI Service", "live", False),
        ("Discount", None, "extra_financial", False),
        ("Discount", None, "decoy_mapping", False),
        ("Discount", None, "duplicate_description", False),
    ),
)
def test_reprocess_validation_only_accepts_unambiguous_settlement_rows(
    tmp_path: Path,
    printed_label: str,
    printed_description: str | None,
    variant: str,
    accepted: bool,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    printed = source_tables(new_rows, page_sha)
    for column in printed[0]["columns"]:
        column["order"] += 1
    printed[0]["columns"].insert(
        0,
        {
            "id": "metadata",
            "label": "10/07/26,",
            "order": 0,
            "canonical_field": None,
            "evidence": [evidence(page_sha, "metadata-header")],
            "validation_flags": [],
        },
    )
    printed[0]["columns"][2]["order"] = 3
    printed[0]["columns"].insert(
        2,
        {
            "id": "settlement-label",
            "label": "Unit/Days",
            "order": 2,
            "canonical_field": "quantity",
            "evidence": [evidence(page_sha, "settlement-header")],
            "validation_flags": [],
        },
    )
    printed[0]["rows"][0]["cells"].insert(
        0,
        {
            "column_id": "metadata",
            "raw_value": None,
            "evidence": [],
            "validation_flags": ["empty_cell"],
        },
    )
    printed[0]["rows"][0]["cells"].insert(
        2,
        {
            "column_id": "settlement-label",
            "raw_value": None,
            "evidence": [],
            "validation_flags": ["empty_cell"],
        },
    )
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "metadata",
                    "raw_value": "10/07/26, 442",
                    "evidence": [evidence(page_sha, "metadata-value")],
                    "validation_flags": [],
                },
                {
                    "column_id": "description",
                    "raw_value": printed_description,
                    "evidence": (
                        [evidence(page_sha, "billable-description")] if printed_description else []
                    ),
                    "validation_flags": ([] if printed_description else ["empty_cell"]),
                },
                {
                    "column_id": "settlement-label",
                    "raw_value": printed_label,
                    "evidence": [evidence(page_sha, "discount-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "0.00",
                    "evidence": [evidence(page_sha, "discount-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )

    def insert_footer_column(
        index: int,
        *,
        column_id: str,
        canonical_field: str,
        raw_value: str,
    ) -> None:
        for column in printed[0]["columns"][index:]:
            column["order"] += 1
        printed[0]["columns"].insert(
            index,
            {
                "id": column_id,
                "label": column_id.replace("-", " ").title(),
                "order": index,
                "canonical_field": canonical_field,
                "evidence": [evidence(page_sha, f"{column_id}-header")],
                "validation_flags": [],
            },
        )
        printed[0]["rows"][0]["cells"].insert(
            index,
            {
                "column_id": column_id,
                "raw_value": (
                    "100.00"
                    if canonical_field == "net_amount"
                    else ("Package charge" if canonical_field == "description" else None)
                ),
                "evidence": (
                    [evidence(page_sha, "amount-token")]
                    if canonical_field == "net_amount"
                    else (
                        [evidence(page_sha, "description-token")]
                        if canonical_field == "description"
                        else []
                    )
                ),
                "validation_flags": (
                    [] if canonical_field in {"net_amount", "description"} else ["empty_cell"]
                ),
            },
        )
        printed[0]["rows"][1]["cells"].insert(
            index,
            {
                "column_id": column_id,
                "raw_value": raw_value,
                "evidence": [evidence(page_sha, f"{column_id}-value")],
                "validation_flags": [],
            },
        )

    if variant == "extra_financial":
        insert_footer_column(
            2,
            column_id="gross",
            canonical_field="gross_amount",
            raw_value="100.00",
        )
    elif variant == "decoy_mapping":
        insert_footer_column(
            3,
            column_id="decoy-net",
            canonical_field="net_amount",
            raw_value="not recorded",
        )
    elif variant == "duplicate_description":
        insert_footer_column(
            4,
            column_id="second-description",
            canonical_field="description",
            raw_value="MRI Service",
        )

    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    if accepted:
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )
    else:
        with pytest.raises(ValueError, match="unlinked source row.*financial"):
            _validate_result(
                store.job_dir(job_id) / "source.pdf",
                old_result,
                new_result,
                store.job_dir(job_id) / "artifacts",
            )


@pytest.mark.parametrize(
    ("printed_description", "canonical_description"),
    (
        ("Cardiac Investigation", "Cardiac Package Coronary"),
        ("Package Cardiac Investigation", "Cardiac Package Coronary"),
        ("Package Coronary Care Unit", "Package Name Coronary Angiography CAG"),
        (
            "Department Summary Cardiac Investigation",
            "Department Summary Cardiac Surgery",
        ),
    ),
)
def test_reprocess_validation_rejects_weak_one_word_summary_match(
    tmp_path: Path,
    printed_description: str,
    canonical_description: str,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha, role="category_rollup")]
    new_rows[0]["description"] = canonical_description
    printed = source_tables(new_rows, page_sha)
    printed[0]["table_type"] = "category_summary"
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": printed_description,
                    "evidence": [evidence(page_sha, "unlinked-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": "100.00",
                    "evidence": [evidence(page_sha, "unlinked-amount")],
                    "validation_flags": [],
                },
            ],
            "validation_flags": [],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

    with pytest.raises(ValueError, match="unlinked source row.*financial"):
        _validate_result(
            store.job_dir(job_id) / "source.pdf",
            old_result,
            new_result,
            store.job_dir(job_id) / "artifacts",
        )


def test_reprocess_validation_allows_unlinked_non_ledger_text_without_amount(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    printed = source_tables(new_rows, page_sha)
    printed[0]["rows"].append(
        {
            "id": "p1-t1-s1-r2",
            "order": 1,
            "canonical_row_id": None,
            "cells": [
                {
                    "column_id": "description",
                    "raw_value": "Payment due within seven days",
                    "evidence": [evidence(page_sha, "footer-description")],
                    "validation_flags": [],
                },
                {
                    "column_id": "amount",
                    "raw_value": None,
                    "evidence": [],
                    "validation_flags": ["empty_cell"],
                },
            ],
            "validation_flags": ["unlinked_canonical_row"],
        }
    )
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": printed,
    }

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
        row(fixture_row_id("new-row"), page_sha),
        row(fixture_row_id("information-row"), page_sha, role="informational", amount=None),
    ]
    new_result = {
        **old_result,
        "rows": new_rows,
        "source_tables": source_tables(new_rows, page_sha),
    }
    dry_run = reprocess_jobs(root=tmp_path, job_ids=[job_id])
    assert dry_run["would_reprocess"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result

    summary = stage_and_apply(
        root=tmp_path,
        job_ids=[job_id],
        extractor=FakeExtractor(new_result),
    )
    assert summary["reprocessed"] == 1
    assert store.read(job_id)["row_count"] == 2
    current = json.loads((store.job_dir(job_id) / "result.json").read_text())
    assert [item["id"] for item in current["rows"]] == [
        fixture_row_id("new-row"),
        fixture_row_id("information-row"),
    ]
    review = store.read_review(job_id)
    assert review["revision"] == 2
    assert review["approval"] is None
    assert set(review["row_overrides"]) == {fixture_row_id("new-row")}
    assert review["events"][-1]["action"] == "document_reprocessed"
    backup = Path(summary["documents"][0]["backup"])
    assert json.loads((backup / "result.json").read_text()) == old_result
    assert (backup / "artifacts" / "pages" / "page-1.png").is_file()

    rolled_back = rollback_jobs(root=tmp_path, backup_batch=backup.parent)
    assert rolled_back["restored"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result
    assert store.read_review(job_id)["revision"] == 1
    displaced = Path(rolled_back["displaced_root"])
    assert displaced.parent == store.jobs_root / ".reprocess-rollback-current"
    assert json.loads((displaced / ".rollback.json").read_text())["version"] == (
        "reprocess_rollback_batch_v1"
    )
    assert json.loads((displaced / ".committed.json").read_text())["version"] == (
        "reprocess_rollback_commit_v1"
    )
    assert not (store.job_dir(job_id) / ".cutover.json").exists()


def test_reprocess_publishes_fresh_needs_review_quality_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    issue = ValidationIssue(
        id="ambiguous-total",
        code="ambiguous_primary_total",
        severity=ValidationSeverity.BLOCKING,
        message="No unique final bill total context",
        field="document_total",
    )
    report = ValidationReport(
        status=ValidationStatus.NEEDS_REVIEW,
        issues=(issue,),
    )
    monkeypatch.setattr(
        reprocess_module,
        "validate_extraction_result",
        lambda *_: report,
    )

    stage_and_apply(
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

    state = store.read(job_id)
    assert state["status"] == "needs_review"
    assert state["validation_status"] == "needs_review"
    assert state["validation_issue_codes"] == ["ambiguous_primary_total"]
    current = json.loads((store.job_dir(job_id) / "result.json").read_text())
    assert current["semantic_validation"] == report.model_dump(mode="json")


def test_review_migration_preserves_weaker_colliding_fragment_override() -> None:
    page_sha = "a" * 64
    main = row("old-main", page_sha)
    main["description"] = "Mupimet Ointment"
    main["field_evidence"]["description"] = [evidence(page_sha, "main-description")]
    main["field_evidence"]["amount"] = [evidence(page_sha, "main-amount")]
    fragment = row("old-fragment", page_sha)
    fragment["description"] = "5Gm"
    fragment["field_evidence"]["description"] = [evidence(page_sha, "fragment-description")]
    fragment["field_evidence"]["amount"] = [evidence(page_sha, "fragment-total")]
    combined = row("new-combined", page_sha)
    combined["description"] = "Mupimet Ointment 5Gm"
    combined["field_evidence"]["description"] = [
        evidence(page_sha, "main-description"),
        evidence(page_sha, "fragment-description"),
    ]
    combined["field_evidence"]["amount"] = [evidence(page_sha, "main-amount")]
    old_result = {"rows": [fragment, main]}
    new_result = {"rows": [combined]}
    review = {
        "revision": 4,
        "row_overrides": {
            "old-fragment": {
                "changes": {"review_disposition": "rejected"},
                "reason": "duplicate fragment",
            },
            "old-main": {
                "changes": {
                    "quantity_raw": "1",
                    "quantity": "1",
                    "review_disposition": "accepted",
                },
                "reason": "verified charge",
            },
        },
        "added_rows": {},
        "issue_overrides": {},
        "events": [],
    }

    migrated = reprocess_module._migrate_review(
        "job-id",
        old_result,
        new_result,
        review,
    )

    assert migrated["row_overrides"] == {
        "new-combined": {
            **review["row_overrides"]["old-main"],
            "changes": {
                "quantity_raw": "1",
                "quantity": "1",
            },
        }
    }
    assert len(migrated["added_rows"]) == 1
    preserved = next(iter(migrated["added_rows"].values()))
    assert preserved["description"] == "5Gm"
    assert preserved["review_disposition"] == "rejected"
    assert "reviewer_preserved" in preserved["validation_flags"]
    assert migrated["events"][-1]["changes"]["preserved_unmapped_row_overrides"] == ["old-fragment"]


def test_review_migration_does_not_reapply_unchanged_machine_fields() -> None:
    page_sha = "a" * 64
    old = row(fixture_row_id("old-row"), page_sha)
    old["description"] = "07/2026 PI123 Truncated Product"
    upgraded = row(fixture_row_id("new-row"), page_sha)
    upgraded["description"] = "Complete Product Name"
    upgraded["field_evidence"] = deepcopy(old["field_evidence"])
    review = {
        "revision": 1,
        "row_overrides": {
            fixture_row_id("old-row"): {
                "changes": {
                    "description": old["description"],
                    "net_amount_raw": old["net_amount_raw"],
                    "net_amount": old["net_amount"],
                    "quantity_raw": "2",
                    "quantity": "2",
                    "review_disposition": old["review_disposition"],
                    "role": old["role"],
                },
                "reason": "Quantity corrected",
            }
        },
        "added_rows": {},
        "issue_overrides": {},
        "events": [],
    }

    migrated = reprocess_module._migrate_review(
        "job-id",
        {"rows": [old]},
        {"rows": [upgraded]},
        review,
    )

    assert migrated["row_overrides"][fixture_row_id("new-row")]["changes"] == {
        "quantity_raw": "2",
        "quantity": "2",
    }


def test_review_migration_maps_unique_grounded_financial_row_when_token_ids_change() -> None:
    page_sha = "a" * 64
    old = row(fixture_row_id("old-row"), page_sha, amount="30.69")
    old["description"] = "Dispovan 5Ml"
    old["quantity_raw"] = "3"
    old["quantity"] = "3"
    upgraded = row(fixture_row_id("new-row"), page_sha, amount="30.69")
    upgraded["description"] = "Dispovan 5Ml Syringe"
    upgraded["quantity_raw"] = "3"
    upgraded["quantity"] = "3"
    upgraded["field_evidence"] = {
        "description": [evidence(page_sha, "new-description")],
        "amount": [evidence(page_sha, "new-amount")],
    }
    review = {
        "revision": 1,
        "row_overrides": {
            fixture_row_id("old-row"): {
                "changes": {
                    "quantity_raw": "3",
                    "quantity": "3",
                },
                "reason": "Quantity verified",
            }
        },
        "added_rows": {},
        "issue_overrides": {},
        "events": [],
    }

    migrated = reprocess_module._migrate_review(
        "job-id",
        {"rows": [old]},
        {"rows": [upgraded]},
        review,
    )

    assert migrated["row_overrides"] == {
        fixture_row_id("new-row"): {
            **review["row_overrides"][fixture_row_id("old-row")],
            "changes": {},
        }
    }
    assert migrated["added_rows"] == {}


def test_review_migration_subsumes_identical_colliding_corrections() -> None:
    page_sha = "a" * 64
    main = row("old-main", page_sha, amount="60")
    main["description"] = "Gauze Swabs"
    fragment = row("old-fragment", page_sha, amount="60")
    fragment["description"] = "Bill Number ProductName"
    fragment["field_evidence"] = {
        "description": [evidence(page_sha, "fragment-description")],
        "amount": [evidence(page_sha, "fragment-amount")],
    }
    upgraded = row(fixture_row_id("new-row"), page_sha, amount="60")
    upgraded["description"] = "Gauze Swabs 7.5 x 7.5"
    upgraded["quantity_raw"] = "1"
    upgraded["quantity"] = "1"
    upgraded["field_evidence"] = {
        "description": [evidence(page_sha, "new-description")],
        "amount": [evidence(page_sha, "new-amount")],
    }
    correction = {
        "changes": {"quantity_raw": "1", "quantity": "1"},
        "reason": "Quantity verified",
    }
    review = {
        "revision": 2,
        "row_overrides": {
            "old-main": correction,
            "old-fragment": {
                **correction,
                "reason": "Quantity was missing",
            },
        },
        "added_rows": {},
        "issue_overrides": {},
        "events": [],
    }

    migrated = reprocess_module._migrate_review(
        "job-id",
        {"rows": [main, fragment]},
        {"rows": [upgraded]},
        review,
    )

    assert migrated["row_overrides"] == {
        fixture_row_id("new-row"): {
            **correction,
        }
    }
    assert migrated["added_rows"] == {}
    assert migrated["events"][-1]["changes"]["subsumed_row_overrides"] == ["old-fragment"]


def test_manual_rollback_rejects_review_created_after_deployment(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    applied = stage_and_apply(
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
    current_result = json.loads((store.job_dir(job_id) / "result.json").read_text())
    store.mutate_review(
        job_id,
        2,
        lambda review: {
            **review,
            "events": [*review["events"], {"action": "post-deployment-review"}],
        },
    )

    with pytest.raises(ValueError, match="review changed after deployment"):
        rollback_jobs(root=tmp_path, backup_batch=Path(applied["backup_root"]))

    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == current_result
    assert store.read_review(job_id)["revision"] == 3
    assert store.read_review(job_id)["events"][-1]["action"] == "post-deployment-review"


def test_manual_rollback_final_review_compare_and_swap(tmp_path: Path, monkeypatch) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    applied = stage_and_apply(
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
    current_result = json.loads((store.job_dir(job_id) / "result.json").read_text())
    original_job_lock = JobStore.job_lock
    injected = False
    exclusive_count = 0

    def racing_job_lock(self: JobStore, locked_job_id: str, *, exclusive: bool) -> Any:
        nonlocal exclusive_count, injected
        if exclusive:
            exclusive_count += 1
        # The first exclusive acquisition is the stable preflight read. Inject
        # only at the final rollback CAS acquisition.
        if exclusive and exclusive_count == 2 and not injected:
            injected = True
            review_path = self.job_dir(locked_job_id) / "review.json"
            review = json.loads(review_path.read_text())
            review["revision"] += 1
            review["events"].append({"action": "concurrent-review"})
            review_path.write_text(json.dumps(review))
        return original_job_lock(self, locked_job_id, exclusive=exclusive)

    monkeypatch.setattr(JobStore, "job_lock", racing_job_lock)

    with pytest.raises(ValueError, match="review changed during rollback"):
        rollback_jobs(root=tmp_path, backup_batch=Path(applied["backup_root"]))

    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == current_result
    assert store.read_review(job_id)["revision"] == 3
    assert store.read_review(job_id)["events"][-1]["action"] == "concurrent-review"


def test_manual_rollback_batch_failure_restores_live_and_backup_batches(
    tmp_path: Path, monkeypatch
) -> None:
    store, first_id, first_old = setup_job(tmp_path)
    _, second_id, second_old = setup_job(tmp_path)
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    applied = stage_and_apply(
        root=tmp_path,
        job_ids=[first_id, second_id],
        extractor=FakeExtractor(
            {
                **first_old,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    backup_batch = Path(applied["backup_root"])
    current_results = {
        job_id: json.loads((store.job_dir(job_id) / "result.json").read_text())
        for job_id in (first_id, second_id)
    }
    failed_id = sorted((first_id, second_id))[1]
    failed_backup = backup_batch / failed_id / "result.json"
    failed_live = store.job_dir(failed_id) / "result.json"
    original_replace = Path.replace
    injected = False

    def failing_replace(path: Path, target: Path) -> Path:
        nonlocal injected
        if not injected and path == failed_backup and target == failed_live:
            injected = True
            raise OSError("injected manual rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="injected manual rollback failure"):
        rollback_jobs(root=tmp_path, backup_batch=backup_batch)

    for job_id, old_result in ((first_id, first_old), (second_id, second_old)):
        assert (
            json.loads((store.job_dir(job_id) / "result.json").read_text())
            == (current_results[job_id])
        )
        assert store.read_review(job_id)["revision"] == 2
        assert json.loads((backup_batch / job_id / "result.json").read_text()) == old_result
        assert (backup_batch / job_id / "artifacts" / "pages" / "page-1.png").is_file()
        assert not (store.job_dir(job_id) / ".cutover.json").exists()


def test_store_recovery_restores_interrupted_manual_rollback(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    applied = stage_and_apply(
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
    backup_dir = Path(applied["backup_root"]) / job_id
    job_dir = store.job_dir(job_id)
    current_result = json.loads((job_dir / "result.json").read_text())
    current_review = store.read_review(job_id)
    displaced_root = store.jobs_root / ".reprocess-rollback-current" / "interrupted"
    displaced = displaced_root / job_id
    displaced.mkdir(parents=True)
    shutil.copy2(job_dir / "state.json", displaced / "state.json")
    shutil.copy2(job_dir / "review.json", displaced / "review.json")
    (job_dir / "artifacts").replace(displaced / "artifacts")
    (job_dir / "result.json").replace(displaced / "result.json")
    (backup_dir / "artifacts").replace(job_dir / "artifacts")
    (backup_dir / "result.json").replace(job_dir / "result.json")
    (job_dir / "state.json").write_text((backup_dir / "state.json").read_text())
    (job_dir / "review.json").write_text((backup_dir / "review.json").read_text())
    commit_marker = displaced_root / ".committed.json"
    (displaced_root / ".rollback.json").write_text(
        json.dumps(
            {
                "version": "reprocess_rollback_batch_v1",
                "job_ids": [job_id],
            }
        )
    )
    (job_dir / ".cutover.json").write_text(
        json.dumps(
            {
                "version": "job_rollback_v1",
                "job_id": job_id,
                "backup_dir": str(backup_dir),
                "displaced_dir": str(displaced),
                "commit_marker": str(commit_marker),
            }
        )
    )

    store.recover()

    assert json.loads((job_dir / "result.json").read_text()) == current_result
    assert store.read_review(job_id) == current_review
    assert json.loads((backup_dir / "result.json").read_text()) == old_result
    assert (backup_dir / "artifacts" / "pages" / "page-1.png").is_file()
    assert not (job_dir / ".cutover.json").exists()


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
        stage_reprocess_jobs(
            root=tmp_path,
            job_ids=[job_id],
            extractor=MutatingExtractor(extracted_result),
        )
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_direct_apply_requires_the_two_phase_visual_audit_workflow(
    tmp_path: Path,
) -> None:
    _, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    extractor = FakeExtractor(
        {
            **old_result,
            "rows": new_rows,
            "source_tables": source_tables(new_rows, page_sha),
        }
    )

    with pytest.raises(ValueError, match="two-phase.*visual audit"):
        reprocess_jobs(
            root=tmp_path,
            job_ids=[job_id],
            apply=True,
            extractor=extractor,
        )

    assert extractor.calls == []


def test_stage_and_apply_are_separate_review_checked_phases(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    write_passing_visual_audit(Path(staged["staging_root"]))

    applied = apply_staged_jobs(
        root=tmp_path,
        stage_batch=Path(staged["staging_root"]),
    )
    assert applied["reprocessed"] == 1
    assert json.loads((store.job_dir(job_id) / "result.json").read_text())["rows"][0][
        "id"
    ] == fixture_row_id("new-row")


def test_stage_extracts_identical_sources_once_and_preserves_per_job_reviews(
    tmp_path: Path,
) -> None:
    store, first_id, first_old = setup_job(
        tmp_path,
        source_name="First copy.pdf",
    )
    _, second_id, second_old = setup_job(
        tmp_path,
        source_name="Second copy.pdf",
    )
    second_review = store.read_review(second_id)
    second_review["revision"] = 7
    second_review["row_overrides"][fixture_row_id("old-row")]["changes"]["description"] = (
        "Second reviewer correction"
    )
    (store.job_dir(second_id) / "review.json").write_text(json.dumps(second_review))
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    extractor = FakeExtractor(
        {
            **first_old,
            "source_name": "extractor representative name",
            "rows": new_rows,
            "source_tables": source_tables(new_rows, page_sha),
        }
    )

    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[second_id, first_id],
        extractor=extractor,
    )

    staging_root = Path(staged["staging_root"])
    ordered_ids = sorted((first_id, second_id))
    representative_id = ordered_ids[0]
    assert extractor.calls == [store.job_dir(representative_id) / "source.pdf"]
    manifest = json.loads((staging_root / "manifest.json").read_text())
    assert manifest["version"] == "reprocess_stage_v2"
    assert manifest["sealed_at"] == manifest["created_at"]
    assert len(manifest["sources"]) == 1
    source_group = manifest["sources"][0]
    assert source_group["source_sha256"] == first_old["source_sha256"]
    assert source_group["representative_job_id"] == representative_id
    assert source_group["job_ids"] == ordered_ids
    assert source_group["page_count"] == 1
    assert source_group["source_names"] == ["First copy.pdf", "Second copy.pdf"]
    assert len(source_group["extraction_sha256"]) == 64
    assert all(len(document["staged_payload_sha256"]) == 64 for document in manifest["documents"])
    for job_id, old_result in (
        (first_id, first_old),
        (second_id, second_old),
    ):
        staged_result = json.loads((staging_root / job_id / "result.json").read_text())
        assert staged_result["source_name"] == old_result["source_name"]
        assert (staging_root / job_id / "artifacts" / "pages" / "page-1.png").is_file()

    first_review = json.loads((staging_root / first_id / "review.json").read_text())
    second_staged_review = json.loads((staging_root / second_id / "review.json").read_text())
    assert first_review["revision"] == 2
    assert second_staged_review["revision"] == 8
    assert first_review["row_overrides"][fixture_row_id("new-row")]["changes"]["description"] == (
        "Reviewer package charge"
    )
    assert (
        second_staged_review["row_overrides"][fixture_row_id("new-row")]["changes"]["description"]
        == "Second reviewer correction"
    )
    assert first_review["events"][-1]["target_id"] == first_id
    assert second_staged_review["events"][-1]["target_id"] == second_id


def test_default_gpu_stage_holds_shared_lock_for_every_source_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, first_id, first_old = setup_job(
        tmp_path,
        source_name="First source.pdf",
        source=b"%PDF-first-source",
    )
    _, second_id, _ = setup_job(
        tmp_path,
        source_name="Second source.pdf",
        source=b"%PDF-second-source",
    )
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    events: list[tuple[str, Any]] = []

    class TrackingLock:
        closed = False

        def close(self) -> None:
            self.closed = True

    tracking_lock = TrackingLock()

    def tracking_inference_lock(locked_store: JobStore) -> TrackingLock:
        events.append(("lock_enter", locked_store.inference_lock_path))
        return tracking_lock

    class DefaultGpuExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            assert not tracking_lock.closed
            events.append(
                (
                    "constructed",
                    (vl_url, paddle_device, vl_device),
                )
            )

        def extract(
            self,
            source: Path,
            artifact_root: Path,
        ) -> dict[str, Any]:
            assert not tracking_lock.closed
            events.append(("extract", source))
            source_sha = digest(source.read_bytes())
            return {
                **deepcopy(first_old),
                "source_sha256": source_sha,
                "document_id": source_sha,
                "rows": deepcopy(new_rows),
                "source_tables": source_tables(new_rows, page_sha),
            }

    monkeypatch.setattr(JobStore, "acquire_inference_lock", tracking_inference_lock)
    monkeypatch.setattr(
        reprocess_module,
        "OfflineExtractor",
        DefaultGpuExtractor,
    )

    try:
        staged = stage_reprocess_jobs(
            root=tmp_path,
            job_ids=[first_id, second_id],
            paddle_device="gpu:0",
            vl_device="cuda:0",
        )

        assert staged["staged"] == 2
        assert events[0] == ("lock_enter", store.inference_lock_path)
        assert events[1] == (
            "constructed",
            ("http://127.0.0.1:8111", "gpu:0", "cuda:0"),
        )
        assert {event[1] for event in events if event[0] == "extract"} == {
            store.job_dir(first_id) / "source.pdf",
            store.job_dir(second_id) / "source.pdf",
        }
        assert not tracking_lock.closed
    finally:
        retained = getattr(reprocess_module, "_gpu_inference_locks", {})
        for lock in retained.values():
            lock.close()
        retained.clear()


def test_repeated_default_gpu_stages_reuse_the_process_lifetime_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    acquire_calls: list[Path] = []
    original_acquire = JobStore.acquire_inference_lock

    def tracking_acquire(locked_store: JobStore) -> Any:
        acquire_calls.append(locked_store.inference_lock_path)
        return original_acquire(locked_store)

    class DefaultGpuExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
        ) -> dict[str, Any]:
            return {
                **deepcopy(old_result),
                "rows": deepcopy(new_rows),
                "source_tables": source_tables(new_rows, page_sha),
            }

    monkeypatch.setattr(JobStore, "acquire_inference_lock", tracking_acquire)
    monkeypatch.setattr(reprocess_module, "OfflineExtractor", DefaultGpuExtractor)

    try:
        for stage_name in ("first", "second"):
            staged = stage_reprocess_jobs(
                root=tmp_path,
                job_ids=[job_id],
                stage_root=tmp_path / stage_name,
                paddle_device="gpu:0",
            )
            assert staged["staged"] == 1
        assert acquire_calls == [store.inference_lock_path]
    finally:
        retained = getattr(reprocess_module, "_gpu_inference_locks", {})
        for lock in retained.values():
            lock.close()
        retained.clear()


@pytest.mark.parametrize(
    ("fail", "expected"),
    (
        (False, ("staged", None)),
        (True, ("failed", "maintenance extraction failed")),
    ),
    ids=("success", "exception"),
)
def test_default_gpu_stage_keeps_lock_until_maintenance_process_exits(
    tmp_path: Path,
    fail: bool,
    expected: tuple[str, str | None],
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    holder_context = multiprocessing.get_context("fork")
    holder_receiver, holder_sender = holder_context.Pipe(duplex=False)
    holder = holder_context.Process(
        target=_run_default_gpu_stage_then_wait,
        args=(str(tmp_path), job_id, old_result, fail, holder_sender),
    )
    holder.start()
    holder_sender.close()
    contender = None
    contender_receiver = None
    try:
        assert holder_receiver.poll(5)
        status, detail, holder_pid = holder_receiver.recv()
        assert (status, detail) == expected
        assert holder_pid == holder.pid

        contender_context = multiprocessing.get_context("spawn")
        contender_receiver, contender_sender = contender_context.Pipe(duplex=False)
        contender = contender_context.Process(
            target=_report_reprocess_lock_entry,
            args=(str(tmp_path), contender_sender),
        )
        contender.start()
        contender_sender.close()
        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "ready",
            str(store.inference_lock_path),
        )
        assert not contender_receiver.poll(0.25)

        holder.terminate()
        holder.join(timeout=5)
        assert holder.pid == holder_pid
        assert not holder.is_alive()

        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "entered",
            str(store.inference_lock_path),
        )
    finally:
        holder_receiver.close()
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if contender_receiver is not None:
            contender_receiver.close()
        if contender is not None:
            contender.join(timeout=5)
            if contender.is_alive():
                contender.terminate()
                contender.join(timeout=5)

    assert holder.exitcode is not None
    assert contender is not None
    assert contender.exitcode == 0


def test_forked_competing_stage_reacquires_after_owner_process_exits(
    tmp_path: Path,
) -> None:
    _, job_id, old_result = setup_job(tmp_path)
    context = multiprocessing.get_context("fork")
    owner_receiver, owner_sender = context.Pipe(duplex=False)
    child_receiver, child_sender = context.Pipe(duplex=False)
    owner = context.Process(
        target=_run_default_gpu_stage_then_fork_competing_stage,
        args=(
            str(tmp_path),
            job_id,
            old_result,
            owner_sender,
            child_sender,
        ),
    )
    owner.start()
    owner_sender.close()
    child_sender.close()
    child_pid = None
    try:
        assert owner_receiver.poll(5)
        assert owner_receiver.recv() == ("owner_staged", owner.pid)
        assert owner_receiver.poll(5)
        event, child_pid = owner_receiver.recv()
        assert event == "child_started"
        assert not child_receiver.poll(0.25)

        owner.terminate()
        owner.join(timeout=5)
        assert not owner.is_alive()

        assert child_receiver.poll(5)
        assert child_receiver.recv() == ("child_staged", child_pid)
    finally:
        owner_receiver.close()
        child_receiver.close()
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, 15)

    assert owner.exitcode is not None


def test_forked_unrelated_child_does_not_extend_owner_lock_lifetime(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    owner_context = multiprocessing.get_context("fork")
    owner_receiver, owner_sender = owner_context.Pipe(duplex=False)
    owner = owner_context.Process(
        target=_run_default_gpu_stage_then_fork_survivor,
        args=(str(tmp_path), job_id, old_result, owner_sender),
    )
    owner.start()
    owner_sender.close()
    survivor_pid = None
    contender = None
    contender_receiver = None
    try:
        assert owner_receiver.poll(5)
        event, owner_pid, survivor_pid = owner_receiver.recv()
        assert event == "forked_survivor"
        assert owner_pid == owner.pid
        owner.join(timeout=5)
        assert not owner.is_alive()
        os.kill(survivor_pid, 0)

        contender_context = multiprocessing.get_context("spawn")
        contender_receiver, contender_sender = contender_context.Pipe(duplex=False)
        contender = contender_context.Process(
            target=_report_reprocess_lock_entry,
            args=(str(tmp_path), contender_sender),
        )
        contender.start()
        contender_sender.close()
        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "ready",
            str(store.inference_lock_path),
        )
        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "entered",
            str(store.inference_lock_path),
        )
    finally:
        owner_receiver.close()
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=5)
        if contender_receiver is not None:
            contender_receiver.close()
        if contender is not None:
            contender.join(timeout=5)
            if contender.is_alive():
                contender.terminate()
                contender.join(timeout=5)
        if survivor_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(survivor_pid, 15)

    assert owner.exitcode == 0
    assert contender is not None
    assert contender.exitcode == 0


def test_injected_reprocess_extractor_does_not_acquire_gpu_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    def fail_if_locked(locked_store: JobStore) -> Any:
        raise AssertionError(
            f"injected extractor unexpectedly locked {locked_store.inference_lock_path}"
        )

    monkeypatch.setattr(JobStore, "acquire_inference_lock", fail_if_locked)

    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        paddle_device="gpu:0",
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )

    assert staged["staged"] == 1


def test_falsey_injected_reprocess_extractor_remains_dependency_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    class FalseyExtractor(FakeExtractor):
        def __bool__(self) -> bool:
            return False

    class UnexpectedDefaultExtractor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("default extractor was constructed")

    monkeypatch.setattr(
        reprocess_module,
        "OfflineExtractor",
        UnexpectedDefaultExtractor,
    )

    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        paddle_device="gpu:0",
        extractor=FalseyExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )

    assert staged["staged"] == 1


def test_duplicate_review_change_during_staging_aborts_the_group(
    tmp_path: Path,
) -> None:
    store, first_id, first_old = setup_job(tmp_path, source_name="First copy.pdf")
    _, second_id, second_old = setup_job(tmp_path, source_name="Second copy.pdf")
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    extracted_result = {
        **first_old,
        "rows": [row(fixture_row_id("new-row"), page_sha)],
    }
    extracted_result["source_tables"] = source_tables(
        extracted_result["rows"],
        page_sha,
    )
    changed_id = sorted((first_id, second_id))[1]

    class MutatingExtractor(FakeExtractor):
        def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
            review_path = store.job_dir(changed_id) / "review.json"
            review = json.loads(review_path.read_text())
            review["revision"] += 1
            review["events"].append({"action": "concurrent-review"})
            review_path.write_text(json.dumps(review))
            return super().extract(source, artifact_root)

    with pytest.raises(ValueError, match="review changed while staging"):
        stage_reprocess_jobs(
            root=tmp_path,
            job_ids=[first_id, second_id],
            extractor=MutatingExtractor(extracted_result),
        )

    assert json.loads((store.job_dir(first_id) / "result.json").read_text()) == first_old
    assert json.loads((store.job_dir(second_id) / "result.json").read_text()) == (second_old)


def test_duplicate_review_change_before_apply_aborts_without_any_cutover(
    tmp_path: Path,
) -> None:
    store, first_id, first_old = setup_job(tmp_path, source_name="First copy.pdf")
    _, second_id, second_old = setup_job(tmp_path, source_name="Second copy.pdf")
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[first_id, second_id],
        extractor=FakeExtractor(
            {
                **first_old,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    changed_id = sorted((first_id, second_id))[1]
    store.mutate_review(
        changed_id,
        1,
        lambda review: {
            **review,
            "events": [*review["events"], {"action": "concurrent-review"}],
        },
    )
    backup_root = tmp_path / "audit-approved-backups"

    with pytest.raises(ValueError, match="review changed after staging"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert (staging_root / first_id / "result.json").is_file()
    assert (staging_root / second_id / "result.json").is_file()
    assert not list(staging_root.glob(".apply-*"))
    assert json.loads((store.job_dir(first_id) / "result.json").read_text()) == first_old
    assert json.loads((store.job_dir(second_id) / "result.json").read_text()) == (second_old)


def test_apply_rejects_swapped_duplicate_reviews(
    tmp_path: Path,
) -> None:
    store, first_id, first_old = setup_job(tmp_path, source_name="First copy.pdf")
    _, second_id, second_old = setup_job(tmp_path, source_name="Second copy.pdf")
    second_review = store.read_review(second_id)
    second_review["revision"] = 4
    second_review["row_overrides"][fixture_row_id("old-row")]["changes"]["description"] = (
        "Second correction"
    )
    (store.job_dir(second_id) / "review.json").write_text(json.dumps(second_review))
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[first_id, second_id],
        extractor=FakeExtractor(
            {
                **first_old,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    staging_root = Path(staged["staging_root"])
    first_staged_review = (staging_root / first_id / "review.json").read_text()
    second_staged_review = (staging_root / second_id / "review.json").read_text()
    (staging_root / first_id / "review.json").write_text(second_staged_review)
    (staging_root / second_id / "review.json").write_text(first_staged_review)
    manifest_path = staging_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for document in manifest["documents"]:
        document["staged_payload_sha256"] = reprocess_module._tree_digest(
            staging_root / document["job_id"]
        )
    manifest_path.write_text(json.dumps(manifest))
    write_passing_visual_audit(staging_root)
    backup_root = tmp_path / "swapped-review-backups"

    with pytest.raises(ValueError, match="staged review migration"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert (staging_root / first_id / "result.json").is_file()
    assert (staging_root / second_id / "result.json").is_file()
    assert not list(staging_root.glob(".apply-*"))
    assert json.loads((store.job_dir(first_id) / "result.json").read_text()) == first_old
    assert json.loads((store.job_dir(second_id) / "result.json").read_text()) == (second_old)


def test_apply_rejects_staged_result_changed_after_visual_audit(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    result_path = staging_root / job_id / "result.json"
    changed_result = json.loads(result_path.read_text())
    changed_result["diagnostics"] = [{"tampered_after_audit": True}]
    result_path.write_text(json.dumps(changed_result))
    backup_root = tmp_path / "changed-stage-backups"

    with pytest.raises(ValueError, match="staged payload digest"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_apply_revalidates_staged_payload_after_acquiring_locks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    result_path = staging_root / job_id / "result.json"
    original_job_lock = JobStore.job_lock
    injected = False

    def racing_job_lock(self: JobStore, locked_job_id: str, *, exclusive: bool) -> Any:
        nonlocal injected
        if exclusive and not injected:
            injected = True
            changed_result = json.loads(result_path.read_text())
            changed_result["diagnostics"] = [{"changed_before_lock": True}]
            result_path.write_text(json.dumps(changed_result))
        return original_job_lock(self, locked_job_id, exclusive=exclusive)

    monkeypatch.setattr(JobStore, "job_lock", racing_job_lock)
    backup_root = tmp_path / "racing-stage-backups"

    with pytest.raises(ValueError, match="staged payload digest"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_apply_atomically_claims_the_staged_payload_before_cutover(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    known_result_path = staging_root / job_id / "result.json"
    original_cutover = reprocess_module._cutover_job
    injected = False

    def racing_cutover(**kwargs: Any) -> Path:
        nonlocal injected
        if known_result_path.is_file():
            injected = True
            changed_result = json.loads(known_result_path.read_text())
            changed_result["diagnostics"] = [{"changed_after_final_check": True}]
            known_result_path.write_text(json.dumps(changed_result))
        return original_cutover(**kwargs)

    monkeypatch.setattr(reprocess_module, "_cutover_job", racing_cutover)

    apply_staged_jobs(root=tmp_path, stage_batch=staging_root)

    assert injected is False
    live_result = json.loads((store.job_dir(job_id) / "result.json").read_text())
    assert live_result.get("diagnostics") == old_result["diagnostics"]


def test_apply_rejects_a_symlinked_staged_payload_root(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    stage_dir = staging_root / job_id
    redirected = tmp_path / "redirected-staged-payload"
    stage_dir.replace(redirected)
    stage_dir.symlink_to(redirected, target_is_directory=True)
    backup_root = tmp_path / "symlink-stage-backups"

    with pytest.raises(ValueError, match="symbolic link|outside staging"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_apply_cleanup_never_follows_a_symlinked_claim_root(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    manifest_bytes = (staging_root / "manifest.json").read_bytes()
    claimed_root = staging_root / f".apply-{hashlib.sha256(manifest_bytes).hexdigest()}"
    orphaned_stage = tmp_path / "orphaned-staged-payload"
    (staging_root / job_id).replace(orphaned_stage)
    claimed_root.symlink_to(store.jobs_root, target_is_directory=True)

    with pytest.raises(ValueError, match="apply-owned staging namespace is unsafe"):
        apply_staged_jobs(root=tmp_path, stage_batch=staging_root)

    assert store.job_dir(job_id).is_dir()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result
    assert claimed_root.is_symlink()
    assert not (staging_root / job_id).exists()


def test_apply_rejects_an_existing_cutover_journal_without_overwriting_it(
    tmp_path: Path,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    journal_path = store.job_dir(job_id) / ".cutover.json"
    prior_journal = {
        "version": "job_cutover_v1",
        "job_id": job_id,
        "stage_dir": str(staging_root / job_id),
        "backup_dir": str(tmp_path / "prior-backup" / job_id),
        "commit_marker": str(tmp_path / "prior-backup" / ".committed.json"),
    }
    journal_path.write_text(json.dumps(prior_journal))

    with pytest.raises(
        JobTransactionError,
        match="job_cutover_recovery_required",
    ):
        apply_staged_jobs(root=tmp_path, stage_batch=staging_root)

    assert json.loads(journal_path.read_text()) == prior_journal
    assert (staging_root / job_id / "result.json").is_file()
    assert not list(staging_root.glob(".apply-*"))
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_apply_preserves_claimed_payload_when_cutover_requires_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)

    def recovery_required_cutover(**kwargs: Any) -> Path:
        prepared = kwargs["prepared"]
        journal_path = store.job_dir(job_id) / ".cutover.json"
        journal_path.write_text(
            json.dumps(
                {
                    "version": "job_cutover_v1",
                    "job_id": job_id,
                    "stage_dir": str(prepared.stage_dir),
                    "backup_dir": str(kwargs["backup_root"] / job_id),
                    "commit_marker": str(kwargs["commit_marker"]),
                }
            )
        )
        raise RuntimeError("cutover recovery required")

    monkeypatch.setattr(
        reprocess_module,
        "_cutover_job",
        recovery_required_cutover,
    )

    with pytest.raises(RuntimeError, match="cutover recovery required"):
        apply_staged_jobs(root=tmp_path, stage_batch=staging_root)

    journal = json.loads((store.job_dir(job_id) / ".cutover.json").read_text())
    claimed_stage = Path(journal["stage_dir"])
    assert claimed_stage.is_dir()
    assert claimed_stage.parent.name.startswith(".apply-")
    assert not (staging_root / job_id).exists()


def test_apply_restores_mixed_preexisting_claims_when_validation_aborts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, first_id, first_old = setup_job(tmp_path)
    _, second_id, second_old = setup_job(tmp_path)
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[first_id, second_id],
        extractor=FakeExtractor(
            {
                **first_old,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    staging_root = Path(staged["staging_root"])
    write_passing_visual_audit(staging_root)
    manifest_bytes = (staging_root / "manifest.json").read_bytes()
    claimed_root = staging_root / f".apply-{hashlib.sha256(manifest_bytes).hexdigest()}"
    claimed_root.mkdir()
    ordered_ids = sorted((first_id, second_id))
    (staging_root / ordered_ids[0]).replace(claimed_root / ordered_ids[0])
    changed_result_path = staging_root / ordered_ids[1] / "result.json"
    changed_result = json.loads(changed_result_path.read_text())
    changed_result["diagnostics"] = [{"invalid_resumed_stage": True}]
    changed_result_path.write_text(json.dumps(changed_result))
    original_job_lock = JobStore.job_lock
    exclusive_locks: list[str] = []

    def tracking_job_lock(self: JobStore, locked_job_id: str, *, exclusive: bool) -> Any:
        if exclusive:
            exclusive_locks.append(locked_job_id)
        return original_job_lock(self, locked_job_id, exclusive=exclusive)

    monkeypatch.setattr(JobStore, "job_lock", tracking_job_lock)

    with pytest.raises(ValueError, match="staged payload digest"):
        apply_staged_jobs(root=tmp_path, stage_batch=staging_root)

    assert exclusive_locks == ordered_ids
    assert (staging_root / first_id).is_dir()
    assert (staging_root / second_id).is_dir()
    assert not claimed_root.exists()
    assert json.loads((store.job_dir(first_id) / "result.json").read_text()) == first_old
    assert json.loads((store.job_dir(second_id) / "result.json").read_text()) == (second_old)


def test_apply_rejects_visual_audit_replayed_for_a_new_extraction(
    tmp_path: Path,
) -> None:
    _, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    first_rows = [row(fixture_row_id("first-new-row"), page_sha)]
    first_stage = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        stage_root=tmp_path / "first-staging",
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": first_rows,
                "source_tables": source_tables(first_rows, page_sha),
            }
        ),
    )
    first_staging_root = Path(first_stage["staging_root"])
    write_passing_visual_audit(first_staging_root)
    old_audit = (first_staging_root / "visual-audit.json").read_text()

    second_rows = [row(fixture_row_id("second-new-row"), page_sha)]
    second_stage = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        stage_root=tmp_path / "second-staging",
        extractor=FakeExtractor(
            {
                **old_result,
                "rows": second_rows,
                "source_tables": source_tables(second_rows, page_sha),
            }
        ),
    )
    second_staging_root = Path(second_stage["staging_root"])
    (second_staging_root / "visual-audit.json").write_text(old_audit)
    backup_root = tmp_path / "replayed-audit-backups"

    with pytest.raises(ValueError, match="predates the sealed stage"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=second_staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()


@pytest.mark.parametrize(
    ("audit_case", "message"),
    [
        ("missing", "visual audit is missing"),
        ("incomplete_pages", "page inventory"),
        ("failed_page", "page 1 did not pass"),
        ("mismatched_sha", "source inventory"),
        ("invalid_sha_type", "source inventory"),
        ("duplicate_source", "source inventory"),
        ("missing_source", "source inventory"),
        ("manifest_page_count_mismatch", "staged page count"),
        ("boolean_page_number", "page inventory"),
        ("blank_reviewer", "reviewer"),
        ("blank_reviewed_at", "reviewed_at"),
        ("non_utc_reviewed_at", "reviewed_at"),
    ],
)
def test_apply_rejects_invalid_visual_audit_before_creating_backups(
    tmp_path: Path,
    audit_case: str,
    message: str,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    if audit_case != "missing":
        if audit_case == "manifest_page_count_mismatch":
            manifest_path = staging_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sources"][0]["page_count"] = 2
            manifest_path.write_text(json.dumps(manifest))
        audit = write_passing_visual_audit(staging_root)
        if audit_case == "incomplete_pages":
            audit["sources"][0]["pages"] = []
        elif audit_case == "failed_page":
            audit["sources"][0]["pages"][0]["status"] = "fail"
        elif audit_case == "mismatched_sha":
            audit["sources"][0]["source_sha256"] = "0" * 64
        elif audit_case == "invalid_sha_type":
            audit["sources"][0]["source_sha256"] = []
        elif audit_case == "duplicate_source":
            audit["sources"].append(deepcopy(audit["sources"][0]))
        elif audit_case == "missing_source":
            audit["sources"] = []
        elif audit_case == "boolean_page_number":
            audit["sources"][0]["pages"][0]["page_number"] = True
        elif audit_case == "blank_reviewer":
            audit["sources"][0]["reviewer"] = " "
        elif audit_case == "blank_reviewed_at":
            audit["sources"][0]["reviewed_at"] = ""
        elif audit_case == "non_utc_reviewed_at":
            audit["sources"][0]["reviewed_at"] = "2026-07-23T10:00:00+10:00"
        (staging_root / "visual-audit.json").write_text(json.dumps(audit))
    backup_root = tmp_path / "rejected-backups"

    with pytest.raises(ValueError, match=message):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result
    assert store.read_review(job_id)["revision"] == 1


@pytest.mark.parametrize("failed_check", sorted(VISUAL_AUDIT_CHECKS))
def test_apply_rejects_each_failed_source_level_visual_check(
    tmp_path: Path,
    failed_check: str,
) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    audit = write_passing_visual_audit(staging_root)
    audit["sources"][0]["checks"][failed_check] = "fail"
    (staging_root / "visual-audit.json").write_text(json.dumps(audit))
    backup_root = tmp_path / "rejected-check-backups"

    with pytest.raises(ValueError, match=f"{failed_check}.*did not pass"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=staging_root,
            backup_root=backup_root,
        )

    assert not backup_root.exists()
    assert json.loads((store.job_dir(job_id) / "result.json").read_text()) == old_result


def test_unmappable_reviewed_row_is_preserved_as_reviewer_row(tmp_path: Path) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    replacement = row(fixture_row_id("unrelated-row"), page_sha)
    replacement["description"] = "Unrelated upgraded row"
    replacement["field_evidence"]["description"][0]["token_ids"] = ["different-description"]
    replacement["field_evidence"]["amount"][0]["token_ids"] = ["different-amount"]

    summary = stage_and_apply(
        root=tmp_path,
        job_ids=[job_id],
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


def test_review_started_during_cutover_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    write_passing_visual_audit(Path(staged["staging_root"]))
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


def test_cutover_failure_restores_the_complete_old_workspace(tmp_path: Path, monkeypatch) -> None:
    store, job_id, old_result = setup_job(tmp_path)
    page_sha = old_result["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]

    class NewArtifactExtractor(FakeExtractor):
        def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
            result = super().extract(source, artifact_root)
            (artifact_root / "new-only.txt").write_text("new")
            return result

    staged = stage_reprocess_jobs(
        root=tmp_path,
        job_ids=[job_id],
        extractor=NewArtifactExtractor(
            {
                **old_result,
                "rows": new_rows,
                "source_tables": source_tables(new_rows, page_sha),
            }
        ),
    )
    write_passing_visual_audit(Path(staged["staging_root"]))
    job_dir = store.job_dir(job_id)
    original_replace = Path.replace
    injected = False

    def failing_replace(path: Path, target: Path) -> Path:
        nonlocal injected
        if (
            not injected
            and path.name == "result.json"
            and path.parent.name == job_id
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
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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


def test_batch_failure_rolls_back_jobs_already_cut_over(tmp_path: Path, monkeypatch) -> None:
    first_store, first_id, first_old = setup_job(tmp_path)
    _, second_id, second_old = setup_job(tmp_path)
    page_sha = first_old["page_assets"][0]["artifact_sha256"]
    new_rows = [row(fixture_row_id("new-row"), page_sha)]
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
    staging_root = Path(staged["staging_root"])
    manifest = json.loads((staging_root / "manifest.json").read_text())
    assert [source["job_ids"] for source in manifest["sources"]] == [sorted((first_id, second_id))]
    write_passing_visual_audit(staging_root)
    ordered_ids = sorted((first_id, second_id))
    failed_id = ordered_ids[1]
    failed_live = first_store.job_dir(failed_id) / "result.json"
    original_replace = Path.replace
    injected = False

    def failing_replace(path: Path, target: Path) -> Path:
        nonlocal injected
        if (
            not injected
            and path.name == "result.json"
            and path.parent.name == failed_id
            and target == failed_live
        ):
            injected = True
            raise OSError("second job failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="second job failed"):
        apply_staged_jobs(
            root=tmp_path,
            stage_batch=Path(staged["staging_root"]),
        )

    assert json.loads((first_store.job_dir(first_id) / "result.json").read_text()) == first_old
    assert json.loads((first_store.job_dir(second_id) / "result.json").read_text()) == second_old
    assert first_store.read_review(first_id)["revision"] == 1
    assert first_store.read_review(second_id)["revision"] == 1
