from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import fitz

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
)
from gmoney.extraction.validation import validate_extraction_result


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page(width=100, height=200)
    document.save(source)
    document.close()
    source_sha = _sha(source.read_bytes())

    artifact_root = tmp_path / "artifacts"
    page_path = artifact_root / "pages" / "page-1.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"grounded page artifact")
    page_sha = _sha(page_path.read_bytes())
    evidence = EvidenceRef(
        page_number=1,
        table_id="p1-t1",
        polygon=Polygon(
            points=(
                Point(x=1, y=1),
                Point(x=20, y=1),
                Point(x=20, y=10),
                Point(x=1, y=10),
            )
        ),
        artifact_sha256=page_sha,
        token_ids=("description-token", "amount-token"),
    )
    row = CanonicalRow(
        id=uuid4(),
        contract_version="canonical_row_v2",
        document_id=source_sha,
        page_number=1,
        table_id="p1-t1",
        row_order=0,
        description="Consultation",
        net_amount_raw="100.00",
        net_amount="100.00",
        evidence=(evidence,),
        field_evidence={"description": (evidence,), "amount": (evidence,)},
        source_routes=("ocr_spatial_graph",),
    )
    columns = (
        SourceColumn(
            id="description",
            label="Description",
            order=0,
            canonical_field="description",
            evidence=(evidence,),
        ),
        SourceColumn(
            id="amount",
            label="Amount",
            order=1,
            canonical_field="net_amount",
            evidence=(evidence,),
        ),
    )
    source_row = SourceRow(
        id="p1-t1-s1-r1",
        order=0,
        canonical_row_id=str(row.id),
        cells=(
            SourceCell(
                column_id="description",
                raw_value="Consultation",
                evidence=(evidence,),
            ),
            SourceCell(
                column_id="amount",
                raw_value="100.00",
                evidence=(evidence,),
            ),
        ),
    )
    table = SourceTable(
        id="p1-t1-s1",
        page_number=1,
        table_id="p1-t1",
        columns=columns,
        rows=(source_row,),
    )
    result: dict[str, object] = {
        "output_version": "offline_accuracy_spine_v3",
        "document_id": source_sha,
        "source_sha256": source_sha,
        "source_name": source.name,
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
        "rows": [row.model_dump(mode="json")],
        "source_tables": [table.model_dump(mode="json")],
        "diagnostics": [],
        "document_total": None,
        "document_totals": [],
    }
    return source, artifact_root, result


def test_validation_returns_a_complete_structured_report(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "passed"
    assert [issue.code for issue in report.issues] == ["document_total_unavailable"]
    assert report.issues[0].severity == "warning"

    del result["output_version"]
    result["diagnostics"] = [
        {"page_number": 1, "table_id": "p1-t1", "financial_form_suspected": True}
    ]
    failed = validate_extraction_result(source, result, artifact_root)

    assert {issue.code for issue in failed.issues} == {
        "unsupported_output_contract",
        "financial_form_unresolved",
        "document_total_unavailable",
    }
    assert all(issue.id and issue.severity for issue in failed.issues)
    assert failed.status == "failed"


def test_linked_unmapped_financial_lane_is_not_invisible(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    table = SourceTable.model_validate(result["source_tables"][0])
    evidence = table.rows[0].cells[0].evidence
    extra_column = SourceColumn(
        id="copay",
        label="Co-pay amount",
        order=2,
        validation_flags=("synthetic_header", "inferred_financial_lane"),
    )
    extra_cell = SourceCell(
        column_id="copay",
        raw_value="25.00",
        evidence=evidence,
    )
    table = table.model_copy(
        update={
            "columns": (*table.columns, extra_column),
            "rows": (
                table.rows[0].model_copy(update={"cells": (*table.rows[0].cells, extra_cell)}),
            ),
        }
    )
    result["source_tables"] = [table.model_dump(mode="json")]

    report = validate_extraction_result(source, result, artifact_root)

    issue = next(item for item in report.issues if item.code == "unmapped_financial_source_cell")
    assert issue.source_row_id == "p1-t1-s1-r1"
    assert issue.canonical_row_id == table.rows[0].canonical_row_id
    assert issue.field == "copay"


def test_informational_row_cannot_publish_money(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    published_payload = dict(result["rows"][0])
    published_payload["role"] = "informational"
    published = CanonicalRow.model_validate(published_payload)
    result["rows"] = [published.model_dump(mode="json")]

    report = validate_extraction_result(source, result, artifact_root)

    issue = next(
        item for item in report.issues if item.code == "informational_row_has_money"
    )
    assert issue.canonical_row_id == str(published.id)
    assert issue.field == "net_amount"


def test_receipt_duplicate_pairs_remain_individual_structured_issues(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    result["receipt_duplicate_pairs"] = [
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "canonical_row_ids": ["row-a", "row-b"],
            "source_row_ids": ["source-a", "source-b"],
        },
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "canonical_row_ids": ["row-a", "row-c"],
            "source_row_ids": ["source-a", "source-c"],
        },
    ]

    report = validate_extraction_result(source, result, artifact_root)
    duplicates = [
        issue for issue in report.issues if issue.code == "possible_duplicate_supporting_charge"
    ]

    assert len(duplicates) == 2
    assert duplicates[0].id != duplicates[1].id
    assert duplicates[0].related_canonical_row_ids == ("row-a", "row-b")
