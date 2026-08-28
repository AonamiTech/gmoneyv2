from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    ExtractionResultV5,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
)
from gmoney.demo.store import JobStore
from gmoney.extraction.offline import ExtractionDraft, OfflineExtractor, PageExtractionUnit
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
        "output_version": "offline_accuracy_spine_v5",
        "document_total_version": "document_total_v3",
        "document_totals_version": "document_totals_v2",
        "document_id": source_sha,
        "source_sha256": source_sha,
        "source_name": source.name,
        "pages": 1,
        "page_assets": [
            {
                "document_sha256": source_sha,
                "page_number": 1,
                "artifact_sha256": page_sha,
                "width": 100,
                "height": 200,
                "dpi": 300,
                "renderer": "test",
                "renderer_version": "1",
                "relative_path": "pages/page-1.png",
            }
        ],
        "token_manifest": [
            {
                "token_id": token_id,
                "page_number": 1,
                "table_ids": ["p1-t1"],
                "text": "Consultation" if token_id == "description-token" else "100.00",
                "polygon": evidence.polygon.model_dump(mode="json"),
                "artifact_sha256": page_sha,
                "artifact_relative_path": "pages/page-1.png",
                "confidence": 0.99,
            }
            for token_id in evidence.token_ids
        ],
        "rows": [row.model_dump(mode="json")],
        "source_tables": [table.model_dump(mode="json")],
        "diagnostics": [
            {
                "diagnostic_id": "p1-page-inventory",
                "diagnostic_kind": "page",
                "page_number": 1,
                "page_classification": "financial",
            },
            {
                "diagnostic_id": "p1-t1-inventory",
                "diagnostic_kind": "table",
                "page_number": 1,
                "table_id": "p1-t1",
                "source_table_id": "p1-t1-s1",
            },
        ],
        "document_total": None,
        "document_totals": [],
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
    return source, artifact_root, result


def test_validation_returns_a_complete_structured_report(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "passed"
    assert [issue.code for issue in report.issues] == ["document_total_unavailable"]
    assert report.issues[0].severity == "warning"

    del result["output_version"]
    result["diagnostics"] = [
        {
            "diagnostic_id": "p1-page-inventory",
            "diagnostic_kind": "page",
            "page_number": 1,
            "financial_form_suspected": True,
        },
        {
            "diagnostic_id": "p1-t1-inventory",
            "diagnostic_kind": "table",
            "page_number": 1,
            "table_id": "p1-t1",
            "source_table_id": "p1-t1-s1",
        },
    ]
    failed = validate_extraction_result(source, result, artifact_root)

    assert {issue.code for issue in failed.issues} == {
        "extraction_result_contract_invalid",
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
    rows = result["rows"]
    tables = result["source_tables"]
    assert isinstance(rows, list) and isinstance(tables, list)
    table = tables[0]
    assert isinstance(table, dict) and isinstance(table["rows"], list)
    canonical_ids = [str(rows[0]["id"])]
    source_ids = [str(table["rows"][0]["id"])]
    for index, suffix in enumerate(("b", "c"), start=1):
        duplicate_row = json.loads(json.dumps(rows[0]))
        duplicate_row.update(id=str(uuid4()), row_order=index)
        duplicate_source_row = json.loads(json.dumps(table["rows"][0]))
        duplicate_source_row.update(
            id=f"p1-t1-s1-r{suffix}",
            order=index,
            canonical_row_id=duplicate_row["id"],
        )
        rows.append(duplicate_row)
        table["rows"].append(duplicate_source_row)
        canonical_ids.append(str(duplicate_row["id"]))
        source_ids.append(str(duplicate_source_row["id"]))
    result["receipt_duplicate_pairs"] = [
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "canonical_row_ids": canonical_ids[:2],
            "source_row_ids": source_ids[:2],
            "match_basis": "exact_reference",
        },
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "canonical_row_ids": [canonical_ids[0], canonical_ids[2]],
            "source_row_ids": [source_ids[0], source_ids[2]],
            "match_basis": "dated_grounded_charge",
        },
    ]

    report = validate_extraction_result(source, result, artifact_root)
    duplicates = [
        issue for issue in report.issues if issue.code == "possible_duplicate_supporting_charge"
    ]

    assert len(duplicates) == 2
    assert duplicates[0].id != duplicates[1].id
    assert duplicates[0].related_canonical_row_ids == tuple(canonical_ids[:2])

    store = JobStore(tmp_path / "jobs")
    state = store.create("duplicate-receipts.pdf")
    store.update(state["id"], status="queued")
    assert store.claim_queued(state["id"]) is not None
    result["semantic_validation"] = report.model_dump(mode="json")
    assert store.publish_processing_outcome(
        state["id"],
        result,
        status="needs_review",
        validation_status="needs_review",
        validation_issue_count=2,
        validation_issue_codes=["possible_duplicate_supporting_charge"],
    )
    published = store.read_result(state["id"])
    assert len(published["receipt_duplicate_pairs"]) == 2
    json.dumps(published)


def test_empty_extraction_requires_complete_grounded_page_classification(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    result["rows"] = []
    result["source_tables"] = []
    result["token_manifest"] = []
    result["diagnostics"] = [
        {
            "diagnostic_id": "p1-page",
            "diagnostic_kind": "page",
            "page_number": 1,
            "page_classification": "unclassified",
        }
    ]

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert "empty_extraction_unclassified" in {issue.code for issue in report.issues}

    result["diagnostics"][0].update(
        page_classification="blank",
        demonstrably_blank=True,
    )
    passed = validate_extraction_result(source, result, artifact_root)
    assert passed.status == "passed"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rows", [None, {"id": 42}]),
        ("source_tables", [None, {"id": 42}]),
    ),
)
def test_malformed_nested_payloads_return_fatal_reports(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    result[field] = value

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    assert "extraction_result_contract_invalid" in {
        issue.code for issue in report.issues
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rows", ["bad"]),
        ("source_tables", ["bad"]),
    ),
)
def test_non_object_nested_entries_fail_closed_without_throwing(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    result[field] = value

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    assert {issue.code for issue in report.issues} == {
        "extraction_result_contract_invalid"
    }


def test_primary_total_and_page_assets_are_decoded_by_the_envelope(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    primary = {
        "total_version": "document_total_v3",
        "amount_raw": "100.00",
        "amount": "100.00",
        "label": "Grand Total",
        "scope": "document",
        "page_number": "not-a-page",
        "evidence": result["rows"][0]["evidence"][0],
        "confidence": 1,
        "context_id": "final-1",
        "context_kind": "document_final",
    }
    result["document_total"] = primary
    result["document_totals"] = [primary]
    del result["page_assets"][0]["width"]
    del result["page_assets"][0]["height"]

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    fields = {issue.field for issue in report.issues}
    assert "document_total.page_number" in fields
    assert "page_assets.0.width" in fields
    assert "page_assets.0.height" in fields


@pytest.mark.parametrize(
    ("token_id", "replacement"),
    (
        ("description-token", "UNRELATED"),
        ("amount-token", "999.99"),
    ),
)
def test_evidence_token_text_must_support_published_values(
    tmp_path: Path,
    token_id: str,
    replacement: str,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    next(
        token for token in result["token_manifest"] if token["token_id"] == token_id
    )["text"] = replacement

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert {
        "canonical_evidence_value_mismatch",
        "source_cell_evidence_value_mismatch",
    }.intersection(issue.code for issue in report.issues)


def test_table_diagnostic_inventory_is_exact(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    duplicate = dict(result["diagnostics"][1])
    duplicate["diagnostic_id"] = "duplicate-table-diagnostic"
    orphan = dict(result["diagnostics"][1])
    orphan.update(
        diagnostic_id="orphan-table-diagnostic",
        table_id="p1-t9",
        source_table_id="p1-t9-s1",
    )
    result["diagnostics"].extend((duplicate, orphan))

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    assert {
        "table_diagnostic_inventory_incomplete",
        "orphan_table_diagnostic",
    }.issubset(issue.code for issue in report.issues)


def test_unlocatable_table_recovery_keeps_exact_diagnostic_inventory(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    table_digest = "a" * 64
    result["recovery"] = {
        "attempted": True,
        "targets": [
            {
                "page_number": 1,
                "table_id": "p1-t1",
                "selected": "baseline",
                "status": "recovery_target_not_located",
                "baseline_unit_sha256": table_digest,
                "candidate_unit_sha256": None,
                "selected_unit_sha256": table_digest,
                "removed_issue_ids": [],
            }
        ],
        "untargeted_units_sha256": "b" * 64,
    }

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert "recovery_target_not_located" in {
        issue.code for issue in report.issues
    }
    assert "page_diagnostic_inventory_incomplete" not in {
        issue.code for issue in report.issues
    }


def test_recovery_token_retains_crop_identity_and_page_transform(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    crop_path = artifact_root / "crops" / "recovered.png"
    crop_path.parent.mkdir()
    crop_path.write_bytes(b"recovery crop")
    crop_sha = _sha(crop_path.read_bytes())
    token = result["token_manifest"][0]
    token.update(
        source_artifact_sha256=crop_sha,
        source_artifact_relative_path="crops/recovered.png",
        source_polygon=token["polygon"],
        source_width=100,
        source_height=200,
        source_to_page_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "passed"
    token["source_artifact_sha256"] = "f" * 64
    failed = validate_extraction_result(source, result, artifact_root)
    assert "recovery_token_provenance_invalid" in {
        issue.code for issue in failed.issues
    }


def _draft_from_result(result: dict[str, object]) -> ExtractionDraft:
    envelope = ExtractionResultV5.model_validate(result)
    unit = PageExtractionUnit(
        page_asset=envelope.page_assets[0],
        ocr_tokens=(),
        token_manifest=envelope.token_manifest,
        source_tables=envelope.source_tables,
        canonical_rows=envelope.rows,
        diagnostics=tuple(
            item.model_dump(mode="json") for item in envelope.diagnostics
        ),
        total_candidates=envelope.document_totals,
    )
    return ExtractionDraft(
        publication_result=result,
        page_units=(unit,),
        provider_usage=envelope.provider_usage.model_dump(mode="json"),
        hospital=envelope.hospital,
        hospital_id=envelope.hospital_id,
    )


def _as_recovery_candidate(result: dict[str, object]) -> None:
    result["provider_usage"] = {
        "initial": {"gemini_calls": 0, "gemini_measured_cost_usd": "0"},
        "recovery": {
            "gemini_calls": 0,
            "gemini_measured_cost_usd": "0",
            "gemini_allowed": False,
        },
        "aggregate": {"gemini_calls": 0, "gemini_measured_cost_usd": "0"},
    }
    result["recovery"] = {
        "attempted": True,
        "targets": [
            {
                "page_number": 1,
                "table_id": "p1-t1",
                "selected": "candidate",
                "status": "recovered",
                "baseline_unit_sha256": "a" * 64,
                "candidate_unit_sha256": "b" * 64,
                "selected_unit_sha256": "b" * 64,
                "removed_issue_ids": [],
            }
        ],
        "untargeted_units_sha256": "c" * 64,
    }


def test_recover_draft_requires_semantic_improvement_and_preserves_charges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, artifact_root, baseline_result = _fixture(tmp_path)
    baseline_result["token_manifest"][0]["text"] = "UNRELATED"
    baseline = _draft_from_result(baseline_result)
    candidate_result = json.loads(json.dumps(baseline_result))
    candidate_result["token_manifest"][0]["text"] = "Consultation"
    _as_recovery_candidate(candidate_result)
    candidate = _draft_from_result(candidate_result)
    extractor = object.__new__(OfflineExtractor)

    def fake_extract(*args: object, **kwargs: object) -> dict[str, object]:
        sink = kwargs["_draft_sink"]
        sink.update(
            page_units=candidate.page_units,
            provider_usage=candidate.provider_usage,
            hospital=candidate.hospital,
            hospital_id=candidate.hospital_id,
        )
        return candidate.publication_result

    monkeypatch.setattr(extractor, "extract", fake_extract)
    recovered = extractor.recover_draft(
        source,
        artifact_root,
        baseline,
        ((1, "p1-t1"),),
    )

    assert recovered.publication_result["recovery"]["targets"][0]["selected"] == "candidate"
    assert recovered.page_units[0].token_manifest[0].text == "Consultation"

    unsafe_result = json.loads(json.dumps(candidate_result))
    unsafe_result["rows"][0]["net_amount"] = "999.99"
    unsafe_result["rows"][0]["net_amount_raw"] = "999.99"
    unsafe_result["source_tables"][0]["rows"][0]["cells"][1]["raw_value"] = "999.99"
    unsafe_result["token_manifest"][1]["text"] = "999.99"
    unsafe = _draft_from_result(unsafe_result)

    def unsafe_extract(*args: object, **kwargs: object) -> dict[str, object]:
        sink = kwargs["_draft_sink"]
        sink.update(
            page_units=unsafe.page_units,
            provider_usage=unsafe.provider_usage,
            hospital=unsafe.hospital,
            hospital_id=unsafe.hospital_id,
        )
        return unsafe.publication_result

    monkeypatch.setattr(extractor, "extract", unsafe_extract)
    declined = extractor.recover_draft(
        source,
        artifact_root,
        baseline,
        ((1, "p1-t1"),),
    )

    target = declined.publication_result["recovery"]["targets"][0]
    assert target["selected"] == "baseline"
    assert target["status"] == "recovery_no_safe_improvement"


def test_unique_document_final_context_requires_primary_total(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    evidence = result["rows"][0]["evidence"][0]
    result["document_totals"] = [
        {
            "amount_raw": "100.00",
            "amount": "100.00",
            "label": "Net Bill Amount",
            "scope": "document",
            "page_number": 1,
            "evidence": evidence,
            "confidence": 0.99,
            "context_id": "p1:summary:document_final:o1",
            "context_kind": "document_final",
        }
    ]
    result["document_total"] = None

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert "primary_total_missing_for_unique_context" in {
        issue.code for issue in report.issues
    }


def test_recovery_provider_usage_is_separate_and_gemini_free(tmp_path: Path) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    result["provider_usage"] = {
        "initial": {"gemini_calls": 1, "gemini_measured_cost_usd": "0.02"},
        "recovery": {"gemini_calls": 1, "gemini_measured_cost_usd": "0.01"},
        "aggregate": {"gemini_calls": 2, "gemini_measured_cost_usd": "0.03"},
    }

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    assert "recovery_gemini_invoked" in {issue.code for issue in report.issues}
