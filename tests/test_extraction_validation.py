from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
from gmoney.extraction.document_total import DocumentTotalCandidate
from gmoney.extraction.offline import (
    ExtractionDraft,
    OfflineExtractor,
    PageExtractionUnit,
    TableExtractionUnit,
    _financial_inventory_matches,
    _merge_targeted_page_units,
    _recovery_preserves_grounded_charges,
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
    description_evidence = evidence.model_copy(update={"token_ids": ("description-fragment",)})
    amount_evidence = evidence.model_copy(update={"token_ids": ("amount-fragment",)})
    row_anchor = "row-" + "a" * 24
    table_anchor = "table-" + "b" * 24
    row = CanonicalRow(
        id=uuid4(),
        contract_version="canonical_row_v2",
        document_id=source_sha,
        page_number=1,
        table_id="p1-t1",
        row_order=0,
        row_anchor=row_anchor,
        description="Consultation",
        net_amount_raw="100.00",
        net_amount="100.00",
        evidence=(evidence,),
        field_evidence={
            "description": (description_evidence,),
            "amount": (amount_evidence,),
        },
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
        row_anchor=row_anchor,
        order=0,
        canonical_row_id=str(row.id),
        cells=(
            SourceCell(
                column_id="description",
                raw_value="Consultation",
                evidence=(description_evidence,),
            ),
            SourceCell(
                column_id="amount",
                raw_value="100.00",
                evidence=(amount_evidence,),
            ),
        ),
    )
    table = SourceTable(
        id="p1-t1-s1",
        page_number=1,
        table_id="p1-t1",
        table_anchor=table_anchor,
        columns=columns,
        rows=(source_row,),
    )
    result: dict[str, object] = {
        "output_version": "offline_accuracy_spine_v5",
        "contract_revision": 2,
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
        ]
        + [
            {
                "token_id": fragment_id,
                "page_number": 1,
                "table_ids": ["p1-t1"],
                "text": text,
                "polygon": evidence.polygon.model_dump(mode="json"),
                "artifact_sha256": page_sha,
                "artifact_relative_path": "pages/page-1.png",
                "confidence": 0.99,
                "parent_token_id": parent_id,
                "character_start": 0,
                "character_end": len(text),
                "fragment_role": role,
            }
            for fragment_id, parent_id, text, role in (
                (
                    "description-fragment",
                    "description-token",
                    "Consultation",
                    "description",
                ),
                ("amount-fragment", "amount-token", "100.00", "amount"),
            )
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
        "raw_total_candidates": [],
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


def test_validation_hashes_shared_artifacts_once_per_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    calls: list[Path] = []

    def tracked_sha256_file(path: Path) -> str:
        resolved = path.resolve()
        calls.append(resolved)
        return _sha(resolved.read_bytes())

    monkeypatch.setattr(
        "gmoney.extraction.validation.sha256_file",
        tracked_sha256_file,
    )

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "passed"
    assert calls.count(source.resolve()) == 1
    assert calls.count((artifact_root / "pages" / "page-1.png").resolve()) == 1


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

    issue = next(item for item in report.issues if item.code == "informational_row_has_money")
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
    directory = store.job_dir(state["id"])
    (directory / "source.pdf").write_bytes(source.read_bytes())
    (directory / "artifacts").mkdir()
    (directory / "artifacts" / "pages").mkdir()
    (directory / "artifacts" / "pages" / "page-1.png").write_bytes(
        (artifact_root / "pages" / "page-1.png").read_bytes()
    )
    store.update(state["id"], status="queued")
    assert store.claim_queued(state["id"]) is not None
    assert store.publish_processing_outcome(state["id"], result)
    certified_state = store.read(state["id"])
    assert certified_state["status"] == "needs_review"
    assert certified_state["_certification_valid"] is True
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
    assert "extraction_result_contract_invalid" in {issue.code for issue in report.issues}


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
    assert {issue.code for issue in report.issues} == {"extraction_result_contract_invalid"}


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
        ("description-token", "Consultation unrelated text"),
        ("amount-token", "999.99"),
        ("amount-token", "100.00 999.99"),
    ),
)
def test_evidence_token_text_must_support_published_values(
    tmp_path: Path,
    token_id: str,
    replacement: str,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    fragment_id = token_id.replace("-token", "-fragment")
    parent = next(token for token in result["token_manifest"] if token["token_id"] == token_id)
    fragment = next(token for token in result["token_manifest"] if token["token_id"] == fragment_id)
    parent["text"] = replacement
    fragment["text"] = replacement
    fragment["character_end"] = len(replacement)

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
    assert "recovery_target_not_located" in {issue.code for issue in report.issues}
    assert "page_diagnostic_inventory_incomplete" not in {issue.code for issue in report.issues}


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
    assert "recovery_token_provenance_invalid" in {issue.code for issue in failed.issues}


def test_revision_four_preprocessing_raw_page_must_match_page_asset(
    tmp_path: Path,
) -> None:
    _source, _artifact_root, result = _fixture(tmp_path)
    asset = result["page_assets"][0]
    page_sha = asset["artifact_sha256"]
    quality = {
        "page_number": 1,
        "artifact_sha256": page_sha,
        "width": 100,
        "height": 200,
        "dpi": 300,
        "mean_luminance": 200,
        "contrast_stddev": 40,
        "laplacian_variance": 100,
        "edge_density": 0.05,
        "estimated_skew_degrees": 0,
    }
    transform = {
        "page_number": 1,
        "source_width": 100,
        "source_height": 200,
        "derived_width": 100,
        "derived_height": 200,
        "forward_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "inverse_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    }
    result["contract_revision"] = 4
    result["page_preprocessing"] = [
        {
            "page_number": 1,
            "raw_artifact_sha256": page_sha,
            "raw_artifact_relative_path": "pages/not-the-page-asset.png",
            "raw_quality": quality,
            "candidates": [
                {
                    "variant": "raw",
                    "artifact_sha256": page_sha,
                    "artifact_relative_path": "pages/not-the-page-asset.png",
                    "width": 100,
                    "height": 200,
                    "dpi": 300,
                    "transform": transform,
                    "quality": quality,
                    "selected": True,
                }
            ],
            "selected_variant": "raw",
        }
    ]

    with pytest.raises(ValueError, match="does not match page asset"):
        ExtractionResultV5.model_validate(result)


def _draft_from_result(result: dict[str, object]) -> ExtractionDraft:
    envelope = ExtractionResultV5.model_validate(result)
    table_units = tuple(
        TableExtractionUnit(
            page_number=table.page_number,
            table_id=table.table_id,
            source_table=table,
            row_candidates=tuple(row for row in envelope.rows if row.table_id == table.table_id),
            crop_relative_path=None,
            crop_box=None,
            diagnostics=tuple(
                item.model_dump(mode="json")
                for item in envelope.diagnostics
                if item.table_id == table.table_id
            ),
            normalized_fragments=tuple(
                token
                for token in envelope.token_manifest
                if token.fragment_role and table.table_id in token.table_ids
            ),
            recovery_tokens=(),
            raw_total_candidates=(),
            provider_usage=envelope.provider_usage.model_dump(mode="json"),
        )
        for table in envelope.source_tables
    )
    unit = PageExtractionUnit(
        page_asset=envelope.page_assets[0],
        ocr_tokens=(),
        token_manifest=envelope.token_manifest,
        table_units=table_units,
        unassigned_row_candidates=(),
        diagnostics=tuple(item.model_dump(mode="json") for item in envelope.diagnostics),
        total_candidates=tuple(
            DocumentTotalCandidate(
                total=item.total,
                label_priority=item.label_priority,
                vertical_position=item.vertical_position,
                local_context=item.local_context,
            )
            for item in envelope.raw_total_candidates
        ),
        provider_usage=envelope.provider_usage.model_dump(mode="json"),
    )
    return ExtractionDraft(
        document_id=envelope.document_id,
        source_sha256=envelope.source_sha256,
        source_name=envelope.source_name,
        page_units=(unit,),
        provider_usage=envelope.provider_usage.model_dump(mode="json"),
        hospital=envelope.hospital,
        hospital_id=envelope.hospital_id,
        alias_registry_revision=envelope.alias_registry_revision,
        profile_registry_revision=envelope.profile_registry_revision,
        applied_alias_ids=envelope.applied_alias_ids,
        suppressed_repeated_source_tables=tuple(
            (item.page_number, item.table_id) for item in envelope.suppressed_repeated_source_tables
        ),
        recovery_metadata=envelope.recovery.model_dump(mode="json"),
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


def _copy_recovery_artifacts(artifact_root: Path) -> None:
    recovery_root = artifact_root / "recovery"
    for child in tuple(artifact_root.iterdir()):
        if child == recovery_root:
            continue
        target = recovery_root / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif child.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def test_recover_draft_requires_semantic_improvement_and_preserves_charges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, artifact_root, baseline_result = _fixture(tmp_path)
    baseline_result["token_manifest"][0]["text"] = "UNRELATED"
    baseline_result["token_manifest"][2]["text"] = "UNRELATED"
    baseline_result["token_manifest"][2]["character_end"] = len("UNRELATED")
    baseline = _draft_from_result(baseline_result)
    candidate_result = json.loads(json.dumps(baseline_result))
    candidate_result["token_manifest"][0]["text"] = "Consultation"
    candidate_result["token_manifest"][2]["text"] = "Consultation"
    candidate_result["token_manifest"][2]["character_end"] = len("Consultation")
    _as_recovery_candidate(candidate_result)
    candidate = _draft_from_result(candidate_result)
    extractor = object.__new__(OfflineExtractor)
    _copy_recovery_artifacts(artifact_root)

    def fake_extract(*args: object, **kwargs: object) -> dict[str, object]:
        sink = kwargs["_draft_sink"]
        sink.update(
            document_id=candidate.document_id,
            source_sha256=candidate.source_sha256,
            source_name=candidate.source_name,
            page_units=candidate.page_units,
            provider_usage=candidate.provider_usage,
            hospital=candidate.hospital,
            hospital_id=candidate.hospital_id,
            alias_registry_revision=candidate.alias_registry_revision,
            profile_registry_revision=candidate.profile_registry_revision,
            applied_alias_ids=candidate.applied_alias_ids,
            suppressed_repeated_source_tables=(candidate.suppressed_repeated_source_tables),
            recovery_metadata=candidate.recovery_metadata,
        )
        return candidate.result

    monkeypatch.setattr(extractor, "extract", fake_extract)
    recovered = extractor.recover_draft(
        source,
        artifact_root,
        baseline,
        ((1, "p1-t1"),),
    )

    assert recovered.result["recovery"]["targets"][0]["selected"] == "candidate"
    assert recovered.page_units[0].token_manifest[0].text == "Consultation"
    assert recovered.page_units[0].page_asset is baseline.page_units[0].page_asset
    assert recovered.page_units[0].token_manifest[0].artifact_relative_path.startswith("recovery/")
    assert (
        artifact_root / recovered.page_units[0].token_manifest[0].artifact_relative_path
    ).is_file()

    unsafe_result = json.loads(json.dumps(candidate_result))
    unsafe_result["rows"][0]["net_amount"] = "999.99"
    unsafe_result["rows"][0]["net_amount_raw"] = "999.99"
    unsafe_result["source_tables"][0]["rows"][0]["cells"][1]["raw_value"] = "999.99"
    unsafe_result["token_manifest"][1]["text"] = "999.99"
    unsafe = _draft_from_result(unsafe_result)

    def unsafe_extract(*args: object, **kwargs: object) -> dict[str, object]:
        sink = kwargs["_draft_sink"]
        sink.update(
            document_id=unsafe.document_id,
            source_sha256=unsafe.source_sha256,
            source_name=unsafe.source_name,
            page_units=unsafe.page_units,
            provider_usage=unsafe.provider_usage,
            hospital=unsafe.hospital,
            hospital_id=unsafe.hospital_id,
            alias_registry_revision=unsafe.alias_registry_revision,
            profile_registry_revision=unsafe.profile_registry_revision,
            applied_alias_ids=unsafe.applied_alias_ids,
            suppressed_repeated_source_tables=(unsafe.suppressed_repeated_source_tables),
            recovery_metadata=unsafe.recovery_metadata,
        )
        return unsafe.result

    monkeypatch.setattr(extractor, "extract", unsafe_extract)
    declined = extractor.recover_draft(
        source,
        artifact_root,
        baseline,
        ((1, "p1-t1"),),
    )

    target = declined.result["recovery"]["targets"][0]
    assert target["selected"] == "baseline"
    assert target["status"] == "recovery_no_safe_improvement"
    assert declined.page_units[0].page_asset is baseline.page_units[0].page_asset


def test_recovery_cannot_move_a_blocker_to_another_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    second = json.loads(json.dumps(result["rows"][0]))
    second_id = str(uuid4())
    second["id"] = second_id
    second["row_order"] = 1
    second["row_anchor"] = "row-" + "c" * 24
    replacements = {
        "description-token": "description-token-2",
        "description-fragment": "description-fragment-2",
        "amount-token": "amount-token-2",
        "amount-fragment": "amount-fragment-2",
    }

    def replace_token_ids(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("token_ids"), list):
                value["token_ids"] = [
                    replacements.get(token_id, token_id) for token_id in value["token_ids"]
                ]
            for nested in value.values():
                replace_token_ids(nested)
        elif isinstance(value, list):
            for nested in value:
                replace_token_ids(nested)

    replace_token_ids(second)
    second_source = json.loads(json.dumps(result["source_tables"][0]["rows"][0]))
    second_source["id"] = "p1-t1-s1-r2"
    second_source["order"] = 1
    second_source["row_anchor"] = second["row_anchor"]
    second_source["canonical_row_id"] = second_id
    replace_token_ids(second_source)
    result["rows"].append(second)
    result["source_tables"][0]["rows"].append(second_source)
    for token in list(result["token_manifest"]):
        cloned = json.loads(json.dumps(token))
        cloned["token_id"] = replacements.get(cloned["token_id"], cloned["token_id"])
        if cloned.get("parent_token_id"):
            cloned["parent_token_id"] = replacements.get(
                cloned["parent_token_id"], cloned["parent_token_id"]
            )
        result["token_manifest"].append(cloned)

    baseline_result = json.loads(json.dumps(result))
    for token in baseline_result["token_manifest"]:
        if token["token_id"] in {"description-token", "description-fragment"}:
            token["text"] = "UNRELATED"
            if token["token_id"] == "description-fragment":
                token["character_end"] = len("UNRELATED")
    baseline = _draft_from_result(baseline_result)

    candidate_result = json.loads(json.dumps(result))
    for token in candidate_result["token_manifest"]:
        if token["token_id"] in {"description-token-2", "description-fragment-2"}:
            token["text"] = "UNRELATED"
            if token["token_id"] == "description-fragment-2":
                token["character_end"] = len("UNRELATED")
    _as_recovery_candidate(candidate_result)
    candidate = _draft_from_result(candidate_result)
    extractor = object.__new__(OfflineExtractor)
    _copy_recovery_artifacts(artifact_root)

    def fake_extract(*args: object, **kwargs: object) -> dict[str, object]:
        sink = kwargs["_draft_sink"]
        sink.update(
            document_id=candidate.document_id,
            source_sha256=candidate.source_sha256,
            source_name=candidate.source_name,
            page_units=candidate.page_units,
            provider_usage=candidate.provider_usage,
            hospital=candidate.hospital,
            hospital_id=candidate.hospital_id,
            alias_registry_revision=candidate.alias_registry_revision,
            profile_registry_revision=candidate.profile_registry_revision,
            applied_alias_ids=candidate.applied_alias_ids,
            suppressed_repeated_source_tables=candidate.suppressed_repeated_source_tables,
            recovery_metadata=candidate.recovery_metadata,
        )
        return candidate.result

    monkeypatch.setattr(extractor, "extract", fake_extract)
    recovered = extractor.recover_draft(
        source,
        artifact_root,
        baseline,
        ((1, "p1-t1"),),
    )

    target = recovered.result["recovery"]["targets"][0]
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
    result["raw_total_candidates"] = [
        {
            "candidate_id": "total-candidate-1",
            "total": result["document_totals"][0],
            "label_priority": 5,
            "vertical_position": 190.0,
            "local_context": "Net Bill Amount",
            "page_number": 1,
            "table_id": "p1-t1",
            "table_anchor": "table-" + "b" * 24,
            "region_kind": "table",
            "summary_block_ordinal": 1,
            "context_evidence": [evidence],
        }
    ]
    result["document_total"] = None

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert "primary_total_missing_for_unique_context" in {issue.code for issue in report.issues}


def test_unique_raw_document_final_context_cannot_be_omitted(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    evidence = result["rows"][0]["evidence"][0]
    raw_total = {
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
    result["raw_total_candidates"] = [
        {
            "candidate_id": "total-candidate-1",
            "total": raw_total,
            "label_priority": 5,
            "vertical_position": 190.0,
            "local_context": "Net Bill Amount",
            "page_number": 1,
            "table_id": "p1-t1",
            "table_anchor": "table-" + "b" * 24,
            "region_kind": "table",
            "summary_block_ordinal": 1,
            "context_evidence": [evidence],
        }
    ]
    result["document_totals"] = []
    result["document_total"] = None

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "needs_review"
    assert "raw_document_final_total_omitted" in {issue.code for issue in report.issues}


def test_raw_total_candidate_and_context_evidence_are_validated(
    tmp_path: Path,
) -> None:
    source, artifact_root, result = _fixture(tmp_path)
    evidence = json.loads(json.dumps(result["rows"][0]["evidence"][0]))
    evidence["token_ids"] = ["nonexistent-total-token"]
    total = {
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
    result["raw_total_candidates"] = [
        {
            "candidate_id": "corrupt-total-candidate",
            "total": total,
            "label_priority": 5,
            "vertical_position": 190.0,
            "local_context": "Net Bill Amount",
            "page_number": 1,
            "table_id": "p1-t1",
            "table_anchor": "table-" + "b" * 24,
            "region_kind": "table",
            "summary_block_ordinal": 1,
            "context_evidence": [evidence],
        }
    ]

    report = validate_extraction_result(source, result, artifact_root)

    assert report.status == "failed"
    matching = [issue for issue in report.issues if issue.code == "evidence_artifact_mismatch"]
    assert {issue.field for issue in matching} == {
        "raw_total_candidates.total.evidence",
        "raw_total_candidates.context_evidence",
    }


def test_recovery_cannot_delete_an_unlinked_financial_printed_row(
    tmp_path: Path,
) -> None:
    _source, _artifact_root, result = _fixture(tmp_path)
    unlinked = json.loads(json.dumps(result["source_tables"][0]["rows"][0]))
    unlinked.update(
        id="p1-t1-s1-r-unlinked",
        order=1,
        canonical_row_id=None,
        row_anchor="row-" + "d" * 24,
    )
    result["source_tables"][0]["rows"].append(unlinked)
    baseline = _draft_from_result(result)
    candidate_result = json.loads(json.dumps(result))
    candidate_result["source_tables"][0]["rows"] = candidate_result["source_tables"][0]["rows"][:1]
    candidate = _draft_from_result(candidate_result)

    assert not _recovery_preserves_grounded_charges(
        baseline,
        candidate,
        ((1, "p1-t1"),),
        {},
    )


def test_financial_inventory_allows_link_creation_without_token_id_identity() -> None:
    baseline = (
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "table_anchor": "table-anchor",
            "row_anchor": "row-anchor",
            "source_row_id": "baseline-source",
            "canonical_row_id": None,
            "description": "consultation",
            "bounds": (10.0, 10.0, 90.0, 30.0),
            "fields": {
                "net_amount": {
                    "value": "100.00",
                    "lineage_sha256": "same-root-lineage",
                    "grounding_strength": 1,
                }
            },
        },
    )
    candidate = (
        {
            **baseline[0],
            "source_row_id": "new-source-id",
            "canonical_row_id": "new-canonical-link",
        },
    )

    safe, preserved, added = _financial_inventory_matches(baseline, candidate, {})

    assert safe is True
    assert preserved == 1
    assert added == 0


def test_financial_inventory_rejects_deleting_linked_row_with_missing_amount() -> None:
    baseline = (
        {
            "page_number": 1,
            "table_id": "p1-t1",
            "table_anchor": "table-anchor",
            "row_anchor": "row-anchor",
            "source_row_id": "source-row",
            "canonical_row_id": "canonical-with-missing-amount",
            "canonical_amount_missing": True,
            "description": "consultation",
            "bounds": (10.0, 10.0, 90.0, 30.0),
            "fields": {
                "net_amount": {
                    "value": "100.00",
                    "lineage_sha256": "grounded-lineage",
                    "grounding_strength": 1,
                }
            },
        },
    )

    safe, preserved, added = _financial_inventory_matches(baseline, (), {})

    assert safe is False
    assert preserved == 0
    assert added == 0


def test_targeted_page_merge_preserves_untargeted_raw_unit_byte_identity(
    tmp_path: Path,
) -> None:
    _source, _artifact_root, result = _fixture(tmp_path)
    baseline = _draft_from_result(result)
    first = baseline.page_units[0]
    second = replace(
        first,
        page_asset=first.page_asset.model_copy(update={"page_number": 2}),
        provider_usage={"initial": {"paddle_calls": 7}},
    )
    recovered_first = replace(
        first,
        provider_usage={"recovery": {"paddle_calls": 1}},
    )

    merged = _merge_targeted_page_units(
        (first, second),
        (recovered_first,),
        ((1, "p1-t1"),),
    )

    assert merged[0] is not recovered_first
    assert merged[0].table_units[0] is recovered_first.table_units[0]
    assert merged[0].page_asset is first.page_asset
    assert merged[0].residual_digest() == first.residual_digest()
    assert merged[1] is second
    assert merged[1].raw_digest() == second.raw_digest()


def test_table_target_preserves_sibling_table_on_same_page_byte_for_byte(
    tmp_path: Path,
) -> None:
    _source, _artifact_root, result = _fixture(tmp_path)
    draft = _draft_from_result(result)
    page = draft.page_units[0]
    first_table = page.table_units[0]
    sibling_source = first_table.source_table.model_copy(
        update={"id": "p1-t2-source", "table_id": "p1-t2"}, deep=True
    )
    sibling = replace(
        first_table,
        table_id="p1-t2",
        source_table=sibling_source,
        provider_usage={"initial": {"paddle_calls": 2}},
    )
    baseline_page = replace(page, table_units=(first_table, sibling))
    recovered_first = replace(first_table, provider_usage={"recovery": {"paddle_calls": 1}})
    unsafe_sibling = replace(sibling, provider_usage={"recovery": {"paddle_calls": 99}})
    candidate_page = replace(page, table_units=(recovered_first, unsafe_sibling))

    merged = _merge_targeted_page_units((baseline_page,), (candidate_page,), ((1, "p1-t1"),))

    assert merged[0].table_units[0] is recovered_first
    assert merged[0].table_units[1] is sibling
    assert merged[0].table_units[1].raw_digest() == sibling.raw_digest()


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


JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=40),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5)
    ),
    max_leaves=30,
)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(JSON_VALUES)
def test_validator_is_total_for_arbitrary_nested_json(
    tmp_path: Path,
    payload: object,
) -> None:
    case_root = tmp_path / str(uuid4())
    case_root.mkdir()
    source, artifact_root, _result = _fixture(case_root)

    report = validate_extraction_result(source, payload, artifact_root)

    assert report.status == "failed"
    assert report.fatal
