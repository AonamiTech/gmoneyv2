from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gmoney.contracts.authority import (
    BaselineDocumentV1,
    BaselineManifestV1,
    ContinuationKind,
    EvaluatorManifestV1,
    GateDecision,
    GoldDocumentV2,
    M0M4GateReportV1,
    MetricObservationV1,
    ReviewDecision,
    ReviewKind,
    ReviewRecordV1,
    ReviewState,
    RowKind,
    SourceClass,
    SourceManifestV1,
    TableKind,
    TotalKind,
    cell_id_for,
    column_id_for,
    row_id_for,
    source_document_id_for,
    table_id_for,
)

SOURCE_SHA = "a" * 64
ARTIFACT_SHA = "b" * 64


def _polygon(x: int = 0, y: int = 0, width: int = 10, height: int = 10) -> dict:
    return {
        "points": [
            {"x": x, "y": y},
            {"x": x + width, "y": y},
            {"x": x + width, "y": y + height},
        ]
    }


def _source_table(*, two_rows: bool = False) -> dict:
    rows = [
        {
            "polygon": _polygon(0, 20),
            "cells": [
                {"polygon": _polygon(0, 20)},
                {"polygon": _polygon(20, 20)},
            ],
        }
    ]
    if two_rows:
        rows.append(
            {
                "polygon": _polygon(0, 35),
                "cells": [
                    {"polygon": _polygon(0, 35)},
                    {"polygon": _polygon(20, 35)},
                ],
            }
        )
    return {
        "polygon": _polygon(),
        "table_kind": TableKind.ITEM_LEDGER,
        "columns": [
            {"polygon": _polygon(0, 0, 10, 15)},
            {"polygon": _polygon(20, 0, 10, 15)},
        ],
        "rows": rows,
    }


def _source_page(
    *, two_rows: bool = False, source_class: SourceClass = SourceClass.FLAT_SCAN
) -> dict:
    return {
        "source_sha256": SOURCE_SHA,
        "page_number": 1,
        "artifact_sha256": ARTIFACT_SHA,
        "artifact_relative_path": "pages/0001.png",
        "width": 100,
        "height": 100,
        "dpi": 300,
        "source_class": source_class,
        "tables": [_source_table(two_rows=two_rows)],
    }


def _source_manifest(*, two_rows: bool = False, **overrides) -> SourceManifestV1:
    payload = {
        "source_sha256": SOURCE_SHA,
        "source_mime_type": "application/pdf",
        "source_size_bytes": 1234,
        "page_count": 1,
        "pages": [_source_page(two_rows=two_rows)],
        "source_class": SourceClass.FLAT_SCAN,
        "cohorts": ("production14", "staging159"),
        "split": "holdout",
        "group_id": "document-group-01",
    }
    payload.update(overrides)
    return SourceManifestV1(**payload)


def _gold_table(
    *,
    description: str = "Printed item",
    label: str = "Amount",
    raw: str = "12.50",
    two_rows: bool = False,
) -> dict:
    rows = [
        {
            "polygon": _polygon(0, 20),
            "row_kind": RowKind.DETAIL,
            "description": description,
            "amount": Decimal(raw),
            "cells": [
                {"polygon": _polygon(0, 20), "raw_value": description},
                {"polygon": _polygon(20, 20), "raw_value": raw, "normalized_value": raw},
            ],
        }
    ]
    if two_rows:
        rows.append(
            {
                "polygon": _polygon(0, 35),
                "row_kind": RowKind.DETAIL,
                "description": "Second printed item",
                "amount": Decimal("4.00"),
                "cells": [
                    {"polygon": _polygon(0, 35), "raw_value": "Second printed item"},
                    {"polygon": _polygon(20, 35), "raw_value": "4.00", "normalized_value": "4.00"},
                ],
            }
        )
    return {
        "polygon": _polygon(),
        "table_kind": TableKind.ITEM_LEDGER,
        "columns": [
            {"polygon": _polygon(0, 0, 10, 15), "label": "Description"},
            {"polygon": _polygon(20, 0, 10, 15), "label": label, "canonical_field": "amount"},
        ],
        "rows": rows,
    }


def _gold_document(
    *,
    description: str = "Printed item",
    label: str = "Amount",
    raw: str = "12.50",
    two_rows: bool = False,
    continuations: list[dict] | None = None,
    totals: list[dict] | None = None,
) -> GoldDocumentV2:
    source = _source_manifest(two_rows=two_rows)
    return GoldDocumentV2(
        source_sha256=SOURCE_SHA,
        source_manifest_sha256=source.manifest_sha256,
        page_count=1,
        pages=[
            {
                "source_sha256": SOURCE_SHA,
                "page_number": 1,
                "artifact_sha256": ARTIFACT_SHA,
                "width": 100,
                "height": 100,
                "dpi": 300,
                "source_class": SourceClass.FLAT_SCAN,
                "tables": [
                    _gold_table(
                        description=description,
                        label=label,
                        raw=raw,
                        two_rows=two_rows,
                    )
                ],
            }
        ],
        continuations=continuations or (),
        totals=totals or (),
        annotation_group_id="group-01",
        split="holdout",
    )


def _review_payload(**overrides) -> dict:
    payload = {
        "document_id": source_document_id_for(SOURCE_SHA),
        "source_sha256": SOURCE_SHA,
        "review_kind": ReviewKind.INDEPENDENT_A,
        "reviewer_identity": "reviewer-a",
        "model_identity": "human-visual-review",
        "tool_identity": "luna-visual-audit",
        "reasoning_identity": "independent-first-v1",
        "prompt_identity": "authority-independent-a-v1",
        "prompt_sha256": "c" * 64,
        "image_identities": ("page-1",),
        "image_sha256s": (ARTIFACT_SHA,),
        "image_manifest_sha256": "d" * 64,
        "model_config_sha256": "f" * 64,
        "observations_sha256": "1" * 64,
    }
    payload.update(overrides)
    return payload


def _evaluator() -> EvaluatorManifestV1:
    return EvaluatorManifestV1(
        evaluator_id="authority-metrics",
        evaluator_version="authority_metrics_v1",
        code_sha256="1" * 64,
        configuration_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        gold_manifest_sha256="4" * 64,
        metric_names=("row_recall", "amount_precision"),
        normalization_policy="decimal-v1",
        matching_policy="geometry-order-v1",
        unreadable_policy="exclude-from-denominator-v1",
    )


def test_source_manifest_is_versioned_frozen_and_hashes_nested_content() -> None:
    manifest = _source_manifest()
    assert manifest.manifest_version == "source_manifest_v1"
    assert manifest.document_id == source_document_id_for(SOURCE_SHA)
    assert len(manifest.manifest_sha256) == 64
    with pytest.raises((TypeError, ValidationError)):
        manifest.cohorts = ("production14",)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SourceManifestV1.model_validate({**manifest.model_dump(mode="json"), "unknown": True})


def test_source_manifest_rejects_page_count_hash_and_class_mismatches() -> None:
    with pytest.raises(ValidationError, match="page_count"):
        _source_manifest(page_count=2)
    with pytest.raises(ValidationError, match="source class"):
        _source_manifest(pages=[_source_page(source_class=SourceClass.CAMERA_PHOTO)])

    manifest = _source_manifest()
    tampered = manifest.model_dump(mode="json")
    tampered["manifest_sha256"] = "9" * 64
    with pytest.raises(ValidationError, match="hash"):
        SourceManifestV1.model_validate(tampered)

    with pytest.raises(ValidationError, match="canonical order"):
        _source_manifest(cohorts=("staging159", "production14"))
    with pytest.raises(ValidationError, match="unique"):
        _source_manifest(cohorts=("production14", "production14"))
    with pytest.raises(ValidationError, match="sealed master"):
        _source_manifest(cohorts=("production14",), sealed_master=True)


def test_structural_ids_normalize_geometry_and_ignore_transcription() -> None:
    integer_polygon = _polygon()
    float_polygon = {
        "points": [
            {"x": float(point["x"]), "y": float(point["y"])}
            for point in integer_polygon["points"]
        ]
    }
    page_id = "1" * 64
    table_id = table_id_for(page_id, 0, integer_polygon)
    assert table_id_for(page_id, 0, float_polygon) == table_id
    column_id = column_id_for(table_id, 0, integer_polygon)
    row_id = row_id_for(table_id, 0, integer_polygon)
    assert cell_id_for(row_id, column_id, integer_polygon) == cell_id_for(
        row_id, column_id, float_polygon
    )

    first = _gold_document()
    second = _gold_document(description="Corrected transcription", label="Net", raw="99.99")
    assert first.document_id == second.document_id
    assert first.gold_sha256 != second.gold_sha256
    first_table = first.pages[0].tables[0]
    second_table = second.pages[0].tables[0]
    assert first_table.table_id == second_table.table_id
    assert first_table.columns[1].column_id == second_table.columns[1].column_id
    assert first_table.rows[0].row_id == second_table.rows[0].row_id
    assert first_table.rows[0].cells[1].cell_id == second_table.rows[0].cells[1].cell_id


def test_gold_document_requires_source_identity_and_rejects_unreadable_values() -> None:
    gold = _gold_document()
    assert gold.gold_version == "gold_document_v2"
    bad_document = gold.model_dump(mode="json")
    bad_document["document_id"] = "8" * 64
    with pytest.raises(ValidationError, match="derived from source"):
        GoldDocumentV2.model_validate(bad_document)

    table = _gold_table()
    table["rows"][0]["cells"][0]["readable"] = False
    table["rows"][0]["cells"][0]["raw_value"] = "invented"
    with pytest.raises(ValidationError, match="unreadable"):
        GoldDocumentV2(
            source_sha256=SOURCE_SHA,
            source_manifest_sha256=_source_manifest().manifest_sha256,
            page_count=1,
            pages=[
                {
                    "source_sha256": SOURCE_SHA,
                    "page_number": 1,
                    "artifact_sha256": ARTIFACT_SHA,
                    "width": 100,
                    "height": 100,
                    "dpi": 300,
                    "source_class": SourceClass.FLAT_SCAN,
                    "tables": [table],
                }
            ],
            annotation_group_id="group-01",
            split="holdout",
        )


def test_gold_continuation_and_total_references_are_structural() -> None:
    source = _source_manifest(two_rows=True)
    rows = _gold_document(two_rows=True).pages[0].tables[0].rows
    continuation = {
        "source_sha256": SOURCE_SHA,
        "continuation_order": 0,
        "from_row_id": rows[0].row_id,
        "to_row_id": rows[1].row_id,
        "continuation_kind": ContinuationKind.CROSS_PAGE,
    }
    total = {
        "source_sha256": SOURCE_SHA,
        "total_order": 0,
        "scope_id": _gold_document(two_rows=True).pages[0].tables[0].table_id,
        "total_kind": TotalKind.DOCUMENT_TOTAL,
        "polygon": _polygon(0, 60),
        "raw_value": "16.50",
        "normalized_value": "16.50",
    }
    gold = GoldDocumentV2(
        source_sha256=SOURCE_SHA,
        source_manifest_sha256=source.manifest_sha256,
        page_count=1,
        pages=[
            {
                "source_sha256": SOURCE_SHA,
                "page_number": 1,
                "artifact_sha256": ARTIFACT_SHA,
                "width": 100,
                "height": 100,
                "dpi": 300,
                "source_class": SourceClass.FLAT_SCAN,
                "tables": [_gold_table(two_rows=True)],
            }
        ],
        continuations=[continuation],
        totals=[total],
        annotation_group_id="group-01",
        split="holdout",
    )
    assert gold.continuations[0].continuation_id
    assert gold.totals[0].normalized_value == Decimal("16.50")

    bad_continuation = dict(continuation, to_row_id="9" * 64)
    with pytest.raises(ValidationError, match="missing row"):
        _gold_document(two_rows=True, continuations=[bad_continuation])

    bad_total = dict(total, scope_id="9" * 64)
    with pytest.raises(ValidationError, match="missing scope"):
        GoldDocumentV2(
            source_sha256=SOURCE_SHA,
            source_manifest_sha256=source.manifest_sha256,
            page_count=1,
            pages=[
                {
                    "source_sha256": SOURCE_SHA,
                    "page_number": 1,
                    "artifact_sha256": ARTIFACT_SHA,
                    "width": 100,
                    "height": 100,
                    "dpi": 300,
                    "source_class": SourceClass.FLAT_SCAN,
                    "tables": [_gold_table(two_rows=True)],
                }
            ],
            totals=[bad_total],
            annotation_group_id="group-01",
            split="holdout",
        )


def test_review_identity_is_content_bound_but_not_decision_bound() -> None:
    first = ReviewRecordV1(**_review_payload())
    changed_decision = ReviewRecordV1(
        **_review_payload(decision=ReviewDecision.ACCEPT)
    )
    assert first.review_id == changed_decision.review_id
    assert first.review_id == ReviewRecordV1(
        **_review_payload(review_kind="independent_a")
    ).review_id

    with pytest.raises(ValidationError, match="machine output"):
        ReviewRecordV1(**_review_payload(machine_output_sha256="7" * 64))
    with pytest.raises(ValidationError, match="blind"):
        ReviewRecordV1(**_review_payload(blind=False))
    with pytest.raises(ValidationError, match="unfrozen"):
        ReviewRecordV1(
            **_review_payload(frozen_at=datetime.now(UTC), frozen_by="reviewer")
        )


def test_frozen_review_records_have_a_reproducible_freeze_digest() -> None:
    frozen = ReviewRecordV1(
        **_review_payload(
            state=ReviewState.FROZEN,
            frozen_at=datetime(2026, 9, 1, tzinfo=UTC),
            frozen_by="reviewer-a",
        )
    )
    assert frozen.frozen_payload_sha256 is not None
    tampered = frozen.model_dump(mode="json")
    tampered["frozen_payload_sha256"] = "8" * 64
    with pytest.raises(ValidationError, match="frozen review payload"):
        ReviewRecordV1.model_validate(tampered)


def test_adjudicator_requires_independent_first_a_and_b_records() -> None:
    review_a = ReviewRecordV1(**_review_payload(review_kind=ReviewKind.INDEPENDENT_A))
    review_b = ReviewRecordV1(
        **_review_payload(
            review_kind=ReviewKind.INDEPENDENT_B,
            reviewer_identity="reviewer-b",
            prompt_identity="authority-independent-b-v1",
        )
    )
    adjudicator = ReviewRecordV1(
        **_review_payload(
            review_kind=ReviewKind.ADJUDICATOR,
            reviewer_identity="adjudicator",
            prompt_identity="authority-adjudicator-v1",
            blind=False,
            independent_first=True,
            independent_review_ids=(review_a.review_id, review_b.review_id),
        )
    )
    assert adjudicator.independent_first is True
    with pytest.raises(ValidationError, match="independent-first"):
        ReviewRecordV1(
            **_review_payload(
                review_kind=ReviewKind.ADJUDICATOR,
                blind=False,
                independent_review_ids=(review_a.review_id, review_b.review_id),
            )
        )
    with pytest.raises(ValidationError, match="both independent"):
        ReviewRecordV1(
            **_review_payload(
                review_kind=ReviewKind.ADJUDICATOR,
                blind=False,
                independent_first=True,
                independent_review_ids=(review_a.review_id,),
            )
        )


def test_evaluator_and_baseline_manifests_are_digest_bound_and_unique() -> None:
    evaluator = _evaluator()
    assert len(evaluator.manifest_sha256) == 64
    metric = MetricObservationV1(
        metric_name="row_recall",
        value=1.0,
        numerator=1,
        denominator=1,
        cohort="production14",
    )
    document = BaselineDocumentV1(
        document_id=source_document_id_for(SOURCE_SHA),
        source_sha256=SOURCE_SHA,
        result_sha256="5" * 64,
        metrics=(metric,),
        status="complete",
    )
    baseline = BaselineManifestV1(
        source_manifest_sha256="6" * 64,
        gold_manifest_sha256="7" * 64,
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        documents=(document,),
        release_revision="release-1",
        model_identity="baseline-model",
        configuration_sha256="8" * 64,
    )
    assert len(baseline.baseline_id) == 64
    assert len(baseline.manifest_sha256) == 64

    changed_result = BaselineManifestV1(
        source_manifest_sha256="6" * 64,
        gold_manifest_sha256="7" * 64,
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        documents=(document.model_copy(update={"result_sha256": "9" * 64}),),
        release_revision="release-1",
        model_identity="baseline-model",
        configuration_sha256="8" * 64,
    )
    changed_status = BaselineManifestV1(
        source_manifest_sha256="6" * 64,
        gold_manifest_sha256="7" * 64,
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        documents=(document.model_copy(update={"status": "failed"}),),
        release_revision="release-1",
        model_identity="baseline-model",
        configuration_sha256="8" * 64,
    )
    changed_metric = BaselineManifestV1(
        source_manifest_sha256="6" * 64,
        gold_manifest_sha256="7" * 64,
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        documents=(
            document.model_copy(
                update={"metrics": (metric.model_copy(update={"value": 0.5}),)}
            ),
        ),
        release_revision="release-1",
        model_identity="baseline-model",
        configuration_sha256="8" * 64,
    )
    assert baseline.baseline_id not in {
        changed_result.baseline_id,
        changed_status.baseline_id,
        changed_metric.baseline_id,
    }
    with pytest.raises(ValidationError, match="unique source hashes"):
        BaselineManifestV1(
            source_manifest_sha256="6" * 64,
            gold_manifest_sha256="7" * 64,
            evaluator_manifest_sha256=evaluator.manifest_sha256,
            documents=(document, document),
            release_revision="release-1",
            model_identity="baseline-model",
            configuration_sha256="8" * 64,
        )


def test_gate_report_enforces_m0_to_m4_promotion_rules_and_hashes() -> None:
    cohorts = tuple(
        {
            "cohort": cohort,
            "required_count": required_count,
            "observed_count": required_count,
            "passed": True,
        }
        for cohort, required_count in (
            ("production14", 14),
            ("passing36", 36),
            ("staging159", 159),
        )
    )
    report = M0M4GateReportV1(
        milestone="M4",
        decision=GateDecision.PROMOTE,
        source_manifest_sha256="1" * 64,
        gold_manifest_sha256="2" * 64,
        evaluator_manifest_sha256="3" * 64,
        baseline_manifest_sha256="4" * 64,
        candidate_revision="candidate-1",
        candidate_configuration_sha256="5" * 64,
        cohorts=cohorts,
    )
    assert report.report_version == "m0_m4_gate_report_v1"
    assert len(report.report_sha256) == 64
    with pytest.raises(ValidationError, match="blocking"):
        M0M4GateReportV1(
            milestone="M4",
            decision=GateDecision.PROMOTE,
            source_manifest_sha256="1" * 64,
            gold_manifest_sha256="2" * 64,
            evaluator_manifest_sha256="3" * 64,
            baseline_manifest_sha256="4" * 64,
            candidate_revision="candidate-1",
            candidate_configuration_sha256="5" * 64,
            cohorts=cohorts,
            blocking_reasons=("gold_missing",),
        )
    with pytest.raises(ValidationError, match="blocking reason"):
        M0M4GateReportV1(
            milestone="M2",
            decision=GateDecision.HOLD,
            source_manifest_sha256="1" * 64,
            gold_manifest_sha256="2" * 64,
            evaluator_manifest_sha256="3" * 64,
            baseline_manifest_sha256="4" * 64,
            candidate_revision="candidate-1",
            candidate_configuration_sha256="5" * 64,
            cohorts=cohorts,
        )

    with pytest.raises(ValidationError, match="exactly"):
        M0M4GateReportV1(
            milestone="M4",
            decision=GateDecision.PROMOTE,
            source_manifest_sha256="1" * 64,
            gold_manifest_sha256="2" * 64,
            evaluator_manifest_sha256="3" * 64,
            baseline_manifest_sha256="4" * 64,
            candidate_revision="candidate-1",
            candidate_configuration_sha256="5" * 64,
            cohorts=cohorts[:-1],
        )

    with pytest.raises(ValidationError, match="requires exactly 14"):
        M0M4GateReportV1(
            milestone="M4",
            decision=GateDecision.PROMOTE,
            source_manifest_sha256="1" * 64,
            gold_manifest_sha256="2" * 64,
            evaluator_manifest_sha256="3" * 64,
            baseline_manifest_sha256="4" * 64,
            candidate_revision="candidate-1",
            candidate_configuration_sha256="5" * 64,
            cohorts=tuple(
                (
                    {
                        "cohort": "production14",
                        "required_count": 1,
                        "observed_count": 1,
                        "passed": True,
                    },
                    *cohorts[1:],
                )
            ),
        )
