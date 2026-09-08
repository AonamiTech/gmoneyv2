from __future__ import annotations

from copy import deepcopy

import pytest

from gmoney.contracts.authority import GoldDocumentV2, SourceClass, TableKind
from gmoney.contracts.evidence import Point, Polygon
from gmoney.evaluation import authority_metrics as metrics


def _table(
    rows: list[dict],
    *,
    table_id: str = "table-1",
    polygon: list[float] | None = None,
    columns: list[dict] | None = None,
) -> dict:
    return {
        "table_id": table_id,
        "page_number": 1,
        "polygon": polygon or [0, 0, 100, 100],
        "columns": columns
        or [
            {
                "id": "description",
                "canonical_field": "description",
                "order": 0,
                "polygon": [0, 0, 55, 100],
            },
            {"id": "amount", "canonical_field": "amount", "order": 1, "polygon": [55, 0, 100, 100]},
        ],
        "rows": rows,
    }


def _row(
    order: int = 0,
    *,
    description: str = "Dressing",
    amount: str | None = "₹1,000.00",
    readable: bool = True,
    evidence: bool = True,
) -> dict:
    evidence_value = [{"token_ids": [f"token-{order}"]}] if evidence else []
    return {
        "id": f"row-{order}",
        "order": order,
        "polygon": [0, order * 20, 100, order * 20 + 15],
        "cells": [
            {
                "column_id": "description",
                "canonical_field": "description",
                "raw_value": description if readable else None,
                "readable": readable,
                "evidence": evidence_value,
            },
            {
                "column_id": "amount",
                "canonical_field": "amount",
                "raw_value": amount if readable else None,
                "readable": readable,
                "evidence": evidence_value,
            },
        ],
    }


def _document(
    rows: list[dict],
    *,
    table: dict | None = None,
    totals: list[dict] | None = None,
    key: str = "source-a",
) -> dict:
    return {
        "source_sha256": key,
        "tables": [table or _table(rows)],
        "totals": totals or [],
    }


def test_duplicate_rows_are_not_collapsed_into_recall() -> None:
    gold = _document([_row()])
    actual = _document([_row(), _row(1)])

    report = metrics.evaluate_document(gold, actual)

    assert report.metrics["matched_rows"] == 1
    assert report.metrics["row_recall"] == 1.0
    assert report.metrics["duplicates"] == 1
    assert any(row.status == "duplicate" for row in report.rows)


def test_columns_follow_geometry_and_canonical_role_after_shift() -> None:
    gold_columns = [
        {
            "id": "description",
            "canonical_field": "description",
            "order": 0,
            "polygon": [0, 0, 40, 100],
        },
        {"id": "amount", "canonical_field": "amount", "order": 1, "polygon": [60, 0, 100, 100]},
    ]
    actual_columns = [
        {"id": "amount-new", "canonical_field": "amount", "order": 0, "polygon": [60, 0, 100, 100]},
        {
            "id": "description-new",
            "canonical_field": "description",
            "order": 1,
            "polygon": [0, 0, 40, 100],
        },
    ]
    gold_row = _row()
    gold_row["cells"] = [
        {"column_id": "description", "canonical_field": "description", "raw_value": "Dressing"},
        {"column_id": "amount", "canonical_field": "amount", "raw_value": "1000.00"},
    ]
    actual_row = deepcopy(gold_row)
    actual_row["cells"] = [
        {
            "column_id": "amount-new",
            "canonical_field": "amount",
            "raw_value": "1000.00",
            "evidence": ["a"],
        },
        {
            "column_id": "description-new",
            "canonical_field": "description",
            "raw_value": "Dressing",
            "evidence": ["b"],
        },
    ]

    report = metrics.evaluate_document(
        _document([gold_row], table=_table([gold_row], columns=gold_columns)),
        _document([actual_row], table=_table([actual_row], columns=actual_columns)),
    )

    assert report.metrics["matched_columns"] == 2
    assert report.metrics["column_assignment_recall"] == 1.0
    assert report.metrics["exact_cells"] == 2
    assert not any("wrong_column" in row.issues for row in report.rows)


def test_swapped_values_emit_wrong_column_instead_of_a_lucky_match() -> None:
    gold = _document([_row()])
    swapped = _row()
    swapped["cells"] = [
        {
            "column_id": "description",
            "canonical_field": "description",
            "raw_value": "1000.00",
            "evidence": ["a"],
        },
        {
            "column_id": "amount",
            "canonical_field": "amount",
            "raw_value": "Dressing",
            "evidence": ["b"],
        },
    ]

    report = metrics.evaluate_document(gold, _document([swapped]))

    assert report.metrics["wrong_column"] >= 1
    assert any("wrong_column" in row.issues for row in report.rows)


def test_critical_exact_requires_a_critical_prediction_role() -> None:
    gold = _document([_row()])
    actual_row = _row()
    actual_row["cells"][1]["canonical_field"] = "description"

    report = metrics.evaluate_document(gold, _document([actual_row]))

    assert report.metrics["critical_gold"] == 1
    assert report.metrics["critical_actual"] == 0
    assert report.metrics["critical_exact"] == 0
    assert report.metrics["critical_numeric_cell_recall"] == 0.0
    assert report.floors["critical_numeric_cell_precision"] is False
    assert metrics.EVALUATOR_VERSION == "authority_metrics_v3"


def test_unreadable_gold_cell_is_excluded_from_value_recall_but_not_completeness() -> None:
    gold_row = _row(readable=False, amount=None)
    actual_row = _row(readable=False, amount=None)
    report = metrics.evaluate_document(_document([gold_row]), _document([actual_row]))

    assert report.metrics["gold_unreadable_cells"] == 2
    assert report.metrics["correct_unreadable_cells"] == 2
    assert report.metrics["unreadable_recall"] == 1.0
    assert report.metrics["unreadable_mishandled"] == 0
    assert report.metrics["complete_document_exact"] == 1.0


def test_asserting_a_value_for_unreadable_source_is_mishandled() -> None:
    gold_row = _row(readable=False, amount=None)
    actual_row = _row(readable=True, amount="10.00")
    report = metrics.evaluate_document(_document([gold_row]), _document([actual_row]))

    assert report.metrics["unreadable_mishandled"] == 2
    assert any(row.status == "unreadable_mishandled" for row in report.rows)


def test_totals_use_money_normalized_to_inr_cents_and_are_reported_separately() -> None:
    totals = [
        {
            "total_id": "grand",
            "label": "Grand Total",
            "kind": "document_total",
            "scope": "document",
            "amount": "₹1,000.004",
            "evidence": [{"token_ids": ["total"]}],
        }
    ]
    actual_totals = [
        {
            "total_id": "grand",
            "label": "Grand Total",
            "kind": "document_total",
            "scope": "document",
            "amount": "1000.004",
            "evidence": [{"token_ids": ["total"]}],
        }
    ]
    report = metrics.evaluate_document(
        _document([_row()], totals=totals),
        _document([_row()], totals=actual_totals),
    )

    # ROUND_HALF_UP to INR 0.01 makes both values 1000.00 and the total exact.
    assert report.metrics["exact_totals"] == 1
    assert report.metrics["grand_total_exact_accuracy"] == 1.0
    assert report.totals[0].gold_value == "1000.00"
    assert report.totals[0].actual_value == "1000.00"


def test_wrong_table_is_distinguished_from_a_missed_row() -> None:
    gold = _document([_row()], table=_table([_row()], table_id="gold", polygon=[0, 0, 40, 40]))
    actual = _document(
        [_row()], table=_table([_row()], table_id="other", polygon=[60, 60, 100, 100])
    )

    report = metrics.evaluate_document(gold, actual)

    assert report.metrics["matched_tables"] == 0
    assert report.metrics["wrong_table"] >= 1
    assert any(row.status == "wrong_table" for row in report.rows)


def test_document_report_is_byte_stable_for_repeated_evaluation() -> None:
    gold = _document([_row(), _row(1)], key="source-b")
    actual = _document([_row(1), _row()], key="source-b")

    first = metrics.evaluate_document(gold, actual).to_dict()
    second = metrics.evaluate_document(gold, actual).to_dict()

    assert first == second
    assert metrics.identity_sha256(first) == metrics.identity_sha256(second)


def test_cohort_floor_boundary_uses_greater_than_or_equal() -> None:
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(metrics.TABLE_MAGIC_FLOORS, "critical_numeric_cell_precision", 1.0)
        pair = {"gold": _document([_row()]), "actual": _document([_row()])}
        exact = metrics.evaluate_cohort([pair])
        assert exact.floors["critical_numeric_cell_precision"] is True

        monkeypatch.setitem(metrics.TABLE_MAGIC_FLOORS, "critical_numeric_cell_precision", 1.000001)
        below = metrics.evaluate_cohort([pair])
        assert below.floors["critical_numeric_cell_precision"] is False
        assert below.passed is False
    finally:
        monkeypatch.undo()


def test_empty_documents_and_cohorts_are_rejected() -> None:
    with pytest.raises(ValueError, match="gold document may not be empty"):
        metrics.evaluate_document({}, {})
    with pytest.raises(ValueError, match="cohorts may not be empty"):
        metrics.evaluate_cohort([], [])


def test_zero_denominator_floors_fail_and_auto_accept_coverage_is_explicit() -> None:
    report = metrics.evaluate_document(_document([_row()]), _document([_row()]))

    assert report.metrics["auto_accepted_document_coverage"] == 0.0
    assert report.metrics["auto_accepted_documents"] == 0
    assert report.metrics["auto_accepted_document_correctness"] == 0.0
    assert report.floors["auto_accepted_document_correctness"] is False
    assert report.floors["grand_total_exact_accuracy"] is False
    assert report.passed is False

    cohort = metrics.evaluate_cohort(
        [{"gold": _document([_row()]), "actual": _document([_row()])}]
    )
    assert cohort.metrics["auto_accepted_document_coverage"] == 0.0
    assert cohort.floors["auto_accepted_document_correctness"] is False
    assert cohort.floors["grand_total_exact_accuracy"] is False
    assert cohort.passed is False


def test_machine_normalization_uses_visible_raw_value() -> None:
    actual_row = _row(amount="999.00")
    actual_row["cells"][0]["normalized_value"] = "Dressing"
    actual_row["cells"][1]["normalized_value"] = "1000.00"

    report = metrics.evaluate_document(_document([_row()]), _document([actual_row]))

    amount = next(outcome for outcome in report.cells if outcome.canonical_role == "amount")
    assert amount.actual_value == "999.00"
    assert amount.exact is False


def test_cohort_pairing_rejects_duplicates_reordering_and_set_mismatch() -> None:
    first = _document([_row()], key="source-a")
    second = _document([_row()], key="source-b")

    report = metrics.evaluate_cohort([first, second], [first, second])
    assert [item.document_key for item in report.documents] == ["source-a", "source-b"]

    with pytest.raises(ValueError, match="order"):
        metrics.evaluate_cohort([first, second], [second, first])
    with pytest.raises(ValueError, match="sets"):
        metrics.evaluate_cohort([first], [second])
    with pytest.raises(ValueError, match="duplicate"):
        metrics.evaluate_cohort([first, first], [first, first])


def test_authority_v2_models_are_consumed_without_projection() -> None:
    def polygon(left: float, top: float, right: float, bottom: float) -> Polygon:
        return Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        )

    source_sha = "a" * 64
    table_polygon = polygon(0, 0, 100, 100)
    gold = GoldDocumentV2.model_validate(
        {
            "source_sha256": source_sha,
            "source_manifest_sha256": "b" * 64,
            "page_count": 1,
            "pages": [
                {
                    "source_sha256": source_sha,
                    "page_number": 1,
                    "artifact_sha256": "c" * 64,
                    "width": 100,
                    "height": 100,
                    "dpi": 72,
                    "source_class": SourceClass.FLAT_SCAN,
                    "tables": [
                        {
                            "polygon": table_polygon,
                            "table_order": 0,
                            "table_kind": TableKind.ITEM_LEDGER,
                            "columns": [
                                {
                                    "polygon": polygon(0, 0, 50, 100),
                                    "column_order": 0,
                                    "canonical_field": "description",
                                },
                                {
                                    "polygon": polygon(50, 0, 100, 100),
                                    "column_order": 1,
                                    "canonical_field": "amount",
                                },
                            ],
                            "rows": [
                                {
                                    "polygon": polygon(0, 0, 100, 20),
                                    "row_order": 0,
                                    "row_kind": "detail",
                                    "cells": [
                                        {
                                            "polygon": polygon(0, 0, 50, 20),
                                            "column_order": 0,
                                            "raw_value": "Dressing",
                                            "normalized_value": "Dressing",
                                        },
                                        {
                                            "polygon": polygon(50, 0, 100, 20),
                                            "column_order": 1,
                                            "raw_value": "₹1,000.00",
                                            "normalized_value": "1000.00",
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            "annotation_group_id": "group-a",
            "split": "holdout",
        }
    )
    actual = {
        "source_sha256": source_sha,
        "source_tables": [
            {
                "page_number": 1,
                "table_id": gold.pages[0].tables[0].table_id,
                "table_kind": "item_ledger",
                "polygon": table_polygon,
                "columns": [
                    {"id": "description", "order": 0, "canonical_field": "description"},
                    {"id": "amount", "order": 1, "canonical_field": "amount"},
                ],
                "rows": [
                    {
                        "id": "actual-row",
                        "order": 0,
                        "polygon": polygon(0, 0, 100, 20),
                        "cells": [
                            {
                                "column_id": "description",
                                "raw_value": "Dressing",
                                "evidence": [{"token_ids": ["description-token"]}],
                            },
                            {
                                "column_id": "amount",
                                "raw_value": "1000.00",
                                "evidence": [{"token_ids": ["amount-token"]}],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    report = metrics.evaluate_document(gold, actual)

    assert report.gold_identity_sha256 == gold.gold_sha256
    assert report.metrics["matched_tables"] == 1
    assert report.metrics["exact_cells"] == 2
    assert report.metrics["critical_numeric_cell_recall"] == 1.0


def test_v6_source_tables_use_canonical_artifact_geometry_instead_of_list_order() -> None:
    top_row = _row(description="Top service", amount="100.00")
    bottom_row = _row(description="Bottom service", amount="200.00")
    gold = {
        "source_sha256": "a" * 64,
        "tables": [
            _table([top_row], table_id="gold-top", polygon=[0, 0, 100, 40]),
            _table([bottom_row], table_id="gold-bottom", polygon=[0, 60, 100, 100]),
        ],
    }

    actual_top = _table([deepcopy(top_row)], table_id="actual-top")
    actual_bottom = _table([deepcopy(bottom_row)], table_id="actual-bottom")
    actual_top.pop("polygon")
    actual_bottom.pop("polygon")
    actual = {
        "source_sha256": "a" * 64,
        # Producer order is intentionally opposite source-space order.
        "source_tables": [actual_bottom, actual_top],
        "canonical_table_artifacts": [
            {
                "page_number": 1,
                "logical_table_id": "actual-top",
                "crop_polygon_in_source_raw": [0, 0, 100, 40],
            },
            {
                "page_number": 1,
                "logical_table_id": "actual-bottom",
                "crop_polygon_in_source_raw": [0, 60, 100, 100],
            },
        ],
    }

    report = metrics.evaluate_document(gold, actual)

    assert report.metrics["matched_tables"] == 2
    assert report.metrics["exact_cells"] == 4
    assert [(item.gold_id, item.actual_id) for item in report.tables] == [
        ("gold-top", "actual-top"),
        ("gold-bottom", "actual-bottom"),
    ]


def test_v6_geometry_join_does_not_invent_ambiguous_source_table_geometry() -> None:
    table = _table([_row()], table_id="actual")
    table.pop("polygon")
    actual = {
        "source_tables": [table],
        "canonical_table_artifacts": [
            {
                "page_number": 1,
                "logical_table_id": "actual",
                "crop_polygon_in_source_raw": [0, 0, 40, 40],
            },
            {
                "page_number": 1,
                "logical_table_id": "actual",
                "crop_polygon_in_source_raw": [60, 60, 100, 100],
            },
        ],
    }

    enriched = metrics._tables(actual)

    assert metrics._geometry(enriched[0]) is None
