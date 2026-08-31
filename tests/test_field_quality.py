from datetime import UTC, datetime

from gmoney.contracts.gold import GoldAnnotation
from gmoney.evaluation.field_quality import (
    DocumentFields,
    evaluate_field_quality,
    evaluate_with_layout_slices,
    normalize_value,
)


def annotation(*, unreadable: bool = False) -> GoldAnnotation:
    return GoldAnnotation.model_validate(
        {
            "annotation_version": "gold_annotation_v2",
            "bill_file": "bill.pdf",
            "document_sha256": "a" * 64,
            "layout_family_id": "layout-a",
            "image_review": {
                "reviewer": "Codex",
                "method": "codex_image_review",
                "passes": 2,
                "reviewed_at": datetime.now(UTC).isoformat(),
            },
            "rows": [
                {
                    "page_number": 1,
                    "table_id": "p1-t1",
                    "row_order": 0,
                    "description": "Room Charge",
                    "service_date": "31/08/2026",
                    "request_no": "A-10",
                    "quantity": "2",
                    "rate": "50",
                    "gross_amount": "100",
                    "discount": "5",
                    "amount": "95",
                }
            ],
            "source_tables": [
                {
                    "page_number": 1,
                    "table_id": "p1-t1",
                    "columns": [
                        {"id": "c1", "label": "Service", "order": 0},
                        {"id": "c2", "label": "Net Amount", "order": 1},
                    ],
                    "rows": [
                        {
                            "order": 0,
                            "cells": [
                                {"column_id": "c1", "raw_value": "Room Charge"},
                                {
                                    "column_id": "c2",
                                    "raw_value": None if unreadable else "95.00",
                                    "readable": not unreadable,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )


def actual(*, amount: str = "95.00", extra_column: bool = False) -> dict[str, object]:
    columns = [
        {"id": "c1", "label": "Service", "order": 0},
        {"id": "c2", "label": "Net Amount", "order": 1},
    ]
    cells = [
        {"column_id": "c1", "raw_value": "Room charge"},
        {"column_id": "c2", "raw_value": amount},
    ]
    if extra_column:
        columns.append({"id": "c3", "label": "Invented", "order": 2})
        cells.append({"column_id": "c3", "raw_value": "bad"})
    return {
        "rows": [
            {
                "page_number": 1,
                "table_id": "p1-t1",
                "row_order": 0,
                "description": "room charge",
                "service_date_iso": "2026-08-31",
                "request_no": "A10",
                "quantity": "2.0",
                "unit_price": "50.00",
                "gross_amount": "100.00",
                "discount": "5.00",
                "net_amount": amount,
            }
        ],
        "source_tables": [
            {
                "page_number": 1,
                "table_id": "p1-t1",
                "columns": columns,
                "rows": [{"order": 0, "cells": cells}],
            }
        ],
    }


def document(*, gold: GoldAnnotation | None = None, result: dict | None = None) -> DocumentFields:
    return DocumentFields(
        document_id="doc-1",
        layout_family_id="layout-a",
        gold=gold or annotation(),
        actual=result or actual(),
    )


def test_field_quality_passes_normalized_exact_values() -> None:
    report = evaluate_with_layout_slices((document(),))

    assert report["passed"] is True
    assert report["aggregate"]["canonical_fields"]["net_amount"]["accuracy"] == 1
    assert report["aggregate"]["source_column_families"]["net_amount"]["cells"]["passed"]


def test_field_quality_fails_wrong_value_for_canonical_and_source_cell() -> None:
    report = evaluate_field_quality((document(result=actual(amount="96.00")),))

    assert report["passed"] is False
    assert "canonical:net_amount" in report["blocking_reasons"]
    assert "source:net_amount:cells" in report["blocking_reasons"]


def test_unreadable_gold_cell_is_excluded_without_hiding_canonical_error() -> None:
    report = evaluate_field_quality(
        (document(gold=annotation(unreadable=True), result=actual(amount="96.00")),)
    )

    net_source = report["source_column_families"]["net_amount"]
    assert net_source["excluded_unreadable_cells"] == 1
    assert net_source["cells"]["observed"] is False
    assert "canonical:net_amount" in report["blocking_reasons"]


def test_hallucinated_dynamic_column_fails() -> None:
    report = evaluate_field_quality((document(result=actual(extra_column=True)),))

    assert "source:invented:header" in report["blocking_reasons"]
    assert "source:invented:cells" in report["blocking_reasons"]


def test_threshold_is_strictly_greater_than_95_percent() -> None:
    documents = []
    for index in range(20):
        gold = annotation().model_copy(
            update={"document_sha256": f"{index:064x}"},
        )
        result = actual(amount="96.00" if index == 0 else "95.00")
        documents.append(
            DocumentFields(
                document_id=f"doc-{index}",
                layout_family_id="layout-a",
                gold=gold,
                actual=result,
            )
        )

    report = evaluate_field_quality(documents)
    metric = report["canonical_fields"]["net_amount"]
    assert metric["accuracy"] == 0.95
    assert metric["passed"] is False


def test_field_normalizers_are_typed() -> None:
    assert normalize_value("service_date", "31 Aug 2026") == "2026-08-31"
    assert normalize_value("request_no", "A-10 / 2") == "a102"
    assert normalize_value("net_amount", "1,234.5") == "1234.50"
