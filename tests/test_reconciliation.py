from __future__ import annotations

from decimal import Decimal
from typing import Any

from gmoney.extraction.reconciliation import match_outcome, reconcile


def _row(
    row_id: str,
    amount: str | None,
    *,
    role: str = "detail",
    description: str = "Item",
    section: str | None = None,
    page: int = 1,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "page_number": page,
        "role": role,
        "review_disposition": "accepted",
        "description": description,
        "section": section,
        "net_amount": amount,
        **fields,
    }


def _total(amount: str, *, label: str = "Total Bill Amount", kind: str = "bill_total") -> dict:
    return {
        "amount": amount,
        "amount_raw": amount,
        "label": label,
        "kind": kind,
        "scope": "document",
        "page_number": 1,
    }


def _table(
    table_id: str, page: int, entries: list[tuple[str, ...]], *, net_column: bool = True
) -> dict[str, Any]:
    """entries: ("row", canonical_id) or ("printed", label, amount)."""
    columns = [
        {"id": "c0", "label": "Description", "order": 0, "canonical_field": "description"},
        {
            "id": "c1",
            "label": "Amount",
            "order": 1,
            "canonical_field": "net_amount" if net_column else None,
        },
    ]
    rows = []
    for order, entry in enumerate(entries):
        if entry[0] == "row":
            rows.append(
                {
                    "id": f"{table_id}-s{order}",
                    "order": order,
                    "canonical_row_id": entry[1],
                    "cells": [{"column_id": "c0", "raw_value": entry[1]}],
                }
            )
        else:
            rows.append(
                {
                    "id": f"{table_id}-s{order}",
                    "order": order,
                    "canonical_row_id": None,
                    "cells": [
                        {"column_id": "c0", "raw_value": entry[1]},
                        {"column_id": "c1", "raw_value": entry[2]},
                    ],
                }
            )
    return {
        "id": table_id,
        "table_id": table_id,
        "page_number": page,
        "columns": columns,
        "rows": rows,
    }


def _failed(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in report["checks"] if check["outcome"] == "fail" and check["blocking"]]


def test_missing_sections_flag_grand_total_and_section_with_difference() -> None:
    rows = [
        _row("ph1", "50000.00", section="Pharmacy", page=3),
        _row("ph2", "23161.72", section="Pharmacy", page=3),
        _row("pr1", "700.00", section="Procedure", page=2),
    ]
    result = {
        "document_total": _total("94140.00"),
        "rows": rows,
        "source_tables": [
            _table("p2-t1", 2, [("row", "pr1"), ("printed", "Total Amount", "890.00")]),
            _table(
                "p3-t1",
                3,
                [("row", "ph1"), ("row", "ph2"), ("printed", "Sub Total", "73,161.72")],
            ),
        ],
    }
    report = reconcile(result)
    assert report["status"] == "flagged"
    failed = {check["id"]: check for check in _failed(report)}
    assert failed["C1"]["expected"] == "94140.00"
    assert failed["C1"]["actual"] == "73861.72"
    assert failed["C1"]["difference"] == "-20278.28"
    section = failed["C2:p2:p2-t1:p2-t1-s1"]
    assert section["section"] == "Procedure"
    assert section["expected"] == "890.00"
    assert section["difference"] == "-190.00"
    passed = next(check for check in report["checks"] if check["id"] == "C2:p3:p3-t1:p3-t1-s2")
    assert passed["outcome"] == "pass"
    assert any("Procedure" in reason and "-190.00" in reason for reason in report["reasons"])


def test_return_emitted_as_positive_charge_is_flagged() -> None:
    rows = [
        _row("a", "1000.00", section="Item Issues"),
        _row("bifilac", "126.56", section="Return Item", description="Bifilac"),
    ]
    report = reconcile({"document_total": _total("873.44"), "rows": rows})
    assert report["status"] == "flagged"
    assert _failed(report)[0]["difference"] == "253.12"

    rows[1]["role"] = "refund"
    assert reconcile({"document_total": _total("873.44"), "rows": rows})["status"] == "verified"


def test_footer_bill_number_emitted_as_charge_is_flagged() -> None:
    rows = [_row("a", "500.00"), _row("junk", "4172", description="Bill No", page=4)]
    report = reconcile({"document_total": _total("500.00"), "rows": rows})
    assert report["status"] == "flagged"
    assert _failed(report)[0]["difference"] == "4172.00"


def test_package_bill_with_unpriced_inclusions_is_verified() -> None:
    rows = [
        _row("pkg", "11457.00", role="category_rollup", description="Package Charges"),
        _row("pkg-repeat", "11457.00", role="category_rollup", description="Package Charges"),
        *[
            _row(f"inc{index}", None, role="informational", description=f"Inclusion {index}")
            for index in range(88)
        ],
    ]
    report = reconcile({"document_total": _total("11457.00"), "rows": rows})
    assert report["status"] == "verified"
    assert report["checks"][0]["evidence"] == {"basis": "package_rollup"}


def test_whole_rupee_rounding_passes_but_fractional_gap_fails() -> None:
    rows = [_row("a", "94000.18"), _row("b", "140.00")]
    report = reconcile({"document_total": _total("94140.00"), "rows": rows})
    assert report["status"] == "verified"
    assert report["checks"][0]["outcome"] == "rounded"
    assert (
        reconcile({"document_total": _total("69335.00"), "rows": [_row("a", "69335.18")]})[
            "checks"
        ][0]["outcome"]
        == "rounded"
    )

    rows = [_row("a", "94000.50"), _row("b", "140.00")]
    report = reconcile({"document_total": _total("94140.25"), "rows": rows})
    assert report["status"] == "flagged"
    assert report["checks"][0]["outcome"] == "fail"
    assert match_outcome(Decimal("10"), Decimal("10.99")) == "rounded"
    assert match_outcome(Decimal("10"), Decimal("11")) == "fail"


def test_nested_issue_return_and_sub_totals_reconcile() -> None:
    rows = [
        _row("i1", "100.00", section="Item Issues"),
        _row("i2", "50.00", section="Item Issues"),
        _row("r1", "-20.00", role="refund", section="Item Returns"),
        _row("r2", "10.00", role="refund", section="Item Returns"),
    ]
    table = _table(
        "p1-t1",
        1,
        [
            ("row", "i1"),
            ("row", "i2"),
            ("printed", "Item Issues Total", "150.00"),
            ("row", "r1"),
            ("row", "r2"),
            ("printed", "Item Returns Total", "30.00"),
            ("printed", "Sub Total", "120.00"),
            ("printed", "Discount", "5.00"),
        ],
    )
    report = reconcile({"document_total": _total("120.00"), "rows": rows, "source_tables": [table]})
    section_checks = [check for check in report["checks"] if check["kind"] == "section_total"]
    assert [check["outcome"] for check in section_checks] == ["pass", "pass", "pass"]
    assert report["status"] == "verified"
    assert any(item["label"] == "discount" for item in report["reported"])


def test_subtotal_without_known_columns_matches_any_number() -> None:
    rows = [_row("a", "40.00"), _row("b", "60.00")]
    table = _table(
        "p1-t1",
        1,
        [("row", "a"), ("row", "b"), ("printed", "Total", "100.00")],
        net_column=False,
    )
    report = reconcile({"document_total": _total("100.00"), "rows": rows, "source_tables": [table]})
    assert report["status"] == "verified"


def test_missing_printed_total_is_unprovable() -> None:
    report = reconcile({"rows": [_row("a", "10.00")]})
    assert report["status"] == "unprovable"
    assert report["reasons"]


def test_rejected_rows_and_row_arithmetic_do_not_block() -> None:
    rows = [
        _row("a", "200.00", quantity="2", unit_price="150.00"),
        {**_row("junk", "4172"), "review_disposition": "rejected"},
    ]
    report = reconcile({"document_total": _total("200.00"), "rows": rows})
    assert report["status"] == "verified"
    arithmetic = next(check for check in report["checks"] if check["kind"] == "row_arithmetic")
    assert arithmetic["outcome"] == "fail"
    assert arithmetic["blocking"] is False


def test_summary_rollups_must_equal_grand_total() -> None:
    rows = [
        _row("s1", "500.00", role="category_rollup", description="Pharmacy"),
        _row("s2", "400.00", role="category_rollup", description="Room"),
        _row("d1", "500.00"),
        _row("d2", "500.00"),
    ]
    report = reconcile({"document_total": _total("1000.00"), "rows": rows})
    assert report["status"] == "flagged"
    summary = next(check for check in report["checks"] if check["id"] == "C3")
    assert summary["difference"] == "-100.00"
