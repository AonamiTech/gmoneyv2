from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

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
        # Printed in the page-1 summary and again above the inclusions on page 2.
        _row(
            "pkg",
            "11457.00",
            role="category_rollup",
            description="Package Charges",
            table_id="p1-t1",
        ),
        _row(
            "pkg-repeat",
            "11457.00",
            role="category_rollup",
            description="Package Charges",
            page=2,
            table_id="p2-t1",
        ),
        *[
            _row(f"inc{index}", None, role="informational", description=f"Inclusion {index}")
            for index in range(88)
        ],
    ]
    report = reconcile({"document_total": _total("11457.00"), "rows": rows})
    assert report["status"] == "verified"
    assert report["checks"][0]["evidence"] == {"basis": "distinct_category_rollups"}
    assert report["checks"][0]["actual"] == "11457.00"


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
    section_checks = [
        check
        for check in report["checks"]
        if check["kind"] in {"section_total", "section_aggregate"}
    ]
    assert [(check["kind"], check["outcome"]) for check in section_checks] == [
        ("section_total", "pass"),
        ("section_total", "pass"),
        ("section_aggregate", "pass"),
    ]
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


def test_package_rollup_does_not_hide_a_separate_charge() -> None:
    rows = [
        _row("pkg", "100.00", role="category_rollup", description="Package Charges"),
        _row("extra", "50.00", role="category_rollup", description="Pharmacy Outside Package"),
    ]
    report = reconcile({"document_total": _total("100.00"), "rows": rows})
    assert report["status"] == "flagged"
    assert report["checks"][0]["actual"] == "150.00"
    assert report["checks"][0]["difference"] == "50.00"

    rows[0]["table_id"] = "p1-t1"
    rows[1]["table_id"] = "p1-t1"
    rows.append(
        _row(
            "pkg-repeat",
            "100.00",
            role="category_rollup",
            description="PACKAGE charges",
            page=2,
            table_id="p2-t1",
        )
    )
    report = reconcile({"document_total": _total("150.00"), "rows": rows})
    assert report["status"] == "verified"


def test_section_total_cannot_reuse_rows_of_a_previous_section() -> None:
    # Section A prints 50 for rows 100 + 50 (wrong), section B prints 80 for row 30
    # (wrong; 80 = 50 + 30 borrows A's last row).  The old back-search passed both.
    rows = [
        _row("a1", "100.00", section="Room"),
        _row("a2", "50.00", section="Room"),
        _row("b1", "30.00", section="Pharmacy"),
    ]
    table = _table(
        "p1-t1",
        1,
        [
            ("row", "a1"),
            ("row", "a2"),
            ("printed", "Sub Total", "50.00"),
            ("row", "b1"),
            ("printed", "Sub Total", "80.00"),
        ],
    )
    report = reconcile({"document_total": _total("180.00"), "rows": rows, "source_tables": [table]})
    sections = [check for check in report["checks"] if check["kind"] == "section_total"]
    assert [(check["outcome"], check["expected"], check["actual"]) for check in sections] == [
        ("fail", "50.00", "150.00"),
        ("fail", "80.00", "30.00"),
    ]
    assert sections[1]["section"] == "Pharmacy"
    assert report["status"] == "flagged"


def test_section_total_may_start_at_a_section_heading_but_not_mid_section() -> None:
    rows = [
        _row("a1", "100.00"),
        _row("b1", "30.00"),
        _row("b2", "20.00"),
    ]
    entries: list[tuple[str, ...]] = [
        ("row", "a1"),
        ("printed", "Pharmacy Charges", ""),
        ("row", "b1"),
        ("row", "b2"),
        ("printed", "Total", "50.00"),
    ]
    table = _table("p1-t1", 1, entries)
    table["rows"][1]["cells"] = [{"column_id": "c0", "raw_value": "Pharmacy Charges"}]
    report = reconcile({"document_total": _total("150.00"), "rows": rows, "source_tables": [table]})
    section = next(check for check in report["checks"] if check["kind"] == "section_total")
    assert section["outcome"] == "pass"
    assert section["evidence"]["run_row_ids"] == ["b1", "b2"]

    # Without the heading, a printed 20 must not match the trailing row alone.
    entries = [("row", "b1"), ("row", "b2"), ("printed", "Total", "20.00")]
    report = reconcile(
        {
            "document_total": _total("50.00"),
            "rows": rows[1:],
            "source_tables": [_table("p1-t1", 1, entries)],
        }
    )
    section = next(check for check in report["checks"] if check["kind"] == "section_total")
    assert section["outcome"] == "fail"


def test_aggregate_total_uses_only_the_preceding_closed_sections() -> None:
    rows = [
        _row("i1", "100.00", section="Item Issues"),
        _row("r1", "-20.00", role="refund", section="Item Returns"),
        _row("x1", "40.00", section="Room"),
    ]
    table = _table(
        "p1-t1",
        1,
        [
            ("row", "i1"),
            ("printed", "Item Issues Total", "100.00"),
            ("row", "r1"),
            ("printed", "Item Returns Total", "20.00"),
            ("printed", "Sub Total", "80.00"),
            ("row", "x1"),
            ("printed", "Room Total", "40.00"),
            ("printed", "Sub Total", "999.00"),
        ],
    )
    report = reconcile({"document_total": _total("120.00"), "rows": rows, "source_tables": [table]})
    outcomes = [
        (check["kind"], check["outcome"])
        for check in report["checks"]
        if check["kind"].startswith("section")
    ]
    assert outcomes == [
        ("section_total", "pass"),
        ("section_total", "pass"),
        ("section_aggregate", "pass"),
        ("section_total", "pass"),
        ("section_aggregate", "fail"),
    ]
    assert report["checks"][0]["outcome"] == "pass"
    assert report["status"] == "flagged"


def test_package_relabelled_on_its_detail_page_counts_once() -> None:
    rows = [
        _row("s", "50000.00", role="category_rollup", description="Package Charges", table_id="t1"),
        _row(
            "d",
            "50000.00",
            role="category_rollup",
            description="Knee Replacement Package",
            page=2,
            table_id="t2",
        ),
    ]
    report = reconcile({"document_total": _total("50000.00"), "rows": rows})
    assert report["status"] == "verified"


def test_identical_charges_in_one_table_both_count() -> None:
    rows = [
        _row("p", "40000.00", role="category_rollup", description="Package Charges", table_id="t1"),
        _row("c1", "500.00", role="category_rollup", description="Consultation", table_id="t1"),
        _row("c2", "500.00", role="category_rollup", description="Consultation", table_id="t1"),
    ]
    report = reconcile({"document_total": _total("41000.00"), "rows": rows})
    assert report["status"] == "verified"
    assert report["checks"][0]["actual"] == "41000.00"


def test_package_plus_priced_extras_outside_the_package_reconciles() -> None:
    rows = [
        _row("pkg", "50000.00", role="category_rollup", description="Package", table_id="t1"),
        _row("x1", "3000.00", description="Implant outside package", table_id="t2"),
        _row("x2", "2000.00", description="Pharmacy outside package", table_id="t2"),
    ]
    report = reconcile({"document_total": _total("55000.00"), "rows": rows})
    assert report["status"] == "verified"
    assert report["checks"][0]["evidence"] == {"basis": "granular_plus_rollups"}
    assert not any(check["id"] == "C3" for check in report["checks"])
    # A summary that does not match is still caught: rows 5000 + roll-up 50000 vs 60000.
    assert reconcile({"document_total": _total("60000.00"), "rows": rows})["status"] == "flagged"


def _multi_page_pharmacy(page_one_label: str, page_two_start: list[tuple[str, ...]]):
    rows = [
        _row("a", "100.00", section="Pharmacy"),
        _row("b", "200.00", section="Pharmacy"),
        _row("c", "300.00", section="Pharmacy", page=2),
        _row("d", "100.00", section="Pharmacy", page=2),
        _row("r1", "500.00", section="Room", page=3),
    ]
    tables = [
        _table("p1-t1", 1, [("row", "a"), ("row", "b"), ("printed", page_one_label, "300.00")]),
        _table(
            "p2-t1",
            2,
            [*page_two_start, ("row", "c"), ("row", "d"), ("printed", "Pharmacy Total", "700.00")],
        ),
        _table("p3-t1", 3, [("row", "r1"), ("printed", "Room Total", "500.00")]),
    ]
    return {"document_total": _total("1200.00"), "rows": rows, "source_tables": tables}


@pytest.mark.parametrize(
    ("page_one_label", "page_two_start"),
    [
        ("Sub Total", []),
        ("Page Total", [("printed", "Brought Forward", "300.00")]),
    ],
)
def test_section_spanning_pages_with_page_sub_totals_reconciles(
    page_one_label: str, page_two_start: list[tuple[str, ...]]
) -> None:
    report = reconcile(_multi_page_pharmacy(page_one_label, page_two_start))
    assert report["status"] == "verified", report["reasons"]
    kinds = [check["kind"] for check in report["checks"] if check["kind"].startswith("section")]
    assert kinds == ["section_total", "section_total_continued", "section_total"]


def test_wrong_cross_page_section_total_is_still_flagged() -> None:
    result = _multi_page_pharmacy("Sub Total", [])
    result["source_tables"][1]["rows"][-1]["cells"][1]["raw_value"] = "650.00"
    report = reconcile(result)
    assert report["status"] == "flagged"
    failed = next(check for check in report["checks"] if check["outcome"] == "fail")
    assert (failed["expected"], failed["actual"], failed["section"]) == (
        "650.00",
        "400.00",
        "Pharmacy",
    )


def test_discount_between_sub_total_and_net_total_reconciles() -> None:
    rows = [_row("a", "600.00", section="Room"), _row("b", "400.00", section="Room")]
    table = _table(
        "p1-t1",
        1,
        [
            ("row", "a"),
            ("row", "b"),
            ("printed", "Sub Total", "1000.00"),
            ("printed", "Less Discount", "100.00"),
            ("printed", "Net Total", "900.00"),
        ],
    )
    report = reconcile(
        {"document_total": _total("1000.00"), "rows": rows, "source_tables": [table]}
    )
    outcomes = [
        (check["kind"], check["outcome"])
        for check in report["checks"]
        if check["kind"].startswith("section")
    ]
    assert outcomes == [("section_total", "pass"), ("section_aggregate", "pass")]
    assert report["status"] == "verified"


def test_cross_page_total_cannot_borrow_another_sections_rows() -> None:
    rows = [
        _row("a", "100.00", section="Room"),
        _row("b1", "30.00", section="Pharmacy", page=2),
        _row("a_dup", "100.00", section="Room", page=3),
    ]
    tables = [
        _table("t1", 1, [("row", "a"), ("printed", "Room Total", "100.00")]),
        _table("t2", 2, [("row", "b1"), ("printed", "Pharmacy Total", "130.00")]),
        _table("t3", 3, [("row", "a_dup")]),
    ]
    report = reconcile({"document_total": _total("230.00"), "rows": rows, "source_tables": tables})
    assert report["status"] == "flagged"
    pharmacy = next(check for check in report["checks"] if check["label"] == "pharmacy total")
    assert (pharmacy["kind"], pharmacy["expected"], pharmacy["actual"]) == (
        "section_total",
        "130.00",
        "30.00",
    )


def test_gross_sub_total_after_an_in_between_discount_reconciles() -> None:
    rows = [
        _row("a", "1000.00", section="Room"),
        _row("b", "500.00", section="Pharmacy"),
        _row("z", "200.00", section="Lab", page=2),
    ]
    tables = [
        _table(
            "t",
            1,
            [
                ("row", "a"),
                ("printed", "Room Total", "1000.00"),
                ("printed", "Discount", "100.00"),
                ("row", "b"),
                ("printed", "Pharmacy Total", "500.00"),
                ("printed", "Sub Total", "1500.00"),
            ],
        ),
        _table("lab", 2, [("row", "z"), ("printed", "Lab Total", "200.00")]),
    ]
    report = reconcile({"document_total": _total("1700.00"), "rows": rows, "source_tables": tables})
    assert report["status"] == "verified", report["reasons"]


def test_summary_rollup_cannot_stand_in_for_dropped_detail_rows() -> None:
    # Summary lists Pharmacy 300 and Room 700; the Room detail page was dropped, so
    # granular rows are only the Pharmacy 300.  Rows + Room roll-up would equal 1000.
    rows = [
        _row("s1", "300.00", role="category_rollup", description="Pharmacy", section="Pharmacy"),
        _row("s2", "700.00", role="category_rollup", description="Room", section="Room"),
        _row("d1", "300.00", section="Pharmacy", page=2),
    ]
    report = reconcile({"document_total": _total("1000.00"), "rows": rows})
    assert report["status"] == "flagged"


def test_positive_returns_total_matches_printed_positive_refunds_only() -> None:
    from gmoney.extraction.reconciliation import return_total_targets

    reader_refunds = [{"role": "refund", "net_amount": "126.56"}]
    heuristic_refunds = [{"role": "refund", "net_amount": "-126.56"}]
    assert Decimal("-126.56") in return_total_targets(Decimal("126.56"), reader_refunds, -1)
    # Heuristic behaviour is unchanged: only the printed value itself.
    assert return_total_targets(Decimal("126.56"), heuristic_refunds, -1) == {Decimal("126.56")}
    assert return_total_targets(Decimal("500.00"), reader_refunds, 1) == {Decimal("500.00")}
