from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole
from gmoney.extraction.teleocr_reader import (
    charge_rows,
    drop_rows_already_read,
    needs_full_page_read,
    printed_total_rows,
    read_otsl_rows,
    uncovered_band,
)

FIXTURES = Path(__file__).parent / "fixtures" / "teleocr"


def _otsl(*rows: list[str]) -> str:
    return "".join(
        "".join(f"<fcel>{cell}" if cell else "<ecel>" for cell in row) + "<nl>" for row in rows
    )


def test_recorded_teleocr_procedure_table_yields_full_names_amounts_and_printed_total() -> None:
    content = (FIXTURES / "bill6_page2_procedure.otsl").read_text()
    rows = read_otsl_rows(content)

    details = charge_rows(rows)
    assert [row.description for row in details] == [
        "Chemotherapy Pushing Charges (Long CT Per Day)",
        "Chemotherapy Pushing Charges (Ultra Short)",
    ]
    assert [row.amount for row in details] == [Decimal("700.00"), Decimal("190.00")]
    assert [row.rate for row in details] == [Decimal("700.00"), Decimal("190.00")]
    assert [row.quantity for row in details] == [Decimal("1"), Decimal("1")]
    assert [row.service_date for row in details] == ["10/02/2026", "11/02/2026"]
    assert {row.section for row in details} == {"Bed Procedure"}
    assert all(row.role is RowRole.DETAIL for row in details)

    totals = printed_total_rows(rows)
    assert [(row.description, row.amount) for row in totals] == [
        ("Total Amount", Decimal("890.00"))
    ]


def test_multiline_drug_descriptions_are_merged_both_ways() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Item Name", "Qty", "Rate", "Amount"],
            ["Advamab", "", "", ""],
            ["100mg Inj", "1", "25000.00", "25000.00"],
            ["Inj Neukine", "2", "1500.00", "3000.00"],
            ["300 Mcg", "", "", ""],
            ["Adripeg 50mg", "1", "4000.00", "4000.00"],
        )
    )
    assert [(row.description, row.amount) for row in charge_rows(rows)] == [
        ("Advamab 100mg Inj", Decimal("25000.00")),
        ("Inj Neukine 300 Mcg", Decimal("3000.00")),
        ("Adripeg 50mg", Decimal("4000.00")),
    ]
    assert "reader_multiline_merged" in charge_rows(rows)[0].validation_flags


def test_issued_date_group_rows_become_service_date_context() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Description", "Qty", "Amount"],
            ["Issued Date : 10/02/2026", "", ""],
            ["Pantop 40", "2", "120.00"],
            ["Issued Date", "11/02/2026", ""],
            ["Dolo 650", "1", "30.00"],
        )
    )
    details = charge_rows(rows)
    assert [(row.description, row.service_date) for row in details] == [
        ("Pantop 40", "10/02/2026"),
        ("Dolo 650", "11/02/2026"),
    ]
    assert len(rows) == 2


def test_rows_under_return_heading_are_refunds_with_printed_amount() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Item Name", "Qty", "Amount"],
            ["Item Issues", "", ""],
            ["Taxol 100mg", "1", "5000.00"],
            ["Item Issues Total", "", "5000.00"],
            ["Return Item", "", ""],
            ["Bifilac", "1", "126.56"],
            ["Item Returns Total", "", "126.56"],
        )
    )
    details = charge_rows(rows)
    assert [(row.role, row.description, row.amount) for row in details] == [
        (RowRole.DETAIL, "Taxol 100mg", Decimal("5000.00")),
        (RowRole.REFUND, "Bifilac", Decimal("126.56")),
    ]
    assert details[1].section == "Return Item"
    assert [row.description for row in printed_total_rows(rows)] == [
        "Item Issues Total",
        "Item Returns Total",
    ]


def test_item_name_column_wins_over_code_column() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Sr", "Particulars", "", "Qty", "Amount"],
            ["1", "SER0943548", "GRAM STAIN", "1", "250.00"],
            ["2", "SER0943611", "CBC", "1", "400.00"],
        )
    )
    assert [(row.description, row.service_code) for row in charge_rows(rows)] == [
        ("GRAM STAIN", "SER0943548"),
        ("CBC", "SER0943611"),
    ]


def test_header_and_footer_identifiers_are_never_charges() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Service Name", "Qty", "Amount"],
            ["Registration Charges", "1", "500"],
            ["Bill No.", "", "4172"],
            ["UHID", "", "88231"],
        )
    )
    assert [(row.description, row.amount) for row in charge_rows(rows)] == [
        ("Registration Charges", Decimal("500"))
    ]
    metadata = [row for row in rows if row.role is RowRole.METADATA]
    assert len(metadata) == 2
    assert all("reader_identifier_row" in row.validation_flags for row in metadata)


def test_settlement_lines_are_not_charges() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Description", "Amount"],
            ["Room Rent", "1000.00"],
            ["Advance Received", "500.00"],
            ["Discount", "50.00"],
        )
    )
    assert [row.description for row in charge_rows(rows)] == ["Room Rent"]
    assert {row.role for row in rows if row.description != "Room Rent"} == {RowRole.PAYMENT}


def test_unparseable_reply_yields_no_rows() -> None:
    assert read_otsl_rows("The image shows a hospital bill.") == ()


def _token(token_id: str, text: str, x: float, y: float) -> OcrToken:
    return OcrToken(
        token_id=token_id,
        page_number=1,
        text=text,
        polygon=Polygon(
            points=(
                Point(x=x, y=y),
                Point(x=x + 40, y=y),
                Point(x=x + 40, y=y + 10),
                Point(x=x, y=y + 10),
            )
        ),
        artifact_sha256="a" * 64,
        confidence=0.99,
        model_name="PP-OCRv6",
        model_version="test",
    )


def test_missed_table_guard_triggers_on_amounts_outside_detected_tables() -> None:
    tokens = (
        _token("t1", "700.00", 400, 120),
        _token("t2", "Bed Charges", 20, 500),
        _token("t3", "11,100.00", 400, 520),
        _token("t4", "650.00", 400, 700),
        _token("t5", "Page 2", 20, 900),
    )
    table_boxes = [(0, 100, 600, 200)]
    assert needs_full_page_read(tokens, table_boxes)
    assert uncovered_band(tokens, table_boxes, width=600, height=1000) == (0, 480, 600, 750)
    assert not needs_full_page_read(tokens, [(0, 100, 600, 800)])
    assert not needs_full_page_read(tokens[:2], table_boxes)


def test_full_page_read_is_deduped_against_rows_already_read() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Description", "Amount"],
            ["Chemotherapy Pushing Charges (Ultra Short)", "190.00"],
            ["Bed Charges", "11100.00"],
        )
    )
    kept = drop_rows_already_read(rows, [("chemotherapy pushing charges (ultra short)", "190.00")])
    assert [row.description for row in kept] == ["Bed Charges"]
