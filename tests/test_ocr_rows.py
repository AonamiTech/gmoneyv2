from decimal import Decimal

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import (
    RowRole,
    SourceCell,
    SourceColumn,
    TableType,
)
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_rows import (
    _clean_description,
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
)
from gmoney.extraction.offline import (
    _link_source_tables,
    _populate_grounded_service_date_cell,
    _recover_grounded_service_dates,
    _recovery_prior_schemas,
)
from gmoney.extraction.rows import CandidateLedgerRow


def token(index: int, text: str, box: tuple[float, float, float, float]) -> OcrToken:
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
        artifact_sha256="a" * 64,
        model_name="fixture",
        model_version="1",
    )


def test_clean_description_collapses_an_exact_ocr_phrase_echo() -> None:
    description, service_date, request_no = _clean_description(
        "Emeset 2 Ml Inj Emeset 2 Ml Inj"
    )

    assert description == "Emeset 2 Ml Inj"
    assert service_date is None
    assert request_no is None


@pytest.mark.parametrize(
    ("printed_description", "expected"),
    (
        (
            "Becosules Cap 20S 25230730 Becosules Cap",
            "Becosules Cap 20S 25230730",
        ),
        ("Neo | Care Pad Neo I Care Pad", "Neo | Care Pad"),
    ),
)
def test_clean_description_collapses_a_grounded_ocr_suffix_echo(
    printed_description: str,
    expected: str,
) -> None:
    description, service_date, request_no = _clean_description(
        printed_description
    )

    assert description == expected
    assert service_date is None
    assert request_no is None


@pytest.mark.parametrize(
    "printed_description",
    (
        "Vitamin C and D Vitamin C",
        "Type I Type II",
    ),
)
def test_clean_description_preserves_legitimate_repeated_phrases(
    printed_description: str,
) -> None:
    description, service_date, request_no = _clean_description(
        printed_description
    )

    assert description == printed_description
    assert service_date is None
    assert request_no is None


def skewed_token(
    index: int,
    text: str,
    box: tuple[float, float, float, float],
    slope: float,
) -> OcrToken:
    left, top, right, bottom = box
    return token(index, text, box).model_copy(
        update={
            "polygon": Polygon(
                points=(
                    Point(x=left, y=top + slope * left),
                    Point(x=right, y=top + slope * right),
                    Point(x=right, y=bottom + slope * right),
                    Point(x=left, y=bottom + slope * left),
                )
            )
        }
    )


def rotated_token(
    index: int,
    text: str,
    box: tuple[float, float, float, float],
    rise: float,
) -> OcrToken:
    left, top, right, bottom = box
    return token(index, text, box).model_copy(
        update={
            "polygon": Polygon(
                points=(
                    Point(x=left, y=top),
                    Point(x=right, y=top + rise),
                    Point(x=right, y=bottom + rise),
                    Point(x=left, y=bottom),
                )
            )
        }
    )


def test_reconstructs_split_description_and_uses_rightmost_amount() -> None:
    tokens = (
        token(0, "Patient", (10, 5, 80, 15)),
        token(1, "Description", (100, 30, 250, 45)),
        token(2, "Qty", (500, 30, 550, 45)),
        token(3, "M.R.P.", (650, 30, 720, 45)),
        token(4, "Amount", (850, 30, 950, 45)),
        token(5, "Registration", (100, 70, 280, 85)),
        token(6, "1", (510, 100, 530, 115)),
        token(7, "250.00", (650, 100, 720, 115)),
        token(8, "500.00", (850, 100, 930, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert result.diagnostics["header_found"] is True
    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Registration"
    assert result.rows[0].candidate.amount == Decimal("500.00")
    assert result.rows[0].field_token_ids["amount"] == ("token-8",)


def test_multiline_description_cells_use_top_to_bottom_reading_order() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(
            2,
            "Dressing and Plaster Charges-EVALUATION",
            (100, 70, 500, 90),
        ),
        token(3, "UNDER ANAESTHRSIA", (100, 78, 340, 98)),
        token(4, "500.00", (870, 70, 940, 90)),
        token(5, "Procedure Charges-URINARY", (100, 115, 430, 135)),
        token(6, "CATHETERIZATION", (100, 123, 300, 143)),
        token(7, "750.00", (870, 115, 940, 135)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 160),
    )

    expected = [
        "Dressing and Plaster Charges-EVALUATION UNDER ANAESTHRSIA",
        "Procedure Charges-URINARY CATHETERIZATION",
    ]
    assert [row.candidate.description for row in result.rows] == expected
    description_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "description"
    )
    assert [
        next(
            cell.raw_value
            for cell in row.cells
            if cell.column_id == description_column.id
        )
        for row in result.source_tables[0].rows
    ] == expected


def test_upright_skew_is_deskewed_before_row_grouping() -> None:
    slope = -0.025
    tokens = (
        skewed_token(0, "Service Name", (100, 100, 300, 125), slope),
        skewed_token(1, "Patient Amount", (600, 100, 740, 125), slope),
        skewed_token(2, "Company Amount", (750, 100, 900, 125), slope),
        skewed_token(3, "Total Amount", (910, 100, 1000, 125), slope),
        skewed_token(4, "BED CHARGES", (100, 150, 300, 175), slope),
        skewed_token(5, "0.00", (600, 150, 680, 175), slope),
        skewed_token(6, "16200.00", (750, 150, 850, 175), slope),
        skewed_token(7, "16200.00", (910, 150, 1000, 175), slope),
        skewed_token(8, "LABORATORY", (100, 200, 300, 225), slope),
        skewed_token(9, "0.00", (600, 200, 680, 225), slope),
        skewed_token(10, "6422.00", (750, 200, 850, 225), slope),
        skewed_token(11, "6422.00", (910, 200, 1000, 225), slope),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 50, 1020, 260),
    )
    assert [row.candidate.description for row in result.rows] == [
        "BED CHARGES",
        "LABORATORY",
    ]
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("16200.00"),
        Decimal("6422.00"),
    ]
    assert result.schema is not None
    assert result.schema.table_type is TableType.CATEGORY_SUMMARY
    assert result.diagnostics["deskew_slope"] == slope


def test_single_laboratory_charge_does_not_reclassify_general_ledger() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Laboratory Charges", (100, 70, 330, 85)),
        token(3, "2,400.00", (870, 70, 950, 85)),
        token(4, "Procedure Charges", (100, 100, 330, 115)),
        token(5, "4,500.00", (870, 100, 950, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )

    assert result.schema is not None
    assert result.schema.table_type is TableType.ITEM_LEDGER
    assert result.source_tables[0].table_type is TableType.ITEM_LEDGER
    assert [row.candidate.table_type for row in result.rows] == [
        TableType.ITEM_LEDGER,
        TableType.ITEM_LEDGER,
    ]
    assert result.rows[0].candidate.category == "laboratory"


def test_merged_discount_and_amount_token_uses_rightmost_value() -> None:
    tokens = (
        token(0, "Description", (100, 30, 250, 45)),
        token(1, "Rate", (600, 30, 680, 45)),
        token(2, "Qty", (700, 30, 760, 45)),
        token(3, "Discount Amount", (780, 30, 950, 45)),
        token(4, "PTCA", (100, 70, 220, 85)),
        token(5, "250000.00", (600, 70, 690, 85)),
        token(6, "1", (710, 70, 730, 85)),
        token(7, "0.00 250000.00", (780, 70, 950, 85)),
        token(8, "CAG", (100, 100, 220, 115)),
        token(9, "25000.00", (600, 100, 690, 115)),
        token(10, "1", (710, 100, 730, 115)),
        token(11, "0.00 25000.00", (780, 100, 950, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("250000.00"),
        Decimal("25000.00"),
    ]
    assert result.rows[0].field_token_ids["amount"] == ("token-7",)


def test_concatenated_discount_and_amount_keeps_full_currency_value() -> None:
    tokens = (
        token(0, "Description", (100, 30, 250, 45)),
        token(1, "Rate", (600, 30, 680, 45)),
        token(2, "Qty", (700, 30, 760, 45)),
        token(3, "Discount Amount", (780, 30, 950, 45)),
        token(4, "SYNERGY STENT", (100, 70, 300, 85)),
        token(5, "41145.00", (600, 70, 690, 85)),
        token(6, "1", (710, 70, 730, 85)),
        token(7, "0.0041145.00", (780, 70, 950, 85)),
        token(8, "SECOND ROW", (100, 100, 300, 115)),
        token(9, "100.00", (600, 100, 690, 115)),
        token(10, "1", (710, 100, 730, 115)),
        token(11, "0.00100.00", (780, 100, 950, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("41145.00"),
        Decimal("100.00"),
    ]


def test_schema_is_inherited_across_continuation_page() -> None:
    header_page = (
        token(0, "Description", (100, 30, 250, 45)),
        token(1, "Qty", (500, 30, 550, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Needle", (100, 70, 220, 85)),
        token(4, "1", (510, 70, 530, 85)),
        token(5, "10.00", (850, 70, 930, 85)),
    )
    first = reconstruct_ocr_rows(
        header_page,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    continuation = tuple(
        value.model_copy(update={"page_number": 2, "token_id": f"p2-{value.token_id}"})
        for value in (
            token(6, "Syringe", (100, 30, 220, 45)),
            token(7, "2", (510, 30, 530, 45)),
            token(8, "20.00", (850, 30, 930, 45)),
        )
    )
    second = reconstruct_ocr_rows(
        continuation,
        page_number=2,
        table_id="p2-t1",
        box=(80, 20, 980, 70),
        prior_schemas=(first.schema,) if first.schema else (),
    )
    assert second.diagnostics["schema_inherited"] is True
    assert second.rows[0].candidate.description == "Syringe"
    assert second.rows[0].candidate.amount == Decimal("20.00")


def test_compact_itemname_header_keeps_ledger_with_repeated_bill_metadata() -> None:
    tokens = (
        token(0, "Bill No. UHID Bill Date", (100, 20, 450, 35)),
        token(1, "Patient Name Age/Sex Category", (100, 45, 500, 60)),
        token(2, "Mobile Number Address", (100, 70, 400, 85)),
        token(3, "ItemName", (100, 100, 300, 115)),
        token(4, "Quantity", (500, 100, 580, 115)),
        token(5, "Net Amount", (850, 100, 950, 115)),
        token(6, "Diaper", (100, 130, 250, 145)),
        token(7, "41", (510, 130, 540, 145)),
        token(8, "2395.00", (850, 130, 930, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 170),
    )
    assert result.diagnostics["header_found"] is True
    assert result.schema is not None
    assert result.schema.table_type is TableType.ITEM_LEDGER
    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Diaper"
    assert result.rows[0].candidate.amount == Decimal("2395.00")


def test_compact_unitprice_header_preserves_rate_quantity_and_amount_columns() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "UnitPrice", (520, 30, 620, 45)),
        token(2, "Quantity", (650, 30, 750, 45)),
        token(3, "Amount", (850, 30, 950, 45)),
        token(4, "Room :408--A/C", (100, 70, 330, 85)),
        token(5, "3000.00", (530, 70, 610, 85)),
        token(6, "4.00", (680, 70, 730, 85)),
        token(7, "12000.00", (850, 70, 940, 85)),
        token(8, "Consultation", (100, 100, 300, 115)),
        token(9, "350.00", (530, 100, 610, 115)),
        token(10, "1.00", (680, 100, 730, 115)),
        token(11, "350.00", (850, 100, 930, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert result.schema is not None
    assert result.rows[0].candidate.rate == Decimal("3000.00")
    assert result.rows[0].candidate.quantity == Decimal("4.00")
    assert result.rows[0].candidate.amount == Decimal("12000.00")
    assert result.rows[1].candidate.rate == Decimal("350.00")
    assert result.rows[1].candidate.quantity == Decimal("1.00")


def test_amount_rs_unit_days_compound_header_preserves_rate_quantity_and_total() -> None:
    tokens = (
        token(0, "Sr.N", (50, 30, 90, 45)),
        token(1, "Particular", (100, 30, 420, 45)),
        token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(3, "Total", (880, 30, 970, 45)),
        token(4, "1", (50, 70, 70, 85)),
        token(5, "Consulting Charges", (100, 70, 360, 85)),
        token(6, "1,000.00", (620, 70, 700, 85)),
        token(7, "2", (760, 70, 780, 85)),
        token(8, "2,000.00", (890, 70, 960, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.rate == Decimal("1000.00")
    assert result.rows[0].candidate.quantity == Decimal("2")
    assert result.rows[0].candidate.amount == Decimal("2000.00")


def test_staggered_amount_rs_unit_days_header_keeps_rate_before_total() -> None:
    tokens = (
        token(0, "Total", (880, 20, 970, 35)),
        token(1, "Sr.N", (50, 45, 90, 60)),
        token(2, "Amount Rs. Unit/Days", (610, 45, 820, 60)),
        token(3, "Particular", (100, 70, 420, 85)),
        token(4, "1.", (50, 110, 70, 125)),
        token(5, "Registration", (100, 110, 360, 125)),
        token(6, "300.00", (620, 110, 700, 125)),
        token(7, "1", (760, 110, 780, 125)),
        token(8, "300.00", (890, 110, 960, 125)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 10, 980, 150),
    )

    assert result.rows[0].candidate.rate == Decimal("300.00")
    assert result.rows[0].candidate.quantity == Decimal("1")
    assert result.rows[0].candidate.amount == Decimal("300.00")
    assert [
        (column.label, column.canonical_field)
        for column in result.source_tables[0].columns
    ] == [
        ("Sr.N", None),
        ("Particular", "description"),
        ("Amount Rs.", "unit_price"),
        ("Unit/Days", "quantity"),
        ("Total", "net_amount"),
    ]


def test_amount_rs_unit_days_without_total_remains_amount_and_quantity() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 420, 45)),
        token(1, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(2, "Consulting Charges", (100, 70, 360, 85)),
        token(3, "1,000.00", (620, 70, 700, 85)),
        token(4, "2", (760, 70, 780, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )

    assert result.rows[0].candidate.rate is None
    assert result.rows[0].candidate.quantity == Decimal("2")
    assert result.rows[0].candidate.amount == Decimal("1000.00")
    assert any(
        column.canonical_field == "net_amount"
        for column in result.source_tables[0].columns
    )


def test_financial_row_with_blank_particular_uses_grounded_serial_description() -> None:
    tokens = (
        token(0, "Sr.N", (50, 30, 90, 45)),
        token(1, "Particular", (100, 30, 420, 45)),
        token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(3, "Total", (880, 30, 970, 45)),
        token(4, "0.", (50, 70, 70, 85)),
        token(5, "300.00", (620, 70, 700, 85)),
        token(6, "1", (760, 70, 780, 85)),
        token(7, "300.00", (890, 70, 960, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )

    assert len(result.rows) == 1
    candidate = result.rows[0].candidate
    assert candidate.description == "0."
    assert candidate.rate == Decimal("300.00")
    assert candidate.quantity == Decimal("1")
    assert candidate.amount == Decimal("300.00")
    assert "missing_printed_description" in candidate.validation_flags
    assert result.rows[0].field_token_ids["description"] == ("token-4",)


def test_financial_row_splits_merged_serial_and_description_across_boundary() -> None:
    tokens = (
        token(0, "#", (50, 30, 70, 45)),
        token(1, "Particulars", (100, 30, 300, 45)),
        token(2, "Batch", (400, 30, 470, 45)),
        token(3, "Expiry", (520, 30, 590, 45)),
        token(4, "Rate", (650, 30, 700, 45)),
        token(5, "Qty", (760, 30, 800, 45)),
        token(6, "Amount", (880, 30, 960, 45)),
        token(7, "5 IV SET", (50, 70, 195, 85)),
        token(8, "26D041", (400, 70, 470, 85)),
        token(9, "Mar-2031", (520, 70, 590, 85)),
        token(10, "340.00", (650, 70, 700, 85)),
        token(11, "1", (760, 70, 780, 85)),
        token(12, "340.00", (880, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)

    assert len(canonical) == 1
    assert canonical[0].description == "IV SET"
    assert canonical[0].net_amount == Decimal("340.00")
    assert canonical[0].field_evidence["description"][0].token_ids == ("token-7",)
    columns = {column.label: column for column in linked[0].columns}
    linked_row = linked[0].rows[0]
    cells = {cell.column_id: cell for cell in linked_row.cells}
    assert linked_row.canonical_row_id == str(canonical[0].id)
    assert cells[columns["#"].id].raw_value == "5"
    assert cells[columns["Particulars"].id].raw_value == "IV SET"
    assert {
        token_id
        for label in ("#", "Particulars")
        for item in cells[columns[label].id].evidence
        for token_id in item.token_ids
    } == {"token-7"}


def test_slanted_serial_descriptions_start_distinct_financial_rows() -> None:
    tokens = (
        token(0, "#", (50, 10, 70, 50)),
        token(1, "Particulars", (100, 10, 300, 50)),
        token(2, "Rate", (650, 10, 700, 50)),
        token(3, "Qty", (760, 10, 800, 50)),
        token(4, "Amount", (880, 10, 960, 50)),
        token(5, "53.30", (650, 50, 700, 90)),
        token(6, "1", (760, 50, 780, 90)),
        token(7, "53.30", (880, 50, 950, 90)),
        token(8, "1 ZEPOXIN INJ", (50, 55, 300, 95)),
        token(9, "2 ONDET 2ML", (50, 80, 280, 120)),
        token(10, "12.72", (650, 90, 700, 130)),
        token(11, "NNNN", (760, 90, 800, 130)),
        token(12, "25.44", (880, 90, 950, 130)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 0, 980, 140),
    )

    assert [
        (
            item.candidate.description,
            item.candidate.rate,
            item.candidate.quantity,
            item.candidate.amount,
        )
        for item in result.rows
    ] == [
        ("1 ZEPOXIN INJ", Decimal("53.30"), Decimal("1"), Decimal("53.30")),
        ("2 ONDET 2ML", Decimal("12.72"), Decimal("2"), Decimal("25.44")),
    ]
    assert "quantity_derived_from_rate_amount" in (
        result.rows[1].candidate.validation_flags
    )


def test_blank_quantity_is_derived_only_from_exact_line_arithmetic() -> None:
    tokens = (
        token(0, "Particulars", (100, 30, 420, 45)),
        token(1, "Rate", (650, 30, 700, 45)),
        token(2, "Qty", (760, 30, 800, 45)),
        token(3, "Amount", (880, 30, 960, 45)),
        token(4, "SYRINGE 20ML", (100, 70, 420, 85)),
        token(5, "28.00", (650, 70, 700, 85)),
        token(6, "56.00", (880, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )

    assert len(result.rows) == 1
    candidate = result.rows[0].candidate
    assert candidate.quantity == Decimal("2")
    assert "quantity_derived_from_rate_amount" in (
        candidate.validation_flags
    )
    assert set(result.rows[0].field_token_ids["quantity"]) == {
        "token-5",
        "token-6",
    }


def test_added_to_bill_footer_does_not_extend_last_description() -> None:
    tokens = (
        token(0, "#", (50, 30, 70, 45)),
        token(1, "Particulars", (100, 30, 420, 45)),
        token(2, "Rate", (650, 30, 700, 45)),
        token(3, "Qty", (760, 30, 800, 45)),
        token(4, "Amount", (880, 30, 960, 45)),
        token(5, "22 DECMAX 4MG TABLET", (50, 70, 420, 85)),
        token(6, "5.00", (650, 70, 700, 85)),
        token(7, "6", (760, 70, 780, 85)),
        token(8, "30.00", (880, 70, 950, 85)),
        token(9, "(Added to Bill)", (100, 100, 420, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 130),
    )

    assert [item.candidate.description for item in result.rows] == [
        "22 DECMAX 4MG TABLET"
    ]
    assert len(result.source_tables[0].rows) == 1
    description_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "description"
    )
    description_cell = next(
        cell
        for cell in result.source_tables[0].rows[0].cells
        if cell.column_id == description_column.id
    )
    assert description_cell.raw_value == "22 DECMAX 4MG TABLET"


def test_pharmacy_quantity_fused_into_amount_is_split_and_grounded() -> None:
    tokens = (
        token(0, "#", (50, 30, 70, 45)),
        token(1, "Particulars", (100, 30, 420, 45)),
        token(2, "Rate", (650, 30, 700, 45)),
        token(3, "Qty", (760, 30, 800, 45)),
        token(4, "Amount", (880, 30, 960, 45)),
        token(5, "15 FOSAPRINEON INJ", (50, 70, 420, 85)),
        token(6, "4,585.00", (650, 70, 720, 85)),
        token(7, "1 4,585.00", (760, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)

    assert len(canonical) == 1
    assert canonical[0].quantity == Decimal("1")
    assert canonical[0].unit_price == Decimal("4585.00")
    assert canonical[0].net_amount == Decimal("4585.00")
    columns = {
        column.canonical_field: column
        for column in linked[0].columns
        if column.canonical_field is not None
    }
    cells = {cell.column_id: cell for cell in linked[0].rows[0].cells}
    quantity = cells[columns["quantity"].id]
    amount = cells[columns["net_amount"].id]
    assert quantity.raw_value == "1"
    assert amount.raw_value == "4,585.00"
    assert {
        token_id
        for cell in (quantity, amount)
        for item in cell.evidence
        for token_id in item.token_ids
    } == {"token-7"}


def test_expiry_and_rate_merged_token_is_split_and_grounded() -> None:
    tokens = (
        token(0, "#", (50, 30, 70, 45)),
        token(1, "Particulars", (100, 30, 360, 45)),
        token(2, "Expiry", (480, 30, 560, 45)),
        token(3, "Rate", (650, 30, 700, 45)),
        token(4, "Qty", (760, 30, 800, 45)),
        token(5, "Amount", (880, 30, 960, 45)),
        token(6, "13 DOXODEL 50", (50, 70, 360, 85)),
        token(7, "Sep-2027 3,833.00", (480, 70, 700, 85)),
        token(8, "2", (760, 70, 780, 85)),
        token(9, "7,666.00", (880, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)

    assert len(canonical) == 1
    assert canonical[0].unit_price == Decimal("3833.00")
    assert canonical[0].quantity == Decimal("2")
    assert canonical[0].net_amount == Decimal("7666.00")
    columns = {column.label: column for column in linked[0].columns}
    cells = {cell.column_id: cell for cell in linked[0].rows[0].cells}
    assert cells[columns["Expiry"].id].raw_value == "Sep-2027"
    assert cells[columns["Rate"].id].raw_value == "3,833.00"
    assert {
        token_id
        for label in ("Expiry", "Rate")
        for item in cells[columns[label].id].evidence
        for token_id in item.token_ids
    } == {"token-7"}


def test_source_table_excludes_distant_text_beyond_final_column_boundary() -> None:
    tokens = (
        token(0, "Sr.N", (50, 30, 90, 45)),
        token(1, "Particular", (100, 30, 420, 45)),
        token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(3, "Total", (880, 30, 970, 45)),
        token(4, "2.", (50, 70, 70, 85)),
        token(5, "CONSULTING CHARGES PAR DAY", (100, 70, 420, 85)),
        token(6, "1,000.00", (620, 70, 700, 85)),
        token(7, "2", (760, 70, 780, 85)),
        token(8, "2,000", (890, 70, 960, 85)),
        token(9, "114119", (1110, 70, 1180, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 1200, 110),
    )

    total_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "net_amount"
    )
    cells = {
        cell.column_id: cell for cell in result.source_tables[0].rows[0].cells
    }
    assert cells[total_column.id].raw_value == "2,000"
    assert result.rows[0].candidate.amount == Decimal("2000")


def test_source_table_keeps_shifted_value_within_final_lane_tolerance() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 420, 45)),
        token(1, "Disc", (820, 30, 860, 45)),
        token(2, "Net Amount", (870, 30, 950, 45)),
        token(3, "Consulting Charges", (100, 70, 360, 85)),
        token(4, "0", (830, 70, 850, 85)),
        token(5, "2,000", (920, 70, 980, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(0, 20, 1000, 110),
    )

    total_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "net_amount"
    )
    cells = {
        cell.column_id: cell for cell in result.source_tables[0].rows[0].cells
    }
    assert cells[total_column.id].raw_value == "2,000"
    assert result.rows[0].candidate.amount == Decimal("2000")


def test_source_table_keeps_shifted_outer_date_within_structured_tolerance() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 420, 45)),
        token(1, "Net Amount", (770, 30, 850, 45)),
        token(2, "Date", (850, 30, 910, 45)),
        token(3, "Consulting Charges", (100, 70, 360, 85)),
        token(4, "2,000", (780, 70, 840, 85)),
        token(5, "15/07/2026", (920, 70, 1000, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(0, 20, 1000, 110),
    )

    date_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "service_date_raw"
    )
    cells = {
        cell.column_id: cell for cell in result.source_tables[0].rows[0].cells
    }
    assert cells[date_column.id].raw_value == "15/07/2026"
    assert result.rows[0].candidate.service_date == "15/07/2026"


def test_numeric_row_marker_grouped_with_header_is_not_exposed_as_a_column() -> None:
    tokens = (
        token(0, "Sr.N", (50, 30, 90, 45)),
        token(1, "Particular", (100, 30, 420, 45)),
        token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(3, "Total", (880, 30, 970, 45)),
        token(4, "0.", (50, 39, 70, 54)),
        token(5, "1.", (50, 70, 70, 85)),
        token(6, "Registration", (100, 70, 360, 85)),
        token(7, "300.00", (620, 70, 700, 85)),
        token(8, "1", (760, 70, 780, 85)),
        token(9, "300.00", (890, 70, 960, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 980, 110),
    )

    assert [column.label for column in result.source_tables[0].columns] == [
        "Sr.N",
        "Particular",
        "Amount Rs.",
        "Unit/Days",
        "Total",
    ]
    assert [cell.raw_value for cell in result.source_tables[0].rows[0].cells] == [
        "1.",
        "Registration",
        "300.00",
        "1",
        "300.00",
    ]


def test_amount_rs_without_distinct_total_remains_the_final_amount_column() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 420, 45)),
        token(1, "Amount Rs.", (850, 30, 950, 45)),
        token(2, "Consulting Charges", (100, 70, 360, 85)),
        token(3, "2,000.00", (870, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert result.rows[0].candidate.rate is None
    assert result.rows[0].candidate.amount == Decimal("2000.00")
    assert [
        (column.label, column.canonical_field) for column in result.source_tables[0].columns
    ] == [
        ("Particular", "description"),
        ("Amount Rs.", "net_amount"),
    ]


def test_source_table_keeps_unknown_columns_and_raw_ocr_cells() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 360, 45)),
        token(1, "Co-pay %", (520, 30, 610, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Procedure", (100, 70, 300, 85)),
        token(4, "10", (540, 70, 570, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    source_tables = getattr(result, "source_tables", ())

    assert len(source_tables) == 1
    assert [column.label for column in source_tables[0].columns] == [
        "Particular",
        "Co-pay %",
        "Amount",
    ]
    assert [column.canonical_field for column in source_tables[0].columns] == [
        "description",
        None,
        "net_amount",
    ]
    assert [cell.raw_value for cell in source_tables[0].rows[0].cells] == [
        "Procedure",
        "10",
        "4,500.00",
    ]
    assert all(cell.evidence for cell in source_tables[0].rows[0].cells)


def test_unmapped_text_column_is_isolated_from_canonical_description() -> None:
    tokens = (
        token(0, "Description", (100, 30, 360, 45)),
        token(1, "Coverage", (500, 30, 650, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Procedure", (100, 70, 300, 85)),
        token(4, "Cashless", (510, 70, 630, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)

    assert canonical[0].description == "Procedure"
    assert result.rows[0].field_token_ids["description"] == ("token-3",)
    assert [cell.raw_value for cell in linked[0].rows[0].cells] == [
        "Procedure",
        "Cashless",
        "4,500.00",
    ]
    description_column = next(
        column for column in linked[0].columns if column.canonical_field == "description"
    )
    description_cell = next(
        cell
        for cell in linked[0].rows[0].cells
        if cell.column_id == description_column.id
    )
    assert {
        token_id
        for item in description_cell.evidence
        for token_id in item.token_ids
    } == {"token-3"}


def test_unmapped_text_column_before_description_is_isolated() -> None:
    tokens = (
        token(0, "Coverage", (100, 30, 250, 45)),
        token(1, "Description", (400, 30, 650, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Cashless", (110, 70, 230, 85)),
        token(4, "Procedure", (410, 70, 610, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert result.rows[0].candidate.description == "Procedure"
    assert result.rows[0].field_token_ids["description"] == ("token-4",)
    assert [cell.raw_value for cell in result.source_tables[0].rows[0].cells] == [
        "Cashless",
        "Procedure",
        "4,500.00",
    ]


def test_repeated_header_repartitions_unmapped_column_before_description() -> None:
    tokens = (
        token(0, "Description", (100, 30, 350, 45)),
        token(1, "Coverage", (500, 30, 650, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Procedure one", (100, 70, 320, 85)),
        token(4, "Cashless", (510, 70, 630, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
        token(6, "Coverage", (90, 110, 220, 125)),
        token(7, "Description", (300, 110, 520, 125)),
        token(8, "Plan", (620, 110, 700, 125)),
        token(9, "Amount", (850, 110, 950, 125)),
        token(10, "Reimbursed", (100, 150, 210, 165)),
        token(11, "Procedure two", (310, 150, 500, 165)),
        token(12, "Cashless", (620, 150, 700, 165)),
        token(13, "3,000.00", (860, 150, 940, 165)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 190),
    )

    assert result.diagnostics["header_segments"] == 2
    assert [row.candidate.description for row in result.rows] == [
        "Procedure one",
        "Procedure two",
    ]
    assert [
        [cell.raw_value for cell in table.rows[0].cells]
        for table in result.source_tables
    ] == [
        ["Procedure one", "Cashless", "4,500.00"],
        ["Reimbursed", "Procedure two", "Cashless", "3,000.00"],
    ]


def test_source_table_keeps_fully_unknown_grounded_headers() -> None:
    tokens = (
        token(0, "Charge", (100, 30, 360, 45)),
        token(1, "Co-pay %", (520, 30, 610, 45)),
        token(2, "Value", (850, 30, 950, 45)),
        token(3, "Procedure", (100, 70, 300, 85)),
        token(4, "10", (540, 70, 570, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert [column.label for column in result.source_tables[0].columns] == [
        "Charge",
        "Co-pay %",
        "Value",
    ]
    assert [column.canonical_field for column in result.source_tables[0].columns] == [
        None,
        None,
        None,
    ]
    assert [cell.raw_value for cell in result.source_tables[0].rows[0].cells] == [
        "Procedure",
        "10",
        "4,500.00",
    ]


def test_source_tables_restart_at_arbitrary_header_after_preamble() -> None:
    tokens = (
        token(0, "From Date", (80, 10, 170, 25)),
        token(1, ":06-07-2026", (190, 10, 300, 25)),
        token(2, "To Date", (380, 10, 450, 25)),
        token(3, ":08-07-2026", (470, 10, 580, 25)),
        token(4, "Sales", (80, 35, 140, 50)),
        token(5, "SR. ISSUE NO", (80, 60, 190, 75)),
        token(6, "ISSUE DATE", (240, 60, 340, 75)),
        token(7, "DISCOUNT", (430, 60, 510, 75)),
        token(8, "ROUND OFF", (550, 60, 640, 75)),
        token(9, "NET AMT BILL TYPE", (700, 60, 850, 75)),
        token(10, "HOSPITAL", (900, 60, 980, 75)),
        token(11, "1 S.19201", (80, 95, 180, 110)),
        token(12, "06-07-2026", (240, 95, 340, 110)),
        token(13, "0.00", (450, 95, 500, 110)),
        token(14, "0.00", (570, 95, 620, 110)),
        token(15, "8292.77 CREDIT", (700, 95, 850, 110)),
        token(16, "Y", (925, 95, 940, 110)),
        token(17, "2 S.19273", (80, 130, 180, 145)),
        token(18, "07-07-2026", (240, 130, 340, 145)),
        token(19, "0.00", (450, 130, 500, 145)),
        token(20, "0.00", (570, 130, 620, 145)),
        token(21, "631.89 CREDIT", (700, 130, 850, 145)),
        token(22, "Y", (925, 130, 940, 145)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 1000, 160),
    )

    assert len(result.source_tables) == 2
    ledger = result.source_tables[1]
    assert [column.label for column in ledger.columns] == [
        "SR. ISSUE NO",
        "ISSUE DATE",
        "DISCOUNT",
        "ROUND OFF",
        "NET AMT BILL TYPE",
        "HOSPITAL",
    ]
    assert [cell.raw_value for cell in ledger.rows[0].cells] == [
        "1 S.19201",
        "06-07-2026",
        "0.00",
        "0.00",
        "8292.77 CREDIT",
        "Y",
    ]


def test_source_tables_restart_at_arbitrary_header_after_valid_ledger() -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (300, 20, 520, 35)),
        token(2, "Units", (600, 20, 660, 35)),
        token(3, "Service Amt", (700, 20, 790, 35)),
        token(4, "Disc Amt", (800, 20, 870, 35)),
        token(5, "Net Amt", (900, 20, 970, 35)),
        token(6, "27/06/2026", (80, 55, 160, 70)),
        token(7, "Suction Catheter", (300, 55, 520, 70)),
        token(8, "1.00", (610, 55, 650, 70)),
        token(9, "91.00", (710, 55, 780, 70)),
        token(10, "0.00", (810, 55, 860, 70)),
        token(11, "91.00", (910, 55, 960, 70)),
        token(12, "Advance/Receipt detail", (80, 90, 280, 105)),
        token(13, "Receipt No", (300, 120, 420, 135)),
        token(14, "Receipt Date", (520, 120, 640, 135)),
        token(15, "Card Charges", (700, 120, 810, 135)),
        token(16, "Receipt Amount", (870, 120, 980, 135)),
        token(17, "OPA1/26/306 (EFT)", (300, 155, 450, 170)),
        token(18, "23/05/2026 1:02PM", (520, 155, 660, 170)),
        token(19, "0.00", (730, 155, 780, 170)),
        token(20, "5000.00", (900, 155, 970, 170)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 1000, 190),
    )

    assert len(result.source_tables) == 2
    assert [
        column.label for column in result.source_tables[1].columns
    ] == [
        "Receipt No",
        "Receipt Date",
        "Card Charges",
        "Receipt Amount",
    ]
    assert [
        cell.raw_value for cell in result.source_tables[1].rows[0].cells
    ] == [
        "OPA1/26/306 (EFT)",
        "23/05/2026 1:02PM",
        "0.00",
        "5000.00",
    ]
    assert [
        aligned.candidate.description for aligned in result.rows
    ] == ["Suction Catheter"]


def test_arbitrary_data_rows_are_not_promoted_to_repeated_headers() -> None:
    tokens = (
        token(0, "REFERENCE", (80, 20, 200, 35)),
        token(1, "KIND", (420, 20, 500, 35)),
        token(2, "VALUE", (740, 20, 850, 35)),
        token(3, "S.19201", (80, 60, 200, 75)),
        token(4, "CREDIT", (420, 60, 500, 75)),
        token(5, "8292.77 CREDIT", (740, 60, 880, 75)),
        token(6, "CURRENT", (80, 90, 200, 105)),
        token(7, "SALES", (420, 90, 500, 105)),
        token(8, "CREDIT", (740, 90, 850, 105)),
        token(9, "S.19202", (80, 120, 200, 135)),
        token(10, "CREDIT", (420, 120, 500, 135)),
        token(11, "631.89", (740, 120, 850, 135)),
        token(12, "S.19203", (80, 150, 200, 165)),
        token(13, "CREDIT", (420, 150, 500, 165)),
        token(14, "500.00", (740, 150, 850, 165)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 900, 180),
    )

    assert len(result.source_tables) == 1
    table = result.source_tables[0]
    assert [column.label for column in table.columns] == [
        "REFERENCE",
        "KIND",
        "VALUE",
    ]
    assert [
        [cell.raw_value for cell in source_row.cells]
        for source_row in table.rows
    ] == [
        ["S.19201", "CREDIT", "8292.77 CREDIT"],
        ["CURRENT", "SALES", "CREDIT"],
        ["S.19202", "CREDIT", "631.89"],
        ["S.19203", "CREDIT", "500.00"],
    ]


def test_headerish_nil_data_row_is_not_promoted_to_repeated_header() -> None:
    tokens = (
        token(0, "REFERENCE", (80, 20, 200, 35)),
        token(1, "KIND", (420, 20, 500, 35)),
        token(2, "VALUE", (740, 20, 850, 35)),
        token(3, "S.1", (80, 60, 200, 75)),
        token(4, "CREDIT", (420, 60, 500, 75)),
        token(5, "100.00", (740, 60, 850, 75)),
        token(6, "SERVICE", (80, 90, 200, 105)),
        token(7, "CREDIT", (420, 90, 500, 105)),
        token(8, "NIL", (740, 90, 850, 105)),
        token(9, "S.2", (80, 120, 200, 135)),
        token(10, "CREDIT", (420, 120, 500, 135)),
        token(11, "631.89", (740, 120, 850, 135)),
        token(12, "S.3", (80, 150, 200, 165)),
        token(13, "CREDIT", (420, 150, 500, 165)),
        token(14, "500.00", (740, 150, 850, 165)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 900, 180),
    )

    assert len(result.source_tables) == 1
    assert [
        [cell.raw_value for cell in source_row.cells]
        for source_row in result.source_tables[0].rows
    ] == [
        ["S.1", "CREDIT", "100.00"],
        ["SERVICE", "CREDIT", "NIL"],
        ["S.2", "CREDIT", "631.89"],
        ["S.3", "CREDIT", "500.00"],
    ]


def test_unicode_nil_data_row_is_not_promoted_to_repeated_header() -> None:
    tokens = (
        token(0, "विवरण", (80, 20, 200, 35)),
        token(1, "प्रकार", (420, 20, 500, 35)),
        token(2, "राशि", (740, 20, 850, 35)),
        token(3, "सेवा", (80, 60, 200, 75)),
        token(4, "उधार", (420, 60, 500, 75)),
        token(5, "100.00", (740, 60, 850, 75)),
        token(6, "दवा", (80, 90, 200, 105)),
        token(7, "नकद", (420, 90, 500, 105)),
        token(8, "शून्य", (740, 90, 850, 105)),
        token(9, "जाँच", (80, 120, 200, 135)),
        token(10, "उधार", (420, 120, 500, 135)),
        token(11, "631.89", (740, 120, 850, 135)),
        token(12, "कमरा", (80, 150, 200, 165)),
        token(13, "उधार", (420, 150, 500, 165)),
        token(14, "500.00", (740, 150, 850, 165)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 900, 180),
    )

    assert len(result.source_tables) == 1
    assert [
        [cell.raw_value for cell in source_row.cells]
        for source_row in result.source_tables[0].rows
    ] == [
        ["सेवा", "उधार", "100.00"],
        ["दवा", "नकद", "शून्य"],
        ["जाँच", "उधार", "631.89"],
        ["कमरा", "उधार", "500.00"],
    ]


def test_source_table_row_links_to_its_grounded_canonical_row() -> None:
    tokens = (
        token(0, "Particular", (100, 30, 360, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Procedure", (100, 70, 300, 85)),
        token(3, "4,500.00", (860, 70, 940, 85)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    assert linked[0].rows[0].canonical_row_id == str(canonical[0].id)


def test_merged_date_and_description_in_date_lane_is_grounded_and_split() -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Units", (600, 20, 660, 35)),
        token(3, "Service Amt", (700, 20, 790, 35)),
        token(4, "Disc Amt", (800, 20, 870, 35)),
        token(5, "Net Amount", (900, 20, 970, 35)),
        token(6, "14/07/2026 NORMAL DELIVERY", (80, 60, 330, 75)),
        token(7, "1.00", (610, 60, 650, 75)),
        token(8, "95000.00", (700, 60, 790, 75)),
        token(9, "0.00", (810, 60, 860, 75)),
        token(10, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "NORMAL DELIVERY"
    assert result.rows[0].candidate.service_date == "14/07/2026"
    assert result.rows[0].candidate.amount == Decimal("95000.00")
    assert result.rows[0].field_token_ids["description"] == ("token-6",)
    assert result.rows[0].field_token_ids["service_date"] == ("token-6",)

    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    table = linked[0]
    assert table.rows[0].canonical_row_id == str(canonical[0].id)
    cells_by_field = {
        column.canonical_field: next(
            cell
            for cell in table.rows[0].cells
            if cell.column_id == column.id
        )
        for column in table.columns
        if column.canonical_field is not None
    }
    assert cells_by_field["service_date_raw"].raw_value == "14/07/2026"
    assert cells_by_field["description"].raw_value == "NORMAL DELIVERY"
    assert (
        cells_by_field["service_date_raw"].validation_flags
        == ("split_from_merged_ocr_token",)
    )
    assert (
        cells_by_field["description"].validation_flags
        == ("split_from_merged_ocr_token",)
    )
    assert cells_by_field["service_date_raw"].evidence
    assert cells_by_field["description"].evidence


def test_slanted_rows_do_not_shift_total_amount_into_prior_charge() -> None:
    tokens = (
        token(0, "Service Name", (100, 10, 300, 30)),
        token(1, "Qty / Days", (600, 10, 680, 30)),
        token(2, "Amount", (730, 10, 810, 30)),
        token(3, "Total Amount", (900, 10, 990, 30)),
        token(4, "Diagnostics", (100, 50, 260, 90)),
        token(5, "500.00", (900, 80, 980, 120)),
        token(6, "500.00", (730, 95, 810, 135)),
        token(7, "1.00", (610, 105, 670, 145)),
        token(8, "COMPLETE BLOOD COUNT(CBC)", (100, 110, 430, 150)),
        token(9, "1,000.00", (900, 120, 980, 160)),
        token(10, "1,000.00", (730, 135, 810, 175)),
        token(11, "1.00", (610, 145, 670, 185)),
        token(12, "RENAL FUNCTION TEST(RFT)", (100, 150, 410, 190)),
        token(13, "1,500.00", (900, 160, 980, 200)),
        token(14, "Sub Total : Diagnostics", (100, 190, 390, 230)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 1000, 240),
    )

    assert [
        (aligned.candidate.description, aligned.candidate.amount)
        for aligned in result.rows
    ] == [
        ("COMPLETE BLOOD COUNT(CBC)", Decimal("500.00")),
        ("RENAL FUNCTION TEST(RFT)", Decimal("1000.00")),
    ]
    amount_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "net_amount"
    )
    assert [
        cell.raw_value
        for source_row in result.source_tables[0].rows
        if (cell := next(
            item
            for item in source_row.cells
            if item.column_id == amount_column.id
        )).raw_value
    ] == ["500.00", "1,000.00", "1,500.00"]


def test_separate_description_token_shifted_into_wide_date_lane_is_split() -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (550, 20, 720, 35)),
        token(2, "Units", (740, 20, 790, 35)),
        token(3, "Service Amt", (805, 20, 875, 35)),
        token(4, "Disc Amt", (890, 20, 945, 35)),
        token(5, "Net Amount", (960, 20, 1030, 35)),
        token(6, "27/06/2026", (80, 60, 160, 75)),
        token(7, "PT INR - PROTHROMBIN TIME", (180, 60, 430, 75)),
        token(8, "1.00", (745, 60, 785, 75)),
        token(9, "450.00", (810, 60, 870, 75)),
        token(10, "0.00", (895, 60, 940, 75)),
        token(11, "450.00", (965, 60, 1025, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 1040, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "PT INR - PROTHROMBIN TIME"
    assert result.rows[0].candidate.service_date == "27/06/2026"
    assert result.rows[0].field_token_ids["description"] == ("token-7",)
    assert result.rows[0].field_token_ids["service_date"] == ("token-6",)

    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    columns = {
        column.canonical_field: column
        for column in linked[0].columns
        if column.canonical_field is not None
    }
    cells = {
        cell.column_id: cell for cell in linked[0].rows[0].cells
    }
    assert linked[0].rows[0].canonical_row_id == str(canonical[0].id)
    assert cells[columns["service_date_raw"].id].raw_value == "27/06/2026"
    assert (
        cells[columns["description"].id].raw_value
        == "PT INR - PROTHROMBIN TIME"
    )
    assert "split_from_merged_ocr_token" in (
        cells[columns["service_date_raw"].id].validation_flags
    )
    assert "split_from_merged_ocr_token" in (
        cells[columns["description"].id].validation_flags
    )


@pytest.mark.parametrize(
    "shifted_text",
    (
        "9:05AM",
        "MNEIPI/123",
        "Service Code AB123",
    ),
)
def test_shifted_date_lane_metadata_is_not_promoted_to_description(
    shifted_text: str,
) -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (550, 20, 720, 35)),
        token(2, "Net Amount", (960, 20, 1030, 35)),
        token(3, "27/06/2026", (80, 60, 160, 75)),
        token(4, shifted_text, (180, 60, 430, 75)),
        token(5, "450.00", (965, 60, 1025, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 1040, 100),
    )

    assert result.rows == ()


def test_partial_description_in_date_lane_is_grounded_and_merged() -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Net Amount", (900, 20, 970, 35)),
        token(3, "14/07/2026 NORMAL", (80, 60, 330, 75)),
        token(4, "DELIVERY", (350, 60, 520, 75)),
        token(5, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "NORMAL DELIVERY"
    assert result.rows[0].field_token_ids["description"] == (
        "token-3",
        "token-4",
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    table = linked[0]
    assert table.rows[0].canonical_row_id == str(canonical[0].id)
    cells_by_field = {
        column.canonical_field: next(
            cell
            for cell in table.rows[0].cells
            if cell.column_id == column.id
        )
        for column in table.columns
        if column.canonical_field is not None
    }
    assert cells_by_field["service_date_raw"].raw_value == "14/07/2026"
    assert cells_by_field["description"].raw_value == "NORMAL DELIVERY"
    assert {
        token_id
        for evidence in cells_by_field["description"].evidence
        for token_id in evidence.token_ids
    } == {"token-3", "token-4"}


@pytest.mark.parametrize(
    ("date_cell_text", "description_cell_text", "description_token_ids"),
    (
        (
            "14/07/2026 MNEIPI/123 NORMAL DELIVERY",
            None,
            ("token-3",),
        ),
        (
            "14/07/2026 MNEIPI/123 NORMAL",
            "DELIVERY",
            ("token-3", "token-4"),
        ),
    ),
)
def test_request_prefix_in_date_lane_is_preserved_while_description_is_split(
    date_cell_text: str,
    description_cell_text: str | None,
    description_token_ids: tuple[str, ...],
) -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Net Amount", (900, 20, 970, 35)),
        token(3, date_cell_text, (80, 60, 330, 75)),
        *(
            (token(4, description_cell_text, (350, 60, 520, 75)),)
            if description_cell_text
            else ()
        ),
        token(5, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "NORMAL DELIVERY"
    assert result.rows[0].candidate.request_no == "MNEIPI/123"
    assert result.rows[0].field_token_ids["description"] == description_token_ids
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    table = linked[0]
    assert table.rows[0].canonical_row_id == str(canonical[0].id)
    cells_by_field = {
        column.canonical_field: next(
            cell
            for cell in table.rows[0].cells
            if cell.column_id == column.id
        )
        for column in table.columns
        if column.canonical_field is not None
    }
    assert cells_by_field["service_date_raw"].raw_value == "14/07/2026"
    assert (
        cells_by_field["description"].raw_value
        == "MNEIPI/123 NORMAL DELIVERY"
    )
    assert {
        token_id
        for evidence in cells_by_field["description"].evidence
        for token_id in evidence.token_ids
    } == set(description_token_ids)


@pytest.mark.parametrize(
    ("date_cell_text", "description_cell_text", "printed_description"),
    (
        (
            "14/07/2026 NORMAL DELIVERY Batch: X",
            None,
            "NORMAL DELIVERY Batch: X",
        ),
        (
            "14/07/2026 NORMAL Batch: X",
            "DELIVERY",
            "NORMAL Batch: X DELIVERY",
        ),
    ),
)
def test_batch_suffix_in_date_lane_is_preserved_while_description_is_cleaned(
    date_cell_text: str,
    description_cell_text: str | None,
    printed_description: str,
) -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Net Amount", (900, 20, 970, 35)),
        token(3, date_cell_text, (80, 60, 330, 75)),
        *(
            (token(4, description_cell_text, (350, 60, 520, 75)),)
            if description_cell_text
            else ()
        ),
        token(5, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "NORMAL DELIVERY"
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    table = linked[0]
    assert table.rows[0].canonical_row_id == str(canonical[0].id)
    cells_by_field = {
        column.canonical_field: next(
            cell
            for cell in table.rows[0].cells
            if cell.column_id == column.id
        )
        for column in table.columns
        if column.canonical_field is not None
    }
    assert cells_by_field["service_date_raw"].raw_value == "14/07/2026"
    assert cells_by_field["description"].raw_value == printed_description


@pytest.mark.parametrize(
    "suffix",
    (
        "1.00",
        "HSN1234",
        "HSN 1234",
        "HSN: 1234",
        "SAC 9983",
        "Service Code AB123",
        "Request No ABC123",
        "Request No: ABC123",
        "Reference No ABC123",
        "Ref No ABC123",
        "CASHLESS",
    ),
)
def test_merged_date_metadata_is_not_promoted_to_description(suffix: str) -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Net Amount", (900, 20, 970, 35)),
        token(3, f"14/07/2026 {suffix}", (80, 60, 330, 75)),
        token(4, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert result.rows == ()
    assert [
        [cell.raw_value for cell in row.cells]
        for row in result.source_tables[0].rows
    ] == [[f"14/07/2026 {suffix}", None, "95000.00"]]


@pytest.mark.parametrize(
    "description",
    (
        "DIALYSIS",
        "DELIVERY",
        "XRAY",
        "MRI",
        "ROOM",
        "COVID-19",
        "COVID19",
        "H1N1",
        "B12",
    ),
)
def test_merged_date_single_word_charge_is_preserved(description: str) -> None:
    tokens = (
        token(0, "Date", (80, 20, 160, 35)),
        token(1, "Particulars", (350, 20, 520, 35)),
        token(2, "Net Amount", (900, 20, 970, 35)),
        token(3, f"14/07/2026 {description}", (80, 60, 330, 75)),
        token(4, "95000.00", (900, 60, 970, 75)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == description
    assert result.rows[0].candidate.service_date == "14/07/2026"
    assert result.rows[0].candidate.amount == Decimal("95000.00")


def test_source_table_synthesizes_grounded_columns_without_a_header() -> None:
    tokens = (
        token(0, "Procedure", (100, 30, 300, 45)),
        token(1, "10", (540, 30, 570, 45)),
        token(2, "4,500.00", (860, 30, 940, 45)),
        token(3, "Medicine", (100, 70, 300, 85)),
        token(4, "20", (540, 70, 570, 85)),
        token(5, "2,000.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert [column.label for column in result.source_tables[0].columns] == [
        "Column 1",
        "Column 2",
        "Column 3",
    ]
    assert all(
        column.validation_flags == ("synthetic_header",)
        for column in result.source_tables[0].columns
    )
    assert [
        [cell.raw_value for cell in row.cells] for row in result.source_tables[0].rows
    ] == [
        ["Procedure", "10", "4,500.00"],
        ["Medicine", "20", "2,000.00"],
    ]


def test_headerless_source_date_lane_recovers_grounded_canonical_dates() -> None:
    tokens = (
        token(0, "1.", (70, 30, 90, 45)),
        token(1, "27/07/2026", (150, 30, 245, 45)),
        token(2, "Consultation", (300, 30, 500, 45)),
        token(3, "1,000.00", (860, 30, 940, 45)),
        token(4, "2.", (70, 70, 90, 85)),
        token(5, "28/07/2026", (150, 70, 245, 85)),
        token(6, "Procedure", (300, 70, 500, 85)),
        token(7, "4,500.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(50, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    recovered = _recover_grounded_service_dates(
        result.source_tables,
        canonical,
    )
    linked = _link_source_tables(result.source_tables, recovered)

    assert [row.service_date_raw for row in recovered] == [
        "27/07/2026",
        "28/07/2026",
    ]
    assert [row.service_date_iso for row in recovered] == [
        "2026-07-27",
        "2026-07-28",
    ]
    assert all(
        "service_date_recovered_from_source_cell" in row.validation_flags
        for row in recovered
    )
    date_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_date_raw"
    )
    assert date_column.label == "Date"
    assert "inferred_column_role" in date_column.validation_flags
    assert [
        next(
            cell.raw_value
            for cell in source_row.cells
            if cell.column_id == date_column.id
        )
        for source_row in linked[0].rows
        if source_row.canonical_row_id is not None
    ] == ["27/07/2026", "28/07/2026"]


def test_headerless_date_lane_displays_date_merged_into_another_cell() -> None:
    tokens = (
        token(0, "15/07/2026", (80, 30, 170, 45)),
        token(1, "MNEIPI/100 Item One", (250, 30, 580, 45)),
        token(2, "25.00", (700, 30, 760, 45)),
        token(3, "1.00", (790, 30, 830, 45)),
        token(4, "25.00", (900, 30, 960, 45)),
        token(
            5,
            "15/07/2026 16:17:00 - MNEIPI/101 Item Two",
            (250, 70, 580, 85),
        ),
        token(6, "30.00", (700, 70, 760, 85)),
        token(7, "1.00", (790, 70, 830, 85)),
        token(8, "30.00", (900, 70, 960, 85)),
        token(9, "15/07/2026", (80, 110, 170, 125)),
        token(10, "MNEIPI/102 Item Three", (250, 110, 580, 125)),
        token(11, "35.00", (700, 110, 760, 125)),
        token(12, "1.00", (790, 110, 830, 125)),
        token(13, "35.00", (900, 110, 960, 125)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 20, 980, 145),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    assert len(canonical) == 3
    merged_source_cell = next(
        cell
        for table in result.source_tables
        for row in table.rows
        for cell in row.cells
        if cell.raw_value
        == "15/07/2026 16:17:00 - MNEIPI/101 Item Two"
    )
    columns = (
        SourceColumn(
            id="date",
            label="Date",
            order=0,
            canonical_field="service_date_raw",
            validation_flags=("synthetic_header",),
        ),
        SourceColumn(
            id="printed",
            label="Column 2",
            order=1,
            validation_flags=("synthetic_header",),
        ),
    )
    source_cells = (
        SourceCell(
            column_id="date",
            validation_flags=("empty_cell",),
        ),
        SourceCell(
            column_id="printed",
            raw_value=merged_source_cell.raw_value,
            evidence=merged_source_cell.evidence,
        ),
    )

    split = _populate_grounded_service_date_cell(
        source_cells,
        columns,
        canonical[1],
    )
    cells = {cell.column_id: cell for cell in split}
    assert cells["date"].raw_value == "15/07/2026 16:17:00"
    assert cells["date"].evidence
    assert (
        "split_from_merged_ocr_token"
        in cells["date"].validation_flags
    )
    assert (
        cells["printed"].raw_value
        == "15/07/2026 16:17:00 - MNEIPI/101 Item Two"
    )


def test_grounded_date_lane_corrects_conflated_expiry_and_service_dates() -> None:
    tokens = (
        token(0, "27/07/2026", (150, 30, 245, 45)),
        token(1, "Syringe", (300, 30, 500, 45)),
        token(2, "1,000.00", (860, 30, 940, 45)),
        token(3, "28/07/2026", (150, 70, 245, 85)),
        token(4, "Procedure", (300, 70, 500, 85)),
        token(5, "4,500.00", (860, 70, 940, 85)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(50, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    corrupted = canonical[0].model_copy(
        update={
            "service_date_raw": "30/05/2028 27/07/2026",
            "service_date_iso": None,
        }
    )

    recovered = _recover_grounded_service_dates(
        result.source_tables,
        (corrupted, canonical[1]),
    )

    assert recovered[0].service_date_raw == "27/07/2026"
    assert recovered[0].service_date_iso == "2026-07-27"
    assert (
        "service_date_corrected_from_source_cell"
        in recovered[0].validation_flags
    )
    assert {
        token_id
        for evidence in recovered[0].field_evidence["service_date"]
        for token_id in evidence.token_ids
    } == {"token-0"}


def test_payment_and_expiry_dates_are_not_recovered_as_service_dates() -> None:
    charge_tokens = (
        token(0, "Product", (100, 30, 300, 45)),
        token(1, "Expiry", (500, 30, 580, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Medicine", (100, 70, 300, 85)),
        token(4, "01/03/2028", (500, 70, 590, 85)),
        token(5, "2,000.00", (860, 70, 940, 85)),
    )
    charge = reconstruct_ocr_rows(
        charge_tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        charge.rows,
    )
    recovered = _recover_grounded_service_dates(
        charge.source_tables,
        canonical,
    )

    assert recovered[0].service_date_raw is None
    assert all(
        column.canonical_field != "service_date_raw"
        for column in _link_source_tables(
            charge.source_tables,
            recovered,
        )[0].columns
    )


def test_unmapped_batch_expiry_value_does_not_become_service_date() -> None:
    tokens = (
        token(0, "Product", (100, 30, 300, 45)),
        token(1, "Generated By", (500, 30, 650, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "Roceft 1.5gm Inj", (100, 70, 300, 85)),
        token(4, "B6CS01B 01/03/2028", (500, 70, 680, 85)),
        token(5, "2,000.00", (860, 70, 940, 85)),
        token(6, "Nipro Syringe", (100, 100, 300, 115)),
        token(7, "26C17K33 01/02/2031", (500, 100, 690, 115)),
        token(8, "200.00", (860, 100, 940, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    recovered = _recover_grounded_service_dates(
        result.source_tables,
        canonical,
    )

    assert [row.service_date_raw for row in recovered] == [None, None]


def test_sale_date_line_applies_only_within_its_grounded_item_group() -> None:
    tokens = (
        token(0, "S.NO", (60, 20, 100, 35)),
        token(1, "SALE NO", (150, 20, 240, 35)),
        token(2, "PARTICULARS", (300, 20, 520, 35)),
        token(3, "QTY", (650, 20, 700, 35)),
        token(4, "RATE", (750, 20, 810, 35)),
        token(5, "AMOUNT", (880, 20, 960, 35)),
        token(6, "1", (70, 55, 85, 70)),
        token(7, "S54281", (150, 55, 220, 70)),
        token(8, "Medicine A", (300, 55, 470, 70)),
        token(9, "1", (670, 55, 680, 70)),
        token(10, "10.00", (760, 55, 800, 70)),
        token(11, "10.00", (900, 55, 950, 70)),
        token(12, "03/06/2026", (150, 85, 245, 100)),
        token(13, "2", (70, 115, 85, 130)),
        token(14, "Medicine B", (300, 115, 470, 130)),
        token(15, "1", (670, 115, 680, 130)),
        token(16, "20.00", (760, 115, 800, 130)),
        token(17, "20.00", (900, 115, 950, 130)),
        token(18, "3", (70, 150, 85, 165)),
        token(19, "S54300", (150, 150, 220, 165)),
        token(20, "Medicine C", (300, 150, 470, 165)),
        token(21, "1", (670, 150, 680, 165)),
        token(22, "30.00", (760, 150, 800, 165)),
        token(23, "30.00", (900, 150, 950, 165)),
        token(24, "04/06/2026", (150, 180, 245, 195)),
        token(25, "4", (70, 210, 85, 225)),
        token(26, "Medicine D", (300, 210, 470, 225)),
        token(27, "1", (670, 210, 680, 225)),
        token(28, "40.00", (760, 210, 800, 225)),
        token(29, "40.00", (900, 210, 950, 225)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 10, 980, 245),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    recovered = _recover_grounded_service_dates(
        result.source_tables,
        canonical,
    )

    assert [row.description for row in recovered] == [
        "Medicine A",
        "Medicine B",
        "Medicine C",
        "Medicine D",
    ]
    assert [row.service_date_iso for row in recovered] == [
        "2026-06-03",
        "2026-06-03",
        "2026-06-04",
        "2026-06-04",
    ]
    assert all(
        "service_date_inherited_from_group" in row.validation_flags
        for row in recovered
    )


def test_source_table_skips_title_before_headerless_rows() -> None:
    tokens = (
        token(0, "Charges", (100, 5, 300, 20)),
        token(1, "Procedure", (100, 40, 300, 55)),
        token(2, "10", (540, 40, 570, 55)),
        token(3, "4,500.00", (860, 40, 940, 55)),
        token(4, "Medicine", (100, 70, 300, 85)),
        token(5, "20", (540, 70, 570, 85)),
        token(6, "2,000.00", (860, 70, 940, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 980, 100),
    )

    assert len(result.rows) == 2
    assert [column.label for column in result.source_tables[0].columns] == [
        "Column 1",
        "Column 2",
        "Column 3",
    ]
    assert [
        [cell.raw_value for cell in row.cells] for row in result.source_tables[0].rows
    ] == [
        ["Procedure", "10", "4,500.00"],
        ["Medicine", "20", "2,000.00"],
    ]


def test_repeated_shifted_headers_reassign_rate_and_quantity_lanes() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "UnitPrice", (520, 30, 620, 45)),
        token(2, "Quantity", (650, 30, 750, 45)),
        token(3, "Amount", (850, 30, 950, 45)),
        token(4, "Room :408--A/C", (100, 70, 330, 85)),
        token(5, "3000.00", (530, 70, 610, 85)),
        token(6, "4.00", (680, 70, 730, 85)),
        token(7, "12000.00", (850, 70, 940, 85)),
        token(8, "Consultation : -Rs 700.00", (100, 105, 410, 120)),
        token(9, "Description", (100, 140, 300, 155)),
        token(10, "UnitPrice", (650, 140, 750, 155)),
        token(11, "Quantity", (770, 140, 850, 155)),
        token(12, "Amount", (880, 140, 970, 155)),
        token(13, "Consultation one", (100, 180, 330, 195)),
        token(14, "350.00", (660, 180, 740, 195)),
        token(15, "1.00", (790, 180, 830, 195)),
        token(16, "350.00", (890, 180, 960, 195)),
        token(17, "Consultation two", (100, 210, 330, 225)),
        token(18, "350.00", (660, 210, 740, 225)),
        token(19, "1.00", (790, 210, 830, 225)),
        token(20, "350.00", (890, 210, 960, 225)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 990, 250),
    )
    assert result.diagnostics["header_segments"] == 2
    assert [row.candidate.rate for row in result.rows] == [
        Decimal("3000.00"),
        Decimal("350.00"),
        Decimal("350.00"),
    ]
    assert [row.candidate.quantity for row in result.rows] == [
        Decimal("4.00"),
        Decimal("1.00"),
        Decimal("1.00"),
    ]


def test_wrapped_header_allows_numeric_label_beside_named_columns() -> None:
    tokens = (
        token(0, "Item Name", (100, 20, 300, 35)),
        token(1, "Tax", (500, 20, 580, 35)),
        token(2, "1", (650, 20, 670, 35)),
        token(3, "Net", (850, 20, 900, 35)),
        token(4, "Amount", (850, 45, 950, 60)),
        token(5, "Procedure", (100, 90, 300, 105)),
        token(6, "10", (520, 90, 560, 105)),
        token(7, "1", (650, 90, 670, 105)),
        token(8, "4,500.00", (860, 90, 940, 105)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 130),
    )

    assert result.diagnostics["header_found"] is True
    assert [column.label for column in result.source_tables[0].columns] == [
        "Item Name",
        "Tax",
        "Net",
        "Amount",
    ]


def test_header_does_not_absorb_distant_metadata_line() -> None:
    tokens = (
        token(0, "Discharge Date: 08-07-2026 03:58 PM", (100, 10, 400, 25)),
        token(1, "TPA: MEDICLAIM", (500, 10, 650, 25)),
        token(2, "No.", (100, 100, 150, 115)),
        token(3, "Code", (200, 100, 270, 115)),
        token(4, "Service Name", (400, 100, 650, 115)),
        token(5, "Amount", (850, 100, 950, 115)),
        token(6, "1.A", (100, 140, 150, 155)),
        token(7, "Room Rent", (400, 140, 600, 155)),
        token(8, "16,000.00", (850, 140, 950, 155)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 980, 180),
    )

    assert [column.label for column in result.source_tables[0].columns] == [
        "No.",
        "Code",
        "Service Name",
        "Amount",
    ]


def test_header_does_not_absorb_redundant_adjacent_bill_date_metadata() -> None:
    tokens = (
        token(0, "Bill Date", (800, 5, 900, 20)),
        token(1, "Date", (80, 30, 160, 45)),
        token(2, "Particulars", (300, 30, 550, 45)),
        token(3, "Units", (600, 30, 660, 45)),
        token(4, "Service Amt", (700, 30, 790, 45)),
        token(5, "Disc Amt", (800, 30, 870, 45)),
        token(6, "Net Amt", (900, 30, 970, 45)),
        token(7, "27/06/2026", (80, 70, 160, 85)),
        token(8, "Suction Catheter", (300, 70, 520, 85)),
        token(9, "1.00", (610, 70, 650, 85)),
        token(10, "91.00", (710, 70, 780, 85)),
        token(11, "0.00", (810, 70, 860, 85)),
        token(12, "91.00", (910, 70, 960, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 990, 110),
    )

    assert [column.label for column in result.source_tables[0].columns] == [
        "Date",
        "Particulars",
        "Units",
        "Service Amt",
        "Disc Amt",
        "Net Amt",
    ]
    assert result.source_tables[0].columns[0].canonical_field == (
        "service_date_raw"
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    cells = {
        cell.column_id: cell for cell in linked[0].rows[0].cells
    }
    assert cells[linked[0].columns[0].id].raw_value == "27/06/2026"
    assert linked[0].rows[0].canonical_row_id == str(canonical[0].id)


def test_structured_field_does_not_borrow_from_adjacent_printed_column() -> None:
    tokens = (
        token(0, "No.", (100, 30, 150, 45)),
        token(1, "Code", (160, 30, 220, 45)),
        token(2, "Service Name", (400, 30, 650, 45)),
        token(3, "Amount", (850, 30, 950, 45)),
        token(4, "1.A", (100, 70, 150, 85)),
        token(5, "Room Rent", (400, 70, 600, 85)),
        token(6, "16,000.00", (850, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert result.rows[0].candidate.service_code is None
    table = result.source_tables[0]
    cells = {cell.column_id: cell for cell in table.rows[0].cells}
    no_column = next(column for column in table.columns if column.label == "No.")
    code_column = next(column for column in table.columns if column.label == "Code")
    assert cells[no_column.id].raw_value == "1.A"
    assert cells[code_column.id].raw_value is None


def test_structured_field_selects_valid_token_inside_its_printed_lane() -> None:
    tokens = (
        token(0, "No.", (100, 30, 150, 45)),
        token(1, "Code", (190, 30, 250, 45)),
        token(2, "Service Name", (400, 30, 650, 45)),
        token(3, "Amount", (850, 30, 950, 45)),
        token(4, "1.A", (150, 70, 190, 85)),
        token(5, "AB123", (260, 70, 300, 85)),
        token(6, "Room Rent", (400, 70, 600, 85)),
        token(7, "16,000.00", (850, 70, 950, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert result.rows[0].candidate.service_code == "AB123"
    table = result.source_tables[0]
    cells = {cell.column_id: cell for cell in table.rows[0].cells}
    no_column = next(column for column in table.columns if column.label == "No.")
    code_column = next(column for column in table.columns if column.label == "Code")
    assert cells[no_column.id].raw_value == "1.A"
    assert cells[code_column.id].raw_value == "AB123"


def test_structured_field_accepts_stacked_fragments_in_same_printed_lane() -> None:
    tokens = (
        token(0, "Service", (100, 20, 180, 35)),
        token(1, "Particular", (400, 20, 650, 35)),
        token(2, "Amount", (850, 45, 950, 60)),
        token(3, "Date", (105, 45, 165, 60)),
        token(4, "15/07/2026", (100, 90, 200, 105)),
        token(5, "Procedure", (400, 90, 600, 105)),
        token(6, "4,500.00", (850, 90, 950, 105)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 130),
    )

    assert result.diagnostics["header_found"] is True
    assert result.rows[0].candidate.service_date == "15/07/2026"


def test_repeated_header_ignores_numeric_section_preamble() -> None:
    tokens = (
        token(0, "Date", (195, 20, 269, 40)),
        token(1, "Code", (351, 20, 438, 40)),
        token(2, "Service Name (Notes)", (578, 20, 910, 40)),
        token(3, "Rate", (1492, 20, 1568, 40)),
        token(4, "Qty.", (1669, 20, 1733, 40)),
        token(5, "Amount", (1841, 20, 1962, 40)),
        token(6, "Disc", (2068, 20, 2138, 40)),
        token(7, "Net Amt", (2184, 20, 2311, 40)),
        token(8, "06-07-2026", (127, 60, 299, 80)),
        token(9, "Room Rent", (577, 60, 900, 80)),
        token(10, "8000", (1483, 60, 1565, 80)),
        token(11, "2", (1712, 60, 1737, 80)),
        token(12, "16000.00", (1833, 60, 1955, 80)),
        token(13, "0", (2108, 60, 2133, 80)),
        token(14, "16000.00", (2234, 60, 2357, 80)),
        token(15, "ACCOMMODATION CHARGES", (127, 100, 604, 120)),
        token(16, "Total Rs. 16000.00/-", (2055, 100, 2356, 120)),
        token(17, "1.A", (127, 135, 182, 155)),
        token(
            18,
            "Room Rent (06-07-2026 to 08-07-2026 MICU)",
            (431, 135, 1120, 155),
        ),
        token(19, "8000", (1484, 135, 1566, 155)),
        token(20, "2 Days", (1626, 135, 1738, 155)),
        token(21, "16000.00", (1817, 135, 1957, 155)),
        token(22, "0", (2107, 135, 2135, 155)),
        token(23, "16000.00", (2218, 135, 2357, 155)),
        token(24, "Date", (195, 175, 269, 195)),
        token(25, "Code", (351, 175, 438, 195)),
        token(26, "Service Name (Notes)", (578, 175, 910, 195)),
        token(27, "Rate", (1492, 175, 1568, 195)),
        token(28, "Qty.", (1669, 175, 1733, 195)),
        token(29, "Amount", (1841, 175, 1962, 195)),
        token(30, "Disc", (2068, 175, 2138, 195)),
        token(31, "Net Amt", (2184, 175, 2311, 195)),
        token(32, "CONSULTATION CHARGES", (128, 215, 560, 235)),
        token(33, "Total Rs. 7500.00/-", (2067, 215, 2351, 235)),
        token(34, "06-07-2026", (127, 255, 299, 275)),
        token(
            35,
            "IP VISIT CHARGE(ICU) (Dr. SWAPNIL JAISWAL)",
            (577, 255, 1327, 275),
        ),
        token(36, "2500", (1483, 255, 1565, 275)),
        token(37, "3", (1712, 255, 1737, 275)),
        token(38, "7500.00", (1833, 255, 1955, 275)),
        token(39, "0", (2108, 255, 2133, 275)),
        token(40, "7500.00", (2234, 255, 2357, 275)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(100, 0, 2380, 300),
    )

    assert result.diagnostics["header_segments"] == 2
    repeated = result.source_tables[1]
    description_column = next(
        column
        for column in repeated.columns
        if column.canonical_field == "description"
    )
    cells = {cell.column_id: cell for cell in repeated.rows[-1].cells}
    assert description_column.label == "Service Name (Notes)"
    assert cells[description_column.id].raw_value == (
        "IP VISIT CHARGE(ICU) (Dr. SWAPNIL JAISWAL)"
    )
    room_rent = next(
        row
        for row in result.rows
        if row.candidate.description.startswith("Room Rent")
    )
    assert room_rent.candidate.quantity == Decimal("2")


def test_quantity_with_printed_days_unit_is_canonical_numeric_quantity() -> None:
    tokens = (
        token(0, "No.", (129, 941, 183, 978)),
        token(1, "Code", (208, 942, 293, 978)),
        token(2, "Service Name (Notes)", (433, 942, 764, 982)),
        token(3, "Rate", (1489, 945, 1566, 982)),
        token(4, "Qty.", (1666, 944, 1730, 988)),
        token(5, "Amount", (1840, 946, 1960, 982)),
        token(6, "Disc", (2065, 946, 2134, 983)),
        token(7, "Net Amt.", (2184, 947, 2315, 982)),
        token(8, "ACCOMMODATION CHARGES", (127, 1020, 604, 1050)),
        token(9, "Total Rs. 16000.00/-", (2055, 1020, 2356, 1050)),
        token(10, "1.A", (127, 1108, 182, 1145)),
        token(
            11,
            "Room Rent (06-07-2026 to 08-07-2026 MICU)",
            (431, 1108, 1120, 1152),
        ),
        token(12, "8000", (1484, 1112, 1566, 1149)),
        token(13, "2 Days", (1626, 1112, 1738, 1155)),
        token(14, "16000.00", (1817, 1114, 1957, 1149)),
        token(15, "0", (2107, 1114, 2135, 1151)),
        token(16, "16000.00", (2218, 1115, 2357, 1150)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(100, 900, 2380, 1180),
    )

    room_rent = next(
        row
        for row in result.rows
        if row.candidate.description.startswith("Room Rent")
    )
    assert room_rent.candidate.quantity == Decimal("2")
    table = next(
        table
        for table in result.source_tables
        if any(column.canonical_field == "quantity" for column in table.columns)
    )
    quantity_column = next(
        column
        for column in table.columns
        if column.canonical_field == "quantity"
    )
    quantity_cell = next(
        cell
        for printed_row in table.rows
        for cell in printed_row.cells
        if cell.column_id == quantity_column.id
        and cell.raw_value == "2 Days"
    )
    assert quantity_cell.raw_value == "2 Days"


def test_vertically_split_decimal_suffix_stays_in_one_numeric_cell() -> None:
    tokens = (
        token(0, "Date", (195, 1183, 269, 1220)),
        token(1, "Code", (351, 1180, 438, 1222)),
        token(2, "Service Name (Notes)", (578, 1184, 910, 1224)),
        token(3, "Rate", (1492, 1186, 1568, 1224)),
        token(4, "Qty.", (1669, 1186, 1733, 1230)),
        token(5, "Amount", (1841, 1189, 1962, 1225)),
        token(6, "Disc", (2068, 1187, 2138, 1225)),
        token(7, "Net Amt", (2184, 1190, 2311, 1222)),
        token(8, "19720.9", (1436, 3140, 1558, 3176)),
        token(9, "08-07-2026", (122, 3156, 293, 3194)),
        token(10, "MEDICINE CHARGES", (574, 3158, 915, 3195)),
        token(11, "1", (1708, 3160, 1731, 3196)),
        token(12, "19720.96", (1811, 3159, 1951, 3197)),
        token(13, "0", (2103, 3161, 2128, 3196)),
        token(14, "19720.96", (2211, 3159, 2353, 3197)),
        token(15, "6", (1533, 3179, 1561, 3216)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(100, 1100, 2380, 3250),
    )

    row = next(
        row
        for row in result.rows
        if row.candidate.description == "MEDICINE CHARGES"
    )
    assert row.candidate.rate == Decimal("19720.96")
    assert row.candidate.quantity == Decimal("1")
    assert row.candidate.amount == Decimal("19720.96")
    assert "line_arithmetic_mismatch" not in row.candidate.validation_flags
    assert set(row.field_token_ids["rate"]) == {"token-8", "token-15"}


def test_category_total_with_header_words_does_not_reset_quantity_and_rate() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Rate", (650, 30, 740, 45)),
        token(2, "Qty", (770, 30, 830, 45)),
        token(3, "Amount", (880, 30, 970, 45)),
        token(4, "BLOOD GROUP & RH TYPE", (100, 70, 400, 85)),
        token(5, "32.00", (660, 70, 730, 85)),
        token(6, "1.00", (780, 70, 820, 85)),
        token(7, "32.00", (890, 70, 960, 85)),
        token(8, "TOTAL FOR BLOOD BANK INVESTIGATIONS", (100, 105, 520, 120)),
        token(9, "32.00", (890, 105, 960, 120)),
        token(10, "FIRST VISIT CARDIAC SURGEON", (100, 140, 480, 155)),
        token(11, "350.00", (660, 140, 730, 155)),
        token(12, "1.00", (780, 140, 820, 155)),
        token(13, "350.00", (890, 140, 960, 155)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 990, 180),
    )
    assert result.diagnostics["header_segments"] == 1
    assert [row.candidate.description for row in result.rows] == [
        "BLOOD GROUP & RH TYPE",
        "FIRST VISIT CARDIAC SURGEON",
    ]
    assert [row.candidate.quantity for row in result.rows] == [
        Decimal("1.00"),
        Decimal("1.00"),
    ]
    assert [row.candidate.rate for row in result.rows] == [
        Decimal("32.00"),
        Decimal("350.00"),
    ]


def test_compound_quantity_unitprice_header_splits_both_semantic_lanes() -> None:
    tokens = (
        token(0, "No", (80, 30, 95, 45)),
        token(1, "Description", (100, 30, 300, 45)),
        token(2, "Expiry Date", (500, 30, 620, 45)),
        token(3, "Quantity UnitPrice", (640, 30, 850, 45)),
        token(4, "Amount", (880, 30, 970, 45)),
        token(5, "5575450", (80, 70, 95, 85)),
        token(6, "NS 100 ML FLEXIDRIP", (100, 70, 390, 85)),
        token(7, "Aug/2028 1.00", (500, 70, 710, 85)),
        token(8, "44.93", (780, 70, 840, 85)),
        token(9, "44.93", (890, 70, 960, 85)),
        token(10, "5576674", (80, 100, 95, 115)),
        token(11, "KABICEFTAM 3GM", (100, 100, 360, 115)),
        token(12, "Aug/2027 3.00", (500, 100, 710, 115)),
        token(13, "1624.00", (760, 100, 845, 115)),
        token(14, "4872.00", (880, 100, 960, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 990, 140),
    )
    assert [row.candidate.quantity for row in result.rows] == [
        Decimal("1.00"),
        Decimal("3.00"),
    ]
    assert [row.candidate.rate for row in result.rows] == [
        Decimal("44.93"),
        Decimal("1624.00"),
    ]
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("44.93"),
        Decimal("4872.00"),
    ]


def test_compound_rate_quantity_data_token_splits_both_semantic_values() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Rate Qty", (600, 30, 760, 45)),
        token(2, "Amount", (850, 30, 950, 45)),
        token(3, "SURGICAL BLADE", (100, 70, 350, 85)),
        token(4, "33.00 1.00", (600, 70, 760, 85)),
        token(5, "33.00", (870, 70, 940, 85)),
        token(6, "BED DRY SHEET", (100, 100, 350, 115)),
        token(7, "-275.00 3.00", (600, 100, 760, 115)),
        token(8, "-825.00", (860, 100, 940, 115)),
        token(9, "SODAC INJECTION", (100, 130, 350, 145)),
        token(10, "35.7620.0", (600, 130, 760, 145)),
        token(11, "715.20", (860, 130, 940, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )
    assert [row.candidate.rate for row in result.rows] == [
        Decimal("33.00"),
        Decimal("-275.00"),
        Decimal("35.76"),
    ]
    assert [row.candidate.quantity for row in result.rows] == [
        Decimal("1.00"),
        Decimal("3.00"),
        Decimal("20.0"),
    ]
    assert not any(row.candidate.validation_flags for row in result.rows)


def test_repeated_procedure_header_is_not_published_as_a_charge() -> None:
    tokens = (
        token(0, "Item Name", (100, 30, 260, 45)),
        token(1, "Service Code", (520, 30, 620, 45)),
        token(2, "Date", (650, 30, 710, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "MRI Head", (100, 70, 260, 85)),
        token(5, "RI089", (530, 70, 600, 85)),
        token(6, "18/01/2026", (650, 70, 750, 85)),
        token(7, "2475.00", (880, 70, 960, 85)),
        token(8, "Day Care Procedure", (100, 110, 310, 125)),
        token(9, "Procedure Name", (100, 140, 280, 155)),
        token(10, "Service Code", (520, 140, 620, 155)),
        token(11, "Date", (650, 140, 710, 155)),
        token(12, "HSN Code Quantity", (720, 140, 850, 155)),
        token(13, "Net Amount", (870, 140, 970, 155)),
        token(14, "Urinary Bladder Catheterisation", (100, 180, 400, 195)),
        token(15, "GP009", (530, 180, 600, 195)),
        token(16, "19/01/2026", (650, 180, 750, 195)),
        token(17, "1", (800, 180, 815, 195)),
        token(18, "630.00", (890, 180, 960, 195)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 990, 220),
    )
    assert result.diagnostics["header_segments"] == 2
    assert [row.candidate.description for row in result.rows] == [
        "MRI Head",
        "Urinary Bladder Catheterisation",
    ]
    assert [row.candidate.service_code for row in result.rows] == ["RI089", "GP009"]
    assert [row.candidate.service_date for row in result.rows] == [
        "18/01/2026",
        "19/01/2026",
    ]
    assert result.rows[1].candidate.quantity == Decimal("1")
    assert result.rows[1].candidate.amount == Decimal("630.00")


def test_alphabetic_service_date_is_structured_and_removed_from_description() -> None:
    tokens = (
        token(0, "Description", (100, 30, 420, 45)),
        token(1, "Rate", (600, 30, 680, 45)),
        token(2, "Qty", (700, 30, 760, 45)),
        token(3, "Amount", (850, 30, 950, 45)),
        token(
            4,
            "12-Feb-2026 SER0944134 234A (SHARING A/C) - 12-Feb-2026 13:29 to 12",
            (100, 70, 560, 85),
        ),
        token(5, "2062.50", (600, 70, 680, 85)),
        token(6, "1.00", (710, 70, 750, 85)),
        token(7, "2062.50", (850, 70, 930, 85)),
        token(8, "13-Feb-2026 SER0945309 OXYGEN", (100, 100, 520, 115)),
        token(9, "77.00", (600, 100, 670, 115)),
        token(10, "24.00", (710, 100, 755, 115)),
        token(11, "1848.00", (850, 100, 930, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    first = result.rows[0].candidate
    assert first.service_date == "12-Feb-2026"
    assert first.description == "SER0944134 234A (SHARING A/C)"
    assert result.rows[0].field_token_ids["service_date"] == ("token-4",)


def test_description_continuation_after_amount_extends_previous_row() -> None:
    tokens = (
        token(0, "Description", (100, 30, 250, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Renal Function Test (Bun/Creat", (100, 70, 420, 85)),
        token(3, "975.00", (850, 70, 930, 85)),
        token(4, "/Urea/Electrolytes) /", (100, 100, 380, 115)),
        token(5, "RFT", (100, 130, 160, 145)),
        token(6, "Total for Investigation", (500, 160, 780, 175)),
        token(7, "975.00", (850, 160, 930, 175)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 190),
    )
    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == (
        "Renal Function Test (Bun/Creat /Urea/Electrolytes) / RFT"
    )
    assert result.rows[0].field_token_ids["description"] == (
        "token-2",
        "token-4",
        "token-5",
    )


def test_printed_connector_merges_grounded_canonical_and_source_rows() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(
            2,
            "Special Instruments/Equipments Charges-",
            (100, 70, 500, 85),
        ),
        token(3, "8,500.00", (870, 70, 950, 85)),
        token(4, "CIRCUMCISION STAPLER ZSR", (100, 100, 430, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )

    expected = "Special Instruments/Equipments Charges- CIRCUMCISION STAPLER ZSR"
    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == expected
    assert result.rows[0].field_token_ids["description"] == ("token-2", "token-4")
    assert {"token-2", "token-3", "token-4"} <= set(result.rows[0].evidence_token_ids)

    source_table = result.source_tables[0]
    assert len(source_table.rows) == 1
    assert [row.order for row in source_table.rows] == [0]
    description_column = next(
        column for column in source_table.columns if column.canonical_field == "description"
    )
    description_cell = next(
        cell
        for cell in source_table.rows[0].cells
        if cell.column_id == description_column.id
    )
    assert description_cell.raw_value == expected
    assert {"token-2", "token-4"} <= {
        token_id
        for evidence in description_cell.evidence
        for token_id in evidence.token_ids
    }


def test_provider_fusion_preserves_printed_amount_and_evidence() -> None:
    tokens = (
        token(0, "Description", (100, 30, 250, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "CEF 500", (100, 70, 250, 85)),
        token(3, "549.61", (850, 70, 930, 85)),
    )
    reconstruction = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    provider = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("CEFTUM 500", "999.00"),
        description="CEFTUM 500",
        amount=Decimal("999.00"),
        table_type=TableType.PHARMACY,
    )
    fused = fuse_provider_descriptions(reconstruction.rows, (provider,))
    assert fused[0].candidate.description == "CEFTUM 500"
    assert fused[0].candidate.amount == Decimal("549.61")
    assert fused[0].field_token_ids == reconstruction.rows[0].field_token_ids


def test_department_amount_table_is_a_category_summary() -> None:
    tokens = (
        token(0, "Department", (100, 30, 250, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "LABORATORY", (100, 70, 260, 85)),
        token(3, "0.00", (850, 70, 930, 85)),
        token(4, "CITY PHARMACY", (100, 100, 280, 115)),
        token(5, "100.00", (850, 100, 930, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert result.schema is not None
    assert result.schema.table_type is TableType.CATEGORY_SUMMARY
    assert {row.candidate.role for row in result.rows} == {RowRole.CATEGORY_ROLLUP}


def test_headerless_serial_table_with_zero_tail_is_a_category_summary() -> None:
    values = []
    for row in range(5):
        top = 30 + row * 30
        values.extend(
            (
                token(row * 4, str(row + 1), (100, top, 120, top + 15)),
                token(row * 4 + 1, f"DEPARTMENT {row + 1}", (200, top, 380, top + 15)),
                token(row * 4 + 2, f"{(row + 1) * 100}.00", (700, top, 780, top + 15)),
                token(row * 4 + 3, "0.00", (850, top, 930, top + 15)),
            )
        )
    result = reconstruct_ocr_rows(
        tuple(values),
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 200),
    )
    assert result.schema is not None
    assert result.schema.table_type is TableType.CATEGORY_SUMMARY
    assert {row.candidate.role for row in result.rows} == {RowRole.CATEGORY_ROLLUP}


def test_demographic_fragments_and_payments_are_not_detail_rows() -> None:
    metadata = (
        token(0, "UHID Original Bill No", (100, 30, 400, 45)),
        token(1, "20", (850, 30, 900, 45)),
        token(2, "Contact No D.O.A", (100, 70, 400, 85)),
        token(3, "17", (850, 70, 900, 85)),
        token(4, "Sponsor Billing Category", (100, 100, 450, 115)),
        token(5, "11", (850, 100, 900, 115)),
    )
    metadata_result = reconstruct_ocr_rows(
        metadata,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert metadata_result.schema is not None
    assert metadata_result.schema.table_type is TableType.METADATA
    assert {row.candidate.role for row in metadata_result.rows} == {RowRole.UNRESOLVED}

    payment = (
        token(6, "Payment Mode", (100, 30, 300, 45)),
        token(7, "Amount", (850, 30, 950, 45)),
        token(8, "Receipt Ref AD/26-27/7413", (100, 70, 450, 85)),
        token(9, "20000.00", (850, 70, 940, 85)),
        token(10, "Amount Refunded", (100, 100, 350, 115)),
        token(11, "-4528.00", (850, 100, 940, 115)),
    )
    payment_result = reconstruct_ocr_rows(
        payment,
        page_number=1,
        table_id="p1-t2",
        box=(80, 20, 980, 140),
    )
    assert payment_result.schema is not None
    assert payment_result.schema.table_type is TableType.PAYMENT
    assert {row.candidate.role for row in payment_result.rows} == {RowRole.PAYMENT}


def test_document_totals_and_advance_are_not_detail_rows() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Angiography", (100, 70, 300, 85)),
        token(3, "1000.00", (850, 70, 940, 85)),
        token(4, "Total Bill Amount", (100, 100, 350, 115)),
        token(5, "1000.00", (850, 100, 940, 115)),
        token(6, "Total Discount Amount", (100, 130, 380, 145)),
        token(7, "0.00", (850, 130, 940, 145)),
        token(8, "Advance Received", (100, 160, 350, 175)),
        token(9, "500.00", (850, 160, 940, 175)),
        token(10, "Amount To Be Received", (100, 190, 400, 205)),
        token(11, "500.00", (850, 190, 940, 205)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 220),
    )
    assert [row.candidate.description for row in result.rows] == [
        "Angiography",
        "Advance Received",
    ]
    assert result.rows[1].candidate.role is RowRole.PAYMENT


def test_footer_totals_short_fragments_and_standalone_expiry_are_not_rows() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Medicine", (100, 70, 300, 85)),
        token(3, "100.00", (850, 70, 940, 85)),
        token(4, "P", (100, 100, 130, 115)),
        token(5, "1493283.00", (850, 100, 940, 115)),
        token(6, "CGST Amount @2.5%", (100, 130, 350, 145)),
        token(7, "0.00", (850, 130, 940, 145)),
        token(8, "Payer Receivable (INR)", (100, 160, 400, 175)),
        token(9, "1493283.00", (850, 160, 940, 175)),
        token(10, "Round Off Amount", (100, 190, 350, 205)),
        token(11, "0.58", (850, 190, 940, 205)),
        token(12, "ExpDate", (100, 220, 250, 235)),
        token(13, "146.40", (850, 220, 940, 235)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 250),
    )
    assert [
        (row.candidate.description, row.candidate.amount)
        for row in result.rows
        if row.candidate.role is RowRole.DETAIL
    ] == [("Medicine", Decimal("100.00"))]
    assert [
        row.candidate.description for row in result.rows if row.candidate.role is RowRole.UNRESOLVED
    ] == ["ExpDate"]


def test_item_movement_totals_and_payer_footer_are_not_charges() -> None:
    tokens = (
        token(0, "Description", (100, 30, 400, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "SURGICAL BLADE", (100, 70, 350, 85)),
        token(3, "33.00", (870, 70, 940, 85)),
        token(4, "Item Issues Total", (100, 100, 350, 115)),
        token(5, "9885.83", (850, 100, 940, 115)),
        token(6, "Credit From UNITED INDIA INSURANCE", (100, 130, 500, 145)),
        token(7, "124615.00", (850, 130, 950, 145)),
        token(8, "Rupees In One Lakh Only", (100, 160, 450, 175)),
        token(9, "124615.00", (850, 160, 950, 175)),
        token(10, "Patient Wise Total", (100, 190, 350, 205)),
        token(11, "0.00", (850, 190, 940, 205)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 230),
    )
    assert [
        row.candidate.description
        for row in result.rows
        if row.candidate.role in {RowRole.DETAIL, RowRole.REFUND}
    ] == ["SURGICAL BLADE"]


def test_contact_footer_phone_number_is_metadata_not_a_charge() -> None:
    tokens = (
        token(0, "Description", (100, 30, 400, 45)),
        token(1, "Amount", (850, 30, 950, 45)),
        token(2, "Consultation", (100, 70, 300, 85)),
        token(3, "350.00", (850, 70, 930, 85)),
        token(4, "Doctor Appointments Free Home Sample Co", (100, 100, 600, 115)),
        token(5, "9962725555", (850, 100, 950, 115)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 140),
    )
    assert [row.candidate.description for row in result.rows] == [
        "Consultation",
        "Doctor Appointments Free Home Sample Co",
    ]
    assert result.rows[1].candidate.role is RowRole.UNRESOLVED


def test_headerless_settlement_region_is_terminal_payment() -> None:
    tokens = (
        token(0, "Payer Received", (100, 30, 350, 45)),
        token(1, "0.00", (850, 30, 940, 45)),
        token(2, "Patient Balance", (100, 60, 350, 75)),
        token(3, "0.04", (850, 60, 940, 75)),
        token(4, "Payer Receivable (INR)", (100, 90, 400, 105)),
        token(5, "1493283.00", (850, 90, 940, 105)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 120),
    )
    assert result.schema is not None
    assert result.schema.table_type is TableType.PAYMENT
    assert {row.candidate.role for row in result.rows} <= {
        RowRole.PAYMENT,
        RowRole.UNRESOLVED,
    }


def test_wrapped_header_keeps_date_code_and_amount_in_distinct_lanes() -> None:
    tokens = (
        token(0, "Service", (100, 20, 180, 32)),
        token(1, "Net", (890, 20, 930, 32)),
        token(2, "Service Name", (100, 42, 280, 54)),
        token(3, "Date", (650, 42, 710, 54)),
        token(4, "Code", (520, 64, 580, 76)),
        token(5, "Amount", (870, 64, 960, 76)),
        token(6, "Package Name : Coronary Angiography", (100, 100, 430, 114)),
        token(7, "20/01/2026 - 21/01/2026", (650, 100, 790, 114)),
        token(8, "11457.00", (870, 100, 960, 114)),
        token(9, "Complete Haemogram", (100, 135, 360, 149)),
        token(10, "LB012", (520, 135, 580, 149)),
        token(11, "20/01/2026 10:24:02", (650, 135, 800, 149)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 170),
    )
    assert result.diagnostics["header_found"] is True
    assert (
        result.diagnostics["column_centers"]["amount"]
        > result.diagnostics["column_centers"]["service_date"]
    )
    assert len(result.rows) == 2
    charge, component = (row.candidate for row in result.rows)
    assert charge.amount == Decimal("11457.00")
    assert charge.service_date == "20/01/2026 - 21/01/2026"
    assert component.role is RowRole.INFORMATIONAL
    assert component.amount is None
    assert component.service_code == "LB012"
    assert component.service_date == "20/01/2026 10:24:02"
    assert result.rows[1].field_token_ids["service_date"] == ("token-11",)


def test_invalid_stamp_text_is_not_published_as_a_missing_service_code() -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Service Code", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(9, "PATNA", (430, 120, 500, 155), 10),
        token(10, "20/01/2026 09:44:58", (700, 130, 850, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.service_code is None
    code_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_code"
    )
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    code_cell = next(
        cell
        for cell in gloves_source_row.cells
        if cell.column_id == code_column.id
    )
    assert code_cell.raw_value is None
    assert code_cell.evidence == ()
    assert "excluded_oversized_overlay" in code_cell.validation_flags


@pytest.mark.parametrize(
    ("overlay_text", "overlay_box", "supporting_overlay_tokens"),
    (
        ("No: 10&10/1, Radhkrishnan", (300, 120, 650, 155), ()),
        (
            "Road,",
            (530, 120, 610, 135),
            (
                rotated_token(11, "VL. Ltd.", (530, 100, 610, 115), 19),
                rotated_token(12, "600 087.", (530, 160, 610, 175), 19),
                rotated_token(
                    13,
                    "Valasaravakkam,",
                    (700, 100, 820, 115),
                    19,
                ),
            ),
        ),
        (
            "J032 51250.",
            (430, 120, 650, 155),
            (
                rotated_token(11, "Road,", (530, 100, 610, 115), 19),
                rotated_token(12, "600 087.", (530, 160, 610, 175), 19),
                rotated_token(
                    13,
                    "Valasaravakkam,",
                    (700, 100, 820, 115),
                    19,
                ),
            ),
        ),
    ),
)
def test_invalid_stamp_text_is_not_published_as_a_missing_request_number(
    overlay_text: str,
    overlay_box: tuple[float, float, float, float],
    supporting_overlay_tokens: tuple[OcrToken, ...],
) -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Bill Number", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(
            9,
            overlay_text,
            overlay_box,
            19,
        ),
        token(10, "20/01/2026 09:44:58", (700, 130, 850, 145)),
        *supporting_overlay_tokens,
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 180),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.request_no is None
    request_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "request_no"
    )
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    request_cell = next(
        cell
        for cell in gloves_source_row.cells
        if cell.column_id == request_column.id
    )
    assert request_cell.raw_value is None
    assert request_cell.evidence == ()
    assert "excluded_oversized_overlay" in request_cell.validation_flags


@pytest.mark.parametrize(
    ("printed_request", "preceding_request", "following_request"),
    (
        ("No: 12345", "REQ100", "REQ200"),
        ("REQ 150.", "REQ 100.", "REQ 200."),
    ),
)
def test_plausible_slanted_request_number_remains_for_strict_validation(
    printed_request: str,
    preceding_request: str,
    following_request: str,
) -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Bill Number", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(9, printed_request, (530, 120, 610, 135), 19),
        token(10, "20/01/2026 09:44:58", (700, 130, 850, 145)),
        rotated_token(11, preceding_request, (530, 100, 610, 115), 19),
        rotated_token(12, following_request, (530, 160, 610, 175), 19),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 180),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.request_no is None
    request_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "request_no"
    )
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    request_cell = next(
        cell
        for cell in gloves_source_row.cells
        if cell.column_id == request_column.id
    )
    assert request_cell.raw_value == printed_request
    assert request_cell.evidence
    assert "excluded_oversized_overlay" not in request_cell.validation_flags


def test_financial_row_values_survive_surrounding_cross_column_overlay() -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Bill Number", (430, 30, 520, 45)),
        token(2, "HSN", (570, 30, 640, 45)),
        token(3, "Date", (700, 30, 760, 45)),
        token(4, "Net Amount", (870, 30, 970, 45)),
        rotated_token(5, "Road,", (430, 85, 520, 100), 19),
        rotated_token(6, "Valasaravakkam,", (570, 85, 690, 100), 19),
        token(10, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(11, "REQ 150.", (430, 125, 520, 140), 19),
        rotated_token(12, "6210 4071", (570, 125, 660, 140), 19),
        token(13, "20/01/2026 09:44:58", (700, 130, 850, 145)),
        token(14, "100.00", (880, 130, 960, 145)),
        rotated_token(15, "600 087.", (430, 165, 520, 180), 19),
        rotated_token(16, "P:044-42649097", (570, 165, 690, 180), 19),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 200),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.request_no is None
    assert gloves.hsn_code is None
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    columns_by_field = {
        column.canonical_field: column for column in linked[0].columns
    }
    cells_by_column = {
        cell.column_id: cell for cell in gloves_source_row.cells
    }
    request_cell = cells_by_column[columns_by_field["request_no"].id]
    hsn_cell = cells_by_column[columns_by_field["hsn_code"].id]
    assert request_cell.raw_value == "REQ 150."
    assert hsn_cell.raw_value == "6210 4071"
    assert request_cell.evidence
    assert hsn_cell.evidence
    assert "excluded_oversized_overlay" not in request_cell.validation_flags
    assert "excluded_oversized_overlay" not in hsn_cell.validation_flags


def test_linked_source_row_splits_grounded_date_from_cross_column_ocr_token() -> None:
    tokens = (
        token(0, "Date", (100, 30, 170, 45)),
        token(1, "Particulars", (300, 30, 500, 45)),
        token(2, "Rate", (650, 30, 710, 45)),
        token(3, "Qty", (760, 30, 800, 45)),
        token(4, "Amount", (880, 30, 970, 45)),
        token(
            5,
            "15/07/2026 11:31:00 - MNEIPI/265586 BED BATH ADULT WIPES GINNI",
            (100, 70, 600, 85),
        ),
        token(6, "570.00", (650, 70, 710, 85)),
        token(7, "1.00", (760, 70, 800, 85)),
        token(8, "570.00", (880, 70, 970, 85)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    assert len(canonical) == 1
    assert canonical[0].service_date_raw == "15/07/2026"
    assert canonical[0].request_no == "MNEIPI/265586"
    columns = {column.canonical_field: column for column in linked[0].columns}
    cells = {cell.column_id: cell for cell in linked[0].rows[0].cells}
    date_cell = cells[columns["service_date_raw"].id]
    description_cell = cells[columns["description"].id]
    assert date_cell.raw_value == "15/07/2026 11:31:00"
    assert date_cell.evidence
    assert "split_from_merged_ocr_token" in date_cell.validation_flags
    assert description_cell.raw_value == "MNEIPI/265586 BED BATH ADULT WIPES GINNI"
    assert description_cell.evidence
    assert "split_from_merged_ocr_token" in description_cell.validation_flags


def test_wide_date_cell_extracts_grounded_date_and_request_prefix() -> None:
    tokens = (
        token(0, "Date", (100, 30, 170, 45)),
        token(1, "Particulars", (300, 30, 500, 45)),
        token(2, "Rate", (650, 30, 710, 45)),
        token(3, "Qty", (760, 30, 800, 45)),
        token(4, "Amount", (880, 30, 970, 45)),
        token(5, "15/07/2026 14:51:00 - MNEIPI/265604", (100, 70, 340, 85)),
        token(6, "FOLEY CATHETER 2 WAY 14 FR", (300, 70, 600, 85)),
        token(7, "210.00", (650, 70, 710, 85)),
        token(8, "1.00", (760, 70, 800, 85)),
        token(9, "210.00", (880, 70, 970, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 110),
    )

    assert len(result.rows) == 1
    candidate = result.rows[0].candidate
    assert candidate.service_date == "15/07/2026"
    assert candidate.request_no == "MNEIPI/265604"
    assert result.rows[0].field_token_ids["service_date"] == ("token-5",)
    assert result.rows[0].field_token_ids["request_no"] == ("token-5",)
    date_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "service_date_raw"
    )
    date_cell = next(
        cell
        for cell in result.source_tables[0].rows[0].cells
        if cell.column_id == date_column.id
    )
    assert date_cell.raw_value == "15/07/2026 14:51:00 - MNEIPI/265604"
    assert date_cell.evidence


def test_printed_table_synthesizes_repeated_unlabeled_numeric_column() -> None:
    tokens = (
        token(0, "Date", (100, 30, 170, 45)),
        token(1, "Particulars", (300, 30, 500, 45)),
        token(2, "Rate", (650, 30, 710, 45)),
        token(3, "Qty", (760, 30, 800, 45)),
        token(4, "Amount", (880, 30, 970, 45)),
        *tuple(
            value
            for row, top in enumerate((70, 100, 130), start=1)
            for value in (
                token(
                    row * 10,
                    f"15/07/2026 11:31:00 - MNEIPI/{265585 + row} Item {row}",
                    (100, top, 600, top + 15),
                ),
                token(row * 10 + 1, "570.00", (650, top, 710, top + 15)),
                token(row * 10 + 2, "1.00", (760, top, 800, top + 15)),
                token(row * 10 + 3, "0.00", (820, top, 860, top + 15)),
                token(row * 10 + 4, "570.00", (880, top, 970, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )

    table = result.source_tables[0]
    assert [column.canonical_field for column in table.columns] == [
        "service_date_raw",
        "description",
        "unit_price",
        "quantity",
        None,
        "net_amount",
    ]
    assert table.columns[4].label == "Column 5"
    assert table.columns[4].validation_flags == ("synthetic_header",)
    for row in table.rows:
        cells = {cell.column_id: cell for cell in row.cells}
        assert cells[table.columns[3].id].raw_value == "1.00"
        assert cells[table.columns[4].id].raw_value == "0.00"
        assert cells[table.columns[5].id].raw_value == "570.00"


def test_printed_summary_synthesizes_all_repeated_unlabeled_amount_lanes() -> None:
    tokens = (
        token(0, "Cash Summary", (120, 20, 390, 35)),
        token(1, "Credit Summary", (870, 20, 1160, 35)),
        token(2, "Total Summary", (1650, 20, 1920, 35)),
        *tuple(
            value
            for row, (top, cash_label, credit_value, total_label, total_value) in enumerate(
                (
                    (60, "Total Cash Sales", "22678.93", "Total Sales", "22678.93"),
                    (90, "Total Cash Return", "2957.97", "Total Return", "2957.97"),
                    (120, "Net Cash Sales", "19720.96", "Total Sales", "19720.96"),
                ),
                start=1,
            )
            for value in (
                token(row * 10, cash_label, (130, top, 420, top + 15)),
                token(row * 10 + 1, "0.00", (680, top, 755, top + 15)),
                token(
                    row * 10 + 2,
                    cash_label.replace("Cash", "Credit"),
                    (870, top, 1180, top + 15),
                ),
                token(row * 10 + 3, credit_value, (1360, top, 1520, top + 15)),
                token(row * 10 + 4, total_label, (1650, top, 1860, top + 15)),
                token(row * 10 + 5, total_value, (2060, top, 2220, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(100, 0, 2300, 160),
    )

    table = result.source_tables[0]
    assert [column.label for column in table.columns] == [
        "Cash Summary",
        "Column 2",
        "Credit Summary",
        "Column 4",
        "Total Summary",
        "Column 6",
    ]
    assert [
        [cell.raw_value for cell in row.cells]
        for row in table.rows
    ] == [
        [
            "Total Cash Sales",
            "0.00",
            "Total Credit Sales",
            "22678.93",
            "Total Sales",
            "22678.93",
        ],
        [
            "Total Cash Return",
            "0.00",
            "Total Credit Return",
            "2957.97",
            "Total Return",
            "2957.97",
        ],
        [
            "Net Cash Sales",
            "0.00",
            "Net Credit Sales",
            "19720.96",
            "Total Sales",
            "19720.96",
        ],
    ]
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    assert [row.canonical_row_id for row in linked[0].rows] == [
        str(row.id) for row in canonical
    ]


@pytest.mark.parametrize("gross_center", (760, 770, 774))
def test_shifted_mapped_numeric_lane_does_not_create_a_synthetic_column(
    gross_center: int,
) -> None:
    tokens = (
        token(0, "Description", (180, 30, 420, 45)),
        token(1, "Gross Amount", (650, 30, 770, 45)),
        token(2, "Net Amount", (860, 30, 960, 45)),
        *tuple(
            value
            for row, top in enumerate((70, 100, 130), start=1)
            for value in (
                token(row * 10, f"Item {row}", (180, top, 420, top + 15)),
                token(
                    row * 10 + 1,
                    f"{row * 100}.00",
                    (gross_center - 25, top, gross_center + 25, top + 15),
                ),
                token(
                    row * 10 + 2,
                    f"{row * 90}.00",
                    (885, top, 935, top + 15),
                ),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 20, 980, 170),
    )

    table = result.source_tables[0]
    assert all(column.validation_flags != ("synthetic_header",) for column in table.columns)
    gross_column = next(
        column for column in table.columns if column.canonical_field == "gross_amount"
    )
    for row, expected in zip(table.rows, ("100.00", "200.00", "300.00"), strict=True):
        cells = {cell.column_id: cell for cell in row.cells}
        assert cells[gross_column.id].raw_value == expected


def test_shifted_rightmost_net_lane_does_not_create_a_synthetic_column() -> None:
    tokens = (
        token(0, "Sr. No.", (80, 30, 150, 45)),
        token(1, "Description", (220, 30, 480, 45)),
        token(2, "Net Amount", (850, 30, 950, 45)),
        *tuple(
            value
            for row, top in enumerate((70, 100, 130), start=1)
            for value in (
                token(row * 10, str(row), (100, top, 130, top + 15)),
                token(row * 10 + 1, f"Item {row}", (220, top, 480, top + 15)),
                token(
                    row * 10 + 2,
                    f"{row * 100}.00",
                    (930, top, 990, top + 15),
                ),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 20, 1000, 170),
    )

    table = result.source_tables[0]
    assert all(
        column.validation_flags != ("synthetic_header",)
        for column in table.columns
    )
    net_column = next(
        column for column in table.columns if column.canonical_field == "net_amount"
    )
    assert [
        next(
            cell.raw_value
            for cell in source_row.cells
            if cell.column_id == net_column.id
        )
        for source_row in table.rows
    ] == ["100.00", "200.00", "300.00"]


def test_left_aligned_total_header_uses_repeated_right_aligned_value_lane() -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "UNITS", (520, 20, 580, 35)),
        token(2, "CHARGES", (650, 20, 760, 35)),
        token(3, "TOTAL", (830, 20, 900, 35)),
        *tuple(
            value
            for row, (top, description, quantity, charge, total) in enumerate(
                (
                    (60, "Registration Charges", "1", "500.00", "500.00"),
                    (90, "Single Room A/C", "4", "2500.00", "10000.00"),
                    (120, "Lactation Counselling", "1", "1500.00", "1500.00"),
                ),
                start=1,
            )
            for value in (
                token(row * 10, description, (100, top, 390, top + 15)),
                token(row * 10 + 1, quantity, (530, top, 570, top + 15)),
                token(row * 10 + 2, charge, (690, top, 760, top + 15)),
                token(row * 10 + 3, total, (900, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 160),
    )

    assert [row.candidate.description for row in result.rows] == [
        "Registration Charges",
        "Single Room A/C",
        "Lactation Counselling",
    ]
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("500.00"),
        Decimal("10000.00"),
        Decimal("1500.00"),
    ]
    assert result.diagnostics["column_centers"]["amount"] > 0.9
    table = result.source_tables[0]
    assert [column.label for column in table.columns] == [
        "DESCRIPTION",
        "UNITS",
        "CHARGES",
        "TOTAL",
    ]
    assert [
        [cell.raw_value for cell in row.cells]
        for row in table.rows
    ] == [
        ["Registration Charges", "1", "500.00", "500.00"],
        ["Single Room A/C", "4", "2500.00", "10000.00"],
        ["Lactation Counselling", "1", "1500.00", "1500.00"],
    ]


def test_left_aligned_total_header_uses_integer_arithmetic_value_lane() -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "UNITS", (520, 20, 580, 35)),
        token(2, "CHARGES", (650, 20, 760, 35)),
        token(3, "TOTAL", (830, 20, 900, 35)),
        *tuple(
            value
            for row, (top, description, quantity, charge, total) in enumerate(
                (
                    (60, "Registration Charges", "1", "500", "500"),
                    (90, "Single Room A/C", "4", "2500", "10000"),
                    (120, "Lactation Counselling", "1", "1500", "1500"),
                ),
                start=1,
            )
            for value in (
                token(row * 10, description, (100, top, 390, top + 15)),
                token(row * 10 + 1, quantity, (530, top, 570, top + 15)),
                token(row * 10 + 2, charge, (690, top, 760, top + 15)),
                token(row * 10 + 3, total, (900, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 160),
    )

    assert [row.candidate.amount for row in result.rows] == [
        Decimal("500"),
        Decimal("10000"),
        Decimal("1500"),
    ]
    assert result.diagnostics["column_centers"]["amount"] > 0.9


def test_left_aligned_total_header_accepts_two_consistent_rows() -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "UNITS", (520, 20, 580, 35)),
        token(2, "CHARGES", (650, 20, 760, 35)),
        token(3, "TOTAL", (830, 20, 900, 35)),
        token(10, "Registration Charges", (100, 60, 390, 75)),
        token(11, "1", (530, 60, 570, 75)),
        token(12, "500", (690, 60, 760, 75)),
        token(13, "500", (900, 60, 980, 75)),
        token(20, "Single Room A/C", (100, 90, 390, 105)),
        token(21, "4", (530, 90, 570, 105)),
        token(22, "2500", (690, 90, 760, 105)),
        token(23, "10000", (900, 90, 980, 105)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 130),
    )

    assert [row.candidate.amount for row in result.rows] == [
        Decimal("500"),
        Decimal("10000"),
    ]
    assert result.diagnostics["column_centers"]["amount"] > 0.9


def test_left_aligned_total_header_rejects_ambiguous_right_side_lanes() -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "TOTAL", (790, 20, 850, 35)),
        *tuple(
            value
            for row, (top, description, total, exterior) in enumerate(
                (
                    (60, "Registration Charges", "500.00", "50.00"),
                    (90, "Single Room A/C", "1000.00", "100.00"),
                    (120, "Lactation Counselling", "1500.00", "150.00"),
                ),
                start=1,
            )
            for value in (
                token(row * 10, description, (100, top, 390, top + 15)),
                token(row * 10 + 1, total, (860, top, 920, top + 15)),
                token(row * 10 + 2, exterior, (940, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 160),
    )

    assert result.diagnostics["column_centers"]["amount"] < 0.86
    assert all(
        row.candidate.amount not in {
            Decimal("50.00"),
            Decimal("100.00"),
            Decimal("150.00"),
        }
        for row in result.rows
    )


@pytest.mark.parametrize(
    "references",
    (
        ("101", "102", "103"),
        ("101.01", "102.02", "103.03"),
    ),
)
def test_left_aligned_total_header_rejects_unlabeled_reference_lane(
    references: tuple[str, str, str],
) -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "TOTAL", (790, 20, 850, 35)),
        *tuple(
            value
            for row, (top, description, reference) in enumerate(
                zip(
                    (60, 90, 120),
                    (
                        "Registration Charges",
                        "Single Room A/C",
                        "Lactation Counselling",
                    ),
                    references,
                    strict=True,
                ),
                start=1,
            )
            for value in (
                token(row * 10, description, (100, top, 390, top + 15)),
                token(row * 10 + 1, reference, (940, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 160),
    )

    assert result.diagnostics["column_centers"]["amount"] < 0.86
    assert all(row.candidate.amount is None for row in result.rows)


def test_left_aligned_total_header_rejects_mixed_arithmetic_exterior_lane() -> None:
    rows = tuple(
        (
            60 + index * 25,
            f"Charge {index}",
            "1",
            str(index * 100),
            str(index * 100 if index <= 2 else 9000 + index),
        )
        for index in range(1, 11)
    )
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "UNITS", (520, 20, 580, 35)),
        token(2, "CHARGES", (650, 20, 760, 35)),
        token(3, "TOTAL", (790, 20, 850, 35)),
        *tuple(
            value
            for row, (top, description, quantity, charge, exterior) in enumerate(
                rows,
                start=1,
            )
            for value in (
                token(row * 10, description, (100, top, 390, top + 15)),
                token(row * 10 + 1, quantity, (530, top, 570, top + 15)),
                token(row * 10 + 2, charge, (690, top, 760, top + 15)),
                token(row * 10 + 3, exterior, (940, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 340),
    )

    assert result.diagnostics["column_centers"]["amount"] < 0.86
    assert all(
        row.candidate.amount not in {
            Decimal("100"),
            Decimal("200"),
            *(Decimal(str(9000 + index)) for index in range(3, 11)),
        }
        for row in result.rows
    )


@pytest.mark.parametrize("intermediate_header", ("CORPORATE", "DISCHARGE", "SEPARATE"))
def test_left_aligned_total_header_requires_explicit_financial_lane_label(
    intermediate_header: str,
) -> None:
    tokens = (
        token(0, "DESCRIPTION", (100, 20, 260, 35)),
        token(1, "UNITS", (520, 20, 580, 35)),
        token(2, intermediate_header, (650, 20, 760, 35)),
        token(3, "TOTAL", (790, 20, 850, 35)),
        *tuple(
            value
            for row, (top, quantity, intermediate, exterior) in enumerate(
                (
                    (60, "1", "500", "500"),
                    (90, "2", "500", "1000"),
                    (120, "3", "500", "1500"),
                ),
                start=1,
            )
            for value in (
                token(row * 10, f"Charge {row}", (100, top, 390, top + 15)),
                token(row * 10 + 1, quantity, (530, top, 570, top + 15)),
                token(row * 10 + 2, intermediate, (690, top, 760, top + 15)),
                token(row * 10 + 3, exterior, (940, top, 980, top + 15)),
            )
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 0, 1000, 160),
    )

    assert result.diagnostics["column_centers"]["amount"] < 0.86


@pytest.mark.parametrize(
    ("printed_code", "code_box", "rotated"),
    (
        ("12345", (530, 130, 600, 145), False),
        ("ABC", (530, 130, 600, 145), False),
        ("SVC 42", (530, 130, 600, 145), False),
        ("AB_12", (530, 130, 600, 145), False),
        ("ABC", (535, 120, 605, 155), False),
        ("ABC", (430, 120, 500, 155), False),
        ("LB012 PATNA", (430, 120, 560, 155), True),
    ),
)
def test_plausible_unparsed_service_code_remains_for_strict_validation(
    printed_code: str,
    code_box: tuple[float, float, float, float],
    rotated: bool,
) -> None:
    printed_code_token = (
        rotated_token(9, printed_code, code_box, 10)
        if rotated
        else token(9, printed_code, code_box)
    )
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Service Code", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        printed_code_token,
        token(10, "20/01/2026 09:44:58", (700, 130, 850, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.service_code is None
    code_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_code"
    )
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    code_cell = next(
        cell
        for cell in gloves_source_row.cells
        if cell.column_id == code_column.id
    )
    assert code_cell.raw_value == printed_code
    assert code_cell.evidence
    assert "excluded_oversized_overlay" not in code_cell.validation_flags


@pytest.mark.parametrize("printed_code", ("12345", "ABC", "SVC 42", "AB_12"))
def test_rotated_stamp_cannot_erase_an_aligned_code_in_the_same_cell(
    printed_code: str,
) -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Service Code", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(9, "PATNA", (370, 115, 440, 150), 10),
        token(10, printed_code, (535, 130, 605, 145)),
        token(11, "20/01/2026 09:44:58", (700, 130, 850, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    assert gloves.service_code is None
    code_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_code"
    )
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    code_cell = next(
        cell
        for cell in gloves_source_row.cells
        if cell.column_id == code_column.id
    )
    assert printed_code in (code_cell.raw_value or "")
    assert code_cell.evidence
    assert "excluded_oversized_overlay" not in code_cell.validation_flags


def test_overlay_filter_resolves_reordered_source_cells_by_column_id() -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Service Code", (520, 30, 620, 45)),
        token(2, "Date", (700, 30, 760, 45)),
        token(3, "Net Amount", (870, 30, 970, 45)),
        token(4, "Package Name : Coronary Angiography", (100, 70, 430, 85)),
        token(5, "20/01/2026 - 21/01/2026", (700, 70, 850, 85)),
        token(6, "11457.00", (880, 70, 960, 85)),
        token(7, "Pharmacy", (100, 100, 230, 115)),
        token(8, "Gloves Sterile 7", (100, 130, 300, 145)),
        rotated_token(9, "PATNA", (430, 120, 500, 155), 10),
        token(10, "20/01/2026 09:44:58", (700, 130, 850, 145)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 170),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    reordered = tuple(
        table.model_copy(
            update={
                "rows": tuple(
                    row.model_copy(update={"cells": tuple(reversed(row.cells))})
                    for row in table.rows
                )
            }
        )
        for table in result.source_tables
    )

    linked = _link_source_tables(reordered, canonical)

    gloves = next(row for row in canonical if row.description == "Gloves Sterile 7")
    gloves_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(gloves.id)
    )
    cells = {cell.column_id: cell for cell in gloves_source_row.cells}
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    code_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_code"
    )
    assert cells[description_column.id].raw_value == "Gloves Sterile 7"
    assert cells[code_column.id].raw_value is None


def test_wrapped_repeated_header_updates_columns_without_becoming_a_row() -> None:
    tokens = (
        token(0, "Service", (100, 20, 180, 32)),
        token(1, "Net", (890, 20, 930, 32)),
        token(2, "Service Name", (100, 42, 240, 54)),
        token(3, "Date", (650, 42, 710, 54)),
        token(4, "Code", (520, 64, 580, 76)),
        token(5, "Amount", (870, 64, 960, 76)),
        token(6, "First service", (100, 100, 280, 114)),
        token(7, "LB001", (520, 100, 580, 114)),
        token(8, "20/01/2026", (650, 100, 750, 114)),
        token(9, "100.00", (870, 100, 950, 114)),
        token(10, "Service", (100, 140, 180, 152)),
        token(11, "Net", (890, 140, 930, 152)),
        token(12, "Service Name", (100, 162, 240, 174)),
        token(13, "Date", (650, 162, 710, 174)),
        token(14, "Code", (520, 184, 580, 196)),
        token(15, "Amount", (870, 184, 960, 196)),
        token(16, "Second service", (100, 220, 280, 234)),
        token(17, "LB002", (520, 220, 580, 234)),
        token(18, "21/01/2026", (650, 220, 750, 234)),
        token(19, "200.00", (870, 220, 950, 234)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 250),
    )

    assert [row.candidate.description for row in result.rows] == [
        "First service",
        "Second service",
    ]
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("100.00"),
        Decimal("200.00"),
    ]
    assert [row.candidate.service_date for row in result.rows] == [
        "20/01/2026",
        "21/01/2026",
    ]
    assert result.diagnostics["header_segments"] == 2


def test_description_before_subtotal_extends_previous_serial_row() -> None:
    tokens = (
        token(0, "Sr. No.", (80, 30, 140, 45)),
        token(1, "Service Name", (200, 30, 430, 45)),
        token(2, "Bill Amount", (650, 30, 760, 45)),
        token(3, "Discount Amount", (770, 30, 860, 45)),
        token(4, "Total Amount", (870, 30, 970, 45)),
        token(5, "1", (90, 70, 105, 85)),
        token(6, "Package(IPD) - Coronary", (200, 70, 500, 85)),
        token(7, "11457.00", (880, 70, 960, 85)),
        token(8, "Angiography (CAG)", (200, 100, 430, 115)),
        token(9, "Total", (200, 130, 280, 145)),
        token(10, "11457.00", (880, 130, 960, 145)),
        token(11, "Total Bill Amount", (650, 160, 830, 175)),
        token(12, "11457.00", (880, 160, 960, 175)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(70, 20, 980, 190),
    )
    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == ("Package(IPD) - Coronary Angiography (CAG)")
    assert result.rows[0].candidate.amount == Decimal("11457.00")
    assert result.rows[0].candidate.role is RowRole.CATEGORY_ROLLUP
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    linked_row = next(row for row in linked[0].rows if row.canonical_row_id is not None)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    assert next(
        cell.raw_value
        for cell in linked_row.cells
        if cell.column_id == description_column.id
    ) == "Package(IPD) - Coronary Angiography (CAG)"


def test_nonanchored_note_before_total_stays_separate_in_printed_table() -> None:
    tokens = (
        token(0, "Description", (100, 30, 300, 45)),
        token(1, "Amount", (870, 30, 970, 45)),
        token(2, "Procedure Charges", (100, 70, 330, 85)),
        token(3, "100.00", (880, 70, 960, 85)),
        token(4, "Insurance Note", (450, 100, 550, 115)),
        token(5, "Total", (770, 130, 830, 145)),
        token(6, "100.00", (880, 130, 960, 145)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 160),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    printed_descriptions = [
        next(
            cell.raw_value
            for cell in source_row.cells
            if cell.column_id == description_column.id
        )
        for source_row in linked[0].rows
        if any(
            cell.raw_value
            for cell in source_row.cells
            if cell.column_id == description_column.id
        )
    ]

    assert [row.description for row in canonical] == ["Procedure Charges"]
    assert printed_descriptions[:2] == ["Procedure Charges", "Insurance Note"]


def test_payment_details_before_total_does_not_extend_previous_charge() -> None:
    tokens = (
        token(0, "Service Name", (200, 30, 430, 45)),
        token(1, "Total Amount", (870, 30, 970, 45)),
        token(2, "Procedure Charges", (200, 70, 430, 85)),
        token(3, "11,457.00", (880, 70, 960, 85)),
        token(4, "Payment Details", (600, 100, 760, 115)),
        token(5, "Total Bill Amount", (650, 130, 830, 145)),
        token(6, "11,457.00", (880, 130, 960, 145)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(180, 20, 980, 160),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Procedure Charges"
    assert "token-4" not in result.rows[0].evidence_token_ids

    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    payment_source_row = next(
        row
        for row in linked[0].rows
        if any(cell.raw_value == "Payment Details" for cell in row.cells)
    )
    assert payment_source_row.canonical_row_id is None


@pytest.mark.parametrize(
    "footer_parts",
    (
        ("Page 1 of 2",),
        ("Page", "1", "of", "2"),
        ("Page 1", "of 2"),
        ("Page", "1/2"),
        ("This bill was created using PRESCO IPD",),
    ),
)
def test_printed_table_stops_at_explicit_bill_page_footer(
    footer_parts: tuple[str, ...],
) -> None:
    tokens = (
        token(0, "Particular", (100, 30, 420, 45)),
        token(1, "Amount", (600, 30, 700, 45)),
        token(2, "Unit/Days", (760, 30, 840, 45)),
        token(3, "Registration", (100, 70, 360, 85)),
        token(4, "300.00", (620, 70, 690, 85)),
        token(5, "1", (780, 70, 800, 85)),
        token(6, "10/07/26,", (20, 110, 95, 125)),
        *tuple(
            token(
                7 + index,
                part,
                (500 + index * 70, 110, 560 + index * 70, 125),
            )
            for index, part in enumerate(footer_parts)
        ),
        token(20, "13", (620, 150, 650, 165)),
        token(21, "Background dashboard", (100, 180, 360, 195)),
        token(22, "4177", (620, 180, 680, 195)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(0, 20, 900, 220),
    )

    assert len(result.source_tables) == 1
    table = result.source_tables[0]
    assert len(table.rows) == 1
    values = {
        column.label: next(
            cell.raw_value
            for cell in table.rows[0].cells
            if cell.column_id == column.id
        )
        for column in table.columns
    }
    assert values == {
        "Particular": "Registration",
        "Amount": "300.00",
        "Unit/Days": "1",
    }
    assert [
        (row.candidate.description, row.candidate.amount) for row in result.rows
    ] == [("Registration", Decimal("300.00"))]


@pytest.mark.parametrize(
    "payment_heading",
    (
        (token(12, "Payment Details", (600, 130, 760, 145)),),
        (
            token(12, "Payment", (600, 130, 680, 145)),
            token(18, "Details", (690, 130, 760, 145)),
        ),
        (token(12, "ID 410 Payment Details", (600, 130, 760, 145)),),
        (token(12, "410 Payment Details", (600, 130, 760, 145)),),
        (token(12, "Payment Details 410.00", (600, 130, 800, 145)),),
        (token(12, "Payment Details: 410", (600, 130, 800, 145)),),
    ),
)
@pytest.mark.parametrize(
    "margin_text",
    ("410", "ID 410", "#410", "23-Jul-2026"),
)
@pytest.mark.parametrize(
    "margin_box",
    ((465, 130, 555, 145), (50, 130, 140, 145)),
)
@pytest.mark.parametrize("payment_amount", (None, "410.00"))
def test_payment_heading_ends_pending_charge_before_polluted_summary(
    payment_heading: tuple[OcrToken, ...],
    margin_text: str,
    margin_box: tuple[int, int, int, int],
    payment_amount: str | None,
) -> None:
    tokens = (
        token(0, "Sr.N", (50, 30, 90, 45)),
        token(1, "Particular", (100, 30, 420, 45)),
        token(2, "Amount Rs. Unit/Days", (610, 30, 820, 45)),
        token(3, "Total", (880, 30, 970, 45)),
        token(4, "1.", (50, 70, 70, 85)),
        token(5, "Registration", (100, 70, 360, 85)),
        token(6, "300.00", (620, 70, 700, 85)),
        token(7, "1", (760, 70, 780, 85)),
        token(8, "300", (890, 70, 950, 85)),
        token(9, "13.", (50, 100, 75, 115)),
        token(10, "Others-ENEMA PROCEDURE", (100, 100, 430, 115)),
        token(11, margin_text, margin_box),
        *payment_heading,
        *(
            (token(19, payment_amount, (890, 130, 960, 145)),)
            if payment_amount is not None
            else ()
        ),
        token(13, "70855", (0, 160, 50, 175)),
        token(14, "Total Bill Amount", (650, 160, 830, 175)),
        token(15, "1,03,276.00", (880, 160, 970, 175)),
        token(16, "Discount (Rs.):", (760, 190, 870, 205)),
        token(17, "0.00", (900, 190, 960, 205)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 1000, 220),
    )

    assert [row.candidate.description for row in result.rows] == ["Registration"]
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    unpriced_row = next(
        row
        for row in linked[0].rows
        if any(
            cell.column_id == description_column.id
            and cell.raw_value == "Others-ENEMA PROCEDURE"
            for cell in row.cells
        )
    )
    assert unpriced_row.canonical_row_id is None


def test_headerless_inherited_schema_uses_description_lane_for_payment_footer() -> None:
    first = reconstruct_ocr_rows(
        (
            token(0, "Particular", (100, 30, 420, 45)),
            token(1, "Rate", (610, 30, 700, 45)),
            token(2, "Qty", (750, 30, 800, 45)),
            token(3, "Amount", (880, 30, 970, 45)),
            token(4, "Registration", (100, 70, 360, 85)),
            token(5, "300.00", (620, 70, 700, 85)),
            token(6, "1", (760, 70, 780, 85)),
            token(7, "300", (890, 70, 950, 85)),
        ),
        page_number=1,
        table_id="p1-t1",
        box=(40, 20, 1000, 100),
    )
    continuation = tuple(
        value.model_copy(
            update={"page_number": 2, "token_id": f"p2-{value.token_id}"}
        )
        for value in (
            token(8, "Registration", (100, 30, 360, 45)),
            token(9, "300.00", (620, 30, 700, 45)),
            token(10, "1", (760, 30, 780, 45)),
            token(11, "300", (890, 30, 950, 45)),
            token(12, "Others-ENEMA PROCEDURE", (100, 60, 430, 75)),
            token(13, "Payment", (700, 90, 770, 105)),
            token(14, "Details", (780, 90, 850, 105)),
            token(15, "Total Bill Amount", (650, 120, 830, 135)),
            token(16, "300", (890, 120, 950, 135)),
        )
    )

    second = reconstruct_ocr_rows(
        continuation,
        page_number=2,
        table_id="p2-t1",
        box=(40, 20, 1000, 150),
        prior_schemas=(first.schema,) if first.schema else (),
    )

    assert second.diagnostics["schema_inherited"] is True
    assert [row.candidate.description for row in second.rows] == ["Registration"]


def test_wrapped_payment_details_charge_in_description_lane_is_retained() -> None:
    tokens = (
        token(0, "Description", (100, 30, 430, 45)),
        token(1, "Amount", (870, 30, 970, 45)),
        token(2, "Payment", (100, 70, 180, 85)),
        token(3, "Details", (190, 70, 260, 85)),
        token(4, "Charge", (306, 70, 380, 85)),
        token(5, "100.00", (880, 100, 960, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Payment Details Charge"
    assert result.rows[0].candidate.amount == Decimal("100.00")


@pytest.mark.parametrize(
    "description",
    ("Payment Details", "Payment Details Charge"),
)
def test_priced_payment_description_is_not_treated_as_footer_heading(
    description: str,
) -> None:
    tokens = (
        token(0, "Description", (100, 30, 430, 45)),
        token(1, "Amount", (870, 30, 970, 45)),
        token(2, description, (100, 70, 350, 85)),
        token(3, "100.00", (880, 70, 960, 85)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == description
    assert result.rows[0].candidate.amount == Decimal("100.00")


@pytest.mark.parametrize(
    "footer_text",
    (
        "Payment Details",
        "Receipt Information",
        "Settlement Mode",
        "Payment Summary",
        "Receipt History",
        "Settlement Status",
        "Payment Breakup",
    ),
)
def test_connector_before_payment_heading_keeps_linked_printed_description(
    footer_text: str,
) -> None:
    tokens = (
        token(0, "Description", (100, 30, 430, 45)),
        token(1, "Amount", (870, 30, 970, 45)),
        token(2, "Procedure Charges-", (100, 70, 350, 85)),
        token(3, "11,457.00", (880, 70, 960, 85)),
        token(4, footer_text, (100, 100, 270, 115)),
        token(5, "Total Bill Amount", (650, 130, 830, 145)),
        token(6, "11,457.00", (880, 130, 960, 145)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 160),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Procedure Charges"
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    charge_source_row = next(
        row
        for row in linked[0].rows
        if row.canonical_row_id == str(canonical[0].id)
    )
    printed_description = next(
        cell.raw_value
        for cell in charge_source_row.cells
        if cell.column_id == description_column.id
    )
    assert printed_description is not None
    assert printed_description.rstrip(" -") == canonical[0].description
    assert footer_text not in printed_description
    payment_source_row = next(
        row
        for row in linked[0].rows
        if any(cell.raw_value == footer_text for cell in row.cells)
    )
    assert payment_source_row.canonical_row_id is None


@pytest.mark.parametrize("footer_text", ("Amount Paid", "Amount Received"))
def test_connector_before_amount_footer_does_not_merge_source_footer(
    footer_text: str,
) -> None:
    tokens = (
        token(0, "Description", (100, 30, 430, 45)),
        token(1, "Amount", (870, 30, 970, 45)),
        token(2, "Procedure Charges-", (100, 70, 350, 85)),
        token(3, "11,457.00", (880, 70, 960, 85)),
        token(4, footer_text, (100, 100, 270, 115)),
        token(5, "Total Bill Amount", (650, 130, 830, 145)),
        token(6, "11,457.00", (880, 130, 960, 145)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 160),
    )

    assert result.rows[0].candidate.description == "Procedure Charges"
    description_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "description"
    )
    assert [
        cell.raw_value
        for row in result.source_tables[0].rows
        for cell in row.cells
        if cell.column_id == description_column.id and cell.raw_value
    ][:2] == ["Procedure Charges-", footer_text]


def test_headerless_continuation_inherits_date_without_inventing_amount() -> None:
    header = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Date", (650, 30, 710, 45)),
        token(2, "Net Amount", (870, 30, 970, 45)),
        token(3, "Package", (100, 70, 280, 85)),
        token(4, "20/01/2026", (650, 70, 750, 85)),
        token(5, "100.00", (880, 70, 960, 85)),
    )
    first = reconstruct_ocr_rows(
        header,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 100),
    )
    continuation = tuple(
        value.model_copy(update={"page_number": 2, "token_id": f"p2-{value.token_id}"})
        for value in (
            token(6, "Included medicine", (100, 30, 330, 45)),
            token(7, "21/01/2026 11:47:02", (650, 30, 800, 45)),
        )
    )
    second = reconstruct_ocr_rows(
        continuation,
        page_number=2,
        table_id="p2-t1",
        box=(80, 20, 980, 70),
        prior_schemas=(first.schema,) if first.schema else (),
    )
    assert second.diagnostics["schema_inherited"] is True
    assert len(second.rows) == 1
    assert second.rows[0].candidate.role is RowRole.INFORMATIONAL
    assert second.rows[0].candidate.service_date == "21/01/2026 11:47:02"
    assert second.rows[0].candidate.amount is None


def test_rows_before_later_section_header_inherit_the_prior_page_schema() -> None:
    first = reconstruct_ocr_rows(
        (
            token(0, "Description", (100, 20, 300, 35)),
            token(1, "Date", (500, 20, 560, 35)),
            token(2, "UnitPrice", (530, 20, 610, 35)),
            token(3, "Quantity", (620, 20, 700, 35)),
            token(4, "Amount", (900, 20, 970, 35)),
            token(5, "Urine Culture", (100, 55, 300, 70)),
            token(6, "07/02/2026", (500, 55, 580, 70)),
            token(7, "460.00", (540, 55, 600, 70)),
            token(8, "1.00", (630, 55, 690, 70)),
            token(9, "460.00", (910, 55, 960, 70)),
        ),
        page_number=1,
        table_id="p1-t1",
        box=(80, 10, 980, 90),
    )
    page_two = tuple(
        value.model_copy(update={"page_number": 2, "token_id": f"p2-{value.token_id}"})
        for value in (
            token(10, "Glucose Random", (100, 10, 300, 25)),
            token(11, "08/02/2026", (500, 10, 580, 25)),
            token(12, "40.00", (690, 10, 750, 25)),
            token(13, "1.00", (790, 10, 830, 25)),
            token(14, "40.00", (890, 10, 950, 25)),
            token(15, "Renal Function Test", (100, 40, 300, 55)),
            token(16, "08/02/2026", (500, 40, 580, 55)),
            token(17, "500.00", (690, 40, 750, 55)),
            token(18, "1.00", (790, 40, 830, 55)),
            token(19, "500.00", (890, 40, 950, 55)),
            token(20, "Glucose Random", (100, 70, 300, 85)),
            token(21, "09/02/2026", (500, 70, 580, 85)),
            token(22, "40.00", (690, 70, 750, 85)),
            token(23, "5.00", (790, 70, 830, 85)),
            token(24, "200.00", (890, 70, 950, 85)),
            token(25, "Description", (100, 105, 300, 120)),
            token(26, "Date", (500, 105, 560, 120)),
            token(27, "UnitPrice", (680, 105, 750, 120)),
            token(28, "Quantity", (780, 105, 840, 120)),
            token(29, "Amount", (880, 105, 960, 120)),
            token(30, "CT Whole Abdomen", (100, 140, 300, 155)),
            token(31, "06/02/2026", (500, 140, 580, 155)),
            token(32, "3450.00", (690, 140, 750, 155)),
            token(33, "1.00", (790, 140, 830, 155)),
            token(34, "3450.00", (890, 140, 950, 155)),
        )
    )

    second = reconstruct_ocr_rows(
        page_two,
        page_number=2,
        table_id="p2-t1",
        box=(80, 0, 980, 175),
        prior_schemas=(first.schema,) if first.schema else (),
    )

    assert [row.candidate.description for row in second.rows] == [
        "Glucose Random",
        "Renal Function Test",
        "Glucose Random",
        "CT Whole Abdomen",
    ]
    assert [row.candidate.amount for row in second.rows] == [
        Decimal("40.00"),
        Decimal("500.00"),
        Decimal("200.00"),
        Decimal("3450.00"),
    ]
    assert [row.candidate.service_date for row in second.rows] == [
        "08/02/2026",
        "08/02/2026",
        "09/02/2026",
        "06/02/2026",
    ]
    assert [row.candidate.rate for row in second.rows] == [
        Decimal("40.00"),
        Decimal("500.00"),
        Decimal("40.00"),
        Decimal("3450.00"),
    ]
    assert [row.candidate.quantity for row in second.rows] == [
        Decimal("1.00"),
        Decimal("1.00"),
        Decimal("5.00"),
        Decimal("1.00"),
    ]
    assert len({table.id for table in second.source_tables}) == len(
        second.source_tables
    )
    assert any(
        "pre_header_continuation" in table.validation_flags
        for table in second.source_tables
    )


def test_leading_dash_suffixed_date_is_folded_into_linked_source_row() -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Date", (650, 30, 710, 45)),
        token(2, "Net Amount", (870, 30, 970, 45)),
        token(3, "20/01/2026 -", (650, 70, 760, 85)),
        token(4, "Package Name: Coronary Angiography", (100, 100, 430, 115)),
        token(5, "21/01/2026", (650, 100, 750, 115)),
        token(6, "11,457.00", (880, 100, 960, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    date_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "service_date_raw"
    )
    source_row = next(row for row in linked[0].rows if row.canonical_row_id)
    date_cell = next(
        cell for cell in source_row.cells if cell.column_id == date_column.id
    )
    canonical_date_ids = {
        token_id
        for item in canonical[0].field_evidence["service_date"]
        for token_id in item.token_ids
    }
    source_date_ids = {
        token_id for item in date_cell.evidence for token_id in item.token_ids
    }

    assert canonical[0].service_date_raw == "20/01/2026 - 21/01/2026"
    assert date_cell.raw_value == "20/01/2026 - 21/01/2026"
    assert canonical_date_ids <= source_date_ids


def test_date_only_line_can_ground_following_row_without_current_date_token() -> None:
    tokens = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Date", (650, 30, 710, 45)),
        token(2, "Net Amount", (870, 30, 970, 45)),
        token(3, "20/01/2026", (650, 70, 750, 85)),
        token(4, "Package Name: Coronary Angiography", (100, 100, 430, 115)),
        token(5, "11,457.00", (880, 100, 960, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 130),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == (
        "Package Name: Coronary Angiography"
    )
    assert result.rows[0].candidate.service_date == "20/01/2026"
    assert result.rows[0].candidate.amount == Decimal("11457.00")
    assert set(result.rows[0].field_token_ids["service_date"]) == {"token-3"}


def test_headerless_date_only_continuation_keeps_every_printed_row_linkable() -> None:
    header = (
        token(0, "Service Name", (100, 30, 300, 45)),
        token(1, "Date", (650, 30, 710, 45)),
        token(2, "Net Amount", (870, 30, 970, 45)),
        token(3, "Package", (100, 70, 280, 85)),
        token(4, "20/01/2026", (650, 70, 750, 85)),
        token(5, "100.00", (880, 70, 960, 85)),
    )
    first = reconstruct_ocr_rows(
        header,
        page_number=1,
        table_id="p1-t1",
        box=(80, 20, 980, 100),
    )
    continuation = tuple(
        value.model_copy(update={"page_number": 2, "token_id": f"p2-{value.token_id}"})
        for value in (
            token(6, "20/01/2026 09:44:58", (650, 5, 800, 20)),
            token(7, "Euroflex Ns 500ml", (100, 40, 330, 55)),
            token(8, "20/01/2026 09:44:58", (650, 40, 800, 55)),
            token(9, "Extension Set", (100, 70, 300, 85)),
            token(10, "20/01/2026 09:44:58", (650, 70, 800, 85)),
            token(11, "Total", (770, 100, 830, 115)),
            token(12, "100.00", (880, 100, 960, 115)),
        )
    )

    second = reconstruct_ocr_rows(
        continuation,
        page_number=2,
        table_id="p2-t1",
        box=(80, 0, 980, 125),
        prior_schemas=(first.schema,) if first.schema else (),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        second.rows,
    )
    linked = _link_source_tables(second.source_tables, canonical)
    linked_descriptions = {
        row.canonical_row_id
        for table in linked
        for row in table.rows
        if row.canonical_row_id is not None
    }

    assert [row.candidate.description for row in second.rows] == [
        "Euroflex Ns 500ml",
        "Extension Set",
    ]
    assert linked_descriptions == {str(row.id) for row in canonical}
    assert any(
        "Euroflex Ns 500ml" in {cell.raw_value for cell in row.cells}
        for table in second.source_tables
        for row in table.rows
    )
    source_token_ids = {
        token_id
        for table in second.source_tables
        for row in table.rows
        for cell in row.cells
        for item in cell.evidence
        for token_id in item.token_ids
    }
    assert "p2-token-6" in source_token_ids


def test_client_pharmacy_aliases_keep_date_product_gross_and_net_in_their_lanes() -> None:
    tokens = (
        token(0, "BillDate", (80, 25, 150, 40)),
        token(1, "Bill Number", (170, 25, 260, 40)),
        token(2, "ProductName", (290, 25, 430, 40)),
        token(3, "Qty", (610, 25, 650, 40)),
        token(4, "Rate", (700, 25, 750, 40)),
        token(5, "Service Amt", (790, 25, 870, 40)),
        token(6, "Total", (910, 25, 960, 40)),
        token(7, "20/07/2026", (80, 70, 150, 85)),
        token(8, "B-441", (170, 70, 250, 85)),
        token(9, "Ceftriaxone 1g Inj", (290, 70, 500, 85)),
        token(10, "2", (620, 70, 635, 85)),
        token(11, "125.00", (700, 70, 755, 85)),
        token(12, "250.00", (800, 70, 860, 85)),
        token(13, "225.00", (910, 70, 965, 85)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 15, 980, 100),
    )
    assert len(result.rows) == 1
    candidate = result.rows[0].candidate
    assert candidate.description == "Ceftriaxone 1g Inj"
    assert candidate.service_date == "20/07/2026"
    assert candidate.request_no == "B-441"
    assert candidate.quantity == Decimal("2")
    assert candidate.rate == Decimal("125.00")
    assert candidate.gross_amount == Decimal("250.00")
    assert candidate.amount == Decimal("225.00")


@pytest.mark.parametrize("merged_right", (440, 650, 850))
@pytest.mark.parametrize("timestamp_separator", (" ", ", ", "; ", " - ", ".", ". "))
@pytest.mark.parametrize("request_prefix", ("PI", "P|"))
@pytest.mark.parametrize("request_suffix", ("", "."))
def test_pharmacy_expiry_date_does_not_replace_left_transaction_date(
    merged_right: int,
    timestamp_separator: str,
    request_prefix: str,
    request_suffix: str,
) -> None:
    tokens = (
        token(0, "Date/ Time", (80, 25, 170, 40)),
        token(1, "Bill Number", (200, 25, 290, 40)),
        token(2, "ProductName", (320, 25, 450, 40)),
        token(3, "Batch No", (500, 25, 570, 40)),
        token(4, "Expiry", (620, 25, 680, 40)),
        token(5, "Date", (690, 25, 730, 40)),
        token(6, "Qty", (770, 25, 805, 40)),
        token(7, "Rate", (840, 25, 885, 40)),
        token(8, "Total", (920, 25, 970, 40)),
        token(
            9,
            f"10/07/2026{timestamp_separator}02:48 am "
            f"{request_prefix}261015227{request_suffix} Patient Coat",
            (80, 70, merged_right, 85),
        ),
        token(10, "Patient Coat", (320, 70, 430, 85)),
        token(12, "62104070", (500, 70, 570, 85)),
        token(13, "1", (780, 70, 795, 85)),
        token(14, "250", (845, 70, 880, 85)),
        token(15, "250", (930, 70, 965, 85)),
        token(
            16,
            f"10/07/2026{timestamp_separator}02:49 am "
            f"{request_prefix}261015228{request_suffix}",
            (80, 100, 290, 115),
        ),
        token(18, "Betadine Scrub 50 ML", (320, 100, 465, 115)),
        token(19, "MD06126", (500, 100, 570, 115)),
        token(20, "30/09/2027", (630, 100, 725, 115)),
        token(21, "1", (780, 100, 795, 115)),
        token(22, "107.1", (840, 100, 885, 115)),
        token(23, "107.1", (920, 100, 970, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(60, 15, 980, 130),
    )

    assert [row.candidate.description for row in result.rows] == [
        "Patient Coat",
        "Betadine Scrub 50 ML",
    ]
    assert [row.candidate.service_date for row in result.rows] == [
        "10/07/2026",
        "10/07/2026",
    ]
    assert [row.candidate.amount for row in result.rows] == [
        Decimal("250"),
        Decimal("107.1"),
    ]
    assert result.diagnostics["column_centers"]["service_date"] < 0.2
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    columns_by_field = {
        column.canonical_field: column
        for column in linked[0].columns
        if column.canonical_field is not None
    }
    first_cells = {
        column.canonical_field: next(
            cell
            for cell in linked[0].rows[0].cells
            if cell.column_id == column.id
        )
        for column in columns_by_field.values()
    }
    assert first_cells["service_date_raw"].raw_value == (
        f"10/07/2026{timestamp_separator}02:48 am"
    )
    assert first_cells["request_no"].raw_value == f"{request_prefix}261015227"
    assert first_cells["description"].raw_value == "Patient Coat"
    assert {
        token_id
        for evidence in first_cells["description"].evidence
        for token_id in evidence.token_ids
    } == {"token-9", "token-10"}
    assert all(
        cell.validation_flags == ("split_from_merged_ocr_token",)
        for cell in (
            first_cells["service_date_raw"],
            first_cells["request_no"],
            first_cells["description"],
        )
    )
    if merged_right == 850:
        assert all(
            {
                token_id
                for evidence in cell.evidence
                for token_id in evidence.token_ids
            }
            == {"token-9"}
            for cell in (
                first_cells["service_date_raw"],
                first_cells["request_no"],
            )
        )
        assert all(
            min(point.x for point in cell.evidence[0].polygon.points) == 80
            and max(point.x for point in cell.evidence[0].polygon.points) == 850
            for cell in (
                first_cells["service_date_raw"],
                first_cells["request_no"],
                first_cells["description"],
            )
        )
        batch_column = next(
            column for column in linked[0].columns if column.label == "Batch No"
        )
        batch_cell = next(
            cell
            for cell in linked[0].rows[0].cells
            if cell.column_id == batch_column.id
        )
        assert batch_cell.raw_value == "62104070"
        assert {
            token_id
            for evidence in batch_cell.evidence
            for token_id in evidence.token_ids
        } == {"token-12"}
        assert min(
            point.x
            for evidence in batch_cell.evidence
            for point in evidence.polygon.points
        ) == 500
        assert max(
            point.x
            for evidence in batch_cell.evidence
            for point in evidence.polygon.points
        ) == 570
    second_cells = {
        column.canonical_field: next(
            cell
            for cell in linked[0].rows[1].cells
            if cell.column_id == column.id
        )
        for column in columns_by_field.values()
    }
    assert second_cells["service_date_raw"].raw_value == (
        f"10/07/2026{timestamp_separator}02:49 am"
    )
    assert second_cells["request_no"].raw_value == f"{request_prefix}261015228"
    assert second_cells["description"].raw_value == "Betadine Scrub 50 ML"


@pytest.mark.parametrize(
    ("merged_expiry_quantity", "quantity"),
    (("Aug/2028 1.00", "1.00"), ("Aug/20282.00", "2.00")),
)
def test_linked_pharmacy_row_splits_quantity_merged_with_expiry(
    merged_expiry_quantity: str,
    quantity: str,
) -> None:
    tokens = (
        token(0, "Date", (80, 25, 150, 40)),
        token(1, "ProductName", (290, 25, 430, 40)),
        token(2, "Expiry", (620, 25, 680, 40)),
        token(3, "Quantity", (750, 25, 820, 40)),
        token(4, "Rate", (840, 25, 885, 40)),
        token(5, "Amount", (920, 25, 970, 40)),
        token(6, "06/02/2026", (80, 70, 150, 85)),
        token(7, "NS 100 ML FLEXIDRIP CLARIS", (290, 70, 540, 85)),
        token(8, merged_expiry_quantity, (620, 70, 810, 85)),
        token(9, "44.93", (840, 70, 885, 85)),
        token(
            10,
            str(Decimal(quantity) * Decimal("44.93")),
            (920, 70, 970, 85),
        ),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(60, 15, 980, 100),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    columns_by_field = {
        column.canonical_field: column
        for column in linked[0].columns
        if column.canonical_field is not None
    }
    linked_cells = {
        cell.column_id: cell
        for cell in linked[0].rows[0].cells
    }
    expiry_column = next(
        column for column in linked[0].columns if column.label == "Expiry"
    )
    quantity_cell = linked_cells[columns_by_field["quantity"].id]
    expiry_cell = linked_cells[expiry_column.id]

    assert canonical[0].quantity == Decimal(quantity)
    assert quantity_cell.raw_value == quantity
    assert expiry_cell.raw_value == "Aug/2028"
    assert quantity_cell.validation_flags == ("split_from_merged_ocr_token",)
    assert {
        token_id
        for evidence in quantity_cell.evidence
        for token_id in evidence.token_ids
    } == {"token-8"}


def test_linked_pharmacy_row_consolidates_grounded_adjacent_description() -> None:
    tokens = (
        token(0, "Date/ Time", (80, 10, 170, 25)),
        token(1, "Bill Number", (200, 10, 290, 25)),
        token(2, "ProductName", (320, 10, 450, 25)),
        token(3, "Batch No", (500, 10, 570, 25)),
        token(4, "Expiry Date", (620, 10, 730, 25)),
        token(5, "Qty", (770, 10, 805, 25)),
        token(6, "Rate", (840, 10, 885, 25)),
        token(7, "Total", (920, 10, 970, 25)),
        token(8, "tals & Fertility Centre", (200, 50, 300, 105)),
        token(9, "Easyadlide Skin", (320, 70, 450, 85)),
        token(
            10,
            "12/07/2026,06:00 pm PI262015226.",
            (80, 85, 290, 105),
        ),
        token(11, "OD260403", (500, 87, 570, 103)),
        token(12, "31/03/2031", (620, 87, 730, 103)),
        token(13, "36.5", (840, 87, 885, 103)),
        token(14, "-73", (920, 87, 970, 103)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(60, 0, 980, 120),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )

    linked = _link_source_tables(result.source_tables, canonical)

    assert len(canonical) == 1
    linked_row = next(
        row for row in linked[0].rows if row.canonical_row_id is not None
    )
    donor_row = linked[0].rows[0]
    columns = {
        column.canonical_field: column
        for column in linked[0].columns
        if column.canonical_field is not None
    }
    linked_cells = {cell.column_id: cell for cell in linked_row.cells}
    donor_cells = {cell.column_id: cell for cell in donor_row.cells}
    description = linked_cells[columns["description"].id]
    donor_description = donor_cells[columns["description"].id]

    assert linked_cells[columns["service_date_raw"].id].raw_value == (
        "12/07/2026,06:00 pm"
    )
    assert linked_cells[columns["request_no"].id].raw_value == "PI262015226"
    assert description.raw_value == "Easyadlide Skin"
    assert {
        token_id
        for item in description.evidence
        for token_id in item.token_ids
    } == {"token-9"}
    assert "redistributed_from_adjacent_source_row" in (
        description.validation_flags
    )
    assert donor_description.raw_value is None
    assert not donor_description.evidence
    assert "redistributed_to_linked_source_row" in (
        donor_description.validation_flags
    )


def test_linked_row_does_not_guess_between_adjacent_description_donors() -> None:
    tokens = (
        token(0, "ProductName", (320, 10, 450, 25)),
        token(1, "Rate", (840, 10, 885, 25)),
        token(2, "Total", (920, 10, 970, 25)),
        token(3, "Grounded Item", (320, 50, 450, 65)),
        token(4, "25", (840, 75, 885, 90)),
        token(5, "25", (920, 75, 970, 90)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(300, 0, 980, 110),
    )
    canonical = canonicalize_rows(
        "d" * 64,
        1,
        "p1-t1",
        "a" * 64,
        result.rows,
    )
    source_table = result.source_tables[0]
    donor, financial = source_table.rows
    duplicate_donor = donor.model_copy(
        update={
            "id": f"{donor.id}-duplicate",
            "order": 2,
        }
    )
    ambiguous_table = source_table.model_copy(
        update={"rows": (donor, financial, duplicate_donor)}
    )

    linked = _link_source_tables((ambiguous_table,), canonical)

    linked_row = next(
        row for row in linked[0].rows if row.canonical_row_id is not None
    )
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    description = next(
        cell
        for cell in linked_row.cells
        if cell.column_id == description_column.id
    )
    assert description.raw_value is None
    assert all(
        next(
            cell
            for cell in source_row.cells
            if cell.column_id == description_column.id
        ).raw_value
        == "Grounded Item"
        for source_row in (linked[0].rows[0], linked[0].rows[2])
    )


def test_pharmacy_description_continuation_in_same_lane_is_grounded() -> None:
    tokens = (
        token(9, "IP Pharmacy", (300, 0, 430, 15)),
        token(0, "ProductName", (300, 25, 450, 40)),
        token(10, "Batch No", (500, 25, 570, 40)),
        token(11, "Expiry", (600, 25, 660, 40)),
        token(1, "Qty", (700, 25, 750, 40)),
        token(2, "Rate", (800, 25, 850, 40)),
        token(3, "Total", (900, 25, 960, 40)),
        token(4, "Betadine Scrub 50", (300, 70, 470, 85)),
        token(12, "MD06126", (500, 70, 570, 85)),
        token(5, "1", (710, 70, 730, 85)),
        token(6, "107.1", (800, 70, 850, 85)),
        token(7, "107.1", (900, 70, 960, 85)),
        token(8, "ML", (300, 100, 330, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(280, -5, 980, 130),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Betadine Scrub 50 ML"
    assert result.rows[0].field_token_ids["description"] == ("token-4", "token-8")
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    description_cell = next(
        cell
        for cell in linked[0].rows[0].cells
        if cell.column_id == description_column.id
    )
    assert description_cell.raw_value == "Betadine Scrub 50 ML"
    assert {
        token_id
        for evidence in description_cell.evidence
        for token_id in evidence.token_ids
    } == {"token-4", "token-8"}
    assert linked[0].rows[0].canonical_row_id == str(canonical[0].id)


@pytest.mark.parametrize(
    ("printed_amount", "expected_role", "expected_flags"),
    (
        (
            "23.93",
            RowRole.DETAIL,
            ("positive_amount_in_return_section",),
        ),
        ("-23.93", RowRole.REFUND, ()),
    ),
)
def test_pharmacy_return_section_marks_missing_sign_and_valid_refund_arithmetic(
    printed_amount: str,
    expected_role: RowRole,
    expected_flags: tuple[str, ...],
) -> None:
    tokens = (
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Qty", (700, 25, 750, 40)),
        token(2, "Rate", (800, 25, 850, 40)),
        token(3, "Total", (900, 25, 960, 40)),
        token(4, "IP Pharmacy Returns", (300, 65, 500, 80)),
        token(5, "Metronidazole IV 100ML", (300, 105, 500, 120)),
        token(6, "1", (710, 105, 730, 120)),
        token(7, "23.93", (800, 105, 850, 120)),
        token(8, printed_amount, (900, 105, 960, 120)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=6,
        table_id="p6-t1",
        box=(280, 15, 980, 140),
    )

    assert len(result.rows) == 1
    candidate = result.rows[0].candidate
    assert candidate.description == "Metronidazole IV 100ML"
    assert candidate.amount == Decimal(printed_amount)
    assert candidate.role is expected_role
    assert candidate.validation_flags == expected_flags


def test_pharmacy_return_section_ends_at_the_next_explicit_pharmacy_section() -> None:
    tokens = (
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Qty", (700, 25, 750, 40)),
        token(2, "Rate", (800, 25, 850, 40)),
        token(3, "Total", (900, 25, 960, 40)),
        token(4, "IP Pharmacy Returns", (300, 65, 500, 80)),
        token(5, "Returned item", (300, 105, 500, 120)),
        token(6, "1", (710, 105, 730, 120)),
        token(7, "10.00", (800, 105, 850, 120)),
        token(8, "-10.00", (900, 105, 960, 120)),
        token(9, "IP Pharmacy Details", (300, 145, 500, 160)),
        token(10, "Issued item", (300, 185, 500, 200)),
        token(11, "1", (710, 185, 730, 200)),
        token(12, "20.00", (800, 185, 850, 200)),
        token(13, "20.00", (900, 185, 960, 200)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=6,
        table_id="p6-t1",
        box=(280, 15, 980, 220),
    )

    assert tuple(row.candidate.amount for row in result.rows) == (
        Decimal("-10.00"),
        Decimal("20.00"),
    )
    assert tuple(row.candidate.validation_flags for row in result.rows) == ((), ())


def test_pharmacy_return_section_continues_to_the_next_page_table() -> None:
    first_page = reconstruct_ocr_rows(
        (
            token(0, "ProductName", (300, 25, 450, 40)),
            token(1, "Qty", (700, 25, 750, 40)),
            token(2, "Rate", (800, 25, 850, 40)),
            token(3, "Total", (900, 25, 960, 40)),
            token(4, "IP Pharmacy Returns", (300, 65, 500, 80)),
            token(5, "Returned item", (300, 105, 500, 120)),
            token(6, "1", (710, 105, 730, 120)),
            token(7, "10.00", (800, 105, 850, 120)),
            token(8, "-10.00", (900, 105, 960, 120)),
        ),
        page_number=1,
        table_id="p1-t1",
        box=(280, 15, 980, 140),
    )
    assert first_page.schema is not None

    second_page = reconstruct_ocr_rows(
        (
            token(20, "ProductName", (300, 25, 450, 40)),
            token(21, "Qty", (700, 25, 750, 40)),
            token(22, "Rate", (800, 25, 850, 40)),
            token(23, "Total", (900, 25, 960, 40)),
            token(24, "Continued item", (300, 65, 500, 80)),
            token(25, "1", (710, 65, 730, 80)),
            token(26, "20.00", (800, 65, 850, 80)),
            token(27, "20.00", (900, 65, 960, 80)),
        ),
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 100),
        prior_schemas=(first_page.schema,),
    )
    assert second_page.schema is not None
    normal_page = reconstruct_ocr_rows(
        (
            token(40, "IP Pharmacy Details", (300, 0, 500, 15)),
            token(41, "ProductName", (300, 25, 450, 40)),
            token(42, "Qty", (700, 25, 750, 40)),
            token(43, "Rate", (800, 25, 850, 40)),
            token(44, "Total", (900, 25, 960, 40)),
            token(45, "Issued item", (300, 65, 500, 80)),
            token(46, "1", (710, 65, 730, 80)),
            token(47, "30.00", (800, 65, 850, 80)),
            token(48, "30.00", (900, 65, 960, 80)),
        ),
        page_number=3,
        table_id="p3-t1",
        box=(280, 0, 980, 100),
        prior_schemas=(first_page.schema, second_page.schema),
    )

    assert first_page.schema.in_return_section
    assert second_page.rows[0].candidate.validation_flags == (
        "positive_amount_in_return_section",
    )
    assert normal_page.rows[0].candidate.validation_flags == ()


def test_recovery_does_not_inherit_its_own_table_end_state() -> None:
    tokens = (
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Qty", (700, 25, 750, 40)),
        token(2, "Rate", (800, 25, 850, 40)),
        token(3, "Total", (900, 25, 960, 40)),
        token(4, "Issued item", (300, 65, 500, 80)),
        token(5, "1", (710, 65, 730, 80)),
        token(6, "20.00", (800, 65, 850, 80)),
        token(7, "20.00", (900, 65, 960, 80)),
        token(8, "IP Pharmacy Returns", (300, 105, 500, 120)),
        token(9, "Returned item", (300, 145, 500, 160)),
        token(10, "1", (710, 145, 730, 160)),
        token(11, "10.00", (800, 145, 850, 160)),
        token(12, "-10.00", (900, 145, 960, 160)),
    )
    baseline = reconstruct_ocr_rows(
        tokens,
        page_number=6,
        table_id="p6-t1",
        box=(280, 15, 980, 180),
    )
    assert baseline.schema is not None
    assert baseline.schema.in_return_section

    recovery_schemas = _recovery_prior_schemas(
        (baseline.schema,),
        page_number=6,
        table_id="p6-t1",
    )
    recovered = reconstruct_ocr_rows(
        tokens,
        page_number=6,
        table_id="p6-t1",
        box=(280, 15, 980, 180),
        prior_schemas=recovery_schemas,
    )

    assert recovery_schemas == ()
    assert tuple(row.candidate.validation_flags for row in recovered.rows) == (
        (),
        (),
    )
    assert tuple(row.candidate.role for row in recovered.rows) == (
        RowRole.DETAIL,
        RowRole.REFUND,
    )


def test_pharmacy_description_wrap_inside_numeric_row_envelope_is_grounded() -> None:
    tokens = (
        token(99, "IP Pharmacy", (300, 0, 430, 15)),
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Batch No", (500, 25, 570, 40)),
        token(2, "Qty", (700, 25, 750, 40)),
        token(3, "Rate", (800, 25, 850, 40)),
        token(4, "Total", (900, 25, 960, 40)),
        token(5, "Dispovan 10Ml", (300, 70, 470, 90)),
        token(6, "621103JP1", (500, 70, 590, 110)),
        token(7, "5", (710, 70, 730, 110)),
        token(8, "13.3", (800, 70, 850, 110)),
        token(9, "66.5", (900, 70, 960, 110)),
        token(10, "Syringe", (300, 100, 390, 120)),
        token(11, "Next Product", (300, 140, 450, 160)),
        token(12, "1", (710, 140, 730, 160)),
        token(13, "20", (800, 140, 850, 160)),
        token(14, "20", (900, 140, 960, 160)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(280, -5, 980, 175),
    )

    assert [row.candidate.description for row in result.rows] == [
        "Dispovan 10Ml Syringe",
        "Next Product",
    ]
    assert result.rows[0].field_token_ids["description"] == (
        "token-5",
        "token-10",
    )
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    description_column = next(
        column
        for column in linked[0].columns
        if column.canonical_field == "description"
    )
    description_cell = next(
        cell
        for cell in linked[0].rows[0].cells
        if cell.column_id == description_column.id
    )
    assert description_cell.raw_value == "Dispovan 10Ml Syringe"
    assert {
        token_id
        for evidence in description_cell.evidence
        for token_id in evidence.token_ids
    } == {"token-5", "token-10"}
    assert linked[0].rows[0].canonical_row_id == str(canonical[0].id)


@pytest.mark.parametrize(
    "bill_total_tokens",
    (
        (token(9, "BILL TOTAL", (700, 160, 790, 180)),),
        (
            token(9, "BILL", (700, 160, 740, 180)),
            token(13, "TOTAL", (745, 160, 790, 180)),
        ),
        (
            rotated_token(
                9,
                "BILL TOTAL",
                (700, 160, 790, 180),
                6,
            ),
        ),
        (
            token(9, "BILL", (700, 160, 740, 180)),
            rotated_token(
                13,
                "TOTAL",
                (745, 160, 790, 180),
                6,
            ),
        ),
    ),
    ids=(
        "merged-label-token",
        "split-label-tokens",
        "rotated-merged-label-token",
        "rotated-split-total-token",
    ),
)
@pytest.mark.parametrize(
    ("overlay_tokens", "printed_total_description"),
    (
        ((), None),
        (
            (
                rotated_token(
                    14,
                    "Hospital address",
                    (300, 155, 550, 180),
                    15,
                ),
            ),
            "Hospital address",
        ),
    ),
    ids=("clean-total-line", "rotated-overlay-on-total-line"),
)
def test_bill_total_in_quantity_lane_does_not_consume_pending_description(
    bill_total_tokens: tuple[OcrToken, ...],
    overlay_tokens: tuple[OcrToken, ...],
    printed_total_description: str | None,
) -> None:
    tokens = (
        token(99, "IP Pharmacy", (300, 0, 430, 15)),
        token(0, "ProductName", (300, 25, 450, 40)),
        token(11, "Batch No", (500, 25, 570, 40)),
        token(1, "Qty", (700, 25, 750, 40)),
        token(2, "Rate", (800, 25, 850, 40)),
        token(3, "Total", (900, 25, 960, 40)),
        token(4, "Dispovan 1ML", (300, 70, 470, 90)),
        token(12, "621103JP1", (500, 70, 590, 90)),
        token(5, "2", (710, 70, 730, 90)),
        token(6, "10.23", (800, 70, 850, 90)),
        token(7, "20.46", (900, 70, 960, 90)),
        token(8, "Unlinked text", (300, 120, 430, 140)),
        *overlay_tokens,
        *bill_total_tokens,
        token(10, "20.46", (900, 160, 960, 180)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(280, -5, 980, 195),
    )

    assert [row.candidate.description for row in result.rows] == ["Dispovan 1ML"]
    assert all(row.candidate.amount == Decimal("20.46") for row in result.rows)
    description_column = next(
        column
        for column in result.source_tables[0].columns
        if column.canonical_field == "description"
    )
    assert [
        next(
            cell.raw_value
            for cell in row.cells
            if cell.column_id == description_column.id
        )
        for row in result.source_tables[0].rows
    ] == ["Dispovan 1ML", "Unlinked text", printed_total_description]

    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    assert [row.canonical_row_id for row in linked[0].rows] == [
        str(canonical[0].id),
        None,
        None,
    ]


def test_pharmacy_unpriced_full_name_is_not_merged_into_previous_product() -> None:
    tokens = (
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Batch No", (500, 25, 570, 40)),
        token(2, "Expiry", (600, 25, 660, 40)),
        token(3, "Qty", (700, 25, 750, 40)),
        token(4, "Rate", (800, 25, 850, 40)),
        token(5, "Total", (900, 25, 960, 40)),
        token(6, "First Product", (300, 70, 470, 85)),
        token(7, "1", (710, 70, 730, 85)),
        token(8, "100", (800, 70, 850, 85)),
        token(9, "100", (900, 70, 960, 85)),
        token(10, "Second Product", (300, 100, 420, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 130),
    )

    assert [row.candidate.description for row in result.rows] == ["First Product"]
    assert any(
        "Second Product" in {cell.raw_value for cell in row.cells}
        for table in result.source_tables
        for row in table.rows
    )


@pytest.mark.parametrize(
    "standalone_product",
    (
        "Glove",
        "Gloves",
        "Injection",
        "Tablet",
        "Capsule",
        "Syringe",
        "Blade",
    ),
)
def test_pharmacy_standalone_form_is_not_merged_into_previous_product(
    standalone_product: str,
) -> None:
    tokens = (
        token(0, "ProductName", (300, 25, 450, 40)),
        token(1, "Batch No", (500, 25, 570, 40)),
        token(2, "Expiry", (600, 25, 660, 40)),
        token(3, "Qty", (700, 25, 750, 40)),
        token(4, "Rate", (800, 25, 850, 40)),
        token(5, "Total", (900, 25, 960, 40)),
        token(6, "First Product", (300, 70, 470, 85)),
        token(7, "1", (710, 70, 730, 85)),
        token(8, "100", (800, 70, 850, 85)),
        token(9, "100", (900, 70, 960, 85)),
        token(10, standalone_product, (300, 100, 420, 115)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 130),
    )

    assert [row.candidate.description for row in result.rows] == ["First Product"]
    assert any(
        standalone_product in {cell.raw_value for cell in row.cells}
        for table in result.source_tables
        for row in table.rows
    )


@pytest.mark.parametrize("merged_right", (480, 500, 541, 600, 650, 850))
def test_fused_product_and_batch_does_not_publish_partial_batch_suffix(
    merged_right: int,
) -> None:
    result = reconstruct_ocr_rows(
        (
            token(0, "ProductName", (300, 25, 450, 40)),
            token(1, "Batch No", (500, 25, 570, 40)),
            token(2, "Qty", (700, 25, 750, 40)),
            token(3, "Rate", (800, 25, 850, 40)),
            token(4, "Total", (900, 25, 960, 40)),
            token(5, "Solution 100Ml MD06126", (320, 70, merged_right, 85)),
            token(6, "MD06126", (500, 70, 570, 85)),
            token(7, "1", (710, 70, 730, 85)),
            token(8, "100", (800, 70, 850, 85)),
            token(9, "100", (900, 70, 960, 85)),
        ),
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Solution 100Ml"
    canonical = canonicalize_rows(
        "d" * 64,
        2,
        "p2-t1",
        "a" * 64,
        result.rows,
    )
    linked = _link_source_tables(result.source_tables, canonical)
    columns = {column.label: column for column in linked[0].columns}
    cells = {cell.column_id: cell for cell in linked[0].rows[0].cells}
    assert cells[columns["ProductName"].id].raw_value == "Solution 100Ml"
    assert cells[columns["Batch No"].id].raw_value == "MD06126"


@pytest.mark.parametrize("expiry_header", ("Expiry Date", "Exp Date", "Date of Expiry"))
def test_pharmacy_expiry_header_never_becomes_service_date(
    expiry_header: str,
) -> None:
    result = reconstruct_ocr_rows(
        (
            token(0, "ProductName", (300, 25, 450, 40)),
            token(1, "Batch No", (500, 25, 570, 40)),
            token(2, expiry_header, (600, 25, 700, 40)),
            token(3, "Qty", (710, 25, 750, 40)),
            token(4, "Rate", (800, 25, 850, 40)),
            token(5, "Total", (900, 25, 960, 40)),
            token(6, "Betadine Solution", (300, 70, 450, 85)),
            token(7, "MD06126", (500, 70, 570, 85)),
            token(8, "30/09/2027", (600, 70, 700, 85)),
            token(9, "1", (710, 70, 750, 85)),
            token(10, "100", (800, 70, 850, 85)),
            token(11, "100", (900, 70, 960, 85)),
        ),
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Betadine Solution"
    assert result.rows[0].candidate.service_date is None


@pytest.mark.parametrize(
    ("date_fragment", "expiry_fragment"),
    (("Date", "of Expiry"), ("Date of", "Expiry")),
)
def test_split_date_of_expiry_header_never_becomes_service_date(
    date_fragment: str,
    expiry_fragment: str,
) -> None:
    result = reconstruct_ocr_rows(
        (
            token(0, "ProductName", (300, 25, 450, 40)),
            token(1, "Batch No", (500, 25, 570, 40)),
            token(2, date_fragment, (600, 25, 630, 40)),
            token(3, expiry_fragment, (640, 25, 700, 40)),
            token(4, "Qty", (710, 25, 750, 40)),
            token(5, "Rate", (800, 25, 850, 40)),
            token(6, "Total", (900, 25, 960, 40)),
            token(7, "Betadine Solution", (300, 70, 450, 85)),
            token(8, "MD06126", (500, 70, 570, 85)),
            token(9, "30/09/2027", (600, 70, 700, 85)),
            token(10, "1", (710, 70, 750, 85)),
            token(11, "100", (800, 70, 850, 85)),
            token(12, "100", (900, 70, 960, 85)),
        ),
        page_number=2,
        table_id="p2-t1",
        box=(280, 15, 980, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Betadine Solution"
    assert result.rows[0].candidate.service_date is None


def test_split_expiry_phrase_does_not_hide_separate_transaction_date() -> None:
    result = reconstruct_ocr_rows(
        (
            token(0, "Date/Time", (50, 25, 140, 40)),
            token(1, "ProductName", (200, 25, 400, 40)),
            token(2, "Batch No", (500, 25, 570, 40)),
            token(3, "Date", (600, 25, 630, 40)),
            token(4, "of Expiry", (640, 25, 700, 40)),
            token(5, "Qty", (710, 25, 750, 40)),
            token(6, "Rate", (800, 25, 850, 40)),
            token(7, "Total", (900, 25, 960, 40)),
            token(8, "10/07/2026", (50, 70, 140, 85)),
            token(9, "Betadine Solution", (200, 70, 400, 85)),
            token(10, "MD06126", (500, 70, 570, 85)),
            token(11, "30/09/2027", (600, 70, 700, 85)),
            token(12, "1", (710, 70, 750, 85)),
            token(13, "100", (800, 70, 850, 85)),
            token(14, "100", (900, 70, 960, 85)),
        ),
        page_number=2,
        table_id="p2-t1",
        box=(30, 15, 980, 100),
    )

    assert len(result.rows) == 1
    assert result.rows[0].candidate.description == "Betadine Solution"
    assert result.rows[0].candidate.service_date == "10/07/2026"


@pytest.mark.parametrize(
    "description",
    (
        "B12345 Injection",
        "ITEM12345 Dressing",
        "CIPLA12345 Tablet",
        "b12345 Injection",
    ),
)
def test_compact_product_prefix_is_not_a_request_without_request_schema(
    description: str,
) -> None:
    result = reconstruct_ocr_rows(
        (
            token(0, "Description", (100, 25, 400, 40)),
            token(1, "Amount", (850, 25, 950, 40)),
            token(2, description, (100, 70, 450, 85)),
            token(3, "500.00", (870, 70, 940, 85)),
        ),
        page_number=1,
        table_id="p1-t1",
        box=(80, 15, 980, 100),
    )

    assert result.rows[0].candidate.description == description
    assert result.rows[0].candidate.request_no is None


@pytest.mark.parametrize(
    ("raw", "description"),
    (
        ("10/07/2026 02:48 Ambulance", "Ambulance"),
        ("10/07/2026 02:48 Ampicillin", "Ampicillin"),
        ("10/07/2026 02:48 PMMA Implant", "PMMA Implant"),
    ),
)
def test_time_parser_does_not_consume_description_prefix(
    raw: str,
    description: str,
) -> None:
    cleaned, service_date, _ = _clean_description(raw)

    assert cleaned == description
    assert service_date == "10/07/2026"


def test_aadhaar_policy_and_pin_identifiers_are_not_published_as_money_rows() -> None:
    tokens = (
        token(0, "Aadhaar No", (100, 25, 220, 40)),
        token(1, "857821813514", (500, 25, 650, 40)),
        token(2, "Policy Number", (100, 60, 250, 75)),
        token(3, "70000931", (500, 60, 620, 75)),
        token(4, "Pin Code", (100, 95, 210, 110)),
        token(5, "100066", (500, 95, 590, 110)),
    )
    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 15, 680, 125),
    )
    assert all(row.candidate.role is RowRole.UNRESOLVED for row in result.rows)


def test_claim_policy_grid_is_metadata_not_a_charge_ledger() -> None:
    tokens = (
        token(0, "P2 REQ ON", (100, 25, 210, 40)),
        token(1, "POLICY TYPE", (260, 25, 380, 40)),
        token(2, "TREATMENT DETAILS", (430, 25, 620, 40)),
        token(3, "CLAIM ELIGIBLE", (720, 25, 880, 40)),
        token(4, "15/07/26, 2:55 PM", (100, 65, 250, 80)),
        token(5, "GMC", (260, 65, 310, 80)),
        token(6, "Laparoscopic Hysterectomy", (430, 65, 650, 80)),
        token(7, "200000", (760, 65, 840, 80)),
    )

    result = reconstruct_ocr_rows(
        tokens,
        page_number=1,
        table_id="p1-t1",
        box=(80, 15, 900, 95),
    )

    assert {table.table_type for table in result.source_tables} == {
        TableType.METADATA
    }
    assert all(row.candidate.role is RowRole.UNRESOLVED for row in result.rows)
