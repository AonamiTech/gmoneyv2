from decimal import Decimal

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_rows import (
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
)
from gmoney.extraction.offline import _link_source_tables
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
