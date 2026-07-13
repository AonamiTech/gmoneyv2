from decimal import Decimal

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType
from gmoney.extraction.ocr_rows import (
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
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
