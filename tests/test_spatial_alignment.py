from decimal import Decimal

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole
from gmoney.extraction.ocr_tokens import paddle_ocr_tokens
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import align_candidate_rows
from gmoney.geometry.transform import translation


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
        model_name="PP-OCRv6-medium",
        model_version="PP-OCRv6",
    )


def test_paddle_tokens_map_crop_polygons_back_to_page() -> None:
    output = {
        "pages": [
            {
                "res": {
                    "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                    "rec_texts": ["Test"],
                    "rec_scores": [0.9],
                }
            }
        ]
    }
    tokens = paddle_ocr_tokens(output, 1, "a" * 64, crop_to_page=translation(100, 200))
    assert tokens[0].polygon.points[0] == Point(x=100, y=200)


def test_duplicate_rows_align_monotonically_to_distinct_evidence() -> None:
    tokens = (
        token(0, "Needle", (10, 10, 80, 20)),
        token(1, "5.00", (200, 10, 240, 20)),
        token(2, "Needle", (10, 40, 80, 50)),
        token(3, "5.00", (200, 40, 240, 50)),
    )
    candidates = tuple(
        CandidateLedgerRow(
            source_row=index,
            role=RowRole.DETAIL,
            cells=("Needle", "5.00"),
            description="Needle",
            amount=Decimal("5.00"),
        )
        for index in range(2)
    )
    aligned = align_candidate_rows(candidates, tokens)
    assert aligned[0].field_token_ids["description"] == ("token-0",)
    assert aligned[0].field_token_ids["amount"] == ("token-1",)
    assert aligned[1].field_token_ids["description"] == ("token-2",)
    assert aligned[1].field_token_ids["amount"] == ("token-3",)
    assert aligned[0].grounding_ratio == aligned[1].grounding_ratio == 1.0


def test_derived_amount_is_reconciled_to_rightmost_ocr_value() -> None:
    tokens = (
        token(0, "Levipil", (10, 10, 80, 20)),
        token(1, "13.77", (150, 10, 190, 20)),
        token(2, "60", (220, 10, 240, 20)),
        token(3, "826.00", (280, 10, 340, 20)),
    )
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("Levipil", "13.77", "60"),
        description="Levipil",
        quantity=Decimal("60"),
        rate=Decimal("13.77"),
        amount=Decimal("826.20"),
        amount_derived=True,
    )
    aligned = align_candidate_rows((candidate,), tokens)
    assert aligned[0].candidate.amount == Decimal("826.00")
    assert aligned[0].field_token_ids["amount"] == ("token-3",)


def test_amount_reconciliation_uses_same_line_and_last_number_in_merged_token() -> None:
    tokens = (
        token(0, "First", (10, 10, 80, 40)),
        token(1, "1 6.50", (250, 10, 340, 40)),
        token(2, "Second", (10, 50, 80, 80)),
        token(3, "2 25.40", (250, 50, 340, 80)),
    )
    candidates = (
        CandidateLedgerRow(
            source_row=1,
            role=RowRole.DETAIL,
            cells=("First", "6.50", "1"),
            description="First",
            quantity=Decimal("1"),
            rate=Decimal("6.50"),
            amount=Decimal("6.50"),
            amount_derived=True,
        ),
        CandidateLedgerRow(
            source_row=2,
            role=RowRole.DETAIL,
            cells=("Second", "12.70", "2"),
            description="Second",
            quantity=Decimal("2"),
            rate=Decimal("12.70"),
            amount=Decimal("25.40"),
            amount_derived=True,
        ),
    )
    aligned = align_candidate_rows(candidates, tokens)
    assert [row.candidate.amount for row in aligned] == [Decimal("6.50"), Decimal("25.40")]
