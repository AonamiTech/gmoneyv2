from decimal import Decimal

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.extraction.document_total import (
    extract_document_total_candidates,
    select_document_total,
    select_document_totals,
)


def token(
    index: int,
    text: str,
    box: tuple[float, float, float, float],
    *,
    page: int = 1,
) -> OcrToken:
    left, top, right, bottom = box
    return OcrToken(
        token_id=f"p{page}-token-{index}",
        page_number=page,
        text=text,
        confidence=0.98,
        polygon=Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
        artifact_sha256=("a" if page == 1 else "b") * 64,
        model_name="fixture",
        model_version="1",
    )


def test_extracts_explicit_final_total_with_grounded_evidence() -> None:
    candidates = extract_document_total_candidates(
        (
            token(0, "Sub Total", (100, 50, 260, 70)),
            token(1, "900.00", (800, 50, 930, 70)),
            token(2, "Net Bill Amount", (100, 100, 310, 120)),
            token(3, "₹ 1,000.50", (780, 100, 930, 120)),
        )
    )
    total = select_document_total(candidates)
    assert total is not None
    assert total.label == "Net Bill Amount"
    assert total.amount == Decimal("1000.50")
    assert total.amount_raw == "₹ 1,000.50"
    assert total.evidence.token_ids == ("p1-token-2", "p1-token-3")


def test_specific_bill_label_outranks_later_generic_grand_total() -> None:
    first = extract_document_total_candidates(
        (
            token(0, "Total Bill Amount", (100, 100, 330, 120)),
            token(1, "1,250.00", (800, 100, 930, 120)),
        )
    )
    later = extract_document_total_candidates(
        (
            token(2, "Grand Total", (100, 600, 270, 620), page=2),
            token(3, "250.00", (800, 600, 930, 620), page=2),
        )
    )
    total = select_document_total((*first, *later))
    assert total is not None
    assert total.amount == Decimal("1250.00")
    assert total.page_number == 1


def test_net_medical_amount_is_an_explicit_bill_total() -> None:
    candidates = extract_document_total_candidates(
        (
            token(0, "Gross Bill Amount", (100, 50, 320, 70)),
            token(1, "36,235.23", (800, 50, 930, 70)),
            token(2, "Net Medical Amount", (100, 80, 340, 100)),
            token(3, "36,235.23", (800, 80, 930, 100)),
            token(4, "Claim Amount", (100, 110, 280, 130)),
            token(5, "36,235.00", (800, 110, 930, 130)),
        )
    )

    total = select_document_total(candidates)

    assert total is not None
    assert total.label == "Net Medical Amount"
    assert total.amount == Decimal("36235.23")
    assert total.evidence.token_ids == ("p1-token-2", "p1-token-3")


def test_later_same_priority_total_wins_and_intermediate_labels_are_rejected() -> None:
    excluded = extract_document_total_candidates(
        (
            token(0, "Patient Grand Total", (100, 50, 320, 70)),
            token(1, "700.00", (800, 50, 930, 70)),
            token(2, "Gross Bill Amount", (100, 80, 320, 100)),
            token(3, "800.00", (800, 80, 930, 100)),
        )
    )
    first = extract_document_total_candidates(
        (
            token(4, "Net Bill Amount", (100, 110, 320, 130)),
            token(5, "900.00", (800, 110, 930, 130)),
        )
    )
    second = extract_document_total_candidates(
        (
            token(6, "Net Bill Amount 950.00", (100, 100, 930, 120), page=2),
        )
    )
    assert excluded == ()
    total = select_document_total((*first, *second))
    assert total is not None
    assert total.amount == Decimal("950.00")
    assert total.page_number == 2


def test_retains_all_explicit_totals_with_scope_and_selects_primary() -> None:
    candidates = extract_document_total_candidates(
        (
            token(0, "Gross Amount 1,200.00", (100, 80, 930, 100)),
            token(1, "Net Amount 1,000.00", (100, 110, 930, 130)),
            token(2, "Amount To Be Received 400.00", (100, 140, 930, 160)),
            token(3, "Total Bill Amount 1,000.00", (100, 170, 930, 190)),
        )
    )
    totals = select_document_totals(candidates)
    assert [(item.label, item.amount, item.scope.value) for item in totals] == [
        ("Gross Amount", Decimal("1200.00"), "document"),
        ("Net Amount", Decimal("1000.00"), "document"),
        ("Amount To Be Received", Decimal("400.00"), "payment"),
        ("Total Bill Amount", Decimal("1000.00"), "document"),
    ]
    primary = select_document_total(candidates)
    assert primary is not None
    assert primary.label == "Total Bill Amount"


def test_pharmacy_grand_total_is_section_scoped() -> None:
    candidates = extract_document_total_candidates(
        (
            token(0, "Pharmacy Detailed Bill", (100, 30, 350, 50)),
            token(1, "Grand Total", (100, 100, 300, 120)),
            token(2, "28,555.00", (800, 100, 930, 120)),
        )
    )
    assert len(candidates) == 1
    assert candidates[0].total.scope.value == "section"
