from decimal import Decimal

from gmoney.contracts.extraction import ReviewDisposition, RowRole
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow


def test_grounded_detail_becomes_accepted_canonical_row() -> None:
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("Blood Test", "500"),
        description="Blood Test",
        amount=Decimal("500"),
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={"description": ("t1",), "amount": ("t2",)},
        evidence_token_ids=("t1", "t2"),
        evidence_box=(10, 20, 200, 40),
        grounding_ratio=1.0,
    )
    rows = canonicalize_rows("document", 1, "table", "a" * 64, (aligned,))
    assert len(rows) == 1
    assert rows[0].review_disposition is ReviewDisposition.ACCEPTED
    assert rows[0].net_amount == Decimal("500")
    assert rows[0].evidence[0].token_ids == ("t1", "t2")


def test_totals_never_enter_canonical_detail_ledger() -> None:
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DOCUMENT_TOTAL,
        cells=("Total", "500"),
        description="Total",
        amount=Decimal("500"),
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={},
        evidence_token_ids=(),
        evidence_box=None,
        grounding_ratio=0,
    )
    assert canonicalize_rows("document", 1, "table", "a" * 64, (aligned,)) == ()


def test_descriptionless_numeric_candidate_stays_out_of_canonical_ledger() -> None:
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("", "500"),
        amount=Decimal("500"),
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={"amount": ("t1",)},
        evidence_token_ids=("t1",),
        evidence_box=(10, 20, 200, 40),
        grounding_ratio=1,
    )
    assert canonicalize_rows("document", 1, "table", "a" * 64, (aligned,)) == ()
