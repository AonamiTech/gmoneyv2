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
    assert rows[0].contract_version == "canonical_row_v2"
    assert set(rows[0].field_evidence) == {"description", "amount"}
    repeated = canonicalize_rows("document", 1, "table", "a" * 64, (aligned,))
    assert repeated[0].id == rows[0].id
    assert repeated[0].candidate_ids == rows[0].candidate_ids


def test_ungrounded_optional_provider_values_are_not_published() -> None:
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("Blood Test", "2", "250", "500"),
        description="Blood Test",
        quantity=Decimal("2"),
        rate=Decimal("250"),
        amount=Decimal("500"),
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={"description": ("t1",), "amount": ("t2",)},
        evidence_token_ids=("t1", "t2"),
        evidence_box=(10, 20, 200, 40),
        grounding_ratio=0.5,
    )
    row = canonicalize_rows("document", 1, "table", "a" * 64, (aligned,))[0]
    assert row.review_disposition is ReviewDisposition.ACCEPTED
    assert row.quantity is None
    assert row.unit_price is None


def test_row_with_ungrounded_amount_is_not_published() -> None:
    candidate = CandidateLedgerRow(
        source_row=1,
        role=RowRole.DETAIL,
        cells=("Blood Test", "500"),
        description="Blood Test",
        amount=Decimal("500"),
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={"description": ("t1",)},
        evidence_token_ids=("t1",),
        evidence_box=(10, 20, 200, 40),
        grounding_ratio=1,
    )
    assert canonicalize_rows("document", 1, "table", "a" * 64, (aligned,)) == ()


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


def test_grounded_informational_row_is_published_without_an_amount() -> None:
    candidate = CandidateLedgerRow(
        source_row=2,
        role=RowRole.INFORMATIONAL,
        cells=("Complete Haemogram", "20/01/2026"),
        description="Complete Haemogram",
        service_date="20/01/2026",
    )
    aligned = AlignedLedgerRow(
        candidate=candidate,
        field_token_ids={"description": ("t1",), "service_date": ("t2",)},
        evidence_token_ids=("t1", "t2"),
        evidence_box=(10, 20, 200, 40),
        grounding_ratio=1,
    )
    row = canonicalize_rows("document", 1, "table", "a" * 64, (aligned,))[0]
    assert row.role is RowRole.INFORMATIONAL
    assert row.service_date_raw == "20/01/2026"
    assert row.service_date_iso == "2026-01-20"
    assert row.net_amount is None
    assert "amount" not in row.field_evidence
