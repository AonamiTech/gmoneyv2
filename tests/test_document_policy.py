from decimal import Decimal

from gmoney.contracts.extraction import CanonicalRow, ReviewDisposition, RowRole, TableType
from gmoney.extraction.offline import _apply_document_role_policy


def row(order: int, role: RowRole, section: str, amount: str) -> CanonicalRow:
    return CanonicalRow(
        contract_version="canonical_row_v2",
        document_id="document",
        page_number=1,
        table_id="table",
        table_type=TableType.ITEM_LEDGER,
        row_order=order,
        role=role,
        review_disposition=ReviewDisposition.ACCEPTED,
        section=section,
        description=f"row-{order}",
        net_amount=Decimal(amount),
        evidence=(),
    )


def test_only_replaced_category_rollup_is_suppressed() -> None:
    rows = [
        row(0, RowRole.CATEGORY_ROLLUP, "pharmacy", "30"),
        row(1, RowRole.CATEGORY_ROLLUP, "bed", "100"),
        row(2, RowRole.DETAIL, "pharmacy", "10"),
        row(3, RowRole.DETAIL, "pharmacy", "20"),
    ]
    selected = _apply_document_role_policy(rows)
    assert [item.net_amount for item in selected] == [Decimal("100"), Decimal("10"), Decimal("20")]


def test_zero_and_exactly_repeated_summary_rows_are_suppressed() -> None:
    repeated = row(1, RowRole.CATEGORY_ROLLUP, "other", "25").model_copy(
        update={"description": "row-0"}
    )
    rows = [
        row(0, RowRole.DETAIL, "service", "25"),
        repeated,
        row(2, RowRole.CATEGORY_ROLLUP, "unused", "0"),
    ]
    assert [item.net_amount for item in _apply_document_role_policy(rows)] == [Decimal("25")]


def test_coincidental_cross_page_sum_does_not_delete_detail() -> None:
    rows = [
        row(0, RowRole.DETAIL, "service", "30"),
        row(1, RowRole.DETAIL, "service", "10"),
        row(2, RowRole.DETAIL, "service", "20"),
    ]
    assert len(_apply_document_role_policy(rows)) == 3
