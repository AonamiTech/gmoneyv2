from decimal import Decimal

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    ReviewDisposition,
    RowRole,
    TableType,
)
from gmoney.extraction.offline import _apply_document_role_policy, _deduplicate


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


def test_repeated_semantic_rollups_keep_the_richest_grounded_row() -> None:
    compact = row(0, RowRole.CATEGORY_ROLLUP, "package", "11457").model_copy(
        update={"description": "Package (IPD) - Coronary"}
    )
    continuation = row(1, RowRole.CATEGORY_ROLLUP, "package", "11457").model_copy(
        update={
            "page_number": 2,
            "description": "Package Name: Coronary Angiography (CAG)",
        }
    )
    abbreviated = row(2, RowRole.CATEGORY_ROLLUP, "package", "11457").model_copy(
        update={"description": "Angiography (CAG)"}
    )
    unrelated = row(3, RowRole.CATEGORY_ROLLUP, "laboratory", "11457").model_copy(
        update={"description": "Laboratory services"}
    )

    selected = _apply_document_role_policy([compact, continuation, abbreviated, unrelated])

    assert [item.description for item in selected] == [
        "Package Name: Coronary Angiography (CAG)",
        "Laboratory services",
    ]


def test_overlapping_table_proposals_keep_the_trustworthy_complete_row() -> None:
    description_evidence = EvidenceRef(
        page_number=1,
        table_id="full",
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=10, y=0),
                Point(x=10, y=10),
                Point(x=0, y=10),
            )
        ),
        artifact_sha256="a" * 64,
        token_ids=("shared-description-token",),
    )
    partial = row(0, RowRole.DETAIL, "implant", "1").model_copy(
        update={
            "description": "ULTIMASTER STENT 3.00MM x 18",
            "field_evidence": {"description": (description_evidence,)},
            "validation_flags": ("line_arithmetic_mismatch",),
        }
    )
    complete = row(1, RowRole.DETAIL, "implant", "40879.66").model_copy(
        update={
            "description": "ULTIMASTER STENT 3.00MM x 18MM",
            "field_evidence": {"description": (description_evidence,)},
        }
    )
    assert _deduplicate([partial, complete]) == [complete]
