from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    ReviewDisposition,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    TableType,
)
from gmoney.extraction.offline import (
    _apply_document_role_policy,
    _deduplicate,
    _suppress_repeated_printed_tables,
)


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


def repeated_table(
    page_number: int,
    *,
    descriptions: tuple[str | None, ...],
    amount: str = "100",
    segment: int = 1,
    evidence_box: tuple[float, float, float, float] = (30, 30, 40, 40),
) -> SourceTable:
    def evidence(token_id: str) -> tuple[EvidenceRef, ...]:
        left, top, right, bottom = evidence_box
        return (
            EvidenceRef(
                page_number=page_number,
                table_id=f"p{page_number}-t1",
                polygon=Polygon(
                    points=(
                        Point(x=left, y=top),
                        Point(x=right, y=top),
                        Point(x=right, y=bottom),
                        Point(x=left, y=bottom),
                    )
                ),
                artifact_sha256="a" * 64,
                token_ids=(token_id,),
            ),
        )

    columns = (
        SourceColumn(
            id="serial",
            label="Sr. No.",
            order=0,
            evidence=evidence(f"header-{page_number}-serial"),
        ),
        SourceColumn(
            id="description",
            label="Particular",
            order=1,
            canonical_field="description",
            evidence=evidence(f"header-{page_number}-description"),
        ),
        SourceColumn(
            id="rate",
            label="Amount Rs.",
            order=2,
            canonical_field="unit_price",
            evidence=evidence(f"header-{page_number}-rate"),
        ),
        SourceColumn(
            id="quantity",
            label="Unit/Days",
            order=3,
            canonical_field="quantity",
            evidence=evidence(f"header-{page_number}-quantity"),
        ),
        SourceColumn(
            id="net",
            label="Total",
            order=4,
            canonical_field="net_amount",
            evidence=evidence(f"header-{page_number}-net"),
        ),
    )
    rows = tuple(
        SourceRow(
            id=f"p{page_number}-s{segment}-r{index}",
            order=index,
            cells=(
                SourceCell(
                    column_id="serial",
                    raw_value=f"{index}.",
                    evidence=evidence(f"p{page_number}-r{index}-serial"),
                ),
                SourceCell(
                    column_id="description",
                    raw_value=description,
                    evidence=(
                        evidence(f"p{page_number}-r{index}-description")
                        if description
                        else ()
                    ),
                ),
                SourceCell(
                    column_id="rate",
                    raw_value=amount,
                    evidence=evidence(f"p{page_number}-r{index}-rate"),
                ),
                SourceCell(
                    column_id="quantity",
                    raw_value="1",
                    evidence=evidence(f"p{page_number}-r{index}-quantity"),
                ),
                SourceCell(
                    column_id="net",
                    raw_value=amount,
                    evidence=evidence(f"p{page_number}-r{index}-net"),
                ),
            ),
        )
        for index, description in enumerate(descriptions)
    )
    return SourceTable(
        id=f"p{page_number}-t1-s{segment}",
        page_number=page_number,
        table_id=f"p{page_number}-t1",
        table_type=TableType.ITEM_LEDGER,
        columns=columns,
        rows=rows,
    )


def grounded_row(
    page_number: int,
    order: int,
    description: str,
    *,
    evidence_box: tuple[float, float, float, float] = (30, 30, 40, 40),
) -> CanonicalRow:
    left, top, right, bottom = evidence_box
    evidence = EvidenceRef(
        page_number=page_number,
        table_id=f"p{page_number}-t1",
        polygon=Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
        artifact_sha256="b" * 64,
        token_ids=(f"canonical-{page_number}-{order}",),
    )
    return row(order, RowRole.DETAIL, "service", "100").model_copy(
        update={
            "page_number": page_number,
            "table_id": f"p{page_number}-t1",
            "description": description,
            "evidence": (evidence,),
        }
    )


def repeated_crop_inputs(
    tmp_path: Path,
    *,
    change_candidate: bool = False,
) -> tuple[
    dict[tuple[int, str], Path],
    dict[tuple[int, str], tuple[int, int, int, int]],
]:
    reference = np.full((80, 80), 255, dtype=np.uint8)
    cv2.rectangle(reference, (5, 5), (74, 74), 0, 2)
    cv2.putText(
        reference,
        "BILL",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        0,
        2,
    )
    candidate = np.full((120, 120), 240, dtype=np.uint8)
    candidate[20:100, 20:100] = reference
    if change_candidate:
        cv2.circle(candidate, (60, 60), 8, 127, -1)
    candidate_path = tmp_path / "candidate.png"
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(candidate_path), candidate)
    assert cv2.imwrite(str(reference_path), reference)
    return (
        {
            (1, "p1-t1"): candidate_path,
            (7, "p7-t1"): reference_path,
        },
        {
            (1, "p1-t1"): (0, 0, 120, 120),
            (7, "p7-t1"): (20, 20, 100, 100),
        },
    )


def test_repeated_printed_table_keeps_most_complete_grounded_rendition(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Charge {index}" for index in range(10))
    shifted = (None, *descriptions[:-1])
    first = repeated_table(1, descriptions=shifted)
    final = repeated_table(7, descriptions=descriptions)
    canonical_rows = [
        grounded_row(page_number, index, description)
        for page_number, values in ((1, shifted), (7, descriptions))
        for index, description in enumerate(values)
        if description is not None
    ]
    crop_paths, crop_boxes = repeated_crop_inputs(tmp_path)

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [first, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert [table.table_id for table in tables] == ["p7-t1"]
    assert {item.page_number for item in selected_rows} == {7}
    assert suppressed == ((1, "p1-t1"),)


def test_similar_serial_ranges_do_not_merge_different_printed_tables(
    tmp_path: Path,
) -> None:
    first = repeated_table(
        1,
        descriptions=tuple(f"Medicine {index}" for index in range(10)),
    )
    second = repeated_table(
        2,
        descriptions=tuple(f"Procedure {index}" for index in range(10)),
    )

    tables, rows, suppressed = _suppress_repeated_printed_tables(
        [first, second],
        [],
        crop_paths={},
        crop_boxes={},
    )

    assert tables == [first, second]
    assert rows == []
    assert suppressed == ()


def test_repeated_labels_do_not_merge_tables_with_different_values(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Daily charge {index}" for index in range(10))
    first = repeated_table(1, descriptions=descriptions, amount="100")
    second = repeated_table(2, descriptions=descriptions, amount="200")

    tables, rows, suppressed = _suppress_repeated_printed_tables(
        [first, second],
        [],
        crop_paths={},
        crop_boxes={},
    )

    assert tables == [first, second]
    assert rows == []
    assert suppressed == ()


def test_near_duplicate_tables_with_unique_rows_are_preserved(
    tmp_path: Path,
) -> None:
    shared = tuple(f"Shared charge {index}" for index in range(8))
    first_descriptions = (*shared, "Left unique 1", "Left unique 2")
    final_descriptions = (*shared, "Right unique 1", "Right unique 2")
    first = repeated_table(1, descriptions=first_descriptions)
    final = repeated_table(7, descriptions=final_descriptions)
    canonical_rows = [
        grounded_row(page_number, index, description)
        for page_number, values in (
            (1, first_descriptions),
            (7, final_descriptions),
        )
        for index, description in enumerate(values)
    ]
    crop_paths, crop_boxes = repeated_crop_inputs(
        tmp_path,
        change_candidate=True,
    )

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [first, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert tables == [first, final]
    assert selected_rows == canonical_rows
    assert suppressed == ()


def test_physical_table_sibling_segment_prevents_partial_suppression(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Charge {index}" for index in range(10))
    duplicate_segment = repeated_table(1, descriptions=descriptions)
    unique_segment = repeated_table(
        1,
        descriptions=(None,),
        segment=2,
        evidence_box=(105, 105, 115, 115),
    )
    final = repeated_table(7, descriptions=descriptions)
    canonical_rows = [
        *(
            grounded_row(1, index, description)
            for index, description in enumerate(descriptions)
        ),
        grounded_row(
            1,
            100,
            "Unique sibling row",
            evidence_box=(105, 105, 115, 115),
        ).model_copy(
            update={
                "role": RowRole.INFORMATIONAL,
                "net_amount": None,
            }
        ),
        *(
            grounded_row(7, index, description)
            for index, description in enumerate(descriptions)
        ),
    ]
    crop_paths, crop_boxes = repeated_crop_inputs(tmp_path)

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [duplicate_segment, unique_segment, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert tables == [duplicate_segment, unique_segment, final]
    assert selected_rows == canonical_rows
    assert suppressed == ()


def test_blank_description_financial_residual_is_preserved(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Charge {index}" for index in range(10))
    duplicate_segment = repeated_table(1, descriptions=descriptions)
    outside_financial_segment = repeated_table(
        1,
        descriptions=(None,),
        segment=2,
        evidence_box=(105, 105, 115, 115),
    )
    final = repeated_table(7, descriptions=descriptions)
    canonical_rows = [
        *(
            grounded_row(1, index, description)
            for index, description in enumerate(descriptions)
        ),
        *(
            grounded_row(7, index, description)
            for index, description in enumerate(descriptions)
        ),
    ]
    crop_paths, crop_boxes = repeated_crop_inputs(tmp_path)

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [duplicate_segment, outside_financial_segment, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert [table.id for table in tables] == [
        outside_financial_segment.id,
        final.id,
    ]
    residual = tables[0]
    assert residual.rows[0].canonical_row_id is None
    assert [
        cell.raw_value
        for cell in residual.rows[0].cells
        if cell.raw_value
    ] == ["0.", "100", "1", "100"]
    assert selected_rows == [
        item for item in canonical_rows if item.page_number == 7
    ]
    assert suppressed == ((1, "p1-t1"),)


def test_boundary_crossing_source_evidence_is_preserved_as_residual(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Charge {index}" for index in range(10))
    crossing = repeated_table(
        1,
        descriptions=descriptions,
        evidence_box=(30, 30, 115, 40),
    )
    final = repeated_table(7, descriptions=descriptions)
    canonical_rows = [
        *(
            grounded_row(1, index, description)
            for index, description in enumerate(descriptions)
        ),
        *(
            grounded_row(7, index, description)
            for index, description in enumerate(descriptions)
        ),
    ]
    crop_paths, crop_boxes = repeated_crop_inputs(tmp_path)

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [crossing, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert [table.id for table in tables] == [crossing.id, final.id]
    assert [
        cell.raw_value
        for cell in tables[0].rows[0].cells
        if cell.raw_value
    ] == ["0.", descriptions[0], "100", "1", "100"]
    assert selected_rows == [
        item for item in canonical_rows if item.page_number == 7
    ]
    assert suppressed == ((1, "p1-t1"),)


def test_candidate_only_footer_outside_base_crop_is_preserved(
    tmp_path: Path,
) -> None:
    descriptions = tuple(f"Charge {index}" for index in range(10))
    repeated = repeated_table(1, descriptions=descriptions)
    candidate_only_footer = repeated_table(
        1,
        descriptions=(None,),
        segment=2,
        evidence_box=(30, 105, 40, 115),
        amount="999",
    )
    final = repeated_table(7, descriptions=descriptions)
    canonical_rows = [
        *(
            grounded_row(1, index, description)
            for index, description in enumerate(descriptions)
        ),
        *(
            grounded_row(7, index, description)
            for index, description in enumerate(descriptions)
        ),
    ]

    crop_paths, crop_boxes = repeated_crop_inputs(tmp_path)

    tables, selected_rows, suppressed = _suppress_repeated_printed_tables(
        [repeated, candidate_only_footer, final],
        canonical_rows,
        crop_paths=crop_paths,
        crop_boxes=crop_boxes,
    )

    assert [table.id for table in tables] == [
        candidate_only_footer.id,
        final.id,
    ]
    assert [
        cell.raw_value
        for cell in tables[0].rows[0].cells
        if cell.raw_value
    ] == ["0.", "999", "1", "999"]
    assert {item.page_number for item in selected_rows} == {7}
    assert suppressed == ((1, "p1-t1"),)


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
