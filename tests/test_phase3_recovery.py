from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import (
    EvidenceRef,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    TableType,
)
from gmoney.contracts.phase3 import (
    GeminiMode,
    ProfileMatch,
    RecoveryReason,
    RecoveryStage,
)
from gmoney.extraction import offline as offline_module
from gmoney.extraction.ocr_rows import (
    ReconstructionResult,
    TableSchemaState,
)
from gmoney.extraction.offline import OfflineExtractor, TableWork
from gmoney.extraction.recovery import (
    decide_recovery,
    description_lane_recovery_regions,
    is_implausibly_low_yield,
    map_crop_tokens_to_page,
    map_page_box_to_crop_pixels,
    map_transformed_tokens_to_page,
    merge_recovery_tokens,
    needs_field_quality_recovery,
    reconstruction_quality,
    replace_tokens_in_regions,
    return_sign_recovery_targets,
    safely_improves_reconstruction,
)
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow
from gmoney.extraction.typed_values import parse_decimal


def _schema(table_type: TableType = TableType.ITEM_LEDGER) -> TableSchemaState:
    return TableSchemaState(
        source_page=1,
        source_table="table-1",
        table_type=table_type,
        column_centers={
            "description": 0.2,
            "quantity": 0.5,
            "rate": 0.7,
            "amount": 0.9,
        },
        confidence=1,
        header_token_ids=("header",),
    )


def _aligned_row(
    source_row: int,
    *,
    description: str = "Anaesthetist",
    service_date: str | None = None,
    quantity: Decimal | None = None,
    rate: Decimal | None = Decimal("100"),
    amount: Decimal = Decimal("100"),
    flags: tuple[str, ...] = (),
    mapped_fields: tuple[str, ...] = ("description", "rate", "amount"),
    role: RowRole = RowRole.DETAIL,
    table_type: TableType = TableType.ITEM_LEDGER,
) -> AlignedLedgerRow:
    field_token_ids = {field: (f"token-{source_row}-{field}",) for field in mapped_fields}
    return AlignedLedgerRow(
        candidate=CandidateLedgerRow(
            source_row=source_row,
            role=role,
            cells=(description, str(quantity or ""), str(rate or ""), str(amount)),
            description=description,
            service_date=service_date,
            quantity=quantity,
            rate=rate,
            amount=amount,
            table_type=table_type,
            source_route="ocr_spatial_graph",
            validation_flags=flags,
        ),
        field_token_ids=field_token_ids,
        evidence_token_ids=tuple(
            token_id for token_ids in field_token_ids.values() for token_id in token_ids
        ),
        evidence_box=(0, 0, 100, 20),
        grounding_ratio=1,
        source_routes=("ocr_spatial_graph",),
    )


def _reconstruction(
    rows: tuple[AlignedLedgerRow, ...],
    *,
    table_type: TableType = TableType.ITEM_LEDGER,
) -> ReconstructionResult:
    return ReconstructionResult(
        rows=rows,
        schema=_schema(table_type),
        diagnostics={
            "table_type": table_type.value,
            "ocr_line_count": len(rows) + 1,
            "ocr_row_count": len(rows),
        },
    )


def _source_table(*raw_values: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        rows=(
            SimpleNamespace(
                cells=tuple(SimpleNamespace(raw_value=raw_value) for raw_value in raw_values)
            ),
        )
    )


def _ocr_token(text: str, artifact_sha256: str, token_id: str | None = None) -> OcrToken:
    return OcrToken(
        token_id=token_id or f"local-{text}",
        page_number=1,
        text=text,
        confidence=1,
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=100, y=0),
                Point(x=100, y=50),
                Point(x=0, y=50),
            )
        ),
        artifact_sha256=artifact_sha256,
        model_name="test",
        model_version="1",
    )


def test_recovery_ladder_is_local_first_and_gemini_last() -> None:
    reconstruction = ReconstructionResult(
        rows=(),
        schema=None,
        diagnostics={"table_type": "unknown"},
    )
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.CHALLENGER)
    assert decision.reasons == (
        RecoveryReason.ZERO_YIELD,
        RecoveryReason.NO_SCHEMA,
        RecoveryReason.UNKNOWN_TABLE,
    )
    assert decision.planned_stages == (
        RecoveryStage.HIGH_RESOLUTION,
        RecoveryStage.PHOTOMETRIC,
        RecoveryStage.CROP_OCR,
        RecoveryStage.LOCAL_VLM,
        RecoveryStage.GEMINI,
        RecoveryStage.REVIEW,
    )


def test_crop_tokens_map_back_to_original_page() -> None:
    source = OcrToken(
        token_id="1",
        page_number=1,
        text="Amount",
        confidence=1,
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=100, y=0),
                Point(x=100, y=50),
                Point(x=0, y=50),
            )
        ),
        artifact_sha256="b" * 64,
        model_name="test",
        model_version="1",
    )
    mapped = map_crop_tokens_to_page(
        (source,),
        (200, 300, 400, 400),
        crop_width=200,
        crop_height=100,
        page_artifact_sha256="a" * 64,
    )[0]
    assert mapped.token_id == "recovery:1"
    assert mapped.polygon.points[0] == Point(x=200, y=300)
    assert mapped.polygon.points[2] == Point(x=300, y=350)


def test_transformed_tokens_map_back_through_perspective_matrix() -> None:
    source = OcrToken(
        token_id="1",
        page_number=1,
        text="Amount",
        confidence=1,
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=10, y=0),
                Point(x=10, y=10),
                Point(x=0, y=10),
            )
        ),
        artifact_sha256="b" * 64,
        model_name="test",
        model_version="1",
    )
    mapped = map_transformed_tokens_to_page(
        (source,),
        ((2.0, 0.0, 100.0), (0.0, 3.0, 200.0), (0.0, 0.0, 1.0)),
        "a" * 64,
    )[0]
    assert mapped.polygon.points[0] == Point(x=100, y=200)
    assert mapped.polygon.points[2] == Point(x=120, y=230)


def _evidence_at(
    token_id: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            page_number=1,
            table_id="table-1",
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


def _description_recovery_source_table() -> SourceTable:
    columns = (
        SourceColumn(
            id="description",
            label="ProductName",
            order=0,
            canonical_field="description",
            evidence=_evidence_at("h-description", 100, 100, 220, 120),
        ),
        SourceColumn(
            id="batch",
            label="Batch No",
            order=1,
            evidence=_evidence_at("h-batch", 400, 100, 470, 120),
        ),
        SourceColumn(
            id="quantity",
            label="Qty",
            order=2,
            canonical_field="quantity",
            evidence=_evidence_at("h-quantity", 500, 100, 540, 120),
        ),
        SourceColumn(
            id="rate",
            label="Rate",
            order=3,
            canonical_field="unit_price",
            evidence=_evidence_at("h-rate", 600, 100, 640, 120),
        ),
        SourceColumn(
            id="total",
            label="Total",
            order=4,
            canonical_field="net_amount",
            evidence=_evidence_at("h-total", 700, 100, 750, 120),
        ),
    )

    def cell(
        column_id: str,
        value: str | None,
        row: int,
    ) -> SourceCell:
        x_by_column = {
            "description": 110,
            "batch": 410,
            "quantity": 510,
            "rate": 610,
            "total": 710,
        }
        return SourceCell(
            column_id=column_id,
            raw_value=value,
            evidence=(
                _evidence_at(
                    f"r{row}-{column_id}",
                    x_by_column[column_id],
                    140 + row * 30,
                    x_by_column[column_id] + 25,
                    160 + row * 30,
                )
                if value
                else ()
            ),
        )

    values = (
        ("Emeset", "A1", "1", "25.44", "25.44"),
        (None, "B2", "3", "37.23", "111.69"),
        ("RoadSterile Water", "C3", "5", "3.04", "15.20"),
        (None, "D4", "4", "153.80", "615.20"),
        (None, None, "BILL TOTAL", None, "767.53"),
    )
    rows = tuple(
        SourceRow(
            id=f"row-{order}",
            order=order,
            cells=tuple(
                cell(column.id, value, order)
                for column, value in zip(columns, row_values, strict=True)
            ),
        )
        for order, row_values in enumerate(values)
    )
    return SourceTable(
        id="table-1-s1",
        page_number=1,
        table_id="table-1",
        table_type=TableType.PHARMACY,
        columns=columns,
        rows=rows,
    )


def _return_sign_recovery_baseline() -> ReconstructionResult:
    table = _description_recovery_source_table()
    first_cells = list(table.rows[0].cells)
    first_cells[-1] = first_cells[-1].model_copy(
        update={
            "raw_value": "23.93",
            "evidence": _evidence_at(
                "token-0-amount",
                710,
                140,
                735,
                160,
            ),
        }
    )
    table = table.model_copy(
        update={"rows": (table.rows[0].model_copy(update={"cells": tuple(first_cells)}),)}
    )
    return ReconstructionResult(
        rows=(
            _aligned_row(
                0,
                amount=Decimal("23.93"),
                flags=("positive_amount_in_return_section",),
                table_type=TableType.PHARMACY,
            ),
        ),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(table,),
    )


def test_description_lane_recovery_targets_only_consecutive_grounded_detail_rows() -> None:
    reconstruction = ReconstructionResult(
        rows=(),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(_description_recovery_source_table(),),
    )

    assert description_lane_recovery_regions(
        reconstruction,
        table_box=(50, 80, 800, 300),
    ) == ((100, 165, 400, 255),)


def test_return_sign_recovery_targets_only_the_flagged_grounded_amount_cell() -> None:
    targets = return_sign_recovery_targets(
        _return_sign_recovery_baseline(),
        table_box=(50, 80, 800, 300),
    )

    assert len(targets) == 1
    assert targets[0].expected_absolute_amount == Decimal("23.93")
    assert targets[0].region == (670, 130, 745, 170)


def test_description_recovery_boundary_ignores_rotated_overlay_only_row() -> None:
    table = _description_recovery_source_table()
    target = table.rows[1]
    overlay = SourceRow(
        id="stamp-overlay",
        order=1,
        cells=(
            SourceCell(
                column_id="description",
                raw_value="Hospital address",
                evidence=_evidence_at(
                    "stamp-overlay",
                    100,
                    155,
                    380,
                    178,
                ),
                validation_flags=("all_text_rotated",),
            ),
            *(
                SourceCell(
                    column_id=column.id,
                    raw_value=None,
                    evidence=(),
                    validation_flags=("empty_cell",),
                )
                for column in table.columns[1:]
            ),
        ),
    )
    reconstruction = ReconstructionResult(
        rows=(),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(table.model_copy(update={"rows": (table.rows[0], overlay, target)}),),
    )

    assert description_lane_recovery_regions(
        reconstruction,
        table_box=(50, 80, 800, 300),
    ) == ((100, 165, 400, 200),)


def test_first_description_recovery_band_stays_below_grounded_headers() -> None:
    table = _description_recovery_source_table()
    first_cells = list(table.rows[0].cells)
    first_cells[0] = first_cells[0].model_copy(update={"raw_value": None, "evidence": ()})
    first_cells[1] = first_cells[1].model_copy(
        update={
            "evidence": _evidence_at(
                "r0-tall-batch-overlay",
                410,
                80,
                470,
                180,
            )
        }
    )
    first_row = table.rows[0].model_copy(update={"cells": tuple(first_cells)})
    reconstruction = ReconstructionResult(
        rows=(),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(table.model_copy(update={"rows": (first_row, *table.rows[1:])}),),
    )

    assert description_lane_recovery_regions(
        reconstruction,
        table_box=(50, 80, 800, 300),
    ) == ((100, 120, 400, 255),)


def test_description_recovery_uses_parseable_gross_when_net_cell_is_blank() -> None:
    table = _description_recovery_source_table()
    gross_column = SourceColumn(
        id="gross",
        label="Gross",
        order=len(table.columns),
        canonical_field="gross_amount",
        evidence=_evidence_at("h-gross", 760, 100, 790, 120),
    )
    target_row = table.rows[1]
    target_cells = [
        cell.model_copy(update={"raw_value": None, "evidence": ()})
        if cell.column_id == "total"
        else cell
        for cell in target_row.cells
    ]
    target_cells.append(
        SourceCell(
            column_id="gross",
            raw_value="111.69",
            evidence=_evidence_at("r1-gross", 760, 170, 790, 190),
        )
    )
    target_row = target_row.model_copy(update={"cells": tuple(target_cells)})
    reconstruction = ReconstructionResult(
        rows=(),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(
            table.model_copy(
                update={
                    "columns": (*table.columns, gross_column),
                    "rows": (target_row,),
                }
            ),
        ),
    )

    assert description_lane_recovery_regions(
        reconstruction,
        table_box=(50, 80, 800, 300),
    ) == ((100, 160, 400, 200),)


def test_description_recovery_does_not_require_a_printed_quantity() -> None:
    table = _description_recovery_source_table()
    target_cells = tuple(
        cell.model_copy(
            update={
                "raw_value": None,
                "evidence": (),
                "validation_flags": ("empty_cell",),
            }
        )
        if cell.column_id == "quantity"
        else cell
        for cell in table.rows[1].cells
    )
    target_row = table.rows[1].model_copy(update={"cells": target_cells})
    reconstruction = ReconstructionResult(
        rows=(),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(table.model_copy(update={"rows": (target_row,)}),),
    )

    assert description_lane_recovery_regions(
        reconstruction,
        table_box=(50, 80, 800, 300),
    ) == ((100, 160, 400, 200),)


def test_normal_yield_with_missing_grounded_descriptions_enters_crop_recovery() -> None:
    reconstruction = ReconstructionResult(
        rows=(_aligned_row(0, description="Emeset"),),
        schema=_schema(TableType.PHARMACY),
        diagnostics={"table_type": TableType.PHARMACY.value},
        source_tables=(_description_recovery_source_table(),),
    )

    assert offline_module._should_attempt_crop_recovery(
        reconstruction,
        parsed_rows=(object(),),
        table_box=(50, 80, 800, 300),
    )


def test_page_region_maps_to_high_resolution_crop_pixels() -> None:
    assert map_page_box_to_crop_pixels(
        (100, 200, 400, 500),
        parent_page_box=(50, 100, 650, 700),
        crop_width=1200,
        crop_height=900,
    ) == (100, 150, 700, 600)


def test_targeted_tokens_replace_only_ocr_inside_recovery_regions() -> None:
    original = (
        _ocr_token("keep-left", "a" * 64, "keep-left").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=10, y=200),
                        Point(x=40, y=200),
                        Point(x=40, y=220),
                        Point(x=10, y=220),
                    )
                )
            }
        ),
        _ocr_token("RoadSterile Water", "a" * 64, "replace").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=120, y=200),
                        Point(x=350, y=200),
                        Point(x=350, y=220),
                        Point(x=120, y=220),
                    )
                )
            }
        ),
        _ocr_token("keep-right", "a" * 64, "keep-right").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=500, y=200),
                        Point(x=550, y=200),
                        Point(x=550, y=220),
                        Point(x=500, y=220),
                    )
                )
            }
        ),
    )
    recovered = (_ocr_token("Sterile Water 10ML", "a" * 64, "recovered"),)

    merged = replace_tokens_in_regions(
        original,
        recovered,
        regions=((100, 185, 400, 245),),
    )

    assert tuple(token.token_id for token in merged) == (
        "keep-left",
        "keep-right",
        "recovered",
    )


def test_targeted_merge_preserves_baseline_evidence_and_adds_only_missing_geometry() -> None:
    baseline = (
        _ocr_token("RoadSterile Water", "a" * 64, "baseline-target").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=120, y=200),
                        Point(x=350, y=200),
                        Point(x=350, y=220),
                        Point(x=120, y=220),
                    )
                )
            }
        ),
        _ocr_token("25.44", "a" * 64, "baseline-amount").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=500, y=200),
                        Point(x=550, y=200),
                        Point(x=550, y=220),
                        Point(x=500, y=220),
                    )
                )
            }
        ),
    )
    high_resolution = (
        _ocr_token("25,44", "a" * 64, "high-amount").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=502, y=201),
                        Point(x=552, y=201),
                        Point(x=552, y=221),
                        Point(x=502, y=221),
                    )
                )
            }
        ),
        _ocr_token("4", "a" * 64, "high-missing-quantity").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=600, y=200),
                        Point(x=620, y=200),
                        Point(x=620, y=220),
                        Point(x=600, y=220),
                    )
                )
            }
        ),
    )
    targeted = (
        _ocr_token("Sterile Water 10ML", "a" * 64, "target-description").model_copy(
            update={
                "polygon": Polygon(
                    points=(
                        Point(x=120, y=200),
                        Point(x=350, y=200),
                        Point(x=350, y=220),
                        Point(x=120, y=220),
                    )
                )
            }
        ),
    )

    merged = merge_recovery_tokens(
        baseline,
        high_resolution,
        targeted,
        regions=((100, 185, 400, 245),),
    )

    assert tuple(token.token_id for token in merged) == (
        "baseline-amount",
        "high-missing-quantity",
        "target-description",
    )


def test_implausibly_low_yield_is_escalated() -> None:
    reconstruction = ReconstructionResult(
        rows=(object(),),
        schema=None,
        diagnostics={"ocr_line_count": 20, "ocr_row_count": 1},
    )
    assert is_implausibly_low_yield(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)
    assert RecoveryReason.LOW_YIELD in decision.reasons


def test_empty_metadata_region_is_terminal_without_recovery() -> None:
    reconstruction = ReconstructionResult(
        rows=(),
        schema=None,
        diagnostics={"table_type": TableType.METADATA.value},
    )
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)
    assert decision.reasons == ()
    assert decision.planned_stages == ()
    assert decision.route == "local_ocr"


@pytest.mark.parametrize(
    "flag",
    (
        "missing_labeled_quantity",
        "missing_labeled_unit_price",
        "line_arithmetic_mismatch",
        "positive_amount_in_return_section",
    ),
)
def test_grounded_field_defects_route_to_local_validation_recovery(flag: str) -> None:
    reconstruction = _reconstruction((_aligned_row(0, flags=(flag,)),))

    assert needs_field_quality_recovery(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)

    assert decision.reasons == (RecoveryReason.VALIDATION_FAILURE,)
    assert decision.route == "local_recovery"
    assert RecoveryStage.CROP_OCR in decision.planned_stages


def test_selected_profile_with_field_defects_still_routes_to_local_recovery() -> None:
    reconstruction = _reconstruction((_aligned_row(0, flags=("missing_labeled_quantity",)),))
    selected_profile = ProfileMatch(
        profile_key="hospital",
        profile_version=1,
        score=0.99,
        margin=0.5,
        selected=True,
    )

    decision = decide_recovery(
        reconstruction,
        gemini_mode=GeminiMode.OFF,
        profile_match=selected_profile,
    )

    assert decision.route == "local_recovery"
    assert decision.reasons == (RecoveryReason.VALIDATION_FAILURE,)


@pytest.mark.parametrize("table_type", (TableType.METADATA, TableType.PAYMENT))
def test_terminal_regions_ignore_field_quality_defects(table_type: TableType) -> None:
    reconstruction = _reconstruction(
        (_aligned_row(0, flags=("missing_labeled_quantity",)),),
        table_type=table_type,
    )

    assert not needs_field_quality_recovery(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)

    assert decision.reasons == ()
    assert decision.planned_stages == ()


def test_haemorrhoidectomy_anaesthetist_same_row_recovery_is_a_safe_improvement() -> None:
    unchanged = tuple(_aligned_row(index, description=f"Service {index}") for index in range(12))
    baseline = _reconstruction(
        (
            *unchanged,
            _aligned_row(
                12,
                flags=("missing_labeled_quantity",),
            ),
        )
    )
    recovered = _reconstruction(
        (
            *unchanged,
            _aligned_row(
                12,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert len(baseline.rows) == len(recovered.rows) == 13
    assert recovered.rows[-1].candidate.quantity == Decimal("1")
    assert "missing_labeled_quantity" not in recovered.rows[-1].candidate.validation_flags
    assert reconstruction_quality(recovered) > reconstruction_quality(baseline)
    assert safely_improves_reconstruction(baseline, recovered)


def test_candidate_with_fewer_canonical_rows_cannot_replace_baseline() -> None:
    baseline = _reconstruction((_aligned_row(0), _aligned_row(1)))
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert not safely_improves_reconstruction(baseline, candidate)


def test_candidate_with_same_aligned_count_but_fewer_publishable_rows_is_rejected() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(0, flags=("missing_labeled_quantity",)),
            _aligned_row(1),
        )
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
            _aligned_row(
                1,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
                role=RowRole.SECTION_TOTAL,
            ),
        )
    )

    assert len(candidate.rows) == len(baseline.rows)
    assert not safely_improves_reconstruction(baseline, candidate)


def test_nonpublishable_rows_cannot_hide_mapped_field_coverage_regression() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
            _aligned_row(1, mapped_fields=(), role=RowRole.SECTION_TOTAL),
        )
    )
    candidate = _reconstruction(
        (
            _aligned_row(0, mapped_fields=("description", "amount")),
            _aligned_row(
                1,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
                role=RowRole.SECTION_TOTAL,
            ),
        )
    )

    assert not safely_improves_reconstruction(baseline, candidate)


def test_equal_aggregate_candidate_cannot_replace_a_different_source_row() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                description="Anaesthetist",
                flags=("missing_labeled_quantity",),
            ),
            _aligned_row(1, description="Assistant surgeon"),
        )
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                description="Anaesthetist",
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
            _aligned_row(7, description="Blood bank charge"),
        )
    )

    assert reconstruction_quality(candidate) > reconstruction_quality(baseline)
    assert not safely_improves_reconstruction(baseline, candidate)


def test_field_gain_on_one_row_cannot_hide_grounded_date_loss_on_another() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                description="Anaesthetist",
                flags=("missing_labeled_quantity",),
            ),
            _aligned_row(
                1,
                description="Assistant surgeon",
                service_date="20/01/2026",
                mapped_fields=("description", "service_date", "rate", "amount"),
            ),
        )
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                description="Anaesthetist",
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
            _aligned_row(1, description="Assistant surgeon"),
        )
    )

    assert reconstruction_quality(candidate) > reconstruction_quality(baseline)
    assert not safely_improves_reconstruction(baseline, candidate)


@pytest.mark.parametrize("table_type", (TableType.PAYMENT, TableType.LABORATORY))
def test_incompatible_terminal_or_table_type_candidate_is_rejected(
    table_type: TableType,
) -> None:
    baseline = _reconstruction((_aligned_row(0, flags=("missing_labeled_quantity",)),))
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
                table_type=table_type,
            ),
        ),
        table_type=table_type,
    )

    assert not safely_improves_reconstruction(baseline, candidate)


@pytest.mark.parametrize(
    "candidate",
    (
        _reconstruction(
            (
                _aligned_row(
                    0,
                    flags=(
                        "missing_labeled_quantity",
                        "missing_labeled_unit_price",
                    ),
                ),
            )
        ),
        _reconstruction(
            (
                _aligned_row(
                    0,
                    quantity=Decimal("1"),
                    mapped_fields=("description", "amount"),
                ),
            )
        ),
    ),
)
def test_validation_or_mapped_field_regression_cannot_replace_baseline(
    candidate: ReconstructionResult,
) -> None:
    baseline = _reconstruction((_aligned_row(0, flags=("missing_labeled_quantity",)),))

    assert not safely_improves_reconstruction(baseline, candidate)


def test_populated_source_cells_break_an_otherwise_equal_quality_tie() -> None:
    rows = (_aligned_row(0),)
    baseline = ReconstructionResult(
        rows=rows,
        schema=_schema(),
        diagnostics={"table_type": TableType.ITEM_LEDGER.value},
        source_tables=(_source_table(None, None),),
    )
    candidate = ReconstructionResult(
        rows=rows,
        schema=baseline.schema,
        diagnostics=baseline.diagnostics,
        source_tables=(_source_table("Anaesthetist", "1"),),
    )

    assert reconstruction_quality(candidate) > reconstruction_quality(baseline)
    assert safely_improves_reconstruction(baseline, candidate)


def test_source_cell_regression_does_not_veto_better_grounded_field_quality() -> None:
    baseline_row = _aligned_row(0, flags=("missing_labeled_quantity",))
    baseline = ReconstructionResult(
        rows=(baseline_row,),
        schema=_schema(),
        diagnostics={"table_type": TableType.ITEM_LEDGER.value},
        source_tables=(_source_table("Anaesthetist", "100"),),
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert safely_improves_reconstruction(baseline, candidate)


def test_recovery_cannot_change_an_existing_grounded_financial_value() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                flags=("missing_labeled_quantity",),
                mapped_fields=("description", "amount"),
            ),
        )
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                amount=Decimal("999"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert not safely_improves_reconstruction(baseline, candidate)


def test_recovery_can_correct_only_a_grounded_missing_refund_sign() -> None:
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                description="Metronidazole IV 100ML",
                amount=Decimal("23.93"),
                flags=(
                    "positive_amount_in_return_section",
                    "missing_labeled_quantity",
                ),
                role=RowRole.DETAIL,
                table_type=TableType.PHARMACY,
            ),
        ),
        table_type=TableType.PHARMACY,
    )
    corrected = _reconstruction(
        (
            _aligned_row(
                0,
                description="Metronidazole IV 100ML",
                amount=Decimal("-23.93"),
                role=RowRole.REFUND,
                table_type=TableType.PHARMACY,
            ),
        ),
        table_type=TableType.PHARMACY,
    )
    wrong_amount = replace(
        corrected,
        rows=(
            _aligned_row(
                0,
                description="Metronidazole IV 100ML",
                amount=Decimal("-24"),
                role=RowRole.REFUND,
                table_type=TableType.PHARMACY,
            ),
        ),
    )
    arithmetically_unsafe = replace(
        corrected,
        rows=(
            replace(
                corrected.rows[0],
                candidate=replace(
                    corrected.rows[0].candidate,
                    validation_flags=("line_arithmetic_mismatch",),
                ),
            ),
        ),
    )
    unflagged = replace(
        baseline,
        rows=(
            replace(
                baseline.rows[0],
                candidate=replace(
                    baseline.rows[0].candidate,
                    validation_flags=(),
                ),
            ),
        ),
    )

    assert safely_improves_reconstruction(baseline, corrected)
    assert not safely_improves_reconstruction(baseline, wrong_amount)
    assert not safely_improves_reconstruction(baseline, arithmetically_unsafe)
    assert not safely_improves_reconstruction(unflagged, corrected)


def test_recovery_can_add_a_grounded_row_with_only_an_optional_field_missing() -> None:
    baseline = _reconstruction(
        (_aligned_row(0, description="Existing item"),),
        table_type=TableType.PHARMACY,
    )
    recovered = _reconstruction(
        (
            _aligned_row(0, description="Existing item"),
            _aligned_row(
                1,
                description="Under Pad 10S",
                rate=Decimal("153.80"),
                amount=Decimal("615.20"),
                flags=("missing_labeled_quantity",),
                table_type=TableType.PHARMACY,
            ),
        ),
        table_type=TableType.PHARMACY,
    )
    arithmetically_unsafe = replace(
        recovered,
        rows=(
            recovered.rows[0],
            replace(
                recovered.rows[1],
                candidate=replace(
                    recovered.rows[1].candidate,
                    validation_flags=("line_arithmetic_mismatch",),
                ),
            ),
        ),
    )

    assert safely_improves_reconstruction(baseline, recovered)
    assert not safely_improves_reconstruction(baseline, arithmetically_unsafe)


@pytest.mark.parametrize(
    "bad_first_variant",
    (False, True),
    ids=("rank-all-variants", "continue-after-bad-first-variant"),
)
def test_crop_recovery_isolates_and_ranks_grounded_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_first_variant: bool,
) -> None:
    page_artifact_sha256 = "a" * 64
    high_artifact_sha256 = "b" * 64
    clahe_artifact_sha256 = "c" * 64
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                flags=(
                    "missing_labeled_quantity",
                    "missing_labeled_unit_price",
                    "line_arithmetic_mismatch",
                ),
            ),
        )
    )
    observed_token_batches: list[tuple[OcrToken, ...]] = []

    monkeypatch.setattr(
        offline_module,
        "render_pdf_region",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "high.png",
            artifact_sha256=high_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "clahe_variant",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "clahe.png",
            artifact_sha256=clahe_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "_cached_prediction",
        lambda path, request, adapter: (
            SimpleNamespace(
                output={"variant": request.options["input_variant"]},
                latency_ms=1,
            ),
            False,
        ),
    )

    def fake_paddle_tokens(output, page_number, artifact_sha256):
        if bad_first_variant and output["variant"] == "high_resolution":
            raise ValueError("bad high-resolution OCR payload")
        return tuple(
            _ocr_token(
                text,
                artifact_sha256,
                f"local-{output['variant']}-{field}",
            )
            for field, text in (
                (
                    ("description", "Anaesthetist"),
                    ("quantity", "1"),
                    ("amount", "100"),
                )
                if output["variant"] == "high_resolution"
                else (
                    ("description", "Anaesthetist"),
                    ("quantity", "1"),
                    ("rate", "100"),
                    ("amount", "100"),
                )
            )
        )

    monkeypatch.setattr(offline_module, "paddle_ocr_tokens", fake_paddle_tokens)
    monkeypatch.setattr(
        offline_module.cv2,
        "imread",
        lambda path, mode: SimpleNamespace(shape=(100, 200, 3)),
    )

    def fake_reconstruct(tokens, **kwargs):
        observed_token_batches.append(tokens)
        field_tokens = {
            token.token_id.rsplit("-", maxsplit=1)[-1]: (token.token_id,) for token in tokens
        }
        quantity = Decimal(
            next(token.text for token in tokens if token.token_id.endswith("-quantity"))
        )
        if len(tokens) == 3:
            row = _aligned_row(
                0,
                quantity=quantity,
                flags=(
                    "missing_labeled_unit_price",
                    "line_arithmetic_mismatch",
                ),
                mapped_fields=(),
            )
        else:
            rate = Decimal(next(token.text for token in tokens if token.token_id.endswith("-rate")))
            row = _aligned_row(
                0,
                quantity=quantity,
                rate=rate,
                mapped_fields=(),
            )
        return _reconstruction(
            (
                replace(
                    row,
                    field_token_ids=field_tokens,
                    evidence_token_ids=tuple(
                        token_id for token_ids in field_tokens.values() for token_id in token_ids
                    ),
                ),
            )
        )

    monkeypatch.setattr(offline_module, "reconstruct_ocr_rows", fake_reconstruct)
    extractor = object.__new__(OfflineExtractor)
    extractor.ocr = object()
    work = TableWork(
        table_id="table-1",
        page_number=1,
        page_artifact_sha256=page_artifact_sha256,
        crop_path=tmp_path / "primary.png",
        crop_sha256="d" * 64,
        box=(200, 300, 400, 400),
    )

    recovered, attempts, recovered_manifest = extractor._recover_crop_ocr(
        source=tmp_path / "bill.pdf",
        artifact_root=tmp_path,
        work=work,
        prior_schemas=(),
        page_artifact_sha256=page_artifact_sha256,
        page_artifact_relative_path="pages/page-1.png",
        baseline=baseline,
        baseline_tokens=(),
    )
    assert recovered_manifest

    assert recovered is not None
    assert recovered.rows[0].candidate.validation_flags == ()
    assert recovered.rows[0].candidate.quantity == Decimal("1")
    assert [len(tokens) for tokens in observed_token_batches] == (
        [4] if bad_first_variant else [3, 4]
    )
    observed_tokens = tuple(token for tokens in observed_token_batches for token in tokens)
    assert all(token.artifact_sha256 == page_artifact_sha256 for token in observed_tokens)
    assert all(token.token_id.startswith("recovery:") for token in observed_tokens)
    assert recovered.rows[0].field_token_ids["quantity"] == ("recovery:local-photometric-quantity",)
    crop_attempts = [attempt for attempt in attempts if attempt.stage is RecoveryStage.CROP_OCR]
    assert [attempt.status for attempt in crop_attempts] == (
        ["failed", "recovered"] if bad_first_variant else ["no_improvement", "recovered"]
    )
    assert [attempt.accepted_rows for attempt in crop_attempts] == [0, 1]


def test_crop_recovery_uses_targeted_description_lane_for_grounded_financial_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_artifact_sha256 = "a" * 64
    high_artifact_sha256 = "b" * 64
    overlay_artifact_sha256 = "c" * 64
    target_artifact_sha256 = "d" * 64
    baseline = _reconstruction(
        (_aligned_row(0, description="Emeset 2ML"),),
        table_type=TableType.PHARMACY,
    )
    source_table = _description_recovery_source_table()
    observed_variants: list[str] = []
    reconstructed_token_batches: list[tuple[OcrToken, ...]] = []

    monkeypatch.setattr(
        offline_module,
        "render_pdf_region",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "high.png",
            artifact_sha256=high_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "clahe_variant",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("skip CLAHE")),
    )
    monkeypatch.setattr(
        offline_module,
        "color_overlay_suppressed_variant",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "overlay.png",
            artifact_sha256=overlay_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "crop_region",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "description-lane.png",
            artifact_sha256=target_artifact_sha256,
        ),
    )

    def fake_prediction(path, request, adapter):
        variant = str(request.options["input_variant"])
        observed_variants.append(variant)
        return (
            SimpleNamespace(
                output={"variant": variant},
                latency_ms=1,
            ),
            False,
        )

    monkeypatch.setattr(offline_module, "_cached_prediction", fake_prediction)
    monkeypatch.setattr(
        offline_module,
        "paddle_ocr_tokens",
        lambda output, page_number, artifact_sha256: (
            _ocr_token(
                ("Ns 500ML" if output["variant"] == "description_lane" else "Emeset 2ML"),
                artifact_sha256,
                (
                    "target-description"
                    if output["variant"] == "description_lane"
                    else "full-description"
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        offline_module.cv2,
        "imread",
        lambda path, mode: SimpleNamespace(
            shape=((60, 300, 3) if str(path).endswith("description-lane.png") else (220, 800, 3))
        ),
    )

    def fake_reconstruct(tokens, **kwargs):
        reconstructed_token_batches.append(tokens)
        if any("target-description" in token.token_id for token in tokens):
            return ReconstructionResult(
                rows=(
                    _aligned_row(0, description="Emeset 2ML"),
                    _aligned_row(1, description="Ns 500ML"),
                ),
                schema=_schema(TableType.PHARMACY),
                diagnostics={"table_type": TableType.PHARMACY.value},
                source_tables=(source_table,),
            )
        return ReconstructionResult(
            rows=(_aligned_row(0, description="Emeset 2ML"),),
            schema=_schema(TableType.PHARMACY),
            diagnostics={"table_type": TableType.PHARMACY.value},
            source_tables=(source_table,),
        )

    monkeypatch.setattr(offline_module, "reconstruct_ocr_rows", fake_reconstruct)
    extractor = object.__new__(OfflineExtractor)
    extractor.ocr = object()
    work = TableWork(
        table_id="table-1",
        page_number=1,
        page_artifact_sha256=page_artifact_sha256,
        crop_path=tmp_path / "primary.png",
        crop_sha256="e" * 64,
        box=(50, 80, 800, 300),
    )

    recovered, attempts, recovered_manifest = extractor._recover_crop_ocr(
        source=tmp_path / "bill.pdf",
        artifact_root=tmp_path,
        work=work,
        prior_schemas=(),
        page_artifact_sha256=page_artifact_sha256,
        page_artifact_relative_path="pages/page-1.png",
        baseline=baseline,
        baseline_tokens=(
            _ocr_token(
                "25.44",
                page_artifact_sha256,
                "page-baseline-amount",
            ).model_copy(
                update={
                    "polygon": Polygon(
                        points=(
                            Point(x=500, y=200),
                            Point(x=550, y=200),
                            Point(x=550, y=220),
                            Point(x=500, y=220),
                        )
                    )
                }
            ),
        ),
    )
    assert recovered_manifest

    assert recovered is not None
    assert tuple(row.candidate.description for row in recovered.rows) == (
        "Emeset 2ML",
        "Ns 500ML",
    )
    assert observed_variants == ["high_resolution", "description_lane"]
    targeted_batch = next(
        batch
        for batch in reconstructed_token_batches
        if any("target-description" in token.token_id for token in batch)
    )
    assert "page-baseline-amount" in {token.token_id for token in targeted_batch}
    assert not any("full-description" in token.token_id for token in targeted_batch)
    assert any(
        attempt.status == "recovered"
        and attempt.reason == "input_variant:high_resolution+description_lane"
        for attempt in attempts
    )


def test_crop_recovery_uses_800dpi_clahe_only_for_a_grounded_refund_sign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_artifact_sha256 = "a" * 64
    baseline = _return_sign_recovery_baseline()
    render_calls: list[tuple[tuple[int, int, int, int], int]] = []
    observed_variants: list[str] = []
    reconstructed_batches: list[tuple[OcrToken, ...]] = []

    def fake_render(source, output, page_number, box, **kwargs):
        output_dpi = int(kwargs.get("output_dpi", 400))
        render_calls.append((box, output_dpi))
        return SimpleNamespace(
            output_path=tmp_path / f"render-{output_dpi}.png",
            artifact_sha256=("b" if output_dpi == 400 else "c") * 64,
        )

    monkeypatch.setattr(offline_module, "render_pdf_region", fake_render)
    monkeypatch.setattr(
        offline_module,
        "clahe_variant",
        lambda source, output: SimpleNamespace(
            output_path=(
                tmp_path / "sign-clahe.png" if "800" in str(source) else tmp_path / "high-clahe.png"
            ),
            artifact_sha256=("d" if "800" in str(source) else "e") * 64,
        ),
    )

    def fake_prediction(path, request, adapter):
        variant = str(request.options["input_variant"])
        observed_variants.append(variant)
        return SimpleNamespace(output={"variant": variant}, latency_ms=1), False

    monkeypatch.setattr(offline_module, "_cached_prediction", fake_prediction)
    monkeypatch.setattr(
        offline_module,
        "paddle_ocr_tokens",
        lambda output, page_number, artifact_sha256: (
            _ocr_token(
                ("-23.93" if output["variant"] == "return_sign_800dpi_clahe" else "23.93"),
                artifact_sha256,
                "local-amount",
            ),
        ),
    )
    monkeypatch.setattr(
        offline_module.cv2,
        "imread",
        lambda path, mode: SimpleNamespace(shape=(100, 100, 3)),
    )

    def fake_reconstruct(tokens, **kwargs):
        reconstructed_batches.append(tokens)
        if any(parse_decimal(token.text) == Decimal("-23.93") for token in tokens):
            return replace(
                baseline,
                rows=(
                    _aligned_row(
                        0,
                        amount=Decimal("-23.93"),
                        role=RowRole.REFUND,
                        table_type=TableType.PHARMACY,
                    ),
                ),
            )
        return baseline

    monkeypatch.setattr(offline_module, "reconstruct_ocr_rows", fake_reconstruct)
    extractor = object.__new__(OfflineExtractor)
    extractor.ocr = object()
    work = TableWork(
        table_id="table-1",
        page_number=1,
        page_artifact_sha256=page_artifact_sha256,
        crop_path=tmp_path / "primary.png",
        crop_sha256="f" * 64,
        box=(50, 80, 800, 300),
    )
    baseline_amount_token = _ocr_token(
        "23.93",
        page_artifact_sha256,
        "token-0-amount",
    ).model_copy(
        update={
            "polygon": Polygon(
                points=(
                    Point(x=710, y=140),
                    Point(x=735, y=140),
                    Point(x=735, y=160),
                    Point(x=710, y=160),
                )
            )
        }
    )

    recovered, attempts, recovered_manifest = extractor._recover_crop_ocr(
        source=tmp_path / "bill.pdf",
        artifact_root=tmp_path,
        work=work,
        prior_schemas=(),
        page_artifact_sha256=page_artifact_sha256,
        page_artifact_relative_path="pages/page-1.png",
        baseline=baseline,
        baseline_tokens=(baseline_amount_token,),
    )
    assert recovered_manifest

    assert recovered is not None
    assert recovered.rows[0].candidate.amount == Decimal("-23.93")
    assert recovered.rows[0].candidate.role is RowRole.REFUND
    assert ((670, 130, 745, 170), 800) in render_calls
    assert "return_sign_800dpi_clahe" in observed_variants
    sign_batch = next(
        tokens
        for tokens in reconstructed_batches
        if any(token.text == "-23.93" for token in tokens)
    )
    assert not any(token.text == "23.93" for token in sign_batch)
    assert any(
        attempt.status == "recovered" and "return_sign" in str(attempt.reason)
        for attempt in attempts
    )
