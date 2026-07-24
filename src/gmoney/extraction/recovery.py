from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from functools import lru_cache
from math import ceil

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType
from gmoney.contracts.phase3 import (
    AdjudicationResponse,
    GeminiMode,
    ProfileMatch,
    RecoveryReason,
    RecoveryStage,
    RouteDecision,
)
from gmoney.extraction.canonicalize import is_publishable_aligned_row
from gmoney.extraction.ocr_rows import ReconstructionResult
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow
from gmoney.extraction.typed_values import parse_decimal

FIELD_QUALITY_FLAGS = frozenset(
    {
        "missing_labeled_quantity",
        "missing_labeled_unit_price",
        "line_arithmetic_mismatch",
        "positive_amount_in_return_section",
    }
)
MAPPED_CANONICAL_FIELDS = frozenset(
    {
        "description",
        "service_date",
        "request_no",
        "service_code",
        "hsn_code",
        "quantity",
        "rate",
        "gross_amount",
        "discount",
        "amount",
    }
)


@dataclass(frozen=True)
class GroundingResult:
    rows: tuple[AlignedLedgerRow, ...]
    rejected_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReturnSignRecoveryTarget:
    region: tuple[int, int, int, int]
    expected_absolute_amount: Decimal


def is_implausibly_low_yield(reconstruction: ReconstructionResult) -> bool:
    row_count = len(reconstruction.rows)
    line_count = int(reconstruction.diagnostics.get("ocr_line_count") or 0)
    return 0 < row_count <= 2 and line_count >= 12 and row_count / line_count < 0.2


def is_terminal_non_ledger(reconstruction: ReconstructionResult) -> bool:
    table_type = (
        reconstruction.schema.table_type.value
        if reconstruction.schema is not None
        else reconstruction.diagnostics.get("table_type")
    )
    return table_type in {TableType.METADATA.value, TableType.PAYMENT.value}


def needs_field_quality_recovery(reconstruction: ReconstructionResult) -> bool:
    if is_terminal_non_ledger(reconstruction):
        return False
    return any(
        FIELD_QUALITY_FLAGS.intersection(
            getattr(getattr(row, "candidate", None), "validation_flags", ())
        )
        for row in reconstruction.rows
    )


def _field_quality_defects(reconstruction: ReconstructionResult) -> tuple[int, int]:
    flags = tuple(
        flag
        for row in reconstruction.rows
        for flag in row.candidate.validation_flags
    )
    return (
        sum(
            flags.count(flag)
            for flag in (
                "line_arithmetic_mismatch",
                "positive_amount_in_return_section",
            )
        ),
        sum(
            flags.count(flag)
            for flag in ("missing_labeled_quantity", "missing_labeled_unit_price")
        ),
    )


def _is_populated(value: object) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _mapped_field_coverage(reconstruction: ReconstructionResult) -> int:
    return sum(
        1
        for row in reconstruction.rows
        if is_publishable_aligned_row(row)
        for field, token_ids in row.field_token_ids.items()
        if field in MAPPED_CANONICAL_FIELDS
        and token_ids
        and _is_populated(getattr(row.candidate, field, None))
    )


def _populated_source_cells(reconstruction: ReconstructionResult) -> int:
    return sum(
        1
        for table in reconstruction.source_tables
        for row in table.rows
        for cell in row.cells
        if _is_populated(cell.raw_value)
    )


def reconstruction_quality(reconstruction: ReconstructionResult) -> tuple[int, ...]:
    arithmetic_mismatches, missing_labeled_fields = _field_quality_defects(reconstruction)
    publishable_rows = tuple(
        row for row in reconstruction.rows if is_publishable_aligned_row(row)
    )
    grounded_rows = sum(
        bool(row.evidence_token_ids and row.evidence_box) for row in publishable_rows
    )
    source_rows = sum(len(table.rows) for table in reconstruction.source_tables)
    return (
        -arithmetic_mismatches,
        len(publishable_rows),
        -missing_labeled_fields,
        _mapped_field_coverage(reconstruction),
        _populated_source_cells(reconstruction),
        grounded_rows,
        source_rows,
    )


def _table_type(reconstruction: ReconstructionResult) -> TableType:
    if reconstruction.schema is not None:
        return reconstruction.schema.table_type
    try:
        return TableType(str(reconstruction.diagnostics.get("table_type") or "unknown"))
    except ValueError:
        return TableType.UNKNOWN


def _description_similarity(
    baseline: AlignedLedgerRow,
    candidate: AlignedLedgerRow,
) -> float:
    baseline_description = re.sub(
        r"[^a-z0-9]+",
        " ",
        baseline.candidate.description.casefold(),
    ).strip()
    candidate_description = re.sub(
        r"[^a-z0-9]+",
        " ",
        candidate.candidate.description.casefold(),
    ).strip()
    if not baseline_description or not candidate_description:
        return 0.0
    return SequenceMatcher(
        None,
        baseline_description,
        candidate_description,
    ).ratio()


def _geometry_is_compatible(
    baseline: AlignedLedgerRow,
    candidate: AlignedLedgerRow,
) -> bool:
    if baseline.evidence_box is None or candidate.evidence_box is None:
        return False
    _, baseline_top, _, baseline_bottom = baseline.evidence_box
    _, candidate_top, _, candidate_bottom = candidate.evidence_box
    baseline_height = max(1.0, baseline_bottom - baseline_top)
    candidate_height = max(1.0, candidate_bottom - candidate_top)
    overlap = min(baseline_bottom, candidate_bottom) - max(
        baseline_top,
        candidate_top,
    )
    if overlap > 0:
        return True
    baseline_center = (baseline_top + baseline_bottom) / 2
    candidate_center = (candidate_top + candidate_bottom) / 2
    return abs(baseline_center - candidate_center) <= max(
        baseline_height,
        candidate_height,
    ) / 2


def _row_match_score(
    baseline: AlignedLedgerRow,
    candidate: AlignedLedgerRow,
    *,
    baseline_order: int,
    candidate_order: int,
) -> tuple[int, ...] | None:
    description_similarity = _description_similarity(baseline, candidate)
    if description_similarity < 0.75:
        return None
    source_row_matches = (
        baseline.candidate.source_row == candidate.candidate.source_row
    )
    geometry_matches = _geometry_is_compatible(baseline, candidate)
    order_distance = abs(baseline_order - candidate_order)
    if not source_row_matches and not (geometry_matches and order_distance <= 1):
        return None
    source_row_distance = abs(
        baseline.candidate.source_row - candidate.candidate.source_row
    )
    return (
        int(source_row_matches),
        int(geometry_matches),
        int(order_distance == 0),
        round(description_similarity * 1000),
        -source_row_distance,
        -order_distance,
    )


def _match_publishable_rows(
    baseline: ReconstructionResult,
    candidate: ReconstructionResult,
) -> tuple[tuple[AlignedLedgerRow, AlignedLedgerRow], ...] | None:
    baseline_rows = tuple(
        row for row in baseline.rows if is_publishable_aligned_row(row)
    )
    candidate_rows = tuple(
        row for row in candidate.rows if is_publishable_aligned_row(row)
    )
    if len(candidate_rows) < len(baseline_rows):
        return None
    zero_score = (0, 0, 0, 0, 0, 0)

    @lru_cache
    def best_match(
        baseline_order: int,
        candidate_start: int,
    ) -> tuple[tuple[int, ...], int, tuple[int, ...]] | None:
        if baseline_order == len(baseline_rows):
            return zero_score, 1, ()
        remaining = len(baseline_rows) - baseline_order - 1
        best: tuple[tuple[int, ...], int, tuple[int, ...]] | None = None
        for candidate_order in range(
            candidate_start,
            len(candidate_rows) - remaining,
        ):
            row_score = _row_match_score(
                baseline_rows[baseline_order],
                candidate_rows[candidate_order],
                baseline_order=baseline_order,
                candidate_order=candidate_order,
            )
            if row_score is None:
                continue
            suffix = best_match(baseline_order + 1, candidate_order + 1)
            if suffix is None:
                continue
            suffix_score, suffix_count, suffix_mapping = suffix
            total_score = tuple(
                current + following
                for current, following in zip(
                    row_score,
                    suffix_score,
                    strict=True,
                )
            )
            mapping = (candidate_order, *suffix_mapping)
            if best is None or total_score > best[0]:
                best = total_score, suffix_count, mapping
            elif total_score == best[0]:
                best = best[0], min(2, best[1] + suffix_count), best[2]
        return best

    match = best_match(0, 0)
    if match is None or match[1] != 1:
        return None
    return tuple(
        (baseline_row, candidate_rows[candidate_order])
        for baseline_row, candidate_order in zip(
            baseline_rows,
            match[2],
            strict=True,
        )
    )


def _preserves_grounded_fields(
    matches: tuple[tuple[AlignedLedgerRow, AlignedLedgerRow], ...],
) -> bool:
    def equivalent(left: object, right: object) -> bool:
        if isinstance(left, str) and isinstance(right, str):
            return re.sub(r"\s+", " ", left).strip().casefold() == re.sub(
                r"\s+",
                " ",
                right,
            ).strip().casefold()
        return left == right

    def is_grounded_refund_sign_correction(
        baseline: AlignedLedgerRow,
        candidate: AlignedLedgerRow,
        field: str,
        baseline_value: object,
        candidate_value: object,
    ) -> bool:
        return (
            field == "amount"
            and "positive_amount_in_return_section"
            in baseline.candidate.validation_flags
            and baseline.candidate.role is RowRole.DETAIL
            and candidate.candidate.role is RowRole.REFUND
            and bool(candidate.field_token_ids.get(field))
            and isinstance(baseline_value, Decimal)
            and isinstance(candidate_value, Decimal)
            and baseline_value > 0
            and candidate_value == -baseline_value
            and "positive_amount_in_return_section"
            not in candidate.candidate.validation_flags
        )

    for baseline, candidate in matches:
        for field in MAPPED_CANONICAL_FIELDS:
            baseline_value = getattr(baseline.candidate, field, None)
            candidate_value = getattr(candidate.candidate, field, None)
            if (
                baseline.field_token_ids.get(field)
                and _is_populated(baseline_value)
                and (
                    not candidate.field_token_ids.get(field)
                    or not _is_populated(candidate_value)
                    or not equivalent(baseline_value, candidate_value)
                )
            ):
                if is_grounded_refund_sign_correction(
                    baseline,
                    candidate,
                    field,
                    baseline_value,
                    candidate_value,
                ):
                    continue
                return False
    return True


def safely_improves_reconstruction(
    baseline: ReconstructionResult,
    candidate: ReconstructionResult,
) -> bool:
    baseline_type = _table_type(baseline)
    candidate_type = _table_type(candidate)
    if is_terminal_non_ledger(candidate):
        return False
    if baseline_type is not TableType.UNKNOWN and candidate_type is not baseline_type:
        return False
    baseline_publishable = sum(is_publishable_aligned_row(row) for row in baseline.rows)
    candidate_publishable = sum(is_publishable_aligned_row(row) for row in candidate.rows)
    if candidate_publishable < baseline_publishable:
        return False
    matched_rows = _match_publishable_rows(baseline, candidate)
    if matched_rows is None or not _preserves_grounded_fields(matched_rows):
        return False
    for baseline_row, candidate_row in matched_rows:
        baseline_flags = baseline_row.candidate.validation_flags
        candidate_flags = candidate_row.candidate.validation_flags
        baseline_defects = (
            baseline_flags.count("line_arithmetic_mismatch"),
            baseline_flags.count("positive_amount_in_return_section"),
            sum(
                baseline_flags.count(flag)
                for flag in (
                    "missing_labeled_quantity",
                    "missing_labeled_unit_price",
                )
            ),
        )
        candidate_defects = (
            candidate_flags.count("line_arithmetic_mismatch"),
            candidate_flags.count("positive_amount_in_return_section"),
            sum(
                candidate_flags.count(flag)
                for flag in (
                    "missing_labeled_quantity",
                    "missing_labeled_unit_price",
                )
            ),
        )
        if any(
            candidate_count > baseline_count
            for candidate_count, baseline_count in zip(
                candidate_defects,
                baseline_defects,
                strict=True,
            )
        ):
            return False
    matched_candidate_ids = {
        id(candidate_row) for _, candidate_row in matched_rows
    }
    if any(
        {
            "line_arithmetic_mismatch",
            "positive_amount_in_return_section",
        }.intersection(row.candidate.validation_flags)
        for row in candidate.rows
        if is_publishable_aligned_row(row)
        and id(row) not in matched_candidate_ids
    ):
        return False
    if _mapped_field_coverage(candidate) < _mapped_field_coverage(baseline):
        return False
    return reconstruction_quality(candidate) > reconstruction_quality(baseline)


def decide_recovery(
    reconstruction: ReconstructionResult,
    *,
    gemini_mode: GeminiMode,
    vlm_truncated: bool = False,
    profile_match: ProfileMatch | None = None,
) -> RouteDecision:
    reasons: list[RecoveryReason] = []
    if not is_terminal_non_ledger(reconstruction):
        if not reconstruction.rows:
            reasons.append(RecoveryReason.ZERO_YIELD)
        elif is_implausibly_low_yield(reconstruction):
            reasons.append(RecoveryReason.LOW_YIELD)
        if needs_field_quality_recovery(reconstruction):
            reasons.append(RecoveryReason.VALIDATION_FAILURE)
        if reconstruction.schema is None:
            reasons.append(RecoveryReason.NO_SCHEMA)
        if reconstruction.diagnostics.get("table_type") == TableType.UNKNOWN.value:
            reasons.append(RecoveryReason.UNKNOWN_TABLE)
        if vlm_truncated:
            reasons.append(RecoveryReason.VLM_TRUNCATED)
        if profile_match is not None and profile_match.profile_key and not profile_match.selected:
            reasons.append(RecoveryReason.PROFILE_AMBIGUOUS)
    stages: list[RecoveryStage] = []
    if reasons:
        stages.extend(
            (
                RecoveryStage.HIGH_RESOLUTION,
                RecoveryStage.PHOTOMETRIC,
                RecoveryStage.CROP_OCR,
                RecoveryStage.LOCAL_VLM,
            )
        )
        if gemini_mode is not GeminiMode.OFF:
            stages.append(RecoveryStage.GEMINI)
        stages.append(RecoveryStage.REVIEW)
    route = "profile_fast" if profile_match and profile_match.selected else "local_ocr"
    if reasons and (
        route != "profile_fast" or RecoveryReason.VALIDATION_FAILURE in reasons
    ):
        route = "local_recovery"
    return RouteDecision(
        route=route,
        reasons=tuple(dict.fromkeys(reasons)),
        planned_stages=tuple(stages),
        profile_key=profile_match.profile_key if profile_match else None,
        profile_version=profile_match.profile_version if profile_match else None,
        profile_score=profile_match.score if profile_match else None,
        profile_margin=profile_match.margin if profile_match else None,
    )


def map_crop_tokens_to_page(
    tokens: tuple[OcrToken, ...],
    crop_box: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
    page_artifact_sha256: str,
) -> tuple[OcrToken, ...]:
    left, top, right, bottom = crop_box
    x_scale = (right - left) / max(1, crop_width)
    y_scale = (bottom - top) / max(1, crop_height)
    return tuple(
        token.model_copy(
            update={
                "polygon": Polygon(
                    points=tuple(
                        Point(x=left + point.x * x_scale, y=top + point.y * y_scale)
                        for point in token.polygon.points
                    )
                ),
                "artifact_sha256": page_artifact_sha256,
                "token_id": f"recovery:{token.token_id}",
            }
        )
        for token in tokens
    )


def _evidence_bounds(
    evidence: tuple[object, ...],
) -> tuple[float, float, float, float] | None:
    polygons = tuple(
        item.polygon
        for item in evidence
        if getattr(item, "token_ids", ()) and getattr(item, "polygon", None) is not None
    )
    if not polygons:
        return None
    bounds = tuple(_polygon_bounds(polygon) for polygon in polygons)
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def _row_evidence_bounds(row: object) -> tuple[float, float, float, float] | None:
    bounds = tuple(
        bound
        for cell in getattr(row, "cells", ())
        if "all_text_rotated"
        not in getattr(cell, "validation_flags", ())
        if (bound := _evidence_bounds(getattr(cell, "evidence", ()))) is not None
    )
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def return_sign_recovery_targets(
    reconstruction: ReconstructionResult,
    *,
    table_box: tuple[int, int, int, int],
) -> tuple[ReturnSignRecoveryTarget, ...]:
    """Locate grounded amount cells whose minus sign needs a focused OCR retry."""
    table_left, table_top, table_right, table_bottom = table_box
    targets: list[ReturnSignRecoveryTarget] = []
    for aligned_row in reconstruction.rows:
        if (
            "positive_amount_in_return_section"
            not in aligned_row.candidate.validation_flags
            or aligned_row.candidate.amount is None
            or aligned_row.candidate.amount <= 0
        ):
            continue
        amount_token_ids = {
            token_id.strip()
            for token_id in aligned_row.field_token_ids.get("amount", ())
            if token_id.strip()
        }
        if not amount_token_ids:
            continue
        amount_cell = next(
            (
                cell
                for table in reconstruction.source_tables
                for row in table.rows
                for cell in row.cells
                if parse_decimal(getattr(cell, "raw_value", None))
                == aligned_row.candidate.amount
                and any(
                    amount_token_ids.intersection(
                        token_id.strip()
                        for token_id in getattr(evidence, "token_ids", ())
                        if token_id.strip()
                    )
                    for evidence in getattr(cell, "evidence", ())
                )
            ),
            None,
        )
        if amount_cell is None:
            continue
        bounds = _evidence_bounds(amount_cell.evidence)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        height = max(1.0, bottom - top)
        region = (
            max(table_left, round(left - 2 * height)),
            max(table_top, round(top - height / 2)),
            min(table_right, round(right + height / 2)),
            min(table_bottom, round(bottom + height / 2)),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            continue
        target = ReturnSignRecoveryTarget(
            region=region,
            expected_absolute_amount=aligned_row.candidate.amount,
        )
        if target not in targets:
            targets.append(target)
    return tuple(targets)


def _cell_by_canonical_field(
    table: object,
    row: object,
    canonical_field: str,
) -> object | None:
    column = next(
        (
            item
            for item in getattr(table, "columns", ())
            if getattr(item, "canonical_field", None) == canonical_field
        ),
        None,
    )
    if column is None:
        return None
    return next(
        (
            cell
            for cell in getattr(row, "cells", ())
            if getattr(cell, "column_id", None) == column.id
        ),
        None,
    )


def _has_missing_grounded_detail_description(table: object, row: object) -> bool:
    description = _cell_by_canonical_field(table, row, "description")
    unit_price = _cell_by_canonical_field(table, row, "unit_price")
    financial = next(
        (
            cell
            for field in ("net_amount", "gross_amount")
            if (cell := _cell_by_canonical_field(table, row, field)) is not None
            and parse_decimal(getattr(cell, "raw_value", None)) is not None
        ),
        None,
    )
    return (
        description is not None
        and not str(getattr(description, "raw_value", "") or "").strip()
        and unit_price is not None
        and parse_decimal(getattr(unit_price, "raw_value", None)) is not None
        and financial is not None
    )


def description_lane_recovery_regions(
    reconstruction: ReconstructionResult,
    *,
    table_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Find grounded detail rows whose description lane needs targeted OCR."""
    table_left, table_top, table_right, table_bottom = table_box
    regions: list[tuple[int, int, int, int]] = []
    for table in reconstruction.source_tables:
        columns = table.columns
        description_index = next(
            (
                index
                for index, column in enumerate(columns)
                if column.canonical_field == "description"
            ),
            None,
        )
        if description_index is None or description_index + 1 >= len(columns):
            continue
        description_bounds = _evidence_bounds(columns[description_index].evidence)
        next_bounds = _evidence_bounds(columns[description_index + 1].evidence)
        if description_bounds is None or next_bounds is None:
            continue
        grounded_header_bounds = tuple(
            bounds
            for column in columns
            if (bounds := _evidence_bounds(column.evidence)) is not None
        )
        if not grounded_header_bounds:
            continue
        data_top = max(
            table_top,
            ceil(max(bounds[3] for bounds in grounded_header_bounds)),
        )
        left = max(table_left, round(description_bounds[0]))
        right = min(table_right, round(next_bounds[0]))
        if right <= left:
            continue

        row_bounds = tuple(_row_evidence_bounds(row) for row in table.rows)
        target_indexes = tuple(
            index
            for index, row in enumerate(table.rows)
            if row_bounds[index] is not None
            and _has_missing_grounded_detail_description(table, row)
        )
        target_groups: list[list[int]] = []
        for target_index in target_indexes:
            if (
                not target_groups
                or target_index - target_groups[-1][-1] > 2
            ):
                target_groups.append([target_index])
            else:
                target_groups[-1].append(target_index)
        for indexes in target_groups:
            first_index, last_index = indexes[0], indexes[-1]
            first_bounds = row_bounds[first_index]
            last_bounds = row_bounds[last_index]
            if first_bounds is None or last_bounds is None:
                continue
            previous_bounds = next(
                (
                    row_bounds[index]
                    for index in range(first_index - 1, -1, -1)
                    if row_bounds[index] is not None
                ),
                None,
            )
            next_row_bounds = next(
                (
                    row_bounds[index]
                    for index in range(last_index + 1, len(row_bounds))
                    if row_bounds[index] is not None
                ),
                None,
            )
            first_height = max(1.0, first_bounds[3] - first_bounds[1])
            last_height = max(1.0, last_bounds[3] - last_bounds[1])
            top = (
                round((previous_bounds[3] + first_bounds[1]) / 2)
                if previous_bounds is not None
                else round(first_bounds[1] - first_height / 2)
            )
            bottom = (
                round((last_bounds[3] + next_row_bounds[1]) / 2)
                if next_row_bounds is not None
                else round(last_bounds[3] + last_height / 2)
            )
            top = max(data_top, top)
            bottom = min(table_bottom, bottom)
            if bottom > top:
                regions.append((left, top, right, bottom))
    return tuple(regions)


def map_page_box_to_crop_pixels(
    page_box: tuple[int, int, int, int],
    *,
    parent_page_box: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int]:
    parent_left, parent_top, parent_right, parent_bottom = parent_page_box
    left, top, right, bottom = page_box
    parent_width = max(1, parent_right - parent_left)
    parent_height = max(1, parent_bottom - parent_top)
    return (
        max(0, min(crop_width, round((left - parent_left) * crop_width / parent_width))),
        max(0, min(crop_height, round((top - parent_top) * crop_height / parent_height))),
        max(0, min(crop_width, round((right - parent_left) * crop_width / parent_width))),
        max(
            0,
            min(crop_height, round((bottom - parent_top) * crop_height / parent_height)),
        ),
    )


def replace_tokens_in_regions(
    original: tuple[OcrToken, ...],
    replacements: tuple[OcrToken, ...],
    *,
    regions: tuple[tuple[int, int, int, int], ...],
) -> tuple[OcrToken, ...]:
    def inside_region(token: OcrToken) -> bool:
        left, top, right, bottom = _bounds(token)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return any(
            region_left <= center_x <= region_right
            and region_top <= center_y <= region_bottom
            for region_left, region_top, region_right, region_bottom in regions
        )

    return (
        *(token for token in original if not inside_region(token)),
        *replacements,
    )


def merge_recovery_tokens(
    baseline: tuple[OcrToken, ...],
    recovered: tuple[OcrToken, ...],
    targeted: tuple[OcrToken, ...],
    *,
    regions: tuple[tuple[int, int, int, int], ...],
) -> tuple[OcrToken, ...]:
    """Preserve page OCR, supplement missing geometry, then replace target lanes."""

    def inside_region(token: OcrToken) -> bool:
        left, top, right, bottom = _bounds(token)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return any(
            region_left <= center_x <= region_right
            and region_top <= center_y <= region_bottom
            for region_left, region_top, region_right, region_bottom in regions
        )

    preserved = tuple(token for token in baseline if not inside_region(token))

    def substantially_overlaps_preserved(token: OcrToken) -> bool:
        left, top, right, bottom = _bounds(token)
        area = max(1.0, (right - left) * (bottom - top))
        for existing in preserved:
            other_left, other_top, other_right, other_bottom = _bounds(existing)
            intersection = max(
                0.0,
                min(right, other_right) - max(left, other_left),
            ) * max(
                0.0,
                min(bottom, other_bottom) - max(top, other_top),
            )
            other_area = max(
                1.0,
                (other_right - other_left) * (other_bottom - other_top),
            )
            if intersection / min(area, other_area) >= 0.5:
                return True
        return False

    supplements = tuple(
        token
        for token in recovered
        if not inside_region(token)
        and not substantially_overlaps_preserved(token)
    )
    return (*preserved, *supplements, *targeted)


def _bounds(token: OcrToken) -> tuple[float, float, float, float]:
    xs = [point.x for point in token.polygon.points]
    ys = [point.y for point in token.polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [point.x for point in polygon.points]
    ys = [point.y for point in polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def _overlaps(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(
        left[3], right[3]
    ) > max(left[1], right[1])


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _description_supported(value: str, tokens: list[OcrToken]) -> bool:
    evidence_text = " ".join(token.text for token in tokens if token.text != "[REDACTED]")
    if not evidence_text:
        return False
    return SequenceMatcher(None, _normalize(value), _normalize(evidence_text)).ratio() >= 0.25


def _numeric_supported(value: str, tokens: list[OcrToken]) -> bool:
    parsed = parse_decimal(value)
    if parsed is None:
        return False
    return any(parsed in _numeric_values(token.text) for token in tokens)


def _numeric_values(text: str) -> tuple[Decimal, ...]:
    output: list[Decimal] = []
    whole = parse_decimal(text)
    if whole is not None:
        output.append(whole)
    for part in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        value = parse_decimal(part)
        if value is not None:
            output.append(value)
    return tuple(dict.fromkeys(output))


def ground_adjudication(
    response: AdjudicationResponse,
    *,
    validation_tokens: tuple[OcrToken, ...],
    evidence_tokens: tuple[OcrToken, ...],
    crop_width: int,
    crop_height: int,
    table_type: TableType,
    column_centers: dict[str, float] | None = None,
    column_tolerance: float = 0.1,
) -> GroundingResult:
    validation_by_id = {token.token_id: token for token in validation_tokens}
    evidence_by_id = {token.token_id: token for token in evidence_tokens}
    output: list[AlignedLedgerRow] = []
    rejected: list[str] = list(response.rejected_reasons)
    numeric_fields = {"amount", "quantity", "rate", "gross_amount", "discount"}
    optional_text_fields = {
        "service_date",
        "request_no",
        "service_code",
        "hsn_code",
        "section",
    }
    allowed_fields = {"description", *numeric_fields, *optional_text_fields}
    for row in response.rows:
        fields: dict[str, str] = {}
        field_token_ids: dict[str, tuple[str, ...]] = {}
        valid = True
        for field in row.fields:
            if field.name not in allowed_fields:
                rejected.append(f"row_{row.row_order}:unknown_field:{field.name}")
                valid = False
                continue
            if any(
                token_id not in validation_by_id or token_id not in evidence_by_id
                for token_id in field.token_ids
            ):
                rejected.append(f"row_{row.row_order}:unknown_token:{field.name}")
                valid = False
                continue
            bounds = _polygon_bounds(field.polygon)
            if (
                bounds[0] < 0
                or bounds[1] < 0
                or bounds[2] > crop_width
                or bounds[3] > crop_height
            ):
                rejected.append(f"row_{row.row_order}:out_of_crop:{field.name}")
                valid = False
                continue
            referenced = [validation_by_id[token_id] for token_id in field.token_ids]
            if not any(_overlaps(bounds, _bounds(token)) for token in referenced):
                rejected.append(f"row_{row.row_order}:polygon_token_mismatch:{field.name}")
                valid = False
                continue
            expected_center = (column_centers or {}).get(field.name)
            if expected_center is not None:
                actual_center = ((bounds[0] + bounds[2]) / 2) / max(1, crop_width)
                if abs(actual_center - expected_center) > column_tolerance:
                    rejected.append(f"row_{row.row_order}:column_mismatch:{field.name}")
                    valid = False
                    continue
            supported = (
                _numeric_supported(field.raw_value, referenced)
                if field.name in numeric_fields
                else _description_supported(field.raw_value, referenced)
            )
            if not supported:
                rejected.append(f"row_{row.row_order}:unsupported_value:{field.name}")
                valid = False
                continue
            fields[field.name] = field.raw_value
            field_token_ids[field.name] = field.token_ids
        if not valid or "description" not in fields or "amount" not in fields:
            continue
        amount = parse_decimal(fields["amount"])
        if amount is None:
            continue
        evidence_ids = tuple(
            dict.fromkeys(
                token_id
                for token_ids in field_token_ids.values()
                for token_id in token_ids
            )
        )
        page_boxes = [_bounds(evidence_by_id[token_id]) for token_id in evidence_ids]
        evidence_box = (
            min(item[0] for item in page_boxes),
            min(item[1] for item in page_boxes),
            max(item[2] for item in page_boxes),
            max(item[3] for item in page_boxes),
        )
        role = RowRole.REFUND if amount < 0 else row.role
        if role not in {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}:
            rejected.append(f"row_{row.row_order}:unsupported_role:{role}")
            continue
        candidate = CandidateLedgerRow(
            source_row=row.row_order,
            role=role,
            cells=tuple(field.raw_value for field in row.fields),
            section=fields.get("section"),
            description=fields["description"],
            service_date=fields.get("service_date"),
            request_no=fields.get("request_no"),
            service_code=fields.get("service_code"),
            hsn_code=fields.get("hsn_code"),
            quantity=parse_decimal(fields.get("quantity")),
            rate=parse_decimal(fields.get("rate")),
            gross_amount=parse_decimal(fields.get("gross_amount")),
            discount=parse_decimal(fields.get("discount")),
            amount=amount,
            table_type=table_type,
            source_route="gemini_grounded",
        )
        output.append(
            AlignedLedgerRow(
                candidate=candidate,
                field_token_ids=field_token_ids,
                evidence_token_ids=evidence_ids,
                evidence_box=evidence_box,
                grounding_ratio=1.0,
                source_routes=("gemini_grounded",),
            )
        )
    return GroundingResult(tuple(output), tuple(dict.fromkeys(rejected)))
