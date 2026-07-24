from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Annotated, Any
from uuid import uuid4

import cv2
import typer

from gmoney.contracts.evidence import OcrToken
from gmoney.contracts.extraction import (
    CanonicalRow,
    DocumentTotal,
    EvidenceRef,
    PageType,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    TableType,
)
from gmoney.contracts.phase3 import (
    AdjudicationRequest,
    GeminiMode,
    LayoutObservation,
    ProfileLifecycle,
    ProfileMatch,
    RecoveryAttempt,
    RecoveryReason,
    RecoveryStage,
)
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
    DocumentTotalCandidate,
    extract_document_total_candidates,
    select_document_total,
    select_document_totals,
)
from gmoney.extraction.hospital import detect_hospital
from gmoney.extraction.ocr_rows import (
    DATE_PREFIX,
    ReconstructionResult,
    TableSchemaState,
    _clean_description,
    _request_prefix_match,
    _structured_field_value_is_valid,
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
    row_category,
    tokens_in_box,
)
from gmoney.extraction.ocr_tokens import paddle_ocr_tokens
from gmoney.extraction.otsl import parse_otsl, split_otsl_tables
from gmoney.extraction.recovery import (
    decide_recovery,
    description_lane_recovery_regions,
    ground_adjudication,
    is_implausibly_low_yield,
    is_terminal_non_ledger,
    map_crop_tokens_to_page,
    map_page_box_to_crop_pixels,
    merge_recovery_tokens,
    needs_field_quality_recovery,
    reconstruction_quality,
    replace_tokens_in_regions,
    return_sign_recovery_targets,
    safely_improves_reconstruction,
)
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.extraction.spatial import align_candidate_rows
from gmoney.extraction.typed_values import parse_decimal, parse_service_date
from gmoney.geometry.crop import (
    clahe_variant,
    color_overlay_suppressed_variant,
    crop_region,
    render_pdf_region,
)
from gmoney.geometry.render import render_pdf
from gmoney.inference.contracts import InferenceRequest, InferenceResponse
from gmoney.inference.gemini import (
    AdjudicationAdapter,
    GeminiAdjudicationAdapter,
    cached_adjudication,
    validate_promotion,
)
from gmoney.inference.ocr_table_fallback import propose_tables_from_ocr
from gmoney.inference.paddle import (
    PaddleDocLayoutV3Adapter,
    PaddleOcrV6Adapter,
    PaddleOcrVlAdapter,
)
from gmoney.inference.redaction import redact_crop
from gmoney.profiles.lifecycle import deterministic_shadow_sample
from gmoney.profiles.matching import match_profile, profile_to_schema
from gmoney.profiles.repository import JsonProfileRepository
from gmoney.settings import Settings, get_settings

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class TableWork:
    table_id: str
    page_number: int
    page_artifact_sha256: str
    crop_path: Path
    crop_sha256: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VlAsset:
    path: Path
    artifact_sha256: str
    identity: str


def _should_attempt_crop_recovery(
    reconstruction: ReconstructionResult,
    *,
    parsed_rows: Collection[object],
    table_box: tuple[int, int, int, int],
) -> bool:
    return not is_terminal_non_ledger(reconstruction) and (
        not parsed_rows
        or is_implausibly_low_yield(reconstruction)
        or needs_field_quality_recovery(reconstruction)
        or bool(
            description_lane_recovery_regions(
                reconstruction,
                table_box=table_box,
            )
        )
    )


def _page_type(table_type: TableType) -> PageType:
    return {
        TableType.PHARMACY: PageType.PHARMACY,
        TableType.LABORATORY: PageType.LABORATORY,
        TableType.CATEGORY_SUMMARY: PageType.CATEGORY_SUMMARY,
        TableType.PACKAGE_SUMMARY: PageType.CATEGORY_SUMMARY,
        TableType.PAYMENT: PageType.RECEIPT_PAYMENT,
        TableType.METADATA: PageType.METADATA,
        TableType.ITEM_LEDGER: PageType.ITEMIZED_CHARGES,
    }.get(table_type, PageType.MIXED)


def _cached_prediction(
    cache_path: Path,
    request: InferenceRequest,
    adapter: Any,
) -> tuple[InferenceResponse, bool]:
    """Reuse only an inference result tied to the same immutable input and model."""
    if cache_path.exists():
        envelope = json.loads(cache_path.read_text())
        if (
            envelope.get("artifact_sha256") == request.artifact_sha256
            and envelope.get("options") == request.options
            and envelope.get("model_spec") == adapter.spec.model_dump(mode="json")
        ):
            return InferenceResponse.model_validate(envelope["response"]), True

    response = adapter.predict(request)
    envelope = {
        "artifact_sha256": request.artifact_sha256,
        "options": request.options,
        "model_spec": adapter.spec.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(cache_path)
    return response, False


def _layout_boxes(output: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    pages = output.get("pages") or []
    if not pages:
        return []
    boxes = pages[0].get("res", {}).get("boxes") or []
    return [
        tuple(round(value) for value in box["coordinate"])
        for box in boxes
        if box.get("label") == "table" and float(box.get("score") or 0) >= 0.3
    ]


def _safe_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    horizontal_padding: int = 20,
    vertical_padding: int = 100,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, left - horizontal_padding),
        max(0, top - vertical_padding),
        min(width, right + horizontal_padding),
        min(height, bottom + vertical_padding),
    )


def _merge_table_boxes(
    layout_boxes: list[tuple[int, int, int, int]],
    geometry_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Fuse layout proposals with OCR row geometry, expanding partial tables."""
    merged = list(layout_boxes)
    for proposal in geometry_boxes:
        p_left, p_top, p_right, p_bottom = proposal
        p_area = (p_right - p_left) * (p_bottom - p_top)
        match_index: int | None = None
        for index, candidate in enumerate(merged):
            left, top, right, bottom = candidate
            intersection = max(0, min(right, p_right) - max(left, p_left)) * max(
                0, min(bottom, p_bottom) - max(top, p_top)
            )
            candidate_area = (right - left) * (bottom - top)
            if intersection / max(1, min(candidate_area, p_area)) >= 0.5:
                match_index = index
                break
        if match_index is None:
            merged.append(proposal)
            continue
        left, top, right, bottom = merged[match_index]
        merged[match_index] = (
            min(left, p_left),
            min(top, p_top),
            max(right, p_right),
            max(bottom, p_bottom),
        )
    return sorted(merged, key=lambda box: (box[1], box[0]))


def _write_vl_image(image, output: Path) -> VlAsset:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write VLM image: {temporary}")
    temporary.replace(output)
    return VlAsset(output, sha256_file(output), output.stem)


def _vl_asset(work: TableWork, artifact_root: Path, orientation: str) -> VlAsset:
    if orientation == "upright":
        return VlAsset(work.crop_path, work.crop_sha256, work.table_id)
    image = cv2.imread(str(work.crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read table crop: {work.crop_path}")
    rotation = (
        cv2.ROTATE_90_CLOCKWISE if orientation == "clockwise_90" else cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    rotated = cv2.rotate(image, rotation)
    return _write_vl_image(
        rotated,
        artifact_root / "crops" / f"{work.table_id}-oriented.png",
    )


def _vertical_vl_tiles(
    asset: VlAsset,
    artifact_root: Path,
    table_id: str,
    *,
    tile_height: int = 1600,
    overlap: int = 160,
) -> tuple[VlAsset, ...]:
    image = cv2.imread(str(asset.path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read VLM image: {asset.path}")
    height = image.shape[0]
    if height <= tile_height:
        return ()
    output: list[VlAsset] = []
    top = 0
    index = 1
    while top < height:
        bottom = min(height, top + tile_height)
        output.append(
            _write_vl_image(
                image[top:bottom],
                artifact_root / "crops" / f"{table_id}-tile-{index}.png",
            )
        )
        if bottom == height:
            break
        top = bottom - overlap
        index += 1
    return tuple(output)


def _deduplicate(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    selected: dict[tuple[object, ...], CanonicalRow] = {}
    for row in rows:
        description = re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip()
        description_evidence_ids = tuple(
            sorted(
                token_id
                for evidence in row.field_evidence.get("description", ())
                for token_id in evidence.token_ids
            )
        )
        key = (
            row.page_number,
            ("description_evidence", description_evidence_ids)
            if description_evidence_ids
            else ("fallback", row.net_amount, description),
        )
        current = selected.get(key)
        score = (
            not row.validation_flags,
            "ocr_spatial_graph" in row.source_routes,
            len(row.field_evidence),
            len(description),
        )
        current_score = (
            bool(current and not current.validation_flags),
            bool(current and "ocr_spatial_graph" in current.source_routes),
            len(current.field_evidence) if current else -1,
            len(current.description or "") if current else -1,
        )
        if current is None or score > current_score:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row.page_number, row.row_order))


@dataclass(frozen=True)
class _PrintedTableFeatures:
    descriptions: frozenset[str]
    serials: frozenset[int]
    numeric_values: tuple[Decimal, ...]
    complete_financial_rows: int
    populated_canonical_fields: int


def _printed_table_features(table: SourceTable) -> _PrintedTableFeatures:
    description_columns = tuple(
        column
        for column in table.columns
        if column.canonical_field == "description"
    )
    financial_columns = tuple(
        column
        for column in table.columns
        if column.canonical_field in {"gross_amount", "net_amount"}
    )
    numeric_columns = tuple(
        column
        for column in table.columns
        if column.canonical_field
        in {
            "quantity",
            "unit_price",
            "gross_amount",
            "discount",
            "net_amount",
        }
    )
    serial_columns = tuple(
        column
        for column in table.columns
        if column.canonical_field is None
        and (
            "serial"
            in re.sub(
                r"[^a-z0-9]+", " ", column.label.casefold()
            ).split()
            or re.sub(
                r"[^a-z0-9]+", "", column.label.casefold()
            ).startswith("sr")
        )
    )
    descriptions: set[str] = set()
    serials: set[int] = set()
    numeric_values: list[Decimal] = []
    complete_financial_rows = 0
    populated_fields: set[str] = set()
    for row in table.rows:
        cells = {cell.column_id: cell for cell in row.cells}
        description = re.sub(
            r"[^a-z0-9]+",
            " ",
            " ".join(
                cells[column.id].raw_value or ""
                for column in description_columns
            ).casefold(),
        ).strip()
        if description:
            descriptions.add(description)
        row_has_financial_value = False
        for column in numeric_columns:
            raw_value = cells[column.id].raw_value
            if raw_value and (parsed := parse_decimal(raw_value)) is not None:
                numeric_values.append(parsed)
        for column in financial_columns:
            raw_value = cells[column.id].raw_value
            if raw_value and parse_decimal(raw_value) is not None:
                row_has_financial_value = True
                if column.canonical_field:
                    populated_fields.add(column.canonical_field)
        if description and row_has_financial_value:
            complete_financial_rows += 1
        for column in serial_columns:
            raw_value = (cells[column.id].raw_value or "").strip()
            if re.fullmatch(r"\d{1,4}\.?", raw_value):
                serials.add(int(raw_value.rstrip(".")))
    return _PrintedTableFeatures(
        descriptions=frozenset(descriptions),
        serials=frozenset(serials),
        numeric_values=tuple(sorted(numeric_values)),
        complete_financial_rows=complete_financial_rows,
        populated_canonical_fields=len(populated_fields),
    )


def _merged_printed_table_features(
    tables: tuple[SourceTable, ...],
) -> _PrintedTableFeatures:
    features = tuple(_printed_table_features(table) for table in tables)
    return _PrintedTableFeatures(
        descriptions=frozenset(
            description
            for feature in features
            for description in feature.descriptions
        ),
        serials=frozenset(
            serial for feature in features for serial in feature.serials
        ),
        numeric_values=tuple(
            sorted(
                value
                for feature in features
                for value in feature.numeric_values
            )
        ),
        complete_financial_rows=sum(
            feature.complete_financial_rows for feature in features
        ),
        populated_canonical_fields=max(
            (
                feature.populated_canonical_fields
                for feature in features
            ),
            default=0,
        ),
    )


def _evidence_region_position(
    evidence: Any,
    region: tuple[float, float, float, float],
) -> str:
    points = evidence.polygon.points
    if not points:
        return "boundary"
    left, top, right, bottom = region
    inside = tuple(
        left <= point.x <= right and top <= point.y <= bottom
        for point in points
    )
    if all(inside):
        return "inside"
    polygon_left = min(point.x for point in points)
    polygon_top = min(point.y for point in points)
    polygon_right = max(point.x for point in points)
    polygon_bottom = max(point.y for point in points)
    if (
        polygon_right < left
        or polygon_left > right
        or polygon_bottom < top
        or polygon_top > bottom
    ):
        return "outside"
    return "boundary"


def _exact_embedded_crop_region(
    *,
    candidate_path: Path,
    candidate_box: tuple[int, int, int, int],
    representative_path: Path,
) -> tuple[float, float, float, float] | None:
    """Return the candidate-page region occupied by an exact repeated crop."""
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
    representative = cv2.imread(
        str(representative_path),
        cv2.IMREAD_GRAYSCALE,
    )
    if candidate is None or representative is None:
        return None
    if min(candidate.shape) < 16 or min(representative.shape) < 16:
        return None
    if float(candidate.std()) < 5 or float(representative.std()) < 5:
        return None

    def exact_location(
        image: Any,
        template: Any,
    ) -> tuple[int, int] | None:
        if (
            template.shape[0] > image.shape[0]
            or template.shape[1] > image.shape[1]
        ):
            return None
        match = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
        _, _, location, _ = cv2.minMaxLoc(match)
        left, top = location
        patch = image[
            top : top + template.shape[0],
            left : left + template.shape[1],
        ]
        if cv2.countNonZero(cv2.absdiff(patch, template)) != 0:
            return None
        return left, top

    representative_location = exact_location(candidate, representative)
    if representative_location is not None:
        crop_left, crop_top, crop_right, crop_bottom = candidate_box
        scale_x = (crop_right - crop_left) / candidate.shape[1]
        scale_y = (crop_bottom - crop_top) / candidate.shape[0]
        match_left, match_top = representative_location
        return (
            crop_left + match_left * scale_x,
            crop_top + match_top * scale_y,
            crop_left
            + (match_left + representative.shape[1]) * scale_x,
            crop_top
            + (match_top + representative.shape[0]) * scale_y,
        )

    if exact_location(representative, candidate) is not None:
        return tuple(float(value) for value in candidate_box)
    return None


def _residual_tables_after_exact_repeat(
    *,
    key: tuple[int, str],
    tables: tuple[SourceTable, ...],
    rows: list[CanonicalRow],
    region: tuple[float, float, float, float],
) -> tuple[SourceTable, ...] | None:
    canonical_rows = tuple(
        row
        for row in rows
        if (row.page_number, row.table_id) == key
    )
    if not canonical_rows or any(
        not row.evidence
        or not all(
            _evidence_region_position(evidence, region) == "inside"
            for evidence in row.evidence
        )
        for row in canonical_rows
    ):
        return None

    residual_tables: list[SourceTable] = []
    for table in tables:
        residual_rows: list[Any] = []
        for source_row in table.rows:
            residual_cells: list[SourceCell] = []
            has_outside_value = False
            for cell in source_row.cells:
                if not cell.raw_value or not cell.raw_value.strip():
                    residual_cells.append(cell)
                    continue
                if not cell.evidence:
                    return None
                positions = tuple(
                    _evidence_region_position(evidence, region)
                    for evidence in cell.evidence
                )
                if all(position == "inside" for position in positions):
                    residual_cells.append(
                        cell.model_copy(
                            update={
                                "raw_value": None,
                                "evidence": (),
                                "validation_flags": tuple(
                                    dict.fromkeys(
                                        (
                                            *cell.validation_flags,
                                            "suppressed_exact_repeated_crop",
                                        )
                                    )
                                ),
                            }
                        )
                    )
                else:
                    has_outside_value = True
                    residual_cells.append(cell)
            if has_outside_value:
                residual_rows.append(
                    source_row.model_copy(
                        update={
                            "order": len(residual_rows),
                            "canonical_row_id": None,
                            "cells": tuple(residual_cells),
                            "validation_flags": tuple(
                                dict.fromkeys(
                                    (
                                        *source_row.validation_flags,
                                        "repeated_crop_residual",
                                    )
                                )
                            ),
                        }
                    )
                )
        if residual_rows:
            residual_tables.append(
                table.model_copy(
                    update={
                        "rows": tuple(residual_rows),
                        "validation_flags": tuple(
                            dict.fromkeys(
                                (
                                    *table.validation_flags,
                                    "repeated_crop_residual",
                                )
                            )
                        ),
                    }
                )
            )
    return tuple(residual_tables)


def _suppress_repeated_printed_tables(
    tables: list[SourceTable],
    rows: list[CanonicalRow],
    *,
    crop_paths: dict[tuple[int, str], Path],
    crop_boxes: dict[tuple[int, str], tuple[int, int, int, int]],
) -> tuple[
    list[SourceTable],
    list[CanonicalRow],
    tuple[tuple[int, str], ...],
]:
    """Keep one grounded rendition when a print overlay repeats across pages.

    Browser-generated PDFs can paginate a fixed bill preview several times.
    OCR overlap is only a candidate search. Suppression additionally requires
    exact pixel containment of one physical table crop and proof that every
    billable row from the candidate lies inside that repeated crop. This keeps
    near-duplicate continuation pages and sibling source-table segments.
    """
    grouped: dict[tuple[int, str], list[SourceTable]] = {}
    for table in tables:
        grouped.setdefault(
            (table.page_number, table.table_id),
            [],
        ).append(table)
    physical_keys = list(grouped)
    physical_tables = [
        tuple(grouped[key]) for key in physical_keys
    ]
    features = [
        _merged_printed_table_features(group)
        for group in physical_tables
    ]
    connected: dict[int, set[int]] = {
        index: {index} for index in range(len(physical_keys))
    }
    for left_index, left_key in enumerate(physical_keys):
        left_features = features[left_index]
        if (
            len(left_features.serials) < 8
            or len(left_features.descriptions) < 8
            or len(left_features.numeric_values) < 8
        ):
            continue
        for right_index in range(
            left_index + 1,
            len(physical_keys),
        ):
            right_key = physical_keys[right_index]
            right_features = features[right_index]
            if (
                left_key[0] == right_key[0]
                or len(right_features.serials) < 8
                or len(right_features.descriptions) < 8
                or len(right_features.numeric_values) < 8
            ):
                continue
            serial_overlap = len(
                left_features.serials & right_features.serials
            )
            description_overlap = len(
                left_features.descriptions & right_features.descriptions
            )
            numeric_overlap = sum(
                (
                    Counter(left_features.numeric_values)
                    & Counter(right_features.numeric_values)
                ).values()
            )
            if (
                serial_overlap * 5
                < min(
                    len(left_features.serials),
                    len(right_features.serials),
                )
                * 4
                or description_overlap * 4
                < min(
                    len(left_features.descriptions),
                    len(right_features.descriptions),
                )
                * 3
                or numeric_overlap * 5
                < min(
                    len(left_features.numeric_values),
                    len(right_features.numeric_values),
                )
                * 4
            ):
                continue
            connected[left_index].add(right_index)
            connected[right_index].add(left_index)

    suppressed_keys: set[tuple[int, str]] = set()
    residual_tables_by_key: dict[
        tuple[int, str],
        tuple[SourceTable, ...],
    ] = {}
    visited: set[int] = set()
    for start in range(len(physical_keys)):
        if start in visited:
            continue
        component: set[int] = set()
        pending = [start]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(connected[index] - component)
        visited.update(component)
        if len(component) < 2:
            continue

        def grounding_score(index: int) -> tuple[int, int, int, int, int]:
            table_features = features[index]
            return (
                table_features.complete_financial_rows,
                table_features.populated_canonical_fields,
                len(table_features.descriptions),
                len(table_features.serials),
                physical_keys[index][0],
            )

        keep = max(component, key=grounding_score)
        keep_key = physical_keys[keep]
        representative_path = crop_paths.get(keep_key)
        if representative_path is None:
            continue
        for index in component:
            if index == keep or index not in connected[keep]:
                continue
            candidate_key = physical_keys[index]
            candidate_features = features[index]
            if not candidate_features.serials.issubset(
                features[keep].serials
            ):
                continue
            candidate_path = crop_paths.get(candidate_key)
            candidate_box = crop_boxes.get(candidate_key)
            if candidate_path is None or candidate_box is None:
                continue
            repeated_region = _exact_embedded_crop_region(
                candidate_path=candidate_path,
                candidate_box=candidate_box,
                representative_path=representative_path,
            )
            if repeated_region is None:
                continue
            residual_tables = _residual_tables_after_exact_repeat(
                key=candidate_key,
                tables=physical_tables[index],
                rows=rows,
                region=repeated_region,
            )
            if residual_tables is None:
                continue
            suppressed_keys.add(candidate_key)
            residual_tables_by_key[candidate_key] = residual_tables

    selected_tables: list[SourceTable] = []
    emitted_residuals: set[tuple[int, str]] = set()
    for table in tables:
        key = (table.page_number, table.table_id)
        if key not in suppressed_keys:
            selected_tables.append(table)
            continue
        if key not in emitted_residuals:
            selected_tables.extend(residual_tables_by_key[key])
            emitted_residuals.add(key)
    selected_rows = [
        row
        for row in rows
        if (row.page_number, row.table_id) not in suppressed_keys
    ]
    return (
        selected_tables,
        selected_rows,
        tuple(sorted(suppressed_keys)),
    )


def _contains_service_code_fragment(value: str) -> bool:
    return any(
        _structured_field_value_is_valid("service_code", fragment)
        for fragment in re.findall(r"[A-Za-z0-9./-]{3,30}", value)
    )


def _source_cell_is_oversized_overlay(
    cell: SourceCell,
    column: SourceColumn,
    columns: tuple[SourceColumn, ...],
    row_cells: tuple[SourceCell, ...],
) -> bool:
    def evidence_heights(item: SourceCell) -> tuple[float, ...]:
        return tuple(
            max(point.y for point in evidence.polygon.points)
            - min(point.y for point in evidence.polygon.points)
            for evidence in item.evidence
        )

    own_heights = evidence_heights(cell)
    peer_heights = tuple(
        height
        for peer in row_cells
        if peer.column_id != cell.column_id
        for height in evidence_heights(peer)
        if height > 0
    )
    if not own_heights or len(peer_heights) < 2:
        return False
    peer_height = median(peer_heights)
    if peer_height <= 0 or max(own_heights) <= peer_height * 2:
        return False

    cell_xs = tuple(
        point.x for evidence in cell.evidence for point in evidence.polygon.points
    )
    column_xs = tuple(
        point.x for evidence in column.evidence for point in evidence.polygon.points
    )
    table_xs = tuple(
        point.x
        for source_column in columns
        for evidence in source_column.evidence
        for point in evidence.polygon.points
    )
    if not cell_xs or not column_xs or not table_xs:
        return False
    table_width = max(table_xs) - min(table_xs)
    cell_center = (min(cell_xs) + max(cell_xs)) / 2
    column_center = (min(column_xs) + max(column_xs)) / 2
    return (
        table_width > 0
        and abs(cell_center - column_center) > table_width * 0.08
    )


def _source_cell_is_in_rotated_overlay_cluster(
    cell: SourceCell,
    table: SourceTable,
    source_row: SourceRow,
    role: str,
) -> bool:
    row_index = next(
        (
            index
            for index, candidate in enumerate(table.rows)
            if candidate.id == source_row.id
        ),
        None,
    )
    if row_index is None:
        return False
    neighboring_rows = table.rows[
        max(0, row_index - 2) : min(len(table.rows), row_index + 3)
    ]
    rotated_cells_in_lane = tuple(
        candidate
        for neighboring_row in neighboring_rows
        for candidate in neighboring_row.cells
        if candidate.column_id == cell.column_id
        and candidate.raw_value
        and "all_text_rotated" in candidate.validation_flags
        and not _structured_field_value_is_valid(role, candidate.raw_value)
    )
    return len(rotated_cells_in_lane) >= 3


def _source_cell_is_invalid_structured_overlay(
    cell: SourceCell,
    column: SourceColumn,
    table: SourceTable,
    source_row: SourceRow,
    canonical: CanonicalRow,
) -> bool:
    roles = {
        "service_date_raw": "service_date",
        "request_no": "request_no",
        "service_code": "service_code",
        "hsn_code": "hsn_code",
    }
    field = column.canonical_field
    role = roles.get(field or "")
    if (
        field is None
        or role is None
        or getattr(canonical, field) is not None
        or not cell.raw_value
        or _structured_field_value_is_valid(role, cell.raw_value)
        or "all_text_rotated" not in cell.validation_flags
    ):
        return False
    if (
        role == "service_code"
        and _contains_service_code_fragment(cell.raw_value)
    ):
        return False
    if _structured_field_value_is_valid(
        role,
        re.sub(r"[\s:#]+", "", cell.raw_value),
    ):
        return False
    return _source_cell_is_oversized_overlay(
        cell,
        column,
        table.columns,
        source_row.cells,
    ) or _source_cell_is_in_rotated_overlay_cluster(
        cell,
        table,
        source_row,
        role,
    )


def _filter_evidence_token_ids(
    evidence_refs: tuple[EvidenceRef, ...],
    token_ids: set[str],
) -> tuple[EvidenceRef, ...]:
    return tuple(
        evidence.model_copy(
            update={
                "token_ids": tuple(
                    token_id
                    for token_id in evidence.token_ids
                    if token_id in token_ids
                )
            }
        )
        for evidence in evidence_refs
        if token_ids.intersection(evidence.token_ids)
    )


def _split_grounded_date_request_description(
    cells: tuple[SourceCell, ...],
    columns: tuple[SourceColumn, ...],
    canonical: CanonicalRow,
) -> tuple[SourceCell, ...]:
    columns_by_field = {
        column.canonical_field: column
        for column in columns
        if column.canonical_field is not None
    }
    date_column = columns_by_field.get("service_date_raw")
    request_column = columns_by_field.get("request_no")
    description_column = columns_by_field.get("description")
    if (
        date_column is None
        or request_column is None
        or description_column is None
        or not canonical.service_date_raw
        or not canonical.request_no
    ):
        return cells

    cells_by_id = {cell.column_id: cell for cell in cells}
    merged: tuple[SourceCell, re.Match[str], re.Match[str]] | None = None
    for cell in cells:
        raw_value = cell.raw_value or ""
        date_match = DATE_PREFIX.match(raw_value)
        if date_match is None:
            continue
        request_match = _request_prefix_match(
            raw_value[date_match.end() :].strip(),
            allow_compact=True,
        )
        if request_match is not None:
            merged = (cell, date_match, request_match)
            break
    if merged is None:
        return cells

    merged_cell, date_match, request_match = merged
    assert merged_cell.raw_value is not None
    printed_date = merged_cell.raw_value[: date_match.end()].strip(" -:")
    remainder = merged_cell.raw_value[date_match.end() :].strip()
    printed_request = request_match.group(0).strip()
    merged_tail = remainder[request_match.end() :].strip(" -:")
    canonical_description_prefix = re.match(
        re.escape(canonical.description),
        merged_tail,
        re.IGNORECASE,
    )
    if canonical_description_prefix is not None:
        merged_description = merged_tail[: canonical_description_prefix.end()]
        merged_residual = merged_tail[canonical_description_prefix.end() :].strip(
            " -:;/,[]"
        )
    else:
        merged_description = merged_tail
        merged_residual = ""
    existing_description = (
        ""
        if cells_by_id[description_column.id] is merged_cell
        else (cells_by_id[description_column.id].raw_value or "")
    )
    printed_description, _, _ = _clean_description(
        " ".join(
            value
            for value in (merged_description, existing_description)
            if value
        )
    )
    def normalize(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            value.casefold(),
        ).strip()

    date_matches = bool(
        (
            canonical.service_date_iso
            and parse_service_date(printed_date) == canonical.service_date_iso
        )
        or normalize(printed_date) == normalize(canonical.service_date_raw)
    )
    if (
        not date_matches
        or printed_request.casefold() != canonical.request_no.casefold()
        or (
            merged_description
            and normalize(printed_description) != normalize(canonical.description)
        )
    ):
        return cells
    if (
        cells_by_id[date_column.id] is not merged_cell
        and cells_by_id[date_column.id].raw_value
    ) or (
        cells_by_id[request_column.id] is not merged_cell
        and cells_by_id[request_column.id].raw_value
    ):
        return cells

    split_flag = "split_from_merged_ocr_token"
    merged_token_ids = {
        token_id
        for evidence in merged_cell.evidence
        for token_id in evidence.token_ids
    }

    def grounded_evidence(field: str, source: tuple[SourceCell, ...]) -> tuple:
        field_ids = {
            token_id
            for evidence in canonical.field_evidence.get(field, ())
            for token_id in evidence.token_ids
        }
        source_evidence = _filter_evidence_token_ids(
            tuple(
            evidence
            for cell in source
            for evidence in cell.evidence
            if field_ids.intersection(evidence.token_ids)
            ),
            field_ids,
        )
        source_ids = {
            token_id
            for evidence in source_evidence
            for token_id in evidence.token_ids
        }
        return source_evidence if field_ids and field_ids.issubset(source_ids) else ()

    date_evidence = grounded_evidence("service_date", (merged_cell,))
    request_evidence = grounded_evidence("request_no", (merged_cell,))
    description_sources = (
        (
            (merged_cell,)
            if cells_by_id[description_column.id] is merged_cell
            else (merged_cell, cells_by_id[description_column.id])
        )
        if merged_description
        else (cells_by_id[description_column.id],)
    )
    description_evidence = (
        grounded_evidence("description", description_sources)
        if merged_description
        else ()
    )
    if (
        not merged_token_ids
        or not date_evidence
        or not request_evidence
        or (merged_description and not description_evidence)
    ):
        return cells

    def split_cell(
        cell: SourceCell,
        raw_value: str,
        evidence: tuple,
    ) -> SourceCell:
        return cell.model_copy(
            update={
                "raw_value": raw_value,
                "evidence": evidence,
                "validation_flags": tuple(
                    dict.fromkeys(
                        (
                            *(
                                flag
                                for flag in cell.validation_flags
                                if flag != "empty_cell"
                            ),
                            split_flag,
                        )
                    )
                ),
            }
        )

    structured_column_ids = {
        date_column.id,
        request_column.id,
        description_column.id,
    }
    if merged_cell.column_id not in structured_column_ids:
        consumed_ids = {
            token_id
            for field in ("service_date", "request_no", "description")
            for evidence in canonical.field_evidence.get(field, ())
            for token_id in evidence.token_ids
        }
        residual_ids = merged_token_ids - consumed_ids
        residual_evidence = _filter_evidence_token_ids(
            merged_cell.evidence,
            residual_ids,
        )
        if merged_residual and not residual_evidence:
            return cells
        cells_by_id[merged_cell.column_id] = merged_cell.model_copy(
            update={
                "raw_value": merged_residual or None,
                "evidence": residual_evidence,
                "validation_flags": tuple(
                    dict.fromkeys(
                        (
                            *(
                                flag
                                for flag in merged_cell.validation_flags
                                if flag != "empty_cell"
                            ),
                            "redistributed_merged_ocr_token",
                        )
                    )
                ),
            }
        )
    cells_by_id[date_column.id] = split_cell(
        cells_by_id[date_column.id],
        printed_date,
        date_evidence,
    )
    cells_by_id[request_column.id] = split_cell(
        cells_by_id[request_column.id],
        printed_request,
        request_evidence,
    )
    if merged_description:
        cells_by_id[description_column.id] = split_cell(
            cells_by_id[description_column.id],
            printed_description,
            description_evidence,
        )
    return tuple(cells_by_id[cell.column_id] for cell in cells)


def _redistribute_grounded_description_from_adjacent_cell(
    cells: tuple[SourceCell, ...],
    columns: tuple[SourceColumn, ...],
    canonical: CanonicalRow,
) -> tuple[SourceCell, ...]:
    description_column = next(
        (
            column
            for column in columns
            if column.canonical_field == "description"
        ),
        None,
    )
    if description_column is None or not canonical.description:
        return cells
    adjacent_column = next(
        (
            column
            for column in columns
            if column.order == description_column.order + 1
        ),
        None,
    )
    if adjacent_column is None:
        return cells

    cells_by_id = {cell.column_id: cell for cell in cells}
    description_cell = cells_by_id[description_column.id]
    merged_cell = cells_by_id[adjacent_column.id]
    if description_cell.raw_value or not merged_cell.raw_value:
        return cells
    description_match = re.search(
        re.escape(canonical.description),
        merged_cell.raw_value,
        re.IGNORECASE,
    )
    if description_match is None:
        return cells
    printed_description = merged_cell.raw_value[
        description_match.start() : description_match.end()
    ]
    leading_residual = merged_cell.raw_value[
        : description_match.start()
    ].strip(" -:;/,[]")
    trailing_residual = merged_cell.raw_value[
        description_match.end() :
    ].strip(" -:;/,[]")
    if (
        leading_residual
        and trailing_residual
        and leading_residual.casefold() != trailing_residual.casefold()
    ):
        return cells
    residual = leading_residual or trailing_residual
    residual_words = residual.split()
    midpoint = len(residual_words) // 2
    if (
        midpoint
        and len(residual_words) == midpoint * 2
        and [word.casefold() for word in residual_words[:midpoint]]
        == [word.casefold() for word in residual_words[midpoint:]]
    ):
        residual = " ".join(residual_words[:midpoint])

    canonical_ids = {
        token_id
        for evidence in canonical.field_evidence.get("description", ())
        for token_id in evidence.token_ids
    }
    merged_ids = {
        token_id
        for evidence in merged_cell.evidence
        for token_id in evidence.token_ids
    }
    residual_ids = merged_ids - canonical_ids
    description_evidence = _filter_evidence_token_ids(
        merged_cell.evidence,
        canonical_ids,
    )
    residual_evidence = _filter_evidence_token_ids(
        merged_cell.evidence,
        residual_ids,
    )
    if (
        not residual
        or not description_evidence
        or not residual_evidence
    ):
        return cells

    split_flag = "split_from_merged_ocr_token"
    cells_by_id[description_column.id] = description_cell.model_copy(
        update={
            "raw_value": printed_description,
            "evidence": description_evidence,
            "validation_flags": tuple(
                dict.fromkeys(
                    (
                        *(
                            flag
                            for flag in description_cell.validation_flags
                            if flag != "empty_cell"
                        ),
                        split_flag,
                    )
                )
            ),
        }
    )
    cells_by_id[adjacent_column.id] = merged_cell.model_copy(
        update={
            "raw_value": residual,
            "evidence": residual_evidence,
            "validation_flags": tuple(
                dict.fromkeys(
                    (
                        *merged_cell.validation_flags,
                        split_flag,
                    )
                )
            ),
        }
    )
    return tuple(cells_by_id[cell.column_id] for cell in cells)


def _trim_grounded_duplicate_adjacent_description(
    cells: tuple[SourceCell, ...],
    columns: tuple[SourceColumn, ...],
    canonical: CanonicalRow,
) -> tuple[SourceCell, ...]:
    description_column = next(
        (
            column
            for column in columns
            if column.canonical_field == "description"
        ),
        None,
    )
    if description_column is None or not canonical.description:
        return cells
    adjacent_column = next(
        (
            column
            for column in columns
            if column.order == description_column.order + 1
        ),
        None,
    )
    if adjacent_column is None:
        return cells

    cells_by_id = {cell.column_id: cell for cell in cells}
    description_cell = cells_by_id[description_column.id]
    adjacent_cell = cells_by_id[adjacent_column.id]
    description_value = description_cell.raw_value or ""
    adjacent_value = adjacent_cell.raw_value or ""
    if (
        not description_value
        or not adjacent_value
        or not adjacent_cell.evidence
    ):
        return cells
    suffix = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(adjacent_value.strip())}\s*$",
        description_value,
        re.IGNORECASE,
    )
    if suffix is None:
        return cells
    printed_description = description_value[: suffix.start()].rstrip(" -:;/,[]")

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    canonical_ids = {
        token_id
        for evidence in canonical.field_evidence.get("description", ())
        for token_id in evidence.token_ids
    }
    printed_ids = {
        token_id
        for evidence in description_cell.evidence
        for token_id in evidence.token_ids
    }
    if (
        not printed_description
        or normalize(printed_description) != normalize(canonical.description)
        or not canonical_ids
        or not canonical_ids.issubset(printed_ids)
    ):
        return cells
    cells_by_id[description_column.id] = description_cell.model_copy(
        update={
            "raw_value": printed_description,
            "validation_flags": tuple(
                dict.fromkeys(
                    (
                        *description_cell.validation_flags,
                        "split_duplicate_adjacent_value",
                    )
                )
            ),
        }
    )
    return tuple(cells_by_id[cell.column_id] for cell in cells)


def _split_grounded_date_from_description(
    cells: tuple[SourceCell, ...],
    columns: tuple[SourceColumn, ...],
    canonical: CanonicalRow,
) -> tuple[SourceCell, ...]:
    columns_by_field = {
        column.canonical_field: column
        for column in columns
        if column.canonical_field is not None
    }
    date_column = columns_by_field.get("service_date_raw")
    description_column = columns_by_field.get("description")
    if (
        date_column is None
        or description_column is None
        or date_column.order >= description_column.order
        or not canonical.service_date_iso
    ):
        return cells
    cells_by_id = {cell.column_id: cell for cell in cells}
    date_cell = cells_by_id[date_column.id]
    description_cell = cells_by_id[description_column.id]
    date_match = DATE_PREFIX.match(date_cell.raw_value or "")
    date_remainder = (
        date_cell.raw_value[date_match.end() :].strip()
        if date_cell.raw_value and date_match is not None
        else ""
    )
    split_from_date_cell = bool(date_match is not None and date_remainder)
    if split_from_date_cell:
        merged_cell = date_cell
    elif not date_cell.raw_value and description_cell.raw_value:
        merged_cell = description_cell
    else:
        return cells
    assert merged_cell.raw_value is not None
    match = DATE_PREFIX.match(merged_cell.raw_value)
    if match is None:
        return cells
    printed_date = merged_cell.raw_value[: match.end()].strip(" -:")
    remaining_description = merged_cell.raw_value[match.end() :].strip()
    if split_from_date_cell and description_cell.raw_value:
        remaining_description = " ".join(
            (remaining_description, description_cell.raw_value)
        )
    if (
        not remaining_description
        or parse_service_date(printed_date) != canonical.service_date_iso
    ):
        return cells
    date_token_ids = {
        token_id
        for evidence in canonical.field_evidence.get("service_date", ())
        for token_id in evidence.token_ids
    }
    merged_token_ids = {
        token_id
        for evidence in merged_cell.evidence
        for token_id in evidence.token_ids
    }
    if not date_token_ids or not date_token_ids.issubset(merged_token_ids):
        return cells
    date_evidence = tuple(
        evidence
        for evidence in merged_cell.evidence
        if date_token_ids.intersection(evidence.token_ids)
    )
    if not date_evidence:
        return cells
    description_evidence = description_cell.evidence
    if split_from_date_cell:
        date_description, _, printed_request = _clean_description(
            merged_cell.raw_value
        )
        existing_description, _, _ = _clean_description(
            description_cell.raw_value or ""
        )
        comparable_description = " ".join(
            value
            for value in (date_description, existing_description)
            if value
        )
        if printed_request and printed_request != canonical.request_no:
            return cells
        normalized_remaining = re.sub(
            r"[^a-z0-9]+",
            " ",
            comparable_description.casefold(),
        ).strip()
        normalized_canonical = re.sub(
            r"[^a-z0-9]+",
            " ",
            canonical.description.casefold(),
        ).strip()
        description_token_ids = {
            token_id
            for evidence in canonical.field_evidence.get("description", ())
            for token_id in evidence.token_ids
        }
        description_source_evidence = (
            *merged_cell.evidence,
            *description_cell.evidence,
        )
        description_source_token_ids = {
            token_id
            for evidence in description_source_evidence
            for token_id in evidence.token_ids
        }
        if (
            not description_token_ids
            or not description_token_ids.issubset(description_source_token_ids)
            or normalized_remaining != normalized_canonical
        ):
            return cells
        description_evidence = tuple(
            evidence
            for evidence in description_source_evidence
            if description_token_ids.intersection(evidence.token_ids)
        )
        if not description_evidence:
            return cells
    split_flag = "split_from_merged_ocr_token"
    cells_by_id[date_column.id] = date_cell.model_copy(
        update={
            "raw_value": printed_date,
            "evidence": date_evidence,
            "validation_flags": tuple(
                dict.fromkeys(
                    (
                        *(
                            flag
                            for flag in date_cell.validation_flags
                            if flag != "empty_cell"
                        ),
                        split_flag,
                    )
                )
            ),
        }
    )
    cells_by_id[description_column.id] = description_cell.model_copy(
        update={
            "raw_value": remaining_description,
            "evidence": description_evidence,
            "validation_flags": tuple(
                dict.fromkeys(
                    (
                        *(
                            flag
                            for flag in description_cell.validation_flags
                            if flag != "empty_cell"
                        ),
                        split_flag,
                    )
                )
            ),
        }
    )
    return tuple(cells_by_id[cell.column_id] for cell in cells)


def _link_source_tables(
    tables: tuple[SourceTable, ...] | list[SourceTable],
    canonical_rows: tuple[CanonicalRow, ...] | list[CanonicalRow],
) -> tuple[SourceTable, ...]:
    """Link printed rows only when OCR field evidence identifies one canonical row."""

    def evidence_ids(items: object) -> set[str]:
        return {
            token_id
            for item in items or ()
            for token_id in getattr(item, "token_ids", ())
        }

    linked_tables: list[SourceTable] = []
    for table in tables:
        candidates = tuple(
            row
            for row in canonical_rows
            if row.page_number == table.page_number and row.table_id == table.table_id
        )
        linked_rows = []
        for source_row in table.rows:
            source_ids = {
                token_id
                for cell in source_row.cells
                for evidence in cell.evidence
                for token_id in evidence.token_ids
            }
            scored: list[tuple[tuple[int, int, int], CanonicalRow]] = []
            for candidate in candidates:
                description_ids = evidence_ids(
                    candidate.field_evidence.get("description", ())
                )
                if candidate.role in {
                    RowRole.DETAIL,
                    RowRole.REFUND,
                    RowRole.CATEGORY_ROLLUP,
                }:
                    anchor_ids = evidence_ids(candidate.field_evidence.get("amount", ()))
                elif candidate.role is RowRole.INFORMATIONAL:
                    anchor_ids = set().union(
                        *(
                            evidence_ids(candidate.field_evidence.get(field, ()))
                            for field in (
                                "service_date",
                                "request_no",
                                "service_code",
                                "hsn_code",
                            )
                        )
                    )
                else:
                    anchor_ids = description_ids
                anchor_overlap = len(anchor_ids & source_ids)
                if anchor_overlap == 0:
                    continue
                all_ids = evidence_ids(candidate.evidence)
                scored.append(
                    (
                        (
                            anchor_overlap,
                            len(description_ids & source_ids),
                            len(all_ids & source_ids),
                        ),
                        candidate,
                    )
                )
            scored.sort(key=lambda item: item[0], reverse=True)
            canonical_row_id = None
            flags = source_row.validation_flags
            linked_cells = source_row.cells
            if scored and (len(scored) == 1 or scored[0][0] != scored[1][0]):
                matched = scored[0][1]
                canonical_row_id = str(matched.id)
                columns_by_id = {column.id: column for column in table.columns}
                linked_cells = _split_grounded_date_request_description(
                    linked_cells,
                    table.columns,
                    matched,
                )
                linked_cells = _redistribute_grounded_description_from_adjacent_cell(
                    linked_cells,
                    table.columns,
                    matched,
                )
                linked_cells = _trim_grounded_duplicate_adjacent_description(
                    linked_cells,
                    table.columns,
                    matched,
                )
                linked_cells = _split_grounded_date_from_description(
                    linked_cells,
                    table.columns,
                    matched,
                )
                linked_cells = tuple(
                    cell.model_copy(
                        update={
                            "raw_value": None,
                            "evidence": (),
                            "validation_flags": tuple(
                                dict.fromkeys(
                                    (
                                        *cell.validation_flags,
                                        "excluded_oversized_overlay",
                                    )
                                )
                            ),
                        }
                    )
                    if _source_cell_is_invalid_structured_overlay(
                        cell,
                        columns_by_id[cell.column_id],
                        table,
                        source_row,
                        matched,
                    )
                    else cell
                    for cell in linked_cells
                )
            elif scored:
                flags = tuple(dict.fromkeys((*flags, "canonical_link_ambiguous")))
            linked_rows.append(
                source_row.model_copy(
                    update={
                        "canonical_row_id": canonical_row_id,
                        "cells": linked_cells,
                        "validation_flags": flags,
                    }
                )
            )
        linked_tables.append(table.model_copy(update={"rows": tuple(linked_rows)}))
    return tuple(linked_tables)


def _apply_document_role_policy(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    """Apply the authoritative finest-available charge policy.

    A category charge is retained only when no item-level row covers that same
    category. Unlike the old exact-sum heuristic, this cannot delete an
    unrelated legitimate row merely because later values happen to add to it.
    """
    detailed_categories = {
        row.section for row in rows if row.role in {RowRole.DETAIL, RowRole.REFUND} and row.section
    }
    detailed_keys = {
        (
            re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip(),
            row.net_amount,
        )
        for row in rows
        if row.role in {RowRole.DETAIL, RowRole.REFUND}
    }
    selected = [
        row
        for row in rows
        if not (
            row.role is RowRole.CATEGORY_ROLLUP
            and (
                row.net_amount == 0
                or (row.section is not None and row.section in detailed_categories)
                or (
                    re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip(),
                    row.net_amount,
                )
                in detailed_keys
            )
        )
    ]

    # Cross-page category similarity alone is not repeated-print provenance.
    # Exact duplicate overlays are removed earlier only after pixel and
    # full-evidence containment checks; preserve every remaining occurrence.
    return [row.model_copy(update={"row_order": order}) for order, row in enumerate(selected)]


def _apply_profile_constraints(reconstruction, profile):
    schema = reconstruction.schema
    if schema is not None:
        schema = replace(
            schema,
            table_type=profile.table_type,
            confidence=max(schema.confidence, profile.retrieval_threshold),
        )
    optional_fields = {
        "service_date",
        "request_no",
        "service_code",
        "hsn_code",
        "quantity",
        "rate",
        "gross_amount",
        "discount",
    }
    rows = []
    for row in reconstruction.rows:
        candidate = row.candidate
        role = candidate.role
        if profile.table_type in {TableType.CATEGORY_SUMMARY, TableType.PACKAGE_SUMMARY}:
            role = RowRole.CATEGORY_ROLLUP
        elif profile.table_type is TableType.PAYMENT:
            role = RowRole.PAYMENT
        elif candidate.amount is not None and candidate.amount < 0:
            role = RowRole.REFUND
        elif role in {RowRole.CATEGORY_ROLLUP, RowRole.PAYMENT}:
            role = RowRole.DETAIL
        updates = {field: None for field in optional_fields & set(profile.unsupported_fields)}
        rows.append(
            replace(
                row,
                candidate=replace(
                    candidate,
                    role=role,
                    table_type=profile.table_type,
                    **updates,
                ),
                field_token_ids={
                    name: ids
                    for name, ids in row.field_token_ids.items()
                    if name not in profile.unsupported_fields
                },
            )
        )
    return replace(reconstruction, rows=tuple(rows), schema=schema)


def _heavy_disagrees(reconstruction, provider_candidates) -> bool:
    local = {
        (
            re.sub(r"[^a-z0-9]+", " ", (row.candidate.description or "").casefold()).strip(),
            row.candidate.amount,
        )
        for row in reconstruction.rows
    }
    heavy = {
        (
            re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip(),
            row.amount,
        )
        for row in provider_candidates
        if row.description and row.amount is not None
    }
    return bool(heavy and heavy != local)


class OfflineExtractor:
    def __init__(
        self,
        vl_url: str,
        *,
        paddle_device: str = "cpu",
        vl_device: str = "cpu",
        hospital_id: str | None = None,
        profile_registry: Path | None = None,
        gemini_mode: GeminiMode = GeminiMode.OFF,
        gemini_adapter: AdjudicationAdapter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.hospital_id = hospital_id
        self.gemini_mode = gemini_mode
        self.gemini_promotion = None
        if gemini_mode is GeminiMode.ENABLED:
            promotion_path = self.settings.gemini_promotion_path
            if promotion_path is None:
                raise ValueError("enabled Gemini mode requires a frozen promotion decision")
            self.gemini_promotion = validate_promotion(
                promotion_path,
                model=self.settings.gemini_model,
                prompt_version=self.settings.gemini_prompt_version,
                redaction_version=self.settings.gemini_redaction_version,
            )
        self.ocr = PaddleOcrV6Adapter(device=paddle_device)
        self.layout = PaddleDocLayoutV3Adapter(device=paddle_device)
        self.vl = PaddleOcrVlAdapter(base_url=vl_url, device=vl_device)
        self.profiles = (
            JsonProfileRepository(profile_registry).list_profiles()
            if profile_registry is not None and profile_registry.exists()
            else ()
        )
        self.gemini = gemini_adapter
        if (
            self.gemini is None
            and gemini_mode is not GeminiMode.OFF
            and self.settings.gemini_api_key
        ):
            self.gemini = GeminiAdjudicationAdapter(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model,
                timeout_seconds=self.settings.gemini_timeout_seconds,
                input_cost_usd_per_million=self.settings.gemini_input_cost_usd_per_million,
                output_cost_usd_per_million=self.settings.gemini_output_cost_usd_per_million,
            )

    def _profile_match(
        self,
        *,
        document_id: str,
        page_number: int,
        page_width: int,
        page_height: int,
        box: tuple[int, int, int, int],
        reconstruction,
        tokens,
        profiles=None,
        include_shadow: bool = False,
    ) -> ProfileMatch | None:
        candidates = self.profiles if profiles is None else profiles
        if not candidates:
            return None
        table_type = (
            reconstruction.schema.table_type
            if reconstruction.schema is not None
            else TableType(str(reconstruction.diagnostics.get("table_type") or "unknown"))
        )
        token_by_id = {token.token_id: token for token in tokens}
        header_ids = reconstruction.schema.header_token_ids if reconstruction.schema else ()
        header_tokens = tuple(
            token_by_id[token_id].text for token_id in header_ids if token_id in token_by_id
        )
        if not header_tokens:
            scoped = tokens_in_box(tokens, box)
            top_limit = box[1] + (box[3] - box[1]) * 0.2
            header_tokens = tuple(
                token.text
                for token in scoped
                if min(point.y for point in token.polygon.points) <= top_limit
            )
        left, top, right, bottom = box
        observation = LayoutObservation(
            document_id=document_id,
            hospital_id=self.hospital_id,
            page_number=page_number,
            page_type=_page_type(table_type),
            table_type=table_type,
            page_aspect_ratio=page_width / page_height,
            table_box=(
                left / page_width,
                top / page_height,
                right / page_width,
                bottom / page_height,
            ),
            header_tokens=header_tokens,
            column_centers=(
                reconstruction.schema.column_centers if reconstruction.schema is not None else {}
            ),
        )
        return match_profile(candidates, observation, include_shadow=include_shadow)

    def _recover_crop_ocr(
        self,
        *,
        source: Path,
        artifact_root: Path,
        work: TableWork,
        prior_schemas: tuple[TableSchemaState, ...],
        page_artifact_sha256: str,
        baseline: ReconstructionResult,
        baseline_tokens: tuple[OcrToken, ...],
    ) -> tuple[ReconstructionResult | None, tuple[RecoveryAttempt, ...]]:
        attempts: list[RecoveryAttempt] = []
        try:
            high_resolution = render_pdf_region(
                source,
                artifact_root / "crops" / f"{work.table_id}-400dpi.png",
                work.page_number,
                work.box,
            )
        except Exception as error:
            attempts.append(
                RecoveryAttempt(
                    stage=RecoveryStage.HIGH_RESOLUTION,
                    status="failed",
                    reason=f"render_error:{type(error).__name__}",
                )
            )
            return None, tuple(attempts)
        attempts.append(
            RecoveryAttempt(
                stage=RecoveryStage.HIGH_RESOLUTION,
                artifact_sha256=high_resolution.artifact_sha256,
                status="rendered",
            )
        )
        assets = [
            (
                "high_resolution",
                high_resolution.output_path,
                high_resolution.artifact_sha256,
                f"{work.table_id}.400dpi.ocr.json",
            )
        ]
        try:
            photometric = clahe_variant(
                high_resolution.output_path,
                artifact_root / "crops" / f"{work.table_id}-400dpi-clahe.png",
            )
        except Exception as error:
            attempts.append(
                RecoveryAttempt(
                    stage=RecoveryStage.PHOTOMETRIC,
                    status="failed",
                    reason=f"variant_error:{type(error).__name__}",
                )
            )
        else:
            attempts.append(
                RecoveryAttempt(
                    stage=RecoveryStage.PHOTOMETRIC,
                    artifact_sha256=photometric.artifact_sha256,
                    status="prepared",
                )
            )
            assets.append(
                (
                    "photometric",
                    photometric.output_path,
                    photometric.artifact_sha256,
                    f"{work.table_id}.400dpi-clahe.ocr.json",
                )
            )
        reconstructed: list[
            tuple[
                str,
                str,
                bool,
                int,
                tuple[OcrToken, ...],
                ReconstructionResult,
            ]
        ] = []
        for variant, image_path, artifact_sha256, cache_name in assets:
            request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=artifact_sha256,
                image_path=str(image_path.resolve()),
                page_number=work.page_number,
                options={
                    "recovery_stage": RecoveryStage.CROP_OCR.value,
                    "input_variant": variant,
                    "source_crop_sha256": work.crop_sha256,
                },
            )
            try:
                response, cache_hit = _cached_prediction(
                    artifact_root / "inference" / cache_name,
                    request,
                    self.ocr,
                )
            except Exception as error:
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.CROP_OCR,
                        artifact_sha256=artifact_sha256,
                        status="failed",
                        reason=f"{variant}:ocr_error:{type(error).__name__}",
                    )
                )
                continue
            try:
                local_tokens = paddle_ocr_tokens(
                    response.output,
                    work.page_number,
                    artifact_sha256,
                )
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot read recovered crop: {image_path}")
                mapped_tokens = map_crop_tokens_to_page(
                    local_tokens,
                    work.box,
                    image.shape[1],
                    image.shape[0],
                    page_artifact_sha256,
                )
                reconstruction = reconstruct_ocr_rows(
                    mapped_tokens,
                    page_number=work.page_number,
                    table_id=work.table_id,
                    box=work.box,
                    prior_schemas=prior_schemas,
                )
            except Exception as error:
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.CROP_OCR,
                        artifact_sha256=artifact_sha256,
                        cache_hit=cache_hit,
                        latency_ms=response.latency_ms,
                        status="failed",
                        reason=(
                            f"{variant}:reconstruction_error:"
                            f"{type(error).__name__}"
                        ),
                    )
                )
                continue
            reconstructed.append(
                (
                    variant,
                    artifact_sha256,
                    cache_hit,
                    response.latency_ms,
                    mapped_tokens,
                    reconstruction,
                )
            )
        targeted_inputs = tuple(
            (
                item,
                tuple(
                    dict.fromkeys(
                        description_lane_recovery_regions(
                            item[-1],
                            table_box=work.box,
                        )
                    )
                ),
            )
            for item in reconstructed
            if description_lane_recovery_regions(
                item[-1],
                table_box=work.box,
            )
        )
        if targeted_inputs:
            try:
                overlay_suppressed = color_overlay_suppressed_variant(
                    high_resolution.output_path,
                    artifact_root
                    / "crops"
                    / f"{work.table_id}-400dpi-color-suppressed.png",
                )
                overlay_image = cv2.imread(
                    str(overlay_suppressed.output_path),
                    cv2.IMREAD_COLOR,
                )
                if overlay_image is None:
                    raise ValueError(
                        f"cannot read color-suppressed crop: "
                        f"{overlay_suppressed.output_path}"
                    )
            except Exception as error:
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.PHOTOMETRIC,
                        status="failed",
                        reason=(
                            "description_lane_variant_error:"
                            f"{type(error).__name__}"
                        ),
                    )
                )
            else:
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.PHOTOMETRIC,
                        artifact_sha256=overlay_suppressed.artifact_sha256,
                        status="prepared",
                        reason="description_lane_color_suppression",
                    )
                )
                for item, regions in targeted_inputs:
                    (
                        variant,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = item
                    recovered_tokens: list[OcrToken] = []
                    target_cache_hits: list[bool] = []
                    target_latency_ms = 0
                    for region_index, region in enumerate(regions, start=1):
                        pixel_box = map_page_box_to_crop_pixels(
                            region,
                            parent_page_box=work.box,
                            crop_width=overlay_image.shape[1],
                            crop_height=overlay_image.shape[0],
                        )
                        if (
                            pixel_box[2] <= pixel_box[0]
                            or pixel_box[3] <= pixel_box[1]
                        ):
                            continue
                        region_slug = "-".join(str(value) for value in region)
                        try:
                            targeted_crop = crop_region(
                                overlay_suppressed.output_path,
                                artifact_root
                                / "crops"
                                / (
                                    f"{work.table_id}-400dpi-description-"
                                    f"{region_slug}.png"
                                ),
                                work.page_number,
                                pixel_box,
                            )
                            target_request = InferenceRequest(
                                request_id=str(uuid4()),
                                artifact_sha256=targeted_crop.artifact_sha256,
                                image_path=str(targeted_crop.output_path.resolve()),
                                page_number=work.page_number,
                                options={
                                    "recovery_stage": RecoveryStage.CROP_OCR.value,
                                    "input_variant": "description_lane",
                                    "source_crop_sha256": work.crop_sha256,
                                },
                            )
                            target_response, target_cache_hit = _cached_prediction(
                                artifact_root
                                / "inference"
                                / (
                                    f"{work.table_id}.400dpi-description-"
                                    f"{region_slug}.ocr.json"
                                ),
                                target_request,
                                self.ocr,
                            )
                            local_target_tokens = tuple(
                                token.model_copy(
                                    update={
                                        "token_id": (
                                            f"description:{region_index}:"
                                            f"{token.token_id}"
                                        )
                                    }
                                )
                                for token in paddle_ocr_tokens(
                                    target_response.output,
                                    work.page_number,
                                    targeted_crop.artifact_sha256,
                                )
                            )
                            target_image = cv2.imread(
                                str(targeted_crop.output_path),
                                cv2.IMREAD_COLOR,
                            )
                            if target_image is None:
                                raise ValueError(
                                    f"cannot read description crop: "
                                    f"{targeted_crop.output_path}"
                                )
                            recovered_tokens.extend(
                                map_crop_tokens_to_page(
                                    local_target_tokens,
                                    region,
                                    target_image.shape[1],
                                    target_image.shape[0],
                                    page_artifact_sha256,
                                )
                            )
                            target_cache_hits.append(target_cache_hit)
                            target_latency_ms += target_response.latency_ms
                        except Exception as error:
                            attempts.append(
                                RecoveryAttempt(
                                    stage=RecoveryStage.CROP_OCR,
                                    status="failed",
                                    reason=(
                                        "description_lane:"
                                        f"{type(error).__name__}"
                                    ),
                                )
                            )
                    if not recovered_tokens:
                        continue
                    combined_tokens = merge_recovery_tokens(
                        baseline_tokens,
                        (),
                        tuple(recovered_tokens),
                        regions=regions,
                    )
                    try:
                        reconstruction = reconstruct_ocr_rows(
                            combined_tokens,
                            page_number=work.page_number,
                            table_id=work.table_id,
                            box=work.box,
                            prior_schemas=prior_schemas,
                        )
                    except Exception as error:
                        attempts.append(
                            RecoveryAttempt(
                                stage=RecoveryStage.CROP_OCR,
                                artifact_sha256=overlay_suppressed.artifact_sha256,
                                status="failed",
                                reason=(
                                    "description_lane:reconstruction_error:"
                                    f"{type(error).__name__}"
                                ),
                            )
                        )
                        continue
                    reconstructed.append(
                        (
                            f"{variant}+description_lane",
                            overlay_suppressed.artifact_sha256,
                            all(target_cache_hits),
                            target_latency_ms,
                            combined_tokens,
                            reconstruction,
                        )
                    )
        sign_targets = return_sign_recovery_targets(
            baseline,
            table_box=work.box,
        )
        if sign_targets:
            recovered_sign_tokens: list[OcrToken] = []
            recovered_sign_regions: list[tuple[int, int, int, int]] = []
            sign_cache_hits: list[bool] = []
            sign_latency_ms = 0
            sign_artifact_sha256 = page_artifact_sha256
            for target_index, target in enumerate(sign_targets, start=1):
                region = target.region
                region_slug = "-".join(str(value) for value in region)
                try:
                    high_resolution_sign = render_pdf_region(
                        source,
                        artifact_root
                        / "crops"
                        / f"{work.table_id}-800dpi-return-sign-{region_slug}.png",
                        work.page_number,
                        region,
                        output_dpi=800,
                    )
                    enhanced_sign = clahe_variant(
                        high_resolution_sign.output_path,
                        artifact_root
                        / "crops"
                        / (
                            f"{work.table_id}-800dpi-return-sign-"
                            f"{region_slug}-clahe.png"
                        ),
                    )
                    sign_request = InferenceRequest(
                        request_id=str(uuid4()),
                        artifact_sha256=enhanced_sign.artifact_sha256,
                        image_path=str(enhanced_sign.output_path.resolve()),
                        page_number=work.page_number,
                        options={
                            "recovery_stage": RecoveryStage.CROP_OCR.value,
                            "input_variant": "return_sign_800dpi_clahe",
                            "source_crop_sha256": work.crop_sha256,
                        },
                    )
                    sign_response, sign_cache_hit = _cached_prediction(
                        artifact_root
                        / "inference"
                        / (
                            f"{work.table_id}.800dpi-return-sign-"
                            f"{region_slug}.ocr.json"
                        ),
                        sign_request,
                        self.ocr,
                    )
                    local_sign_tokens = paddle_ocr_tokens(
                        sign_response.output,
                        work.page_number,
                        enhanced_sign.artifact_sha256,
                    )
                    matching_sign_tokens = tuple(
                        token
                        for token in local_sign_tokens
                        if parse_decimal(token.text)
                        == -target.expected_absolute_amount
                    )
                    if len(matching_sign_tokens) != 1:
                        raise ValueError("exact negative amount not uniquely recovered")
                    sign_image = cv2.imread(
                        str(enhanced_sign.output_path),
                        cv2.IMREAD_COLOR,
                    )
                    if sign_image is None:
                        raise ValueError(
                            f"cannot read return sign crop: "
                            f"{enhanced_sign.output_path}"
                        )
                    prefixed_sign_tokens = tuple(
                        token.model_copy(
                            update={
                                "token_id": (
                                    f"return-sign:{target_index}:"
                                    f"{token.token_id}"
                                )
                            }
                        )
                        for token in matching_sign_tokens
                    )
                    recovered_sign_tokens.extend(
                        map_crop_tokens_to_page(
                            prefixed_sign_tokens,
                            region,
                            sign_image.shape[1],
                            sign_image.shape[0],
                            page_artifact_sha256,
                        )
                    )
                    recovered_sign_regions.append(region)
                    sign_cache_hits.append(sign_cache_hit)
                    sign_latency_ms += sign_response.latency_ms
                    sign_artifact_sha256 = enhanced_sign.artifact_sha256
                except Exception as error:
                    attempts.append(
                        RecoveryAttempt(
                            stage=RecoveryStage.CROP_OCR,
                            status="failed",
                            reason=(
                                "return_sign_800dpi_clahe:"
                                f"{type(error).__name__}"
                            ),
                        )
                    )
            if recovered_sign_tokens:
                sign_bases = (
                    (
                        "baseline",
                        page_artifact_sha256,
                        True,
                        0,
                        baseline_tokens,
                        baseline,
                    ),
                    *tuple(reconstructed),
                )
                for item in sign_bases:
                    (
                        variant,
                        _,
                        cache_hit,
                        latency_ms,
                        candidate_tokens,
                        _,
                    ) = item
                    combined_tokens = replace_tokens_in_regions(
                        candidate_tokens,
                        tuple(recovered_sign_tokens),
                        regions=tuple(recovered_sign_regions),
                    )
                    try:
                        reconstruction = reconstruct_ocr_rows(
                            combined_tokens,
                            page_number=work.page_number,
                            table_id=work.table_id,
                            box=work.box,
                            prior_schemas=prior_schemas,
                        )
                    except Exception as error:
                        attempts.append(
                            RecoveryAttempt(
                                stage=RecoveryStage.CROP_OCR,
                                artifact_sha256=sign_artifact_sha256,
                                status="failed",
                                reason=(
                                    "return_sign_800dpi_clahe:"
                                    f"reconstruction_error:{type(error).__name__}"
                                ),
                            )
                        )
                        continue
                    reconstructed.append(
                        (
                            f"{variant}+return_sign_800dpi_clahe",
                            sign_artifact_sha256,
                            cache_hit and all(sign_cache_hits),
                            latency_ms + sign_latency_ms,
                            combined_tokens,
                            reconstruction,
                        )
                    )
        safe_candidates = [
            item
            for item in reconstructed
            if safely_improves_reconstruction(baseline, item[-1])
        ]
        selected_item = max(
            safe_candidates,
            key=lambda item: reconstruction_quality(item[-1]),
            default=None,
        )
        for item in reconstructed:
            (
                variant,
                artifact_sha256,
                cache_hit,
                latency_ms,
                _,
                reconstruction,
            ) = item
            selected = item is selected_item
            attempts.append(
                RecoveryAttempt(
                    stage=RecoveryStage.CROP_OCR,
                    artifact_sha256=artifact_sha256,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    produced_rows=len(reconstruction.rows),
                    accepted_rows=len(reconstruction.rows) if selected else 0,
                    status=(
                        "recovered"
                        if selected
                        else ("no_improvement" if reconstruction.rows else "no_rows")
                    ),
                    reason=f"input_variant:{variant}",
                )
            )
        selected = selected_item[-1] if selected_item is not None else None
        return selected, tuple(attempts)

    def extract(
        self,
        source: Path,
        artifact_root: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        manifest = render_pdf(source, artifact_root / "pages", dpi=300)
        if progress:
            progress(0, len(manifest.pages))
        document_id = manifest.document_sha256
        all_rows: list[CanonicalRow] = []
        all_source_tables: list[SourceTable] = []
        source_table_crop_paths: dict[tuple[int, str], Path] = {}
        source_table_crop_boxes: dict[
            tuple[int, str],
            tuple[int, int, int, int],
        ] = {}
        document_total_candidates: list[DocumentTotalCandidate] = []
        diagnostics: list[dict[str, Any]] = []
        schema_states: list[TableSchemaState] = []
        hospital = None
        gemini_calls = 0
        gemini_cost = Decimal("0")
        gemini_provider_disabled_reason: str | None = None
        for page_asset in manifest.pages:
            page_path = artifact_root / "pages" / page_asset.relative_path
            ocr_request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=page_asset.artifact_sha256,
                image_path=str(page_path.resolve()),
                page_number=page_asset.page_number,
            )
            ocr_response, ocr_cache_hit = _cached_prediction(
                artifact_root / "inference" / f"page-{page_asset.page_number}.ocr.json",
                ocr_request,
                self.ocr,
            )
            tokens = paddle_ocr_tokens(
                ocr_response.output,
                page_asset.page_number,
                page_asset.artifact_sha256,
            )
            document_total_candidates.extend(extract_document_total_candidates(tokens))
            if page_asset.page_number == 1:
                hospital = detect_hospital(
                    tokens,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                )
            layout_request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=page_asset.artifact_sha256,
                image_path=str(page_path.resolve()),
                page_number=page_asset.page_number,
            )
            layout_response, layout_cache_hit = _cached_prediction(
                artifact_root / "inference" / f"page-{page_asset.page_number}.layout.json",
                layout_request,
                self.layout,
            )
            layout_boxes = _layout_boxes(layout_response.output)
            result = (ocr_response.output.get("pages") or [{}])[0].get("res") or {}
            proposals = propose_tables_from_ocr(
                result.get("rec_boxes") or [],
                result.get("rec_texts") or [],
            )
            geometry_boxes = [
                tuple(round(value) for value in proposal.box) for proposal in proposals
            ]
            boxes = _merge_table_boxes(layout_boxes, geometry_boxes)
            route = (
                "layout+ocr_geometry"
                if layout_boxes and geometry_boxes
                else ("layout" if layout_boxes else "ocr_geometry")
            )

            table_work: list[TableWork] = []
            for table_index, box in enumerate(boxes):
                table_id = f"p{page_asset.page_number}-t{table_index + 1}"
                safe_box = _safe_box(box, page_asset.width, page_asset.height)
                crop = crop_region(
                    page_path,
                    artifact_root / "crops" / f"{table_id}.png",
                    page_asset.page_number,
                    safe_box,
                )
                source_table_crop_paths[
                    (page_asset.page_number, table_id)
                ] = crop.output_path
                source_table_crop_boxes[
                    (page_asset.page_number, table_id)
                ] = safe_box
                table_work.append(
                    TableWork(
                        table_id=table_id,
                        page_number=page_asset.page_number,
                        page_artifact_sha256=page_asset.artifact_sha256,
                        crop_path=crop.output_path,
                        crop_sha256=crop.artifact_sha256,
                        box=safe_box,
                    )
                )

            if not table_work:
                diagnostics.append(
                    {
                        "page_number": page_asset.page_number,
                        "route": route,
                        "ocr_latency_ms": ocr_response.latency_ms,
                        "ocr_cache_hit": ocr_cache_hit,
                        "layout_latency_ms": layout_response.latency_ms,
                        "layout_cache_hit": layout_cache_hit,
                        "table_count": 0,
                        "status": "no_table_detected",
                    }
                )
            for work in table_work:
                reconstruction = reconstruct_ocr_rows(
                    tokens,
                    page_number=work.page_number,
                    table_id=work.table_id,
                    box=work.box,
                    prior_schemas=tuple(schema_states),
                )
                profile_match = self._profile_match(
                    document_id=document_id,
                    page_number=work.page_number,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                    box=work.box,
                    reconstruction=reconstruction,
                    tokens=tokens,
                )
                shadow_profile_match = self._profile_match(
                    document_id=document_id,
                    page_number=work.page_number,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                    box=work.box,
                    reconstruction=reconstruction,
                    tokens=tokens,
                    profiles=tuple(
                        profile
                        for profile in self.profiles
                        if profile.lifecycle is ProfileLifecycle.SHADOW
                    ),
                    include_shadow=True,
                )
                selected_profile = None
                if profile_match is not None and profile_match.selected:
                    selected_profile = next(
                        profile
                        for profile in self.profiles
                        if profile.profile_key == profile_match.profile_key
                        and profile.profile_version == profile_match.profile_version
                    )
                    guided = reconstruct_ocr_rows(
                        tokens,
                        page_number=work.page_number,
                        table_id=work.table_id,
                        box=work.box,
                        prior_schemas=(
                            *schema_states,
                            profile_to_schema(selected_profile, work.page_number, work.table_id),
                        ),
                    )
                    guided = _apply_profile_constraints(guided, selected_profile)
                    route_name = (
                        f"profile_guided:{selected_profile.profile_key}"
                        f"@{selected_profile.profile_version}"
                    )
                    reconstruction = replace(
                        guided,
                        rows=tuple(
                            replace(
                                row,
                                source_routes=tuple(
                                    dict.fromkeys((*row.source_routes, route_name))
                                ),
                            )
                            for row in guided.rows
                        ),
                    )
                profile_heavy_sample = bool(
                    selected_profile is not None
                    and deterministic_shadow_sample(
                        document_id,
                        selected_profile.profile_key,
                    )
                )
                if reconstruction.schema is not None:
                    schema_states.append(reconstruction.schema)
                parsed_rows = list(
                    canonicalize_rows(
                        document_id,
                        work.page_number,
                        work.table_id,
                        work.page_artifact_sha256,
                        reconstruction.rows,
                    )
                )
                recovery_attempts: list[RecoveryAttempt] = []
                if _should_attempt_crop_recovery(
                    reconstruction,
                    parsed_rows=parsed_rows,
                    table_box=work.box,
                ):
                    recovered, attempts = self._recover_crop_ocr(
                        source=source,
                        artifact_root=artifact_root,
                        work=work,
                        prior_schemas=tuple(schema_states),
                        page_artifact_sha256=work.page_artifact_sha256,
                        baseline=reconstruction,
                        baseline_tokens=tokens_in_box(tokens, work.box),
                    )
                    recovery_attempts.extend(attempts)
                    if recovered is not None:
                        reconstruction = recovered
                        if reconstruction.schema is not None:
                            schema_states.append(reconstruction.schema)
                        parsed_rows = list(
                            canonicalize_rows(
                                document_id,
                                work.page_number,
                                work.table_id,
                                work.page_artifact_sha256,
                                reconstruction.rows,
                            )
                        )
                content = ""
                candidate_count = 0
                provider_candidates = []
                vl_latency_ms = 0
                vl_cache_hit = False
                vl_finish_reason: str | None = None
                vl_error: str | None = None
                vl_truncated = False
                vl_tile_count = 0
                vl_retry_reason: str | None = None
                gemini_invoked = False
                gemini_cache_hit = False
                gemini_grounded_rows = 0
                gemini_rejected_reasons: tuple[str, ...] = ()
                gemini_redaction_safe: bool | None = None
                gemini_masked_tokens = 0
                # OCR is the deterministic primary route. The heavy parser is
                # invoked only when OCR could not reconstruct a trustworthy
                # table, avoiding context pressure on dense, already-readable
                # ledgers.
                advisor_eligible = bool(
                    not is_terminal_non_ledger(reconstruction)
                    and (
                        reconstruction.schema is None
                        or reconstruction.schema.table_type
                        not in {
                            TableType.CATEGORY_SUMMARY,
                            TableType.PACKAGE_SUMMARY,
                            TableType.PAYMENT,
                            TableType.METADATA,
                        }
                    )
                )
                use_vl = bool(
                    advisor_eligible
                    and (
                        profile_heavy_sample
                        or not (profile_match and profile_match.selected and parsed_rows)
                    )
                    and (
                        profile_heavy_sample
                        or not parsed_rows
                        or is_implausibly_low_yield(reconstruction)
                        or reconstruction.diagnostics.get("orientation") != "upright"
                        or (
                            not reconstruction.diagnostics.get("header_found")
                            and not reconstruction.diagnostics.get("schema_inherited")
                        )
                    )
                )
                rows_before_vl = len(parsed_rows)
                if use_vl:
                    scoped_tokens = tokens_in_box(tokens, work.box)
                    vl_cache_hits: list[bool] = []
                    vl_contents: list[str] = []
                    orientation = str(reconstruction.diagnostics.get("orientation") or "upright")
                    primary_asset = _vl_asset(work, artifact_root, orientation)
                    vl_jobs: list[tuple[VlAsset, str, int | None]] = [
                        (primary_asset, f"{work.table_id}.vl.json", None)
                    ]
                    job_index = 0
                    while job_index < len(vl_jobs):
                        asset, cache_name, tile_index = vl_jobs[job_index]
                        vl_request = InferenceRequest(
                            request_id=str(uuid4()),
                            artifact_sha256=asset.artifact_sha256,
                            image_path=str(asset.path.resolve()),
                            page_number=work.page_number,
                            options={
                                "prompt": "Table Recognition:",
                                "prompt_version": "phase2-grounded-v2",
                                "max_tokens": 8192 if tile_index is not None else 16384,
                                "context_total": 24576,
                                "parallel_slots": 1,
                                "tiling_policy": "orientation-aware-overlap-v2",
                                "asset_identity": asset.identity,
                                "tile_index": tile_index,
                            },
                        )
                        try:
                            vl_response, cache_hit = _cached_prediction(
                                artifact_root / "inference" / cache_name,
                                vl_request,
                                self.vl,
                            )
                        except Exception as error:
                            vl_error = f"provider_error:{type(error).__name__}"
                            vl_retry_reason = vl_error
                            break
                        vl_cache_hits.append(cache_hit)
                        vl_latency_ms += vl_response.latency_ms
                        response_content = str(vl_response.output.get("content") or "")
                        vl_contents.append(response_content)
                        response_candidate_count = 0
                        for table in split_otsl_tables(parse_otsl(response_content)):
                            candidates = extract_candidate_rows(table)
                            if reconstruction.schema is not None:
                                candidates = tuple(
                                    replace(
                                        candidate,
                                        role=(
                                            RowRole.CATEGORY_ROLLUP
                                            if reconstruction.schema.table_type
                                            in {
                                                TableType.CATEGORY_SUMMARY,
                                                TableType.PACKAGE_SUMMARY,
                                            }
                                            and candidate.role is RowRole.DETAIL
                                            else candidate.role
                                        ),
                                        table_type=reconstruction.schema.table_type,
                                        category=row_category(
                                            candidate.description or "",
                                            reconstruction.schema.table_type,
                                        ),
                                    )
                                    for candidate in candidates
                                )
                            provider_candidates.extend(candidates)
                            response_candidate_count += len(candidates)
                            if not profile_heavy_sample:
                                aligned = align_candidate_rows(candidates, scoped_tokens)
                                parsed_rows.extend(
                                    canonicalize_rows(
                                        document_id,
                                        work.page_number,
                                        work.table_id,
                                        work.page_artifact_sha256,
                                        aligned,
                                        starting_order=len(parsed_rows),
                                    )
                                )
                        candidate_count += response_candidate_count
                        response_truncated = bool(vl_response.output.get("truncated"))
                        vl_truncated = vl_truncated or response_truncated
                        if job_index == 0:
                            vl_finish_reason = vl_response.output.get("finish_reason")
                            if response_truncated or response_candidate_count == 0:
                                vl_retry_reason = (
                                    "truncated" if response_truncated else "no_candidates"
                                )
                                tiles = _vertical_vl_tiles(
                                    primary_asset,
                                    artifact_root,
                                    work.table_id,
                                )
                                vl_tile_count = len(tiles)
                                vl_jobs.extend(
                                    (
                                        tile_asset,
                                        f"{work.table_id}.tile-{index}.vl.json",
                                        index,
                                    )
                                    for index, tile_asset in enumerate(tiles, start=1)
                                )
                        job_index += 1

                    vl_cache_hit = bool(vl_cache_hits) and all(vl_cache_hits)
                    content = "\n".join(vl_contents)
                    recovery_attempts.append(
                        RecoveryAttempt(
                            stage=RecoveryStage.LOCAL_VLM,
                            cache_hit=vl_cache_hit,
                            latency_ms=vl_latency_ms,
                            produced_rows=candidate_count,
                            accepted_rows=(
                                0
                                if profile_heavy_sample
                                else max(0, len(parsed_rows) - rows_before_vl)
                            ),
                            status=(
                                "provider_failed"
                                if vl_error
                                else (
                                    "truncated"
                                    if vl_truncated
                                    else ("recovered" if candidate_count else "no_rows")
                                )
                            ),
                            reason=vl_retry_reason,
                        )
                    )
                profile_heavy_disagreement = bool(
                    profile_heavy_sample and _heavy_disagrees(reconstruction, provider_candidates)
                )
                fused_rows = (
                    reconstruction.rows
                    if profile_heavy_sample
                    else fuse_provider_descriptions(
                        reconstruction.rows,
                        tuple(provider_candidates),
                    )
                )
                if fused_rows != reconstruction.rows:
                    parsed_rows.extend(
                        canonicalize_rows(
                            document_id,
                            work.page_number,
                            work.table_id,
                            work.page_artifact_sha256,
                            fused_rows,
                            starting_order=len(parsed_rows),
                        )
                    )
                table_type = (
                    reconstruction.schema.table_type
                    if reconstruction.schema is not None
                    else TableType.UNKNOWN
                )
                route_decision = decide_recovery(
                    reconstruction,
                    gemini_mode=self.gemini_mode,
                    vlm_truncated=vl_truncated,
                    profile_match=profile_match,
                )
                needs_gemini_recovery = bool(
                    not is_terminal_non_ledger(reconstruction)
                    and (not parsed_rows or is_implausibly_low_yield(reconstruction))
                )
                gemini_block_reason: str | None = None
                if needs_gemini_recovery and self.gemini_mode is not GeminiMode.OFF:
                    if self.gemini is None:
                        gemini_block_reason = "adapter_unavailable"
                    elif gemini_provider_disabled_reason:
                        gemini_block_reason = gemini_provider_disabled_reason
                    elif gemini_calls >= self.settings.gemini_max_calls_per_document:
                        gemini_block_reason = "call_budget_exhausted"
                    elif gemini_cost >= Decimal(
                        str(self.settings.gemini_max_cost_usd_per_document)
                    ):
                        gemini_block_reason = "cost_budget_exhausted"
                eligible_for_gemini = bool(
                    needs_gemini_recovery
                    and self.gemini_mode is not GeminiMode.OFF
                    and gemini_block_reason is None
                )
                if eligible_for_gemini:
                    try:
                        redaction = redact_crop(
                            work.crop_path,
                            artifact_root / "crops" / f"{work.table_id}-redacted.png",
                            tokens,
                            work.box,
                        )
                    except Exception as error:
                        gemini_block_reason = f"redaction_error:{type(error).__name__}"
                        recovery_attempts.append(
                            RecoveryAttempt(
                                stage=RecoveryStage.GEMINI,
                                status="redaction_blocked",
                                reason=gemini_block_reason,
                            )
                        )
                    else:
                        gemini_redaction_safe = redaction.safe
                        gemini_masked_tokens = len(redaction.masked_token_ids)
                        if redaction.safe:
                            request = AdjudicationRequest(
                                request_id=str(uuid4()),
                                document_id=document_id,
                                page_number=work.page_number,
                                table_id=work.table_id,
                                masked_crop_path=str(redaction.path.resolve()),
                                masked_crop_sha256=redaction.artifact_sha256,
                                page_type=_page_type(table_type),
                                table_type=table_type,
                                tokens=redaction.tokens,
                                unresolved_reasons=route_decision.reasons,
                                candidate_rows=tuple(asdict(row) for row in provider_candidates),
                                prompt_version=self.settings.gemini_prompt_version,
                                redaction_version=self.settings.gemini_redaction_version,
                            )
                            try:
                                response, gemini_cache_hit = cached_adjudication(
                                    artifact_root / "inference" / f"{work.table_id}.gemini.json",
                                    request,
                                    self.gemini,
                                )
                            except Exception as error:
                                gemini_invoked = True
                                gemini_calls += 1
                                gemini_provider_disabled_reason = (
                                    f"provider_error:{type(error).__name__}"
                                )
                                gemini_block_reason = gemini_provider_disabled_reason
                                recovery_attempts.append(
                                    RecoveryAttempt(
                                        stage=RecoveryStage.GEMINI,
                                        artifact_sha256=redaction.artifact_sha256,
                                        status="provider_failed",
                                        reason=gemini_block_reason,
                                    )
                                )
                            else:
                                gemini_invoked = True
                                gemini_calls += 0 if gemini_cache_hit else 1
                                if not gemini_cache_hit:
                                    gemini_cost += response.measured_cost_usd
                                try:
                                    image = cv2.imread(str(redaction.path), cv2.IMREAD_COLOR)
                                    if image is None:
                                        raise ValueError("cannot read redacted crop")
                                    grounded = ground_adjudication(
                                        response,
                                        validation_tokens=redaction.tokens,
                                        evidence_tokens=tokens_in_box(tokens, work.box),
                                        crop_width=image.shape[1],
                                        crop_height=image.shape[0],
                                        table_type=table_type,
                                        column_centers=(
                                            reconstruction.schema.column_centers
                                            if reconstruction.schema is not None
                                            else None
                                        ),
                                    )
                                except Exception as error:
                                    gemini_block_reason = f"grounding_error:{type(error).__name__}"
                                    recovery_attempts.append(
                                        RecoveryAttempt(
                                            stage=RecoveryStage.GEMINI,
                                            artifact_sha256=redaction.artifact_sha256,
                                            cache_hit=gemini_cache_hit,
                                            latency_ms=response.latency_ms,
                                            produced_rows=len(response.rows),
                                            status="grounding_failed",
                                            reason=gemini_block_reason,
                                        )
                                    )
                                else:
                                    gemini_grounded_rows = len(grounded.rows)
                                    gemini_rejected_reasons = grounded.rejected_reasons
                                    if self.gemini_mode is GeminiMode.ENABLED and grounded.rows:
                                        parsed_rows.extend(
                                            canonicalize_rows(
                                                document_id,
                                                work.page_number,
                                                work.table_id,
                                                work.page_artifact_sha256,
                                                grounded.rows,
                                                starting_order=len(parsed_rows),
                                            )
                                        )
                                    recovery_attempts.append(
                                        RecoveryAttempt(
                                            stage=RecoveryStage.GEMINI,
                                            artifact_sha256=redaction.artifact_sha256,
                                            cache_hit=gemini_cache_hit,
                                            latency_ms=response.latency_ms,
                                            produced_rows=len(response.rows),
                                            accepted_rows=(
                                                gemini_grounded_rows
                                                if self.gemini_mode is GeminiMode.ENABLED
                                                else 0
                                            ),
                                            status=(
                                                "challenger"
                                                if self.gemini_mode is GeminiMode.CHALLENGER
                                                else ("recovered" if grounded.rows else "rejected")
                                            ),
                                            reason=(
                                                ",".join(grounded.rejected_reasons)
                                                if grounded.rejected_reasons
                                                else None
                                            ),
                                        )
                                    )
                        else:
                            gemini_rejected_reasons = redaction.reasons
                            gemini_block_reason = "redaction_blocked"
                            recovery_attempts.append(
                                RecoveryAttempt(
                                    stage=RecoveryStage.GEMINI,
                                    status="redaction_blocked",
                                    reason=",".join(redaction.reasons),
                                )
                            )
                elif (
                    needs_gemini_recovery
                    and self.gemini_mode is not GeminiMode.OFF
                    and gemini_block_reason
                ):
                    recovery_attempts.append(
                        RecoveryAttempt(
                            stage=RecoveryStage.GEMINI,
                            status="not_invoked",
                            reason=gemini_block_reason,
                        )
                    )
                terminal_unresolved = bool(
                    not is_terminal_non_ledger(reconstruction)
                    and (
                        not parsed_rows
                        or (
                            RecoveryReason.LOW_YIELD in route_decision.reasons
                            and len(parsed_rows) <= len(reconstruction.rows)
                        )
                    )
                )
                if terminal_unresolved and route_decision.reasons:
                    recovery_attempts.append(
                        RecoveryAttempt(
                            stage=RecoveryStage.REVIEW,
                            status="pending",
                            reason=",".join(reason.value for reason in route_decision.reasons),
                        )
                    )
                all_rows.extend(parsed_rows)
                all_source_tables.extend(reconstruction.source_tables)
                diagnostics.append(
                    {
                        "page_number": page_asset.page_number,
                        "route": route,
                        "ocr_latency_ms": ocr_response.latency_ms,
                        "ocr_cache_hit": ocr_cache_hit,
                        "layout_latency_ms": layout_response.latency_ms,
                        "layout_cache_hit": layout_cache_hit,
                        "table_id": work.table_id,
                        "crop_sha256": work.crop_sha256,
                        "box": work.box,
                        "vl_invoked": use_vl,
                        "vl_latency_ms": vl_latency_ms,
                        "vl_cache_hit": vl_cache_hit,
                        "vl_finish_reason": vl_finish_reason,
                        "vl_truncated": vl_truncated,
                        "vl_tile_count": vl_tile_count,
                        "vl_retry_reason": vl_retry_reason,
                        "vl_error": vl_error,
                        "candidate_count": candidate_count,
                        "canonical_count": len(parsed_rows),
                        "phase3_route": route_decision.model_dump(mode="json"),
                        "profile_match": (
                            profile_match.model_dump(mode="json") if profile_match else None
                        ),
                        "shadow_profile_match": (
                            shadow_profile_match.model_dump(mode="json")
                            if shadow_profile_match
                            else None
                        ),
                        "profile_heavy_sample": profile_heavy_sample,
                        "profile_heavy_disagreement": profile_heavy_disagreement,
                        "recovery_attempts": [
                            attempt.model_dump(mode="json") for attempt in recovery_attempts
                        ],
                        "gemini_mode": self.gemini_mode.value,
                        "gemini_invoked": gemini_invoked,
                        "gemini_cache_hit": gemini_cache_hit,
                        "gemini_grounded_rows": gemini_grounded_rows,
                        "gemini_rejected_reasons": gemini_rejected_reasons,
                        "gemini_redaction_safe": gemini_redaction_safe,
                        "gemini_masked_tokens": gemini_masked_tokens,
                        "gemini_block_reason": gemini_block_reason,
                        "content": content,
                        **reconstruction.diagnostics,
                    }
                )
            if progress:
                progress(page_asset.page_number, len(manifest.pages))

        (
            selected_source_tables,
            selected_rows,
            suppressed_source_tables,
        ) = _suppress_repeated_printed_tables(
            all_source_tables,
            all_rows,
            crop_paths=source_table_crop_paths,
            crop_boxes=source_table_crop_boxes,
        )
        rows = _apply_document_role_policy(_deduplicate(selected_rows))
        source_tables = _link_source_tables(selected_source_tables, rows)
        document_totals = select_document_totals(document_total_candidates)
        document_total: DocumentTotal | None = select_document_total(document_total_candidates)
        return {
            "output_version": "offline_accuracy_spine_v3",
            "document_total_version": DOCUMENT_TOTAL_VERSION,
            "document_totals_version": DOCUMENT_TOTALS_VERSION,
            "document_total": (
                document_total.model_dump(mode="json") if document_total is not None else None
            ),
            "document_totals": [total.model_dump(mode="json") for total in document_totals],
            "document_id": document_id,
            "hospital_id": self.hospital_id,
            "hospital": hospital,
            "source_sha256": sha256_file(source),
            "source_name": source.name,
            "pages": len(manifest.pages),
            "page_assets": [
                {
                    "page_number": page.page_number,
                    "artifact_sha256": page.artifact_sha256,
                    "width": page.width,
                    "height": page.height,
                    "relative_path": str(Path("pages") / page.relative_path),
                }
                for page in manifest.pages
            ],
            "source_tables": [table.model_dump(mode="json") for table in source_tables],
            "suppressed_repeated_source_tables": [
                {
                    "page_number": page_number,
                    "table_id": table_id,
                }
                for page_number, table_id in suppressed_source_tables
            ],
            "rows": [row.model_dump(mode="json") for row in rows],
            "diagnostics": diagnostics,
            "provider_usage": {
                "gemini_mode": self.gemini_mode.value,
                "gemini_calls": gemini_calls,
                "gemini_measured_cost_usd": str(gemini_cost),
                "gemini_provider_disabled_reason": gemini_provider_disabled_reason,
                "gemini_promotion_manifest_sha256": (
                    self.gemini_promotion.frozen_manifest_sha256 if self.gemini_promotion else None
                ),
            },
        }


@app.command("run")
def run(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option(file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    hospital_id: str | None = None,
    profile_registry: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    gemini_mode: GeminiMode = GeminiMode.OFF,
) -> None:
    result = OfflineExtractor(
        vl_url,
        paddle_device=paddle_device,
        vl_device=vl_device,
        hospital_id=hospital_id,
        profile_registry=profile_registry,
        gemini_mode=gemini_mode,
    ).extract(source, artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "document_id": result["document_id"],
                "pages": result["pages"],
                "rows": len(result["rows"]),
            }
        )
    )


if __name__ == "__main__":
    app()
