from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Annotated, Any
from uuid import uuid4

import cv2
import typer

from gmoney.contracts.evidence import OcrToken, PageAsset, Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    DerivedFieldProvenance,
    EvidenceRef,
    PageType,
    RawTotalCandidate,
    ReceiptSourceMetadata,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    TableType,
    TokenManifestEntry,
)
from gmoney.contracts.phase3 import (
    AdjudicationRequest,
    GeminiMode,
    LayoutObservation,
    LayoutProfile,
    ProfileLifecycle,
    ProfileMatch,
    RecoveryAttempt,
    RecoveryReason,
    RecoveryStage,
)
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.date_context import service_date_from_context
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
    DocumentTotalCandidate,
    assign_document_total_contexts,
    extract_document_total_candidates,
    select_document_total,
    select_document_totals,
)
from gmoney.extraction.hospital import detect_hospital
from gmoney.extraction.ocr_rows import (
    DATE_PREFIX,
    DATE_SPAN,
    ReconstructionResult,
    TableSchemaState,
    _clean_description,
    _request_prefix_match,
    _split_merged_serial_description,
    _structured_field_value_is_valid,
    fuse_provider_descriptions,
    matched_header_alias_ids,
    reconstruct_ocr_rows,
    row_category,
    set_header_aliases,
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
from gmoney.extraction.spatial import AlignedLedgerRow, align_candidate_rows
from gmoney.extraction.typed_values import (
    parse_decimal,
    parse_quantity,
    parse_service_date,
)
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
from gmoney.profiles.aliases import (
    CANONICAL_TO_HEADER_ROLE,
    AliasRegistryUnavailable,
    JsonAliasRepository,
)
from gmoney.profiles.lifecycle import deterministic_shadow_sample
from gmoney.profiles.matching import match_profile, profile_to_schema
from gmoney.profiles.repository import JsonProfileRepository, combined_hospital_name_owners
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


def _recovery_prior_schemas(
    schemas: tuple[TableSchemaState, ...] | list[TableSchemaState],
    *,
    page_number: int,
    table_id: str,
) -> tuple[TableSchemaState, ...]:
    """Exclude a table's final state from retries that start at its beginning."""
    return tuple(
        schema
        for schema in schemas
        if (
            schema.source_page != page_number
            or schema.source_table != table_id
        )
    )


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


def _needs_full_page_financial_recovery(tokens: tuple[OcrToken, ...]) -> bool:
    """Identify financially relevant forms when neither table detector fires."""
    text = re.sub(
        r"[^a-z0-9₹./:-]+",
        " ",
        " ".join(token.text for token in tokens).casefold(),
    ).strip()
    if not text:
        return False
    local_labels = any(
        marker in text
        for marker in (
            "amount",
            "bill no",
            "date",
            "invoice no",
            "payment",
            "receipt no",
            "receipt book",
            "reservation receipt",
            "distribution receipt",
            "charges towards",
            "payment received",
        )
    )
    money_like = bool(
        re.search(
            r"(?:₹|\brs\.?\s*)?\d[\d,]*(?:\.\d{1,2})?(?:\s*/-)?",
            text,
        )
    )
    return local_labels and money_like


def _assess_no_table_page(
    page_path: Path,
    tokens: tuple[OcrToken, ...],
) -> dict[str, Any]:
    """Classify a detector-empty page without trusting baseline OCR alone."""
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        evidence = tuple(token.token_id for token in tokens)
        return {
            "demonstrably_blank": False,
            "financial_form_suspected": _needs_full_page_financial_recovery(tokens),
            "financial_form_classification": "unresolved",
            "financial_form_classification_evidence": evidence,
            "form_signals": ("image_unreadable",),
        }

    normalized_text = re.sub(
        r"[^a-z0-9₹./:-]+",
        " ",
        " ".join(token.text for token in tokens).casefold(),
    ).strip()
    foreground = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    foreground_ratio = float(cv2.countNonZero(foreground)) / float(foreground.size)
    demonstrably_blank = bool(
        not re.search(r"[a-z0-9]", normalized_text)
        and float(image.std()) < 2.0
        and foreground_ratio < 0.001
    )

    signals: list[str] = []
    if _needs_full_page_financial_recovery(tokens):
        signals.append("financial_ocr_text")
    if re.search(r"(?:₹|\brs\.?\s*)\d|\d[\d,]*\.\d{2}\b", normalized_text):
        signals.append("currency_geometry")
    receipt_terms = (
        "receipt",
        "amount received",
        "charges towards",
        "payment received",
        "bill no",
        "invoice no",
    )
    if any(term in normalized_text for term in receipt_terms):
        signals.append("financial_form_language")

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    minimum_line = max(80, image.shape[1] // 5)
    lines = cv2.HoughLinesP(
        edges,
        1,
        3.141592653589793 / 180,
        threshold=60,
        minLineLength=minimum_line,
        maxLineGap=12,
    )
    line_count = 0 if lines is None else len(lines)
    if line_count >= 3 and foreground_ratio >= 0.002:
        signals.append("form_line_structure")

    reference_present = bool(
        re.search(r"\b(?:receipt|invoice|bill|reference|ref)\s*(?:no|number|#)", normalized_text)
    )
    date_present = bool(DATE_SPAN.search(normalized_text))
    amount_present = bool(
        re.search(r"(?:₹|\brs\.?\s*)\d|\d[\d,]*\.\d{2}\b", normalized_text)
    )
    billable_language = any(
        marker in normalized_text
        for marker in ("charge", "charges towards", "invoice", "bill amount")
    )
    nonbillable_language = any(
        marker in normalized_text
        for marker in (
            "advance received",
            "deposit received",
            "amount refunded",
            "refund issued",
            "settlement",
        )
    )
    suspected = bool(
        not demonstrably_blank
        and (
            (
                amount_present
                and (signals or reference_present or date_present)
            )
            or "form_line_structure" in signals
        )
    )
    classification = (
        "blank"
        if demonstrably_blank
        else (
            "recognized_nonbillable"
            if suspected and nonbillable_language and not billable_language
            else ("unresolved" if suspected else "nonfinancial")
        )
    )
    classification_ids = tuple(
        token.token_id
        for token in tokens
        if token.text.strip()
        and (
            parse_decimal(token.text) is not None
            or any(
                marker in token.text.casefold()
                for marker in (
                    "amount",
                    "bill",
                    "charge",
                    "date",
                    "deposit",
                    "invoice",
                    "payment",
                    "receipt",
                    "reference",
                    "settlement",
                )
            )
        )
    )

    return {
        "demonstrably_blank": demonstrably_blank,
        "financial_form_suspected": suspected,
        "financial_form_classification": classification,
        "financial_form_classification_evidence": classification_ids,
        "form_signals": tuple(dict.fromkeys(signals)),
        "foreground_ratio": round(foreground_ratio, 6),
        "form_line_count": line_count,
    }


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

    def grounded_position(row: CanonicalRow) -> tuple[float, float]:
        points = tuple(
            point
            for item in (
                *row.evidence,
                *(
                    evidence
                    for field_evidence in row.field_evidence.values()
                    for evidence in field_evidence
                ),
            )
            for point in item.polygon.points
        )
        if not points:
            return float("inf"), float("inf")
        return min(point.y for point in points), min(point.x for point in points)

    return sorted(
        selected.values(),
        key=lambda row: (
            row.page_number,
            *grounded_position(row),
            row.table_id,
            row.row_order,
        ),
    )


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
    if peer_height <= 0:
        return False
    maximum_height = max(own_heights)
    if maximum_height > peer_height * 3:
        return True
    if maximum_height <= peer_height * 2:
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
    table: SourceTable,
    source_row: SourceRow,
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
    structured_roles = {
        "service_date_raw": "service_date",
        "request_no": "request_no",
        "service_code": "service_code",
        "hsn_code": "hsn_code",
    }
    roles_by_column = {
        column.id: structured_roles[column.canonical_field]
        for column in table.columns
        if column.canonical_field in structured_roles
    }
    financial_column_ids = {
        column.id
        for column in table.columns
        if column.canonical_field in {"net_amount", "gross_amount"}
    }

    def row_has_financial_value(row: SourceRow) -> bool:
        return any(
            candidate.column_id in financial_column_ids
            and candidate.raw_value
            and parse_decimal(candidate.raw_value) is not None
            for candidate in row.cells
        )

    rotated_structured_cells = tuple(
        (neighboring_row, candidate)
        for neighboring_row in neighboring_rows
        for candidate in neighboring_row.cells
        if candidate.column_id in roles_by_column
        and candidate.raw_value
        and "all_text_rotated" in candidate.validation_flags
        and not _structured_field_value_is_valid(
            roles_by_column[candidate.column_id],
            candidate.raw_value,
        )
    )
    return (
        len(rotated_structured_cells) >= 3
        and len(
            {
                candidate.column_id
                for _, candidate in rotated_structured_cells
            }
        )
        >= 2
        and len(
            {
                neighboring_row.id
                for neighboring_row, _ in rotated_structured_cells
                if not row_has_financial_value(neighboring_row)
            }
        )
        >= 2
    )


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
    if _source_cell_is_oversized_overlay(
        cell,
        column,
        table.columns,
        source_row.cells,
    ):
        return True
    if _structured_field_value_is_valid(
        role,
        re.sub(r"[\s:#]+", "", cell.raw_value),
    ):
        return False
    return _source_cell_is_in_rotated_overlay_cluster(
        table,
        source_row,
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
    merged_tail = remainder[request_match.end() :].strip(" -:.;,")
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


def _populate_grounded_service_date_cell(
    cells: tuple[SourceCell, ...],
    columns: tuple[SourceColumn, ...],
    canonical: CanonicalRow,
) -> tuple[SourceCell, ...]:
    """Display a grounded date that OCR merged into another Printed cell."""
    date_columns = tuple(
        column
        for column in columns
        if column.canonical_field == "service_date_raw"
    )
    if len(date_columns) != 1 or not canonical.service_date_iso:
        return cells
    date_column = date_columns[0]
    cells_by_id = {cell.column_id: cell for cell in cells}
    date_cell = cells_by_id[date_column.id]
    if date_cell.raw_value:
        return cells

    date_token_ids = {
        token_id
        for evidence in canonical.field_evidence.get("service_date", ())
        for token_id in evidence.token_ids
    }
    if not date_token_ids:
        return cells
    candidates: list[tuple[str, tuple[EvidenceRef, ...]]] = []
    for cell in cells:
        if cell.column_id == date_column.id or not cell.raw_value:
            continue
        match = DATE_PREFIX.match(cell.raw_value)
        if match is None:
            continue
        printed_date = cell.raw_value[: match.end()].strip(" -:")
        if parse_service_date(printed_date) != canonical.service_date_iso:
            continue
        cell_token_ids = {
            token_id
            for evidence in cell.evidence
            for token_id in evidence.token_ids
        }
        if not date_token_ids.issubset(cell_token_ids):
            continue
        date_evidence = _filter_evidence_token_ids(
            cell.evidence,
            date_token_ids,
        )
        if date_evidence:
            candidates.append((printed_date, date_evidence))
    if len(candidates) != 1:
        return cells

    printed_date, date_evidence = candidates[0]
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
                        "split_from_merged_ocr_token",
                    )
                )
            ),
        }
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
    cells_by_id = {cell.column_id: cell for cell in cells}
    description_cell = cells_by_id[description_column.id]
    if description_cell.raw_value:
        return cells

    adjacent_matches = tuple(
        (column, cells_by_id[column.id], description_match)
        for column in columns
        if abs(column.order - description_column.order) == 1
        and (merged_value := cells_by_id[column.id].raw_value)
        and (
            description_match := re.search(
                re.escape(canonical.description),
                merged_value,
                re.IGNORECASE,
            )
        )
    )
    if len(adjacent_matches) != 1:
        return cells
    adjacent_column, merged_cell, description_match = adjacent_matches[0]
    assert merged_cell.raw_value is not None
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
    merged_serial = _split_merged_serial_description(merged_cell.raw_value)
    normalized_adjacent_label = re.sub(
        r"[^a-z0-9#]+",
        " ",
        adjacent_column.label.casefold(),
    ).strip()
    serial_split_is_grounded = bool(
        adjacent_column.order == description_column.order - 1
        and normalized_adjacent_label
        in {"#", "no", "s no", "serial no", "sr n", "sr no"}
        and merged_serial is not None
        and merged_serial[0] == residual
        and re.sub(r"[^a-z0-9]+", " ", merged_serial[1].casefold()).strip()
        == re.sub(
            r"[^a-z0-9]+",
            " ",
            canonical.description.casefold(),
        ).strip()
        and canonical_ids
        and canonical_ids == merged_ids
    )
    if serial_split_is_grounded:
        description_evidence = merged_cell.evidence
        residual_evidence = merged_cell.evidence
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


def _consolidate_grounded_adjacent_descriptions(
    table: SourceTable,
    canonical_rows: tuple[CanonicalRow, ...],
) -> SourceTable:
    description_column = next(
        (
            column
            for column in table.columns
            if column.canonical_field == "description"
        ),
        None,
    )
    if description_column is None:
        return table
    canonical_by_id = {str(row.id): row for row in canonical_rows}
    rows = list(table.rows)

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    for linked_index, linked_row in enumerate(tuple(rows)):
        if linked_row.canonical_row_id is None:
            continue
        canonical = canonical_by_id.get(linked_row.canonical_row_id)
        if canonical is None or not canonical.description:
            continue
        linked_cells = {
            cell.column_id: cell for cell in rows[linked_index].cells
        }
        target = linked_cells[description_column.id]
        if target.raw_value and target.raw_value.strip():
            continue
        description_ids = {
            token_id
            for evidence in canonical.field_evidence.get("description", ())
            for token_id in evidence.token_ids
        }
        if not description_ids:
            continue
        donors: list[tuple[int, SourceCell, tuple[EvidenceRef, ...]]] = []
        for donor_index in range(
            max(0, linked_index - 1),
            min(len(rows), linked_index + 2),
        ):
            if donor_index == linked_index:
                continue
            donor_row = rows[donor_index]
            if donor_row.canonical_row_id is not None:
                continue
            donor = next(
                cell
                for cell in donor_row.cells
                if cell.column_id == description_column.id
            )
            if (
                not donor.raw_value
                or normalized(donor.raw_value)
                != normalized(canonical.description)
            ):
                continue
            donor_ids = {
                token_id
                for evidence in donor.evidence
                for token_id in evidence.token_ids
            }
            if not description_ids.issubset(donor_ids):
                continue
            grounded = _filter_evidence_token_ids(
                donor.evidence,
                description_ids,
            )
            if grounded:
                donors.append((donor_index, donor, grounded))
        if len(donors) != 1:
            continue

        donor_index, donor, grounded = donors[0]
        linked_cells[description_column.id] = target.model_copy(
            update={
                "raw_value": donor.raw_value,
                "evidence": grounded,
                "validation_flags": tuple(
                    dict.fromkeys(
                        (
                            *(
                                flag
                                for flag in target.validation_flags
                                if flag != "empty_cell"
                            ),
                            "redistributed_from_adjacent_source_row",
                        )
                    )
                ),
            }
        )
        rows[linked_index] = rows[linked_index].model_copy(
            update={
                "cells": tuple(
                    linked_cells[cell.column_id]
                    for cell in rows[linked_index].cells
                )
            }
        )
        donor_cells = {
            cell.column_id: cell for cell in rows[donor_index].cells
        }
        donor_cells[description_column.id] = donor.model_copy(
            update={
                "raw_value": None,
                "evidence": (),
                "validation_flags": tuple(
                    dict.fromkeys(
                        (
                            *donor.validation_flags,
                            "empty_cell",
                            "redistributed_to_linked_source_row",
                        )
                    )
                ),
            }
        )
        rows[donor_index] = rows[donor_index].model_copy(
            update={
                "cells": tuple(
                    donor_cells[cell.column_id]
                    for cell in rows[donor_index].cells
                )
            }
        )
    return table.model_copy(update={"rows": tuple(rows)})


_SERVICE_DATE_TABLE_TYPES = {
    TableType.CATEGORY_SUMMARY,
    TableType.ITEM_LEDGER,
    TableType.LABORATORY,
    TableType.PHARMACY,
}
def _source_cell_service_date(
    cell: SourceCell,
    column: SourceColumn,
) -> tuple[str, str, tuple[EvidenceRef, ...]] | None:
    raw = re.sub(r"\s+", " ", cell.raw_value or "").strip()
    if not raw or not cell.evidence:
        return None
    parsed = service_date_from_context(
        raw,
        column_label=column.label,
        canonical_field=column.canonical_field,
    )
    if parsed is None:
        return None
    value, iso = parsed
    evidence = tuple(item for item in cell.evidence if item.token_ids)
    if not evidence:
        return None
    return value, iso, evidence


def _append_unique_evidence(
    existing: tuple[EvidenceRef, ...],
    additions: tuple[EvidenceRef, ...],
) -> tuple[EvidenceRef, ...]:
    output = list(existing)
    signatures = {
        (
            item.page_number,
            item.table_id,
            item.artifact_sha256,
            item.token_ids,
        )
        for item in output
    }
    for item in additions:
        signature = (
            item.page_number,
            item.table_id,
            item.artifact_sha256,
            item.token_ids,
        )
        if signature not in signatures:
            output.append(item)
            signatures.add(signature)
    return tuple(output)


def _with_grounded_service_date(
    row: CanonicalRow,
    *,
    raw: str,
    iso: str,
    evidence: tuple[EvidenceRef, ...],
    inherited: bool,
) -> CanonicalRow:
    correcting_ambiguous_date = bool(
        row.service_date_raw
        and row.service_date_iso is None
        and len(
            {
                parsed
                for match in DATE_SPAN.finditer(row.service_date_raw)
                if (parsed := parse_service_date(match.group(0))) is not None
            }
        )
        > 1
    )
    if row.service_date_raw and not correcting_ambiguous_date:
        return row
    field_evidence = dict(row.field_evidence)
    field_evidence["service_date"] = evidence
    flags = (
        "service_date_corrected_from_source_cell"
        if correcting_ambiguous_date
        else (
            "service_date_inherited_from_group"
            if inherited
            else "service_date_recovered_from_source_cell"
        )
    )
    return row.model_copy(
        update={
            "service_date_raw": raw,
            "service_date_iso": iso,
            "field_evidence": field_evidence,
            "evidence": _append_unique_evidence(row.evidence, evidence),
            "validation_flags": tuple(
                dict.fromkeys((*row.validation_flags, flags))
            ),
        }
    )


def _recover_grounded_service_dates(
    source_tables: tuple[SourceTable, ...] | list[SourceTable],
    canonical_rows: tuple[CanonicalRow, ...] | list[CanonicalRow],
    *,
    already_linked: bool = False,
) -> list[CanonicalRow]:
    """Recover printed charge dates without borrowing document metadata.

    Canonical rows and source rows are first linked by their existing amount
    evidence. A date can then be recovered only from that linked charge row, or
    from one unlinked date-only row whose adjacency proves a grouped/continued
    charge date. Payment and metadata tables never participate.
    """

    rows = list(canonical_rows)
    row_indexes = {str(row.id): index for index, row in enumerate(rows)}
    initially_linked = (
        tuple(source_tables)
        if already_linked
        else _link_source_tables(source_tables, rows)
    )

    def parsed_date(iso: str) -> date:
        return date.fromisoformat(iso)

    trusted_dates = [
        parsed_date(row.service_date_iso)
        for row in rows
        if row.service_date_iso
    ]
    anchor_markers = (
        "admission date",
        "bill date",
        "date of admission",
        "date of discharge",
        "discharge date",
        "print date",
        "print time",
        "receipt date",
    )
    for table in initially_linked:
        columns = {column.id: column for column in table.columns}
        for source_row in table.rows:
            for cell in source_row.cells:
                raw = re.sub(r"\s+", " ", cell.raw_value or "").strip()
                context = " ".join(
                    (
                        columns[cell.column_id].label.casefold(),
                        raw.casefold(),
                    )
                )
                if (
                    "expiry" in context
                    or not any(marker in context for marker in anchor_markers)
                ):
                    continue
                for match in DATE_SPAN.finditer(raw):
                    if iso := parse_service_date(match.group(0)):
                        trusted_dates.append(parsed_date(iso))

    column_dates: dict[tuple[str, str], tuple[date, ...]] = {}
    for table in initially_linked:
        columns = {column.id: column for column in table.columns}
        for column in table.columns:
            values = tuple(
                parsed_date(candidate[1])
                for source_row in table.rows
                for cell in source_row.cells
                if cell.column_id == column.id
                if (
                    candidate := _source_cell_service_date(
                        cell,
                        columns[cell.column_id],
                    )
                )
                is not None
            )
            column_dates[(table.id, column.id)] = values

    def direct_candidate_is_admissible(
        table: SourceTable,
        column: SourceColumn,
        candidate: tuple[str, str, tuple[EvidenceRef, ...]],
        canonical: CanonicalRow,
    ) -> bool:
        normalized_description = re.sub(
            r"[^a-z0-9]+",
            " ",
            (canonical.description or "").casefold(),
        ).strip()
        if (
            not normalized_description
            or parse_service_date(canonical.description) is not None
            or normalized_description.startswith(
                ("advance ", "payment ", "receipt ", "refund ")
            )
            or normalized_description
            in {"advance", "cash", "cashless", "debit", "finance", "payment"}
            or re.fullmatch(r"rc\d[\da-z/-]*", normalized_description)
        ):
            return False
        candidate_date = parsed_date(candidate[1])
        anchor_close = bool(
            not trusted_dates
            or min(abs((candidate_date - anchor).days) for anchor in trusted_dates)
            <= 120
        )
        if not anchor_close:
            return False
        normalized_label = re.sub(
            r"[^a-z0-9]+",
            " ",
            column.label.casefold(),
        ).strip()
        explicit_date_lane = bool(
            column.canonical_field == "service_date_raw"
            or (
                any(word in normalized_label.split() for word in ("date", "time"))
                and "expiry" not in normalized_label
            )
        )
        if explicit_date_lane:
            return True
        if not trusted_dates and DATE_PREFIX.match(candidate[0]) is None:
            return False
        dates_in_column = column_dates.get((table.id, column.id), ())
        if (
            not trusted_dates
            and dates_in_column
            and (max(dates_in_column) - min(dates_in_column)).days > 180
        ):
            return False
        nearby_support = sum(
            abs((candidate_date - other).days) <= 45
            for other in dates_in_column
        )
        return nearby_support >= 2

    def update(
        canonical_id: str,
        candidate: tuple[str, str, tuple[EvidenceRef, ...]],
        *,
        inherited: bool,
    ) -> bool:
        index = row_indexes.get(canonical_id)
        if index is None or rows[index].role not in {
            RowRole.DETAIL,
            RowRole.REFUND,
            RowRole.CATEGORY_ROLLUP,
        }:
            return False
        raw, iso, evidence = candidate
        updated = _with_grounded_service_date(
            rows[index],
            raw=raw,
            iso=iso,
            evidence=evidence,
            inherited=inherited,
        )
        changed = updated is not rows[index]
        rows[index] = updated
        return changed

    for table in initially_linked:
        if table.table_type not in _SERVICE_DATE_TABLE_TYPES:
            continue
        columns = {column.id: column for column in table.columns}

        # Prefer a date printed on the same grounded billable row.
        for source_row in table.rows:
            if source_row.canonical_row_id is None:
                continue
            canonical_index = row_indexes.get(source_row.canonical_row_id)
            if canonical_index is None:
                continue
            canonical = rows[canonical_index]
            candidates = tuple(
                (cell, candidate)
                for cell in source_row.cells
                if (
                    candidate := _source_cell_service_date(
                        cell,
                        columns[cell.column_id],
                    )
                )
                is not None
                and direct_candidate_is_admissible(
                    table,
                    columns[cell.column_id],
                    candidate,
                    canonical,
                )
            )
            unique_dates = {
                (raw, iso) for _, (raw, iso, _) in candidates
            }
            if len(unique_dates) == 1 and candidates:
                update(
                    source_row.canonical_row_id,
                    candidates[0][1],
                    inherited=False,
                )

        # A transaction/sale date is sometimes printed on its own line directly
        # after the first item. The non-date reference in the same lane, or OCR
        # description evidence shared with the preceding row, proves direction.
        for donor_index, donor_row in enumerate(table.rows):
            if donor_row.canonical_row_id is not None:
                continue
            donor_candidates = tuple(
                (
                    cell,
                    candidate,
                )
                for cell in donor_row.cells
                if (
                    candidate := _source_cell_service_date(
                        cell,
                        columns[cell.column_id],
                    )
                )
                is not None
            )
            if len(donor_candidates) != 1:
                continue
            if any(
                parse_decimal(cell.raw_value or "") is not None
                for cell in donor_row.cells
            ):
                continue
            donor_cell, candidate = donor_candidates[0]
            previous = (
                table.rows[donor_index - 1] if donor_index > 0 else None
            )
            following = (
                table.rows[donor_index + 1]
                if donor_index + 1 < len(table.rows)
                else None
            )
            associated_previous = False
            if previous is not None and previous.canonical_row_id is not None:
                previous_cells = {
                    cell.column_id: cell for cell in previous.cells
                }
                lane_value = (
                    previous_cells[donor_cell.column_id].raw_value or ""
                ).strip()
                previous_row = rows[
                    row_indexes[previous.canonical_row_id]
                ]
                description_ids = {
                    token_id
                    for item in previous_row.field_evidence.get(
                        "description", ()
                    )
                    for token_id in item.token_ids
                }
                donor_other_ids = {
                    token_id
                    for cell in donor_row.cells
                    if cell.column_id != donor_cell.column_id
                    for item in cell.evidence
                    for token_id in item.token_ids
                }
                associated_previous = bool(
                    (
                        lane_value
                        and DATE_SPAN.search(lane_value) is None
                        and parse_decimal(lane_value) is None
                        and any(character.isdigit() for character in lane_value)
                    )
                    or description_ids.intersection(donor_other_ids)
                )
                if associated_previous:
                    update(
                        previous.canonical_row_id,
                        candidate,
                        inherited=True,
                    )

            if associated_previous:
                # Carry within this explicitly delimited transaction group.
                for grouped_row in table.rows[donor_index + 1 :]:
                    grouped_cells = {
                        cell.column_id: cell for cell in grouped_row.cells
                    }
                    lane_value = (
                        grouped_cells[donor_cell.column_id].raw_value or ""
                    ).strip()
                    if lane_value:
                        break
                    if grouped_row.canonical_row_id is not None:
                        update(
                            grouped_row.canonical_row_id,
                            candidate,
                            inherited=True,
                        )
                continue

            # A bare date immediately before one charge applies only to that row.
            if (
                following is not None
                and following.canonical_row_id is not None
                and all(
                    cell.column_id == donor_cell.column_id
                    or not (cell.raw_value or "").strip()
                    for cell in donor_row.cells
                )
            ):
                update(
                    following.canonical_row_id,
                    candidate,
                    inherited=True,
                )
    return rows


def _promote_grounded_date_column(
    table: SourceTable,
    canonical_rows: tuple[CanonicalRow, ...],
) -> SourceTable:
    """Map one unambiguous raw date lane after canonical evidence proves it."""
    date_evidence_ids = {
        token_id
        for row in canonical_rows
        if row.service_date_raw
        for item in row.field_evidence.get("service_date", ())
        for token_id in item.token_ids
    }
    if not date_evidence_ids or table.table_type not in _SERVICE_DATE_TABLE_TYPES:
        return table

    support: Counter[str] = Counter()
    for source_row in table.rows:
        for cell in source_row.cells:
            cell_ids = {
                token_id
                for item in cell.evidence
                for token_id in item.token_ids
            }
            if date_evidence_ids.intersection(cell_ids):
                support[cell.column_id] += 1
    if not support:
        return table
    ordered = support.most_common()
    if ordered[0][1] < 2 or (
        len(ordered) > 1 and ordered[0][1] == ordered[1][1]
    ):
        return table
    selected_id = ordered[0][0]
    selected = next(column for column in table.columns if column.id == selected_id)
    if selected.canonical_field not in {None, "service_date_raw"}:
        return table

    supported_values = tuple(
        cell.raw_value.strip()
        for row in table.rows
        for cell in row.cells
        if cell.column_id == selected_id
        and cell.raw_value
        and cell.raw_value.strip()
        and date_evidence_ids.intersection(
            {
                token_id
                for item in cell.evidence
                for token_id in item.token_ids
            }
        )
    )
    if len(supported_values) < 2 or any(
        service_date_from_context(
            value,
            column_label=selected.label,
            canonical_field=selected.canonical_field,
        )
        is None
        for value in supported_values
    ):
        return table

    columns = tuple(
        column.model_copy(
            update={
                "label": (
                    "Date"
                    if "synthetic_header" in column.validation_flags
                    else column.label
                ),
                "canonical_field": "service_date_raw",
                "validation_flags": tuple(
                    dict.fromkeys(
                        (*column.validation_flags, "inferred_column_role")
                    )
                ),
            }
        )
        if column.id == selected_id
        else column
        for column in table.columns
    )
    return table.model_copy(update={"columns": columns})


def _link_source_tables(
    tables: tuple[SourceTable, ...] | list[SourceTable],
    canonical_rows: tuple[CanonicalRow, ...] | list[CanonicalRow],
) -> tuple[SourceTable, ...]:
    """Link only strict, unique mutual-best source/canonical pairs."""

    def evidence_ids(items: object) -> set[str]:
        return {
            token_id
            for item in items or ()
            for token_id in getattr(item, "token_ids", ())
        }

    def evidence_bounds(items: object) -> tuple[float, float, float, float] | None:
        points = tuple(
            point
            for item in items or ()
            for point in getattr(getattr(item, "polygon", None), "points", ())
        )
        if not points:
            return None
        return (
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    def geometry_score(
        source_items: object,
        candidate_items: object,
    ) -> tuple[int, int, int]:
        source = evidence_bounds(source_items)
        candidate = evidence_bounds(candidate_items)
        if source is None or candidate is None:
            return (0, -10**9, -10**9)
        vertical_overlap = max(0.0, min(source[3], candidate[3]) - max(source[1], candidate[1]))
        minimum_height = max(1.0, min(source[3] - source[1], candidate[3] - candidate[1]))
        vertical_ratio = round(vertical_overlap / minimum_height * 1000)
        vertical_distance = round(
            abs((source[1] + source[3]) / 2 - (candidate[1] + candidate[3]) / 2)
        )
        horizontal_distance = round(
            abs((source[0] + source[2]) / 2 - (candidate[0] + candidate[2]) / 2)
        )
        return (vertical_ratio, -vertical_distance, -horizontal_distance)

    candidates_by_table: dict[tuple[int, str | None], tuple[CanonicalRow, ...]] = {}
    for table in tables:
        candidates_by_table[(table.page_number, table.table_id)] = tuple(
            row
            for row in canonical_rows
            if row.page_number == table.page_number and row.table_id == table.table_id
        )

    scores: dict[tuple[str, str], tuple[int, ...]] = {}
    source_rows: dict[str, tuple[SourceTable, SourceRow]] = {}
    canonical_by_id = {str(row.id): row for row in canonical_rows}
    for table in tables:
        description_column = next(
            (column for column in table.columns if column.canonical_field == "description"),
            None,
        )
        amount_column_ids = {
            column.id
            for column in table.columns
            if column.canonical_field in {"gross_amount", "net_amount"}
        }
        for source_row in table.rows:
            source_key = f"{table.id}:{source_row.id}"
            source_rows[source_key] = (table, source_row)
            source_evidence = tuple(
                evidence for cell in source_row.cells for evidence in cell.evidence
            )
            amount_evidence = tuple(
                evidence
                for cell in source_row.cells
                if cell.column_id in amount_column_ids
                for evidence in cell.evidence
            )
            source_ids = evidence_ids(source_evidence)
            printed_description = next(
                (
                    cell.raw_value or ""
                    for cell in source_row.cells
                    if description_column is not None
                    and cell.column_id == description_column.id
                ),
                "",
            )
            for candidate in candidates_by_table.get(
                (table.page_number, table.table_id), ()
            ):
                description_ids = evidence_ids(
                    candidate.field_evidence.get("description", ())
                )
                if candidate.role in {
                    RowRole.DETAIL,
                    RowRole.REFUND,
                    RowRole.CATEGORY_ROLLUP,
                }:
                    anchor_evidence = candidate.field_evidence.get("amount", ())
                elif candidate.role is RowRole.INFORMATIONAL:
                    anchor_evidence = tuple(
                        evidence
                        for field in ("service_date", "request_no", "service_code", "hsn_code")
                        for evidence in candidate.field_evidence.get(field, ())
                    )
                else:
                    anchor_evidence = candidate.field_evidence.get("description", ())
                anchor_ids = evidence_ids(anchor_evidence)
                anchor_overlap = len(anchor_ids & source_ids)
                if anchor_overlap == 0:
                    continue
                similarity = round(
                    SequenceMatcher(
                        None,
                        re.sub(r"\s+", " ", printed_description.casefold()).strip(),
                        re.sub(r"\s+", " ", (candidate.description or "").casefold()).strip(),
                    ).ratio()
                    * 1000
                )
                source_anchor = amount_evidence or source_evidence
                scores[(source_key, str(candidate.id))] = (
                    anchor_overlap,
                    len(description_ids & source_ids),
                    similarity,
                    *geometry_score(source_anchor, anchor_evidence),
                    len(evidence_ids(candidate.evidence) & source_ids),
                    -abs(source_row.order - candidate.row_order),
                )

    def unique_best(
        values: list[tuple[tuple[int, ...], str]],
    ) -> str | None:
        values.sort(key=lambda item: item[0], reverse=True)
        if not values or (len(values) > 1 and values[0][0] == values[1][0]):
            return None
        return values[0][1]

    source_best: dict[str, str | None] = {}
    for source_key in source_rows:
        source_best[source_key] = unique_best(
            [
                (score, canonical_id)
                for (candidate_source, canonical_id), score in scores.items()
                if candidate_source == source_key
            ]
        )
    canonical_best: dict[str, str | None] = {}
    for canonical_id in canonical_by_id:
        canonical_best[canonical_id] = unique_best(
            [
                (score, source_key)
                for (source_key, candidate_canonical), score in scores.items()
                if candidate_canonical == canonical_id
            ]
        )

    linked: list[SourceTable] = []
    for table in tables:
        rows: list[SourceRow] = []
        for source_row in table.rows:
            source_key = f"{table.id}:{source_row.id}"
            canonical_id = source_best.get(source_key)
            accepted = bool(
                canonical_id is not None
                and canonical_best.get(canonical_id) == source_key
            )
            flags = tuple(
                flag
                for flag in source_row.validation_flags
                if flag != "canonical_link_ambiguous"
            )
            if not accepted and any(key[0] == source_key for key in scores):
                flags = tuple(dict.fromkeys((*flags, "canonical_link_ambiguous")))
            rows.append(
                source_row.model_copy(
                    update={
                        "canonical_row_id": canonical_id if accepted else None,
                        "validation_flags": flags,
                    }
                )
            )
        linked.append(table.model_copy(update={"rows": tuple(rows)}))
    return tuple(linked)



def _finalize_linked_source_cells(
    tables: tuple[SourceTable, ...],
    canonical_rows: tuple[CanonicalRow, ...] | list[CanonicalRow],
) -> tuple[SourceTable, ...]:
    """Apply grounded display cleanup after one-to-one linkage is complete."""
    canonical = {str(row.id): row for row in canonical_rows}
    finalized: list[SourceTable] = []
    for table in tables:
        columns_by_id = {column.id: column for column in table.columns}
        rows: list[SourceRow] = []
        for source_row in table.rows:
            matched = canonical.get(source_row.canonical_row_id or "")
            if matched is None:
                rows.append(source_row)
                continue
            cells = _split_grounded_date_request_description(
                source_row.cells,
                table.columns,
                matched,
            )
            cells = _redistribute_grounded_description_from_adjacent_cell(
                cells,
                table.columns,
                matched,
            )
            cells = _trim_grounded_duplicate_adjacent_description(
                cells,
                table.columns,
                matched,
            )
            cells = _split_grounded_date_from_description(
                cells,
                table.columns,
                matched,
            )
            cells = _populate_grounded_service_date_cell(
                cells,
                table.columns,
                matched,
            )
            cells = tuple(
                cell.model_copy(
                    update={
                        "raw_value": None,
                        "evidence": (),
                        "validation_flags": tuple(
                            dict.fromkeys(
                                (*cell.validation_flags, "excluded_oversized_overlay")
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
                for cell in cells
            )
            rows.append(source_row.model_copy(update={"cells": cells}))
        linked_table = table.model_copy(update={"rows": tuple(rows)})
        finalized.append(
            _consolidate_grounded_adjacent_descriptions(
                linked_table,
                tuple(
                    row
                    for row in canonical_rows
                    if row.page_number == table.page_number
                    and row.table_id == table.table_id
                ),
            )
        )
    return tuple(finalized)


def _finalize_linked_service_dates(
    tables: tuple[SourceTable, ...],
    canonical_rows: list[CanonicalRow],
) -> tuple[SourceTable, ...]:
    canonical = {str(row.id): row for row in canonical_rows}
    finalized: list[SourceTable] = []
    for original in tables:
        candidates = tuple(
            row
            for row in canonical_rows
            if row.page_number == original.page_number
            and row.table_id == original.table_id
        )
        table = _promote_grounded_date_column(original, candidates)
        rows = tuple(
            source_row.model_copy(
                update={
                    "cells": _populate_grounded_service_date_cell(
                        source_row.cells,
                        table.columns,
                        canonical[source_row.canonical_row_id],
                    )
                }
            )
            if source_row.canonical_row_id in canonical
            else source_row
            for source_row in table.rows
        )
        finalized.append(table.model_copy(update={"rows": rows}))
    return tuple(finalized)


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


def _flag_possible_supporting_receipt_duplicates(
    rows: list[CanonicalRow],
    source_tables: tuple[SourceTable, ...],
) -> tuple[list[CanonicalRow], list[dict[str, Any]]]:
    """Keep billable receipts visible while making possible duplicates reviewable."""
    output = list(rows)
    duplicate_pairs: list[dict[str, Any]] = []
    generic_words = {
        "amount",
        "bill",
        "charge",
        "charges",
        "collection",
        "receipt",
        "towards",
    }

    def normalized(value: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()

    canonical_to_source = {
        source_row.canonical_row_id: (table.table_id, source_row.id)
        for table in source_tables
        for source_row in table.rows
        if source_row.canonical_row_id
    }
    source_by_canonical = {
        source_row.canonical_row_id: source_row
        for table in source_tables
        for source_row in table.rows
        if source_row.canonical_row_id
    }

    def issuer_context(row: CanonicalRow) -> str:
        source_row = source_by_canonical.get(str(row.id))
        if source_row is not None and source_row.receipt_metadata is not None:
            issuer = normalized(source_row.receipt_metadata.issuer_normalized)
            if issuer:
                return issuer
        section = normalized(row.section)
        if section and section not in {"supporting receipt", "receipt", "charges"}:
            return section
        words = [
            word
            for word in normalized(row.description).split()
            if word not in generic_words and not word.isdigit()
        ]
        return " ".join(words[:3])

    def reference_context(row: CanonicalRow) -> str:
        source_row = source_by_canonical.get(str(row.id))
        if source_row is not None and source_row.receipt_metadata is not None:
            reference = normalized(source_row.receipt_metadata.reference_normalized)
            if reference:
                return reference
        return normalized(row.request_no)

    seen_pairs: set[tuple[str, str]] = set()

    for index, receipt in enumerate(output):
        if not (
            "supporting_receipt_charge" in receipt.validation_flags
            or "receipt_form" in receipt.source_routes
        ):
            continue
        receipt_description = normalized(receipt.description)
        possible_duplicate = False
        for other_index, other in enumerate(output):
            if other_index == index:
                continue
            pair_key = tuple(sorted((str(receipt.id), str(other.id))))
            if pair_key in seen_pairs:
                continue
            issuers_compatible = bool(
                issuer_context(receipt)
                and issuer_context(receipt) == issuer_context(other)
            )
            reference_match = bool(
                reference_context(receipt)
                and reference_context(receipt) == reference_context(other)
                and issuers_compatible
            )
            descriptions_match = (
                SequenceMatcher(
                    None,
                    receipt_description,
                    normalized(other.description),
                ).ratio()
                >= 0.80
            )
            dated_amount_match = bool(
                receipt.net_amount is not None
                and receipt.net_amount == other.net_amount
                and receipt.service_date_iso
                and receipt.service_date_iso == other.service_date_iso
                and descriptions_match
                and issuers_compatible
                and abs(receipt.page_number - other.page_number) <= 1
            )
            if not (reference_match or dated_amount_match):
                continue
            possible_duplicate = True
            seen_pairs.add(pair_key)
            source_refs = tuple(
                reference[1]
                for row_id in pair_key
                if (reference := canonical_to_source.get(row_id)) is not None
            )
            table_refs = tuple(
                reference[0]
                for row_id in pair_key
                if (reference := canonical_to_source.get(row_id)) is not None
            )
            duplicate_pairs.append(
                {
                    "canonical_row_ids": pair_key,
                    "source_row_ids": source_refs,
                    "page_number": min(receipt.page_number, other.page_number),
                    "table_id": table_refs[0] if len(set(table_refs)) == 1 else None,
                    "match_basis": (
                        "exact_reference" if reference_match else "dated_grounded_charge"
                    ),
                }
            )
        if possible_duplicate:
            output[index] = receipt.model_copy(
                update={
                    "validation_flags": tuple(
                        dict.fromkeys(
                            (
                                *receipt.validation_flags,
                                "possible_duplicate_supporting_charge",
                            )
                        )
                    )
                }
            )
    return output, duplicate_pairs


def _attach_receipt_source_metadata(
    tables: tuple[SourceTable, ...],
    canonical_rows: list[CanonicalRow],
) -> tuple[SourceTable, ...]:
    canonical = {str(row.id): row for row in canonical_rows}
    output: list[SourceTable] = []
    for table in tables:
        columns = {column.id: column for column in table.columns}
        rows: list[SourceRow] = []
        for source_row in table.rows:
            row = canonical.get(source_row.canonical_row_id or "")
            if row is None or not (
                "supporting_receipt_charge" in row.validation_flags
                or "receipt_form" in row.source_routes
            ):
                rows.append(source_row)
                continue
            reference_cell = next(
                (
                    cell
                    for cell in source_row.cells
                    if columns[cell.column_id].canonical_field == "request_no"
                    and (cell.raw_value or "").strip()
                ),
                None,
            )
            issuer_cell = next(
                (
                    cell
                    for cell in source_row.cells
                    if any(
                        marker in re.sub(
                            r"[^a-z0-9]+",
                            " ",
                            columns[cell.column_id].label.casefold(),
                        ).split()
                        for marker in ("issuer", "hospital", "provider", "payee")
                    )
                    and (cell.raw_value or "").strip()
                    and bool(cell.evidence)
                ),
                None,
            )
            issuer_raw = re.sub(
                r"\s+", " ", (issuer_cell.raw_value if issuer_cell else "")
            ).strip()
            reference_raw = re.sub(
                r"\s+", " ", (reference_cell.raw_value if reference_cell else "")
            ).strip()
            rows.append(
                source_row.model_copy(
                    update={
                        "receipt_metadata": ReceiptSourceMetadata(
                            issuer_raw=issuer_raw or None,
                            issuer_normalized=re.sub(
                                r"[^a-z0-9]+", " ", issuer_raw.casefold()
                            ).strip()
                            or None,
                            reference_raw=reference_raw or None,
                            reference_normalized=re.sub(
                                r"[^a-z0-9]+", " ", reference_raw.casefold()
                            ).strip()
                            or None,
                        )
                    }
                )
            )
        output.append(table.model_copy(update={"rows": tuple(rows)}))
    return tuple(output)


def _materialize_printed_cell_fragments(
    tables: tuple[SourceTable, ...],
    token_lookup: dict[str, TokenManifestEntry],
) -> tuple[
    tuple[SourceTable, ...],
    tuple[TokenManifestEntry, ...],
    dict[tuple[str, str, str], str],
]:
    """Give every mapped Printed value one exact, purpose-built fragment."""
    fragments: dict[str, TokenManifestEntry] = {}
    assignments: dict[tuple[str, str, str], str] = {}
    occupied: dict[str, list[tuple[int, int]]] = {}
    output: list[SourceTable] = []

    def normalized(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    def bounds(polygon: Polygon) -> tuple[float, float, float, float]:
        return (
            min(point.x for point in polygon.points),
            min(point.y for point in polygon.points),
            max(point.x for point in polygon.points),
            max(point.y for point in polygon.points),
        )

    def rectangle(left: float, top: float, right: float, bottom: float) -> Polygon:
        return Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        )

    def span_polygon(token: TokenManifestEntry, start: int, end: int) -> Polygon:
        left, top, right, bottom = bounds(token.polygon)
        length = max(1, len(token.text))
        width = right - left
        return rectangle(
            left + width * start / length,
            top,
            left + width * end / length,
            bottom,
        )

    for table in tables:
        columns = {column.id: column for column in table.columns}
        column_centers: dict[str, float] = {}
        for column in table.columns:
            polygons = tuple(
                evidence.polygon
                for row in table.rows
                for cell in row.cells
                if cell.column_id == column.id
                for evidence in cell.evidence
            )
            if not polygons:
                polygons = tuple(item.polygon for item in column.evidence)
            if polygons:
                column_centers[column.id] = median(
                    (bounds(polygon)[0] + bounds(polygon)[2]) / 2
                    for polygon in polygons
                )
        materialized_rows: list[SourceRow] = []
        for source_row in table.rows:
            materialized_cells: list[SourceCell] = []
            for cell in source_row.cells:
                raw = (cell.raw_value or "").strip()
                canonical_role = columns[cell.column_id].canonical_field
                role = canonical_role or f"source:{cell.column_id}"
                if not raw:
                    materialized_cells.append(cell)
                    continue
                aligned_role = {
                    "service_date_raw": "service_date",
                    "unit_price": "rate",
                    "net_amount": "amount",
                }.get(role, role)
                parent_ids = tuple(
                    dict.fromkeys(
                        token_id
                        for evidence in cell.evidence
                        for token_id in evidence.token_ids
                    )
                )
                parents = tuple(
                    token_lookup[token_id]
                    for token_id in parent_ids
                    if token_id in token_lookup
                    and token_lookup[token_id].parent_token_id is None
                    and not token_lookup[token_id].parent_token_ids
                )
                selected_spans: tuple[tuple[int, int], ...] | None = None
                selected_parents: tuple[TokenManifestEntry, ...] = ()
                fragment_polygon: Polygon | None = None
                if parents and normalized(
                    " ".join(item.text for item in parents)
                ) == normalized(raw):
                    selected_parents = parents
                    selected_spans = tuple((0, len(item.text)) for item in parents)
                    all_bounds = tuple(bounds(item.polygon) for item in parents)
                    fragment_polygon = rectangle(
                        min(item[0] for item in all_bounds),
                        min(item[1] for item in all_bounds),
                        max(item[2] for item in all_bounds),
                        max(item[3] for item in all_bounds),
                    )
                elif len(parents) == 1:
                    token = parents[0]
                    occurrences = tuple(
                        match.span()
                        for match in re.finditer(re.escape(raw), token.text, re.IGNORECASE)
                        if not any(
                            match.start() < used_end and match.end() > used_start
                            for used_start, used_end in occupied.get(token.token_id, ())
                        )
                    )
                    if occurrences:
                        target = column_centers.get(cell.column_id)
                        scored = sorted(
                            (
                                abs(
                                    (
                                        bounds(span_polygon(token, start, end))[0]
                                        + bounds(span_polygon(token, start, end))[2]
                                    )
                                    / 2
                                    - target
                                )
                                if target is not None
                                else 0.0,
                                start,
                                end,
                            )
                            for start, end in occurrences
                        )
                        unique = len(scored) == 1 or scored[1][0] - scored[0][0] > 1.0
                        if unique:
                            _, start, end = scored[0]
                            selected_parents = (token,)
                            selected_spans = ((start, end),)
                            fragment_polygon = span_polygon(token, start, end)
                            occupied.setdefault(token.token_id, []).append((start, end))
                if selected_spans is None or fragment_polygon is None:
                    cell = cell.model_copy(
                        update={
                            "validation_flags": tuple(
                                dict.fromkeys(
                                    (*cell.validation_flags, "fragment_occurrence_ambiguous")
                                )
                            )
                        }
                    )
                    materialized_cells.append(cell)
                    continue
                identity = hashlib.sha256(
                    json.dumps(
                        {
                            "parents": [item.token_id for item in selected_parents],
                            "spans": selected_spans,
                            "role": aligned_role,
                            "value": raw,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()[:24]
                fragment_id = f"fragment-{identity}"
                first = selected_parents[0]
                if len(selected_parents) == 1:
                    fragment_kwargs: dict[str, Any] = {
                        "parent_token_id": first.token_id,
                        "character_start": selected_spans[0][0],
                        "character_end": selected_spans[0][1],
                    }
                else:
                    fragment_kwargs = {
                        "parent_token_ids": tuple(
                            item.token_id for item in selected_parents
                        ),
                        "parent_character_spans": selected_spans,
                    }
                fragments[fragment_id] = TokenManifestEntry(
                    token_id=fragment_id,
                    page_number=first.page_number,
                    table_ids=tuple(sorted({*first.table_ids, table.table_id})),
                    text=raw,
                    polygon=fragment_polygon,
                    artifact_sha256=first.artifact_sha256,
                    artifact_relative_path=first.artifact_relative_path,
                    confidence=min(item.confidence for item in selected_parents),
                    fragment_role=aligned_role,
                    **fragment_kwargs,
                )
                for parent in selected_parents:
                    assignments[(source_row.id, aligned_role, parent.token_id)] = fragment_id
                first_evidence = cell.evidence[0]
                cell = cell.model_copy(
                    update={
                        "evidence": (
                            first_evidence.model_copy(
                                update={
                                    "polygon": fragment_polygon,
                                    "token_ids": (fragment_id,),
                                }
                            ),
                        )
                    }
                )
                materialized_cells.append(cell)
            materialized_rows.append(
                source_row.model_copy(update={"cells": tuple(materialized_cells)})
            )
        output.append(table.model_copy(update={"rows": tuple(materialized_rows)}))
    return tuple(output), tuple(fragments.values()), assignments


def _apply_fragment_ids_to_aligned_rows(
    rows: tuple[AlignedLedgerRow, ...],
    tables: tuple[SourceTable, ...],
    assignments: dict[tuple[str, str, str], str],
) -> tuple[AlignedLedgerRow, ...]:
    source_rows = tuple(row for table in tables for row in table.rows)
    output: list[AlignedLedgerRow] = []
    for aligned in rows:
        evidence_ids = set(aligned.evidence_token_ids)
        scored = sorted(
            (
                (
                    sum(
                        (source_row.id, field, token_id) in assignments
                        for field, token_ids in aligned.field_token_ids.items()
                        for token_id in token_ids
                        if token_id in evidence_ids
                    ),
                    source_row,
                )
                for source_row in source_rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scored or scored[0][0] == 0 or (
            len(scored) > 1 and scored[0][0] == scored[1][0]
        ):
            output.append(aligned)
            continue
        source_row = scored[0][1]
        updated = {
            field: tuple(
                assignments.get((source_row.id, field, token_id), token_id)
                for token_id in token_ids
            )
            for field, token_ids in aligned.field_token_ids.items()
        }
        output.append(replace(aligned, field_token_ids=updated))
    return tuple(output)


def _synchronize_canonical_from_printed(
    tables: tuple[SourceTable, ...],
    rows: list[CanonicalRow],
) -> list[CanonicalRow]:
    """Make final Printed fragments authoritative for values and field evidence."""
    by_id = {str(row.id): row for row in rows}
    updates_by_id: dict[str, dict[str, Any]] = {}
    evidence_names = {
        "description": "description",
        "service_date_raw": "service_date",
        "request_no": "request_no",
        "quantity": "quantity",
        "unit_price": "rate",
        "gross_amount": "gross_amount",
        "discount": "discount",
        "net_amount": "amount",
    }
    raw_names = {
        "quantity": "quantity_raw",
        "unit_price": "unit_price_raw",
        "gross_amount": "gross_amount_raw",
        "discount": "discount_raw",
        "net_amount": "net_amount_raw",
    }
    for table in tables:
        columns = {column.id: column for column in table.columns}
        for source_row in table.rows:
            canonical_id = source_row.canonical_row_id
            row = by_id.get(canonical_id or "")
            if row is None:
                continue
            updates = updates_by_id.setdefault(
                str(row.id),
                {"field_evidence": dict(row.field_evidence)},
            )
            for cell in source_row.cells:
                field = columns[cell.column_id].canonical_field
                raw = (cell.raw_value or "").strip()
                if field not in evidence_names or not raw or not cell.evidence:
                    continue
                if field == "description":
                    updates["description"] = raw
                elif field == "request_no":
                    if _structured_field_value_is_valid("request_no", raw):
                        updates["request_no"] = raw
                    else:
                        continue
                elif field == "service_date_raw":
                    parsed = service_date_from_context(
                        raw,
                        column_label=columns[cell.column_id].label,
                        canonical_field="service_date_raw",
                        description=str(updates.get("description") or row.description or ""),
                    )
                    if parsed is None:
                        continue
                    updates["service_date_raw"], updates["service_date_iso"] = parsed
                else:
                    parsed_number = (
                        parse_quantity(raw) if field == "quantity" else parse_decimal(raw)
                    )
                    if parsed_number is None:
                        continue
                    updates[field] = parsed_number
                    updates[raw_names[field]] = raw
                updates["field_evidence"][evidence_names[field]] = cell.evidence
    return [
        row.model_copy(update=updates_by_id.get(str(row.id), {}))
        for row in rows
    ]


def _attach_derived_field_provenance(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    output: list[CanonicalRow] = []
    for row in rows:
        if "quantity_derived_from_rate_amount" not in row.validation_flags:
            output.append(row)
            continue
        if row.quantity is None or row.unit_price is None or row.net_amount is None:
            output.append(row)
            continue
        fields: list[str] = ["net_amount"]
        values: list[Decimal] = [row.net_amount]
        evidence_names: list[str] = ["amount"]
        if row.discount is not None:
            fields.append("discount")
            values.append(row.discount)
            evidence_names.append("discount")
        fields.append("unit_price")
        values.append(row.unit_price)
        evidence_names.append("rate")
        token_groups = tuple(
            tuple(
                token_id
                for evidence in row.field_evidence.get(evidence_name, ())
                for token_id in evidence.token_ids
            )
            for evidence_name in evidence_names
        )
        if any(not token_ids for token_ids in token_groups):
            output.append(row)
            continue
        provenance = DerivedFieldProvenance(
            operation="absolute_net_plus_discount_divided_by_unit_price",
            operand_fields=tuple(fields),
            operand_values=tuple(values),
            operand_evidence_token_ids=token_groups,
            result=row.quantity,
        )
        output.append(
            row.model_copy(
                update={
                    "derived_fields": {**row.derived_fields, "quantity": provenance},
                    "field_evidence": {
                        key: value
                        for key, value in row.field_evidence.items()
                        if key != "quantity"
                    },
                    "quantity_raw": None,
                }
            )
        )
    return output


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


class ExtractionAborted(RuntimeError):
    """Raised at a safe extraction checkpoint after a user abort request."""


def _stable_payload_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _evidence_bounds(evidence: Collection[EvidenceRef]) -> tuple[float, float, float, float] | None:
    points = tuple(point for item in evidence for point in item.polygon.points)
    if not points:
        return None
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _quantized_geometry(
    bounds: tuple[float, float, float, float] | None,
    page: PageAsset,
) -> tuple[int, int, int, int]:
    if bounds is None:
        return (0, 0, 0, 0)
    left, top, right, bottom = bounds
    return tuple(
        int(round(value * 1000))
        for value in (
            left / page.width,
            top / page.height,
            right / page.width,
            bottom / page.height,
        )
    )


def _anchor(kind: str, *parts: object) -> str:
    return f"{kind}-{_stable_payload_digest(parts)[:24]}"


def _assign_geometry_anchors(
    tables: tuple[SourceTable, ...],
    rows: list[CanonicalRow],
    page_assets: Collection[PageAsset],
) -> tuple[tuple[SourceTable, ...], list[CanonicalRow]]:
    assets = {page.page_number: page for page in page_assets}
    ordered_tables: dict[
        int,
        list[tuple[SourceTable, tuple[float, float, float, float] | None]],
    ] = {}
    for table in tables:
        evidence = tuple(
            item
            for column in table.columns
            for item in column.evidence
        ) + tuple(
            item
            for source_row in table.rows
            for cell in source_row.cells
            for item in cell.evidence
        )
        ordered_tables.setdefault(table.page_number, []).append(
            (table, _evidence_bounds(evidence))
        )
    anchored_tables: list[SourceTable] = []
    canonical_anchors: dict[str, str] = {}
    for page_number, candidates in sorted(ordered_tables.items()):
        page = assets[page_number]
        for ordinal, (table, bounds) in enumerate(
            sorted(candidates, key=lambda item: item[1] or (0, 0, 0, 0))
        ):
            table_anchor = _anchor(
                "table",
                page_number,
                _quantized_geometry(bounds, page),
                ordinal,
            )
            anchored_rows: list[SourceRow] = []
            for source_row in table.rows:
                row_bounds = _evidence_bounds(
                    tuple(item for cell in source_row.cells for item in cell.evidence)
                )
                row_anchor = _anchor(
                    "row",
                    table_anchor,
                    _quantized_geometry(row_bounds, page),
                    source_row.order,
                )
                anchored_rows.append(source_row.model_copy(update={"row_anchor": row_anchor}))
                if source_row.canonical_row_id:
                    canonical_anchors[source_row.canonical_row_id] = row_anchor
            anchored_tables.append(
                table.model_copy(
                    update={"table_anchor": table_anchor, "rows": tuple(anchored_rows)}
                )
            )
    anchored_canonical: list[CanonicalRow] = []
    for row in rows:
        row_anchor = canonical_anchors.get(str(row.id))
        if row_anchor is None:
            page = assets[row.page_number]
            row_anchor = _anchor(
                "row",
                row.page_number,
                _quantized_geometry(_evidence_bounds(row.evidence), page),
                row.row_order,
            )
        anchored_canonical.append(row.model_copy(update={"row_anchor": row_anchor}))
    return tuple(anchored_tables), anchored_canonical


def _raw_total_candidate(
    candidate: DocumentTotalCandidate,
    tables: Collection[SourceTable],
) -> RawTotalCandidate:
    table = next(
        (
            item
            for item in tables
            if item.page_number == candidate.total.page_number
            and item.table_id == candidate.total.evidence.table_id
        ),
        None,
    )
    context_id = candidate.total.context_id or ""
    ordinal_match = re.search(r":o(?P<ordinal>\d+)$", context_id)
    payload = {
        "total": candidate.total.model_dump(mode="json"),
        "label_priority": candidate.label_priority,
        "vertical_position": candidate.vertical_position,
        "local_context": candidate.local_context,
        "table_anchor": table.table_anchor if table else None,
    }
    return RawTotalCandidate(
        candidate_id=f"total-{_stable_payload_digest(payload)[:24]}",
        total=candidate.total,
        label_priority=candidate.label_priority,
        vertical_position=candidate.vertical_position,
        local_context=candidate.local_context,
        page_number=candidate.total.page_number,
        table_id=table.table_id if table else candidate.total.evidence.table_id,
        table_anchor=table.table_anchor if table else None,
        table_type=table.table_type if table else None,
        region_kind="table" if table else "page_summary",
        summary_block_ordinal=(
            int(ordinal_match.group("ordinal")) if ordinal_match else 0
        ),
        context_evidence=(candidate.total.evidence,),
    )


def _provider_usage_leaf(
    *,
    gemini_mode: str,
    gemini_allowed: bool,
    gemini_calls: int,
    gemini_cost: Decimal,
    disabled_reason: str | None,
    promotion_sha256: str | None,
) -> dict[str, Any]:
    return {
        "gemini_mode": gemini_mode,
        "gemini_allowed": gemini_allowed,
        "gemini_calls": gemini_calls,
        "gemini_measured_cost_usd": str(gemini_cost),
        "gemini_provider_disabled_reason": disabled_reason,
        "gemini_promotion_manifest_sha256": promotion_sha256,
    }


def _provider_usage_payload(
    *,
    baseline_usage: dict[str, Any] | None,
    gemini_mode: str,
    gemini_allowed: bool,
    gemini_calls: int,
    gemini_cost: Decimal,
    disabled_reason: str | None,
    promotion_sha256: str | None,
) -> dict[str, Any]:
    current = _provider_usage_leaf(
        gemini_mode=gemini_mode,
        gemini_allowed=gemini_allowed,
        gemini_calls=gemini_calls,
        gemini_cost=gemini_cost,
        disabled_reason=disabled_reason,
        promotion_sha256=promotion_sha256,
    )
    if baseline_usage is None:
        initial = current
        recovery = _provider_usage_leaf(
            gemini_mode=gemini_mode,
            gemini_allowed=False,
            gemini_calls=0,
            gemini_cost=Decimal("0"),
            disabled_reason="not_attempted",
            promotion_sha256=promotion_sha256,
        )
    else:
        initial = dict(baseline_usage.get("initial") or baseline_usage)
        recovery = current
    initial_calls = int(initial.get("gemini_calls") or 0)
    recovery_calls = int(recovery.get("gemini_calls") or 0)
    initial_cost = Decimal(str(initial.get("gemini_measured_cost_usd") or "0"))
    recovery_cost = Decimal(str(recovery.get("gemini_measured_cost_usd") or "0"))
    return {
        "initial": initial,
        "recovery": recovery,
        "aggregate": {
            "gemini_calls": initial_calls + recovery_calls,
            "gemini_measured_cost_usd": str(initial_cost + recovery_cost),
        },
    }


def _recovery_metadata(
    baseline_draft: ExtractionDraft | None,
    targets: set[tuple[int, str | None]],
    source_tables: tuple[SourceTable, ...],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    if baseline_draft is None:
        return {
            "attempted": False,
            "targets": [],
            "untargeted_units_sha256": None,
        }
    baseline_tables = {
        (table.page_number, table.table_id): table.model_dump(mode="json")
        for unit in baseline_draft.page_units
        for table in unit.source_tables
    }
    selected_tables = {
        (table.page_number, table.table_id): table.model_dump(mode="json")
        for table in source_tables
    }
    target_records: list[dict[str, Any]] = []
    for page_number, table_id in sorted(targets, key=lambda item: (item[0], item[1] or "")):
        baseline_units = (
            [
                value
                for (page, identity), value in baseline_tables.items()
                if page == page_number and (table_id is None or identity == table_id)
            ]
        )
        candidate_units = (
            [
                value
                for (page, identity), value in selected_tables.items()
                if page == page_number and (table_id is None or identity == table_id)
            ]
        )
        status = next(
            (
                str(item.get("status"))
                for item in diagnostics
                if item.get("page_number") == page_number
                and (table_id is None or item.get("table_id") == table_id)
                and item.get("status") in {
                    "recovery_target_not_located",
                    "recovery_no_safe_improvement",
                }
            ),
            None,
        )
        selected = "baseline" if status or not candidate_units else "candidate"
        selected_units = baseline_units if selected == "baseline" else candidate_units
        target_records.append(
            {
                "page_number": page_number,
                "table_id": table_id,
                "baseline_unit_sha256": _stable_payload_digest(baseline_units),
                "candidate_unit_sha256": (
                    _stable_payload_digest(candidate_units) if candidate_units else None
                ),
                "selected": selected,
                "status": (
                    status
                    or ("recovered" if selected == "candidate" else "recovery_no_safe_improvement")
                ),
                "selected_unit_sha256": _stable_payload_digest(selected_units),
                "removed_issue_ids": [],
            }
        )
    untargeted = [
        value
        for (page, table_id), value in baseline_tables.items()
        if (page, table_id) not in targets and (page, None) not in targets
    ]
    return {
        "attempted": True,
        "targets": target_records,
        "untargeted_units_sha256": _stable_payload_digest(untargeted),
    }


@dataclass(frozen=True)
class TableExtractionUnit:
    """Raw table/crop reconstruction retained before document projection."""

    page_number: int
    table_id: str
    source_table: SourceTable
    row_candidates: tuple[CanonicalRow, ...]
    crop_relative_path: str | None
    crop_box: tuple[int, int, int, int] | None
    diagnostics: tuple[dict[str, Any], ...]
    normalized_fragments: tuple[TokenManifestEntry, ...]
    recovery_tokens: tuple[TokenManifestEntry, ...]
    raw_total_candidates: tuple[DocumentTotalCandidate, ...]
    provider_usage: dict[str, Any]

    def raw_digest(self) -> str:
        return _stable_payload_digest(
            {
                "page_number": self.page_number,
                "table_id": self.table_id,
                "source_table": self.source_table.model_dump(mode="json"),
                "row_candidates": [
                    row.model_dump(mode="json") for row in self.row_candidates
                ],
                "crop_relative_path": self.crop_relative_path,
                "crop_box": self.crop_box,
                "diagnostics": self.diagnostics,
                "normalized_fragments": [
                    item.model_dump(mode="json") for item in self.normalized_fragments
                ],
                "recovery_tokens": [
                    item.model_dump(mode="json") for item in self.recovery_tokens
                ],
                "raw_total_candidates": [
                    {
                        "total": item.total.model_dump(mode="json"),
                        "label_priority": item.label_priority,
                        "vertical_position": item.vertical_position,
                        "local_context": item.local_context,
                    }
                    for item in self.raw_total_candidates
                ],
                "provider_usage": self.provider_usage,
            }
        )


@dataclass(frozen=True)
class PageExtractionUnit:
    """Raw page substrate; it contains no linked or certified public rows."""

    page_asset: PageAsset
    ocr_tokens: tuple[OcrToken, ...]
    token_manifest: tuple[TokenManifestEntry, ...]
    table_units: tuple[TableExtractionUnit, ...]
    unassigned_row_candidates: tuple[CanonicalRow, ...]
    diagnostics: tuple[dict[str, Any], ...]
    total_candidates: tuple[DocumentTotalCandidate, ...]
    provider_usage: dict[str, Any]

    @property
    def source_tables(self) -> tuple[SourceTable, ...]:
        return tuple(unit.source_table for unit in self.table_units)

    @property
    def canonical_rows(self) -> tuple[CanonicalRow, ...]:
        return (
            *(row for unit in self.table_units for row in unit.row_candidates),
            *self.unassigned_row_candidates,
        )

    def raw_digest(self) -> str:
        payload = {
            "page_asset": self.page_asset.model_dump(mode="json"),
            "ocr_tokens": [item.model_dump(mode="json") for item in self.ocr_tokens],
            "token_manifest": [
                item.model_dump(mode="json") for item in self.token_manifest
            ],
            "source_tables": [
                unit.raw_digest() for unit in self.table_units
            ],
            "unassigned_row_candidates": [
                row.model_dump(mode="json") for row in self.unassigned_row_candidates
            ],
            "diagnostics": list(self.diagnostics),
            "total_candidates": [
                {
                    "total": item.total.model_dump(mode="json"),
                    "label_priority": item.label_priority,
                    "vertical_position": item.vertical_position,
                    "local_context": item.local_context,
                }
                for item in self.total_candidates
            ],
            "provider_usage": self.provider_usage,
        }
        return _stable_payload_digest(payload)


@dataclass(frozen=True)
class ExtractionDraft:
    """Internal OCR draft projected only when a public result is requested."""

    document_id: str
    source_sha256: str
    source_name: str
    page_units: tuple[PageExtractionUnit, ...]
    provider_usage: dict[str, Any]
    hospital: dict[str, Any] | None
    hospital_id: str | None
    alias_registry_revision: int | None
    profile_registry_revision: int | None
    applied_alias_ids: tuple[str, ...]
    suppressed_repeated_source_tables: tuple[tuple[int, str], ...]
    recovery_metadata: dict[str, Any]
    worker_release_revision: str | None = None
    validation_recovery_attempted: bool | None = None

    @property
    def result(self) -> dict[str, Any]:
        return _project_extraction_draft(self)

    @property
    def raw_unit_sha256(self) -> dict[str, str]:
        return {
            f"page:{unit.page_asset.page_number}": unit.raw_digest()
            for unit in self.page_units
        }


def _project_extraction_draft(draft: ExtractionDraft) -> dict[str, Any]:
    source_tables = tuple(
        table for unit in draft.page_units for table in unit.source_tables
    )
    rows = _apply_document_role_policy(
        _deduplicate(
            [row for unit in draft.page_units for row in unit.canonical_rows]
        )
    )
    source_tables = _link_source_tables(source_tables, rows)
    canonical_by_id = {str(row.id): row for row in rows}
    source_tables = tuple(
        _promote_grounded_date_column(
            table,
            tuple(
                canonical_by_id[source_row.canonical_row_id]
                for source_row in table.rows
                if source_row.canonical_row_id in canonical_by_id
                and canonical_by_id[source_row.canonical_row_id].role
                in {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}
            ),
        )
        for table in source_tables
    )
    rows = _recover_grounded_service_dates(source_tables, rows, already_linked=True)
    rows = _synchronize_canonical_from_printed(source_tables, rows)
    rows = _attach_derived_field_provenance(rows)
    source_tables = _attach_receipt_source_metadata(source_tables, rows)
    source_tables, rows = _assign_geometry_anchors(
        source_tables,
        rows,
        tuple(unit.page_asset for unit in draft.page_units),
    )
    rows, receipt_duplicate_pairs = _flag_possible_supporting_receipt_duplicates(
        rows,
        source_tables,
    )
    total_candidates = [
        candidate for unit in draft.page_units for candidate in unit.total_candidates
    ]
    document_totals = select_document_totals(total_candidates)
    document_total = select_document_total(total_candidates)
    raw_total_candidates = tuple(
        _raw_total_candidate(candidate, source_tables) for candidate in total_candidates
    )
    token_manifest = tuple(
        {
            token.token_id: token
            for unit in draft.page_units
            for token in unit.token_manifest
        }.values()
    )
    diagnostics = [
        deepcopy(item) for unit in draft.page_units for item in unit.diagnostics
    ]
    result: dict[str, Any] = {
        "output_version": "offline_accuracy_spine_v5",
        "contract_revision": 2,
        "document_total_version": DOCUMENT_TOTAL_VERSION,
        "document_totals_version": DOCUMENT_TOTALS_VERSION,
        "document_total": (
            document_total.model_dump(mode="json") if document_total else None
        ),
        "document_totals": [item.model_dump(mode="json") for item in document_totals],
        "raw_total_candidates": [
            item.model_dump(mode="json") for item in raw_total_candidates
        ],
        "document_id": draft.document_id,
        "hospital_id": draft.hospital_id,
        "hospital": deepcopy(draft.hospital),
        "alias_registry_revision": draft.alias_registry_revision,
        "profile_registry_revision": draft.profile_registry_revision,
        "applied_alias_ids": list(draft.applied_alias_ids),
        "source_sha256": draft.source_sha256,
        "source_name": draft.source_name,
        "pages": len(draft.page_units),
        "page_assets": [
            unit.page_asset.model_dump(mode="json") for unit in draft.page_units
        ],
        "source_tables": [table.model_dump(mode="json") for table in source_tables],
        "token_manifest": [token.model_dump(mode="json") for token in token_manifest],
        "suppressed_repeated_source_tables": [
            {"page_number": page, "table_id": table_id}
            for page, table_id in draft.suppressed_repeated_source_tables
        ],
        "rows": [row.model_dump(mode="json") for row in rows],
        "receipt_duplicate_pairs": receipt_duplicate_pairs,
        "diagnostics": diagnostics,
        "provider_usage": deepcopy(draft.provider_usage),
        "recovery": deepcopy(draft.recovery_metadata),
    }
    if draft.worker_release_revision is not None:
        result["worker_release_revision"] = draft.worker_release_revision
    if draft.validation_recovery_attempted is not None:
        result["validation_recovery_attempted"] = draft.validation_recovery_attempted
    return result


def _issue_semantic_key(issue: object) -> tuple[object, ...]:
    return (
        getattr(issue, "id", None),
        getattr(issue, "code", None),
        getattr(issue, "severity", None),
        getattr(issue, "category", None),
        getattr(issue, "page_number", None),
        getattr(issue, "table_id", None),
        getattr(issue, "table_anchor", None),
        getattr(issue, "row_anchor", None),
        getattr(issue, "source_row_id", None),
        getattr(issue, "canonical_row_id", None),
        getattr(issue, "field", None),
    )


def _issue_is_in_recovery_target(
    issue: object,
    targets: Collection[tuple[int, str | None]],
) -> bool:
    page_number = getattr(issue, "page_number", None)
    table_id = getattr(issue, "table_id", None)
    return any(
        page_number == page and (target_table is None or table_id == target_table)
        for page, target_table in targets
    )


def _recovery_preserves_grounded_charges(
    baseline: ExtractionDraft,
    candidate: ExtractionDraft,
    targets: Collection[tuple[int, str | None]],
    implicated_fields: dict[str, set[str]],
) -> bool:
    billable = {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}

    compared_fields = (
        "role",
        "description",
        "service_date_raw",
        "service_date_iso",
        "request_no",
        "service_code",
        "hsn_code",
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    )

    evidence_names = {
        "description": "description",
        "service_date_raw": "service_date",
        "service_date_iso": "service_date",
        "request_no": "request_no",
        "service_code": "service_code",
        "hsn_code": "hsn_code",
        "quantity": "quantity",
        "unit_price": "rate",
        "gross_amount": "gross_amount",
        "discount": "discount",
        "net_amount": "amount",
    }

    def rows(draft: ExtractionDraft) -> list[CanonicalRow]:
        return [
            row
            for unit in draft.page_units
            for row in unit.canonical_rows
            if row.role in billable and row.net_amount is not None
        ]

    def token_lookup(draft: ExtractionDraft) -> dict[str, TokenManifestEntry]:
        return {
            token.token_id: token
            for unit in draft.page_units
            for token in unit.token_manifest
        }

    baseline_tokens = token_lookup(baseline)
    candidate_tokens = token_lookup(candidate)

    def evidence_digest(row: CanonicalRow, field: str) -> str:
        name = evidence_names.get(field)
        return _stable_payload_digest(
            [
                item.model_dump(mode="json")
                for item in (row.field_evidence.get(name, ()) if name else row.evidence)
            ]
        )

    def grounding_strength(
        row: CanonicalRow,
        field: str,
        manifest: dict[str, TokenManifestEntry],
    ) -> int:
        name = evidence_names.get(field)
        evidence = row.field_evidence.get(name, ()) if name else row.evidence
        token_ids = tuple(
            token_id for item in evidence for token_id in item.token_ids
        )
        if not token_ids:
            return 0
        if len(token_ids) != 1 or token_ids[0] not in manifest:
            return 1
        token = manifest[token_ids[0]]
        expected_role = name
        return int(
            bool(
                expected_role
                and token.fragment_role == expected_role
                and (token.parent_token_id or token.parent_token_ids)
            )
        ) + 1

    available = rows(candidate)
    candidate_by_anchor = {
        row.row_anchor: row for row in available if row.row_anchor
    }
    for baseline_row in rows(baseline):
        targeted_fields = set(
            implicated_fields.get(baseline_row.row_anchor or "", set())
        ) | set(implicated_fields.get(str(baseline_row.id), set()))
        candidate_row = candidate_by_anchor.get(baseline_row.row_anchor or "")
        if candidate_row is None:
            return False
        for field in compared_fields:
            baseline_value = getattr(baseline_row, field)
            candidate_value = getattr(candidate_row, field)
            normalized_field = {
                "unit_price": "rate",
                "net_amount": "amount",
                "service_date_raw": "service_date",
                "service_date_iso": "service_date",
            }.get(field, field)
            targeted = field in targeted_fields or normalized_field in targeted_fields
            if not targeted:
                if baseline_value != candidate_value:
                    return False
                if evidence_digest(baseline_row, field) != evidence_digest(
                    candidate_row, field
                ):
                    return False
            elif baseline_value != candidate_value and grounding_strength(
                candidate_row, field, candidate_tokens
            ) <= grounding_strength(baseline_row, field, baseline_tokens):
                return False
    return True


def _normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _declined_recovery_draft(
    baseline: ExtractionDraft,
    candidate: ExtractionDraft,
    targets: tuple[tuple[int, str | None], ...],
    target_audit: dict[tuple[int, str | None], dict[str, tuple[str, ...]]] | None = None,
) -> ExtractionDraft:
    initial = dict(baseline.provider_usage.get("initial") or {})
    recovery = {
        "gemini_calls": 0,
        "gemini_measured_cost_usd": "0",
        "gemini_allowed": False,
        "gemini_mode": "off",
        "gemini_promotion_manifest_sha256": None,
        "gemini_provider_disabled_reason": "no_safe_improvement",
    }
    provider_usage = {
        "initial": initial,
        "recovery": recovery,
        "aggregate": {
            "gemini_calls": int(initial.get("gemini_calls") or 0),
            "gemini_measured_cost_usd": str(
                Decimal(str(initial.get("gemini_measured_cost_usd") or "0"))
            ),
        },
    }

    def target_digest(draft: ExtractionDraft, page: int, table: str | None) -> str:
        payload = [
            source_table.model_dump(mode="json")
            for unit in draft.page_units
            if unit.page_asset.page_number == page
            for source_table in unit.source_tables
            if table is None or source_table.table_id == table
        ]
        return _stable_payload_digest(payload)

    records = []
    candidate_records = {
        (int(item.get("page_number") or 0), item.get("table_id")): item
        for item in candidate.recovery_metadata.get("targets", [])
    }
    for page, table in targets:
        baseline_digest = target_digest(baseline, page, table)
        candidate_digest = target_digest(candidate, page, table)
        candidate_record = candidate_records.get((page, table), {})
        candidate_status = candidate_record.get("status")
        status = (
            candidate_status
            if candidate_status == "recovery_target_not_located"
            else "recovery_no_safe_improvement"
        )
        audit = (target_audit or {}).get((page, table), {})
        records.append(
            {
                "page_number": page,
                "table_id": table,
                "selected": "baseline",
                "status": status,
                "baseline_unit_sha256": baseline_digest,
                "candidate_unit_sha256": candidate_digest,
                "selected_unit_sha256": baseline_digest,
                "target_issue_ids": list(audit.get("target_issue_ids", ())),
                "removed_issue_ids": list(audit.get("removed_issue_ids", ())),
                "remaining_issue_ids": list(audit.get("remaining_issue_ids", ())),
                "new_issue_ids": list(audit.get("new_issue_ids", ())),
            }
        )
    targeted_pages = {page for page, _table in targets}
    untargeted = {
        key: digest
        for key, digest in baseline.raw_unit_sha256.items()
        if int(key.partition(":")[2]) not in targeted_pages
    }
    recovery_metadata = {
        "attempted": True,
        "targets": records,
        "untargeted_units_sha256": _stable_payload_digest(untargeted),
    }
    return replace(
        baseline,
        provider_usage=provider_usage,
        recovery_metadata=recovery_metadata,
    )


class OfflineExtractor:
    def __init__(
        self,
        vl_url: str,
        *,
        paddle_device: str = "cpu",
        vl_device: str = "cpu",
        hospital_id: str | None = None,
        profile_registry: Path | None = None,
        profiles: tuple[LayoutProfile, ...] | None = None,
        alias_registry: Path | None = None,
        gemini_mode: GeminiMode = GeminiMode.OFF,
        gemini_adapter: AdjudicationAdapter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.hospital_id = hospital_id
        self.alias_registry = alias_registry
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
        self.profile_registry_revision: int | None = None
        if profiles is not None:
            self.profiles = profiles
        elif profile_registry is not None and profile_registry.exists():
            profile_snapshot = JsonProfileRepository(profile_registry).snapshot()
            self.profiles = profile_snapshot.profiles
            self.profile_registry_revision = profile_snapshot.revision
        else:
            self.profiles = ()
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
        page_artifact_relative_path: str,
        baseline: ReconstructionResult,
        baseline_tokens: tuple[OcrToken, ...],
    ) -> tuple[
        ReconstructionResult | None,
        tuple[RecoveryAttempt, ...],
        tuple[TokenManifestEntry, ...],
    ]:
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
            return None, tuple(attempts), ()
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
                tuple[TokenManifestEntry, ...],
            ]
        ] = []

        def manifest_entries(
            local_tokens: tuple[OcrToken, ...],
            mapped_tokens: tuple[OcrToken, ...],
            *,
            image_path: Path,
            artifact_sha256: str,
            width: int,
            height: int,
            page_box: tuple[int, int, int, int],
        ) -> tuple[TokenManifestEntry, ...]:
            left, top, right, bottom = page_box
            matrix = (
                ((right - left) / max(1, width), 0.0, float(left)),
                (0.0, (bottom - top) / max(1, height), float(top)),
                (0.0, 0.0, 1.0),
            )
            return tuple(
                TokenManifestEntry(
                    token_id=mapped.token_id,
                    page_number=mapped.page_number,
                    text=mapped.text,
                    polygon=mapped.polygon,
                    artifact_sha256=page_artifact_sha256,
                    artifact_relative_path=page_artifact_relative_path,
                    confidence=mapped.confidence,
                    source_artifact_sha256=artifact_sha256,
                    source_artifact_relative_path=str(
                        image_path.resolve().relative_to(artifact_root.resolve())
                    ),
                    source_polygon=local.polygon,
                    source_width=width,
                    source_height=height,
                    source_to_page_matrix=matrix,
                )
                for local, mapped in zip(local_tokens, mapped_tokens, strict=True)
            )
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
                    manifest_entries(
                        local_tokens,
                        mapped_tokens,
                        image_path=image_path,
                        artifact_sha256=artifact_sha256,
                        width=image.shape[1],
                        height=image.shape[0],
                        page_box=work.box,
                    ),
                )
            )
        targeted_inputs = tuple(
            (
                item,
                tuple(
                    dict.fromkeys(
                        description_lane_recovery_regions(
                            item[-2],
                            table_box=work.box,
                        )
                    )
                ),
            )
            for item in reconstructed
            if description_lane_recovery_regions(
                item[-2],
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
                        _,
                    ) = item
                    recovered_tokens: list[OcrToken] = []
                    recovered_manifest: list[TokenManifestEntry] = []
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
                            mapped_target_tokens = map_crop_tokens_to_page(
                                    local_target_tokens,
                                    region,
                                    target_image.shape[1],
                                    target_image.shape[0],
                                    page_artifact_sha256,
                                )
                            recovered_tokens.extend(mapped_target_tokens)
                            recovered_manifest.extend(
                                manifest_entries(
                                    local_target_tokens,
                                    mapped_target_tokens,
                                    image_path=targeted_crop.output_path,
                                    artifact_sha256=targeted_crop.artifact_sha256,
                                    width=target_image.shape[1],
                                    height=target_image.shape[0],
                                    page_box=region,
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
                            tuple(recovered_manifest),
                        )
                    )
        sign_targets = return_sign_recovery_targets(
            baseline,
            table_box=work.box,
        )
        if sign_targets:
            recovered_sign_tokens: list[OcrToken] = []
            recovered_sign_manifest: list[TokenManifestEntry] = []
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
                    mapped_sign_tokens = map_crop_tokens_to_page(
                            prefixed_sign_tokens,
                            region,
                            sign_image.shape[1],
                            sign_image.shape[0],
                            page_artifact_sha256,
                        )
                    recovered_sign_tokens.extend(mapped_sign_tokens)
                    recovered_sign_manifest.extend(
                        manifest_entries(
                            prefixed_sign_tokens,
                            mapped_sign_tokens,
                            image_path=enhanced_sign.output_path,
                            artifact_sha256=enhanced_sign.artifact_sha256,
                            width=sign_image.shape[1],
                            height=sign_image.shape[0],
                            page_box=region,
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
                        (),
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
                        candidate_manifest,
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
                            tuple((*candidate_manifest, *recovered_sign_manifest)),
                        )
                    )
        safe_candidates = [
            item
            for item in reconstructed
            if safely_improves_reconstruction(baseline, item[-2])
        ]
        selected_item = max(
            safe_candidates,
            key=lambda item: reconstruction_quality(item[-2]),
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
                _,
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
        selected = selected_item[-2] if selected_item is not None else None
        selected_manifest = selected_item[-1] if selected_item is not None else ()
        return selected, tuple(attempts), selected_manifest

    def extract(
        self,
        source: Path,
        artifact_root: Path,
        progress: Callable[[int, int], None] | None = None,
        *,
        should_abort: Callable[[], bool] | None = None,
        alias_snapshot: dict[str, Any] | None = None,
        profile_identities: dict[str, dict[str, Any]] | None = None,
        profiles: tuple[LayoutProfile, ...] | None = None,
        profile_registry_revision: int | None = None,
        recovery_targets: tuple[tuple[int, str | None], ...] = (),
        baseline_draft: ExtractionDraft | None = None,
        allow_gemini: bool = True,
        _draft_sink: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        set_header_aliases({})
        if self.alias_registry is not None and alias_snapshot is None:
            raise AliasRegistryUnavailable("coordinated alias snapshot is required")
        resolved_hospital_id = self.hospital_id
        job_profiles = self.profiles if profiles is None else profiles
        recovery_target_set = set(recovery_targets)
        if baseline_draft is None and recovery_target_set:
            raise ValueError("targeted recovery requires a baseline result")
        if baseline_draft is not None and not recovery_target_set:
            raise ValueError("baseline result requires targeted recovery locations")

        def abort_checkpoint() -> None:
            if should_abort is not None and should_abort():
                raise ExtractionAborted

        abort_checkpoint()
        manifest = render_pdf(source, artifact_root / "pages", dpi=300)
        abort_checkpoint()
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
        baseline_rows = tuple(
            row
            for unit in (baseline_draft.page_units if baseline_draft else ())
            for row in unit.canonical_rows
        )
        baseline_tables = tuple(
            table
            for unit in (baseline_draft.page_units if baseline_draft else ())
            for table in unit.source_tables
        )
        baseline_diagnostics = tuple(
            diagnostic
            for unit in (baseline_draft.page_units if baseline_draft else ())
            for diagnostic in unit.diagnostics
        )
        baseline_token_manifest = tuple(
            token
            for unit in (baseline_draft.page_units if baseline_draft else ())
            for token in unit.token_manifest
        )
        targeted_pages = {page_number for page_number, _table_id in recovery_target_set}
        if baseline_draft is not None:
            for unit in baseline_draft.page_units:
                for candidate in unit.total_candidates:
                    if candidate.total.page_number in targeted_pages:
                        continue
                    document_total_candidates.append(candidate)
        hospital = baseline_draft.hospital if baseline_draft else None
        if baseline_draft is not None:
            resolved_hospital_id = baseline_draft.hospital_id
            if alias_snapshot is not None and resolved_hospital_id is not None:
                active_aliases = JsonAliasRepository.active_aliases(
                    alias_snapshot, resolved_hospital_id
                )
                set_header_aliases(
                    {
                        alias["normalized_label"]: CANONICAL_TO_HEADER_ROLE[
                            alias["canonical_field"]
                        ]
                        for alias in active_aliases
                        if alias["canonical_field"] in CANONICAL_TO_HEADER_ROLE
                    },
                    {
                        alias["normalized_label"]: alias["alias_id"]
                        for alias in active_aliases
                        if alias["canonical_field"] in CANONICAL_TO_HEADER_ROLE
                    },
                )
        gemini_calls = 0
        gemini_cost = Decimal("0")
        gemini_provider_disabled_reason: str | None = None
        page_tokens: dict[int, tuple[OcrToken, ...]] = {}
        recovery_token_manifest: dict[str, TokenManifestEntry] = {}
        precanonical_fragment_manifest: dict[str, TokenManifestEntry] = {}
        for page_asset in manifest.pages:
            abort_checkpoint()
            page_targets = {
                table_id
                for page_number, table_id in recovery_target_set
                if page_number == page_asset.page_number
            }
            if baseline_draft is not None and not page_targets:
                all_rows.extend(
                    row for row in baseline_rows if row.page_number == page_asset.page_number
                )
                all_source_tables.extend(
                    table
                    for table in baseline_tables
                    if table.page_number == page_asset.page_number
                )
                diagnostics.extend(
                    diagnostic
                    for diagnostic in baseline_diagnostics
                    if diagnostic.get("page_number") == page_asset.page_number
                )
                if progress:
                    progress(page_asset.page_number, len(manifest.pages))
                continue
            if baseline_draft is not None and None not in page_targets:
                all_rows.extend(
                    row
                    for row in baseline_rows
                    if row.page_number == page_asset.page_number
                    and row.table_id not in page_targets
                )
                all_source_tables.extend(
                    table
                    for table in baseline_tables
                    if table.page_number == page_asset.page_number
                    and table.table_id not in page_targets
                )
                diagnostics.extend(
                    diagnostic
                    for diagnostic in baseline_diagnostics
                    if diagnostic.get("page_number") == page_asset.page_number
                    and diagnostic.get("table_id") not in page_targets
                )
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
            abort_checkpoint()
            tokens = paddle_ocr_tokens(
                ocr_response.output,
                page_asset.page_number,
                page_asset.artifact_sha256,
            )
            page_tokens[page_asset.page_number] = tokens
            document_total_candidates.extend(extract_document_total_candidates(tokens))
            if page_asset.page_number == 1:
                hospital = detect_hospital(
                    tokens,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                )
                if alias_snapshot is not None and resolved_hospital_id is None:
                    identity_owners = combined_hospital_name_owners(
                        profile_identities or {},
                        alias_snapshot,
                        str(hospital.get("name") or "") if hospital else None,
                    )
                    resolved_hospital_id = (
                        next(iter(identity_owners))
                        if len(identity_owners) == 1
                        else None
                    )
                if alias_snapshot is not None and resolved_hospital_id is not None:
                    active_aliases = JsonAliasRepository.active_aliases(
                        alias_snapshot, resolved_hospital_id
                    )
                    set_header_aliases(
                        {
                            alias["normalized_label"]: CANONICAL_TO_HEADER_ROLE[
                                alias["canonical_field"]
                            ]
                            for alias in active_aliases
                            if alias["canonical_field"] in CANONICAL_TO_HEADER_ROLE
                        },
                        {
                            alias["normalized_label"]: alias["alias_id"]
                            for alias in active_aliases
                            if alias["canonical_field"] in CANONICAL_TO_HEADER_ROLE
                        },
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
            abort_checkpoint()
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
            no_table_assessment: dict[str, Any] | None = None

            table_work: list[TableWork] = []
            work_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
            if baseline_draft is not None and None not in page_targets:
                for table_id in sorted(str(value) for value in page_targets):
                    diagnostic = next(
                        (
                            item
                            for item in baseline_diagnostics
                            if item.get("page_number") == page_asset.page_number
                            and item.get("table_id") == table_id
                            and isinstance(item.get("box"), (list, tuple))
                            and len(item["box"]) == 4
                        ),
                        None,
                    )
                    if diagnostic is not None:
                        work_boxes.append(
                            (table_id, tuple(int(value) for value in diagnostic["box"]))
                        )
            else:
                work_boxes.extend(
                    (f"p{page_asset.page_number}-t{table_index + 1}", box)
                    for table_index, box in enumerate(boxes)
                )
            for table_id, box in work_boxes:
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

            specific_table_recovery = bool(
                baseline_draft is not None and None not in page_targets
            )
            if not table_work and not specific_table_recovery:
                no_table_assessment = _assess_no_table_page(page_path, tokens)

            if not table_work and not specific_table_recovery and (
                not no_table_assessment["demonstrably_blank"]
                or (page_asset.page_number, None) in recovery_target_set
            ):
                table_id = f"p{page_asset.page_number}-t1"
                safe_box = (0, 0, page_asset.width, page_asset.height)
                crop = crop_region(
                    page_path,
                    artifact_root / "crops" / f"{table_id}.png",
                    page_asset.page_number,
                    safe_box,
                )
                source_table_crop_paths[(page_asset.page_number, table_id)] = (
                    crop.output_path
                )
                source_table_crop_boxes[(page_asset.page_number, table_id)] = safe_box
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
                route = "full_page_form_assessment"

            if not table_work:
                if specific_table_recovery:
                    missing_targets = {str(value) for value in page_targets}
                    all_rows.extend(
                        row
                        for row in baseline_rows
                        if row.page_number == page_asset.page_number
                        and row.table_id in missing_targets
                    )
                    all_source_tables.extend(
                        table
                        for table in baseline_tables
                        if table.page_number == page_asset.page_number
                        and table.table_id in missing_targets
                    )
                    diagnostics.extend(
                        {
                            **diagnostic,
                            "status": "recovery_target_not_located",
                        }
                        for diagnostic in baseline_diagnostics
                        if diagnostic.get("page_number") == page_asset.page_number
                        and diagnostic.get("table_id") in missing_targets
                    )
                else:
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
                            "financial_form_suspected": False,
                            **(no_table_assessment or {}),
                        }
                    )
            for work in table_work:
                abort_checkpoint()
                targeted_recovery = bool(
                    (work.page_number, work.table_id) in recovery_target_set
                    or (work.page_number, None) in recovery_target_set
                )
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
                    profiles=job_profiles,
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
                        for profile in job_profiles
                        if profile.lifecycle is ProfileLifecycle.SHADOW
                    ),
                    include_shadow=True,
                )
                selected_profile = None
                if profile_match is not None and profile_match.selected:
                    selected_profile = next(
                        profile
                        for profile in job_profiles
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
                if targeted_recovery or _should_attempt_crop_recovery(
                    reconstruction,
                    parsed_rows=parsed_rows,
                    table_box=work.box,
                ):
                    recovered, attempts, recovered_manifest = self._recover_crop_ocr(
                        source=source,
                        artifact_root=artifact_root,
                        work=work,
                        prior_schemas=_recovery_prior_schemas(
                            schema_states,
                            page_number=work.page_number,
                            table_id=work.table_id,
                        ),
                        page_artifact_sha256=work.page_artifact_sha256,
                        page_artifact_relative_path=str(
                            Path("pages") / page_asset.relative_path
                        ),
                        baseline=reconstruction,
                        baseline_tokens=tokens_in_box(tokens, work.box),
                    )
                    recovery_attempts.extend(attempts)
                    if recovered is not None:
                        recovery_token_manifest.update(
                            {item.token_id: item for item in recovered_manifest}
                        )
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
                # Normalize every split Printed value into a stable fragment
                # token before the canonical rows used for linking are built.
                # The final Printed grid and canonical candidate fields now
                # consume the same value/evidence representation.
                reconstruction_token_lookup = {
                    token.token_id: TokenManifestEntry(
                        token_id=token.token_id,
                        page_number=token.page_number,
                        text=token.text,
                        polygon=token.polygon,
                        artifact_sha256=token.artifact_sha256,
                        artifact_relative_path=str(
                            Path("pages") / page_asset.relative_path
                        ),
                        confidence=token.confidence,
                    )
                    for token in tokens
                }
                reconstruction_token_lookup.update(recovery_token_manifest)
                materialized_tables, fragments, fragment_assignments = (
                    _materialize_printed_cell_fragments(
                        reconstruction.source_tables,
                        reconstruction_token_lookup,
                    )
                )
                if fragments:
                    reconstruction = replace(
                        reconstruction,
                        source_tables=materialized_tables,
                        rows=_apply_fragment_ids_to_aligned_rows(
                            reconstruction.rows,
                            materialized_tables,
                            fragment_assignments,
                        ),
                    )
                    precanonical_fragment_manifest.update(
                        {item.token_id: item for item in fragments}
                    )
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
                        targeted_recovery
                        or
                        profile_heavy_sample
                        or not (profile_match and profile_match.selected and parsed_rows)
                    )
                    and (
                        targeted_recovery
                        or
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
                        abort_checkpoint()
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
                            abort_checkpoint()
                        except ExtractionAborted:
                            raise
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
                if (
                    needs_gemini_recovery
                    and allow_gemini
                    and self.gemini_mode is not GeminiMode.OFF
                ):
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
                    and allow_gemini
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
                    and allow_gemini
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
                selected_parsed_rows = list(parsed_rows)
                selected_reconstruction_tables = tuple(reconstruction.source_tables)
                recovery_selection_status: str | None = None
                if targeted_recovery and baseline_draft is not None:
                    baseline_target_tables = tuple(
                        table
                        for table in baseline_tables
                        if table.page_number == work.page_number
                        and table.table_id == work.table_id
                    )
                    baseline_target_rows = tuple(
                        row
                        for row in baseline_rows
                        if row.page_number == work.page_number
                        and row.table_id == work.table_id
                    )

                    baseline_payload = [
                        table.model_dump(mode="json") for table in baseline_target_tables
                    ]
                    candidate_payload = [
                        table.model_dump(mode="json")
                        for table in selected_reconstruction_tables
                    ]
                    if (
                        not candidate_payload
                        or _stable_payload_digest(candidate_payload)
                        == _stable_payload_digest(baseline_payload)
                    ):
                        selected_reconstruction_tables = baseline_target_tables
                        selected_parsed_rows = list(baseline_target_rows)
                        recovery_selection_status = "recovery_no_safe_improvement"
                all_rows.extend(selected_parsed_rows)
                all_source_tables.extend(selected_reconstruction_tables)
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
                        "crop_relative_path": str(
                            work.crop_path.resolve().relative_to(artifact_root.resolve())
                        ),
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
                        "status": recovery_selection_status or "extracted",
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
                        "gemini_allowed": allow_gemini,
                        "gemini_invoked": gemini_invoked,
                        "gemini_cache_hit": gemini_cache_hit,
                        "gemini_grounded_rows": gemini_grounded_rows,
                        "gemini_rejected_reasons": gemini_rejected_reasons,
                        "gemini_redaction_safe": gemini_redaction_safe,
                        "gemini_masked_tokens": gemini_masked_tokens,
                        "gemini_block_reason": gemini_block_reason,
                        "content": content,
                        **(no_table_assessment or {}),
                        **reconstruction.diagnostics,
                    }
                )
                abort_checkpoint()
            if progress:
                progress(page_asset.page_number, len(manifest.pages))

        abort_checkpoint()
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
        relative_by_page = {
            page.page_number: str(Path("pages") / page.relative_path)
            for page in manifest.pages
        }
        token_lookup = {item.token_id: item for item in baseline_token_manifest}
        token_lookup.update(
            {
                token.token_id: TokenManifestEntry(
                    token_id=token.token_id,
                    page_number=token.page_number,
                    text=token.text,
                    polygon=token.polygon,
                    artifact_sha256=token.artifact_sha256,
                    artifact_relative_path=relative_by_page[token.page_number],
                    confidence=token.confidence,
                )
                for tokens in page_tokens.values()
                for token in tokens
            }
        )
        token_lookup.update(recovery_token_manifest)
        token_lookup.update(precanonical_fragment_manifest)
        (
            selected_source_tables,
            printed_fragments,
            _fragment_assignments,
        ) = _materialize_printed_cell_fragments(
            selected_source_tables,
            token_lookup,
        )
        token_lookup.update({item.token_id: item for item in printed_fragments})
        rows = _apply_document_role_policy(_deduplicate(selected_rows))
        draft_source_tables = selected_source_tables
        draft_row_candidates = tuple(rows)
        source_tables = _link_source_tables(selected_source_tables, rows)
        canonical_by_id = {str(row.id): row for row in rows}
        source_tables = tuple(
            _promote_grounded_date_column(
                table,
                tuple(
                    canonical_by_id[source_row.canonical_row_id]
                    for source_row in table.rows
                    if source_row.canonical_row_id in canonical_by_id
                    and canonical_by_id[source_row.canonical_row_id].role
                    in {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}
                ),
            )
            for table in source_tables
        )
        rows = _recover_grounded_service_dates(
            source_tables,
            rows,
            already_linked=True,
        )
        rows = _synchronize_canonical_from_printed(source_tables, rows)
        rows = _attach_derived_field_provenance(rows)
        source_tables = _attach_receipt_source_metadata(source_tables, rows)
        source_tables, rows = _assign_geometry_anchors(
            source_tables,
            rows,
            tuple(
                page.model_copy(
                    update={"relative_path": str(Path("pages") / page.relative_path)}
                )
                for page in manifest.pages
            ),
        )
        rows, receipt_duplicate_pairs = _flag_possible_supporting_receipt_duplicates(
            rows,
            source_tables,
        )
        document_total_candidates = list(
            assign_document_total_contexts(document_total_candidates, diagnostics)
        )
        raw_total_candidates = tuple(
            _raw_total_candidate(candidate, source_tables)
            for candidate in document_total_candidates
        )
        document_totals = select_document_totals(document_total_candidates)
        document_total = select_document_total(document_total_candidates)

        table_ids_by_token: dict[str, set[str]] = {}
        for table in source_tables:
            for evidence in (
                *(item for column in table.columns for item in column.evidence),
                *(
                    item
                    for source_row in table.rows
                    for cell in source_row.cells
                    for item in cell.evidence
                ),
            ):
                for token_id in evidence.token_ids:
                    table_ids_by_token.setdefault(token_id, set()).add(table.table_id)
        # Table-scoped recovery may retain other baseline tables on the same
        # page. Keep their fragment tokens, then replace only colliding IDs
        # with the newly selected page/recovery representation.
        token_manifest_by_id = {
            item.token_id: item for item in baseline_token_manifest
        }
        token_manifest_by_id.update(
            {
                token.token_id: TokenManifestEntry(
                    token_id=token.token_id,
                    page_number=token.page_number,
                    table_ids=tuple(sorted(table_ids_by_token.get(token.token_id, ()))),
                    text=token.text,
                    polygon=token.polygon,
                    artifact_sha256=token.artifact_sha256,
                    artifact_relative_path=relative_by_page[token.page_number],
                    confidence=token.confidence,
                )
                for page_number in sorted(page_tokens)
                for token in page_tokens[page_number]
            }
        )
        token_manifest_by_id.update(recovery_token_manifest)
        token_manifest_by_id.update(precanonical_fragment_manifest)
        token_manifest_by_id.update({item.token_id: item for item in printed_fragments})
        token_manifest = tuple(token_manifest_by_id.values())

        normalized_diagnostics: list[dict[str, Any]] = []
        for diagnostic in diagnostics:
            payload = dict(diagnostic)
            payload.setdefault(
                "diagnostic_kind", "table" if payload.get("table_id") else "page"
            )
            payload.setdefault(
                "diagnostic_id",
                f"p{int(payload.get('page_number') or 1)}-"
                f"{payload.get('table_id') or 'page'}-"
                f"{_stable_payload_digest(payload)[:12]}",
            )
            if payload.get("diagnostic_kind") == "table" and not payload.get(
                "source_table_id"
            ):
                matching_tables = tuple(
                    table
                    for table in source_tables
                    if table.page_number == int(payload.get("page_number") or 0)
                    and table.table_id == payload.get("table_id")
                )
                if matching_tables:
                    payload["source_table_id"] = matching_tables[0].id
            normalized_diagnostics.append(payload)
        diagnostic_source_table_ids = {
            str(item.get("source_table_id"))
            for item in normalized_diagnostics
            if item.get("diagnostic_kind") == "table" and item.get("source_table_id")
        }
        for table in source_tables:
            if table.id not in diagnostic_source_table_ids:
                normalized_diagnostics.append(
                    {
                        "diagnostic_id": f"p{table.page_number}-{table.id}-inventory",
                        "diagnostic_kind": "table",
                        "page_number": table.page_number,
                        "table_id": table.table_id,
                        "source_table_id": table.id,
                        "table_type": table.table_type.value,
                        "status": "published_source_table",
                    }
                )
        diagnostic_pages = {
            int(item["page_number"])
            for item in normalized_diagnostics
            if item.get("diagnostic_kind") == "page" and item.get("page_number")
        }
        for page in manifest.pages:
            if page.page_number in diagnostic_pages:
                continue
            page_tables = tuple(
                table for table in source_tables if table.page_number == page.page_number
            )
            normalized_diagnostics.append(
                {
                    "diagnostic_id": f"p{page.page_number}-page-inventory",
                    "diagnostic_kind": "page",
                    "page_number": page.page_number,
                    "status": "financial_tables_detected" if page_tables else "unclassified",
                    "page_classification": "financial" if page_tables else "unclassified",
                    "table_count": len(page_tables),
                }
            )
        diagnostics = normalized_diagnostics
        abort_checkpoint()
        provider_usage = _provider_usage_payload(
            baseline_usage=(baseline_draft.provider_usage if baseline_draft else None),
            gemini_mode=self.gemini_mode.value,
            gemini_allowed=allow_gemini,
            gemini_calls=gemini_calls,
            gemini_cost=gemini_cost,
            disabled_reason=gemini_provider_disabled_reason,
            promotion_sha256=(
                self.gemini_promotion.frozen_manifest_sha256
                if self.gemini_promotion
                else None
            ),
        )
        recovery_metadata = _recovery_metadata(
            baseline_draft,
            recovery_target_set,
            source_tables,
            diagnostics,
        )
        result = {
            "output_version": "offline_accuracy_spine_v5",
            "contract_revision": 2,
            "document_total_version": DOCUMENT_TOTAL_VERSION,
            "document_totals_version": DOCUMENT_TOTALS_VERSION,
            "document_total": (
                document_total.model_dump(mode="json") if document_total is not None else None
            ),
            "document_totals": [total.model_dump(mode="json") for total in document_totals],
            "raw_total_candidates": [
                candidate.model_dump(mode="json") for candidate in raw_total_candidates
            ],
            "document_id": document_id,
            "hospital_id": resolved_hospital_id,
            "hospital": hospital,
            "alias_registry_revision": (
                alias_snapshot["revision"] if alias_snapshot is not None else None
            ),
            "profile_registry_revision": (
                profile_registry_revision
                if profile_registry_revision is not None
                else getattr(self, "profile_registry_revision", None)
            ),
            "applied_alias_ids": list(matched_header_alias_ids()),
            "source_sha256": sha256_file(source),
            "source_name": source.name,
            "pages": len(manifest.pages),
            "page_assets": [
                {
                    "document_sha256": page.document_sha256,
                    "page_number": page.page_number,
                    "artifact_sha256": page.artifact_sha256,
                    "width": page.width,
                    "height": page.height,
                    "dpi": page.dpi,
                    "renderer": page.renderer,
                    "renderer_version": page.renderer_version,
                    "relative_path": str(Path("pages") / page.relative_path),
                }
                for page in manifest.pages
            ],
            "source_tables": [table.model_dump(mode="json") for table in source_tables],
            "token_manifest": [item.model_dump(mode="json") for item in token_manifest],
            "suppressed_repeated_source_tables": [
                {
                    "page_number": page_number,
                    "table_id": table_id,
                }
                for page_number, table_id in suppressed_source_tables
            ],
            "rows": [row.model_dump(mode="json") for row in rows],
            "receipt_duplicate_pairs": receipt_duplicate_pairs,
            "diagnostics": diagnostics,
            "provider_usage": provider_usage,
            "recovery": recovery_metadata,
        }
        if _draft_sink is not None:
            baseline_units = {
                unit.page_asset.page_number: unit
                for unit in (baseline_draft.page_units if baseline_draft else ())
            }
            page_units: list[PageExtractionUnit] = []
            for page in manifest.pages:
                published_asset = page.model_copy(
                    update={"relative_path": str(Path("pages") / page.relative_path)}
                )
                raw_page_tables = tuple(
                    table
                    for table in draft_source_tables
                    if table.page_number == page.page_number
                )
                raw_page_rows = tuple(
                    row
                    for row in draft_row_candidates
                    if row.page_number == page.page_number
                )
                page_manifest = tuple(
                    item
                    for item in token_manifest
                    if item.page_number == page.page_number
                )
                page_diagnostics = tuple(
                    item
                    for item in diagnostics
                    if int(item.get("page_number") or 0) == page.page_number
                )
                page_totals = tuple(
                    candidate
                    for candidate in document_total_candidates
                    if candidate.total.page_number == page.page_number
                )
                table_units = tuple(
                    TableExtractionUnit(
                        page_number=page.page_number,
                        table_id=table.table_id,
                        source_table=table,
                        row_candidates=tuple(
                            row for row in raw_page_rows if row.table_id == table.table_id
                        ),
                        crop_relative_path=(
                            str(
                                source_table_crop_paths[
                                    (page.page_number, table.table_id)
                                ]
                                .resolve()
                                .relative_to(artifact_root.resolve())
                            )
                            if (page.page_number, table.table_id)
                            in source_table_crop_paths
                            else None
                        ),
                        crop_box=source_table_crop_boxes.get(
                            (page.page_number, table.table_id)
                        ),
                        diagnostics=tuple(
                            item
                            for item in page_diagnostics
                            if item.get("table_id") == table.table_id
                        ),
                        normalized_fragments=tuple(
                            item
                            for item in page_manifest
                            if item.fragment_role is not None
                            and table.table_id in item.table_ids
                        ),
                        recovery_tokens=tuple(
                            item
                            for item in page_manifest
                            if item.source_artifact_sha256 is not None
                            and table.table_id in item.table_ids
                        ),
                        raw_total_candidates=tuple(
                            item
                            for item in page_totals
                            if item.total.evidence.table_id == table.table_id
                        ),
                        provider_usage=provider_usage,
                    )
                    for table in raw_page_tables
                )
                assigned_table_ids = {table.table_id for table in raw_page_tables}
                page_units.append(
                    PageExtractionUnit(
                        page_asset=published_asset,
                        ocr_tokens=page_tokens.get(
                            page.page_number,
                            baseline_units.get(page.page_number).ocr_tokens
                            if page.page_number in baseline_units
                            else (),
                        ),
                        token_manifest=page_manifest,
                        table_units=table_units,
                        unassigned_row_candidates=tuple(
                            row
                            for row in raw_page_rows
                            if row.table_id not in assigned_table_ids
                        ),
                        diagnostics=page_diagnostics,
                        total_candidates=page_totals,
                        provider_usage=provider_usage,
                    )
                )
            _draft_sink["page_units"] = tuple(page_units)
            _draft_sink["provider_usage"] = provider_usage
            _draft_sink["hospital"] = hospital
            _draft_sink["hospital_id"] = resolved_hospital_id
            _draft_sink["document_id"] = document_id
            _draft_sink["source_sha256"] = sha256_file(source)
            _draft_sink["source_name"] = source.name
            _draft_sink["alias_registry_revision"] = (
                alias_snapshot["revision"] if alias_snapshot is not None else None
            )
            _draft_sink["profile_registry_revision"] = (
                profile_registry_revision
                if profile_registry_revision is not None
                else getattr(self, "profile_registry_revision", None)
            )
            _draft_sink["applied_alias_ids"] = tuple(matched_header_alias_ids())
            _draft_sink["suppressed_repeated_source_tables"] = tuple(
                suppressed_source_tables
            )
            _draft_sink["recovery_metadata"] = recovery_metadata
        return result

    def extract_draft(
        self,
        source: Path,
        artifact_root: Path,
        progress: Callable[[int, int], None] | None = None,
        **options: Any,
    ) -> ExtractionDraft:
        sink: dict[str, Any] = {}
        self.extract(
            source,
            artifact_root,
            progress,
            _draft_sink=sink,
            **options,
        )
        return ExtractionDraft(
            document_id=sink["document_id"],
            source_sha256=sink["source_sha256"],
            source_name=sink["source_name"],
            page_units=sink["page_units"],
            provider_usage=sink["provider_usage"],
            hospital=sink["hospital"],
            hospital_id=sink["hospital_id"],
            alias_registry_revision=sink["alias_registry_revision"],
            profile_registry_revision=sink["profile_registry_revision"],
            applied_alias_ids=sink["applied_alias_ids"],
            suppressed_repeated_source_tables=sink[
                "suppressed_repeated_source_tables"
            ],
            recovery_metadata=sink["recovery_metadata"],
        )

    def recover_draft(
        self,
        source: Path,
        artifact_root: Path,
        draft: ExtractionDraft,
        recovery_targets: tuple[tuple[int, str | None], ...],
        progress: Callable[[int, int], None] | None = None,
        **options: Any,
    ) -> ExtractionDraft:
        sink: dict[str, Any] = {}
        options.pop("allow_gemini", None)
        options.pop("baseline_draft", None)
        options.pop("recovery_targets", None)
        self.extract(
            source,
            artifact_root,
            progress,
            baseline_draft=draft,
            recovery_targets=recovery_targets,
            allow_gemini=False,
            _draft_sink=sink,
            **options,
        )
        candidate = ExtractionDraft(
            document_id=sink["document_id"],
            source_sha256=sink["source_sha256"],
            source_name=sink["source_name"],
            page_units=sink["page_units"],
            provider_usage=sink["provider_usage"],
            hospital=sink["hospital"],
            hospital_id=sink["hospital_id"],
            alias_registry_revision=sink["alias_registry_revision"],
            profile_registry_revision=sink["profile_registry_revision"],
            applied_alias_ids=sink["applied_alias_ids"],
            suppressed_repeated_source_tables=sink[
                "suppressed_repeated_source_tables"
            ],
            recovery_metadata=sink["recovery_metadata"],
        )
        targeted_pages = {page for page, _table in recovery_targets}
        for unit, digest in draft.raw_unit_sha256.items():
            page_number = int(unit.partition(":")[2])
            if page_number not in targeted_pages and candidate.raw_unit_sha256.get(unit) != digest:
                raise RuntimeError("targeted_recovery_changed_untargeted_raw_unit")
        from gmoney.extraction.validation import validate_extraction_result

        baseline_report = validate_extraction_result(
            source, draft.result, artifact_root
        )
        candidate_report = validate_extraction_result(
            source, candidate.result, artifact_root
        )
        baseline_blocking = tuple(
            issue
            for issue in baseline_report.issues
            if issue.severity.value in {"blocking", "fatal"}
        )
        candidate_blocking = tuple(
            issue
            for issue in candidate_report.issues
            if issue.severity.value in {"blocking", "fatal"}
        )
        baseline_targeted = tuple(
            issue
            for issue in baseline_blocking
            if _issue_is_in_recovery_target(issue, recovery_targets)
        )
        candidate_targeted = tuple(
            issue
            for issue in candidate_blocking
            if _issue_is_in_recovery_target(issue, recovery_targets)
        )
        baseline_keys = {_issue_semantic_key(issue): issue for issue in baseline_blocking}
        candidate_keys = {_issue_semantic_key(issue): issue for issue in candidate_blocking}
        removed_issue_ids = tuple(
            issue.id
            for key, issue in baseline_keys.items()
            if key not in candidate_keys and issue in baseline_targeted
        )
        new_blocking = any(key not in baseline_keys for key in candidate_keys)
        target_audit: dict[
            tuple[int, str | None], dict[str, tuple[str, ...]]
        ] = {}
        for target in recovery_targets:
            baseline_for_target = tuple(
                issue
                for issue in baseline_blocking
                if _issue_is_in_recovery_target(issue, (target,))
            )
            candidate_for_target = tuple(
                issue
                for issue in candidate_blocking
                if _issue_is_in_recovery_target(issue, (target,))
            )
            baseline_target_keys = {
                _issue_semantic_key(issue): issue for issue in baseline_for_target
            }
            candidate_target_keys = {
                _issue_semantic_key(issue): issue for issue in candidate_for_target
            }
            target_audit[target] = {
                "target_issue_ids": tuple(issue.id for issue in baseline_for_target),
                "removed_issue_ids": tuple(
                    issue.id
                    for key, issue in baseline_target_keys.items()
                    if key not in candidate_target_keys
                ),
                "remaining_issue_ids": tuple(
                    issue.id
                    for key, issue in baseline_target_keys.items()
                    if key in candidate_target_keys
                ),
                "new_issue_ids": tuple(
                    issue.id
                    for key, issue in candidate_target_keys.items()
                    if key not in baseline_target_keys
                ),
            }
        implicated_fields: dict[str, set[str]] = {}
        baseline_rows_by_id = {
            str(row.id): row for unit in draft.page_units for row in unit.canonical_rows
        }
        for issue in baseline_targeted:
            if issue.canonical_row_id and issue.field:
                implicated_fields.setdefault(str(issue.canonical_row_id), set()).add(
                    str(issue.field)
                )
                baseline_row = baseline_rows_by_id.get(str(issue.canonical_row_id))
                if baseline_row is not None and baseline_row.row_anchor:
                    implicated_fields.setdefault(baseline_row.row_anchor, set()).add(
                        str(issue.field)
                    )
        candidate_recovery_records = {
            (int(item.get("page_number") or 0), item.get("table_id")): item
            for item in candidate.recovery_metadata.get("targets", [])
        }
        target_status_safe = all(
            candidate_recovery_records.get(target, {}).get("status") == "recovered"
            and target_audit[target]["removed_issue_ids"]
            and not target_audit[target]["new_issue_ids"]
            for target in recovery_targets
        )
        safe = bool(
            removed_issue_ids
            and len(candidate_targeted) < len(baseline_targeted)
            and not new_blocking
            and target_status_safe
            and _recovery_preserves_grounded_charges(
                draft,
                candidate,
                recovery_targets,
                implicated_fields,
            )
        )
        if not safe:
            return _declined_recovery_draft(
                draft,
                candidate,
                recovery_targets,
                target_audit,
            )
        recovery_metadata = deepcopy(candidate.recovery_metadata)
        for target in recovery_metadata.get("targets", []):
            identity = (int(target.get("page_number") or 0), target.get("table_id"))
            audit = target_audit.get(identity, {})
            target["target_issue_ids"] = list(audit.get("target_issue_ids", ()))
            target["removed_issue_ids"] = list(audit.get("removed_issue_ids", ()))
            target["remaining_issue_ids"] = list(audit.get("remaining_issue_ids", ()))
            target["new_issue_ids"] = list(audit.get("new_issue_ids", ()))
            target["status"] = "recovered"
            target["selected"] = "candidate"
        return ExtractionDraft(
            document_id=candidate.document_id,
            source_sha256=candidate.source_sha256,
            source_name=candidate.source_name,
            page_units=candidate.page_units,
            provider_usage=candidate.provider_usage,
            hospital=candidate.hospital,
            hospital_id=candidate.hospital_id,
            alias_registry_revision=candidate.alias_registry_revision,
            profile_registry_revision=candidate.profile_registry_revision,
            applied_alias_ids=candidate.applied_alias_ids,
            suppressed_repeated_source_tables=(
                candidate.suppressed_repeated_source_tables
            ),
            recovery_metadata=recovery_metadata,
        )


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
