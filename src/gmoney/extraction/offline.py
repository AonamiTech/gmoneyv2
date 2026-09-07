from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
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

from gmoney.contracts.evidence import (
    OcrToken,
    PageAsset,
    PagePreprocessingRecord,
    PageQuality,
    Point,
    Polygon,
    PreprocessingCandidate,
    PreprocessingVariant,
    TransformChain,
)
from gmoney.contracts.extraction import (
    CanonicalRow,
    CanonicalTableCrop,
    DerivedFieldProvenance,
    DocumentTotal,
    EvidenceRef,
    ExtractionDiagnostic,
    PageType,
    ProviderUsage,
    RawTotalCandidate,
    ReceiptDuplicatePair,
    ReceiptSourceMetadata,
    RecoveryMetadata,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    SuppressedSourceTable,
    TableAdapterInput,
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
from gmoney.contracts.v6 import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    CanonicalRowV2,
    CanonicalTableArtifact,
    DenseBackwardGridMapping,
    DocumentTotalV2,
    EvidenceRefV2,
    ExtractionResultV6,
    HomographyMapping,
    IdentityMapping,
    LogicalTableSelection,
    RawTotalCandidateV2,
    SourceCellV2,
    SourceColumnV2,
    SourceRowV2,
    SourceTableV2,
    TableAdapterInputV2,
    TableCandidateScore,
    TableMatchEdge,
    TableSelectionRun,
    TokenManifestEntryV2,
    UvdocShadowRun,
    canonical_json,
    canonical_sha256,
)
from gmoney.contracts.v6 import (
    PageArtifact as V6PageArtifact,
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
    map_transformed_tokens_to_page,
    merge_recovery_tokens,
    needs_field_quality_recovery,
    reconstruction_quality,
    replace_tokens_in_regions,
    return_sign_recovery_targets,
    safely_improves_reconstruction,
    safely_realigns_perspective_reconstruction,
)
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.extraction.spatial import AlignedLedgerRow, align_candidate_rows
from gmoney.extraction.table_selection import (
    MATCH_POLICY_VERSION,
    TableCandidateVariant,
    TableProposal,
    match_logical_tables,
    select_stage_one,
    select_stage_two,
    stage_two_key,
)
from gmoney.extraction.typed_values import (
    parse_decimal,
    parse_quantity,
    parse_service_date,
)
from gmoney.geometry.crop import (
    clahe_variant,
    color_overlay_suppressed_variant,
    crop_region,
    resize_region,
)
from gmoney.geometry.dense import load_dense_grid, map_dense_points
from gmoney.geometry.normalize import normalize_quadrilateral_region
from gmoney.geometry.preprocess import (
    ORIENTATION_CONFIDENCE_THRESHOLD,
    PreparedPageCandidate,
    detect_table_quadrilateral,
    prepare_page_candidates,
)
from gmoney.geometry.render import render_pdf
from gmoney.geometry.transform import (
    Matrix,
    apply_matrix,
    compose,
    invert,
    right_angle_rotation,
    translation,
)
from gmoney.geometry.transform import (
    identity as transform_identity,
)
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
    PaddleDocOrientationAdapter,
    PaddleOcrV6Adapter,
    PaddleOcrVlAdapter,
)
from gmoney.inference.redaction import redact_crop
from gmoney.inference.uvdoc import (
    UVDOC_ADAPTER_VERSION,
    PaddleUvdocAdapter,
    UvdocPreparedRun,
    UvdocPreregistration,
    load_preregistration,
)
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
    crop_width: int
    crop_height: int
    candidate_box: tuple[int, int, int, int]
    box: tuple[int, int, int, int]
    source_polygon: Polygon
    crop_to_source_matrix: Matrix
    selected_page_artifact_sha256: str
    selected_variant: PreprocessingVariant
    selected_dpi: int
    tokens: tuple[OcrToken, ...]
    canonical_tokens: tuple[OcrToken, ...]
    adapter_inputs: list[TableAdapterInput]


@dataclass(frozen=True)
class PageInferenceBundle:
    candidate: PreparedPageCandidate
    ocr_response: InferenceResponse
    ocr_cache_hit: bool
    layout_response: InferenceResponse
    layout_cache_hit: bool
    tokens: tuple[OcrToken, ...]
    source_tokens: tuple[OcrToken, ...]
    candidate_layout_boxes: tuple[tuple[int, int, int, int], ...]
    candidate_geometry_boxes: tuple[tuple[int, int, int, int], ...]
    layout_boxes: tuple[tuple[int, int, int, int], ...]
    geometry_boxes: tuple[tuple[int, int, int, int], ...]
    reconstruction_score: tuple[int, ...]


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
        if (schema.source_page != page_number or schema.source_table != table_id)
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
            and envelope.get("canonical_artifact_id") == request.canonical_artifact_id
            and envelope.get("canonical_artifact_sha256") == request.canonical_artifact_sha256
            and envelope.get("options") == request.options
            and envelope.get("model_spec") == adapter.spec.model_dump(mode="json")
        ):
            try:
                response = InferenceResponse.model_validate(envelope["response"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if (
                    response.input_artifact_sha256 == request.artifact_sha256
                    and response.canonical_artifact_sha256 == request.canonical_artifact_sha256
                    and response.canonical_artifact_id == request.canonical_artifact_id
                ):
                    return response, True

    response = adapter.predict(request)
    if response.input_artifact_sha256 != request.artifact_sha256:
        raise ValueError("inference response input artifact hash differs from request")
    if response.canonical_artifact_sha256 != request.canonical_artifact_sha256:
        raise ValueError("inference response canonical artifact hash differs from request")
    if response.canonical_artifact_id != request.canonical_artifact_id:
        raise ValueError("inference response canonical artifact ID differs from request")
    envelope = {
        "artifact_sha256": request.artifact_sha256,
        "canonical_artifact_id": request.canonical_artifact_id,
        "canonical_artifact_sha256": request.canonical_artifact_sha256,
        "options": request.options,
        "model_spec": adapter.spec.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(cache_path)
    return response, False


def _mapped_box(
    box: Collection[float],
    transform: Matrix,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = (float(value) for value in box)
    points = apply_matrix(
        transform,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )
    return (
        round(min(point[0] for point in points)),
        round(min(point[1] for point in points)),
        round(max(point[0] for point in points)),
        round(max(point[1] for point in points)),
    )


def _bounded_mapped_box(
    box: Collection[float],
    transform: Matrix,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _mapped_box(box, transform)
    bounded = (
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(1, min(width, right)),
        max(1, min(height, bottom)),
    )
    if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
        raise ValueError("mapped table box is outside the target artifact")
    return bounded


def _source_polygon(
    candidate_box: tuple[int, int, int, int],
    candidate_to_source: Matrix,
    *,
    source_width: int,
    source_height: int,
) -> Polygon:
    left, top, right, bottom = candidate_box
    mapped = apply_matrix(
        candidate_to_source,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )
    return Polygon(
        points=tuple(
            Point(
                x=max(0.0, min(float(source_width), x)),
                y=max(0.0, min(float(source_height), y)),
            )
            for x, y in mapped
        )
    )


def _polygon_box(
    polygon: Polygon,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xs = tuple(point.x for point in polygon.points)
    ys = tuple(point.y for point in polygon.points)
    left = max(0, min(width - 1, round(min(xs))))
    top = max(0, min(height - 1, round(min(ys))))
    right = max(1, min(width, round(max(xs))))
    bottom = max(1, min(height, round(max(ys))))
    if right <= left or bottom <= top:
        raise ValueError("canonical table crop has an empty source projection")
    return left, top, right, bottom


def _layout_boxes(
    output: dict[str, Any],
    to_page: Matrix | None = None,
) -> list[tuple[int, int, int, int]]:
    pages = output.get("pages") or []
    if not pages:
        return []
    boxes = pages[0].get("res", {}).get("boxes") or []
    transform = to_page or (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    return [
        _mapped_box(box["coordinate"], transform)
        for box in boxes
        if box.get("label") == "table" and float(box.get("score") or 0) >= 0.3
    ]


def _ocr_geometry_boxes(
    output: dict[str, Any],
    to_page: Matrix,
) -> list[tuple[int, int, int, int]]:
    result = (output.get("pages") or [{}])[0].get("res") or {}
    proposals = propose_tables_from_ocr(
        result.get("rec_boxes") or [],
        result.get("rec_texts") or [],
    )
    return [_mapped_box(proposal.box, to_page) for proposal in proposals]


def _orientation_correction(output: dict[str, Any]) -> tuple[int, float]:
    result = (output.get("pages") or [{}])[0].get("res") or {}
    labels = result.get("label_names") or []
    scores = result.get("scores") or []
    if not labels or not scores:
        return 0, 0.0
    try:
        predicted = int(str(labels[0])) % 360
        confidence = float(scores[0])
    except (TypeError, ValueError):
        return 0, 0.0
    if predicted not in {0, 90, 180, 270}:
        return 0, confidence
    # Paddle applies positive OpenCV angles (counter-clockwise in image
    # coordinates); right_angle_rotation uses the inverse image convention.
    return (-predicted) % 360, confidence


_FINANCIAL_TOKEN = re.compile(
    r"(?:₹|\brs\.?\s*)?[-+]?\d[\d,]*(?:\.\d{1,2})?(?:\s*/-)?",
    re.IGNORECASE,
)


def _candidate_metrics(bundle: PageInferenceBundle) -> tuple[int, float, int, int]:
    token_count = len(bundle.tokens)
    confidence = float(median(token.confidence for token in bundle.tokens)) if bundle.tokens else 0
    financial_count = sum(
        bool(_FINANCIAL_TOKEN.fullmatch(token.text.strip())) for token in bundle.tokens
    )
    table_count = len(_merge_table_boxes(list(bundle.layout_boxes), list(bundle.geometry_boxes)))
    return token_count, confidence, financial_count, table_count


def _select_page_inference(
    bundles: tuple[PageInferenceBundle, ...],
) -> tuple[PageInferenceBundle, tuple[PreprocessingCandidate, ...]]:
    raw = bundles[0]
    raw_tokens, raw_confidence, raw_financial, raw_tables = _candidate_metrics(raw)
    raw_reconstruction = raw.reconstruction_score
    safe: list[PageInferenceBundle] = []
    for bundle in bundles[1:]:
        tokens, confidence, financial, tables = _candidate_metrics(bundle)
        retains_text = raw_tokens < 10 or tokens >= max(1, round(raw_tokens * 0.90))
        retains_confidence = confidence >= raw_confidence - 0.03
        retains_financial = financial >= raw_financial
        retains_tables = tables >= raw_tables
        retains_reconstruction = bundle.reconstruction_score >= raw_reconstruction
        improves = bool(
            bundle.reconstruction_score > raw_reconstruction
            or tables > raw_tables
            or financial > raw_financial
            or (tokens >= raw_tokens + max(3, round(raw_tokens * 0.10)))
            or confidence >= raw_confidence + 0.05
        )
        if (
            retains_text
            and retains_confidence
            and retains_financial
            and retains_tables
            and retains_reconstruction
            and improves
        ):
            safe.append(bundle)
    selected = max(
        safe,
        key=lambda bundle: (
            bundle.reconstruction_score,
            _candidate_metrics(bundle)[3],
            _candidate_metrics(bundle)[2],
            _candidate_metrics(bundle)[0],
            _candidate_metrics(bundle)[1],
            -list(PreprocessingVariant).index(bundle.candidate.contract.variant),
        ),
        default=raw,
    )
    annotated: list[PreprocessingCandidate] = []
    for bundle in bundles:
        token_count, confidence, financial_count, table_count = _candidate_metrics(bundle)
        is_selected = bundle is selected
        annotated.append(
            bundle.candidate.contract.model_copy(
                update={
                    "selected": is_selected,
                    "selection_reason": (
                        "selected_safe_structural_improvement"
                        if is_selected and bundle is not raw
                        else (
                            "selected_raw_no_safe_improvement"
                            if is_selected
                            else "rejected_no_safe_structural_improvement"
                        )
                    ),
                    "ocr_token_count": token_count,
                    "ocr_median_confidence": confidence,
                    "financial_token_count": financial_count,
                    "table_proposal_count": table_count,
                    "reconstruction_score": bundle.reconstruction_score,
                    "ocr_latency_ms": bundle.ocr_response.latency_ms,
                    "layout_latency_ms": bundle.layout_response.latency_ms,
                }
            )
        )
    return selected, tuple(annotated)


def _m5_variant(variant: PreprocessingVariant) -> TableCandidateVariant:
    return {
        PreprocessingVariant.RAW: TableCandidateVariant.ORIENTED_RAW,
        PreprocessingVariant.GEOMETRY_300: TableCandidateVariant.PROJECTIVE,
        PreprocessingVariant.CAMERA_400: TableCandidateVariant.PROJECTIVE_ENHANCED,
    }[variant]


def _m5_header_tokens(
    tokens: tuple[OcrToken, ...], box: tuple[int, int, int, int]
) -> tuple[str, ...]:
    scoped = tokens_in_box(tokens, box)
    top_limit = box[1] + (box[3] - box[1]) * 0.2
    return tuple(
        token.text
        for token in scoped
        if min(point.y for point in token.polygon.points) <= top_limit
    )


def _m5_reconstruction_metrics(reconstruction: ReconstructionResult) -> dict[str, int]:
    quality = reconstruction_quality(reconstruction)
    required_columns = (
        len(reconstruction.schema.column_centers) if reconstruction.schema is not None else 0
    )
    publishable_rows = max(0, quality[1])
    grounded_rows = max(0, quality[5])
    populated_cells = max(0, quality[4])
    evidence_linkage = (
        round(grounded_rows * 1_000_000 / publishable_rows) if publishable_rows else 0
    )
    return {
        "critical_error_count": max(0, -quality[0]),
        "required_column_count": required_columns,
        "grounded_complete_row_count": grounded_rows,
        "evidence_linkage_ppm": evidence_linkage,
        "arithmetic_consistency_count": int(quality[0] == 0 and populated_cells > 0),
        "cross_channel_agreement_ppm": 0,
        "conflict_count": max(0, -quality[2]),
        "duplicate_row_count": 0,
        "missing_row_count": max(0, publishable_rows - grounded_rows),
        "publishable_row_count": publishable_rows,
        "populated_cell_count": populated_cells,
    }


def _m5_linear_shadow_decision(
    *,
    source_sha256: str,
    page_asset: PageAsset,
    bundles: tuple[PageInferenceBundle, ...],
    prior_schemas: tuple[TableSchemaState, ...],
    extra_proposals: tuple[TableProposal, ...] = (),
    extra_reconstructions: dict[str, ReconstructionResult] | None = None,
) -> dict[str, Any]:
    """Build the M5 decision graph without changing the published page selection."""

    proposals: list[TableProposal] = []
    reconstructions: dict[str, ReconstructionResult] = {}
    for bundle in bundles:
        variant = _m5_variant(bundle.candidate.contract.variant)
        source_boxes = _merge_table_boxes(list(bundle.layout_boxes), list(bundle.geometry_boxes))
        for reading_order, raw_box in enumerate(source_boxes):
            source_box = _safe_box(
                raw_box,
                page_asset.width,
                page_asset.height,
                horizontal_padding=20,
                vertical_padding=100,
            )
            candidate_box = _bounded_mapped_box(
                source_box,
                bundle.candidate.contract.transform.forward_matrix,
                width=bundle.candidate.contract.width,
                height=bundle.candidate.contract.height,
            )
            proposal_id = canonical_sha256(
                {
                    "policy": MATCH_POLICY_VERSION,
                    "source_sha256": source_sha256,
                    "page_number": page_asset.page_number,
                    "variant": variant.value,
                    "page_artifact_sha256": bundle.candidate.contract.artifact_sha256,
                    "source_box": source_box,
                    "reading_order": reading_order,
                }
            )
            reconstruction = reconstruct_ocr_rows(
                tokens_in_box(bundle.tokens, source_box),
                page_number=page_asset.page_number,
                table_id=proposal_id,
                box=source_box,
                prior_schemas=prior_schemas,
            )
            reconstructions[proposal_id] = reconstruction
            quality = reconstruction_quality(reconstruction)
            financial_count = sum(
                1
                for token in tokens_in_box(bundle.tokens, source_box)
                if parse_decimal(token.text) is not None
            )
            proposals.append(
                TableProposal(
                    proposal_id=proposal_id,
                    source_sha256=source_sha256,
                    page_number=page_asset.page_number,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                    variant=variant,
                    source_box=tuple(float(value) for value in source_box),
                    reading_order=reading_order,
                    header_tokens=_m5_header_tokens(bundle.tokens, source_box),
                    table_type=(
                        reconstruction.schema.table_type.value
                        if reconstruction.schema is not None
                        else str(reconstruction.diagnostics.get("table_type") or "unknown")
                    ),
                    page_artifact_sha256=bundle.candidate.contract.artifact_sha256,
                    candidate_box=candidate_box,
                    transform_valid=True,
                    distortion=abs(bundle.candidate.contract.quality.estimated_skew_degrees),
                    metrics={
                        "lineage_valid": True,
                        "reconstruction_score": quality,
                        "financial_token_count": financial_count,
                        "header_token_count": len(_m5_header_tokens(bundle.tokens, source_box)),
                        "table_confidence_ppm": 1_000_000,
                        "ocr_coverage_ppm": min(
                            1_000_000,
                            len(tokens_in_box(bundle.tokens, source_box)) * 10_000,
                        ),
                        "ocr_confidence_ppm": round(
                            median(
                                [
                                    token.confidence
                                    for token in tokens_in_box(bundle.tokens, source_box)
                                ]
                                or [0.0]
                            )
                            * 1_000_000
                        ),
                    },
                )
            )

    proposals.extend(extra_proposals)
    reconstructions.update(extra_reconstructions or {})
    matching = match_logical_tables(tuple(proposals))
    proposal_by_id = {item.proposal_id: item for item in proposals}
    logical_tables: list[dict[str, Any]] = []
    for logical in matching.logical_tables:
        group = [proposal_by_id[item] for item in logical.proposal_ids]
        finalists = select_stage_one(group)
        scored = tuple(
            (proposal, _m5_reconstruction_metrics(reconstructions[proposal.proposal_id]))
            for proposal in finalists
        )
        winner, winner_metrics = select_stage_two(scored)
        ranked = sorted(scored, key=lambda item: stage_two_key(item[0], item[1]), reverse=True)
        logical_tables.append(
            {
                "logical_table_id": logical.logical_table_id,
                "anchor_proposal_id": logical.anchor_proposal_id,
                "proposal_ids": list(logical.proposal_ids),
                "derivative_only": logical.derivative_only,
                "grounded_rescue": logical.grounded_rescue,
                "finalist_proposal_ids": [item.proposal_id for item in finalists],
                "selected_proposal_id": winner.proposal_id,
                "selected_variant": winner.variant.value,
                "selected_metrics": dict(winner_metrics),
                "candidate_ranking": [
                    {
                        "proposal_id": proposal.proposal_id,
                        "rank": rank,
                        "selected": proposal.proposal_id == winner.proposal_id,
                        "metrics": dict(metrics),
                    }
                    for rank, (proposal, metrics) in enumerate(ranked, 1)
                ],
            }
        )
    return {
        "policy_version": MATCH_POLICY_VERSION,
        "mode": "shadow",
        "status": "complete",
        "page_number": page_asset.page_number,
        "proposal_count": len(proposals),
        "logical_tables": logical_tables,
        "proposals": [
            {
                "proposal_id": item.proposal_id,
                "variant": item.variant.value,
                "source_box": list(item.source_box),
                "candidate_box": list(item.candidate_box or ()),
                "reading_order": item.reading_order,
                "page_artifact_sha256": item.page_artifact_sha256,
                "header_tokens": list(item.header_tokens),
                "table_type": item.table_type,
                "transform_valid": item.transform_valid,
                "distortion": item.distortion,
                "stage_one_metrics": dict(item.metrics),
                "stage_two_metrics": _m5_reconstruction_metrics(
                    reconstructions[item.proposal_id]
                ),
            }
            for item in sorted(proposals, key=lambda value: value.proposal_id)
        ],
        "edges": [
            {
                "anchor_proposal_id": edge.anchor_proposal_id,
                "candidate_proposal_id": edge.candidate_proposal_id,
                "candidate_variant": edge.candidate_variant.value,
                "features": asdict(edge.features),
                "accepted": edge.accepted,
                "reason": edge.reason,
            }
            for edge in matching.edges
        ],
    }


def _m5_map_uvdoc_box_to_source(
    box: tuple[int, int, int, int],
    *,
    mapping: DenseBackwardGridMapping,
    grid: Any,
    oriented_to_source: Matrix,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    def bounded_child(x: int, y: int) -> tuple[float, float]:
        return (
            float(max(0, min(mapping.child_width - 1, x))),
            float(max(0, min(mapping.child_height - 1, y))),
        )

    child_points = (
        bounded_child(left, top),
        bounded_child(right - 1, top),
        bounded_child(right - 1, bottom - 1),
        bounded_child(left, bottom - 1),
    )
    oriented = map_dense_points(mapping, grid, child_points)
    source = apply_matrix(oriented_to_source, oriented)
    xs = [point[0] for point in source]
    ys = [point[1] for point in source]
    return (
        max(0, min(source_width - 1, int(min(xs)))),
        max(0, min(source_height - 1, int(min(ys)))),
        max(1, min(source_width, int(max(xs)) + 1)),
        max(1, min(source_height, int(max(ys)) + 1)),
    )


def _m5_uvdoc_shadow_materials(
    *,
    source_sha256: str,
    page_asset: PageAsset,
    run: UvdocPreparedRun | None,
    artifact_root: Path,
    orientation_degrees: int,
    prior_schemas: tuple[TableSchemaState, ...],
    ocr: PaddleOcrV6Adapter,
    layout: PaddleDocLayoutV3Adapter,
) -> tuple[tuple[TableProposal, ...], dict[str, ReconstructionResult]]:
    """Screen valid UVDoc branches and map their observations back to SOURCE_RAW."""

    if (
        run is None
        or run.status != "valid"
        or run.image_relative_path is None
        or run.image_sha256 is None
        or run.enhanced_relative_path is None
        or run.enhanced_sha256 is None
        or run.grid_relative_path is None
        or run.grid_sha256 is None
        or run.grid_shape is None
        or run.width is None
        or run.height is None
    ):
        return (), {}
    orientation_matrix, oriented_width, oriented_height = right_angle_rotation(
        orientation_degrees, page_asset.width, page_asset.height
    )
    oriented_to_source = invert(orientation_matrix)
    mapping = DenseBackwardGridMapping(
        grid_relative_path=run.grid_relative_path,
        grid_sha256=run.grid_sha256,
        grid_shape=run.grid_shape,
        child_width=run.width,
        child_height=run.height,
        parent_width=oriented_width,
        parent_height=oriented_height,
        padding_mode="zeros",
    )
    grid = load_dense_grid(artifact_root, mapping)
    proposals: list[TableProposal] = []
    reconstructions: dict[str, ReconstructionResult] = {}
    branches = (
        (
            TableCandidateVariant.UVDOC,
            artifact_root / run.image_relative_path,
            run.image_sha256,
        ),
        (
            TableCandidateVariant.UVDOC_ENHANCED,
            artifact_root / run.enhanced_relative_path,
            run.enhanced_sha256,
        ),
    )
    for variant, path, artifact_sha256 in branches:
        suffix = variant.value
        request = InferenceRequest(
            request_id=str(uuid4()),
            artifact_sha256=artifact_sha256,
            image_path=str(path.resolve()),
            page_number=page_asset.page_number,
            options={"preprocessing_policy": MATCH_POLICY_VERSION, "table_stage": "m5_screen"},
        )
        ocr_response, _ocr_cache_hit = _cached_prediction(
            artifact_root / "inference" / f"page-{page_asset.page_number}.{suffix}.ocr.json",
            request,
            ocr,
        )
        layout_response, _layout_cache_hit = _cached_prediction(
            artifact_root / "inference" / f"page-{page_asset.page_number}.{suffix}.layout.json",
            request,
            layout,
        )
        child_tokens = paddle_ocr_tokens(
            ocr_response.output,
            page_asset.page_number,
            artifact_sha256,
        )
        mapped_tokens: list[OcrToken] = []
        for token in child_tokens:
            child_points = tuple(
                (
                    max(0.0, min(float(mapping.child_width - 1), point.x)),
                    max(0.0, min(float(mapping.child_height - 1), point.y)),
                )
                for point in token.polygon.points
            )
            oriented = map_dense_points(mapping, grid, child_points)
            source_points = apply_matrix(oriented_to_source, oriented)
            mapped_tokens.append(
                token.model_copy(
                    update={
                        "artifact_sha256": page_asset.artifact_sha256,
                        "polygon": Polygon(
                            points=tuple(Point(x=point[0], y=point[1]) for point in source_points)
                        ),
                    }
                )
            )
        child_boxes = _merge_table_boxes(
            list(_layout_boxes(layout_response.output)),
            list(_ocr_geometry_boxes(ocr_response.output, transform_identity())),
        )
        for reading_order, child_box in enumerate(child_boxes):
            source_box = _m5_map_uvdoc_box_to_source(
                child_box,
                mapping=mapping,
                grid=grid,
                oriented_to_source=oriented_to_source,
                source_width=page_asset.width,
                source_height=page_asset.height,
            )
            source_box = _safe_box(
                source_box,
                page_asset.width,
                page_asset.height,
                horizontal_padding=20,
                vertical_padding=100,
            )
            proposal_id = canonical_sha256(
                {
                    "policy": MATCH_POLICY_VERSION,
                    "source_sha256": source_sha256,
                    "page_number": page_asset.page_number,
                    "variant": variant.value,
                    "page_artifact_sha256": artifact_sha256,
                    "source_box": source_box,
                    "reading_order": reading_order,
                }
            )
            scoped_tokens = tokens_in_box(tuple(mapped_tokens), source_box)
            reconstruction = reconstruct_ocr_rows(
                scoped_tokens,
                page_number=page_asset.page_number,
                table_id=proposal_id,
                box=source_box,
                prior_schemas=prior_schemas,
            )
            reconstructions[proposal_id] = reconstruction
            quality = reconstruction_quality(reconstruction)
            transform_metrics = run.transform_metrics or {}
            distortion = float(transform_metrics.get("max_displacement_px", 0.0)) / max(
                1, max(run.width, run.height)
            )
            proposals.append(
                TableProposal(
                    proposal_id=proposal_id,
                    source_sha256=source_sha256,
                    page_number=page_asset.page_number,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                    variant=variant,
                    source_box=tuple(float(value) for value in source_box),
                    reading_order=reading_order,
                    header_tokens=_m5_header_tokens(tuple(mapped_tokens), source_box),
                    table_type=(
                        reconstruction.schema.table_type.value
                        if reconstruction.schema is not None
                        else str(reconstruction.diagnostics.get("table_type") or "unknown")
                    ),
                    page_artifact_sha256=artifact_sha256,
                    candidate_box=child_box,
                    transform_valid=True,
                    distortion=distortion,
                    metrics={
                        "lineage_valid": True,
                        "reconstruction_score": quality,
                        "financial_token_count": sum(
                            parse_decimal(token.text) is not None for token in scoped_tokens
                        ),
                        "header_token_count": len(
                            _m5_header_tokens(tuple(mapped_tokens), source_box)
                        ),
                        "table_confidence_ppm": 1_000_000,
                        "ocr_coverage_ppm": min(1_000_000, len(scoped_tokens) * 10_000),
                        "ocr_confidence_ppm": round(
                            median([token.confidence for token in scoped_tokens] or [0.0])
                            * 1_000_000
                        ),
                    },
                )
            )
    return tuple(proposals), reconstructions


def _raw_only_preprocessing_record(
    page: PageAsset,
    quality: PageQuality,
    raw_relative_path: str,
) -> PagePreprocessingRecord:
    """Upgrade an untouched historical page without claiming new inference."""
    matrix = transform_identity()
    candidate = PreprocessingCandidate(
        variant=PreprocessingVariant.RAW,
        artifact_sha256=page.artifact_sha256,
        artifact_relative_path=raw_relative_path,
        width=page.width,
        height=page.height,
        dpi=page.dpi,
        transform=TransformChain(
            page_number=page.page_number,
            source_width=page.width,
            source_height=page.height,
            derived_width=page.width,
            derived_height=page.height,
            forward_matrix=matrix,
            inverse_matrix=matrix,
        ),
        quality=quality,
        selected=True,
        selection_reason="selected_raw_historical_compatibility",
    )
    return PagePreprocessingRecord(
        page_number=page.page_number,
        raw_artifact_sha256=page.artifact_sha256,
        raw_artifact_relative_path=raw_relative_path,
        raw_quality=quality,
        candidates=(candidate,),
        selected_variant=PreprocessingVariant.RAW,
    )


def _page_token_manifest_entry(
    token: OcrToken,
    *,
    artifact_relative_path: str,
    table_ids: tuple[str, ...] = (),
    source: tuple[OcrToken, PreprocessingCandidate] | None = None,
) -> TokenManifestEntry:
    source_fields: dict[str, Any] = {}
    if source is not None:
        source_token, candidate = source
        source_fields = {
            "source_artifact_sha256": candidate.artifact_sha256,
            "source_artifact_relative_path": candidate.artifact_relative_path,
            "source_polygon": source_token.polygon,
            "source_width": candidate.width,
            "source_height": candidate.height,
            "source_to_page_matrix": candidate.transform.inverse_matrix,
        }
    return TokenManifestEntry(
        token_id=token.token_id,
        page_number=token.page_number,
        table_ids=table_ids,
        text=token.text,
        polygon=token.polygon,
        artifact_sha256=token.artifact_sha256,
        artifact_relative_path=artifact_relative_path,
        confidence=token.confidence,
        **source_fields,
    )


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
    amount_present = bool(re.search(r"(?:₹|\brs\.?\s*)\d|\d[\d,]*\.\d{2}\b", normalized_text))
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
            (amount_present and (signals or reference_present or date_present))
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
        column for column in table.columns if column.canonical_field == "description"
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
            "serial" in re.sub(r"[^a-z0-9]+", " ", column.label.casefold()).split()
            or re.sub(r"[^a-z0-9]+", "", column.label.casefold()).startswith("sr")
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
            " ".join(cells[column.id].raw_value or "" for column in description_columns).casefold(),
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
            description for feature in features for description in feature.descriptions
        ),
        serials=frozenset(serial for feature in features for serial in feature.serials),
        numeric_values=tuple(
            sorted(value for feature in features for value in feature.numeric_values)
        ),
        complete_financial_rows=sum(feature.complete_financial_rows for feature in features),
        populated_canonical_fields=max(
            (feature.populated_canonical_fields for feature in features),
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
    inside = tuple(left <= point.x <= right and top <= point.y <= bottom for point in points)
    if all(inside):
        return "inside"
    polygon_left = min(point.x for point in points)
    polygon_top = min(point.y for point in points)
    polygon_right = max(point.x for point in points)
    polygon_bottom = max(point.y for point in points)
    if polygon_right < left or polygon_left > right or polygon_bottom < top or polygon_top > bottom:
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
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
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
            crop_left + (match_left + representative.shape[1]) * scale_x,
            crop_top + (match_top + representative.shape[0]) * scale_y,
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
    canonical_rows = tuple(row for row in rows if (row.page_number, row.table_id) == key)
    if not canonical_rows or any(
        not row.evidence
        or not all(
            _evidence_region_position(evidence, region) == "inside" for evidence in row.evidence
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
                    _evidence_region_position(evidence, region) for evidence in cell.evidence
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
    physical_tables = [tuple(grouped[key]) for key in physical_keys]
    features = [_merged_printed_table_features(group) for group in physical_tables]
    connected: dict[int, set[int]] = {index: {index} for index in range(len(physical_keys))}
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
            serial_overlap = len(left_features.serials & right_features.serials)
            description_overlap = len(left_features.descriptions & right_features.descriptions)
            numeric_overlap = sum(
                (
                    Counter(left_features.numeric_values) & Counter(right_features.numeric_values)
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
            if not candidate_features.serials.issubset(features[keep].serials):
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
    selected_rows = [row for row in rows if (row.page_number, row.table_id) not in suppressed_keys]
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

    cell_xs = tuple(point.x for evidence in cell.evidence for point in evidence.polygon.points)
    column_xs = tuple(point.x for evidence in column.evidence for point in evidence.polygon.points)
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
    return table_width > 0 and abs(cell_center - column_center) > table_width * 0.08


def _source_cell_is_in_rotated_overlay_cluster(
    table: SourceTable,
    source_row: SourceRow,
) -> bool:
    row_index = next(
        (index for index, candidate in enumerate(table.rows) if candidate.id == source_row.id),
        None,
    )
    if row_index is None:
        return False
    neighboring_rows = table.rows[max(0, row_index - 2) : min(len(table.rows), row_index + 3)]
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
        and len({candidate.column_id for _, candidate in rotated_structured_cells}) >= 2
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
    if role == "service_code" and _contains_service_code_fragment(cell.raw_value):
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
                    token_id for token_id in evidence.token_ids if token_id in token_ids
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
        column.canonical_field: column for column in columns if column.canonical_field is not None
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
        merged_residual = merged_tail[canonical_description_prefix.end() :].strip(" -:;/,[]")
    else:
        merged_description = merged_tail
        merged_residual = ""
    existing_description = (
        ""
        if cells_by_id[description_column.id] is merged_cell
        else (cells_by_id[description_column.id].raw_value or "")
    )
    printed_description, _, _ = _clean_description(
        " ".join(value for value in (merged_description, existing_description) if value)
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
        cells_by_id[date_column.id] is not merged_cell and cells_by_id[date_column.id].raw_value
    ) or (
        cells_by_id[request_column.id] is not merged_cell
        and cells_by_id[request_column.id].raw_value
    ):
        return cells

    split_flag = "split_from_merged_ocr_token"
    merged_token_ids = {
        token_id for evidence in merged_cell.evidence for token_id in evidence.token_ids
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
        source_ids = {token_id for evidence in source_evidence for token_id in evidence.token_ids}
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
        grounded_evidence("description", description_sources) if merged_description else ()
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
                            *(flag for flag in cell.validation_flags if flag != "empty_cell"),
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
        column for column in columns if column.canonical_field == "service_date_raw"
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
        cell_token_ids = {token_id for evidence in cell.evidence for token_id in evidence.token_ids}
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
                        *(flag for flag in date_cell.validation_flags if flag != "empty_cell"),
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
        (column for column in columns if column.canonical_field == "description"),
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
    printed_description = merged_cell.raw_value[description_match.start() : description_match.end()]
    leading_residual = merged_cell.raw_value[: description_match.start()].strip(" -:;/,[]")
    trailing_residual = merged_cell.raw_value[description_match.end() :].strip(" -:;/,[]")
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
    merged_ids = {token_id for evidence in merged_cell.evidence for token_id in evidence.token_ids}
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
        and normalized_adjacent_label in {"#", "no", "s no", "serial no", "sr n", "sr no"}
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
    if not residual or not description_evidence or not residual_evidence:
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
        (column for column in columns if column.canonical_field == "description"),
        None,
    )
    if description_column is None or not canonical.description:
        return cells
    adjacent_column = next(
        (column for column in columns if column.order == description_column.order + 1),
        None,
    )
    if adjacent_column is None:
        return cells

    cells_by_id = {cell.column_id: cell for cell in cells}
    description_cell = cells_by_id[description_column.id]
    adjacent_cell = cells_by_id[adjacent_column.id]
    description_value = description_cell.raw_value or ""
    adjacent_value = adjacent_cell.raw_value or ""
    if not description_value or not adjacent_value or not adjacent_cell.evidence:
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
        token_id for evidence in description_cell.evidence for token_id in evidence.token_ids
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
        column.canonical_field: column for column in columns if column.canonical_field is not None
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
        remaining_description = " ".join((remaining_description, description_cell.raw_value))
    if not remaining_description or parse_service_date(printed_date) != canonical.service_date_iso:
        return cells
    date_token_ids = {
        token_id
        for evidence in canonical.field_evidence.get("service_date", ())
        for token_id in evidence.token_ids
    }
    merged_token_ids = {
        token_id for evidence in merged_cell.evidence for token_id in evidence.token_ids
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
        date_description, _, printed_request = _clean_description(merged_cell.raw_value)
        existing_description, _, _ = _clean_description(description_cell.raw_value or "")
        comparable_description = " ".join(
            value for value in (date_description, existing_description) if value
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
            token_id for evidence in description_source_evidence for token_id in evidence.token_ids
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
                        *(flag for flag in date_cell.validation_flags if flag != "empty_cell"),
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
        (column for column in table.columns if column.canonical_field == "description"),
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
        linked_cells = {cell.column_id: cell for cell in rows[linked_index].cells}
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
                cell for cell in donor_row.cells if cell.column_id == description_column.id
            )
            if not donor.raw_value or normalized(donor.raw_value) != normalized(
                canonical.description
            ):
                continue
            donor_ids = {token_id for evidence in donor.evidence for token_id in evidence.token_ids}
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
                            *(flag for flag in target.validation_flags if flag != "empty_cell"),
                            "redistributed_from_adjacent_source_row",
                        )
                    )
                ),
            }
        )
        rows[linked_index] = rows[linked_index].model_copy(
            update={
                "cells": tuple(linked_cells[cell.column_id] for cell in rows[linked_index].cells)
            }
        )
        donor_cells = {cell.column_id: cell for cell in rows[donor_index].cells}
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
            update={"cells": tuple(donor_cells[cell.column_id] for cell in rows[donor_index].cells)}
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
            "validation_flags": tuple(dict.fromkeys((*row.validation_flags, flags))),
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
        tuple(source_tables) if already_linked else _link_source_tables(source_tables, rows)
    )

    def parsed_date(iso: str) -> date:
        return date.fromisoformat(iso)

    trusted_dates = [parsed_date(row.service_date_iso) for row in rows if row.service_date_iso]
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
                if "expiry" in context or not any(marker in context for marker in anchor_markers):
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
            or normalized_description.startswith(("advance ", "payment ", "receipt ", "refund "))
            or normalized_description
            in {"advance", "cash", "cashless", "debit", "finance", "payment"}
            or re.fullmatch(r"rc\d[\da-z/-]*", normalized_description)
        ):
            return False
        candidate_date = parsed_date(candidate[1])
        anchor_close = bool(
            not trusted_dates
            or min(abs((candidate_date - anchor).days) for anchor in trusted_dates) <= 120
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
        nearby_support = sum(abs((candidate_date - other).days) <= 45 for other in dates_in_column)
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
            unique_dates = {(raw, iso) for _, (raw, iso, _) in candidates}
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
            if any(parse_decimal(cell.raw_value or "") is not None for cell in donor_row.cells):
                continue
            donor_cell, candidate = donor_candidates[0]
            previous = table.rows[donor_index - 1] if donor_index > 0 else None
            following = table.rows[donor_index + 1] if donor_index + 1 < len(table.rows) else None
            associated_previous = False
            if previous is not None and previous.canonical_row_id is not None:
                previous_cells = {cell.column_id: cell for cell in previous.cells}
                lane_value = (previous_cells[donor_cell.column_id].raw_value or "").strip()
                previous_row = rows[row_indexes[previous.canonical_row_id]]
                description_ids = {
                    token_id
                    for item in previous_row.field_evidence.get("description", ())
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
                    grouped_cells = {cell.column_id: cell for cell in grouped_row.cells}
                    lane_value = (grouped_cells[donor_cell.column_id].raw_value or "").strip()
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
                    cell.column_id == donor_cell.column_id or not (cell.raw_value or "").strip()
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
            cell_ids = {token_id for item in cell.evidence for token_id in item.token_ids}
            if date_evidence_ids.intersection(cell_ids):
                support[cell.column_id] += 1
    if not support:
        return table
    ordered = support.most_common()
    if ordered[0][1] < 2 or (len(ordered) > 1 and ordered[0][1] == ordered[1][1]):
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
            {token_id for item in cell.evidence for token_id in item.token_ids}
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
                    "Date" if "synthetic_header" in column.validation_flags else column.label
                ),
                "canonical_field": "service_date_raw",
                "validation_flags": tuple(
                    dict.fromkeys((*column.validation_flags, "inferred_column_role"))
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
    *,
    token_lookup: dict[str, TokenManifestEntry] | None = None,
) -> tuple[SourceTable, ...]:
    """Link only strict, unique mutual-best source/canonical pairs."""

    def lineage_roots(token_id: str) -> set[str]:
        pending = [token_id]
        roots: set[str] = set()
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            token = (token_lookup or {}).get(current)
            if token is None:
                roots.add(current)
                continue
            parents = tuple(token.parent_token_ids) or (
                (token.parent_token_id,) if token.parent_token_id else ()
            )
            if parents:
                pending.extend(parents)
            else:
                roots.add(current)
        return roots

    def evidence_ids(items: object) -> set[str]:
        return {
            root
            for item in items or ()
            for token_id in getattr(item, "token_ids", ())
            for root in lineage_roots(token_id)
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
            return (0, -(10**9), -(10**9))
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
                    if description_column is not None and cell.column_id == description_column.id
                ),
                "",
            )
            for candidate in candidates_by_table.get((table.page_number, table.table_id), ()):
                description_ids = evidence_ids(candidate.field_evidence.get("description", ()))
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
                canonical_id is not None and canonical_best.get(canonical_id) == source_key
            )
            flags = tuple(
                flag for flag in source_row.validation_flags if flag != "canonical_link_ambiguous"
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
                            dict.fromkeys((*cell.validation_flags, "excluded_oversized_overlay"))
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
                    if row.page_number == table.page_number and row.table_id == table.table_id
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
            if row.page_number == original.page_number and row.table_id == original.table_id
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
                issuer_context(receipt) and issuer_context(receipt) == issuer_context(other)
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
                        marker
                        in re.sub(
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
            issuer_raw = re.sub(r"\s+", " ", (issuer_cell.raw_value if issuer_cell else "")).strip()
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
                    (bounds(polygon)[0] + bounds(polygon)[2]) / 2 for polygon in polygons
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
                        token_id for evidence in cell.evidence for token_id in evidence.token_ids
                    )
                )
                parents = tuple(
                    token_lookup[token_id]
                    for token_id in parent_ids
                    if token_id in token_lookup
                    and token_lookup[token_id].parent_token_id is None
                    and not token_lookup[token_id].parent_token_ids
                )
                existing_fragment = (
                    token_lookup.get(parent_ids[0]) if len(parent_ids) == 1 else None
                )
                if (
                    existing_fragment is not None
                    and (existing_fragment.parent_token_id or existing_fragment.parent_token_ids)
                    and existing_fragment.fragment_role == aligned_role
                    and normalized(existing_fragment.text) == normalized(raw)
                ):
                    registered_fragment = fragments.get(
                        existing_fragment.token_id,
                        existing_fragment,
                    )
                    if table.table_id not in registered_fragment.table_ids:
                        fragments[existing_fragment.token_id] = registered_fragment.model_copy(
                            update={
                                "table_ids": tuple(
                                    sorted(
                                        {
                                            *registered_fragment.table_ids,
                                            table.table_id,
                                        }
                                    )
                                )
                            }
                        )
                    for parent_id in tuple(existing_fragment.parent_token_ids) or (
                        (existing_fragment.parent_token_id,)
                        if existing_fragment.parent_token_id
                        else ()
                    ):
                        assignments[(source_row.id, aligned_role, parent_id)] = (
                            existing_fragment.token_id
                        )
                    materialized_cells.append(
                        cell.model_copy(
                            update={
                                "validation_flags": tuple(
                                    flag
                                    for flag in cell.validation_flags
                                    if flag != "fragment_occurrence_ambiguous"
                                )
                            }
                        )
                    )
                    continue
                selected_spans: tuple[tuple[int, int], ...] | None = None
                selected_parents: tuple[TokenManifestEntry, ...] = ()
                fragment_polygon: Polygon | None = None
                if len(parents) > 1 and normalized(
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
                        for match in re.finditer(re.escape(raw), token.text)
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
                        "parent_token_ids": tuple(item.token_id for item in selected_parents),
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
        if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
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
    return [row.model_copy(update=updates_by_id.get(str(row.id), {})) for row in rows]


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
                        key: value for key, value in row.field_evidence.items() if key != "quantity"
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
        evidence = tuple(item for column in table.columns for item in column.evidence) + tuple(
            item for source_row in table.rows for cell in source_row.cells for item in cell.evidence
        )
        ordered_tables.setdefault(table.page_number, []).append((table, _evidence_bounds(evidence)))
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
        summary_block_ordinal=(int(ordinal_match.group("ordinal")) if ordinal_match else 0),
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
        baseline_units = [
            value
            for (page, identity), value in baseline_tables.items()
            if page == page_number and (table_id is None or identity == table_id)
        ]
        candidate_units = [
            value
            for (page, identity), value in selected_tables.items()
            if page == page_number and (table_id is None or identity == table_id)
        ]
        status = next(
            (
                str(item.get("status"))
                for item in diagnostics
                if item.get("page_number") == page_number
                and (table_id is None or item.get("table_id") == table_id)
                and item.get("status")
                in {
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
    canonical_crop: CanonicalTableCrop | None = None

    def raw_digest(self) -> str:
        return _stable_payload_digest(
            {
                "page_number": self.page_number,
                "table_id": self.table_id,
                "source_table": self.source_table.model_dump(mode="json"),
                "row_candidates": [row.model_dump(mode="json") for row in self.row_candidates],
                "crop_relative_path": self.crop_relative_path,
                "crop_box": self.crop_box,
                "diagnostics": self.diagnostics,
                "normalized_fragments": [
                    item.model_dump(mode="json") for item in self.normalized_fragments
                ],
                "recovery_tokens": [item.model_dump(mode="json") for item in self.recovery_tokens],
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
                "canonical_crop": (
                    self.canonical_crop.model_dump(mode="json")
                    if self.canonical_crop is not None
                    else None
                ),
            }
        )


def _token_table_owners(
    manifest: Collection[TokenManifestEntry],
) -> dict[str, frozenset[str]]:
    """Attribute fragments and their complete parent lineage to source tables."""

    entries = {token.token_id: token for token in manifest}
    owners = {token_id: set(token.table_ids) for token_id, token in entries.items()}
    changed = True
    while changed:
        changed = False
        for token_id, token in entries.items():
            if not owners[token_id]:
                continue
            for parent_id in (
                *((token.parent_token_id,) if token.parent_token_id else ()),
                *token.parent_token_ids,
            ):
                if parent_id not in owners:
                    continue
                before = len(owners[parent_id])
                owners[parent_id].update(owners[token_id])
                changed = changed or len(owners[parent_id]) != before
    return {token_id: frozenset(value) for token_id, value in owners.items()}


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
    preprocessing: PagePreprocessingRecord | None = None

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
            "token_manifest": [item.model_dump(mode="json") for item in self.token_manifest],
            "source_tables": [unit.raw_digest() for unit in self.table_units],
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
            "preprocessing": (
                self.preprocessing.model_dump(mode="json") if self.preprocessing else None
            ),
        }
        return _stable_payload_digest(payload)

    def residual_digest(self) -> str:
        """Digest only the page-owned no-table recovery substrate."""

        table_owners = _token_table_owners(self.token_manifest)
        payload = {
            "page_asset": self.page_asset.model_dump(mode="json"),
            "ocr_tokens": [item.model_dump(mode="json") for item in self.ocr_tokens],
            "token_manifest": [
                item.model_dump(mode="json")
                for item in self.token_manifest
                if not table_owners.get(item.token_id)
            ],
            "unassigned_row_candidates": [
                row.model_dump(mode="json") for row in self.unassigned_row_candidates
            ],
            "diagnostics": [item for item in self.diagnostics if not item.get("table_id")],
            "total_candidates": [
                {
                    "total": item.total.model_dump(mode="json"),
                    "label_priority": item.label_priority,
                    "vertical_position": item.vertical_position,
                    "local_context": item.local_context,
                }
                for item in self.total_candidates
                if item.total.evidence.table_id is None
            ],
            "provider_usage": self.provider_usage,
            "preprocessing": (
                self.preprocessing.model_dump(mode="json") if self.preprocessing else None
            ),
        }
        return _stable_payload_digest(payload)


def _prefix_recovery_page_units(
    units: tuple[PageExtractionUnit, ...],
    prefix: str,
) -> tuple[PageExtractionUnit, ...]:
    """Make recovery artifacts addressable without overwriting baseline files."""

    def relative(value: str | None) -> str | None:
        return str(Path(prefix) / value) if value else value

    def diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
        output = deepcopy(payload)
        if isinstance(output.get("crop_relative_path"), str):
            output["crop_relative_path"] = relative(output["crop_relative_path"])
        return output

    output: list[PageExtractionUnit] = []
    for unit in units:
        token_manifest = tuple(
            token.model_copy(
                update={
                    "artifact_relative_path": relative(token.artifact_relative_path),
                    "source_artifact_relative_path": relative(token.source_artifact_relative_path),
                }
            )
            for token in unit.token_manifest
        )
        tokens_by_id = {token.token_id: token for token in token_manifest}
        table_units = tuple(
            replace(
                table,
                crop_relative_path=relative(table.crop_relative_path),
                canonical_crop=(
                    table.canonical_crop.model_copy(
                        update={
                            "artifact_relative_path": relative(
                                table.canonical_crop.artifact_relative_path
                            )
                        }
                    )
                    if table.canonical_crop is not None
                    else None
                ),
                diagnostics=tuple(diagnostic(item) for item in table.diagnostics),
                normalized_fragments=tuple(
                    tokens_by_id.get(fragment.token_id, fragment)
                    for fragment in table.normalized_fragments
                ),
            )
            for table in unit.table_units
        )
        output.append(
            replace(
                unit,
                page_asset=unit.page_asset.model_copy(
                    update={"relative_path": relative(unit.page_asset.relative_path)}
                ),
                token_manifest=token_manifest,
                table_units=table_units,
                diagnostics=tuple(diagnostic(item) for item in unit.diagnostics),
                preprocessing=(
                    unit.preprocessing.model_copy(
                        update={
                            "raw_artifact_relative_path": relative(
                                unit.preprocessing.raw_artifact_relative_path
                            ),
                            "candidates": tuple(
                                candidate.model_copy(
                                    update={
                                        "artifact_relative_path": relative(
                                            candidate.artifact_relative_path
                                        )
                                    }
                                )
                                for candidate in unit.preprocessing.candidates
                            ),
                        }
                    )
                    if unit.preprocessing is not None
                    else None
                ),
            )
        )
    return tuple(output)


def _merge_targeted_page_units(
    baseline: tuple[PageExtractionUnit, ...],
    recovered: tuple[PageExtractionUnit, ...],
    recovery_targets: tuple[tuple[int, str | None], ...],
) -> tuple[PageExtractionUnit, ...]:
    """Merge table or page-residual targets without replacing sibling tables."""

    recovered_by_page = {unit.page_asset.page_number: unit for unit in recovered}
    targets_by_page: dict[int, set[str | None]] = defaultdict(set)
    for page_number, table_id in recovery_targets:
        targets_by_page[page_number].add(table_id)

    def merge_page(unit: PageExtractionUnit) -> PageExtractionUnit:
        page_number = unit.page_asset.page_number
        targets = targets_by_page.get(page_number)
        candidate = recovered_by_page.get(page_number)
        if not targets or candidate is None:
            return unit

        table_units = list(unit.table_units)
        token_manifest = list(unit.token_manifest)
        diagnostics = list(unit.diagnostics)
        total_candidates = list(unit.total_candidates)
        unassigned_rows = unit.unassigned_row_candidates
        provider_usage = unit.provider_usage
        preprocessing = unit.preprocessing
        ocr_tokens = unit.ocr_tokens

        for table_id in sorted(
            (value for value in targets if value is not None),
        ):
            recovered_table = next(
                (item for item in candidate.table_units if item.table_id == table_id),
                None,
            )
            if recovered_table is None:
                continue
            table_units = [
                recovered_table if item.table_id == table_id else item for item in table_units
            ]
            if all(item.table_id != table_id for item in unit.table_units):
                table_units.append(recovered_table)
            diagnostics = [item for item in diagnostics if item.get("table_id") != table_id] + [
                deepcopy(item) for item in candidate.diagnostics if item.get("table_id") == table_id
            ]
            total_candidates = [
                item for item in total_candidates if item.total.evidence.table_id != table_id
            ] + [
                item
                for item in candidate.total_candidates
                if item.total.evidence.table_id == table_id
            ]
            baseline_owners = _token_table_owners(token_manifest)
            token_manifest = [
                item
                for item in token_manifest
                if baseline_owners.get(item.token_id) != frozenset({table_id})
            ]
            known_tokens = {item.token_id for item in token_manifest}
            candidate_owners = _token_table_owners(candidate.token_manifest)
            required_token_ids = {
                token_id for token_id, owners in candidate_owners.items() if table_id in owners
            }
            token_manifest.extend(
                item
                for item in candidate.token_manifest
                if item.token_id in required_token_ids and item.token_id not in known_tokens
            )

        if None in targets:
            unassigned_rows = candidate.unassigned_row_candidates
            provider_usage = candidate.provider_usage
            preprocessing = candidate.preprocessing
            ocr_tokens = candidate.ocr_tokens
            diagnostics = [item for item in diagnostics if item.get("table_id")] + [
                deepcopy(item) for item in candidate.diagnostics if not item.get("table_id")
            ]
            total_candidates = [
                item for item in total_candidates if item.total.evidence.table_id is not None
            ] + [
                item for item in candidate.total_candidates if item.total.evidence.table_id is None
            ]
            baseline_owners = _token_table_owners(token_manifest)
            table_tokens = [item for item in token_manifest if baseline_owners.get(item.token_id)]
            table_token_ids = {item.token_id for item in table_tokens}
            candidate_owners = _token_table_owners(candidate.token_manifest)
            token_manifest = table_tokens + [
                item
                for item in candidate.token_manifest
                if not candidate_owners.get(item.token_id) and item.token_id not in table_token_ids
            ]

        return replace(
            unit,
            table_units=tuple(table_units),
            token_manifest=tuple(token_manifest),
            unassigned_row_candidates=unassigned_rows,
            diagnostics=tuple(diagnostics),
            total_candidates=tuple(total_candidates),
            provider_usage=provider_usage,
            preprocessing=preprocessing,
            ocr_tokens=ocr_tokens,
        )

    return tuple(merge_page(unit) for unit in baseline)


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
    artifact_root: Path | None = None
    worker_release_revision: str | None = None
    validation_recovery_attempted: bool | None = None
    uvdoc_shadow_runs: tuple[UvdocPreparedRun, ...] = ()
    table_selection_runs: tuple[dict[str, Any], ...] = ()

    @property
    def result(self) -> dict[str, Any]:
        return _project_extraction_draft(self)

    @property
    def raw_unit_sha256(self) -> dict[str, str]:
        inventory: dict[str, str] = {}
        for unit in self.page_units:
            page_number = unit.page_asset.page_number
            inventory[f"page:{page_number}:residual"] = unit.residual_digest()
            inventory.update(
                {
                    f"table:{page_number}:{table.table_id}": table.raw_digest()
                    for table in unit.table_units
                }
            )
        return inventory


def _project_extraction_draft_v5(draft: ExtractionDraft) -> dict[str, Any]:
    source_tables = tuple(table for unit in draft.page_units for table in unit.source_tables)
    token_lookup = {
        token.token_id: token for unit in draft.page_units for token in unit.token_manifest
    }
    rows = _apply_document_role_policy(
        _deduplicate([row for unit in draft.page_units for row in unit.canonical_rows])
    )
    source_tables = _link_source_tables(
        source_tables,
        rows,
        token_lookup=token_lookup,
    )
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
    token_manifest = tuple(token_lookup.values())
    diagnostics = [deepcopy(item) for unit in draft.page_units for item in unit.diagnostics]
    projected_preprocessing = tuple(
        unit.preprocessing for unit in draft.page_units if unit.preprocessing is not None
    )
    has_complete_preprocessing = len(projected_preprocessing) == len(draft.page_units)
    projected_table_crops = tuple(
        table.canonical_crop
        for unit in draft.page_units
        for table in unit.table_units
        if table.canonical_crop is not None
    )
    table_unit_count = sum(len(unit.table_units) for unit in draft.page_units)
    has_complete_table_crops = len(projected_table_crops) == table_unit_count
    result: dict[str, Any] = {
        "output_version": "offline_accuracy_spine_v5",
        "contract_revision": (
            5
            if has_complete_preprocessing and has_complete_table_crops
            else (4 if has_complete_preprocessing else 3)
        ),
        "document_total_version": DOCUMENT_TOTAL_VERSION,
        "document_totals_version": DOCUMENT_TOTALS_VERSION,
        "document_total": (document_total.model_dump(mode="json") if document_total else None),
        "document_totals": [item.model_dump(mode="json") for item in document_totals],
        "raw_total_candidates": [item.model_dump(mode="json") for item in raw_total_candidates],
        "document_id": draft.document_id,
        "hospital_id": draft.hospital_id,
        "hospital": deepcopy(draft.hospital),
        "alias_registry_revision": draft.alias_registry_revision,
        "profile_registry_revision": draft.profile_registry_revision,
        "applied_alias_ids": list(draft.applied_alias_ids),
        "source_sha256": draft.source_sha256,
        "source_name": draft.source_name,
        "pages": len(draft.page_units),
        "page_assets": [unit.page_asset.model_dump(mode="json") for unit in draft.page_units],
        "page_preprocessing": [
            record.model_dump(mode="json") for record in projected_preprocessing
        ],
        "table_crops": [
            record.model_dump(mode="json")
            for record in (projected_table_crops if has_complete_table_crops else ())
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


def _relative_artifact_path(root: Path, path: str | Path) -> str:
    """Return a stable path rooted at the extraction artifact directory."""

    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root.resolve())
        except ValueError:
            candidate = Path(candidate.name)
    return candidate.as_posix()


def _project_extraction_draft(draft: ExtractionDraft) -> dict[str, Any]:
    """Project a draft to V6 when its immutable artifact substrate is present.

    Fixture-created drafts from the V5 era intentionally keep their old output;
    drafts produced by ``extract_draft`` always carry ``artifact_root`` and use
    the strict V6 writer below.
    """

    result = _project_extraction_draft_v5(draft)
    if draft.artifact_root is None:
        return result
    if len(result.get("page_preprocessing", ())) != int(result.get("pages") or 0):
        return result
    if len(result.get("table_crops", ())) != sum(
        len(unit.table_units) for unit in draft.page_units
    ):
        return result
    return _project_result_v6(
        result,
        draft.artifact_root,
        draft.uvdoc_shadow_runs,
        draft.table_selection_runs,
    )


def _project_result_v6(
    result: dict[str, Any],
    artifact_root: Path,
    uvdoc_runs: tuple[UvdocPreparedRun, ...] = (),
    table_selection_runs: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Build the deterministic V6 graph and convert the complete V5 payload.

    The extraction algorithms still operate on V5 internal models.  This
    boundary is deliberately the only place that changes their public shape,
    which keeps recovery selection and OCR/layout decisions unchanged.
    """

    pages = tuple(PageAsset.model_validate(item) for item in result.get("page_assets", ()))
    preprocessing = tuple(
        PagePreprocessingRecord.model_validate(item)
        for item in result.get("page_preprocessing", ())
    )
    crops = tuple(CanonicalTableCrop.model_validate(item) for item in result.get("table_crops", ()))
    if not pages or len(preprocessing) != len(pages):
        return result

    artifacts: list[ArtifactRef] = []
    page_artifacts: list[V6PageArtifact] = []
    page_raw_ids: dict[int, str] = {}
    page_selected_ids: dict[int, str] = {}
    artifact_by_sha: dict[str, list[ArtifactRef]] = defaultdict(list)
    table_artifacts: list[CanonicalTableArtifact] = []
    table_artifact_by_key: dict[tuple[int, str], ArtifactRef] = {}
    uvdoc_records: list[UvdocShadowRun] = []
    uvdoc_by_page = {run.page_number: run for run in uvdoc_runs}

    def add_artifact(ref: ArtifactRef) -> ArtifactRef:
        artifacts.append(ref)
        artifact_by_sha[ref.image_sha256].append(ref)
        return ref

    for page, record in zip(pages, preprocessing, strict=True):
        root_config = canonical_sha256(
            {
                "document_sha256": result["source_sha256"],
                "page_number": page.page_number,
                "renderer": page.renderer,
                "renderer_version": page.renderer_version,
                "dpi": page.dpi,
            }
        )
        raw = add_artifact(
            ArtifactRef(
                artifact_kind=ArtifactKind.SOURCE_RAW,
                image_sha256=page.artifact_sha256,
                artifact_relative_path=_relative_artifact_path(artifact_root, page.relative_path),
                width=page.width,
                height=page.height,
                producer=page.renderer,
                producer_version=page.renderer_version,
                configuration_sha256=root_config,
                child_to_parent_mapping=IdentityMapping(),
            )
        )
        page_raw_ids[page.page_number] = raw.artifact_id
        orientation_matrix, oriented_width, oriented_height = right_angle_rotation(
            record.orientation_degrees, page.width, page.height
        )
        oriented_path = Path(page.relative_path)
        oriented_sha = page.artifact_sha256
        if record.orientation_degrees:
            source_path = artifact_root / page.relative_path
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError("oriented_raw_source_image_unreadable")
            if record.orientation_degrees == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif record.orientation_degrees == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            else:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            oriented_path = Path("lineage") / "oriented" / f"page-{page.page_number:04d}.png"
            destination = artifact_root / oriented_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError("oriented_raw_image_write_failed")
            oriented_sha = sha256_file(destination)
        oriented_config = canonical_sha256(
            {
                "parent_configuration_sha256": root_config,
                "orientation_degrees": record.orientation_degrees,
            }
        )
        oriented = add_artifact(
            ArtifactRef(
                artifact_kind=ArtifactKind.ORIENTED_RAW,
                image_sha256=oriented_sha,
                artifact_relative_path=_relative_artifact_path(artifact_root, oriented_path),
                width=oriented_width,
                height=oriented_height,
                parent_artifact_id=raw.artifact_id,
                producer="gmoney.orientation",
                producer_version="orientation_v1",
                configuration_sha256=oriented_config,
                child_to_parent_mapping=(
                    IdentityMapping()
                    if record.orientation_degrees == 0
                    else HomographyMapping(child_to_parent_matrix=invert(orientation_matrix))
                ),
            )
        )
        candidates_by_variant: dict[PreprocessingVariant, ArtifactRef] = {}
        for candidate in record.candidates:
            if candidate.variant is PreprocessingVariant.RAW:
                candidates_by_variant[candidate.variant] = oriented
                continue
            candidate_kind = (
                ArtifactKind.PROJECTIVE_ENHANCED
                if candidate.variant is PreprocessingVariant.CAMERA_400
                else ArtifactKind.PROJECTIVE
            )
            candidate_config = canonical_sha256(
                {
                    "page_configuration_sha256": root_config,
                    "policy_version": record.policy_version,
                    "variant": candidate.variant.value,
                    "dpi": candidate.dpi,
                    "operations": candidate.transform.operations,
                }
            )
            candidates_by_variant[candidate.variant] = add_artifact(
                ArtifactRef(
                    artifact_kind=candidate_kind,
                    image_sha256=candidate.artifact_sha256,
                    artifact_relative_path=_relative_artifact_path(
                        artifact_root, candidate.artifact_relative_path
                    ),
                    width=candidate.width,
                    height=candidate.height,
                    parent_artifact_id=oriented.artifact_id,
                    producer="gmoney.preprocessing",
                    producer_version=record.policy_version,
                    configuration_sha256=candidate_config,
                    child_to_parent_mapping=HomographyMapping(
                        child_to_parent_matrix=compose(
                            candidate.transform.inverse_matrix,
                            orientation_matrix,
                        )
                    ),
                )
            )
        selected = candidates_by_variant[record.selected_variant]
        page_selected_ids[page.page_number] = selected.artifact_id
        page_artifacts.append(
            V6PageArtifact(
                artifact=raw,
                page_number=page.page_number,
                dpi=page.dpi,
                role="SOURCE_RAW",
                quality_metrics=record.raw_quality.model_dump(mode="json"),
            )
        )
        page_artifacts.append(
            V6PageArtifact(
                artifact=oriented,
                page_number=page.page_number,
                dpi=page.dpi,
                role="ORIENTED_RAW",
                selected=record.selected_variant is PreprocessingVariant.RAW,
                quality_metrics=record.raw_quality.model_dump(mode="json"),
                transform_metrics={"orientation_degrees": record.orientation_degrees},
            )
        )
        for candidate_variant, artifact in candidates_by_variant.items():
            if candidate_variant is PreprocessingVariant.RAW:
                continue
            candidate = next(
                item for item in record.candidates if item.variant is candidate_variant
            )
            page_artifacts.append(
                V6PageArtifact(
                    artifact=artifact,
                    page_number=page.page_number,
                    dpi=candidate.dpi,
                    role="CANDIDATE",
                    selected=candidate_variant is record.selected_variant,
                    quality_metrics=candidate.quality.model_dump(mode="json"),
                    route_reasons=candidate.route_reasons,
                    transform_metrics={"operations": candidate.transform.operations},
                )
            )
        shadow = uvdoc_by_page.get(page.page_number)
        if shadow is not None and shadow.status == "valid":
            compatibility = shadow.compatibility
            if (
                compatibility is None
                or shadow.parent_image_sha256 != oriented.image_sha256
                or shadow.image_relative_path is None
                or shadow.image_sha256 is None
                or shadow.enhanced_relative_path is None
                or shadow.enhanced_sha256 is None
                or shadow.grid_relative_path is None
                or shadow.grid_sha256 is None
                or shadow.grid_shape is None
                or shadow.width is None
                or shadow.height is None
            ):
                raise RuntimeError("uvdoc_shadow_artifact_metadata_incomplete")
            dense_mapping = DenseBackwardGridMapping(
                grid_relative_path=shadow.grid_relative_path,
                grid_sha256=shadow.grid_sha256,
                grid_shape=shadow.grid_shape,
                child_width=shadow.width,
                child_height=shadow.height,
                parent_width=oriented.width,
                parent_height=oriented.height,
                padding_mode="zeros",
            )
            uvdoc = add_artifact(
                ArtifactRef(
                    artifact_kind=ArtifactKind.UVDOC,
                    image_sha256=shadow.image_sha256,
                    artifact_relative_path=shadow.image_relative_path,
                    width=shadow.width,
                    height=shadow.height,
                    parent_artifact_id=oriented.artifact_id,
                    producer="gmoney.uvdoc",
                    producer_version=UVDOC_ADAPTER_VERSION,
                    configuration_sha256=compatibility.adapter_config_sha256,
                    child_to_parent_mapping=dense_mapping,
                )
            )
            enhanced_config = canonical_sha256(
                {
                    "parent_configuration_sha256": compatibility.adapter_config_sha256,
                    "operations": ("clahe:2.0,8x8", "unsharp:1.0,0.5"),
                }
            )
            enhanced = add_artifact(
                ArtifactRef(
                    artifact_kind=ArtifactKind.UVDOC_ENHANCED,
                    image_sha256=shadow.enhanced_sha256,
                    artifact_relative_path=shadow.enhanced_relative_path,
                    width=shadow.width,
                    height=shadow.height,
                    parent_artifact_id=uvdoc.artifact_id,
                    producer="gmoney.uvdoc_enhancement",
                    producer_version="uvdoc_enhancement_v1",
                    configuration_sha256=enhanced_config,
                    child_to_parent_mapping=IdentityMapping(),
                )
            )
            for candidate, reasons in (
                (uvdoc, ("uvdoc_shadow",)),
                (enhanced, ("uvdoc_shadow", "clahe", "unsharp")),
            ):
                page_artifacts.append(
                    V6PageArtifact(
                        artifact=candidate,
                        page_number=page.page_number,
                        dpi=page.dpi,
                        role="CANDIDATE",
                        selected=False,
                        route_reasons=reasons,
                        transform_metrics=shadow.transform_metrics or {},
                    )
                )
            uvdoc_records.append(
                UvdocShadowRun(
                    page_number=page.page_number,
                    status="valid",
                    reason_code=shadow.reason_code,
                    input_artifact_id=oriented.artifact_id,
                    uvdoc_artifact_id=uvdoc.artifact_id,
                    enhanced_artifact_id=enhanced.artifact_id,
                    model_sha256=compatibility.model_sha256,
                    model_config_sha256=compatibility.model_config_sha256,
                    adapter_config_sha256=compatibility.adapter_config_sha256,
                    paddle_version=compatibility.paddle_version,
                    paddleocr_version=compatibility.paddleocr_version,
                    paddlex_version=compatibility.paddlex_version,
                    reproduction_max_error_by_channel=(shadow.reproduction_max_error_by_channel),
                    transform_metrics=shadow.transform_metrics or {},
                    probe_metrics=shadow.probe_metrics or {},
                )
            )
        elif shadow is not None:
            uvdoc_records.append(
                UvdocShadowRun(
                    page_number=page.page_number,
                    status=shadow.status,
                    reason_code=shadow.reason_code,
                    transform_metrics=shadow.transform_metrics or {},
                    probe_metrics=shadow.probe_metrics or {},
                )
            )
    role_order = {"SOURCE_RAW": 0, "ORIENTED_RAW": 1, "CANDIDATE": 2}
    page_artifacts.sort(
        key=lambda item: (
            item.page_number,
            role_order.get(item.role or "CANDIDATE", 3),
            item.artifact.artifact_id,
        )
    )

    for crop in sorted(crops, key=lambda item: (item.page_number, item.table_id)):
        parent_id = page_selected_ids[crop.page_number]
        crop_ref = add_artifact(
            ArtifactRef(
                artifact_kind=ArtifactKind.TABLE_CROP,
                image_sha256=crop.artifact_sha256,
                artifact_relative_path=_relative_artifact_path(
                    artifact_root, crop.artifact_relative_path
                ),
                width=crop.width,
                height=crop.height,
                parent_artifact_id=parent_id,
                producer="gmoney.table_crop",
                producer_version="canonical_table_crop_v1",
                configuration_sha256=canonical_sha256(
                    {
                        "document_sha256": result["source_sha256"],
                        "page_number": crop.page_number,
                        "table_id": crop.table_id,
                        "candidate_box": crop.candidate_box,
                    }
                ),
                child_to_parent_mapping=HomographyMapping(
                    child_to_parent_matrix=translation(crop.candidate_box[0], crop.candidate_box[1])
                ),
            )
        )
        table_artifact_by_key[(crop.page_number, crop.table_id)] = crop_ref
        table_artifacts.append(
            CanonicalTableArtifact(
                artifact=crop_ref,
                page_number=crop.page_number,
                logical_table_id=crop.table_id,
                page_artifact_id=parent_id,
                crop_polygon_in_page_artifact=Polygon(
                    points=tuple(
                        Point(x=float(x), y=float(y))
                        for x, y in (
                            (crop.candidate_box[0], crop.candidate_box[1]),
                            (crop.candidate_box[2], crop.candidate_box[1]),
                            (crop.candidate_box[2], crop.candidate_box[3]),
                            (crop.candidate_box[0], crop.candidate_box[3]),
                        )
                    )
                ),
                crop_polygon_in_source_raw=crop.source_polygon,
                table_type_hint=None,
            )
        )

    by_sha = {sha: tuple(items) for sha, items in artifact_by_sha.items()}

    def ensure_recovery_artifact(token: TokenManifestEntry) -> None:
        if not token.source_artifact_sha256 or token.source_artifact_sha256 in by_sha:
            return
        owned_key = next(
            (
                (token.page_number, table_id)
                for table_id in token.table_ids
                if (token.page_number, table_id) in table_artifact_by_key
            ),
            None,
        )
        owned_table = table_artifact_by_key[owned_key] if owned_key is not None else None
        parent_id = (
            owned_table.artifact_id if owned_table is not None else page_raw_ids[token.page_number]
        )
        parent = next(item for item in artifacts if item.artifact_id == parent_id)
        source_path = token.source_artifact_relative_path or next(
            item.relative_path for item in pages if item.page_number == token.page_number
        )
        source_width = token.source_width or parent.width
        source_height = token.source_height or parent.height
        if token.source_to_page_matrix is None:
            mapping = IdentityMapping()
        elif owned_table is None:
            mapping = HomographyMapping(child_to_parent_matrix=token.source_to_page_matrix)
        else:
            assert owned_key is not None
            canonical_crop = next(
                item for item in crops if (item.page_number, item.table_id) == owned_key
            )
            mapping = HomographyMapping(
                child_to_parent_matrix=compose(
                    token.source_to_page_matrix,
                    invert(canonical_crop.crop_to_source_matrix),
                )
            )
        recovered = add_artifact(
            ArtifactRef(
                artifact_kind=ArtifactKind.CELL_CROP,
                image_sha256=token.source_artifact_sha256,
                artifact_relative_path=_relative_artifact_path(artifact_root, source_path),
                width=source_width,
                height=source_height,
                parent_artifact_id=parent_id,
                producer="gmoney.recovery",
                producer_version="recovery_v1",
                configuration_sha256=canonical_sha256(
                    {
                        "page_number": token.page_number,
                        "source_artifact_sha256": token.source_artifact_sha256,
                        "source_artifact_relative_path": source_path,
                    }
                ),
                child_to_parent_mapping=mapping,
            )
        )
        by_sha[token.source_artifact_sha256] = (recovered,)

    def artifact_for_token(token: TokenManifestEntry) -> ArtifactRef:
        if token.source_artifact_sha256:
            choices = by_sha.get(token.source_artifact_sha256, ())
            if choices:
                source_path = (
                    _relative_artifact_path(artifact_root, token.source_artifact_relative_path)
                    if token.source_artifact_relative_path
                    else None
                )
                matching_paths = tuple(
                    choice
                    for choice in choices
                    if source_path is not None and choice.artifact_relative_path == source_path
                )
                if matching_paths:
                    return min(matching_paths, key=lambda item: item.artifact_id)

                manifest = {item.artifact_id: item for item in artifacts}
                owned_table_ids = {
                    table_artifact_by_key[(token.page_number, table_id)].artifact_id
                    for table_id in token.table_ids
                    if (token.page_number, table_id) in table_artifact_by_key
                }
                for choice in sorted(choices, key=lambda item: item.artifact_id):
                    ancestor = choice
                    while True:
                        if ancestor.artifact_id in owned_table_ids:
                            return choice
                        if ancestor.parent_artifact_id is None:
                            break
                        ancestor = manifest[ancestor.parent_artifact_id]
                return min(choices, key=lambda item: item.artifact_id)
        choices = by_sha.get(token.artifact_sha256, ())
        if choices:
            return choices[0]
        return next(
            item for item in artifacts if item.artifact_id == page_raw_ids[token.page_number]
        )

    v2_tokens: list[TokenManifestEntryV2] = []
    for raw_token in result.get("token_manifest", ()):
        token = TokenManifestEntry.model_validate(raw_token)
        ensure_recovery_artifact(token)
        artifact = artifact_for_token(token)
        canonical_polygon = token.source_polygon or token.polygon
        source_polygon = token.polygon
        source_artifact_id = page_raw_ids[token.page_number]
        v2 = TokenManifestEntryV2(
            token_id=token.token_id,
            page_number=token.page_number,
            table_ids=token.table_ids,
            text=token.text,
            canonical_polygon=canonical_polygon,
            source_page_polygon=source_polygon,
            source_page_artifact_id=source_artifact_id,
            artifact_id=artifact.artifact_id,
            artifact_sha256=artifact.image_sha256,
            artifact_relative_path=artifact.artifact_relative_path,
            confidence=token.confidence,
            parent_token_id=token.parent_token_id,
            parent_token_ids=token.parent_token_ids,
            parent_character_spans=token.parent_character_spans,
            character_start=token.character_start,
            character_end=token.character_end,
            fragment_role=token.fragment_role,
        )
        v2_tokens.append(v2)
    v2_tokens.sort(key=lambda item: item.token_id)
    token_by_id = {item.token_id: item for item in v2_tokens}

    def artifact_to_source_matrix(artifact_id: str) -> Matrix:
        by_id = {item.artifact_id: item for item in artifacts}
        current = by_id[artifact_id]
        matrix = transform_identity()
        while current.parent_artifact_id is not None:
            mapping = current.child_to_parent_mapping
            if isinstance(mapping, HomographyMapping):
                matrix = compose(matrix, mapping.child_to_parent_matrix)
            current = by_id[current.parent_artifact_id]
        return matrix

    def evidence_v2(raw: EvidenceRef) -> EvidenceRefV2:
        cited = tuple(token_by_id.get(token_id) for token_id in raw.token_ids)
        if any(token is None for token in cited):
            raise ValueError("V1 evidence cites a token missing from the V2 manifest")
        cited_artifact_ids = {token.artifact_id for token in cited if token is not None}
        if len(cited_artifact_ids) > 1:
            raise ValueError("V1 evidence crosses canonical artifact boundaries")
        artifact_id = next(iter(cited_artifact_ids), None)
        if artifact_id is None:
            choices = by_sha.get(raw.artifact_sha256, ())
            artifact_id = (
                min(choices, key=lambda item: item.artifact_id).artifact_id
                if choices
                else page_raw_ids[raw.page_number]
            )
        artifact = next(item for item in artifacts if item.artifact_id == artifact_id)
        source_to_artifact = invert(artifact_to_source_matrix(artifact_id))
        canonical_points = apply_matrix(
            source_to_artifact,
            tuple((point.x, point.y) for point in raw.polygon.points),
        )
        canonical_polygon = Polygon(
            points=tuple(Point(x=max(0.0, x), y=max(0.0, y)) for x, y in canonical_points)
        )
        return EvidenceRefV2(
            artifact_id=artifact_id,
            artifact_sha256=artifact.image_sha256,
            canonical_polygon=canonical_polygon,
            source_page_polygon=raw.polygon,
            source_page_number=raw.page_number,
            source_page_artifact_id=page_raw_ids[raw.page_number],
            ocr_token_ids=raw.token_ids,
            extractor="gmoney.offline",
            model_name="paddleocr",
            model_version="v6",
            recognition_variant=(
                "canonical" if artifact.artifact_kind is ArtifactKind.TABLE_CROP else "page"
            ),
        )

    def source_table_v2(raw: SourceTable) -> SourceTableV2:
        return SourceTableV2(
            **raw.model_dump(exclude={"columns", "rows"}),
            columns=tuple(
                SourceColumnV2(
                    **column.model_dump(exclude={"evidence"}),
                    evidence=tuple(evidence_v2(item) for item in column.evidence),
                )
                for column in raw.columns
            ),
            rows=tuple(
                SourceRowV2(
                    **row.model_dump(exclude={"cells"}),
                    cells=tuple(
                        SourceCellV2(
                            **cell.model_dump(exclude={"evidence"}),
                            evidence=tuple(evidence_v2(item) for item in cell.evidence),
                        )
                        for cell in row.cells
                    ),
                )
                for row in raw.rows
            ),
        )

    source_tables = tuple(
        source_table_v2(SourceTable.model_validate(item))
        for item in result.get("source_tables", ())
    )
    rows: list[CanonicalRowV2] = []
    for item in result.get("rows", ()):
        raw_row = CanonicalRow.model_validate(item)
        rows.append(
            CanonicalRowV2(
                **raw_row.model_dump(exclude={"evidence", "field_evidence"}),
                evidence=tuple(evidence_v2(ref) for ref in raw_row.evidence),
                field_evidence={
                    name: tuple(evidence_v2(ref) for ref in refs)
                    for name, refs in raw_row.field_evidence.items()
                },
            )
        )

    totals: list[Any] = []
    for item in result.get("document_totals", ()):
        raw_total = DocumentTotal.model_validate(item)
        totals.append(
            DocumentTotalV2(
                **raw_total.model_dump(exclude={"evidence"}),
                evidence=evidence_v2(raw_total.evidence),
            )
        )
    document_total = result.get("document_total")
    typed_document_total = None
    if document_total is not None:
        raw_total = DocumentTotal.model_validate(document_total)
        typed_document_total = DocumentTotalV2(
            **raw_total.model_dump(exclude={"evidence"}),
            evidence=evidence_v2(raw_total.evidence),
        )
    raw_candidates: list[RawTotalCandidateV2] = []
    for item in result.get("raw_total_candidates", ()):
        candidate = RawTotalCandidate.model_validate(item)
        total = DocumentTotal.model_validate(candidate.total.model_dump(mode="json"))
        raw_candidates.append(
            RawTotalCandidateV2(
                **candidate.model_dump(exclude={"total", "context_evidence"}),
                total=DocumentTotalV2(
                    **total.model_dump(exclude={"evidence"}),
                    evidence=evidence_v2(total.evidence),
                ),
                context_evidence=tuple(evidence_v2(ref) for ref in candidate.context_evidence),
            )
        )
    all_evidence: dict[str, EvidenceRefV2] = {}
    for table in source_tables:
        for column in table.columns:
            for ref in column.evidence:
                all_evidence[canonical_json(ref).decode()] = ref
        for row in table.rows:
            for cell in row.cells:
                for ref in cell.evidence:
                    all_evidence[canonical_json(ref).decode()] = ref
    for row in rows:
        for ref in row.evidence:
            all_evidence[canonical_json(ref).decode()] = ref
    if typed_document_total:
        all_evidence[canonical_json(typed_document_total.evidence).decode()] = (
            typed_document_total.evidence
        )
    for candidate in raw_candidates:
        all_evidence[canonical_json(candidate.total.evidence).decode()] = candidate.total.evidence
        for ref in candidate.context_evidence:
            all_evidence[canonical_json(ref).decode()] = ref

    adapter_inputs: list[TableAdapterInputV2] = []
    for crop in crops:
        crop_ref = table_artifact_by_key[(crop.page_number, crop.table_id)]
        for trace in crop.adapter_inputs:
            adapter_inputs.append(
                TableAdapterInputV2(
                    page_number=crop.page_number,
                    logical_table_id=crop.table_id,
                    input_artifact_id=trace.input_artifact_id or crop_ref.artifact_id,
                    input_artifact_sha256=trace.input_artifact_sha256,
                    adapter_name=trace.adapter_name,
                    adapter_version=trace.adapter_version or "legacy-v5",
                    configuration_sha256=trace.configuration_sha256
                    or canonical_sha256(
                        {
                            "adapter": trace.adapter_name,
                            "stage": trace.stage,
                            "recognition_variant": trace.recognition_variant,
                        }
                    ),
                    latency_ms=trace.latency_ms,
                    cache_hit=trace.cache_hit,
                    accepted=trace.accepted,
                    stage=trace.stage,
                    recognition_variant=trace.recognition_variant,
                )
            )
    artifacts.sort(key=lambda item: (item.artifact_kind.value, item.artifact_id))
    table_artifacts.sort(key=lambda item: (item.page_number, item.logical_table_id))
    adapter_inputs.sort(
        key=lambda item: (
            item.page_number,
            item.logical_table_id,
            item.input_artifact_id,
            item.adapter_name,
            item.stage,
            item.recognition_variant,
        )
    )
    artifact_manifest = ArtifactManifest(artifacts=tuple(artifacts))
    page_candidate_by_key = {
        (item.page_number, item.artifact.artifact_kind): item.artifact for item in page_artifacts
    }
    candidate_kind_by_variant = {
        "oriented_raw": ArtifactKind.ORIENTED_RAW,
        "projective": ArtifactKind.PROJECTIVE,
        "projective_enhanced": ArtifactKind.PROJECTIVE_ENHANCED,
        "uvdoc": ArtifactKind.UVDOC,
        "uvdoc_enhanced": ArtifactKind.UVDOC_ENHANCED,
    }

    def m5_artifact(page_number: int, proposal: dict[str, Any]) -> ArtifactRef:
        variant = str(proposal["variant"])
        kind = candidate_kind_by_variant[variant]
        candidates = [
            artifact
            for artifact in artifact_by_sha.get(str(proposal.get("page_artifact_sha256")), ())
            if artifact.artifact_kind is kind
        ]
        if candidates:
            return candidates[0]
        fallback = page_candidate_by_key.get((page_number, kind))
        if fallback is None:
            raise RuntimeError("m5_candidate_artifact_missing")
        return fallback

    m5_tables: list[LogicalTableSelection] = []
    for raw_run in table_selection_runs:
        if raw_run.get("status") != "complete":
            continue
        page_number = int(raw_run["page_number"])
        proposals_by_id = {
            str(item["proposal_id"]): item for item in raw_run.get("proposals", ())
        }
        edges_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in raw_run.get("edges", ()):
            edges_by_anchor[str(edge["anchor_proposal_id"])].append(edge)
        for table in raw_run.get("logical_tables", ()):
            logical_id = str(table["logical_table_id"])
            anchor_id = table.get("anchor_proposal_id")
            typed_edges: list[TableMatchEdge] = []
            for edge in edges_by_anchor.get(str(anchor_id), ()) if anchor_id else ():
                anchor = proposals_by_id[str(edge["anchor_proposal_id"])]
                candidate = proposals_by_id[str(edge["candidate_proposal_id"])]
                features = edge["features"]
                reason = str(edge["reason"])
                ambiguous = reason == "abstained_ambiguous"
                typed_edges.append(
                    TableMatchEdge(
                        logical_table_id=logical_id,
                        page_number=page_number,
                        anchor_proposal_id=edge["anchor_proposal_id"],
                        candidate_proposal_id=edge["candidate_proposal_id"],
                        anchor_artifact_id=m5_artifact(page_number, anchor).artifact_id,
                        candidate_artifact_id=m5_artifact(page_number, candidate).artifact_id,
                        candidate_variant=edge["candidate_variant"],
                        match_score=features["weighted_score"],
                        overlap_score=features["polygon_iou"],
                        center_distance_score=features["center_proximity"],
                        reading_order_score=features["reading_order"],
                        table_type_score=features["table_type"],
                        header_similarity_score=features["header_similarity"],
                        decision=(
                            "accepted"
                            if edge["accepted"]
                            else ("ambiguous" if ambiguous else "rejected")
                        ),
                        ambiguous=ambiguous,
                        ambiguity_reason=reason if ambiguous else None,
                    )
                )
            evaluations: list[TableCandidateScore] = []
            for ranking in table.get("candidate_ranking", ()):
                proposal = proposals_by_id[str(ranking["proposal_id"])]
                metrics = ranking["metrics"]
                components = {
                    "critical_safety": 1.0
                    / (1.0 + float(metrics.get("critical_error_count", 0))),
                    "column_coverage": min(
                        1.0, float(metrics.get("required_column_count", 0)) / 5.0
                    ),
                    "grounded_rows": min(
                        1.0, float(metrics.get("grounded_complete_row_count", 0)) / 20.0
                    ),
                    "evidence_linkage": min(
                        1.0, float(metrics.get("evidence_linkage_ppm", 0)) / 1_000_000.0
                    ),
                    "arithmetic_consistency": min(
                        1.0, float(metrics.get("arithmetic_consistency_count", 0))
                    ),
                    "cross_channel_agreement": min(
                        1.0,
                        float(metrics.get("cross_channel_agreement_ppm", 0)) / 1_000_000.0,
                    ),
                    "conflict_safety": 1.0
                    / (1.0 + float(metrics.get("conflict_count", 0))),
                    "duplicate_safety": 1.0
                    / (1.0 + float(metrics.get("duplicate_row_count", 0))),
                    "completeness": 1.0
                    / (1.0 + float(metrics.get("missing_row_count", 0))),
                    "distortion_safety": 1.0 / (1.0 + float(proposal.get("distortion", 0))),
                }
                selected_in_shadow = bool(ranking["selected"])
                evaluations.append(
                    TableCandidateScore(
                        logical_table_id=logical_id,
                        page_number=page_number,
                        candidate_id=proposal["proposal_id"],
                        candidate_artifact_id=m5_artifact(page_number, proposal).artifact_id,
                        candidate_variant=proposal["variant"],
                        score=round(sum(components.values()) / len(components), 12),
                        rank=int(ranking["rank"]),
                        selected=selected_in_shadow,
                        evaluation_status=("selected" if selected_in_shadow else "eligible"),
                        score_components=components,
                    )
                )
            selected = next((item for item in evaluations if item.selected), None)
            m5_tables.append(
                LogicalTableSelection(
                    logical_table_id=logical_id,
                    page_number=page_number,
                    match_edges=tuple(typed_edges),
                    candidate_evaluations=tuple(evaluations),
                    decision="selected" if selected is not None else "no_candidate",
                    selected_candidate_id=(selected.candidate_id if selected is not None else None),
                    selected_candidate_variant=(
                        selected.candidate_variant if selected is not None else None
                    ),
                    selected_artifact_id=(
                        selected.candidate_artifact_id if selected is not None else None
                    ),
                )
            )
    typed_m5_runs = (
        (
            TableSelectionRun(
                document_id=result["document_id"],
                source_sha256=result["source_sha256"],
                mode="shadow",
                logical_tables=tuple(
                    sorted(m5_tables, key=lambda item: (item.page_number, item.logical_table_id))
                ),
            ),
        )
        if m5_tables
        else ()
    )
    diagnostics = tuple(
        ExtractionDiagnostic.model_validate(item) for item in result.get("diagnostics", ())
    )
    provider_usage = ProviderUsage.model_validate(result["provider_usage"])
    recovery = RecoveryMetadata.model_validate(result["recovery"])
    envelope = ExtractionResultV6(
        document_id=result["document_id"],
        source_sha256=result["source_sha256"],
        source_name=result["source_name"],
        pages=result["pages"],
        artifact_manifest=artifact_manifest,
        page_artifacts=tuple(page_artifacts),
        uvdoc_shadow_runs=tuple(sorted(uvdoc_records, key=lambda item: item.page_number)),
        table_selection_runs=typed_m5_runs,
        canonical_table_artifacts=tuple(table_artifacts),
        token_manifest=tuple(v2_tokens),
        evidence=tuple(all_evidence.values()),
        adapter_inputs=tuple(adapter_inputs),
        document_total_version=result["document_total_version"],
        document_totals_version=result["document_totals_version"],
        document_total=typed_document_total,
        document_totals=tuple(totals),
        raw_total_candidates=tuple(raw_candidates),
        hospital_id=result.get("hospital_id"),
        hospital=result.get("hospital"),
        alias_registry_revision=result.get("alias_registry_revision"),
        profile_registry_revision=result.get("profile_registry_revision"),
        applied_alias_ids=tuple(result.get("applied_alias_ids") or ()),
        page_assets=pages,
        page_preprocessing=preprocessing,
        source_tables=source_tables,
        suppressed_repeated_source_tables=tuple(
            SuppressedSourceTable.model_validate(item)
            for item in result.get("suppressed_repeated_source_tables", ())
        ),
        rows=tuple(rows),
        receipt_duplicate_pairs=tuple(
            ReceiptDuplicatePair.model_validate(item)
            for item in result.get("receipt_duplicate_pairs", ())
        ),
        diagnostics=diagnostics,
        provider_usage=provider_usage,
        recovery=recovery,
        worker_release_revision=result.get("worker_release_revision"),
        semantic_validation=result.get("semantic_validation"),
        validation_recovery_attempted=result.get("validation_recovery_attempted"),
    )
    return envelope.model_dump(mode="json")


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


def _token_lineage_signature(
    token_id: str,
    manifest: dict[str, TokenManifestEntry],
    seen: frozenset[str] = frozenset(),
) -> tuple[object, ...]:
    if token_id in seen:
        return ("cycle",)
    token = manifest.get(token_id)
    if token is None:
        return ("missing",)
    parent_ids = tuple(
        dict.fromkeys(
            (
                *((token.parent_token_id,) if token.parent_token_id else ()),
                *token.parent_token_ids,
            )
        )
    )
    if parent_ids:
        return (
            "fragment",
            token.fragment_role,
            _normalize_text(token.text),
            tuple(
                _token_lineage_signature(
                    parent_id,
                    manifest,
                    seen | {token_id},
                )
                for parent_id in parent_ids
            ),
            tuple(token.parent_character_spans),
            token.character_start,
            token.character_end,
        )
    return (
        "root",
        token.page_number,
        token.artifact_sha256,
        _normalize_text(token.text),
        tuple((round(point.x, 2), round(point.y, 2)) for point in token.polygon.points),
    )


def _source_financial_inventory(
    draft: ExtractionDraft,
    target: tuple[int, str | None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Inventory every grounded Printed financial row, independent of linking."""

    manifest = {token.token_id: token for unit in draft.page_units for token in unit.token_manifest}
    entries: list[dict[str, Any]] = []
    financial_fields = {
        "net_amount",
        "gross_amount",
        "unit_price",
        "discount",
    }
    for unit in draft.page_units:
        for table in unit.source_tables:
            if target is not None and (
                table.page_number != target[0]
                or (target[1] is not None and table.table_id != target[1])
            ):
                continue
            columns = {column.id: column for column in table.columns}
            canonical = {str(row.id): row for row in unit.canonical_rows}
            for source_row in table.rows:
                fields: dict[str, dict[str, Any]] = {}
                all_points: list[Point] = []
                description = ""
                for cell in source_row.cells:
                    column = columns.get(cell.column_id)
                    if column is None:
                        continue
                    if column.canonical_field == "description" and cell.raw_value:
                        description = _normalize_text(cell.raw_value)
                    is_financial = bool(
                        column.canonical_field in financial_fields
                        or "inferred_financial_lane" in column.validation_flags
                    )
                    if not is_financial or not cell.raw_value:
                        continue
                    amount = parse_decimal(cell.raw_value)
                    if amount is None:
                        continue
                    evidence = tuple(cell.evidence)
                    token_ids = tuple(token_id for item in evidence for token_id in item.token_ids)
                    if not token_ids:
                        continue
                    all_points.extend(point for item in evidence for point in item.polygon.points)
                    field = column.canonical_field or f"lane:{column.order}"
                    lineage = tuple(
                        _token_lineage_signature(token_id, manifest) for token_id in token_ids
                    )
                    fragment_strength = max(
                        (
                            2
                            if manifest.get(token_id) is not None
                            and manifest[token_id].fragment_role
                            in {field, "amount", "rate", "gross_amount", "discount"}
                            else 1
                        )
                        for token_id in token_ids
                    )
                    fields[field] = {
                        "value": str(amount),
                        "lineage_sha256": _stable_payload_digest(lineage),
                        "grounding_strength": fragment_strength,
                    }
                if not fields:
                    continue
                if all_points:
                    bounds = (
                        round(min(point.x for point in all_points), 2),
                        round(min(point.y for point in all_points), 2),
                        round(max(point.x for point in all_points), 2),
                        round(max(point.y for point in all_points), 2),
                    )
                else:
                    bounds = None
                canonical_row = canonical.get(source_row.canonical_row_id or "")
                entries.append(
                    {
                        "page_number": table.page_number,
                        "table_id": table.table_id,
                        "table_anchor": table.table_anchor,
                        "row_anchor": source_row.row_anchor,
                        "source_row_id": source_row.id,
                        "canonical_row_id": source_row.canonical_row_id,
                        "canonical_amount_missing": bool(
                            canonical_row is not None and canonical_row.net_amount is None
                        ),
                        "description": description,
                        "bounds": bounds,
                        "fields": fields,
                    }
                )
    return tuple(
        sorted(
            entries,
            key=lambda item: (
                item["page_number"],
                item["table_id"],
                item.get("row_anchor") or "",
                item["source_row_id"],
            ),
        )
    )


def _inventory_public_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Remove mutable link/row IDs before hashing a financial inventory."""

    return {
        key: value
        for key, value in entry.items()
        if key
        not in {
            "source_row_id",
            "canonical_row_id",
            "canonical_amount_missing",
        }
    }


def _financial_inventory_digest(entries: Collection[dict[str, Any]]) -> str:
    return _stable_payload_digest([_inventory_public_payload(entry) for entry in entries])


def _bounds_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _financial_inventory_matches(
    baseline: tuple[dict[str, Any], ...],
    candidate: tuple[dict[str, Any], ...],
    implicated_fields: dict[str, set[str]],
) -> tuple[bool, int, int]:
    """Return safe, preserved count, and added count using unique mutual-best matches."""

    def score(first: dict[str, Any], second: dict[str, Any]) -> float:
        if first["page_number"] != second["page_number"]:
            return -1.0
        value_overlap = bool(
            {(field, payload["value"]) for field, payload in first["fields"].items()}
            & {(field, payload["value"]) for field, payload in second["fields"].items()}
        )
        return (
            (0.20 if first["table_id"] == second["table_id"] else 0.0)
            + (0.35 if first.get("row_anchor") == second.get("row_anchor") else 0.0)
            + 0.20
            * SequenceMatcher(
                None,
                first.get("description") or "",
                second.get("description") or "",
            ).ratio()
            + 0.20 * _bounds_iou(first.get("bounds"), second.get("bounds"))
            + (0.15 if value_overlap else 0.0)
        )

    scores = {
        (left, right): score(baseline[left], candidate[right])
        for left in range(len(baseline))
        for right in range(len(candidate))
    }
    matches: dict[int, int] = {}
    for left in range(len(baseline)):
        ranked = sorted(
            ((scores[(left, right)], right) for right in range(len(candidate))),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.55:
            return False, len(matches), max(0, len(candidate) - len(matches))
        best_score, right = ranked[0]
        if len(ranked) > 1 and best_score - ranked[1][0] < 0.05:
            return False, len(matches), max(0, len(candidate) - len(matches))
        reverse_ranked = sorted(
            ((scores[(other, right)], other) for other in range(len(baseline))),
            reverse=True,
        )
        if reverse_ranked[0][1] != left or (
            len(reverse_ranked) > 1 and reverse_ranked[0][0] - reverse_ranked[1][0] < 0.05
        ):
            return False, len(matches), max(0, len(candidate) - len(matches))
        matches[left] = right

    if len(set(matches.values())) != len(matches):
        return False, len(matches), max(0, len(candidate) - len(matches))
    aliases = {
        "net_amount": "amount",
        "unit_price": "rate",
        "gross_amount": "gross_amount",
        "discount": "discount",
    }
    for left, right in matches.items():
        old = baseline[left]
        new = candidate[right]
        targeted = (
            set(implicated_fields.get(old.get("row_anchor") or "", set()))
            | set(implicated_fields.get(old.get("canonical_row_id") or "", set()))
            | set(implicated_fields.get(old.get("source_row_id") or "", set()))
        )
        for field, old_payload in old["fields"].items():
            new_payload = new["fields"].get(field)
            if new_payload is None:
                return False, len(matches), max(0, len(candidate) - len(matches))
            normalized_field = aliases.get(field, field)
            if old_payload["value"] != new_payload["value"]:
                if not ({field, normalized_field} & targeted):
                    return False, len(matches), max(0, len(candidate) - len(matches))
                if new_payload["grounding_strength"] <= old_payload["grounding_strength"]:
                    return False, len(matches), max(0, len(candidate) - len(matches))
            elif old_payload["lineage_sha256"] != new_payload["lineage_sha256"]:
                if new_payload["grounding_strength"] < old_payload["grounding_strength"]:
                    return False, len(matches), max(0, len(candidate) - len(matches))
    return True, len(matches), max(0, len(candidate) - len(matches))


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
        return {token.token_id: token for unit in draft.page_units for token in unit.token_manifest}

    baseline_tokens = token_lookup(baseline)
    candidate_tokens = token_lookup(candidate)

    baseline_inventory = _source_financial_inventory(baseline)
    candidate_inventory = _source_financial_inventory(candidate)
    inventory_safe, _preserved, _added = _financial_inventory_matches(
        baseline_inventory,
        candidate_inventory,
        implicated_fields,
    )
    if not inventory_safe:
        return False

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
        token_ids = tuple(token_id for item in evidence for token_id in item.token_ids)
        if not token_ids:
            return 0
        if len(token_ids) != 1 or token_ids[0] not in manifest:
            return 1
        token = manifest[token_ids[0]]
        expected_role = name
        return (
            int(
                bool(
                    expected_role
                    and token.fragment_role == expected_role
                    and (token.parent_token_id or token.parent_token_ids)
                )
            )
            + 1
        )

    available = rows(candidate)
    candidate_by_anchor = {row.row_anchor: row for row in available if row.row_anchor}
    for baseline_row in rows(baseline):
        targeted_fields = set(implicated_fields.get(baseline_row.row_anchor or "", set())) | set(
            implicated_fields.get(str(baseline_row.id), set())
        )
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
                if evidence_digest(baseline_row, field) != evidence_digest(candidate_row, field):
                    return False
            elif baseline_value != candidate_value and grounding_strength(
                candidate_row, field, candidate_tokens
            ) <= grounding_strength(baseline_row, field, baseline_tokens):
                return False
    return True


def _normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _recovery_unit_key(target: tuple[int, str | None]) -> str:
    page_number, table_id = target
    return (
        f"table:{page_number}:{table_id}"
        if table_id is not None
        else f"page:{page_number}:residual"
    )


def _recovery_unit_digest(
    draft: ExtractionDraft,
    target: tuple[int, str | None],
) -> str:
    return draft.raw_unit_sha256.get(
        _recovery_unit_key(target),
        _stable_payload_digest([]),
    )


def _untargeted_recovery_digest(
    draft: ExtractionDraft,
    targets: Collection[tuple[int, str | None]],
) -> str:
    targeted_keys = {_recovery_unit_key(target) for target in targets}
    return _stable_payload_digest(
        {key: digest for key, digest in draft.raw_unit_sha256.items() if key not in targeted_keys}
    )


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


def _normalize_public_diagnostic(
    diagnostic: dict[str, Any],
    source_tables: Collection[SourceTable],
) -> dict[str, Any]:
    payload = dict(diagnostic)
    payload.setdefault("diagnostic_kind", "table" if payload.get("table_id") else "page")
    payload.setdefault(
        "diagnostic_id",
        f"p{int(payload.get('page_number') or 1)}-"
        f"{payload.get('table_id') or 'page'}-"
        f"{_stable_payload_digest(payload)[:12]}",
    )
    if payload.get("diagnostic_kind") != "table":
        return payload
    if not payload.get("source_table_id"):
        matching_table = next(
            (
                table
                for table in source_tables
                if table.page_number == int(payload.get("page_number") or 0)
                and table.table_id == payload.get("table_id")
            ),
            None,
        )
        if matching_table is not None:
            payload["source_table_id"] = matching_table.id
    if payload.get("table_id") and payload.get("source_table_id"):
        return payload

    attempted_table_id = payload.pop("table_id", None)
    attempted_source_table_id = payload.pop("source_table_id", None)
    payload["diagnostic_kind"] = "page"
    if attempted_table_id:
        payload["attempted_table_id"] = attempted_table_id
    if attempted_source_table_id:
        payload["attempted_source_table_id"] = attempted_source_table_id
    return payload


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
        uvdoc_mode: str | None = None,
        table_selection_mode: str | None = None,
        uvdoc_adapter: PaddleUvdocAdapter | None = None,
        uvdoc_preregistration: UvdocPreregistration | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.uvdoc_mode = uvdoc_mode or self.settings.uvdoc_mode
        self.table_selection_mode = table_selection_mode or getattr(
            self.settings, "table_selection_mode", "off"
        )
        if self.table_selection_mode not in {"off", "shadow", "enabled"}:
            raise ValueError("table_selection_mode_invalid")
        if self.table_selection_mode == "enabled":
            raise ValueError("table_selection_enabled_not_promoted")
        if self.uvdoc_mode == "enabled":
            raise ValueError("uvdoc_enabled_not_promoted")
        self.uvdoc_adapter = uvdoc_adapter
        self.uvdoc_preregistration = uvdoc_preregistration
        self.uvdoc_preregistration_sha256: str | None = None
        self.uvdoc_initialization_error: str | None = None
        if self.uvdoc_mode == "shadow":
            try:
                if self.uvdoc_preregistration is None:
                    preregistration_path = self.settings.uvdoc_preregistration_path
                    if preregistration_path is None:
                        raise ValueError("uvdoc_preregistration_missing")
                    (
                        self.uvdoc_preregistration,
                        self.uvdoc_preregistration_sha256,
                    ) = load_preregistration(preregistration_path)
                if self.uvdoc_adapter is None:
                    model_dir = self.settings.uvdoc_model_dir
                    if model_dir is None:
                        raise ValueError("uvdoc_model_dir_missing")
                    self.uvdoc_adapter = PaddleUvdocAdapter(model_dir, device=paddle_device)
                compatibility = self.uvdoc_adapter.compatibility
                registered_identity = (
                    self.uvdoc_preregistration.model_sha256,
                    self.uvdoc_preregistration.model_config_sha256,
                    self.uvdoc_preregistration.adapter_config_sha256,
                )
                actual_identity = (
                    compatibility.model_sha256,
                    compatibility.model_config_sha256,
                    compatibility.adapter_config_sha256,
                )
                if actual_identity != registered_identity:
                    raise ValueError("uvdoc_preregistered_model_identity_mismatch")
            except Exception as error:  # shadow initialization cannot stop baseline extraction
                self.uvdoc_initialization_error = f"uvdoc_initialization_{type(error).__name__}"
                self.uvdoc_adapter = None
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
        self.orientation = PaddleDocOrientationAdapter(device="cpu")
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

    def _canonical_table_work(
        self,
        *,
        artifact_root: Path,
        page_asset: PageAsset,
        selected: PreparedPageCandidate,
        table_id: str,
        candidate_box: tuple[int, int, int, int],
    ) -> TableWork:
        crop = crop_region(
            selected.path,
            artifact_root / "crops" / f"{table_id}.png",
            page_asset.page_number,
            candidate_box,
        )
        crop_to_source = compose(
            crop.transform.inverse_matrix,
            selected.contract.transform.inverse_matrix,
        )
        source_polygon = _source_polygon(
            candidate_box,
            selected.contract.transform.inverse_matrix,
            source_width=page_asset.width,
            source_height=page_asset.height,
        )
        source_box = _polygon_box(
            source_polygon,
            width=page_asset.width,
            height=page_asset.height,
        )
        request = InferenceRequest(
            request_id=str(uuid4()),
            artifact_sha256=crop.artifact_sha256,
            canonical_artifact_sha256=crop.artifact_sha256,
            image_path=str(crop.output_path.resolve()),
            page_number=page_asset.page_number,
            options={
                "preprocessing_policy": "camera_preprocessing_v1",
                "table_stage": "canonical_table_ocr_v1",
                "recognition_variant": "canonical",
            },
        )
        response, cache_hit = _cached_prediction(
            artifact_root / "inference" / f"{table_id}.canonical.ocr.json",
            request,
            self.ocr,
        )
        local_tokens = tuple(
            token.model_copy(update={"token_id": f"{table_id}:canonical:{token.token_id}"})
            for token in paddle_ocr_tokens(
                response.output,
                page_asset.page_number,
                crop.artifact_sha256,
            )
        )
        mapped_tokens = map_transformed_tokens_to_page(
            local_tokens,
            crop_to_source,
            page_asset.artifact_sha256,
        )
        return TableWork(
            table_id=table_id,
            page_number=page_asset.page_number,
            page_artifact_sha256=page_asset.artifact_sha256,
            crop_path=crop.output_path,
            crop_sha256=crop.artifact_sha256,
            crop_width=crop.transform.derived_width,
            crop_height=crop.transform.derived_height,
            candidate_box=candidate_box,
            box=source_box,
            source_polygon=source_polygon,
            crop_to_source_matrix=crop_to_source,
            selected_page_artifact_sha256=selected.contract.artifact_sha256,
            selected_variant=selected.contract.variant,
            selected_dpi=selected.contract.dpi,
            tokens=mapped_tokens,
            canonical_tokens=local_tokens,
            adapter_inputs=[
                TableAdapterInput(
                    adapter_name=self.ocr.spec.model_name,
                    stage="final_table_ocr",
                    recognition_variant="canonical",
                    input_artifact_sha256=response.input_artifact_sha256,
                    canonical_crop_sha256=crop.artifact_sha256,
                    cache_hit=cache_hit,
                    accepted=True,
                    adapter_version=response.spec.model_version,
                    configuration_sha256=canonical_sha256(
                        {
                            "adapter": response.spec.model_name,
                            "stage": "final_table_ocr",
                            "recognition_variant": "canonical",
                        }
                    ),
                    latency_ms=response.latency_ms,
                )
            ],
        )

    def _recover_crop_ocr(
        self,
        *,
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
            high_resolution = resize_region(
                work.crop_path,
                artifact_root / "crops" / f"{work.table_id}-400dpi.png",
                work.page_number,
                max(1.0, 400 / work.selected_dpi),
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
                canonical_crop_sha256=work.crop_sha256,
                status="rendered",
            )
        )
        assets = [
            (
                "high_resolution",
                high_resolution.output_path,
                high_resolution.artifact_sha256,
                f"{work.table_id}.400dpi.ocr.json",
                compose(
                    high_resolution.transform.inverse_matrix,
                    work.crop_to_source_matrix,
                ),
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
                    canonical_crop_sha256=work.crop_sha256,
                    status="prepared",
                )
            )
            assets.append(
                (
                    "photometric",
                    photometric.output_path,
                    photometric.artifact_sha256,
                    f"{work.table_id}.400dpi-clahe.ocr.json",
                    compose(
                        high_resolution.transform.inverse_matrix,
                        work.crop_to_source_matrix,
                    ),
                )
            )
        try:
            table_quad, quad_confidence, angle_divergence = detect_table_quadrilateral(
                high_resolution.output_path
            )
            if table_quad is not None and quad_confidence >= 0.80:
                perspective = normalize_quadrilateral_region(
                    high_resolution.output_path,
                    artifact_root / "crops" / f"{work.table_id}-400dpi-perspective.png",
                    work.page_number,
                    table_quad,
                )
                perspective_to_page = compose(
                    perspective.transform.inverse_matrix,
                    high_resolution.transform.inverse_matrix,
                    work.crop_to_source_matrix,
                )
                assets.append(
                    (
                        "table_perspective",
                        perspective.output_path,
                        perspective.artifact_sha256,
                        f"{work.table_id}.400dpi-perspective.ocr.json",
                        perspective_to_page,
                    )
                )
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.HIGH_RESOLUTION,
                        artifact_sha256=perspective.artifact_sha256,
                        canonical_crop_sha256=work.crop_sha256,
                        status="prepared",
                        reason=(
                            "table_perspective:"
                            f"confidence={quad_confidence:.3f}:"
                            f"angle_divergence={angle_divergence:.3f}"
                        ),
                    )
                )
        except Exception as error:
            attempts.append(
                RecoveryAttempt(
                    stage=RecoveryStage.HIGH_RESOLUTION,
                    status="failed",
                    reason=f"table_perspective:{type(error).__name__}",
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
            page_box: tuple[int, int, int, int] | None = None,
            source_to_page_matrix: Matrix | None = None,
        ) -> tuple[TokenManifestEntry, ...]:
            matrix = source_to_page_matrix
            if matrix is None:
                if page_box is None:
                    raise ValueError("page_box or source_to_page_matrix is required")
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

        for (
            variant,
            image_path,
            artifact_sha256,
            cache_name,
            source_to_page_matrix,
        ) in assets:
            request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=artifact_sha256,
                canonical_artifact_sha256=work.crop_sha256,
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
                if source_to_page_matrix is None:
                    left, top, right, bottom = work.box
                    source_to_page_matrix = (
                        (
                            (right - left) / max(1, image.shape[1]),
                            0.0,
                            float(left),
                        ),
                        (
                            0.0,
                            (bottom - top) / max(1, image.shape[0]),
                            float(top),
                        ),
                        (0.0, 0.0, 1.0),
                    )
                mapped_tokens = map_transformed_tokens_to_page(
                    local_tokens,
                    source_to_page_matrix,
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
                        reason=(f"{variant}:reconstruction_error:{type(error).__name__}"),
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
                        source_to_page_matrix=source_to_page_matrix,
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
                    artifact_root / "crops" / f"{work.table_id}-400dpi-color-suppressed.png",
                )
                overlay_image = cv2.imread(
                    str(overlay_suppressed.output_path),
                    cv2.IMREAD_COLOR,
                )
                if overlay_image is None:
                    raise ValueError(
                        f"cannot read color-suppressed crop: {overlay_suppressed.output_path}"
                    )
            except Exception as error:
                attempts.append(
                    RecoveryAttempt(
                        stage=RecoveryStage.PHOTOMETRIC,
                        status="failed",
                        reason=(f"description_lane_variant_error:{type(error).__name__}"),
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
                    source_to_overlay = compose(
                        invert(work.crop_to_source_matrix),
                        high_resolution.transform.forward_matrix,
                    )
                    for region_index, region in enumerate(regions, start=1):
                        pixel_box = _bounded_mapped_box(
                            region,
                            source_to_overlay,
                            width=overlay_image.shape[1],
                            height=overlay_image.shape[0],
                        )
                        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                            continue
                        region_slug = "-".join(str(value) for value in region)
                        try:
                            targeted_crop = crop_region(
                                overlay_suppressed.output_path,
                                artifact_root
                                / "crops"
                                / (f"{work.table_id}-400dpi-description-{region_slug}.png"),
                                work.page_number,
                                pixel_box,
                            )
                            target_request = InferenceRequest(
                                request_id=str(uuid4()),
                                artifact_sha256=targeted_crop.artifact_sha256,
                                canonical_artifact_sha256=work.crop_sha256,
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
                                / (f"{work.table_id}.400dpi-description-{region_slug}.ocr.json"),
                                target_request,
                                self.ocr,
                            )
                            local_target_tokens = tuple(
                                token.model_copy(
                                    update={
                                        "token_id": (f"description:{region_index}:{token.token_id}")
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
                                    f"cannot read description crop: {targeted_crop.output_path}"
                                )
                            target_to_source = compose(
                                targeted_crop.transform.inverse_matrix,
                                high_resolution.transform.inverse_matrix,
                                work.crop_to_source_matrix,
                            )
                            mapped_target_tokens = map_transformed_tokens_to_page(
                                local_target_tokens,
                                target_to_source,
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
                                    source_to_page_matrix=target_to_source,
                                )
                            )
                            target_cache_hits.append(target_cache_hit)
                            target_latency_ms += target_response.latency_ms
                        except Exception as error:
                            attempts.append(
                                RecoveryAttempt(
                                    stage=RecoveryStage.CROP_OCR,
                                    status="failed",
                                    reason=(f"description_lane:{type(error).__name__}"),
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
                                    f"description_lane:reconstruction_error:{type(error).__name__}"
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
            source_to_high_resolution = compose(
                invert(work.crop_to_source_matrix),
                high_resolution.transform.forward_matrix,
            )
            for target_index, target in enumerate(sign_targets, start=1):
                region = target.region
                region_slug = "-".join(str(value) for value in region)
                try:
                    sign_pixel_box = _bounded_mapped_box(
                        region,
                        source_to_high_resolution,
                        width=high_resolution.transform.derived_width,
                        height=high_resolution.transform.derived_height,
                    )
                    sign_crop = crop_region(
                        high_resolution.output_path,
                        artifact_root
                        / "crops"
                        / f"{work.table_id}-400dpi-return-sign-{region_slug}.png",
                        work.page_number,
                        sign_pixel_box,
                    )
                    high_resolution_sign = resize_region(
                        sign_crop.output_path,
                        artifact_root
                        / "crops"
                        / f"{work.table_id}-800dpi-return-sign-{region_slug}.png",
                        work.page_number,
                        2.0,
                    )
                    enhanced_sign = clahe_variant(
                        high_resolution_sign.output_path,
                        artifact_root
                        / "crops"
                        / (f"{work.table_id}-800dpi-return-sign-{region_slug}-clahe.png"),
                    )
                    sign_request = InferenceRequest(
                        request_id=str(uuid4()),
                        artifact_sha256=enhanced_sign.artifact_sha256,
                        canonical_artifact_sha256=work.crop_sha256,
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
                        / (f"{work.table_id}.800dpi-return-sign-{region_slug}.ocr.json"),
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
                        if parse_decimal(token.text) == -target.expected_absolute_amount
                    )
                    if len(matching_sign_tokens) != 1:
                        raise ValueError("exact negative amount not uniquely recovered")
                    sign_image = cv2.imread(
                        str(enhanced_sign.output_path),
                        cv2.IMREAD_COLOR,
                    )
                    if sign_image is None:
                        raise ValueError(
                            f"cannot read return sign crop: {enhanced_sign.output_path}"
                        )
                    prefixed_sign_tokens = tuple(
                        token.model_copy(
                            update={"token_id": (f"return-sign:{target_index}:{token.token_id}")}
                        )
                        for token in matching_sign_tokens
                    )
                    sign_to_source = compose(
                        high_resolution_sign.transform.inverse_matrix,
                        sign_crop.transform.inverse_matrix,
                        high_resolution.transform.inverse_matrix,
                        work.crop_to_source_matrix,
                    )
                    mapped_sign_tokens = map_transformed_tokens_to_page(
                        prefixed_sign_tokens,
                        sign_to_source,
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
                            source_to_page_matrix=sign_to_source,
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
                            reason=(f"return_sign_800dpi_clahe:{type(error).__name__}"),
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
            or (
                item[0].startswith("table_perspective")
                and safely_realigns_perspective_reconstruction(baseline, item[-2])
            )
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
                    canonical_crop_sha256=work.crop_sha256,
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
        canonical_table_work: list[TableWork] = []
        canonical_token_manifest: dict[str, TokenManifestEntry] = {}
        document_total_candidates: list[DocumentTotalCandidate] = []
        uvdoc_shadow_runs: list[UvdocPreparedRun] = []
        table_selection_runs: list[dict[str, Any]] = []
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
        baseline_units_by_page = {
            unit.page_asset.page_number: unit
            for unit in (baseline_draft.page_units if baseline_draft else ())
        }
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
        page_token_sources: dict[str, tuple[OcrToken, PreprocessingCandidate]] = {}
        page_preprocessing: dict[int, PagePreprocessingRecord] = {}
        quality_by_page = {quality.page_number: quality for quality in manifest.quality}
        recovery_token_manifest: dict[str, TokenManifestEntry] = {}
        precanonical_fragment_manifest: dict[str, TokenManifestEntry] = {}
        for page_asset in manifest.pages:
            abort_checkpoint()
            page_path = artifact_root / "pages" / page_asset.relative_path
            raw_relative_path = str(Path("pages") / page_asset.relative_path)
            page_targets = {
                table_id
                for page_number, table_id in recovery_target_set
                if page_number == page_asset.page_number
            }
            if baseline_draft is not None and not page_targets:
                baseline_preprocessing = baseline_units_by_page[
                    page_asset.page_number
                ].preprocessing
                page_preprocessing[page_asset.page_number] = (
                    baseline_preprocessing
                    or _raw_only_preprocessing_record(
                        page_asset,
                        quality_by_page[page_asset.page_number],
                        raw_relative_path,
                    )
                )
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
            orientation_request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=page_asset.artifact_sha256,
                image_path=str(page_path.resolve()),
                page_number=page_asset.page_number,
            )
            orientation_response, orientation_cache_hit = _cached_prediction(
                artifact_root / "inference" / f"page-{page_asset.page_number}.orientation.json",
                orientation_request,
                self.orientation,
            )
            orientation_degrees, orientation_confidence = _orientation_correction(
                orientation_response.output
            )
            applied_orientation = (
                orientation_degrees
                if orientation_confidence >= ORIENTATION_CONFIDENCE_THRESHOLD
                else 0
            )
            uvdoc_page_run: UvdocPreparedRun | None = None
            if self.uvdoc_mode == "shadow":
                if self.uvdoc_initialization_error is not None:
                    uvdoc_page_run = UvdocPreparedRun(
                        page_number=page_asset.page_number,
                        status="failed",
                        reason_code=self.uvdoc_initialization_error,
                    )
                    uvdoc_shadow_runs.append(uvdoc_page_run)
                elif self.uvdoc_preregistration is None or not self.uvdoc_preregistration.eligible(
                    document_id, page_asset.page_number
                ):
                    uvdoc_page_run = UvdocPreparedRun(
                        page_number=page_asset.page_number,
                        status="ineligible",
                        reason_code="uvdoc_not_preregistered",
                    )
                    uvdoc_shadow_runs.append(uvdoc_page_run)
                else:
                    oriented_path = page_path
                    if applied_orientation:
                        image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
                        if image is None:
                            raise RuntimeError("oriented_raw_source_image_unreadable")
                        if applied_orientation == 90:
                            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        elif applied_orientation == 180:
                            image = cv2.rotate(image, cv2.ROTATE_180)
                        else:
                            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        oriented_path = (
                            artifact_root
                            / "lineage/oriented"
                            / f"page-{page_asset.page_number:04d}.png"
                        )
                        oriented_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(oriented_path), image):
                            raise RuntimeError("oriented_raw_image_write_failed")
                    try:
                        assert self.uvdoc_adapter is not None
                        uvdoc_page_run = self.uvdoc_adapter.predict(
                            oriented_path,
                            artifact_root,
                            page_number=page_asset.page_number,
                        )
                        uvdoc_shadow_runs.append(uvdoc_page_run)
                    except Exception as error:  # shadow inference cannot change publication
                        uvdoc_page_run = UvdocPreparedRun(
                            page_number=page_asset.page_number,
                            status="failed",
                            reason_code=f"uvdoc_inference_{type(error).__name__}",
                        )
                        uvdoc_shadow_runs.append(uvdoc_page_run)
            prepared_candidates = prepare_page_candidates(
                source_pdf=source,
                raw_path=page_path,
                raw_relative_path=raw_relative_path,
                page=page_asset,
                quality=quality_by_page[page_asset.page_number],
                artifact_root=artifact_root,
                orientation_degrees=applied_orientation,
                orientation_confidence=orientation_confidence,
            )
            bundles: list[PageInferenceBundle] = []
            for prepared in prepared_candidates:
                variant_suffix = (
                    ""
                    if prepared.contract.variant is PreprocessingVariant.RAW
                    else f".{prepared.contract.variant.value}"
                )
                ocr_request = InferenceRequest(
                    request_id=str(uuid4()),
                    artifact_sha256=prepared.contract.artifact_sha256,
                    image_path=str(prepared.path.resolve()),
                    page_number=page_asset.page_number,
                    options={"preprocessing_policy": "camera_preprocessing_v1"},
                )
                ocr_response, ocr_cache_hit = _cached_prediction(
                    artifact_root
                    / "inference"
                    / f"page-{page_asset.page_number}{variant_suffix}.ocr.json",
                    ocr_request,
                    self.ocr,
                )
                abort_checkpoint()
                source_tokens = paddle_ocr_tokens(
                    ocr_response.output,
                    page_asset.page_number,
                    prepared.contract.artifact_sha256,
                )
                tokens = paddle_ocr_tokens(
                    ocr_response.output,
                    page_asset.page_number,
                    page_asset.artifact_sha256,
                    crop_to_page=prepared.contract.transform.inverse_matrix,
                )
                if prepared.contract.variant is not PreprocessingVariant.RAW:
                    in_bounds_pairs = tuple(
                        (token, source_token)
                        for token, source_token in zip(tokens, source_tokens, strict=True)
                        if all(
                            point.x <= page_asset.width and point.y <= page_asset.height
                            for point in token.polygon.points
                        )
                    )
                    tokens = tuple(token for token, _source_token in in_bounds_pairs)
                    source_tokens = tuple(source_token for _token, source_token in in_bounds_pairs)
                layout_request = InferenceRequest(
                    request_id=str(uuid4()),
                    artifact_sha256=prepared.contract.artifact_sha256,
                    image_path=str(prepared.path.resolve()),
                    page_number=page_asset.page_number,
                    options={"preprocessing_policy": "camera_preprocessing_v1"},
                )
                layout_response, layout_cache_hit = _cached_prediction(
                    artifact_root
                    / "inference"
                    / f"page-{page_asset.page_number}{variant_suffix}.layout.json",
                    layout_request,
                    self.layout,
                )
                abort_checkpoint()
                to_page = prepared.contract.transform.inverse_matrix
                candidate_layout_boxes = tuple(
                    box
                    for box in _layout_boxes(layout_response.output)
                    if box[2] > 0
                    and box[3] > 0
                    and box[0] < prepared.contract.width
                    and box[1] < prepared.contract.height
                )
                mapped_layout_boxes = tuple(
                    _bounded_mapped_box(
                        box,
                        to_page,
                        width=page_asset.width,
                        height=page_asset.height,
                    )
                    for box in candidate_layout_boxes
                    if box[2] > 0 and box[3] > 0
                )
                candidate_geometry_boxes = tuple(
                    box
                    for box in _ocr_geometry_boxes(ocr_response.output, transform_identity())
                    if box[2] > 0
                    and box[3] > 0
                    and box[0] < prepared.contract.width
                    and box[1] < prepared.contract.height
                )
                mapped_geometry_boxes = tuple(
                    _bounded_mapped_box(
                        box,
                        to_page,
                        width=page_asset.width,
                        height=page_asset.height,
                    )
                    for box in candidate_geometry_boxes
                    if box[2] > 0 and box[3] > 0
                )
                bundles.append(
                    PageInferenceBundle(
                        candidate=prepared,
                        ocr_response=ocr_response,
                        ocr_cache_hit=ocr_cache_hit,
                        layout_response=layout_response,
                        layout_cache_hit=layout_cache_hit,
                        tokens=tokens,
                        source_tokens=source_tokens,
                        candidate_layout_boxes=candidate_layout_boxes,
                        candidate_geometry_boxes=candidate_geometry_boxes,
                        layout_boxes=mapped_layout_boxes,
                        geometry_boxes=mapped_geometry_boxes,
                        reconstruction_score=reconstruction_quality(
                            reconstruct_ocr_rows(
                                tokens,
                                page_number=page_asset.page_number,
                                table_id=(f"p{page_asset.page_number}-preprocessing-probe"),
                                box=(0, 0, page_asset.width, page_asset.height),
                                prior_schemas=tuple(schema_states),
                            )
                        ),
                    )
                )
            if self.table_selection_mode == "shadow":
                try:
                    uvdoc_proposals, uvdoc_reconstructions = _m5_uvdoc_shadow_materials(
                        source_sha256=document_id,
                        page_asset=page_asset,
                        run=uvdoc_page_run,
                        artifact_root=artifact_root,
                        orientation_degrees=applied_orientation,
                        prior_schemas=tuple(schema_states),
                        ocr=self.ocr,
                        layout=self.layout,
                    )
                    table_selection_runs.append(
                        _m5_linear_shadow_decision(
                            source_sha256=document_id,
                            page_asset=page_asset,
                            bundles=tuple(bundles),
                            prior_schemas=tuple(schema_states),
                            extra_proposals=uvdoc_proposals,
                            extra_reconstructions=uvdoc_reconstructions,
                        )
                    )
                except Exception as error:  # M5 shadow cannot change baseline publication
                    table_selection_runs.append(
                        {
                            "policy_version": MATCH_POLICY_VERSION,
                            "mode": "shadow",
                            "status": "failed",
                            "page_number": page_asset.page_number,
                            "reason": f"table_selection_{type(error).__name__}",
                            "proposal_count": 0,
                            "logical_tables": [],
                            "proposals": [],
                            "edges": [],
                        }
                    )
            selected_bundle, annotated_candidates = _select_page_inference(tuple(bundles))
            selected_candidate = next(
                candidate for candidate in annotated_candidates if candidate.selected
            )
            page_preprocessing[page_asset.page_number] = PagePreprocessingRecord(
                page_number=page_asset.page_number,
                raw_artifact_sha256=page_asset.artifact_sha256,
                raw_artifact_relative_path=raw_relative_path,
                raw_quality=quality_by_page[page_asset.page_number],
                orientation_degrees=applied_orientation,
                orientation_confidence=orientation_confidence,
                candidates=annotated_candidates,
                selected_variant=selected_candidate.variant,
            )
            preprocessing_diagnostic = {
                "preprocessing_policy": "camera_preprocessing_v1",
                "preprocessing_selected_variant": selected_candidate.variant.value,
                "preprocessing_candidate_count": len(annotated_candidates),
                "preprocessing_artifact_sha256": selected_candidate.artifact_sha256,
                "preprocessing_artifact_relative_path": (selected_candidate.artifact_relative_path),
                "preprocessing_operations": selected_candidate.transform.operations,
                "preprocessing_quality_before": quality_by_page[page_asset.page_number].model_dump(
                    mode="json"
                ),
                "preprocessing_quality_after": selected_candidate.quality.model_dump(mode="json"),
                "orientation_degrees": orientation_degrees,
                "orientation_confidence": orientation_confidence,
                "orientation_latency_ms": orientation_response.latency_ms,
                "orientation_cache_hit": orientation_cache_hit,
            }
            tokens = selected_bundle.tokens
            if selected_candidate.variant is not PreprocessingVariant.RAW:
                for token, source_token in zip(
                    selected_bundle.tokens,
                    selected_bundle.source_tokens,
                    strict=True,
                ):
                    page_token_sources[token.token_id] = (source_token, selected_candidate)
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
                        next(iter(identity_owners)) if len(identity_owners) == 1 else None
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
            ocr_response = selected_bundle.ocr_response
            ocr_cache_hit = selected_bundle.ocr_cache_hit
            layout_response = selected_bundle.layout_response
            layout_cache_hit = selected_bundle.layout_cache_hit
            layout_boxes = list(selected_bundle.layout_boxes)
            geometry_boxes = list(selected_bundle.geometry_boxes)
            candidate_layout_boxes = list(selected_bundle.candidate_layout_boxes)
            candidate_geometry_boxes = list(selected_bundle.candidate_geometry_boxes)
            candidate_boxes = _merge_table_boxes(
                candidate_layout_boxes,
                candidate_geometry_boxes,
            )
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
                        stored_candidate_box = diagnostic.get("candidate_box")
                        candidate_box = (
                            tuple(int(value) for value in stored_candidate_box)
                            if isinstance(stored_candidate_box, (list, tuple))
                            and len(stored_candidate_box) == 4
                            else _bounded_mapped_box(
                                tuple(int(value) for value in diagnostic["box"]),
                                selected_candidate.transform.forward_matrix,
                                width=selected_candidate.width,
                                height=selected_candidate.height,
                            )
                        )
                        work_boxes.append((table_id, candidate_box))
            else:
                scale = selected_candidate.dpi / page_asset.dpi
                work_boxes.extend(
                    (
                        f"p{page_asset.page_number}-t{table_index + 1}",
                        _safe_box(
                            box,
                            selected_candidate.width,
                            selected_candidate.height,
                            horizontal_padding=max(1, round(20 * scale)),
                            vertical_padding=max(1, round(100 * scale)),
                        ),
                    )
                    for table_index, box in enumerate(candidate_boxes)
                )
            for table_id, candidate_box in work_boxes:
                work = self._canonical_table_work(
                    artifact_root=artifact_root,
                    page_asset=page_asset,
                    selected=selected_bundle.candidate,
                    table_id=table_id,
                    candidate_box=candidate_box,
                )
                source_table_crop_paths[(page_asset.page_number, table_id)] = work.crop_path
                source_table_crop_boxes[(page_asset.page_number, table_id)] = work.box
                crop_relative_path = str(
                    work.crop_path.resolve().relative_to(artifact_root.resolve())
                )
                canonical_token_manifest.update(
                    {
                        mapped.token_id: TokenManifestEntry(
                            token_id=mapped.token_id,
                            page_number=mapped.page_number,
                            text=mapped.text,
                            polygon=mapped.polygon,
                            artifact_sha256=page_asset.artifact_sha256,
                            artifact_relative_path=raw_relative_path,
                            confidence=mapped.confidence,
                            source_artifact_sha256=work.crop_sha256,
                            source_artifact_relative_path=crop_relative_path,
                            source_polygon=local.polygon,
                            source_width=work.crop_width,
                            source_height=work.crop_height,
                            source_to_page_matrix=work.crop_to_source_matrix,
                        )
                        for local, mapped in zip(
                            work.canonical_tokens,
                            work.tokens,
                            strict=True,
                        )
                    }
                )
                table_work.append(work)
                canonical_table_work.append(work)

            specific_table_recovery = bool(baseline_draft is not None and None not in page_targets)
            if not table_work and not specific_table_recovery:
                no_table_assessment = _assess_no_table_page(page_path, tokens)

            if (
                not table_work
                and not specific_table_recovery
                and (
                    not no_table_assessment["demonstrably_blank"]
                    or (page_asset.page_number, None) in recovery_target_set
                )
            ):
                table_id = f"p{page_asset.page_number}-t1"
                candidate_box = (
                    0,
                    0,
                    selected_candidate.width,
                    selected_candidate.height,
                )
                work = self._canonical_table_work(
                    artifact_root=artifact_root,
                    page_asset=page_asset,
                    selected=selected_bundle.candidate,
                    table_id=table_id,
                    candidate_box=candidate_box,
                )
                source_table_crop_paths[(page_asset.page_number, table_id)] = work.crop_path
                source_table_crop_boxes[(page_asset.page_number, table_id)] = work.box
                crop_relative_path = str(
                    work.crop_path.resolve().relative_to(artifact_root.resolve())
                )
                canonical_token_manifest.update(
                    {
                        mapped.token_id: TokenManifestEntry(
                            token_id=mapped.token_id,
                            page_number=mapped.page_number,
                            text=mapped.text,
                            polygon=mapped.polygon,
                            artifact_sha256=page_asset.artifact_sha256,
                            artifact_relative_path=raw_relative_path,
                            confidence=mapped.confidence,
                            source_artifact_sha256=work.crop_sha256,
                            source_artifact_relative_path=crop_relative_path,
                            source_polygon=local.polygon,
                            source_width=work.crop_width,
                            source_height=work.crop_height,
                            source_to_page_matrix=work.crop_to_source_matrix,
                        )
                        for local, mapped in zip(
                            work.canonical_tokens,
                            work.tokens,
                            strict=True,
                        )
                    }
                )
                table_work.append(work)
                canonical_table_work.append(work)
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
                            **preprocessing_diagnostic,
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
                    work.tokens,
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
                    tokens=work.tokens,
                    profiles=job_profiles,
                )
                shadow_profile_match = self._profile_match(
                    document_id=document_id,
                    page_number=work.page_number,
                    page_width=page_asset.width,
                    page_height=page_asset.height,
                    box=work.box,
                    reconstruction=reconstruction,
                    tokens=work.tokens,
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
                        work.tokens,
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
                        artifact_root=artifact_root,
                        work=work,
                        prior_schemas=_recovery_prior_schemas(
                            schema_states,
                            page_number=work.page_number,
                            table_id=work.table_id,
                        ),
                        page_artifact_sha256=work.page_artifact_sha256,
                        page_artifact_relative_path=str(Path("pages") / page_asset.relative_path),
                        baseline=reconstruction,
                        baseline_tokens=work.tokens,
                    )
                    recovery_attempts.extend(attempts)
                    work.adapter_inputs.extend(
                        TableAdapterInput(
                            adapter_name=self.ocr.spec.model_name,
                            stage="crop_recovery",
                            recognition_variant=(
                                attempt.reason.removeprefix("input_variant:")
                                if attempt.reason and attempt.reason.startswith("input_variant:")
                                else "recovery"
                            ),
                            input_artifact_sha256=attempt.artifact_sha256,
                            canonical_crop_sha256=work.crop_sha256,
                            cache_hit=attempt.cache_hit,
                            accepted=attempt.status == "recovered",
                        )
                        for attempt in attempts
                        if attempt.stage is RecoveryStage.CROP_OCR
                        and attempt.artifact_sha256 is not None
                        and attempt.reason is not None
                        and attempt.reason.startswith("input_variant:")
                    )
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
                        artifact_relative_path=str(Path("pages") / page_asset.relative_path),
                        confidence=token.confidence,
                    )
                    for token in work.tokens
                }
                reconstruction_token_lookup.update(
                    {
                        token.token_id: canonical_token_manifest[token.token_id]
                        for token in work.tokens
                        if token.token_id in canonical_token_manifest
                    }
                )
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
                        or profile_heavy_sample
                        or not (profile_match and profile_match.selected and parsed_rows)
                    )
                    and (
                        targeted_recovery
                        or profile_heavy_sample
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
                    scoped_tokens = work.tokens
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
                            canonical_artifact_sha256=work.crop_sha256,
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
                        work.adapter_inputs.append(
                            TableAdapterInput(
                                adapter_name=self.vl.spec.model_name,
                                stage="local_vlm",
                                recognition_variant=asset.identity,
                                input_artifact_sha256=vl_response.input_artifact_sha256,
                                canonical_crop_sha256=work.crop_sha256,
                                cache_hit=cache_hit,
                                accepted=bool(
                                    response_candidate_count and not profile_heavy_sample
                                ),
                            )
                        )
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
                            work.canonical_tokens,
                            (0, 0, work.crop_width, work.crop_height),
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
                                canonical_crop_sha256=work.crop_sha256,
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
                                work.adapter_inputs.append(
                                    TableAdapterInput(
                                        adapter_name=self.gemini.model,
                                        stage="gemini",
                                        recognition_variant="redacted",
                                        input_artifact_sha256=redaction.artifact_sha256,
                                        canonical_crop_sha256=(response.canonical_crop_sha256),
                                        cache_hit=gemini_cache_hit,
                                        accepted=False,
                                    )
                                )
                                gemini_trace_index = len(work.adapter_inputs) - 1
                                if not gemini_cache_hit:
                                    gemini_cost += response.measured_cost_usd
                                try:
                                    image = cv2.imread(str(redaction.path), cv2.IMREAD_COLOR)
                                    if image is None:
                                        raise ValueError("cannot read redacted crop")
                                    grounded = ground_adjudication(
                                        response,
                                        validation_tokens=redaction.tokens,
                                        evidence_tokens=work.tokens,
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
                                        work.adapter_inputs[gemini_trace_index] = (
                                            work.adapter_inputs[gemini_trace_index].model_copy(
                                                update={"accepted": True}
                                            )
                                        )
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
                        if table.page_number == work.page_number and table.table_id == work.table_id
                    )
                    baseline_target_rows = tuple(
                        row
                        for row in baseline_rows
                        if row.page_number == work.page_number and row.table_id == work.table_id
                    )

                    baseline_payload = [
                        table.model_dump(mode="json") for table in baseline_target_tables
                    ]
                    candidate_payload = [
                        table.model_dump(mode="json") for table in selected_reconstruction_tables
                    ]
                    if not candidate_payload or _stable_payload_digest(
                        candidate_payload
                    ) == _stable_payload_digest(baseline_payload):
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
                        "candidate_box": work.candidate_box,
                        "source_polygon": work.source_polygon.model_dump(mode="json"),
                        "crop_to_source_matrix": work.crop_to_source_matrix,
                        "selected_page_artifact_sha256": (work.selected_page_artifact_sha256),
                        "adapter_inputs": [
                            item.model_dump(mode="json") for item in work.adapter_inputs
                        ],
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
                        **preprocessing_diagnostic,
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
            page.page_number: str(Path("pages") / page.relative_path) for page in manifest.pages
        }
        token_lookup = {item.token_id: item for item in baseline_token_manifest}
        token_lookup.update(
            {
                token.token_id: _page_token_manifest_entry(
                    token,
                    artifact_relative_path=relative_by_page[token.page_number],
                    source=page_token_sources.get(token.token_id),
                )
                for tokens in page_tokens.values()
                for token in tokens
            }
        )
        token_lookup.update(canonical_token_manifest)
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
        source_tables = _link_source_tables(
            selected_source_tables,
            rows,
            token_lookup=token_lookup,
        )
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
                page.model_copy(update={"relative_path": str(Path("pages") / page.relative_path)})
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
        token_manifest_by_id = {item.token_id: item for item in baseline_token_manifest}
        token_manifest_by_id.update(
            {
                token.token_id: _page_token_manifest_entry(
                    token,
                    table_ids=tuple(sorted(table_ids_by_token.get(token.token_id, ()))),
                    artifact_relative_path=relative_by_page[token.page_number],
                    source=page_token_sources.get(token.token_id),
                )
                for page_number in sorted(page_tokens)
                for token in page_tokens[page_number]
            }
        )
        token_manifest_by_id.update(
            {
                token_id: token.model_copy(
                    update={"table_ids": tuple(sorted(table_ids_by_token.get(token_id, ())))}
                )
                for token_id, token in canonical_token_manifest.items()
            }
        )
        token_manifest_by_id.update(recovery_token_manifest)
        token_manifest_by_id.update(precanonical_fragment_manifest)
        token_manifest_by_id.update({item.token_id: item for item in printed_fragments})
        token_manifest = tuple(token_manifest_by_id.values())

        normalized_diagnostics = [
            _normalize_public_diagnostic(diagnostic, source_tables) for diagnostic in diagnostics
        ]
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
                self.gemini_promotion.frozen_manifest_sha256 if self.gemini_promotion else None
            ),
        )
        recovery_metadata = _recovery_metadata(
            baseline_draft,
            recovery_target_set,
            source_tables,
            diagnostics,
        )
        published_preprocessing = tuple(
            page_preprocessing.get(page.page_number)
            or (
                baseline_units_by_page[page.page_number].preprocessing
                if page.page_number in baseline_units_by_page
                else None
            )
            for page in manifest.pages
        )
        if any(record is None for record in published_preprocessing):
            raise RuntimeError("page_preprocessing_record_missing")
        new_table_crops = tuple(
            CanonicalTableCrop(
                page_number=work.page_number,
                table_id=work.table_id,
                source_page_artifact_sha256=work.page_artifact_sha256,
                selected_page_artifact_sha256=work.selected_page_artifact_sha256,
                selected_variant=work.selected_variant.value,
                artifact_sha256=work.crop_sha256,
                artifact_relative_path=str(
                    work.crop_path.resolve().relative_to(artifact_root.resolve())
                ),
                width=work.crop_width,
                height=work.crop_height,
                candidate_box=work.candidate_box,
                source_box=work.box,
                source_polygon=work.source_polygon,
                crop_to_source_matrix=work.crop_to_source_matrix,
                adapter_inputs=tuple(work.adapter_inputs),
            )
            for work in canonical_table_work
        )
        published_table_crops_by_id = {
            (crop.page_number, crop.table_id): crop
            for unit in (baseline_draft.page_units if baseline_draft else ())
            for table in unit.table_units
            if (crop := table.canonical_crop) is not None
        }
        published_table_crops_by_id.update(
            {(crop.page_number, crop.table_id): crop for crop in new_table_crops}
        )
        published_table_crops = tuple(
            published_table_crops_by_id[key] for key in sorted(published_table_crops_by_id)
        )
        result = {
            "output_version": "offline_accuracy_spine_v5",
            "contract_revision": 5,
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
            "page_preprocessing": [
                record.model_dump(mode="json")
                for record in published_preprocessing
                if record is not None
            ],
            "table_crops": [record.model_dump(mode="json") for record in published_table_crops],
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
            baseline_units = baseline_units_by_page
            canonical_crops_by_table = {
                (item.page_number, item.table_id): item for item in published_table_crops
            }
            page_units: list[PageExtractionUnit] = []
            for page in manifest.pages:
                published_asset = page.model_copy(
                    update={"relative_path": str(Path("pages") / page.relative_path)}
                )
                raw_page_tables = tuple(
                    table for table in draft_source_tables if table.page_number == page.page_number
                )
                raw_page_rows = tuple(
                    row for row in draft_row_candidates if row.page_number == page.page_number
                )
                page_manifest = tuple(
                    item for item in token_manifest if item.page_number == page.page_number
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
                                source_table_crop_paths[(page.page_number, table.table_id)]
                                .resolve()
                                .relative_to(artifact_root.resolve())
                            )
                            if (page.page_number, table.table_id) in source_table_crop_paths
                            else None
                        ),
                        crop_box=source_table_crop_boxes.get((page.page_number, table.table_id)),
                        diagnostics=tuple(
                            item
                            for item in page_diagnostics
                            if item.get("table_id") == table.table_id
                        ),
                        normalized_fragments=tuple(
                            item
                            for item in page_manifest
                            if item.fragment_role is not None and table.table_id in item.table_ids
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
                        canonical_crop=canonical_crops_by_table.get(
                            (page.page_number, table.table_id)
                        ),
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
                            row for row in raw_page_rows if row.table_id not in assigned_table_ids
                        ),
                        diagnostics=page_diagnostics,
                        total_candidates=page_totals,
                        provider_usage=provider_usage,
                        preprocessing=(
                            page_preprocessing.get(page.page_number)
                            or (
                                baseline_units[page.page_number].preprocessing
                                if page.page_number in baseline_units
                                else None
                            )
                        ),
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
            _draft_sink["suppressed_repeated_source_tables"] = tuple(suppressed_source_tables)
            _draft_sink["recovery_metadata"] = recovery_metadata
            _draft_sink["artifact_root"] = artifact_root
            _draft_sink["uvdoc_shadow_runs"] = tuple(uvdoc_shadow_runs)
            _draft_sink["table_selection_runs"] = tuple(table_selection_runs)
        if _draft_sink is None:
            return _project_result_v6(
                result,
                artifact_root,
                tuple(uvdoc_shadow_runs),
                tuple(table_selection_runs),
            )
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
            suppressed_repeated_source_tables=sink["suppressed_repeated_source_tables"],
            recovery_metadata=sink["recovery_metadata"],
            artifact_root=Path(artifact_root),
            uvdoc_shadow_runs=sink["uvdoc_shadow_runs"],
            table_selection_runs=sink["table_selection_runs"],
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
        recovery_prefix = "recovery"
        self.extract(
            source,
            artifact_root / recovery_prefix,
            progress,
            baseline_draft=draft,
            recovery_targets=recovery_targets,
            allow_gemini=False,
            _draft_sink=sink,
            **options,
        )
        recovered_page_units = _prefix_recovery_page_units(
            sink["page_units"],
            recovery_prefix,
        )
        candidate = ExtractionDraft(
            document_id=sink["document_id"],
            source_sha256=sink["source_sha256"],
            source_name=sink["source_name"],
            page_units=recovered_page_units,
            provider_usage=sink["provider_usage"],
            hospital=sink["hospital"],
            hospital_id=sink["hospital_id"],
            alias_registry_revision=sink["alias_registry_revision"],
            profile_registry_revision=sink["profile_registry_revision"],
            applied_alias_ids=sink["applied_alias_ids"],
            suppressed_repeated_source_tables=sink["suppressed_repeated_source_tables"],
            recovery_metadata=sink["recovery_metadata"],
            artifact_root=Path(artifact_root),
            uvdoc_shadow_runs=draft.uvdoc_shadow_runs,
            table_selection_runs=draft.table_selection_runs,
        )
        from gmoney.extraction.validation import validate_extraction_result

        candidate_recovery_records = {
            (int(item.get("page_number") or 0), item.get("table_id")): item
            for item in candidate.recovery_metadata.get("targets", [])
        }
        empty_recovery = {
            "attempted": False,
            "targets": [],
            "untargeted_units_sha256": None,
        }
        selected = replace(
            draft,
            provider_usage=candidate.provider_usage,
            recovery_metadata=empty_recovery,
        )
        records: list[dict[str, Any]] = []
        for target in sorted(recovery_targets, key=lambda item: (item[0], item[1] or "")):
            before = selected
            trial = replace(
                selected,
                page_units=_merge_targeted_page_units(
                    selected.page_units,
                    candidate.page_units,
                    (target,),
                ),
                recovery_metadata=empty_recovery,
            )
            before_report = validate_extraction_result(
                source,
                before.result,
                artifact_root,
            )
            trial_report = validate_extraction_result(
                source,
                trial.result,
                artifact_root,
            )
            before_blocking = tuple(
                issue
                for issue in before_report.issues
                if issue.severity.value in {"blocking", "fatal"}
            )
            trial_blocking = tuple(
                issue
                for issue in trial_report.issues
                if issue.severity.value in {"blocking", "fatal"}
            )
            before_target = tuple(
                issue for issue in before_blocking if _issue_is_in_recovery_target(issue, (target,))
            )
            trial_target = tuple(
                issue for issue in trial_blocking if _issue_is_in_recovery_target(issue, (target,))
            )
            before_keys = {_issue_semantic_key(issue): issue for issue in before_blocking}
            trial_keys = {_issue_semantic_key(issue): issue for issue in trial_blocking}
            before_target_keys = {_issue_semantic_key(issue): issue for issue in before_target}
            trial_target_keys = {_issue_semantic_key(issue): issue for issue in trial_target}
            removed = tuple(
                issue.id
                for key, issue in before_target_keys.items()
                if key not in trial_target_keys
            )
            remaining = tuple(
                issue.id for key, issue in before_target_keys.items() if key in trial_target_keys
            )
            new_target = tuple(
                issue.id
                for key, issue in trial_target_keys.items()
                if key not in before_target_keys
            )
            new_global = tuple(
                issue.id for key, issue in trial_keys.items() if key not in before_keys
            )
            implicated_fields: dict[str, set[str]] = {}
            rows_by_id = {
                str(row.id): row for unit in before.page_units for row in unit.canonical_rows
            }
            for issue in before_target:
                if not issue.field:
                    continue
                for identity in (
                    issue.canonical_row_id,
                    issue.source_row_id,
                    issue.row_anchor,
                ):
                    if identity:
                        implicated_fields.setdefault(str(identity), set()).add(str(issue.field))
                baseline_row = rows_by_id.get(str(issue.canonical_row_id or ""))
                if baseline_row is not None and baseline_row.row_anchor:
                    implicated_fields.setdefault(baseline_row.row_anchor, set()).add(
                        str(issue.field)
                    )
            baseline_inventory = _source_financial_inventory(before, target)
            candidate_inventory = _source_financial_inventory(trial, target)
            inventory_safe, preserved_count, added_count = _financial_inventory_matches(
                baseline_inventory,
                candidate_inventory,
                implicated_fields,
            )
            candidate_record = candidate_recovery_records.get(target, {})
            located = bool(candidate_record) and candidate_record.get("status") != (
                "recovery_target_not_located"
            )
            safe = bool(
                located
                and removed
                and len(trial_target) < len(before_target)
                and not new_global
                and inventory_safe
                and _recovery_preserves_grounded_charges(
                    before,
                    trial,
                    (target,),
                    implicated_fields,
                )
            )
            if safe:
                selected = trial
            baseline_inventory_digest = _financial_inventory_digest(baseline_inventory)
            candidate_inventory_digest = _financial_inventory_digest(candidate_inventory)
            records.append(
                {
                    "page_number": target[0],
                    "table_id": target[1],
                    "selected": "candidate" if safe else "baseline",
                    "status": (
                        "recovered"
                        if safe
                        else (
                            "recovery_target_not_located"
                            if not located
                            else "recovery_no_safe_improvement"
                        )
                    ),
                    "baseline_unit_sha256": _recovery_unit_digest(before, target),
                    "candidate_unit_sha256": _recovery_unit_digest(trial, target),
                    "selected_unit_sha256": _recovery_unit_digest(
                        trial if safe else before,
                        target,
                    ),
                    "target_issue_ids": [issue.id for issue in before_target],
                    "removed_issue_ids": list(removed if safe else ()),
                    "remaining_issue_ids": list(remaining),
                    "new_issue_ids": list(new_target),
                    "baseline_financial_inventory_sha256": baseline_inventory_digest,
                    "candidate_financial_inventory_sha256": candidate_inventory_digest,
                    "selected_financial_inventory_sha256": (
                        candidate_inventory_digest if safe else baseline_inventory_digest
                    ),
                    "baseline_financial_row_count": len(baseline_inventory),
                    "candidate_financial_row_count": len(candidate_inventory),
                    "preserved_financial_row_count": preserved_count,
                    "added_financial_row_count": added_count if safe else 0,
                }
            )
        for key, digest in draft.raw_unit_sha256.items():
            if key not in {_recovery_unit_key(target) for target in recovery_targets} and (
                selected.raw_unit_sha256.get(key) != digest
            ):
                raise RuntimeError("targeted_recovery_changed_untargeted_raw_unit")
        recovery_metadata = {
            "attempted": True,
            "targets": records,
            "untargeted_units_sha256": _untargeted_recovery_digest(
                draft,
                recovery_targets,
            ),
        }
        return ExtractionDraft(
            document_id=selected.document_id,
            source_sha256=selected.source_sha256,
            source_name=selected.source_name,
            page_units=selected.page_units,
            provider_usage=candidate.provider_usage,
            hospital=selected.hospital,
            hospital_id=selected.hospital_id,
            alias_registry_revision=selected.alias_registry_revision,
            profile_registry_revision=selected.profile_registry_revision,
            applied_alias_ids=selected.applied_alias_ids,
            suppressed_repeated_source_tables=(selected.suppressed_repeated_source_tables),
            recovery_metadata=recovery_metadata,
            artifact_root=selected.artifact_root,
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
