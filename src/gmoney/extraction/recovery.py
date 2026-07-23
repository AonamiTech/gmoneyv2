from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher

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
from gmoney.extraction.ocr_rows import ReconstructionResult
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow
from gmoney.extraction.typed_values import parse_decimal

FIELD_QUALITY_FLAGS = frozenset(
    {
        "missing_labeled_quantity",
        "missing_labeled_unit_price",
        "line_arithmetic_mismatch",
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
        flags.count("line_arithmetic_mismatch"),
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
    grounded_rows = sum(
        bool(row.evidence_token_ids and row.evidence_box) for row in reconstruction.rows
    )
    source_rows = sum(len(table.rows) for table in reconstruction.source_tables)
    return (
        -arithmetic_mismatches,
        -missing_labeled_fields,
        _mapped_field_coverage(reconstruction),
        _populated_source_cells(reconstruction),
        len(reconstruction.rows),
        grounded_rows,
        source_rows,
    )


def safely_improves_reconstruction(
    baseline: ReconstructionResult,
    candidate: ReconstructionResult,
) -> bool:
    if len(candidate.rows) < len(baseline.rows):
        return False
    if any(
        candidate_count > baseline_count
        for candidate_count, baseline_count in zip(
            _field_quality_defects(candidate),
            _field_quality_defects(baseline),
            strict=True,
        )
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
