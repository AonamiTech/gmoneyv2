from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from gmoney.contracts.common import ContractModel, VersionedContract
from gmoney.contracts.evidence import OcrToken, Polygon
from gmoney.contracts.extraction import PageType, RowRole, TableType


class GeminiMode(StrEnum):
    OFF = "off"
    CHALLENGER = "challenger"
    ENABLED = "enabled"


class RecoveryStage(StrEnum):
    BASELINE = "baseline"
    HIGH_RESOLUTION = "high_resolution"
    PHOTOMETRIC = "photometric"
    CROP_OCR = "crop_ocr"
    LOCAL_VLM = "local_vlm"
    GEMINI = "gemini"
    REVIEW = "review"


class RecoveryReason(StrEnum):
    ZERO_YIELD = "zero_yield"
    LOW_YIELD = "low_yield"
    NO_SCHEMA = "no_schema"
    UNKNOWN_TABLE = "unknown_table"
    UNGROUNDED_FIELD = "ungrounded_field"
    VLM_TRUNCATED = "vlm_truncated"
    PROFILE_AMBIGUOUS = "profile_ambiguous"
    PROFILE_DRIFT = "profile_drift"
    VALIDATION_FAILURE = "validation_failure"


class RouteDecision(ContractModel):
    route_version: str = "phase3_route_v1"
    route: str
    reasons: tuple[RecoveryReason, ...] = ()
    planned_stages: tuple[RecoveryStage, ...] = ()
    profile_key: str | None = None
    profile_version: int | None = Field(default=None, ge=1)
    profile_score: float | None = Field(default=None, ge=0, le=1)
    profile_margin: float | None = Field(default=None, ge=0, le=1)


class RecoveryAttempt(ContractModel):
    stage: RecoveryStage
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    cache_hit: bool = False
    latency_ms: int = Field(default=0, ge=0)
    produced_rows: int = Field(default=0, ge=0)
    accepted_rows: int = Field(default=0, ge=0)
    status: str
    reason: str | None = None


class RecoveryTrace(ContractModel):
    trace_version: str = "phase3_recovery_trace_v1"
    document_id: str
    page_number: int = Field(ge=1)
    table_id: str
    decision: RouteDecision
    attempts: tuple[RecoveryAttempt, ...] = ()
    terminal_stage: RecoveryStage


class AdjudicationField(ContractModel):
    name: str
    raw_value: str
    token_ids: tuple[str, ...]
    polygon: Polygon
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_evidence(self) -> AdjudicationField:
        if not self.token_ids:
            raise ValueError("adjudicated fields require evidence token IDs")
        return self


class AdjudicationRow(ContractModel):
    row_order: int = Field(ge=0)
    role: RowRole
    fields: tuple[AdjudicationField, ...]


class AdjudicationRequest(ContractModel):
    request_version: str = "adjudication_request_v1"
    request_id: str
    document_id: str
    page_number: int = Field(ge=1)
    table_id: str
    masked_crop_path: str
    masked_crop_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_type: PageType | None = None
    table_type: TableType = TableType.UNKNOWN
    tokens: tuple[OcrToken, ...]
    unresolved_reasons: tuple[RecoveryReason, ...]
    candidate_rows: tuple[dict[str, Any], ...] = ()
    prompt_version: str
    schema_version: str = "phase3_adjudication_schema_v1"
    redaction_version: str


class AdjudicationResponse(ContractModel):
    response_version: str = "adjudication_response_v1"
    request_id: str
    provider: str
    model: str
    rows: tuple[AdjudicationRow, ...] = ()
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    measured_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    finish_reason: str | None = None
    rejected_reasons: tuple[str, ...] = ()
    raw_payload_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class GeminiPromotionDecision(ContractModel):
    decision_version: str = "gemini_promotion_v1"
    approved: bool = False
    model: str
    prompt_version: str
    schema_version: str = "phase3_adjudication_schema_v1"
    redaction_version: str
    frozen_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    local_f1: float = Field(ge=0, le=1)
    challenger_f1: float = Field(ge=0, le=1)
    unseen_precision: float = Field(ge=0, le=1)
    unseen_recall: float = Field(ge=0, le=1)
    accepted_ungrounded_rows: int = Field(default=0, ge=0)
    privacy_failures: int = Field(default=0, ge=0)
    decided_at: datetime

    @model_validator(mode="after")
    def validate_approval(self) -> GeminiPromotionDecision:
        if not self.approved:
            return self
        failures = []
        if self.challenger_f1 <= self.local_f1:
            failures.append("challenger_did_not_improve_f1")
        if self.challenger_f1 < 0.9:
            failures.append("challenger_f1_below_90_percent")
        if min(self.unseen_precision, self.unseen_recall) < 0.85:
            failures.append("unseen_metrics_below_85_percent")
        if self.accepted_ungrounded_rows:
            failures.append("accepted_ungrounded_rows")
        if self.privacy_failures:
            failures.append("privacy_failures")
        if failures:
            raise ValueError(f"unsafe Gemini promotion: {', '.join(failures)}")
        return self


class ProfileLifecycle(StrEnum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DRIFTED = "drifted"
    ARCHIVED = "archived"


class ProfileMetrics(ContractModel):
    holdout_bills: int = Field(ge=0)
    holdout_rows: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    amount_accuracy: float = Field(ge=0, le=1)
    precision_ci_lower: float = Field(default=0, ge=0, le=1)
    recall_ci_lower: float = Field(default=0, ge=0, le=1)
    f1_ci_lower: float = Field(default=0, ge=0, le=1)
    negative_recall: float = Field(default=0, ge=0, le=1)
    capture_conditions: int = Field(default=1, ge=1)
    supported_field_failures: tuple[str, ...] = ()
    accepted_ungrounded_rows: int = Field(default=0, ge=0)
    incomplete_pages: int = Field(default=0, ge=0)
    identity_ambiguities: int = Field(default=0, ge=0)
    unsafe_numeric_failures: int = Field(default=0, ge=0)


class LayoutProfile(VersionedContract):
    profile_key: str
    profile_version: int = Field(ge=1)
    lifecycle: ProfileLifecycle = ProfileLifecycle.CANDIDATE
    hospital_id: str | None = None
    hospital_name: str | None = Field(default=None, min_length=2, max_length=200)
    global_family: str | None = None
    page_type: PageType
    table_type: TableType
    page_aspect_ratio: float = Field(gt=0)
    table_box: tuple[float, float, float, float]
    header_tokens: tuple[str, ...] = ()
    stable_anchors: tuple[str, ...] = ()
    column_centers: dict[str, float] = Field(default_factory=dict)
    column_tolerances: dict[str, float] = Field(default_factory=dict)
    supported_fields: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
    retrieval_threshold: float = Field(default=0.8, ge=0, le=1)
    retrieval_margin: float = Field(default=0.1, ge=0, le=1)
    construction_dataset_ids: tuple[str, ...]
    holdout_dataset_ids: tuple[str, ...] = ()
    metrics: ProfileMetrics | None = None
    calibration_version: str | None = None
    supersedes_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> LayoutProfile:
        if not self.hospital_id and not self.global_family:
            raise ValueError("profile requires a hospital or global layout-family association")
        if self.hospital_name and not self.hospital_id:
            raise ValueError("profile hospital_name requires a hospital_id")
        left, top, right, bottom = self.table_box
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("profile table_box must be normalized to [0, 1]")
        if any(not 0 <= value <= 1 for value in self.column_centers.values()):
            raise ValueError("profile column centers must be normalized to [0, 1]")
        if set(self.supported_fields) & set(self.unsupported_fields):
            raise ValueError("profile fields cannot be both supported and unsupported")
        if {"description", "amount"} & set(self.unsupported_fields):
            raise ValueError("core row fields cannot be declared unsupported")
        return self


class LayoutObservation(ContractModel):
    document_id: str
    hospital_id: str | None = None
    page_number: int = Field(ge=1)
    page_type: PageType
    table_type: TableType
    page_aspect_ratio: float = Field(gt=0)
    table_box: tuple[float, float, float, float]
    header_tokens: tuple[str, ...] = ()
    column_centers: dict[str, float] = Field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_geometry(self) -> LayoutObservation:
        left, top, right, bottom = self.table_box
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("observation table_box must be normalized to [0, 1]")
        if any(not 0 <= value <= 1 for value in self.column_centers.values()):
            raise ValueError("observation column centers must be normalized to [0, 1]")
        return self


class ProfileMatch(ContractModel):
    match_version: str = "profile_match_v1"
    profile_key: str | None = None
    profile_version: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    selected: bool
    reasons: tuple[str, ...] = ()


class ProfileEvent(ContractModel):
    profile_key: str
    profile_version: int = Field(ge=1)
    from_state: ProfileLifecycle
    to_state: ProfileLifecycle
    reason: str
    occurred_at: datetime


class DriftObservation(ContractModel):
    profile_key: str
    profile_version: int = Field(ge=1)
    document_id: str
    match_score: float = Field(ge=0, le=1)
    escalated: bool
    heavy_disagreement: bool = False
    hard_incompatibility: bool = False


class DriftDecision(ContractModel):
    profile_key: str
    profile_version: int = Field(ge=1)
    drifted: bool
    immediate: bool = False
    breached_windows: int = Field(default=0, ge=0)
    reasons: tuple[str, ...] = ()


class ProfileRegistrySnapshot(ContractModel):
    registry_version: str = "profile_registry_v1"
    profiles: tuple[LayoutProfile, ...] = ()
    events: tuple[ProfileEvent, ...] = ()

    @model_validator(mode="after")
    def validate_active_versions(self) -> ProfileRegistrySnapshot:
        active_keys = [
            profile.profile_key
            for profile in self.profiles
            if profile.lifecycle is ProfileLifecycle.ACTIVE
        ]
        if len(active_keys) != len(set(active_keys)):
            raise ValueError("profile registry permits only one active version per key")
        return self
