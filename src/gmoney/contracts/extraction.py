from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from gmoney.contracts.common import ContractModel, VersionedContract
from gmoney.contracts.evidence import PageAsset, Polygon


class RowRole(StrEnum):
    DETAIL = "detail"
    INFORMATIONAL = "informational"
    CONTINUATION = "continuation"
    SECTION_HEADER = "section_header"
    CATEGORY_ROLLUP = "category_rollup"
    SECTION_TOTAL = "section_total"
    DOCUMENT_TOTAL = "document_total"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    REFUND = "refund"
    METADATA = "metadata"
    FOOTER_NOISE = "footer_noise"
    UNREADABLE = "unreadable"
    UNRESOLVED = "unresolved"


class PageType(StrEnum):
    ITEMIZED_CHARGES = "itemized_charges"
    PHARMACY = "pharmacy"
    LABORATORY = "laboratory"
    RECEIPT_PAYMENT = "receipt_payment"
    CATEGORY_SUMMARY = "category_summary"
    NARRATIVE = "narrative"
    METADATA = "metadata"
    MIXED = "mixed"
    BLANK = "blank"
    UNREADABLE = "unreadable"


class TableType(StrEnum):
    ITEM_LEDGER = "item_ledger"
    PHARMACY = "pharmacy"
    LABORATORY = "laboratory"
    CATEGORY_SUMMARY = "category_summary"
    PACKAGE_SUMMARY = "package_summary"
    PAYMENT = "payment"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ReviewDisposition(StrEnum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    REJECTED = "rejected"
    UNREADABLE = "unreadable"


class DocumentTotalKind(StrEnum):
    BILL_TOTAL = "bill_total"
    GROSS_TOTAL = "gross_total"
    PAYABLE_TOTAL = "payable_total"
    SETTLEMENT_TOTAL = "settlement_total"


class DocumentTotalScope(StrEnum):
    DOCUMENT = "document"
    SECTION = "section"
    SETTLEMENT = "settlement"
    PAYMENT = "payment"


class EvidenceRef(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str | None = None
    polygon: Polygon
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    token_ids: tuple[str, ...] = ()

    @field_validator("token_ids")
    @classmethod
    def require_non_blank_token_ids(cls, token_ids: tuple[str, ...]) -> tuple[str, ...]:
        if any(not token_id.strip() for token_id in token_ids):
            raise ValueError("evidence token IDs must be non-blank strings")
        return token_ids


class TokenManifestEntry(ContractModel):
    """A published OCR token (or a typed fragment of one) used by evidence."""

    token_id: str
    page_number: int = Field(ge=1)
    table_ids: tuple[str, ...] = ()
    text: str
    polygon: Polygon
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_relative_path: str
    confidence: float = Field(ge=0, le=1)
    parent_token_id: str | None = None
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)
    fragment_role: str | None = None
    # Evidence polygons are always expressed in rendered-page coordinates.  For
    # recovery OCR, retain the artifact and transform that produced those page
    # coordinates instead of pretending that the page image was OCR'd.
    source_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    source_artifact_relative_path: str | None = None
    source_polygon: Polygon | None = None
    source_width: int | None = Field(default=None, gt=0)
    source_height: int | None = Field(default=None, gt=0)
    source_to_page_matrix: tuple[tuple[float, float, float], ...] | None = None

    @model_validator(mode="after")
    def require_consistent_fragment(self) -> "TokenManifestEntry":
        fragment_fields = (
            self.parent_token_id,
            self.character_start,
            self.character_end,
            self.fragment_role,
        )
        if any(value is not None for value in fragment_fields):
            if any(value is None for value in fragment_fields):
                raise ValueError("fragment token metadata must be complete")
            assert self.character_start is not None
            assert self.character_end is not None
            if self.character_end <= self.character_start:
                raise ValueError("fragment token range must be non-empty")
        recovery_fields = (
            self.source_artifact_sha256,
            self.source_artifact_relative_path,
            self.source_polygon,
            self.source_width,
            self.source_height,
            self.source_to_page_matrix,
        )
        if any(value is not None for value in recovery_fields):
            if any(value is None for value in recovery_fields):
                raise ValueError("recovery token provenance must be complete")
            assert self.source_to_page_matrix is not None
            if len(self.source_to_page_matrix) != 3 or any(
                len(row) != 3 for row in self.source_to_page_matrix
            ):
                raise ValueError("source-to-page transform must be 3x3")
        return self


class ReceiptDuplicatePair(ContractModel):
    canonical_row_ids: tuple[str, str]
    source_row_ids: tuple[str, ...] = ()
    page_number: int = Field(ge=1)
    table_id: str | None = None
    match_basis: Literal["exact_reference", "dated_grounded_charge"]


class SuppressedSourceTable(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str


class ProviderUsageSlice(ContractModel):
    gemini_calls: int = Field(ge=0)
    gemini_measured_cost_usd: Decimal = Field(ge=0)
    gemini_allowed: bool | None = None
    gemini_mode: str | None = None
    gemini_promotion_manifest_sha256: str | None = None
    gemini_provider_disabled_reason: str | None = None


class ProviderUsage(ContractModel):
    initial: ProviderUsageSlice
    recovery: ProviderUsageSlice
    aggregate: ProviderUsageSlice


class RecoveryTargetRecord(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str | None = None
    selected: Literal["baseline", "candidate"]
    status: Literal[
        "recovered",
        "recovery_target_not_located",
        "recovery_no_safe_improvement",
    ]
    baseline_unit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_unit_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    selected_unit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    removed_issue_ids: tuple[str, ...] = ()


class RecoveryMetadata(ContractModel):
    attempted: bool
    targets: tuple[RecoveryTargetRecord, ...]
    untargeted_units_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )

    @model_validator(mode="after")
    def require_attempt_metadata(self) -> "RecoveryMetadata":
        if self.attempted != bool(self.targets):
            raise ValueError("recovery attempted state must match its targets")
        if self.attempted and self.untargeted_units_sha256 is None:
            raise ValueError("attempted recovery requires an untargeted-unit digest")
        if not self.attempted and self.untargeted_units_sha256 is not None:
            raise ValueError("unattempted recovery cannot publish a unit digest")
        return self


class ExtractionDiagnostic(ContractModel):
    """Typed diagnostic identity; route-specific measurements live in details."""

    model_config = ConfigDict(extra="allow", frozen=True)

    diagnostic_id: str = Field(min_length=1)
    diagnostic_kind: Literal["page", "table"]
    page_number: int = Field(ge=1)
    table_id: str | None = None
    source_table_id: str | None = None
    status: str | None = None
    page_classification: str | None = None
    financial_form_classification: str | None = None
    financial_form_classification_evidence: tuple[str, ...] = ()
    demonstrably_blank: bool = False
    financial_form_suspected: bool = False
    crop_relative_path: str | None = None
    crop_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    box: tuple[int, int, int, int] | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_kind_identity(self) -> "ExtractionDiagnostic":
        if self.diagnostic_kind == "page" and (
            self.table_id is not None or self.source_table_id is not None
        ):
            raise ValueError("page diagnostics cannot identify a table")
        if self.diagnostic_kind == "table" and (
            not self.table_id or not self.source_table_id
        ):
            raise ValueError("table diagnostics require logical and source table IDs")
        if (self.crop_relative_path is None) != (self.crop_sha256 is None):
            raise ValueError("diagnostic crop path and hash must be paired")
        return self


class SourceColumn(ContractModel):
    id: str
    label: str
    order: int = Field(ge=0)
    canonical_field: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    validation_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_grounded_or_synthetic_header(self) -> "SourceColumn":
        grounded = any(item.token_ids for item in self.evidence)
        synthetic = "synthetic_header" in self.validation_flags
        if not grounded and not synthetic:
            raise ValueError("source header requires grounded OCR evidence")
        if grounded and synthetic:
            raise ValueError("grounded source header cannot be marked synthetic")
        return self


class SourceCell(ContractModel):
    column_id: str
    raw_value: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    validation_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_grounded_value(self) -> "SourceCell":
        if self.raw_value and self.raw_value.strip() and not any(
            item.token_ids for item in self.evidence
        ):
            raise ValueError("non-empty source cell requires grounded OCR evidence")
        return self


class ReceiptSourceMetadata(ContractModel):
    issuer_raw: str | None = None
    issuer_normalized: str | None = None
    reference_raw: str | None = None
    reference_normalized: str | None = None


class SourceRow(ContractModel):
    id: str
    order: int = Field(ge=0)
    canonical_row_id: str | None = None
    cells: tuple[SourceCell, ...]
    validation_flags: tuple[str, ...] = ()
    receipt_metadata: ReceiptSourceMetadata | None = None


class SourceTable(ContractModel):
    id: str
    page_number: int = Field(ge=1)
    table_id: str
    table_type: TableType = TableType.UNKNOWN
    columns: tuple[SourceColumn, ...]
    rows: tuple[SourceRow, ...]
    validation_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_consistent_grid(self) -> "SourceTable":
        if not self.columns or not self.rows:
            raise ValueError("source table requires columns and rows")
        column_ids = tuple(column.id for column in self.columns)
        if len(set(column_ids)) != len(column_ids):
            raise ValueError("source table column ids must be unique")
        if tuple(column.order for column in self.columns) != tuple(range(len(self.columns))):
            raise ValueError("source table columns must use contiguous order")
        if len({row.id for row in self.rows}) != len(self.rows):
            raise ValueError("source table row ids must be unique")
        if tuple(row.order for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("source table rows must use contiguous order")
        expected = set(column_ids)
        for row in self.rows:
            actual = tuple(cell.column_id for cell in row.cells)
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("source row requires exactly one cell per column")
        return self


class DocumentTotal(ContractModel):
    total_version: str = "document_total_v2"
    amount_raw: str
    amount: Decimal
    label: str
    kind: DocumentTotalKind = DocumentTotalKind.BILL_TOTAL
    scope: DocumentTotalScope = DocumentTotalScope.DOCUMENT
    page_number: int = Field(ge=1)
    evidence: EvidenceRef
    confidence: float = Field(ge=0, le=1)
    source_route: str = "page_ocr_final_total"
    context_id: str | None = None
    context_kind: str | None = None


class ProviderCandidate(VersionedContract):
    provider: str
    model: str
    model_version: str
    prompt_version: str | None = None
    profile_version: str | None = None
    crop_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_payload: dict[str, Any]
    evidence: tuple[EvidenceRef, ...]
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: int = Field(ge=0)
    measured_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)


class CanonicalRow(VersionedContract):
    document_id: str
    page_number: int = Field(ge=1)
    table_id: str | None = None
    page_type: PageType | None = None
    table_type: TableType | None = None
    row_order: int = Field(ge=0)
    role: RowRole = RowRole.DETAIL
    review_disposition: ReviewDisposition = ReviewDisposition.PENDING
    section: str | None = None
    description: str | None = None
    service_date_raw: str | None = None
    service_date_iso: str | None = None
    request_no: str | None = None
    service_code: str | None = None
    hsn_code: str | None = None
    quantity_raw: str | None = None
    quantity: Decimal | None = None
    unit_price_raw: str | None = None
    unit_price: Decimal | None = None
    gross_amount_raw: str | None = None
    gross_amount: Decimal | None = None
    discount_raw: str | None = None
    discount: Decimal | None = None
    net_amount_raw: str | None = None
    net_amount: Decimal | None = None
    evidence: tuple[EvidenceRef, ...]
    field_evidence: dict[str, tuple[EvidenceRef, ...]] = Field(default_factory=dict)
    candidate_ids: tuple[str, ...] = ()
    source_routes: tuple[str, ...] = ()
    validation_flags: tuple[str, ...] = ()


class ExtractionResultV5(ContractModel):
    """Complete extractor/publication envelope accepted by the safety gate."""

    output_version: Literal["offline_accuracy_spine_v5"]
    document_total_version: str
    document_totals_version: str
    document_total: DocumentTotal | None
    document_totals: tuple[DocumentTotal, ...]
    document_id: str
    hospital_id: str | None = None
    hospital: dict[str, Any] | None = None
    alias_registry_revision: int | None = None
    profile_registry_revision: int | None = None
    applied_alias_ids: tuple[str, ...] = ()
    source_sha256: str
    source_name: str
    pages: int = Field(ge=1)
    page_assets: tuple[PageAsset, ...]
    source_tables: tuple[SourceTable, ...]
    token_manifest: tuple[TokenManifestEntry, ...]
    suppressed_repeated_source_tables: tuple[SuppressedSourceTable, ...] = ()
    rows: tuple[CanonicalRow, ...]
    receipt_duplicate_pairs: tuple[ReceiptDuplicatePair, ...] = ()
    diagnostics: tuple[ExtractionDiagnostic, ...]
    provider_usage: ProviderUsage
    recovery: RecoveryMetadata
    worker_release_revision: str | None = None
    semantic_validation: dict[str, Any] | None = None
    validation_recovery_attempted: bool | None = None
