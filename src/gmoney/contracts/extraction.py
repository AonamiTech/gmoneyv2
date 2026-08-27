from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from gmoney.contracts.common import ContractModel, VersionedContract
from gmoney.contracts.evidence import Polygon


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
