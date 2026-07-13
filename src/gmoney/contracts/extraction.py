from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field

from gmoney.contracts.common import ContractModel, VersionedContract
from gmoney.contracts.evidence import Polygon


class RowRole(StrEnum):
    DETAIL = "detail"
    SECTION_TOTAL = "section_total"
    DOCUMENT_TOTAL = "document_total"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    REFUND = "refund"
    METADATA = "metadata"
    FOOTER_NOISE = "footer_noise"
    UNREADABLE = "unreadable"


class ReviewDisposition(StrEnum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    REJECTED = "rejected"
    UNREADABLE = "unreadable"


class EvidenceRef(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str | None = None
    polygon: Polygon
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    token_ids: tuple[str, ...] = ()


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
    candidate_ids: tuple[str, ...] = ()
    validation_flags: tuple[str, ...] = ()

