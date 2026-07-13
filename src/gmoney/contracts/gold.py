from decimal import Decimal

from pydantic import Field

from gmoney.contracts.common import ContractModel
from gmoney.contracts.evidence import Polygon


class GoldRow(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str | None = None
    row_order: int | None = Field(default=None, ge=0)
    polygon: Polygon | None = None
    section: str | None = None
    service_date: str | None = None
    request_no: str | None = None
    description: str
    service_code: str | None = None
    hsn_code: str | None = None
    rate: Decimal | None = None
    quantity: Decimal | None = None
    gross_amount: Decimal | None = None
    discount: Decimal | None = None
    amount: Decimal
    source_note: str | None = None


class GoldAnnotation(ContractModel):
    annotation_version: str = "gold_annotation_v1"
    bill_file: str
    unit_key: str | None = None
    hospital_folder: str | None = None
    hospital_id: str | None = None
    variant_id: str | None = None
    source_paths: tuple[str, ...] = ()
    excluded_notes: tuple[str, ...] = ()
    document_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    document_total: Decimal | None = None
    document_total_basis: str | None = None
    rows: tuple[GoldRow, ...]
