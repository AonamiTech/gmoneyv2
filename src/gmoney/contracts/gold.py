from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

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


class GoldSourceColumn(ContractModel):
    id: str
    label: str
    order: int = Field(ge=0)
    canonical_field: str | None = None
    polygon: Polygon | None = None


class GoldSourceCell(ContractModel):
    column_id: str
    raw_value: str | None = None
    polygon: Polygon | None = None
    readable: bool = True
    source_note: str | None = None

    @model_validator(mode="after")
    def unreadable_cells_have_no_asserted_value(self) -> "GoldSourceCell":
        if not self.readable and self.raw_value not in (None, ""):
            raise ValueError("unreadable gold cells cannot assert a value")
        return self


class GoldSourceRow(ContractModel):
    order: int = Field(ge=0)
    cells: tuple[GoldSourceCell, ...]


class GoldSourceTable(ContractModel):
    page_number: int = Field(ge=1)
    table_id: str
    columns: tuple[GoldSourceColumn, ...]
    rows: tuple[GoldSourceRow, ...]

    @model_validator(mode="after")
    def require_consistent_grid(self) -> "GoldSourceTable":
        column_ids = tuple(column.id for column in self.columns)
        if not column_ids or len(set(column_ids)) != len(column_ids):
            raise ValueError("gold source table columns must be non-empty and unique")
        if tuple(column.order for column in self.columns) != tuple(range(len(column_ids))):
            raise ValueError("gold source table columns must use contiguous order")
        expected = set(column_ids)
        if tuple(row.order for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("gold source table rows must use contiguous order")
        for row in self.rows:
            actual = tuple(cell.column_id for cell in row.cells)
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("gold source rows require exactly one cell per column")
        return self


class GoldImageReview(ContractModel):
    reviewer: str = Field(min_length=1)
    method: Literal["human", "codex_image_review", "other"]
    passes: int = Field(ge=1)
    reviewed_at: datetime
    page_asset_sha256: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_valid_page_hashes(self) -> "GoldImageReview":
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.page_asset_sha256
        ):
            raise ValueError("review page asset hashes must be lowercase SHA-256 values")
        return self


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
    layout_family_id: str | None = None
    image_review: GoldImageReview | None = None
    source_tables: tuple[GoldSourceTable, ...] = ()
    rows: tuple[GoldRow, ...]
