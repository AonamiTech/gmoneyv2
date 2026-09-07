"""Authoritative corpus, gold, review, and milestone-gate contracts.

These contracts deliberately sit beside (rather than inside) the extraction
contracts.  A source manifest describes immutable geometry and source quality;
gold adds human transcription to that geometry; review records preserve who or
what made a decision and which frozen inputs were used.  IDs for source
objects are therefore derived only from source hashes and geometry, never from
the transcribed text or numeric values.

The module is intentionally strict at the schema boundary.  Every model is
frozen and rejects unknown fields, every top-level object has an explicit
version, and every digest is a lowercase SHA-256 value.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from gmoney.contracts.common import ContractModel
from gmoney.contracts.evidence import Polygon
from gmoney.contracts.v6 import canonical_sha256

SHA256_PATTERN = r"^[a-f0-9]{64}$"
SHA256: TypeAlias = str
Digest: TypeAlias = Annotated[str, Field(pattern=SHA256_PATTERN)]
AUTHORITY_COHORTS = ("production14", "passing36", "staging159")
AUTHORITY_COHORT_COUNTS = {
    "production14": 14,
    "passing36": 36,
    "staging159": 159,
}
_COHORT_ORDER = {name: index for index, name in enumerate(AUTHORITY_COHORTS)}


class AuthorityModel(ContractModel):
    """Base for contracts whose serialized form is part of an audit trail."""

    # ContractModel already sets extra="forbid" and frozen=True.  Repeating the
    # configuration here makes the boundary explicit and protects this module
    # if the common base is ever relaxed for a backwards-compatible contract.
    model_config = ConfigDict(extra="forbid", frozen=True)


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _identity_hash(kind: str, payload: dict[str, Any]) -> str:
    """Hash a structural identity with a versioned namespace."""

    return canonical_sha256(
        {
            "authority_identity_version": "authority_ids_v1",
            "kind": kind,
            "payload": _json_value(payload),
        }
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return dict(value)


def _enum_value(value: Any) -> Any:
    """Normalize enum instances and raw JSON values before identity hashing."""

    return value.value if isinstance(value, StrEnum) else value


def _polygon_identity(value: Polygon) -> dict[str, Any]:
    """Normalize numeric point spelling before hashing geometry identities."""

    return Polygon.model_validate(value).model_dump(mode="json")


def source_document_id_for(source_sha256: str) -> str:
    """Return the stable document identity for immutable source bytes."""

    return _identity_hash("document", {"source_sha256": source_sha256})


def page_id_for(source_sha256: str, page_number: int, artifact_sha256: str) -> str:
    return _identity_hash(
        "page",
        {
            "source_sha256": source_sha256,
            "page_number": page_number,
            "artifact_sha256": artifact_sha256,
        },
    )


def table_id_for(page_id: str, table_order: int, polygon: Polygon) -> str:
    return _identity_hash(
        "table",
        {
            "page_id": page_id,
            "table_order": table_order,
            "polygon": _polygon_identity(polygon),
        },
    )


def column_id_for(table_id: str, column_order: int, polygon: Polygon) -> str:
    return _identity_hash(
        "column",
        {
            "table_id": table_id,
            "column_order": column_order,
            "polygon": _polygon_identity(polygon),
        },
    )


def row_id_for(table_id: str, row_order: int, polygon: Polygon) -> str:
    return _identity_hash(
        "row",
        {
            "table_id": table_id,
            "row_order": row_order,
            "polygon": _polygon_identity(polygon),
        },
    )


def cell_id_for(row_id: str, column_id: str, polygon: Polygon) -> str:
    return _identity_hash(
        "cell",
        {
            "row_id": row_id,
            "column_id": column_id,
            "polygon": _polygon_identity(polygon),
        },
    )


def continuation_id_for(
    source_sha256: str,
    continuation_order: int,
    from_row_id: str,
    to_row_id: str,
    continuation_kind: str,
) -> str:
    return _identity_hash(
        "continuation",
        {
            "source_sha256": source_sha256,
            "continuation_order": continuation_order,
            "from_row_id": from_row_id,
            "to_row_id": to_row_id,
            "continuation_kind": continuation_kind,
        },
    )


def total_id_for(
    source_sha256: str,
    total_order: int,
    scope_id: str,
    total_kind: str,
    polygon: Polygon,
) -> str:
    return _identity_hash(
        "total",
        {
            "source_sha256": source_sha256,
            "total_order": total_order,
            "scope_id": scope_id,
            "total_kind": total_kind,
            "polygon": _polygon_identity(polygon),
        },
    )


# Friendly aliases used by callers that prefer the shorter names.
document_id_for = source_document_id_for


class SourceClass(StrEnum):
    NATIVE_TEXT_PDF = "NATIVE_TEXT_PDF"
    FLAT_SCAN = "FLAT_SCAN"
    CAMERA_PHOTO = "CAMERA_PHOTO"
    CAMERA_PHOTO_WITH_CURVATURE = "CAMERA_PHOTO_WITH_CURVATURE"
    UNREADABLE_OR_INCOMPLETE = "UNREADABLE_OR_INCOMPLETE"


class Readability(StrEnum):
    READABLE = "readable"
    PARTIAL = "partial"
    UNREADABLE = "unreadable"


class TableKind(StrEnum):
    UNKNOWN = "unknown"
    ITEM_LEDGER = "item_ledger"
    PHARMACY = "pharmacy"
    LABORATORY = "laboratory"
    CATEGORY_SUMMARY = "category_summary"
    RECEIPT = "receipt"
    RECEIPT_PAYMENT = "receipt_payment"
    PAYMENT = "payment"
    NARRATIVE = "narrative"
    METADATA = "metadata"
    MIXED = "mixed"
    BLANK = "blank"
    UNREADABLE = "unreadable"
    TOTAL = "total"


class TotalKind(StrEnum):
    SUBTOTAL = "subtotal"
    TAX = "tax"
    DISCOUNT = "discount"
    DOCUMENT_TOTAL = "document_total"
    PAYMENT = "payment"
    BALANCE = "balance"
    OTHER = "other"


class ContinuationKind(StrEnum):
    CROSS_PAGE = "cross_page"
    WRAPPED_ROW = "wrapped_row"
    CONTINUATION_HEADER = "continuation_header"


class RowKind(StrEnum):
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


class ReviewKind(StrEnum):
    INDEPENDENT_A = "independent_a"
    INDEPENDENT_B = "independent_b"
    ADJUDICATOR = "adjudicator"
    RED_TEAM = "red_team"


class ReviewState(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"
    UNREADABLE = "unreadable"
    NO_CHANGE = "no_change"


class GateDecision(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


def _prepare_row_children(data: dict[str, Any], *, gold: bool) -> dict[str, Any]:
    table_id = data.get("table_id", "")
    polygon = data.get("polygon")
    row_order = data.get("row_order", 0)
    if not data.get("row_id") and polygon is not None:
        data["row_id"] = row_id_for(table_id, row_order, polygon)
    cells: list[dict[str, Any]] = []
    for index, item in enumerate(data.get("cells", ())):
        cell = _as_dict(item)
        cell.setdefault("row_id", data.get("row_id", ""))
        cell.setdefault("column_order", index)
        cells.append(cell)
    if "cells" in data:
        data["cells"] = tuple(cells)
    return data


def _prepare_table_children(data: dict[str, Any], *, gold: bool) -> dict[str, Any]:
    page_id = data.get("page_id", "")
    table_order = data.get("table_order", 0)
    polygon = data.get("polygon")
    if not data.get("table_id") and page_id and polygon is not None:
        data["table_id"] = table_id_for(page_id, table_order, polygon)
    table_id = data.get("table_id", "")
    columns: list[dict[str, Any]] = []
    for index, item in enumerate(data.get("columns", ())):
        column = _as_dict(item)
        column.setdefault("table_id", table_id)
        column.setdefault("column_order", index)
        if not column.get("column_id") and column.get("polygon") is not None:
            column["column_id"] = column_id_for(
                table_id,
                column["column_order"],
                column["polygon"],
            )
        columns.append(column)
    column_ids = [column.get("column_id", "") for column in columns]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data.get("rows", ())):
        row = _as_dict(item)
        row.setdefault("table_id", table_id)
        row.setdefault("row_order", index)
        if not row.get("row_id") and row.get("polygon") is not None:
            row["row_id"] = row_id_for(table_id, row["row_order"], row["polygon"])
        cells: list[dict[str, Any]] = []
        for cell_index, item_cell in enumerate(row.get("cells", ())):
            cell = _as_dict(item_cell)
            cell.setdefault("row_id", row.get("row_id", ""))
            cell.setdefault("column_order", cell_index)
            if not cell.get("column_id") and cell.get("column_order") < len(column_ids):
                cell["column_id"] = column_ids[cell["column_order"]]
            cells.append(cell)
        if "cells" in row:
            row["cells"] = tuple(cells)
        rows.append(row)
    if "columns" in data:
        data["columns"] = tuple(columns)
    if "rows" in data:
        data["rows"] = tuple(rows)
    return data


class SourceCellV1(AuthorityModel):
    cell_id: str = Field(default="", pattern=SHA256_PATTERN)
    row_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_order: int = Field(default=0, ge=0)
    polygon: Polygon
    readability: Readability = Readability.READABLE
    unreadable_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("cell_id") and data.get("polygon") is not None:
            data["cell_id"] = cell_id_for(
                data.get("row_id", ""),
                data.get("column_id", ""),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> SourceCellV1:
        if not self.row_id or not self.column_id:
            raise ValueError("source cells require parent row and column IDs")
        expected = cell_id_for(self.row_id, self.column_id, self.polygon)
        if self.cell_id != expected:
            raise ValueError("source cell ID does not match structural identity")
        if self.readability is Readability.READABLE and self.unreadable_reason:
            raise ValueError("readable source cells cannot have an unreadable reason")
        if self.readability is not Readability.READABLE and not self.unreadable_reason:
            raise ValueError("unreadable source cells require a reason")
        return self


class SourceColumnV1(AuthorityModel):
    column_id: str = Field(default="", pattern=SHA256_PATTERN)
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_order: int = Field(default=0, ge=0)
    polygon: Polygon
    readability: Readability = Readability.READABLE

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("column_id") and data.get("polygon") is not None:
            data["column_id"] = column_id_for(
                data.get("table_id", ""),
                data.get("column_order", 0),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> SourceColumnV1:
        if not self.table_id:
            raise ValueError("source columns require a parent table ID")
        expected = column_id_for(self.table_id, self.column_order, self.polygon)
        if self.column_id != expected:
            raise ValueError("source column ID does not match structural identity")
        return self


class SourceRowV1(AuthorityModel):
    row_id: str = Field(default="", pattern=SHA256_PATTERN)
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    row_order: int = Field(default=0, ge=0)
    polygon: Polygon
    readability: Readability = Readability.READABLE
    cells: tuple[SourceCellV1, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        return _prepare_row_children(_as_dict(value), gold=False)

    @model_validator(mode="after")
    def validate_identity(self) -> SourceRowV1:
        if not self.table_id:
            raise ValueError("source rows require a parent table ID")
        expected = row_id_for(self.table_id, self.row_order, self.polygon)
        if self.row_id != expected:
            raise ValueError("source row ID does not match structural identity")
        if self.readability is Readability.READABLE and any(
            cell.readability is not Readability.READABLE for cell in self.cells
        ):
            raise ValueError("readable source rows cannot contain unreadable cells")
        if any(cell.row_id != self.row_id for cell in self.cells):
            raise ValueError("source row cell belongs to another row")
        return self


class SourceTableV1(AuthorityModel):
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    page_id: str = Field(default="", pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    table_order: int = Field(default=0, ge=0)
    polygon: Polygon
    readability: Readability = Readability.READABLE
    table_kind: TableKind = TableKind.UNKNOWN
    columns: tuple[SourceColumnV1, ...] = ()
    rows: tuple[SourceRowV1, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        return _prepare_table_children(_as_dict(value), gold=False)

    @model_validator(mode="after")
    def validate_identity_and_grid(self) -> SourceTableV1:
        if not self.page_id:
            raise ValueError("source tables require a parent page ID")
        expected = table_id_for(self.page_id, self.table_order, self.polygon)
        if self.table_id != expected:
            raise ValueError("source table ID does not match structural identity")
        if tuple(column.column_order for column in self.columns) != tuple(
            range(len(self.columns))
        ):
            raise ValueError("source table columns must use contiguous order")
        column_ids = tuple(column.column_id for column in self.columns)
        if len(set(column_ids)) != len(column_ids):
            raise ValueError("source table columns must be unique")
        if tuple(row.row_order for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("source table rows must use contiguous order")
        row_ids = tuple(row.row_id for row in self.rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("source table rows must be unique")
        for row in self.rows:
            if row.table_id != self.table_id:
                raise ValueError("source row belongs to another table")
            actual = tuple(cell.column_id for cell in row.cells)
            if row.cells and (len(actual) != len(column_ids) or set(actual) != set(column_ids)):
                raise ValueError("source rows require exactly one cell per column")
        return self


class SourcePageV1(AuthorityModel):
    page_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_relative_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    source_class: SourceClass
    readability: Readability = Readability.READABLE
    tables: tuple[SourceTableV1, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("page_id") and data.get("artifact_sha256") is not None:
            data["page_id"] = page_id_for(
                data.get("source_sha256", ""),
                data.get("page_number", 1),
                data["artifact_sha256"],
            )
        tables = []
        for index, item in enumerate(data.get("tables", ())):
            table = _as_dict(item)
            table.setdefault("page_id", data.get("page_id", ""))
            table.setdefault("page_number", data.get("page_number", 1))
            table.setdefault("table_order", index)
            tables.append(table)
        if "tables" in data:
            data["tables"] = tuple(tables)
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> SourcePageV1:
        expected = page_id_for(self.source_sha256, self.page_number, self.artifact_sha256)
        if self.page_id != expected:
            raise ValueError("source page ID does not match structural identity")
        if tuple(table.table_order for table in self.tables) != tuple(range(len(self.tables))):
            raise ValueError("source page tables must use contiguous order")
        for table in self.tables:
            if table.page_id != self.page_id or table.page_number != self.page_number:
                raise ValueError("source table belongs to another page")
        return self


class SourceContinuationV1(AuthorityModel):
    continuation_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    continuation_order: int = Field(ge=0)
    from_row_id: str = Field(pattern=SHA256_PATTERN)
    to_row_id: str = Field(pattern=SHA256_PATTERN)
    continuation_kind: ContinuationKind
    readable: bool = True
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("continuation_id"):
            data["continuation_id"] = continuation_id_for(
                data.get("source_sha256", ""),
                data.get("continuation_order", 0),
                data.get("from_row_id", ""),
                data.get("to_row_id", ""),
                _enum_value(data.get("continuation_kind", ContinuationKind.CROSS_PAGE)),
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> SourceContinuationV1:
        expected = continuation_id_for(
            self.source_sha256,
            self.continuation_order,
            self.from_row_id,
            self.to_row_id,
            self.continuation_kind.value,
        )
        if self.continuation_id != expected:
            raise ValueError("source continuation ID does not match structural identity")
        if self.from_row_id == self.to_row_id:
            raise ValueError("a continuation must connect two distinct rows")
        return self


class SourceTotalV1(AuthorityModel):
    total_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    total_order: int = Field(ge=0)
    scope_id: str = Field(pattern=SHA256_PATTERN)
    total_kind: TotalKind
    polygon: Polygon
    readability: Readability = Readability.READABLE

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("total_id") and data.get("polygon") is not None:
            data["total_id"] = total_id_for(
                data.get("source_sha256", ""),
                data.get("total_order", 0),
                data.get("scope_id", ""),
                _enum_value(data.get("total_kind", TotalKind.OTHER)),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> SourceTotalV1:
        expected = total_id_for(
            self.source_sha256,
            self.total_order,
            self.scope_id,
            self.total_kind.value,
            self.polygon,
        )
        if self.total_id != expected:
            raise ValueError("source total ID does not match structural identity")
        return self


class SourceManifestV1(AuthorityModel):
    manifest_version: Literal["source_manifest_v1"] = "source_manifest_v1"
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_mime_type: str = Field(min_length=1)
    source_size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    pages: tuple[SourcePageV1, ...]
    source_class: SourceClass
    cohorts: tuple[str, ...] = Field(min_length=1)
    sealed_master: bool = False
    split: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    layout_family_id: str | None = None
    capture_time_bucket: str | None = None
    continuations: tuple[SourceContinuationV1, ...] = ()
    totals: tuple[SourceTotalV1, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @property
    def document_id(self) -> str:
        """Stable source-document identity without duplicating it in the manifest."""

        return source_document_id_for(self.source_sha256)

    @model_validator(mode="before")
    @classmethod
    def prepare_pages_and_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        pages = []
        for index, item in enumerate(data.get("pages", ())):
            page = _as_dict(item)
            page.setdefault("source_sha256", data.get("source_sha256", ""))
            page.setdefault("page_number", index + 1)
            pages.append(SourcePageV1.model_validate(page))
        data["pages"] = tuple(pages)
        continuations = []
        for item in data.get("continuations", ()):
            continuation = _as_dict(item)
            continuation.setdefault("source_sha256", data.get("source_sha256", ""))
            continuations.append(SourceContinuationV1.model_validate(continuation))
        data["continuations"] = tuple(continuations)
        totals = []
        for item in data.get("totals", ()):
            total = _as_dict(item)
            total.setdefault("source_sha256", data.get("source_sha256", ""))
            totals.append(SourceTotalV1.model_validate(total))
        data["totals"] = tuple(totals)
        return data

    @model_validator(mode="after")
    def validate_manifest(self) -> SourceManifestV1:
        if self.page_count != len(self.pages):
            raise ValueError("source manifest page_count does not match pages")
        if tuple(page.page_number for page in self.pages) != tuple(range(1, self.page_count + 1)):
            raise ValueError("source manifest pages must be contiguous and ordered")
        if any(page.source_sha256 != self.source_sha256 for page in self.pages):
            raise ValueError("source page source hash differs from manifest")
        if any(page.source_class is not self.source_class for page in self.pages):
            raise ValueError("source page source class differs from manifest")
        if any(cohort not in _COHORT_ORDER for cohort in self.cohorts):
            raise ValueError("source manifest contains an unknown authority cohort")
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError("source manifest cohorts must be unique")
        if self.cohorts != tuple(sorted(self.cohorts, key=_COHORT_ORDER.__getitem__)):
            raise ValueError("source manifest cohorts must use canonical order")
        if self.sealed_master and "staging159" not in self.cohorts:
            raise ValueError("a sealed master source manifest requires staging159")
        if any(item.source_sha256 != self.source_sha256 for item in self.continuations):
            raise ValueError("source continuation source hash differs from manifest")
        if any(item.source_sha256 != self.source_sha256 for item in self.totals):
            raise ValueError("source total source hash differs from manifest")
        row_ids = {
            row.row_id
            for page in self.pages
            for table in page.tables
            for row in table.rows
        }
        for continuation in self.continuations:
            if continuation.from_row_id not in row_ids or continuation.to_row_id not in row_ids:
                raise ValueError("source continuation references a missing row")
        scope_ids = {
            table.table_id
            for page in self.pages
            for table in page.tables
        } | {source_document_id_for(self.source_sha256)}
        for total in self.totals:
            if total.scope_id not in scope_ids:
                raise ValueError("source total references a missing scope")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256", "created_at"})
        expected_hash = canonical_sha256(payload)
        if not self.manifest_sha256:
            object.__setattr__(self, "manifest_sha256", expected_hash)
        elif self.manifest_sha256 != expected_hash:
            raise ValueError("source manifest hash does not match canonical content")
        return self


class GoldCellV2(AuthorityModel):
    cell_id: str = Field(default="", pattern=SHA256_PATTERN)
    row_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_order: int = Field(default=0, ge=0)
    polygon: Polygon
    readable: bool = True
    raw_value: str | None = None
    normalized_value: str | None = None
    canonical_field: str | None = None
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("cell_id") and data.get("polygon") is not None:
            data["cell_id"] = cell_id_for(
                data.get("row_id", ""),
                data.get("column_id", ""),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity_and_readability(self) -> GoldCellV2:
        if not self.row_id or not self.column_id:
            raise ValueError("gold cells require parent row and column IDs")
        expected = cell_id_for(self.row_id, self.column_id, self.polygon)
        if self.cell_id != expected:
            raise ValueError("gold cell ID does not match structural identity")
        if not self.readable and (self.raw_value not in (None, "") or self.normalized_value):
            raise ValueError("unreadable gold cells cannot assert a value")
        return self


class GoldColumnV2(AuthorityModel):
    column_id: str = Field(default="", pattern=SHA256_PATTERN)
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    column_order: int = Field(default=0, ge=0)
    polygon: Polygon
    label: str | None = None
    canonical_field: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("column_id") and data.get("polygon") is not None:
            data["column_id"] = column_id_for(
                data.get("table_id", ""),
                data.get("column_order", 0),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> GoldColumnV2:
        if not self.table_id:
            raise ValueError("gold columns require a parent table ID")
        expected = column_id_for(self.table_id, self.column_order, self.polygon)
        if self.column_id != expected:
            raise ValueError("gold column ID does not match structural identity")
        return self


class GoldRowV2(AuthorityModel):
    row_id: str = Field(default="", pattern=SHA256_PATTERN)
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    row_order: int = Field(default=0, ge=0)
    polygon: Polygon
    row_kind: RowKind = RowKind.DETAIL
    section: str | None = None
    service_date: str | None = None
    description: str | None = None
    rate: Decimal | None = None
    quantity: Decimal | None = None
    gross_amount: Decimal | None = None
    discount: Decimal | None = None
    amount: Decimal | None = None
    cells: tuple[GoldCellV2, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        return _prepare_row_children(_as_dict(value), gold=True)

    @model_validator(mode="after")
    def validate_identity_and_values(self) -> GoldRowV2:
        if not self.table_id:
            raise ValueError("gold rows require a parent table ID")
        expected = row_id_for(self.table_id, self.row_order, self.polygon)
        if self.row_id != expected:
            raise ValueError("gold row ID does not match structural identity")
        if any(cell.row_id != self.row_id for cell in self.cells):
            raise ValueError("gold row cell belongs to another row")
        if self.row_kind is RowKind.DETAIL and self.description is None and not self.cells:
            raise ValueError("detail gold rows require description or cells")
        return self


class GoldTableV2(AuthorityModel):
    table_id: str = Field(default="", pattern=SHA256_PATTERN)
    page_id: str = Field(default="", pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    table_order: int = Field(default=0, ge=0)
    polygon: Polygon
    table_kind: TableKind = TableKind.UNKNOWN
    readability: Readability = Readability.READABLE
    columns: tuple[GoldColumnV2, ...] = ()
    rows: tuple[GoldRowV2, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        return _prepare_table_children(_as_dict(value), gold=True)

    @model_validator(mode="after")
    def validate_identity_and_grid(self) -> GoldTableV2:
        if not self.page_id:
            raise ValueError("gold tables require a parent page ID")
        expected = table_id_for(self.page_id, self.table_order, self.polygon)
        if self.table_id != expected:
            raise ValueError("gold table ID does not match structural identity")
        if tuple(column.column_order for column in self.columns) != tuple(
            range(len(self.columns))
        ):
            raise ValueError("gold table columns must use contiguous order")
        column_ids = tuple(column.column_id for column in self.columns)
        if len(set(column_ids)) != len(column_ids):
            raise ValueError("gold table columns must be unique")
        if tuple(row.row_order for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("gold table rows must use contiguous order")
        for row in self.rows:
            if row.table_id != self.table_id:
                raise ValueError("gold row belongs to another table")
            actual = tuple(cell.column_id for cell in row.cells)
            if row.cells and (len(actual) != len(column_ids) or set(actual) != set(column_ids)):
                raise ValueError("gold rows require exactly one cell per column")
        return self


class GoldPageV2(AuthorityModel):
    page_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    source_class: SourceClass
    readability: Readability = Readability.READABLE
    tables: tuple[GoldTableV2, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("page_id") and data.get("artifact_sha256") is not None:
            data["page_id"] = page_id_for(
                data.get("source_sha256", ""),
                data.get("page_number", 1),
                data["artifact_sha256"],
            )
        tables = []
        for index, item in enumerate(data.get("tables", ())):
            table = _as_dict(item)
            table.setdefault("page_id", data.get("page_id", ""))
            table.setdefault("page_number", data.get("page_number", 1))
            table.setdefault("table_order", index)
            tables.append(table)
        if "tables" in data:
            data["tables"] = tuple(tables)
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> GoldPageV2:
        expected = page_id_for(self.source_sha256, self.page_number, self.artifact_sha256)
        if self.page_id != expected:
            raise ValueError("gold page ID does not match structural identity")
        if tuple(table.table_order for table in self.tables) != tuple(range(len(self.tables))):
            raise ValueError("gold page tables must use contiguous order")
        for table in self.tables:
            if table.page_id != self.page_id or table.page_number != self.page_number:
                raise ValueError("gold table belongs to another page")
        return self


class GoldContinuationV2(AuthorityModel):
    continuation_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    continuation_order: int = Field(ge=0)
    from_row_id: str = Field(pattern=SHA256_PATTERN)
    to_row_id: str = Field(pattern=SHA256_PATTERN)
    continuation_kind: ContinuationKind
    readable: bool = True
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("continuation_id"):
            data["continuation_id"] = continuation_id_for(
                data.get("source_sha256", ""),
                data.get("continuation_order", 0),
                data.get("from_row_id", ""),
                data.get("to_row_id", ""),
                _enum_value(data.get("continuation_kind", ContinuationKind.CROSS_PAGE)),
            )
        return data

    @model_validator(mode="after")
    def validate_identity(self) -> GoldContinuationV2:
        expected = continuation_id_for(
            self.source_sha256,
            self.continuation_order,
            self.from_row_id,
            self.to_row_id,
            self.continuation_kind.value,
        )
        if self.continuation_id != expected:
            raise ValueError("gold continuation ID does not match structural identity")
        if self.from_row_id == self.to_row_id:
            raise ValueError("a continuation must connect two distinct rows")
        return self


class GoldTotalV2(AuthorityModel):
    total_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    total_order: int = Field(ge=0)
    scope_id: str = Field(pattern=SHA256_PATTERN)
    total_kind: TotalKind
    polygon: Polygon
    readable: bool = True
    raw_value: str | None = None
    normalized_value: Decimal | None = None
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("total_id") and data.get("polygon") is not None:
            data["total_id"] = total_id_for(
                data.get("source_sha256", ""),
                data.get("total_order", 0),
                data.get("scope_id", ""),
                _enum_value(data.get("total_kind", TotalKind.OTHER)),
                data["polygon"],
            )
        return data

    @model_validator(mode="after")
    def validate_identity_and_readability(self) -> GoldTotalV2:
        expected = total_id_for(
            self.source_sha256,
            self.total_order,
            self.scope_id,
            self.total_kind.value,
            self.polygon,
        )
        if self.total_id != expected:
            raise ValueError("gold total ID does not match structural identity")
        if not self.readable and (
            self.raw_value not in (None, "") or self.normalized_value is not None
        ):
            raise ValueError("unreadable gold totals cannot assert a value")
        return self


class GoldDocumentV2(AuthorityModel):
    gold_version: Literal["gold_document_v2"] = "gold_document_v2"
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    document_id: str = Field(default="", pattern=SHA256_PATTERN)
    page_count: int = Field(gt=0)
    pages: tuple[GoldPageV2, ...]
    continuations: tuple[GoldContinuationV2, ...] = ()
    totals: tuple[GoldTotalV2, ...] = ()
    annotation_group_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gold_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def prepare_document(cls, value: Any) -> Any:
        data = _as_dict(value)
        data.setdefault("document_id", source_document_id_for(data.get("source_sha256", "")))
        pages = []
        for index, item in enumerate(data.get("pages", ())):
            page = _as_dict(item)
            page.setdefault("source_sha256", data.get("source_sha256", ""))
            page.setdefault("page_number", index + 1)
            pages.append(GoldPageV2.model_validate(page))
        data["pages"] = tuple(pages)
        continuations = []
        for item in data.get("continuations", ()):
            continuation = _as_dict(item)
            continuation.setdefault("source_sha256", data.get("source_sha256", ""))
            continuations.append(GoldContinuationV2.model_validate(continuation))
        data["continuations"] = tuple(continuations)
        totals = []
        for item in data.get("totals", ()):
            total = _as_dict(item)
            total.setdefault("source_sha256", data.get("source_sha256", ""))
            totals.append(GoldTotalV2.model_validate(total))
        data["totals"] = tuple(totals)
        return data

    @model_validator(mode="after")
    def validate_document(self) -> GoldDocumentV2:
        expected_document_id = source_document_id_for(self.source_sha256)
        if self.document_id != expected_document_id:
            raise ValueError("gold document ID must be derived from source bytes")
        if self.page_count != len(self.pages):
            raise ValueError("gold document page_count does not match pages")
        if tuple(page.page_number for page in self.pages) != tuple(range(1, self.page_count + 1)):
            raise ValueError("gold document pages must be contiguous and ordered")
        if any(page.source_sha256 != self.source_sha256 for page in self.pages):
            raise ValueError("gold page source hash differs from document")
        if any(item.source_sha256 != self.source_sha256 for item in self.continuations):
            raise ValueError("gold continuation source hash differs from document")
        if any(item.source_sha256 != self.source_sha256 for item in self.totals):
            raise ValueError("gold total source hash differs from document")
        row_ids = {
            row.row_id
            for page in self.pages
            for table in page.tables
            for row in table.rows
        }
        for continuation in self.continuations:
            if continuation.from_row_id not in row_ids or continuation.to_row_id not in row_ids:
                raise ValueError("gold continuation references a missing row")
        scope_ids = {
            table.table_id
            for page in self.pages
            for table in page.tables
        } | {self.document_id}
        for total in self.totals:
            if total.scope_id not in scope_ids:
                raise ValueError("gold total references a missing scope")
        payload = self.model_dump(mode="json", exclude={"gold_sha256", "created_at"})
        expected_hash = canonical_sha256(payload)
        if not self.gold_sha256:
            object.__setattr__(self, "gold_sha256", expected_hash)
        elif self.gold_sha256 != expected_hash:
            raise ValueError("gold document hash does not match canonical content")
        return self


class ReviewRecordV1(AuthorityModel):
    review_version: Literal["review_record_v1"] = "review_record_v1"
    review_id: str = Field(default="", pattern=SHA256_PATTERN)
    document_id: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    review_kind: ReviewKind
    reviewer_identity: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    tool_identity: str = Field(min_length=1)
    reasoning_identity: str = Field(
        min_length=1,
        description="Opaque review-strategy identity; never a hidden reasoning trace.",
    )
    prompt_identity: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    image_identities: tuple[str, ...] = ()
    image_sha256s: tuple[Digest, ...] = ()
    image_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    independent_first: bool = False
    blind: bool = True
    independent_review_ids: tuple[Digest, ...] = ()
    machine_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decision: ReviewDecision = ReviewDecision.NEEDS_REVIEW
    observations_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="Digest of the final structured review output; never hidden reasoning.",
    )
    state: ReviewState = ReviewState.DRAFT
    frozen_at: datetime | None = None
    frozen_by: str | None = None
    frozen_payload_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    supersedes_review_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def fill_id(cls, value: Any) -> Any:
        data = _as_dict(value)
        if not data.get("review_id"):
            data["review_id"] = _identity_hash(
                "review",
                {
                    "document_id": data.get("document_id", ""),
                    "source_sha256": data.get("source_sha256", ""),
                    "review_kind": _enum_value(data.get("review_kind", "")),
                    "reviewer_identity": data.get("reviewer_identity", ""),
                    "model_identity": data.get("model_identity", ""),
                    "tool_identity": data.get("tool_identity", ""),
                    "reasoning_identity": data.get("reasoning_identity", ""),
                    "prompt_identity": data.get("prompt_identity", ""),
                    "prompt_sha256": data.get("prompt_sha256", ""),
                    "image_identities": data.get("image_identities", ()),
                    "image_sha256s": data.get("image_sha256s", ()),
                    "image_manifest_sha256": data.get("image_manifest_sha256", ""),
                    "model_config_sha256": data.get("model_config_sha256", ""),
                },
            )
        return data

    @model_validator(mode="after")
    def validate_review(self) -> ReviewRecordV1:
        expected = _identity_hash(
            "review",
            {
                "document_id": self.document_id,
                "source_sha256": self.source_sha256,
                "review_kind": self.review_kind.value,
                "reviewer_identity": self.reviewer_identity,
                "model_identity": self.model_identity,
                "tool_identity": self.tool_identity,
                "reasoning_identity": self.reasoning_identity,
                "prompt_identity": self.prompt_identity,
                "prompt_sha256": self.prompt_sha256,
                "image_identities": self.image_identities,
                "image_sha256s": self.image_sha256s,
                "image_manifest_sha256": self.image_manifest_sha256,
                "model_config_sha256": self.model_config_sha256,
            },
        )
        if self.review_id != expected:
            raise ValueError("review ID does not match immutable review identity")
        if self.document_id != source_document_id_for(self.source_sha256):
            raise ValueError("review document ID must be derived from source bytes")
        if not self.image_sha256s:
            raise ValueError("review requires at least one image identity")
        if self.image_identities and len(self.image_identities) != len(self.image_sha256s):
            raise ValueError("review image identities and hashes must have equal length")
        if len(set(self.image_sha256s)) != len(self.image_sha256s):
            raise ValueError("review image hashes must be unique")
        if self.review_kind in {ReviewKind.INDEPENDENT_A, ReviewKind.INDEPENDENT_B}:
            if not self.blind:
                raise ValueError("independent reviews must be blind")
            if self.independent_first is not False:
                raise ValueError("independent reviews cannot claim adjudicator ordering")
            if self.independent_review_ids:
                raise ValueError("independent reviews cannot reference another review")
            if self.machine_output_sha256 is not None:
                raise ValueError("independent reviews cannot include machine output")
        if self.review_kind is ReviewKind.ADJUDICATOR:
            if not self.independent_first:
                raise ValueError("adjudicator review must be independent-first")
            if len(self.independent_review_ids) < 2:
                raise ValueError("adjudicator review requires both independent reviews")
            if len(set(self.independent_review_ids)) != len(self.independent_review_ids):
                raise ValueError("adjudicator independent reviews must be unique")
            if self.review_id in self.independent_review_ids:
                raise ValueError("adjudicator cannot reference itself")
            if self.blind:
                raise ValueError("adjudicator must see the independent review decisions")
        if self.machine_output_sha256 is not None:
            raise ValueError("reviews cannot include machine output")
        if self.review_kind is ReviewKind.RED_TEAM:
            if self.machine_output_sha256 is not None:
                raise ValueError("red-team review cannot be seeded by machine output")
            if not self.blind:
                raise ValueError("red-team review must be blind to machine output")
        if self.state is ReviewState.FROZEN:
            if self.frozen_at is None or not self.frozen_by:
                raise ValueError("frozen reviews require frozen_at and frozen_by")
            expected_payload = canonical_sha256(
                self.model_dump(
                    mode="json",
                    exclude={"frozen_payload_sha256", "created_at", "state"},
                )
            )
            if (
                self.frozen_payload_sha256 is not None
                and self.frozen_payload_sha256 != expected_payload
            ):
                raise ValueError("frozen review payload hash does not match review content")
            if self.frozen_payload_sha256 is None:
                object.__setattr__(self, "frozen_payload_sha256", expected_payload)
        else:
            if self.frozen_at is not None or self.frozen_by is not None:
                raise ValueError("unfrozen reviews cannot carry freeze metadata")
            if self.frozen_payload_sha256 is not None:
                raise ValueError("unfrozen reviews cannot carry a frozen payload hash")
        if self.state is ReviewState.SUPERSEDED and not self.supersedes_review_id:
            raise ValueError("superseded reviews require supersedes_review_id")
        if self.state is not ReviewState.SUPERSEDED and self.supersedes_review_id is not None:
            raise ValueError("only superseded reviews may reference a superseded record")
        return self


class MetricObservationV1(AuthorityModel):
    metric_name: str = Field(min_length=1)
    value: float
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    cohort: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metric(self) -> MetricObservationV1:
        if not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        if self.denominator == 0 and self.numerator not in (None, 0):
            raise ValueError("metric numerator cannot be positive with zero denominator")
        if (
            self.numerator is not None
            and self.denominator is not None
            and self.numerator > self.denominator
        ):
            raise ValueError("metric numerator cannot exceed denominator")
        return self


class EvaluatorManifestV1(AuthorityModel):
    manifest_version: Literal["evaluator_manifest_v1"] = "evaluator_manifest_v1"
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    code_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    gold_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    metric_names: tuple[str, ...] = ()
    normalization_policy: str = Field(min_length=1)
    matching_policy: str = Field(min_length=1)
    unreadable_policy: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        return data

    @model_validator(mode="after")
    def validate_hash(self) -> EvaluatorManifestV1:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256", "created_at"})
        expected_hash = canonical_sha256(payload)
        if not self.manifest_sha256:
            object.__setattr__(self, "manifest_sha256", expected_hash)
        elif self.manifest_sha256 != expected_hash:
            raise ValueError("evaluator manifest hash does not match canonical content")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("evaluator metric names must be unique")
        return self


class BaselineDocumentV1(AuthorityModel):
    document_id: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    result_sha256: str = Field(pattern=SHA256_PATTERN)
    metrics: tuple[MetricObservationV1, ...] = ()
    status: Literal["complete", "needs_review", "failed"]

    @model_validator(mode="after")
    def validate_document_id(self) -> BaselineDocumentV1:
        if self.document_id != source_document_id_for(self.source_sha256):
            raise ValueError("baseline document ID must be derived from source bytes")
        return self


def _baseline_document_identity(value: Any) -> dict[str, Any]:
    """Return the result-bearing identity used by ``baseline_id``.

    ``baseline_id`` is the stable identity of the evaluated result set, not merely
    the identity of the source cohort.  Keep this helper deliberately explicit so
    that adding a result/status/metric field cannot silently leave the baseline ID
    unchanged.
    """

    data = _as_dict(value)
    metrics = data.get("metrics", ())
    return {
        "source_sha256": data.get("source_sha256", ""),
        "result_sha256": data.get("result_sha256", ""),
        "status": _enum_value(data.get("status", "")),
        "metrics": [_json_value(item) for item in metrics],
    }


class BaselineManifestV1(AuthorityModel):
    manifest_version: Literal["baseline_manifest_v1"] = "baseline_manifest_v1"
    baseline_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    gold_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    documents: tuple[BaselineDocumentV1, ...]
    release_revision: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hashes(cls, value: Any) -> Any:
        data = _as_dict(value)
        documents = tuple(
            BaselineDocumentV1.model_validate(item)
            for item in data.get("documents", ())
        )
        data["documents"] = documents
        if not data.get("baseline_id"):
            data["baseline_id"] = _identity_hash(
                "baseline",
                {
                    "source_manifest_sha256": data.get("source_manifest_sha256", ""),
                    "gold_manifest_sha256": data.get("gold_manifest_sha256", ""),
                    "evaluator_manifest_sha256": data.get("evaluator_manifest_sha256", ""),
                    "release_revision": data.get("release_revision", ""),
                    "model_identity": data.get("model_identity", ""),
                    "configuration_sha256": data.get("configuration_sha256", ""),
                    "documents": [_baseline_document_identity(item) for item in documents],
                },
            )
        return data

    @model_validator(mode="after")
    def validate_manifest(self) -> BaselineManifestV1:
        if not self.documents:
            raise ValueError("baseline manifest requires at least one document")
        expected_id = _identity_hash(
            "baseline",
            {
                "source_manifest_sha256": self.source_manifest_sha256,
                "gold_manifest_sha256": self.gold_manifest_sha256,
                "evaluator_manifest_sha256": self.evaluator_manifest_sha256,
                "release_revision": self.release_revision,
                "model_identity": self.model_identity,
                "configuration_sha256": self.configuration_sha256,
                "documents": [_baseline_document_identity(item) for item in self.documents],
            },
        )
        if self.baseline_id != expected_id:
            raise ValueError("baseline ID does not match canonical identity")
        hashes = [item.source_sha256 for item in self.documents]
        if len(hashes) != len(set(hashes)):
            raise ValueError("baseline documents must have unique source hashes")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256", "created_at"})
        expected_hash = canonical_sha256(payload)
        if not self.manifest_sha256:
            object.__setattr__(self, "manifest_sha256", expected_hash)
        elif self.manifest_sha256 != expected_hash:
            raise ValueError("baseline manifest hash does not match canonical content")
        return self


class GateCohortResultV1(AuthorityModel):
    cohort: str = Field(min_length=1)
    required_count: int = Field(gt=0)
    observed_count: int = Field(ge=0)
    passed: bool
    metrics: tuple[MetricObservationV1, ...] = ()
    blocking_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_count(self) -> GateCohortResultV1:
        expected_count = AUTHORITY_COHORT_COUNTS.get(self.cohort)
        if expected_count is None:
            raise ValueError(f"unknown authority cohort: {self.cohort}")
        if self.required_count != expected_count:
            raise ValueError(
                f"{self.cohort} requires exactly {expected_count} observed documents"
            )
        if self.observed_count > self.required_count:
            raise ValueError("observed cohort count cannot exceed required count")
        if self.passed and self.observed_count != self.required_count:
            raise ValueError("a passing cohort must have exactly its required count")
        return self


class M0M4GateReportV1(AuthorityModel):
    report_version: Literal["m0_m4_gate_report_v1"] = "m0_m4_gate_report_v1"
    milestone: Literal["M0", "M1", "M2", "M3", "M4"]
    decision: GateDecision
    report_id: str = Field(default="", pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    gold_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_revision: str = Field(min_length=1)
    candidate_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    cohorts: tuple[GateCohortResultV1, ...]
    metrics: tuple[MetricObservationV1, ...] = ()
    evidence_sha256s: tuple[Digest, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    report_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hashes(cls, value: Any) -> Any:
        data = _as_dict(value)
        cohorts = tuple(GateCohortResultV1.model_validate(item) for item in data.get("cohorts", ()))
        data["cohorts"] = cohorts
        if not data.get("report_id"):
            data["report_id"] = _identity_hash(
                "gate_report",
                {
                    "milestone": data.get("milestone", ""),
                    "candidate_revision": data.get("candidate_revision", ""),
                    "candidate_configuration_sha256": data.get(
                        "candidate_configuration_sha256", ""
                    ),
                    "source_manifest_sha256": data.get("source_manifest_sha256", ""),
                    "gold_manifest_sha256": data.get("gold_manifest_sha256", ""),
                    "evaluator_manifest_sha256": data.get("evaluator_manifest_sha256", ""),
                    "baseline_manifest_sha256": data.get("baseline_manifest_sha256", ""),
                },
            )
        return data

    @model_validator(mode="after")
    def validate_report(self) -> M0M4GateReportV1:
        expected_cohorts = set(AUTHORITY_COHORTS)
        observed_cohorts = {item.cohort for item in self.cohorts}
        if observed_cohorts != expected_cohorts or len(self.cohorts) != len(expected_cohorts):
            raise ValueError(
                "gate report requires exactly production14, passing36, and staging159 cohorts"
            )
        if tuple(item.cohort for item in self.cohorts) != AUTHORITY_COHORTS:
            raise ValueError("gate report cohorts must use canonical authority order")
        expected_id = _identity_hash(
            "gate_report",
            {
                "milestone": self.milestone,
                "candidate_revision": self.candidate_revision,
                "candidate_configuration_sha256": self.candidate_configuration_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "gold_manifest_sha256": self.gold_manifest_sha256,
                "evaluator_manifest_sha256": self.evaluator_manifest_sha256,
                "baseline_manifest_sha256": self.baseline_manifest_sha256,
            },
        )
        if self.report_id != expected_id:
            raise ValueError("gate report ID does not match canonical identity")
        if len({item.cohort for item in self.cohorts}) != len(self.cohorts):
            raise ValueError("gate report cohorts must be unique")
        if self.decision is GateDecision.PROMOTE:
            has_failed_cohort = any(not cohort.passed for cohort in self.cohorts)
            if self.blocking_reasons or not self.cohorts or has_failed_cohort:
                raise ValueError("a promoted gate cannot have blocking reasons or failed cohorts")
        if self.decision is GateDecision.HOLD and not self.blocking_reasons:
            raise ValueError("a held gate requires at least one blocking reason")
        payload = self.model_dump(mode="json", exclude={"report_sha256", "created_at"})
        expected_hash = canonical_sha256(payload)
        if not self.report_sha256:
            object.__setattr__(self, "report_sha256", expected_hash)
        elif self.report_sha256 != expected_hash:
            raise ValueError("gate report hash does not match canonical content")
        return self


# Short aliases make the structural records convenient to import without
# weakening the explicit versioned names used in serialized manifests.
SourcePage = SourcePageV1
SourceTable = SourceTableV1
SourceColumn = SourceColumnV1
SourceRow = SourceRowV1
SourceCell = SourceCellV1
SourceContinuation = SourceContinuationV1
SourceTotal = SourceTotalV1
GoldPage = GoldPageV2
GoldTable = GoldTableV2
GoldColumn = GoldColumnV2
GoldRow = GoldRowV2
GoldCell = GoldCellV2
GoldContinuation = GoldContinuationV2
GoldTotal = GoldTotalV2


__all__ = [
    "AUTHORITY_COHORTS",
    "AUTHORITY_COHORT_COUNTS",
    "AuthorityModel",
    "BaselineDocumentV1",
    "BaselineManifestV1",
    "ContinuationKind",
    "Digest",
    "EvaluatorManifestV1",
    "GateCohortResultV1",
    "GateDecision",
    "GoldCell",
    "GoldCellV2",
    "GoldColumn",
    "GoldColumnV2",
    "GoldContinuation",
    "GoldContinuationV2",
    "GoldDocumentV2",
    "GoldPage",
    "GoldPageV2",
    "GoldRow",
    "GoldRowV2",
    "GoldTable",
    "GoldTableV2",
    "GoldTotal",
    "GoldTotalV2",
    "MetricObservationV1",
    "M0M4GateReportV1",
    "Readability",
    "ReviewDecision",
    "ReviewKind",
    "ReviewRecordV1",
    "ReviewState",
    "RowKind",
    "SourceCell",
    "SourceCellV1",
    "SourceClass",
    "SourceColumn",
    "SourceColumnV1",
    "SourceContinuation",
    "SourceContinuationV1",
    "SourceManifestV1",
    "SourcePage",
    "SourcePageV1",
    "SourceRow",
    "SourceRowV1",
    "SourceTable",
    "SourceTableV1",
    "SourceTotal",
    "SourceTotalV1",
    "SHA256_PATTERN",
    "TableKind",
    "TotalKind",
    "cell_id_for",
    "column_id_for",
    "continuation_id_for",
    "document_id_for",
    "page_id_for",
    "row_id_for",
    "source_document_id_for",
    "table_id_for",
    "total_id_for",
]
