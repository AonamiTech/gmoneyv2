from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import fitz
from pydantic import Field, ValidationError

from gmoney.contracts.common import ContractModel
from gmoney.contracts.extraction import CanonicalRow, SourceTable
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.date_context import service_date_from_context
from gmoney.extraction.typed_values import parse_decimal, parse_quantity

VALIDATION_VERSION = "extraction_validation_v3"
SUPPORTED_OUTPUT_VERSION = "offline_accuracy_spine_v3"


class ValidationSeverity(StrEnum):
    FATAL = "fatal"
    BLOCKING = "blocking"
    WARNING = "warning"


class ValidationStatus(StrEnum):
    PASSED = "passed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ValidationCategory(StrEnum):
    INTEGRITY = "integrity"
    GROUNDING = "grounding"
    LINKING = "linking"
    SPLITTING = "splitting"
    SERVICE_DATE = "service_date"
    FINANCIAL_FORM = "financial_form"
    DUPLICATE = "duplicate"
    TOTAL = "total"


class RecoveryScope(StrEnum):
    TABLE = "table"
    PAGE = "page"


class ValidationIssue(ContractModel):
    id: str
    code: str
    severity: ValidationSeverity
    message: str
    category: ValidationCategory = ValidationCategory.INTEGRITY
    recovery_scope: RecoveryScope | None = None
    page_number: int | None = Field(default=None, ge=1)
    table_id: str | None = None
    source_row_id: str | None = None
    canonical_row_id: str | None = None
    field: str | None = None
    related_source_row_ids: tuple[str, ...] = ()
    related_canonical_row_ids: tuple[str, ...] = ()


class ValidationReport(ContractModel):
    validation_version: str = VALIDATION_VERSION
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]

    @property
    def fatal(self) -> bool:
        return any(issue.severity is ValidationSeverity.FATAL for issue in self.issues)

    @property
    def recovery_targets(self) -> tuple[tuple[int, str | None], ...]:
        targets: list[tuple[int, str | None]] = []
        for issue in self.issues:
            if (
                issue.severity is not ValidationSeverity.BLOCKING
                or issue.recovery_scope is None
                or issue.page_number is None
            ):
                continue
            target = (
                issue.page_number,
                issue.table_id if issue.recovery_scope is RecoveryScope.TABLE else None,
            )
            if target not in targets:
                targets.append(target)
        page_targets = {page for page, table in targets if table is None}
        return tuple(
            target for target in targets if target[1] is None or target[0] not in page_targets
        )


class ExtractionIntegrityError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__("extraction_integrity_failed")
        self.report = report


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _issue(
    code: str,
    severity: ValidationSeverity,
    message: str,
    *,
    category: ValidationCategory = ValidationCategory.INTEGRITY,
    recovery_scope: RecoveryScope | None = None,
    page_number: int | None = None,
    table_id: str | None = None,
    source_row_id: str | None = None,
    canonical_row_id: str | None = None,
    field: str | None = None,
    related_source_row_ids: tuple[str, ...] = (),
    related_canonical_row_ids: tuple[str, ...] = (),
) -> ValidationIssue:
    if category is ValidationCategory.INTEGRITY:
        if code in {
            "mapped_cell_evidence_mismatch",
            "mapped_cell_value_mismatch",
            "canonical_value_missing_printed",
            "printed_value_missing_canonical",
            "ungrounded_billable_amount",
            "ungrounded_description",
            "unmapped_financial_source_cell",
            "unlinked_financial_source_row",
        }:
            category = ValidationCategory.GROUNDING
        elif code in {
            "canonical_source_link_missing",
            "duplicate_canonical_source_link",
            "unknown_canonical_source_link",
        }:
            category = ValidationCategory.LINKING
        elif code == "service_date_evidence_mismatch":
            category = ValidationCategory.SERVICE_DATE
        elif code == "financial_form_unresolved":
            category = ValidationCategory.FINANCIAL_FORM
        elif code == "possible_duplicate_supporting_charge":
            category = ValidationCategory.DUPLICATE
        elif "total" in code:
            category = ValidationCategory.TOTAL
    if recovery_scope is None and category in {
        ValidationCategory.GROUNDING,
        ValidationCategory.LINKING,
        ValidationCategory.SPLITTING,
        ValidationCategory.SERVICE_DATE,
    }:
        recovery_scope = RecoveryScope.TABLE
    if recovery_scope is None and category is ValidationCategory.FINANCIAL_FORM:
        recovery_scope = RecoveryScope.PAGE
    identity = {
        "code": code,
        "page_number": page_number,
        "table_id": table_id,
        "source_row_id": source_row_id,
        "canonical_row_id": canonical_row_id,
        "field": field,
        "category": category,
        "recovery_scope": recovery_scope,
        "related_source_row_ids": sorted(related_source_row_ids),
        "related_canonical_row_ids": sorted(related_canonical_row_ids),
    }
    issue_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return ValidationIssue(
        id=issue_id,
        code=code,
        severity=severity,
        message=re.sub(r"\s+", " ", message).strip(),
        category=category,
        recovery_scope=recovery_scope,
        page_number=page_number,
        table_id=table_id,
        source_row_id=source_row_id,
        canonical_row_id=canonical_row_id,
        field=field,
        related_source_row_ids=related_source_row_ids,
        related_canonical_row_ids=related_canonical_row_ids,
    )


def _report(issues: list[ValidationIssue]) -> ValidationReport:
    unique = {issue.id: issue for issue in issues}
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.page_number or 0,
                item.table_id or "",
                item.source_row_id or "",
                item.canonical_row_id or "",
                item.field or "",
                item.code,
            ),
        )
    )
    status = (
        ValidationStatus.FAILED
        if any(issue.severity is ValidationSeverity.FATAL for issue in ordered)
        else (
            ValidationStatus.NEEDS_REVIEW
            if any(issue.severity is ValidationSeverity.BLOCKING for issue in ordered)
            else ValidationStatus.PASSED
        )
    )
    return ValidationReport(status=status, issues=ordered)


def _evidence_ids(items: object) -> set[str]:
    return {
        str(token_id)
        for item in items or ()
        for token_id in (
            item.get("token_ids", ())
            if isinstance(item, dict)
            else getattr(item, "token_ids", ())
        )
    }


def _path_uses_symlink(path: Path, root: Path) -> bool:
    current = path
    while current != root and root in current.parents:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _validate_evidence_refs(
    items: object,
    assets: dict[int, dict[str, Any]],
    issues: list[ValidationIssue],
    *,
    page_number: int | None,
    table_id: str | None = None,
    source_row_id: str | None = None,
    canonical_row_id: str | None = None,
    field: str | None = None,
) -> None:
    for evidence in items or ():
        evidence_page = int(getattr(evidence, "page_number", 0) or 0)
        asset = assets.get(evidence_page)
        points = tuple(getattr(getattr(evidence, "polygon", None), "points", ()))
        invalid = bool(
            asset is None
            or evidence_page != page_number
            or getattr(evidence, "artifact_sha256", None) != asset.get("artifact_sha256")
            or not getattr(evidence, "token_ids", ())
            or len(points) < 3
            or any(
                point.x < 0
                or point.y < 0
                for point in points
            )
        )
        if invalid:
            issues.append(
                _issue(
                    "evidence_artifact_mismatch",
                    ValidationSeverity.FATAL,
                    "Evidence does not identify a valid token polygon on its page artifact",
                    page_number=page_number,
                    table_id=table_id,
                    source_row_id=source_row_id,
                    canonical_row_id=canonical_row_id,
                    field=field,
                )
            )


_SUMMARY_WORDS = {
    "amount",
    "bill",
    "charge",
    "charges",
    "ipd",
    "name",
    "package",
    "service",
    "services",
    "summary",
    "total",
}


def _meaningful_summary_words(value: object) -> set[str]:
    return {
        word
        for word in _normalized(value).split()
        if len(word) >= 3 and word not in _SUMMARY_WORDS
    }


def _source_table_signature(
    table: SourceTable,
) -> tuple[Any, tuple[tuple[str, str | None], ...]]:
    return (
        table.table_type,
        tuple(
            (column.id, column.canonical_field)
            for column in table.columns
        ),
    )


def _source_table_slot(table: SourceTable) -> str | None:
    match = re.search(r"(?:^|-)t(?P<slot>\d+)$", table.table_id)
    return match.group("slot") if match is not None else None


def _source_tables_are_contiguous(
    previous: SourceTable,
    following: SourceTable,
) -> bool:
    if _source_table_signature(previous) != _source_table_signature(following):
        return False
    if previous.page_number == following.page_number:
        return previous.table_id == following.table_id
    if previous.page_number + 1 != following.page_number:
        return False
    previous_slot = _source_table_slot(previous)
    following_slot = _source_table_slot(following)
    return (
        previous_slot is None
        or following_slot is None
        or previous_slot == following_slot
    )


def _has_pharmacy_tail_summary(table: SourceTable) -> bool:
    for row in table.rows:
        if row.canonical_row_id is not None:
            continue
        label = _normalized(
            " ".join(
                cell.raw_value.strip()
                for cell in row.cells
                if cell.raw_value
                and cell.raw_value.strip()
                and parse_decimal(cell.raw_value) is None
            )
        )
        words = set(label.split())
        is_summary = (
            "return" in words
            or "returns" in words
            or {"total", "amount"}.issubset(words)
        )
        if is_summary and any(
            parse_decimal(cell.raw_value or "") is not None
            for cell in row.cells
        ):
            return True
    return False


def _unlinked_financial_row_is_explained(
    *,
    table: SourceTable,
    source_tables: tuple[SourceTable, ...],
    source_row: Any,
    cells: dict[str, Any],
    financial_values: tuple[tuple[str, Any], ...],
    canonical_rows: dict[str, dict[str, Any]],
    result: dict[str, Any],
) -> bool:
    """Accept grounded raw aggregates that intentionally are not ledger rows."""
    structured_fields = {
        "service_date_raw",
        "request_no",
        "service_code",
        "hsn_code",
    }
    fields_by_column = {
        column.id: column.canonical_field for column in table.columns
    }
    total_labels = {
        "bill amount",
        "bill total",
        "total",
        "totals",
        "sub total",
        "subtotal",
    }
    total_prefixes = (
        "grand total",
        "gross bill amount",
        "net bill amount",
        "net medical amount",
        "net payable",
        "total bill amount",
        "total gross bill value",
        "total payable amount",
    )

    def is_structured_identifier_cell(cell: Any) -> bool:
        return fields_by_column.get(cell.column_id) in structured_fields

    def is_printed_label_cell(cell: Any) -> bool:
        if not is_structured_identifier_cell(cell):
            return True
        return _normalized(cell.raw_value or "") in total_labels

    def is_section_subtotal_label(value: str) -> bool:
        return (
            value in {"bill total", "sub total", "subtotal"}
            or value.startswith(("sub total ", "subtotal "))
        )

    label_values = tuple(
        cell.raw_value.strip()
        for cell in source_row.cells
        if cell.raw_value and cell.raw_value.strip()
        and parse_decimal(cell.raw_value) is None
        and is_printed_label_cell(cell)
    )
    normalized_label = _normalized(" ".join(label_values))
    discount_labels = {"discount", "discount rs"}
    settlement_labels = {
        "advance received",
        "advance received amount",
        "amount received",
        "amount to be received",
        "balance amount",
        "balance due",
        "company amount",
        "corporate amount",
        "deposit amount",
        "deposit received",
        "net tpa",
        "patient amount",
        "patient balance",
        "patient received",
        "payer amount",
        "payer receivable",
        "payer received",
        "payment detail",
        "payment details",
        "payment mode",
        "payment summary",
        "pre authorization amount",
        "co payment",
        "claim amount",
        "receipt detail",
        "receipt details",
        "receipt history",
        "receipt information",
        "settlement detail",
        "settlement details",
        "settlement mode",
        "settlement status",
        "total discount amount",
    }

    def is_settlement_label(value: str) -> bool:
        normalized = _normalized(value)
        return normalized in discount_labels or normalized in settlement_labels

    def is_structurally_grounded_settlement(
        row_cells: dict[str, Any],
    ) -> bool:
        description_values = tuple(
            row_cells[column.id].raw_value
            for column in table.columns
            if column.canonical_field == "description"
            and row_cells[column.id].raw_value
        )
        financial_columns = tuple(
            column
            for column in table.columns
            if column.canonical_field in {"net_amount", "gross_amount"}
            and (raw_value := row_cells[column.id].raw_value)
            and raw_value.strip()
            and parse_decimal(raw_value) is not None
        )
        if len(financial_columns) != 1:
            return False
        if (
            len(description_values) == 1
            and is_settlement_label(description_values[0])
        ):
            return True
        displaced_labels = tuple(
            row_cells[column.id].raw_value
            for column in table.columns
            if column.canonical_field not in {"net_amount", "gross_amount"}
            and row_cells[column.id].raw_value
            and parse_decimal(row_cells[column.id].raw_value or "") is None
        )
        if (
            not description_values
            and len(displaced_labels) == 1
            and is_settlement_label(displaced_labels[0])
        ):
            return True
        mapped_descriptions = tuple(
            column
            for column in table.columns
            if column.canonical_field == "description"
        )
        if not mapped_descriptions or description_values:
            return False
        financial_index = table.columns.index(financial_columns[0])
        if financial_index == 0:
            return False
        adjacent = row_cells[table.columns[financial_index - 1].id].raw_value
        return bool(adjacent and is_settlement_label(adjacent))

    def cell_bounds(cell: Any) -> tuple[float, float, float, float] | None:
        points = tuple(
            point
            for item in cell.evidence
            if item.token_ids
            for point in item.polygon.points
        )
        if not points:
            return None
        return (
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    description_column = next(
        (
            column
            for column in table.columns
            if column.canonical_field == "description"
        ),
        None,
    )

    def is_grounded_description_continuation(
        preceding_rows: tuple[Any, ...],
        row_index: int,
        *,
        previous_is_continuation: bool,
    ) -> bool:
        if description_column is None or row_index == 0:
            return False
        current = preceding_rows[row_index]
        previous = preceding_rows[row_index - 1]
        if (
            previous.canonical_row_id is None
            and not previous_is_continuation
        ):
            return False
        current_cells = {cell.column_id: cell for cell in current.cells}
        previous_cells = {cell.column_id: cell for cell in previous.cells}
        populated_current = tuple(
            cell
            for cell in current.cells
            if cell.raw_value and cell.raw_value.strip()
        )
        current_description = current_cells[description_column.id]
        previous_description = previous_cells[description_column.id]
        if (
            len(populated_current) != 1
            or populated_current[0].column_id != description_column.id
            or not current_description.raw_value
            or not previous_description.raw_value
        ):
            return False
        current_bounds = cell_bounds(current_description)
        previous_bounds = cell_bounds(previous_description)
        previous_row_bounds = tuple(
            bounds
            for cell in previous.cells
            if cell.column_id != description_column.id
            and cell.raw_value
            and cell.raw_value.strip()
            and (bounds := cell_bounds(cell)) is not None
        )
        if (
            current_bounds is None
            or previous_bounds is None
            or not previous_row_bounds
        ):
            return False
        current_height = max(1.0, current_bounds[3] - current_bounds[1])
        previous_height = max(1.0, previous_bounds[3] - previous_bounds[1])
        line_height = max(current_height, previous_height)
        vertical_gap = current_bounds[1] - previous_bounds[3]
        return (
            current_bounds[1] > previous_bounds[1]
            and current_bounds[1]
            <= max(bounds[3] for bounds in previous_row_bounds)
            and -line_height * 0.25 <= vertical_gap <= line_height * 1.5
            and abs(current_bounds[0] - previous_bounds[0])
            <= max(4.0, line_height * 0.25)
        )

    if is_structurally_grounded_settlement(cells):
        return True

    total_payloads = [
        payload
        for payload in (
            result.get("document_total"),
            *(result.get("document_totals") or []),
        )
        if isinstance(payload, dict) and parse_decimal(str(payload.get("amount"))) is not None
    ]
    total_amounts = {
        parse_decimal(str(payload["amount"]))
        for payload in total_payloads
    }
    if (
        normalized_label in total_labels
        or normalized_label.startswith(total_prefixes)
    ) and all(
        value in total_amounts for _, value in financial_values
    ):
        return True

    summary_words = set(normalized_label.split())
    pharmacy_summary_sign = (
        -1
        if "return" in summary_words or "returns" in summary_words
        else (
            1
            if {"total", "amount"}.issubset(summary_words)
            else 0
        )
    )
    source_row_seen = False
    linked_row_follows = False
    for candidate in table.rows:
        if candidate.id == source_row.id:
            source_row_seen = True
            continue
        if source_row_seen and candidate.canonical_row_id is not None:
            linked_row_follows = True
            break
    if (
        table.table_type.value == "pharmacy"
        and pharmacy_summary_sign
        and not linked_row_follows
    ):
        table_index = next(
            index
            for index, candidate in enumerate(source_tables)
            if candidate is table
        )
        pharmacy_start = table_index
        while pharmacy_start > 0:
            previous = source_tables[pharmacy_start - 1]
            following = source_tables[pharmacy_start]
            if (
                not _source_tables_are_contiguous(previous, following)
                or _has_pharmacy_tail_summary(previous)
            ):
                break
            pharmacy_start -= 1
        pharmacy_row_ids = {
            row.canonical_row_id
            for candidate in source_tables[pharmacy_start : table_index + 1]
            for row in candidate.rows
            if row.canonical_row_id is not None
        }
        pharmacy_rows = tuple(
            row
            for row_id, row in canonical_rows.items()
            if row_id in pharmacy_row_ids
            if row.get("table_type") == "pharmacy"
            and row.get("role") in {"detail", "refund", "category_rollup"}
        )
        if pharmacy_rows and all(
            sum(
                (
                    parsed
                    for row in pharmacy_rows
                    if (parsed := parse_decimal(str(row.get(field)))) is not None
                    and (
                        parsed > 0
                        if pharmacy_summary_sign > 0
                        else parsed < 0
                    )
                ),
                Decimal("0"),
            )
            == value
            for field, value in financial_values
        ):
            return True

    if normalized_label in {"bill total", "total", "totals"}:
        physical_table_rows = [
            row
            for row in canonical_rows.values()
            if int(row.get("page_number") or 0) == table.page_number
            and str(row.get("table_id") or "") == table.table_id
            and row.get("role") in {"detail", "refund", "category_rollup"}
        ]
        if (
            not linked_row_follows
            and physical_table_rows
            and all(
                sum(
                    (
                        parsed
                        for row in physical_table_rows
                        if (parsed := parse_decimal(str(row.get(field)))) is not None
                    ),
                    Decimal("0"),
                )
                == value
                for field, value in financial_values
            )
        ):
            return True

    if is_section_subtotal_label(normalized_label):
        section_rows: list[dict[str, Any]] = []
        section_heading_words: set[str] = set()
        table_index = next(
            index
            for index, candidate in enumerate(source_tables)
            if candidate is table
        )
        section_start = table_index
        while section_start > 0:
            previous = source_tables[section_start - 1]
            following = source_tables[section_start]
            if not _source_tables_are_contiguous(previous, following):
                break
            section_start -= 1
        preceding_rows = tuple(
            preceding
            for candidate in source_tables[section_start : table_index + 1]
            for preceding in candidate.rows
            if candidate is not table or preceding.order < source_row.order
        )
        previous_was_continuation = False
        for preceding_index, preceding in enumerate(preceding_rows):
            preceding_cells = {
                cell.column_id: cell for cell in preceding.cells
            }
            if preceding.canonical_row_id is not None:
                canonical = canonical_rows.get(preceding.canonical_row_id)
                if canonical and canonical.get("role") in {
                    "detail",
                    "refund",
                    "category_rollup",
                }:
                    section_rows.append(canonical)
                previous_was_continuation = False
                continue
            preceding_label = _normalized(
                " ".join(
                    cell.raw_value.strip()
                    for cell in preceding.cells
                    if cell.raw_value
                    and cell.raw_value.strip()
                    and parse_decimal(cell.raw_value) is None
                    and is_printed_label_cell(cell)
                )
            )
            preceding_has_section_text = any(
                cell.raw_value
                and re.search(r"[a-z]", _normalized(cell.raw_value))
                and is_printed_label_cell(cell)
                for cell in preceding.cells
            )
            preceding_has_financial_value = any(
                column.canonical_field in {"net_amount", "gross_amount"}
                and preceding_cells[column.id].raw_value
                and parse_decimal(preceding_cells[column.id].raw_value or "")
                is not None
                for column in table.columns
            )
            preceding_is_financial_boundary = (
                preceding_label in total_labels
                or preceding_label.startswith(total_prefixes)
                or is_section_subtotal_label(preceding_label)
                or is_structurally_grounded_settlement(preceding_cells)
            )
            preceding_is_continuation = is_grounded_description_continuation(
                preceding_rows,
                preceding_index,
                previous_is_continuation=previous_was_continuation,
            )
            if preceding_is_financial_boundary:
                section_rows.clear()
                section_heading_words.clear()
            elif (
                preceding_label
                and preceding_has_section_text
                and not preceding_has_financial_value
                and not preceding_is_continuation
            ):
                section_rows.clear()
                section_heading_words = _meaningful_summary_words(
                    preceding_label
                )
            previous_was_continuation = preceding_is_continuation
        subtotal_scope = re.sub(
            r"^(?:sub\s+total|subtotal)\s*",
            "",
            normalized_label,
        ).strip()
        scope_words = _meaningful_summary_words(subtotal_scope)
        section_words = set().union(
            *(
                _meaningful_summary_words(
                    " ".join(
                        str(value)
                        for value in (
                            row.get("section"),
                            row.get("description"),
                        )
                        if value
                    )
                )
                for row in section_rows
            ),
        )
        section_words.update(section_heading_words)
        if (
            section_rows
            and (not scope_words or scope_words.issubset(section_words))
            and all(
            sum(
                (
                    parsed
                    for row in section_rows
                    if (parsed := parse_decimal(str(row.get(field)))) is not None
                ),
                Decimal("0"),
            )
            == value
            for field, value in financial_values
            )
        ):
            return True

    if table.table_type.value not in {"category_summary", "package_summary"}:
        return False
    printed_descriptions = tuple(
        cells[column.id].raw_value.strip()
        for column in table.columns
        if column.canonical_field == "description"
        and cells[column.id].raw_value
        and cells[column.id].raw_value.strip()
    )
    normalized_descriptions = {
        _normalized(description) for description in printed_descriptions
    }
    if len(normalized_descriptions) != 1:
        return False
    printed_description = printed_descriptions[0]
    printed_words = _meaningful_summary_words(printed_description)
    if not printed_words:
        return False

    normalized_printed_description = _normalized(printed_description)
    numeric_fields = {
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }
    mapped_numeric_values: list[tuple[str, Decimal]] = []
    for column in table.columns:
        field = str(column.canonical_field or "")
        raw_value = cells[column.id].raw_value
        if field not in numeric_fields or not raw_value or not raw_value.strip():
            continue
        parsed = (
            parse_quantity(raw_value)
            if field == "quantity"
            else parse_decimal(raw_value)
        )
        if parsed is None:
            return False
        mapped_numeric_values.append((field, parsed))
    exact_matches: set[str] = set()
    summary_matches: set[str] = set()
    for row_id, canonical in canonical_rows.items():
        if canonical.get("role") not in {
            "detail",
            "refund",
            "category_rollup",
        }:
            continue
        canonical_description = canonical.get("description")
        canonical_words = _meaningful_summary_words(canonical_description)
        if not all(
            parse_decimal(str(canonical.get(field))) == value
            for field, value in mapped_numeric_values
        ):
            continue
        if _normalized(canonical_description) == normalized_printed_description:
            exact_matches.add(row_id)
        if (
            canonical.get("role") == "category_rollup"
            and printed_words.issubset(canonical_words)
            and all(
                parse_decimal(str(canonical.get(field))) == value
                for field, value in financial_values
            )
        ):
            summary_matches.add(row_id)
    return bool(exact_matches) or len(summary_matches) == 1




def _unlinked_financial_is_explained(
    table: SourceTable,
    source_row: Any,
    cells: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    label = _normalized(
        " ".join(
            cell.raw_value or ""
            for cell in source_row.cells
            if parse_decimal(cell.raw_value or "") is None
        )
    )
    if any(
        marker in label
        for marker in (
            "advance",
            "amount paid",
            "balance",
            "claim amount",
            "co payment",
            "deposit",
            "discount",
            "patient amount",
            "payment",
            "receipt amount",
            "refund",
            "settlement",
            "tpa amount",
        )
    ):
        return True
    if any(
        marker in label
        for marker in (
            "bill total",
            "grand total",
            "gross bill amount",
            "net bill amount",
            "net medical amount",
            "net payable",
            "sub total",
            "subtotal",
            "total bill amount",
            "total payable amount",
        )
    ):
        printed = {
            parse_decimal(str(payload.get("amount")))
            for payload in (
                result.get("document_total"),
                *(result.get("document_totals") or ()),
            )
            if isinstance(payload, dict)
        }
        values = {
            parse_decimal(cell.raw_value or "")
            for cell in cells.values()
            if parse_decimal(cell.raw_value or "") is not None
        }
        return bool(values) and values.issubset(printed)
    return bool(
        {"description_continuation", "excluded_oversized_overlay"}.intersection(
            source_row.validation_flags
        )
    )


def _validate_totals(result: dict[str, Any], issues: list[ValidationIssue]) -> None:
    totals = tuple(
        payload
        for payload in result.get("document_totals") or ()
        if isinstance(payload, dict)
    )
    primary = result.get("document_total")
    document_contexts = {
        str(total.get("context_id"))
        for total in totals
        if total.get("scope") == "document"
        and total.get("context_kind") == "document_final"
        and total.get("context_id")
    }
    if primary is not None and primary not in totals:
        issues.append(
            _issue(
                "primary_total_not_in_candidates",
                ValidationSeverity.BLOCKING,
                "Primary total is not a member of the published total candidates",
                page_number=(
                    (primary or {}).get("page_number")
                    if isinstance(primary, dict)
                    else None
                ),
                field="document_total",
            )
        )
    if isinstance(primary, dict) and (
        primary.get("scope") != "document"
        or primary.get("context_kind") != "document_final"
    ):
        issues.append(
            _issue(
                "invalid_primary_total_scope",
                ValidationSeverity.BLOCKING,
                "Primary total is not from a document-final context",
                page_number=primary.get("page_number"),
                field="document_total",
            )
        )
    if len(document_contexts) > 1:
        issues.append(
            _issue(
                "ambiguous_primary_total",
                ValidationSeverity.BLOCKING,
                "Multiple document-final total contexts remain",
                field="document_total",
            )
        )
    if not document_contexts:
        issues.append(
            _issue(
                "document_total_unavailable",
                ValidationSeverity.WARNING,
                "No grounded document-final total context is available",
                field="document_total",
            )
        )
    for context_id in sorted(document_contexts):
        context_amounts = {
            parse_decimal(str(total.get("amount")))
            for total in totals
            if str(total.get("context_id")) == context_id
            and total.get("context_kind") == "document_final"
        }
        context_amounts.discard(None)
        if len(context_amounts) > 1:
            issues.append(
                _issue(
                    "inconsistent_document_total_context",
                    ValidationSeverity.BLOCKING,
                    "A document-final context contains inconsistent total amounts",
                    field="document_total",
                )
            )
    if len(document_contexts) != 1 and primary is not None:
        issues.append(
            _issue(
                "primary_total_requires_unique_context",
                ValidationSeverity.BLOCKING,
                "Primary total requires exactly one document-final context",
                page_number=(primary or {}).get("page_number"),
                field="document_total",
            )
        )


def validate_extraction_result(
    source: Path,
    result: dict[str, Any],
    artifact_root: Path,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if result.get("output_version") != SUPPORTED_OUTPUT_VERSION:
        issues.append(
            _issue(
                "unsupported_output_contract",
                ValidationSeverity.FATAL,
                "Extraction result is missing the supported production output contract",
                field="output_version",
            )
        )
    try:
        source_sha256 = sha256_file(source)
        with fitz.open(source) as document:
            source_pages = document.page_count
    except (OSError, RuntimeError, ValueError, fitz.FileDataError) as error:
        issues.append(
            _issue(
                "source_pdf_unusable",
                ValidationSeverity.FATAL,
                f"Source PDF cannot be validated: {type(error).__name__}",
            )
        )
        return _report(issues)
    if result.get("source_sha256") != source_sha256:
        issues.append(
            _issue(
                "source_hash_mismatch",
                ValidationSeverity.FATAL,
                "Result source hash does not match the source PDF",
                field="source_sha256",
            )
        )
    if result.get("document_id") != source_sha256:
        issues.append(
            _issue(
                "document_identity_mismatch",
                ValidationSeverity.FATAL,
                "Result document identity does not match the source PDF",
                field="document_id",
            )
        )
    if type(result.get("pages")) is not int or result.get("pages") != source_pages:
        issues.append(
            _issue(
                "page_count_mismatch",
                ValidationSeverity.FATAL,
                "Result page count does not match the source PDF",
                field="pages",
            )
        )
    assets = result.get("page_assets")
    assets_by_page: dict[int, dict[str, Any]] = {}
    if not isinstance(assets, list) or len(assets) != source_pages:
        issues.append(
            _issue(
                "page_inventory_incomplete",
                ValidationSeverity.FATAL,
                "Page artifact inventory is incomplete",
                field="page_assets",
            )
        )
    else:
        for expected_page, asset in enumerate(assets, start=1):
            if not isinstance(asset, dict) or asset.get("page_number") != expected_page:
                issues.append(
                    _issue(
                        "page_inventory_invalid",
                        ValidationSeverity.FATAL,
                        "Page inventory has invalid page numbers and is not contiguous",
                        page_number=expected_page,
                        field="page_assets",
                    )
                )
                continue
            assets_by_page[expected_page] = asset
            relative = Path(str(asset.get("relative_path") or ""))
            unresolved = artifact_root / relative
            path = unresolved.resolve()
            if (
                relative.is_absolute()
                or artifact_root.resolve() not in path.parents
                or not unresolved.is_file()
                or _path_uses_symlink(unresolved, artifact_root)
            ):
                issues.append(
                    _issue(
                        "page_artifact_missing",
                        ValidationSeverity.FATAL,
                        "Page artifact is missing",
                        page_number=expected_page,
                        field="page_assets",
                    )
                )
            elif sha256_file(path) != asset.get("artifact_sha256"):
                issues.append(
                    _issue(
                        "page_artifact_hash_mismatch",
                        ValidationSeverity.FATAL,
                        "Page artifact hash does not match the result",
                        page_number=expected_page,
                        field="page_assets",
                    )
                )

    canonical: dict[str, CanonicalRow] = {}
    rows_payload = result.get("rows")
    if not isinstance(rows_payload, list):
        issues.append(
            _issue(
                "canonical_rows_invalid",
                ValidationSeverity.FATAL,
                "Canonical rows are not a list",
                field="rows",
            )
        )
        rows_payload = []
    for index, payload in enumerate(rows_payload):
        try:
            row = CanonicalRow.model_validate(payload)
        except ValidationError as error:
            issues.append(
                _issue(
                    "canonical_row_contract_invalid",
                    ValidationSeverity.FATAL,
                    "Canonical row "
                    f"{index} failed contract validation: {error.errors()[0]['msg']}",
                    page_number=(
                        (payload or {}).get("page_number")
                        if isinstance(payload, dict)
                        else None
                    ),
                    canonical_row_id=str((payload or {}).get("id") or index),
                    field="rows",
                )
            )
            continue
        row_id = str(row.id)
        if row_id in canonical:
            issues.append(
                _issue(
                    "duplicate_canonical_row_id",
                    ValidationSeverity.FATAL,
                    "Canonical row IDs must be globally unique",
                    page_number=row.page_number,
                    table_id=row.table_id,
                    canonical_row_id=row_id,
                    field="id",
                )
            )
            continue
        canonical[row_id] = row
        _validate_evidence_refs(
            row.evidence,
            assets_by_page,
            issues,
            page_number=row.page_number,
            table_id=row.table_id,
            canonical_row_id=row_id,
            field="evidence",
        )
        for evidence_field, evidence in row.field_evidence.items():
            _validate_evidence_refs(
                evidence,
                assets_by_page,
                issues,
                page_number=row.page_number,
                table_id=row.table_id,
                canonical_row_id=row_id,
                field=evidence_field,
            )
        if not row.description or not row.field_evidence.get("description"):
            issues.append(
                _issue(
                    "ungrounded_description",
                    ValidationSeverity.BLOCKING,
                    "Canonical row lacks a grounded description",
                    page_number=row.page_number,
                    table_id=row.table_id,
                    canonical_row_id=str(row.id),
                    field="description",
                )
            )
        if row.role.value in {"detail", "refund", "category_rollup"} and (
            row.net_amount is None or not row.field_evidence.get("amount")
        ):
            issues.append(
                _issue(
                    "ungrounded_billable_amount",
                    ValidationSeverity.BLOCKING,
                    "Billable row lacks a grounded amount",
                    page_number=row.page_number,
                    table_id=row.table_id,
                    canonical_row_id=str(row.id),
                    field="net_amount",
                )
            )
        if row.role.value == "informational":
            financial_field = next(
                (
                    field
                    for field in (
                        "unit_price",
                        "gross_amount",
                        "discount",
                        "net_amount",
                    )
                    if getattr(row, field) is not None
                ),
                None,
            )
            if financial_field is not None:
                issues.append(
                    _issue(
                        "informational_row_has_money",
                        ValidationSeverity.BLOCKING,
                        "Informational row carries a financial value",
                        page_number=row.page_number,
                        table_id=row.table_id,
                        canonical_row_id=row_id,
                        field=financial_field,
                    )
                )
            if not {"service_date", "request_no", "service_code", "hsn_code"}.intersection(
                row.field_evidence
            ):
                issues.append(
                    _issue(
                        "informational_row_missing_typed_evidence",
                        ValidationSeverity.BLOCKING,
                        "Informational row lacks grounded typed evidence",
                        page_number=row.page_number,
                        table_id=row.table_id,
                        canonical_row_id=row_id,
                    )
                )

    tables: tuple[SourceTable, ...] = ()
    source_payload = result.get("source_tables")
    if canonical and not source_payload:
        issues.append(
            _issue(
                "source_tables_missing",
                ValidationSeverity.FATAL,
                "Canonical rows require grounded source tables",
                field="source_tables",
            )
        )
    elif source_payload:
        parsed_tables: list[SourceTable] = []
        table_ids: set[str] = set()
        source_row_ids: set[str] = set()
        for index, payload in enumerate(source_payload):
            try:
                table = SourceTable.model_validate(payload)
            except ValidationError as error:
                issues.append(
                    _issue(
                        "source_table_contract_invalid",
                        ValidationSeverity.FATAL,
                        "Source table "
                        f"{index} failed grounding validation: {error.errors()[0]['msg']}",
                        page_number=(
                            (payload or {}).get("page_number")
                            if isinstance(payload, dict)
                            else None
                        ),
                        table_id=str((payload or {}).get("table_id") or index),
                        field="source_tables",
                    )
                )
                continue
            if table.id in table_ids:
                issues.append(
                    _issue(
                        "duplicate_source_table_id",
                        ValidationSeverity.FATAL,
                        "Source table IDs must be globally unique",
                        page_number=table.page_number,
                        table_id=table.table_id,
                        field="source_tables",
                    )
                )
                continue
            table_ids.add(table.id)
            duplicate_rows = tuple(row.id for row in table.rows if row.id in source_row_ids)
            if duplicate_rows:
                for source_row_id in duplicate_rows:
                    issues.append(
                        _issue(
                            "duplicate_source_row_id",
                            ValidationSeverity.FATAL,
                            "Source row IDs must be globally unique",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row_id,
                            field="source_tables",
                        )
                    )
                continue
            source_row_ids.update(row.id for row in table.rows)
            parsed_tables.append(table)
        tables = tuple(parsed_tables)

    field_evidence = {
        "description": "description",
        "service_date_raw": "service_date",
        "request_no": "request_no",
        "service_code": "service_code",
        "hsn_code": "hsn_code",
        "quantity": "quantity",
        "unit_price": "rate",
        "gross_amount": "gross_amount",
        "discount": "discount",
        "net_amount": "amount",
    }
    numeric_fields = {"quantity", "unit_price", "gross_amount", "discount", "net_amount"}
    linked_canonical: set[str] = set()
    canonical_payloads = {
        row_id: row.model_dump(mode="json") for row_id, row in canonical.items()
    }
    for table in tables:
        columns = {column.id: column for column in table.columns}
        for column in table.columns:
            _validate_evidence_refs(
                column.evidence,
                assets_by_page,
                issues,
                page_number=table.page_number,
                table_id=table.table_id,
                field=column.id,
            )
        for source_row in table.rows:
            for cell in source_row.cells:
                _validate_evidence_refs(
                    cell.evidence,
                    assets_by_page,
                    issues,
                    page_number=table.page_number,
                    table_id=table.table_id,
                    source_row_id=source_row.id,
                    field=cell.column_id,
                )
            cells = {cell.column_id: cell for cell in source_row.cells}
            unmapped_money_cells = tuple(
                cell
                for cell in source_row.cells
                if columns[cell.column_id].canonical_field is None
                and "inferred_financial_lane"
                in columns[cell.column_id].validation_flags
                and parse_decimal(cell.raw_value or "") is not None
            )
            if (
                source_row.canonical_row_id is not None
                and unmapped_money_cells
                and not _unlinked_financial_is_explained(table, source_row, cells, result)
            ):
                for cell in unmapped_money_cells:
                    issues.append(
                        _issue(
                            "unmapped_financial_source_cell",
                            ValidationSeverity.BLOCKING,
                            "Grounded financial value is in an unmapped source column",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=source_row.canonical_row_id,
                            field=cell.column_id,
                        )
                    )
            if source_row.canonical_row_id is None:
                financial_values = tuple(
                    (
                        str(columns[cell.column_id].canonical_field or cell.column_id),
                        parsed,
                    )
                    for cell in source_row.cells
                    if (parsed := parse_decimal(cell.raw_value or "")) is not None
                    and (
                        columns[cell.column_id].canonical_field
                        in {"gross_amount", "net_amount"}
                        or "inferred_financial_lane"
                        in columns[cell.column_id].validation_flags
                    )
                )
                if financial_values and not _unlinked_financial_row_is_explained(
                    table=table,
                    source_tables=tables,
                    source_row=source_row,
                    cells=cells,
                    financial_values=financial_values,
                    canonical_rows=canonical_payloads,
                    result=result,
                ):
                    issues.append(
                        _issue(
                            "unlinked_financial_source_row",
                            ValidationSeverity.BLOCKING,
                            "Unlinked source row contains an unexplained financial value",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            field="net_amount",
                        )
                    )
                continue
            canonical_id = source_row.canonical_row_id
            row = canonical.get(canonical_id)
            if row is None:
                issues.append(
                    _issue(
                        "unknown_canonical_source_link",
                        ValidationSeverity.FATAL,
                        "Source row links an unknown canonical row",
                        page_number=table.page_number,
                        table_id=table.table_id,
                        source_row_id=source_row.id,
                        canonical_row_id=canonical_id,
                    )
                )
                continue
            if canonical_id in linked_canonical:
                issues.append(
                    _issue(
                        "duplicate_canonical_source_link",
                        ValidationSeverity.BLOCKING,
                        "Canonical row is linked by more than one source row",
                        page_number=table.page_number,
                        table_id=table.table_id,
                        source_row_id=source_row.id,
                        canonical_row_id=canonical_id,
                    )
                )
            linked_canonical.add(canonical_id)
            for column in table.columns:
                field = column.canonical_field
                if field not in field_evidence:
                    continue
                cell = cells[column.id]
                canonical_value = getattr(row, field)
                printed = bool(cell.raw_value and cell.raw_value.strip())
                present = canonical_value is not None and (
                    not isinstance(canonical_value, str) or bool(canonical_value.strip())
                )
                canonical_numeric = (
                    parse_decimal(str(canonical_value))
                    if field in numeric_fields and present
                    else None
                )
                derived_quantity = bool(
                    field == "quantity"
                    and "quantity_derived_from_rate_amount" in row.validation_flags
                    and canonical_numeric is not None
                    and canonical_numeric > 0
                    and canonical_numeric == canonical_numeric.to_integral_value()
                    and row.unit_price is not None
                    and row.unit_price > 0
                    and row.net_amount is not None
                    and canonical_numeric * row.unit_price
                    == abs(row.net_amount) + (row.discount or Decimal("0"))
                )
                if printed and not present:
                    issues.append(
                        _issue(
                            "printed_value_missing_canonical",
                            ValidationSeverity.BLOCKING,
                            f"{field} has a printed value but is missing canonical value",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=canonical_id,
                            field=field,
                        )
                    )
                    continue
                if present and not printed:
                    canonical_ids = _evidence_ids(
                        row.field_evidence.get(field_evidence[field])
                    )
                    supporting_ids = {
                        token_id
                        for supporting_column in table.columns
                        if supporting_column.canonical_field in {"unit_price", "net_amount"}
                        for token_id in _evidence_ids(cells[supporting_column.id].evidence)
                    }
                    if derived_quantity and canonical_ids and canonical_ids.issubset(
                        supporting_ids
                    ):
                        continue
                    serial_description = bool(
                        field == "description"
                        and "missing_printed_description" in row.validation_flags
                        and re.fullmatch(r"\d+[.)]?", str(canonical_value).strip())
                        and canonical_ids
                        and any(
                            candidate.canonical_field is None
                            and _normalized(candidate.label)
                            in {"#", "s no", "serial no", "sr n", "sr no"}
                            and (serial_cell := cells[candidate.id]).raw_value
                            and serial_cell.raw_value.strip()
                            == str(canonical_value).strip()
                            and canonical_ids.issubset(
                                _evidence_ids(serial_cell.evidence)
                            )
                            for candidate in table.columns
                        )
                    )
                    if serial_description:
                        continue
                    issues.append(
                        _issue(
                            "canonical_value_missing_printed",
                            ValidationSeverity.BLOCKING,
                            f"{field} has a canonical value but is missing printed value",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=canonical_id,
                            field=field,
                        )
                    )
                    continue
                if not printed:
                    continue
                canonical_ids = _evidence_ids(row.field_evidence.get(field_evidence[field]))
                cell_ids = _evidence_ids(cell.evidence)
                if not canonical_ids or not canonical_ids.issubset(cell_ids):
                    issues.append(
                        _issue(
                            "mapped_cell_evidence_mismatch",
                            ValidationSeverity.BLOCKING,
                            f"Canonical {field} evidence is not in its mapped source cell",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=canonical_id,
                            field=field,
                        )
                    )
                if field in numeric_fields:
                    printed_value = (
                        parse_quantity(cell.raw_value or "")
                        if field == "quantity"
                        else parse_decimal(cell.raw_value or "")
                    )
                    if derived_quantity and printed_value is None:
                        continue
                    if printed_value is None or printed_value != canonical_numeric:
                        issues.append(
                            _issue(
                                "mapped_cell_value_mismatch",
                                ValidationSeverity.BLOCKING,
                                f"Canonical {field} does not match its mapped source cell",
                                page_number=table.page_number,
                                table_id=table.table_id,
                                source_row_id=source_row.id,
                                canonical_row_id=canonical_id,
                                field=field,
                            )
                        )
                elif field == "service_date_raw":
                    parsed = service_date_from_context(
                        cell.raw_value,
                        column_label=column.label,
                        description=row.description,
                    )
                    if parsed is None or parsed[1] != row.service_date_iso:
                        issues.append(
                            _issue(
                                "service_date_evidence_mismatch",
                                ValidationSeverity.BLOCKING,
                                "service_date_raw value does not match its mapped source cell",
                                page_number=table.page_number,
                                table_id=table.table_id,
                                source_row_id=source_row.id,
                                canonical_row_id=canonical_id,
                                field="service_date_raw",
                            )
                        )
                elif field != "description" and _normalized(cell.raw_value) != _normalized(
                    canonical_value
                ):
                    issues.append(
                        _issue(
                            "mapped_cell_value_mismatch",
                            ValidationSeverity.BLOCKING,
                            f"Canonical {field} does not match its mapped source cell",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=canonical_id,
                            field=field,
                        )
                    )

    for row_id, row in canonical.items():
        if row.service_date_raw and {
            "service_date_corrected_from_source_cell",
            "service_date_inherited_from_group",
            "service_date_recovered_from_source_cell",
        }.intersection(row.validation_flags):
            service_ids = _evidence_ids(row.field_evidence.get("service_date", ()))
            grounded = any(
                table.page_number == row.page_number
                and table.table_id == row.table_id
                and service_ids
                and service_ids.issubset(_evidence_ids(cell.evidence))
                and (
                    parsed := service_date_from_context(
                        cell.raw_value,
                        column_label=next(
                            (
                                column.label
                                for column in table.columns
                                if column.id == cell.column_id
                            ),
                            None,
                        ),
                        description=row.description,
                    )
                )
                is not None
                and parsed[1] == row.service_date_iso
                for table in tables
                for source_row in table.rows
                for cell in source_row.cells
            )
            if not grounded:
                issues.append(
                    _issue(
                        "service_date_evidence_mismatch",
                        ValidationSeverity.BLOCKING,
                        "service_date_raw lacks matching grounded source evidence",
                        page_number=row.page_number,
                        table_id=row.table_id,
                        canonical_row_id=row_id,
                        field="service_date_raw",
                    )
                )
        if "ocr_spatial_graph" in row.source_routes and row_id not in linked_canonical:
            issues.append(
                _issue(
                    "canonical_source_link_missing",
                    ValidationSeverity.BLOCKING,
                    "Canonical OCR row lacks a source-table link",
                    page_number=row.page_number,
                    table_id=row.table_id,
                    canonical_row_id=row_id,
                )
            )

    for diagnostic in result.get("diagnostics") or ():
        if not isinstance(diagnostic, dict):
            continue
        diagnostic_page = int(diagnostic.get("page_number") or 1)
        grounded_supporting_charge = any(
            row.page_number == diagnostic_page
            and str(row.id) in linked_canonical
            and row.net_amount is not None
            and bool(row.field_evidence.get("amount"))
            and (
                "supporting_receipt_charge" in row.validation_flags
                or "receipt_form" in row.source_routes
            )
            for row in canonical.values()
        )
        recognized_nonbillable = diagnostic.get("financial_form_classification") in {
            "recognized_nonbillable",
            "nonbillable_payment",
        } and bool(diagnostic.get("financial_form_classification_evidence"))
        if (
            diagnostic.get("financial_form_suspected")
            and not grounded_supporting_charge
            and not recognized_nonbillable
        ):
            issues.append(
                _issue(
                    "financial_form_unresolved",
                    ValidationSeverity.BLOCKING,
                    "Possible financial form could not be grounded",
                    page_number=diagnostic_page,
                    table_id=diagnostic.get("table_id"),
                )
            )

    for pair in result.get("receipt_duplicate_pairs") or ():
        if not isinstance(pair, dict):
            continue
        canonical_ids = tuple(str(value) for value in pair.get("canonical_row_ids") or ())
        source_ids = tuple(str(value) for value in pair.get("source_row_ids") or ())
        issues.append(
            _issue(
                "possible_duplicate_supporting_charge",
                ValidationSeverity.BLOCKING,
                "Supporting receipt may duplicate another grounded charge",
                page_number=pair.get("page_number"),
                table_id=pair.get("table_id"),
                source_row_id=source_ids[0] if source_ids else None,
                canonical_row_id=canonical_ids[0] if canonical_ids else None,
                related_source_row_ids=source_ids,
                related_canonical_row_ids=canonical_ids,
            )
        )

    _validate_totals(result, issues)
    return _report(issues)
