from __future__ import annotations

import hashlib
import json
import re
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

VALIDATION_VERSION = "extraction_validation_v2"
SUPPORTED_OUTPUT_VERSION = "offline_accuracy_spine_v3"


class ValidationSeverity(StrEnum):
    FATAL = "fatal"
    BLOCKING = "blocking"
    WARNING = "warning"


class ValidationStatus(StrEnum):
    PASSED = "passed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ValidationIssue(ContractModel):
    id: str
    code: str
    severity: ValidationSeverity
    message: str
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
        eligible = {
            "canonical_source_link_missing",
            "canonical_value_missing_printed",
            "date_column_ambiguous",
            "financial_form_unresolved",
            "mapped_cell_evidence_mismatch",
            "mapped_cell_value_mismatch",
            "printed_value_missing_canonical",
            "service_date_evidence_mismatch",
            "unmapped_financial_source_cell",
            "unlinked_financial_source_row",
        }
        return tuple(
            dict.fromkeys(
                (issue.page_number, issue.table_id)
                for issue in self.issues
                if issue.severity is ValidationSeverity.BLOCKING
                and issue.code in eligible
                and issue.page_number is not None
            )
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
    page_number: int | None = None,
    table_id: str | None = None,
    source_row_id: str | None = None,
    canonical_row_id: str | None = None,
    field: str | None = None,
    related_source_row_ids: tuple[str, ...] = (),
    related_canonical_row_ids: tuple[str, ...] = (),
) -> ValidationIssue:
    identity = {
        "code": code,
        "page_number": page_number,
        "table_id": table_id,
        "source_row_id": source_row_id,
        "canonical_row_id": canonical_row_id,
        "field": field,
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
                        "Page artifact inventory is not contiguous",
                        page_number=expected_page,
                        field="page_assets",
                    )
                )
                continue
            relative = Path(str(asset.get("relative_path") or ""))
            path = (artifact_root / relative).resolve()
            if artifact_root.resolve() not in path.parents or not path.is_file():
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
        canonical[str(row.id)] = row
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
        for index, payload in enumerate(source_payload):
            try:
                parsed_tables.append(SourceTable.model_validate(payload))
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
    for table in tables:
        columns = {column.id: column for column in table.columns}
        for source_row in table.rows:
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
                money_cells = tuple(
                    cell
                    for cell in source_row.cells
                    if parse_decimal(cell.raw_value or "") is not None
                    and (
                        columns[cell.column_id].canonical_field in {"gross_amount", "net_amount"}
                        or "inferred_financial_lane" in columns[cell.column_id].validation_flags
                    )
                )
                if money_cells and not _unlinked_financial_is_explained(
                    table, source_row, cells, result
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
                if printed and not present:
                    issues.append(
                        _issue(
                            "printed_value_missing_canonical",
                            ValidationSeverity.BLOCKING,
                            f"Printed {field} is missing from its canonical row",
                            page_number=table.page_number,
                            table_id=table.table_id,
                            source_row_id=source_row.id,
                            canonical_row_id=canonical_id,
                            field=field,
                        )
                    )
                    continue
                if present and not printed:
                    issues.append(
                        _issue(
                            "canonical_value_missing_printed",
                            ValidationSeverity.BLOCKING,
                            f"Canonical {field} is missing from its mapped source cell",
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
                    canonical_numeric = parse_decimal(str(canonical_value))
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
                                "Canonical service date does not match the final Date cell",
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
        if diagnostic.get("financial_form_suspected") and not diagnostic.get(
            "canonical_count"
        ):
            issues.append(
                _issue(
                    "financial_form_unresolved",
                    ValidationSeverity.BLOCKING,
                    "Possible financial form could not be grounded",
                    page_number=int(diagnostic.get("page_number") or 1),
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
