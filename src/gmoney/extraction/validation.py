from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import fitz
from pydantic import Field, ValidationError

from gmoney.contracts.common import ContractModel
from gmoney.contracts.extraction import (
    CanonicalRow,
    DocumentTotal,
    ExtractionResultV5,
    SourceTable,
    TokenManifestEntry,
)
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.date_context import service_date_from_context
from gmoney.extraction.typed_values import parse_decimal, parse_quantity, parse_service_date

VALIDATION_VERSION = "extraction_validation_v5"
SUPPORTED_OUTPUT_VERSION = "offline_accuracy_spine_v5"


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


def _validate_provider_and_recovery_metadata(
    result: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    provider = result.get("provider_usage")
    valid_provider = isinstance(provider, dict) and all(
        isinstance(provider.get(key), dict)
        for key in ("initial", "recovery", "aggregate")
    )
    if not valid_provider:
        issues.append(
            _issue(
                "provider_usage_contract_invalid",
                ValidationSeverity.FATAL,
                "Provider usage must separate initial, recovery, and aggregate usage",
                field="provider_usage",
            )
        )
    else:
        assert isinstance(provider, dict)
        parsed: dict[str, tuple[int, Decimal]] = {}
        for key in ("initial", "recovery", "aggregate"):
            payload = provider[key]
            calls = payload.get("gemini_calls")
            try:
                cost = Decimal(str(payload.get("gemini_measured_cost_usd")))
            except Exception:
                cost = Decimal("-1")
            if type(calls) is not int or calls < 0 or cost < 0:
                issues.append(
                    _issue(
                        "provider_usage_contract_invalid",
                        ValidationSeverity.FATAL,
                        f"Provider usage {key} totals are invalid",
                        field="provider_usage",
                    )
                )
                continue
            parsed[key] = (calls, cost)
        if set(parsed) == {"initial", "recovery", "aggregate"} and parsed[
            "aggregate"
        ] != (
            parsed["initial"][0] + parsed["recovery"][0],
            parsed["initial"][1] + parsed["recovery"][1],
        ):
            issues.append(
                _issue(
                    "provider_usage_aggregate_mismatch",
                    ValidationSeverity.FATAL,
                    "Aggregate provider usage does not equal initial plus recovery usage",
                    field="provider_usage",
                )
            )
        if parsed.get("recovery", (0, Decimal("0")))[0] != 0:
            issues.append(
                _issue(
                    "recovery_gemini_invoked",
                    ValidationSeverity.FATAL,
                    "Targeted recovery invoked Gemini",
                    field="provider_usage",
                )
            )

    recovery = result.get("recovery")
    if not isinstance(recovery, dict) or type(recovery.get("attempted")) is not bool:
        issues.append(
            _issue(
                "recovery_metadata_contract_invalid",
                ValidationSeverity.FATAL,
                "Recovery metadata is missing or invalid",
                field="recovery",
            )
        )
        return
    targets = recovery.get("targets")
    if not isinstance(targets, list) or (not recovery["attempted"] and targets):
        issues.append(
            _issue(
                "recovery_metadata_contract_invalid",
                ValidationSeverity.FATAL,
                "Recovery targets do not match the attempted state",
                field="recovery",
            )
        )
        return
    digest_pattern = re.compile(r"^[a-f0-9]{64}$")
    untargeted_digest = recovery.get("untargeted_units_sha256")
    if recovery["attempted"] and not (
        isinstance(untargeted_digest, str) and digest_pattern.fullmatch(untargeted_digest)
    ):
        issues.append(
            _issue(
                "recovery_metadata_contract_invalid",
                ValidationSeverity.FATAL,
                "Recovery is missing the untargeted-unit digest",
                field="recovery",
            )
        )
    for target in targets:
        if not isinstance(target, dict) or (
            type(target.get("page_number")) is not int
            or target["page_number"] < 1
            or target.get("selected") not in {"baseline", "candidate"}
            or not all(
                isinstance(target.get(field), str)
                and digest_pattern.fullmatch(target[field])
                for field in ("baseline_unit_sha256", "selected_unit_sha256")
            )
            or (
                target.get("candidate_unit_sha256") is not None
                and not (
                    isinstance(target["candidate_unit_sha256"], str)
                    and digest_pattern.fullmatch(target["candidate_unit_sha256"])
                )
            )
        ):
            issues.append(
                _issue(
                    "recovery_metadata_contract_invalid",
                    ValidationSeverity.FATAL,
                    "Recovery target digests or selection are invalid",
                    page_number=(
                        target.get("page_number")
                        if isinstance(target, dict)
                        and type(target.get("page_number")) is int
                        and target["page_number"] >= 1
                        else None
                    ),
                    field="recovery",
                )
            )


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _evidence_token_texts(
    evidence: object,
    tokens: dict[str, TokenManifestEntry],
) -> tuple[str, ...]:
    return tuple(
        token.text
        for item in evidence or ()
        for token_id in getattr(item, "token_ids", ())
        if (token := tokens.get(token_id)) is not None
    )


def _token_text_supports_value(
    field: str | None,
    value: object,
    texts: tuple[str, ...],
    *,
    description: object = None,
) -> bool:
    if value is None or not str(value).strip():
        return True
    if not texts:
        return False
    joined = " ".join(texts)
    if field in {
        "amount",
        "net_amount",
        "gross_amount",
        "discount",
        "rate",
        "unit_price",
        "quantity",
    }:
        expected = parse_quantity(str(value)) if field == "quantity" else parse_decimal(str(value))
        observed: set[Decimal] = set()
        for text in (*texts, joined):
            candidates = (text, *re.findall(r"[-+()]?\d[\d,]*(?:\.\d+)?[)]?", text))
            for candidate in candidates:
                parsed = (
                    parse_quantity(candidate)
                    if field == "quantity"
                    else parse_decimal(candidate)
                )
                if parsed is not None:
                    observed.add(parsed)
        if expected is None:
            return _normalized(value) in _normalized(joined)
        return expected in observed
    if field in {"service_date", "service_date_raw", "service_date_iso"}:
        expected_context = service_date_from_context(
            value,
            column_label="Date",
            canonical_field="service_date_raw",
            description=description,
        )
        expected = (
            expected_context[1]
            if expected_context is not None
            else parse_service_date(str(value))
        )
        if expected is None:
            return _normalized(value) in _normalized(joined)
        return any(
            (
                parsed := service_date_from_context(
                    text,
                    column_label="Date",
                    canonical_field="service_date_raw",
                    description=description,
                )
            )
            is not None
            and parsed[1] == expected
            for text in (*texts, joined)
        )
    expected_text = _normalized(value)
    observed_text = _normalized(joined)
    if field in {"request_no", "service_code", "hsn_code"}:
        return (
            expected_text in {_normalized(text) for text in texts}
            or expected_text == observed_text
        )
    return bool(expected_text) and expected_text in observed_text


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


def _contract_error_message(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        details = error.errors()
        if details:
            return str(details[0].get("msg") or "contract validation failed")
    return type(error).__name__


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
    tokens: dict[str, TokenManifestEntry],
    issues: list[ValidationIssue],
    *,
    page_number: int | None,
    table_id: str | None = None,
    source_row_id: str | None = None,
    canonical_row_id: str | None = None,
    field: str | None = None,
) -> None:
    if not isinstance(items, (tuple, list)):
        issues.append(
            _issue(
                "evidence_contract_invalid",
                ValidationSeverity.FATAL,
                "Evidence references are not a collection",
                page_number=page_number,
                table_id=table_id,
                source_row_id=source_row_id,
                canonical_row_id=canonical_row_id,
                field=field,
            )
        )
        return
    for evidence in items:
        evidence_page = int(getattr(evidence, "page_number", 0) or 0)
        asset = assets.get(evidence_page)
        points = tuple(getattr(getattr(evidence, "polygon", None), "points", ()))
        width = float((asset or {}).get("width") or 0)
        height = float((asset or {}).get("height") or 0)
        evidence_token_ids = tuple(getattr(evidence, "token_ids", ()) or ())
        manifest_tokens = tuple(tokens.get(str(token_id)) for token_id in evidence_token_ids)
        polygon_area = abs(
            sum(
                point.x * points[(index + 1) % len(points)].y
                - points[(index + 1) % len(points)].x * point.y
                for index, point in enumerate(points)
            )
            / 2
        ) if len(points) >= 3 else 0
        invalid = bool(
            asset is None
            or evidence_page != page_number
            or (
                table_id is not None
                and getattr(evidence, "table_id", None) != table_id
            )
            or getattr(evidence, "artifact_sha256", None) != asset.get("artifact_sha256")
            or not evidence_token_ids
            or len(points) < 3
            or polygon_area <= 0
            or any(
                point.x < 0
                or point.y < 0
                or point.x > width
                or point.y > height
                for point in points
            )
            or any(token is None for token in manifest_tokens)
            or any(
                token is not None
                and (
                    token.page_number != evidence_page
                    or token.artifact_sha256 != getattr(evidence, "artifact_sha256", None)
                    or (
                        table_id is not None
                        and token.table_ids
                        and table_id not in token.table_ids
                    )
                )
                for token in manifest_tokens
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


def _validate_totals(
    result: dict[str, Any],
    issues: list[ValidationIssue],
    assets: dict[int, dict[str, Any]],
    tokens: dict[str, TokenManifestEntry],
) -> None:
    totals_payload = result.get("document_totals")
    if not isinstance(totals_payload, list):
        issues.append(
            _issue(
                "document_totals_contract_invalid",
                ValidationSeverity.FATAL,
                "Document total candidates are not a list",
                field="document_totals",
            )
        )
        totals_payload = []
    totals = tuple(
        payload
        for payload in totals_payload
        if isinstance(payload, dict)
    )
    if len(totals) != len(totals_payload):
        issues.append(
            _issue(
                "document_total_candidate_invalid",
                ValidationSeverity.FATAL,
                "A document total candidate is not an object",
                field="document_totals",
            )
        )
    for index, payload in enumerate(totals):
        try:
            total = DocumentTotal.model_validate(payload)
        except (ValidationError, TypeError, AttributeError) as error:
            issues.append(
                _issue(
                    "document_total_candidate_invalid",
                    ValidationSeverity.FATAL,
                    f"Document total candidate {index} is invalid: "
                    f"{_contract_error_message(error)}",
                    page_number=(payload.get("page_number") if isinstance(payload, dict) else None),
                    field="document_totals",
                )
            )
            continue
        _validate_evidence_refs(
            (total.evidence,),
            assets,
            tokens,
            issues,
            page_number=total.page_number,
            table_id=total.evidence.table_id,
            field="document_totals",
        )
    primary = result.get("document_total")
    if primary is not None and not isinstance(primary, dict):
        issues.append(
            _issue(
                "document_total_contract_invalid",
                ValidationSeverity.FATAL,
                "Primary total is not an object",
                field="document_total",
            )
        )
        primary = None
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
    if len(document_contexts) == 1 and primary is None:
        issues.append(
            _issue(
                "primary_total_missing_for_unique_context",
                ValidationSeverity.BLOCKING,
                "A unique document-final context requires a primary total",
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


def _validate_extraction_result(
    source: Path,
    result: object,
    artifact_root: Path,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not isinstance(result, dict):
        return _report(
            [
                _issue(
                    "extraction_result_contract_invalid",
                    ValidationSeverity.FATAL,
                    "Extraction result is not an object",
                )
            ]
        )
    try:
        envelope = ExtractionResultV5.model_validate(result)
    except ValidationError as error:
        for contract_error in error.errors(include_url=False):
            location = ".".join(str(item) for item in contract_error.get("loc", ()))
            issues.append(
                _issue(
                    "extraction_result_contract_invalid",
                    ValidationSeverity.FATAL,
                    f"Extraction result field {location or '<root>'} is invalid: "
                    f"{contract_error.get('msg', 'contract validation failed')}",
                    field=location or None,
                )
            )
        return _report(issues)
    result = envelope.model_dump(mode="json")
    _validate_provider_and_recovery_metadata(result, issues)
    for target in result["recovery"]["targets"]:
        status = target["status"]
        if status not in {
            "recovery_target_not_located",
            "recovery_no_safe_improvement",
        }:
            continue
        issues.append(
            _issue(
                status,
                ValidationSeverity.BLOCKING,
                (
                    "Targeted recovery could not locate the baseline unit"
                    if status == "recovery_target_not_located"
                    else "Targeted recovery did not safely improve the baseline unit"
                ),
                category=ValidationCategory.GROUNDING,
                recovery_scope=(
                    RecoveryScope.TABLE if target.get("table_id") else RecoveryScope.PAGE
                ),
                page_number=target["page_number"],
                table_id=target.get("table_id"),
                field="recovery",
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
            if asset.get("document_sha256") != source_sha256:
                issues.append(
                    _issue(
                        "page_artifact_document_mismatch",
                        ValidationSeverity.FATAL,
                        "Page artifact does not identify the validated source document",
                        page_number=expected_page,
                        field="page_assets",
                    )
                )
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

    token_manifest: dict[str, TokenManifestEntry] = {}
    token_payloads = result.get("token_manifest")
    if not isinstance(token_payloads, list):
        issues.append(
            _issue(
                "token_manifest_missing",
                ValidationSeverity.FATAL,
                "The supported extraction contract requires a token manifest",
                field="token_manifest",
            )
        )
        token_payloads = []
    for index, payload in enumerate(token_payloads):
        try:
            token = TokenManifestEntry.model_validate(payload)
        except (ValidationError, TypeError, AttributeError) as error:
            issues.append(
                _issue(
                    "token_manifest_entry_invalid",
                    ValidationSeverity.FATAL,
                    f"Token manifest entry {index} is invalid: {type(error).__name__}",
                    page_number=(payload.get("page_number") if isinstance(payload, dict) else None),
                    field="token_manifest",
                )
            )
            continue
        if token.token_id in token_manifest:
            issues.append(
                _issue(
                    "duplicate_token_manifest_id",
                    ValidationSeverity.FATAL,
                    "Token manifest IDs must be globally unique",
                    page_number=token.page_number,
                    field="token_manifest",
                )
            )
            continue
        asset = assets_by_page.get(token.page_number)
        relative = Path(token.artifact_relative_path)
        unresolved = artifact_root / relative
        path = unresolved.resolve()
        points = token.polygon.points
        width = float((asset or {}).get("width") or 0)
        height = float((asset or {}).get("height") or 0)
        polygon_area = abs(
            sum(
                point.x * points[(position + 1) % len(points)].y
                - points[(position + 1) % len(points)].x * point.y
                for position, point in enumerate(points)
            )
            / 2
        )
        if (
            asset is None
            or token.artifact_sha256 != asset.get("artifact_sha256")
            or relative.is_absolute()
            or artifact_root.resolve() not in path.parents
            or not unresolved.is_file()
            or _path_uses_symlink(unresolved, artifact_root)
            or sha256_file(path) != token.artifact_sha256
            or polygon_area <= 0
            or any(
                point.x > width or point.y > height
                for point in points
            )
        ):
            issues.append(
                _issue(
                    "token_manifest_grounding_invalid",
                    ValidationSeverity.FATAL,
                    "Token manifest entry is not grounded in its declared page artifact",
                    page_number=token.page_number,
                    field="token_manifest",
                )
            )
            continue
        if token.source_artifact_sha256 is not None:
            source_relative = Path(token.source_artifact_relative_path or "")
            source_unresolved = artifact_root / source_relative
            source_path = source_unresolved.resolve()
            source_points = token.source_polygon.points if token.source_polygon else ()
            matrix = token.source_to_page_matrix or ()
            transformed = tuple(
                (
                    matrix[0][0] * point.x
                    + matrix[0][1] * point.y
                    + matrix[0][2],
                    matrix[1][0] * point.x
                    + matrix[1][1] * point.y
                    + matrix[1][2],
                )
                for point in source_points
            )
            provenance_valid = bool(
                not source_relative.is_absolute()
                and artifact_root.resolve() in source_path.parents
                and source_unresolved.is_file()
                and not _path_uses_symlink(source_unresolved, artifact_root)
                and sha256_file(source_path) == token.source_artifact_sha256
                and source_points
                and all(
                    point.x <= (token.source_width or 0)
                    and point.y <= (token.source_height or 0)
                    for point in source_points
                )
                and len(transformed) == len(points)
                and all(
                    abs(actual_x - expected.x) <= 1.0
                    and abs(actual_y - expected.y) <= 1.0
                    for (actual_x, actual_y), expected in zip(
                        transformed, points, strict=True
                    )
                )
            )
            if not provenance_valid:
                issues.append(
                    _issue(
                        "recovery_token_provenance_invalid",
                        ValidationSeverity.FATAL,
                        "Recovery token crop identity or page transform is invalid",
                        page_number=token.page_number,
                        field="token_manifest",
                    )
                )
                continue
        token_manifest[token.token_id] = token
    for token in tuple(token_manifest.values()):
        if token.parent_token_id is None:
            continue
        parent = token_manifest.get(token.parent_token_id)
        if (
            parent is None
            or token.page_number != parent.page_number
            or token.artifact_sha256 != parent.artifact_sha256
            or token.character_start is None
            or token.character_end is None
            or token.character_end > len(parent.text)
            or parent.text[token.character_start : token.character_end] != token.text
        ):
            issues.append(
                _issue(
                    "token_fragment_invalid",
                    ValidationSeverity.FATAL,
                    "Typed token fragment does not match its parent OCR token",
                    page_number=token.page_number,
                    field="token_manifest",
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
        except (ValidationError, TypeError, AttributeError) as error:
            issues.append(
                _issue(
                    "canonical_row_contract_invalid",
                    ValidationSeverity.FATAL,
                    "Canonical row "
                    f"{index} failed contract validation: {_contract_error_message(error)}",
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
            token_manifest,
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
                token_manifest,
                issues,
                page_number=row.page_number,
                table_id=row.table_id,
                canonical_row_id=row_id,
                field=evidence_field,
            )
            canonical_field_values = {
                "description": row.description,
                "service_date": row.service_date_raw or row.service_date_iso,
                "request_no": row.request_no,
                "service_code": row.service_code,
                "hsn_code": row.hsn_code,
                "quantity": row.quantity_raw or row.quantity,
                "rate": row.unit_price_raw or row.unit_price,
                "unit_price": row.unit_price_raw or row.unit_price,
                "gross_amount": row.gross_amount_raw or row.gross_amount,
                "discount": row.discount_raw or row.discount,
                "amount": row.net_amount_raw or row.net_amount,
                "net_amount": row.net_amount_raw or row.net_amount,
            }
            value = canonical_field_values.get(evidence_field)
            derived_quantity = (
                evidence_field == "quantity"
                and "quantity_derived_from_rate_amount" in row.validation_flags
            )
            if (
                value is not None
                and not derived_quantity
                and not _token_text_supports_value(
                    evidence_field,
                    value,
                    _evidence_token_texts(evidence, token_manifest),
                    description=row.description,
                )
            ):
                issues.append(
                    _issue(
                        "canonical_evidence_value_mismatch",
                        ValidationSeverity.BLOCKING,
                        "Canonical field value is not supported by its evidence token text",
                        category=ValidationCategory.GROUNDING,
                        page_number=row.page_number,
                        table_id=row.table_id,
                        canonical_row_id=row_id,
                        field=evidence_field,
                    )
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
    if not isinstance(source_payload, list):
        issues.append(
            _issue(
                "source_tables_contract_invalid",
                ValidationSeverity.FATAL,
                "Source tables are not a list",
                field="source_tables",
            )
        )
        source_payload = []
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
            except (ValidationError, TypeError, AttributeError) as error:
                issues.append(
                    _issue(
                        "source_table_contract_invalid",
                        ValidationSeverity.FATAL,
                        "Source table "
                        f"{index} failed grounding validation: {_contract_error_message(error)}",
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
            columns = {column.id: column for column in table.columns}
            for source_row in table.rows:
                for cell in source_row.cells:
                    raw_value = cell.raw_value
                    if raw_value is None or not raw_value.strip():
                        continue
                    canonical_field = columns[cell.column_id].canonical_field
                    if not _token_text_supports_value(
                        canonical_field,
                        raw_value,
                        _evidence_token_texts(cell.evidence, token_manifest),
                        description=(
                            canonical.get(source_row.canonical_row_id).description
                            if source_row.canonical_row_id in canonical
                            else None
                        ),
                    ):
                        issues.append(
                            _issue(
                                "source_cell_evidence_value_mismatch",
                                ValidationSeverity.BLOCKING,
                                "Printed cell value is not supported by its evidence token text",
                                category=ValidationCategory.GROUNDING,
                                page_number=table.page_number,
                                table_id=table.table_id,
                                source_row_id=source_row.id,
                                field=canonical_field or cell.column_id,
                            )
                        )
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

    diagnostic_payloads = result.get("diagnostics")
    parsed_diagnostics: list[dict[str, Any]] = []
    if not isinstance(diagnostic_payloads, list):
        issues.append(
            _issue(
                "diagnostic_inventory_invalid",
                ValidationSeverity.FATAL,
                "Diagnostics are not a list",
                field="diagnostics",
            )
        )
        diagnostic_payloads = []
    diagnostic_ids: set[str] = set()
    for index, payload in enumerate(diagnostic_payloads):
        if not isinstance(payload, dict):
            issues.append(
                _issue(
                    "diagnostic_entry_invalid",
                    ValidationSeverity.FATAL,
                    f"Diagnostic entry {index} is not an object",
                    field="diagnostics",
                )
            )
            continue
        diagnostic_id = payload.get("diagnostic_id")
        kind = payload.get("diagnostic_kind")
        page = payload.get("page_number")
        if (
            not isinstance(diagnostic_id, str)
            or not diagnostic_id.strip()
            or diagnostic_id in diagnostic_ids
            or kind not in {"page", "table"}
            or type(page) is not int
            or page < 1
            or page > source_pages
            or (
                kind == "table"
                and (
                    not str(payload.get("table_id") or "").strip()
                    or not str(payload.get("source_table_id") or "").strip()
                )
            )
        ):
            issues.append(
                _issue(
                    "diagnostic_entry_invalid",
                    ValidationSeverity.FATAL,
                    "Diagnostic identity, kind, or page is invalid",
                    page_number=page if type(page) is int and page >= 1 else None,
                    table_id=(str(payload.get("table_id")) if payload.get("table_id") else None),
                    field="diagnostics",
                )
            )
            continue
        diagnostic_ids.add(diagnostic_id)
        parsed_diagnostics.append(payload)
        classification_evidence = payload.get("financial_form_classification_evidence")
        if classification_evidence:
            valid_ids = bool(
                isinstance(classification_evidence, (list, tuple))
                and classification_evidence
                and all(
                    isinstance(token_id, str)
                    and token_id in token_manifest
                    and token_manifest[token_id].page_number == page
                    for token_id in classification_evidence
                )
            )
            if not valid_ids:
                issues.append(
                    _issue(
                        "diagnostic_grounding_invalid",
                        ValidationSeverity.FATAL,
                        "Diagnostic classification evidence is not in the token manifest",
                        page_number=page,
                        table_id=(
                            str(payload.get("table_id"))
                            if payload.get("table_id")
                            else None
                        ),
                        field="diagnostics",
                    )
                )
        crop_path_value = payload.get("crop_relative_path")
        crop_sha256 = payload.get("crop_sha256")
        if crop_path_value is not None or crop_sha256 is not None:
            relative = Path(str(crop_path_value or ""))
            unresolved = artifact_root / relative
            resolved = unresolved.resolve()
            valid_crop = bool(
                isinstance(crop_path_value, str)
                and isinstance(crop_sha256, str)
                and re.fullmatch(r"[a-f0-9]{64}", crop_sha256)
                and not relative.is_absolute()
                and artifact_root.resolve() in resolved.parents
                and unresolved.is_file()
                and not _path_uses_symlink(unresolved, artifact_root)
                and sha256_file(resolved) == crop_sha256
            )
            if not valid_crop:
                issues.append(
                    _issue(
                        "diagnostic_artifact_invalid",
                        ValidationSeverity.FATAL,
                        "Diagnostic crop artifact path or hash is invalid",
                        page_number=page,
                        table_id=(
                            str(payload.get("table_id"))
                            if payload.get("table_id")
                            else None
                        ),
                        field="diagnostics",
                    )
                )
    page_diagnostic_counts = {
        page: sum(
            item.get("diagnostic_kind") == "page" and item.get("page_number") == page
            for item in parsed_diagnostics
        )
        for page in range(1, source_pages + 1)
    }
    for page, count in page_diagnostic_counts.items():
        if count != 1:
            issues.append(
                _issue(
                    "page_diagnostic_inventory_incomplete",
                    ValidationSeverity.FATAL,
                    "Each PDF page requires exactly one page diagnostic",
                    page_number=page,
                    field="diagnostics",
                )
            )
    expected_table_diagnostics = {
        (table.page_number, table.id): table for table in tables
    }
    actual_table_diagnostic_counts: Counter[tuple[int, str]] = Counter(
        (int(item["page_number"]), str(item.get("source_table_id")))
        for item in parsed_diagnostics
        if item.get("diagnostic_kind") == "table"
    )
    for identity, table in expected_table_diagnostics.items():
        if actual_table_diagnostic_counts[identity] != 1:
            issues.append(
                _issue(
                    "table_diagnostic_inventory_incomplete",
                    ValidationSeverity.FATAL,
                    "Every source table requires exactly one table diagnostic",
                    page_number=table.page_number,
                    table_id=table.table_id,
                    field="diagnostics",
                )
            )
    for identity in sorted(actual_table_diagnostic_counts):
        if identity not in expected_table_diagnostics:
            issues.append(
                _issue(
                    "orphan_table_diagnostic",
                    ValidationSeverity.FATAL,
                    "Table diagnostic references no published source table",
                    page_number=identity[0],
                    table_id=next(
                        (
                            str(item.get("table_id"))
                            for item in parsed_diagnostics
                            if item.get("diagnostic_kind") == "table"
                            and int(item["page_number"]) == identity[0]
                            and str(item.get("source_table_id")) == identity[1]
                        ),
                        None,
                    ),
                    field="diagnostics",
                )
            )
    if not canonical and not tables:
        allowed_empty_classifications = {
            "blank",
            "recognized_nonfinancial",
            "recognized_nonbillable",
            "nonfinancial",
        }
        for page in range(1, source_pages + 1):
            diagnostic = next(
                (
                    item
                    for item in parsed_diagnostics
                    if item.get("diagnostic_kind") == "page"
                    and item.get("page_number") == page
                ),
                None,
            )
            classification = str(
                (diagnostic or {}).get("page_classification")
                or (diagnostic or {}).get("financial_form_classification")
                or ""
            )
            grounded_nonblank = bool(
                (diagnostic or {}).get("financial_form_classification_evidence")
            )
            demonstrably_blank = bool((diagnostic or {}).get("demonstrably_blank"))
            if (
                classification not in allowed_empty_classifications
                or (
                    classification != "blank"
                    and not demonstrably_blank
                    and not grounded_nonblank
                )
            ):
                issues.append(
                    _issue(
                        "empty_extraction_unclassified",
                        ValidationSeverity.BLOCKING,
                        "Empty extraction lacks a grounded blank or nonbillable classification",
                        category=ValidationCategory.FINANCIAL_FORM,
                        recovery_scope=RecoveryScope.PAGE,
                        page_number=page,
                        field="diagnostics",
                    )
                )

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
                token_manifest,
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
                    token_manifest,
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
                    if derived_quantity and (
                        not canonical_ids
                        or canonical_ids.issubset(supporting_ids)
                        or bool(row.field_evidence.get("rate"))
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
                        canonical_field=column.canonical_field,
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
                        canonical_field="service_date_raw",
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

    for diagnostic in parsed_diagnostics:
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

    duplicate_payloads = result.get("receipt_duplicate_pairs") or []
    if not isinstance(duplicate_payloads, list):
        issues.append(
            _issue(
                "receipt_duplicate_pairs_invalid",
                ValidationSeverity.FATAL,
                "Receipt duplicate pairs are not a list",
                field="receipt_duplicate_pairs",
            )
        )
        duplicate_payloads = []
    for pair in duplicate_payloads:
        if not isinstance(pair, dict):
            issues.append(
                _issue(
                    "receipt_duplicate_pair_invalid",
                    ValidationSeverity.FATAL,
                    "Receipt duplicate pair is not an object",
                    field="receipt_duplicate_pairs",
                )
            )
            continue
        canonical_ids = tuple(str(value) for value in pair.get("canonical_row_ids") or ())
        source_ids = tuple(str(value) for value in pair.get("source_row_ids") or ())
        known_source_rows = {
            source_row.id: table
            for table in tables
            for source_row in table.rows
        }
        referenced_tables = {
            known_source_rows[source_id].table_id
            for source_id in source_ids
            if source_id in known_source_rows
        }
        if (
            any(row_id not in canonical for row_id in canonical_ids)
            or any(source_id not in known_source_rows for source_id in source_ids)
            or (
                pair.get("table_id") is not None
                and referenced_tables
                and referenced_tables != {pair["table_id"]}
            )
        ):
            issues.append(
                _issue(
                    "receipt_duplicate_pair_invalid",
                    ValidationSeverity.FATAL,
                    "Receipt duplicate pair references unknown or inconsistent rows",
                    page_number=pair.get("page_number"),
                    table_id=pair.get("table_id"),
                    field="receipt_duplicate_pairs",
                )
            )
            continue
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

    _validate_totals(result, issues, assets_by_page, token_manifest)
    return _report(issues)


def validate_extraction_result(
    source: Path,
    result: object,
    artifact_root: Path,
) -> ValidationReport:
    """Fail closed for every JSON-compatible payload and validator defect."""

    try:
        return _validate_extraction_result(source, result, artifact_root)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:  # the publication gate must never leak parser bugs
        return _report(
            [
                _issue(
                    "validator_internal_error",
                    ValidationSeverity.FATAL,
                    f"Extraction validation failed closed: {type(error).__name__}",
                    field="result",
                )
            ]
        )
