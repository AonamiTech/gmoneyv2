from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import ExitStack, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, TextIO

import typer
from pydantic import ValidationError

from gmoney.contracts.extraction import SourceTable
from gmoney.demo.review import structural_issues
from gmoney.demo.store import JobStore, is_gpu_device, utc_now
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.offline import OfflineExtractor
from gmoney.extraction.typed_values import (
    parse_decimal,
    parse_quantity,
    parse_service_date,
)

app = typer.Typer(add_completion=False, invoke_without_command=True)
_PRINTED_DATE_REQUEST_SUFFIX = re.compile(
    r"\s*[-:]?\s*[A-Z][A-Z0-9-]{2,}/[A-Z0-9-]+"
    r"(?:\s+(?P<bleed>[A-Z]{1,2}))?\s*$",
    re.IGNORECASE,
)
_gpu_inference_locks: dict[Path, TextIO] = {}
_gpu_inference_locks_guard = Lock()
_gpu_inference_locks_pid = os.getpid()


def _reset_gpu_inference_locks_after_fork() -> None:
    global _gpu_inference_locks_guard, _gpu_inference_locks_pid
    for lock in _gpu_inference_locks.values():
        with suppress(OSError):
            lock.close()
    _gpu_inference_locks.clear()
    _gpu_inference_locks_guard = Lock()
    _gpu_inference_locks_pid = os.getpid()


os.register_at_fork(after_in_child=_reset_gpu_inference_locks_after_fork)


@dataclass(frozen=True)
class PreparedJob:
    job_id: str
    stage_dir: Path
    old_result: dict[str, Any]
    new_result: dict[str, Any]
    migrated_review: dict[str, Any]
    review_marker: tuple[bool, str | None]


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    source: Path
    artifact_root: Path
    source_sha256: str
    source_name: str
    old_result: dict[str, Any]
    review: dict[str, Any]
    review_marker: tuple[bool, str | None]


@dataclass(frozen=True)
class RollbackJob:
    job_id: str
    backup_dir: Path
    displaced_dir: Path
    review_marker: tuple[bool, str | None]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _file_digest(path: Path) -> str:
    return sha256_file(path)


def _tree_digest(root: Path) -> str:
    if root.is_symlink():
        raise ValueError(f"staged payload root is a symbolic link: {root}")
    if not root.is_dir():
        raise ValueError(f"staged directory is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"staged payload contains a symbolic link: {relative_path}")
        if path.is_dir():
            digest.update(b"directory\0")
            digest.update(relative_path.encode())
            digest.update(b"\0")
            continue
        if not path.is_file():
            raise ValueError(f"staged payload contains an unsupported path: {relative_path}")
        digest.update(b"file\0")
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_digest(path)))
    return digest.hexdigest()


def _extraction_digest(stage_dir: Path) -> str:
    digest = hashlib.sha256()
    result_path = stage_dir / "result.json"
    artifact_root = stage_dir / "artifacts"
    if not result_path.is_file():
        raise ValueError(f"staged result is missing: {result_path}")
    digest.update(b"result.json\0")
    digest.update(bytes.fromhex(_file_digest(result_path)))
    digest.update(b"artifacts\0")
    digest.update(bytes.fromhex(_tree_digest(artifact_root)))
    return digest.hexdigest()


def _review_marker(job_dir: Path) -> tuple[bool, str | None]:
    path = job_dir / "review.json"
    return (path.is_file(), _file_digest(path) if path.is_file() else None)


def _marker_payload(marker: tuple[bool, str | None]) -> dict[str, Any]:
    return {"exists": marker[0], "sha256": marker[1]}


def _marker_from_payload(payload: dict[str, Any]) -> tuple[bool, str | None]:
    return bool(payload.get("exists")), (
        str(payload["sha256"]) if payload.get("sha256") is not None else None
    )


def _legacy_rollback_review_is_safe(
    *,
    job_id: str,
    backup_review: dict[str, Any],
    current_review: dict[str, Any],
) -> bool:
    backup_events = backup_review.get("events") or []
    current_events = current_review.get("events") or []
    if int(current_review.get("revision") or 0) != int(
        backup_review.get("revision") or 0
    ) + 1:
        return False
    if current_events[:-1] != backup_events or not current_events:
        return False
    deployment_event = current_events[-1]
    return (
        deployment_event.get("action") == "document_reprocessed"
        and deployment_event.get("target_id") == job_id
        and deployment_event.get("reviewer") == "system-reprocessor"
    )


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _printed_service_date_iso(
    raw_value: str,
    canonical_description: object,
) -> str | None:
    parsed = parse_service_date(raw_value)
    if parsed is not None:
        return parsed
    request_suffix = _PRINTED_DATE_REQUEST_SUFFIX.search(raw_value)
    if request_suffix is None:
        return None
    bleed = (request_suffix.group("bleed") or "").casefold()
    if bleed:
        first_word = next(iter(_normalized(canonical_description).split()), "")
        if not (
            first_word.startswith(bleed)
            or first_word.endswith(bleed)
        ):
            return None
    return parse_service_date(raw_value[: request_suffix.start()])


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
    total_labels = {"bill total", "total", "totals", "sub total", "subtotal"}
    total_prefixes = (
        "grand total",
        "gross bill amount",
        "net bill amount",
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

    if normalized_label in {"bill total", "sub total", "subtotal"}:
        section_rows: list[dict[str, Any]] = []
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
                or is_structurally_grounded_settlement(preceding_cells)
            )
            preceding_is_continuation = is_grounded_description_continuation(
                preceding_rows,
                preceding_index,
                previous_is_continuation=previous_was_continuation,
            )
            if preceding_is_financial_boundary or (
                preceding_label
                and preceding_has_section_text
                and not preceding_has_financial_value
                and not preceding_is_continuation
            ):
                section_rows.clear()
            previous_was_continuation = preceding_is_continuation
        if section_rows and all(
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


def _field_token_ids(row: dict[str, Any], field: str | None = None) -> set[str]:
    evidence_by_field = row.get("field_evidence") or {}
    evidence = evidence_by_field.get(field, []) if field else [
        item for values in evidence_by_field.values() for item in values
    ]
    return {
        str(token_id)
        for item in evidence
        for token_id in (item.get("token_ids") or [])
    }


def _review_mapping_score(
    old_row: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[int, int, int, int, float, int, int] | None:
    if int(candidate.get("page_number") or 0) != int(
        old_row.get("page_number") or 0
    ):
        return None
    old_description_ids = _field_token_ids(old_row, "description")
    old_all_ids = _field_token_ids(old_row)
    description_overlap = len(
        old_description_ids & _field_token_ids(candidate, "description")
    )
    all_overlap = len(old_all_ids & _field_token_ids(candidate))
    similarity = SequenceMatcher(
        None,
        _normalized(old_row.get("description")),
        _normalized(candidate.get("description")),
    ).ratio()
    order_distance = abs(
        int(old_row.get("row_order") or 0)
        - int(candidate.get("row_order") or 0)
    )
    if description_overlap or all_overlap:
        return (
            2,
            description_overlap,
            all_overlap,
            0,
            similarity,
            0,
            -order_distance,
        )

    exact_financial_fields = sum(
        1
        for field in ("net_amount", "gross_amount")
        if (old_value := parse_decimal(str(old_row.get(field)))) is not None
        and parse_decimal(str(candidate.get(field))) == old_value
    )
    if not exact_financial_fields:
        return None
    old_quantity = parse_decimal(str(old_row.get("quantity")))
    candidate_quantity = parse_decimal(str(candidate.get("quantity")))
    if (
        old_quantity is not None
        and candidate_quantity is not None
        and old_quantity != candidate_quantity
    ):
        return None
    quantity_match = int(
        old_quantity is not None and candidate_quantity == old_quantity
    )
    if similarity < 0.5 and not quantity_match:
        return None
    return (
        1,
        0,
        0,
        exact_financial_fields,
        similarity,
        quantity_match,
        -order_distance,
    )


def _map_reviewed_row(
    old_row: dict[str, Any], new_rows: list[dict[str, Any]]
) -> str | None:
    scored: list[
        tuple[tuple[int, int, int, int, float, int, int], str]
    ] = []
    for candidate in new_rows:
        score = _review_mapping_score(old_row, candidate)
        if score is None:
            continue
        scored.append((score, str(candidate["id"])))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _rebase_review_override(
    old_row: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    rebased = json.loads(json.dumps(override))
    rebased["changes"] = {
        field: value
        for field, value in (override.get("changes") or {}).items()
        if field not in old_row or old_row[field] != value
    }
    return rebased


def _merge_compatible_review_overrides(
    preferred: dict[str, Any],
    other: dict[str, Any],
    *,
    target_row: dict[str, Any],
) -> dict[str, Any] | None:
    preferred_changes = preferred.get("changes") or {}
    other_changes = other.get("changes") or {}
    if any(
        field in preferred_changes and preferred_changes[field] != value
        for field, value in other_changes.items()
    ):
        return None
    if (
        "review_disposition" in other_changes
        and "review_disposition" not in preferred_changes
        and other_changes["review_disposition"]
        != target_row.get("review_disposition")
    ):
        return None
    merged = json.loads(json.dumps(preferred))
    merged["changes"] = {**other_changes, **preferred_changes}
    return merged


def _migrate_review(
    job_id: str,
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    old_rows = {str(row["id"]): row for row in old_result.get("rows", [])}
    new_rows = list(new_result.get("rows", []))
    new_rows_by_id = {str(row["id"]): row for row in new_rows}
    new_ids = {str(row["id"]) for row in new_rows}
    migrated_overrides: dict[str, Any] = {}
    migrated_override_sources: dict[str, str] = {}
    migrated_override_members: dict[str, list[str]] = {}
    migrated_added = json.loads(json.dumps(review.get("added_rows", {})))
    preserved_unmapped: list[str] = []
    subsumed_overrides: list[str] = []

    def preserve_override(old_id: str, override: dict[str, Any]) -> None:
        preserved = json.loads(json.dumps(old_rows[old_id]))
        preserved.update(override.get("changes", {}))
        preserved_id = f"reprocessed-{old_id}"
        suffix = 1
        while preserved_id in migrated_added or preserved_id in new_ids:
            suffix += 1
            preserved_id = f"reprocessed-{old_id}-{suffix}"
        preserved["id"] = preserved_id
        preserved["contract_version"] = "canonical_row_reviewer_v1"
        preserved["source_routes"] = sorted(
            {
                *preserved.get("source_routes", []),
                "reviewer",
                "reprocess_preserved",
            }
        )
        preserved["validation_flags"] = sorted(
            {
                *preserved.get("validation_flags", []),
                "reviewer_preserved",
            }
        )
        preserved["review_reason"] = override.get("reason") or (
            "Reviewer correction preserved because the upgraded machine row "
            "could not be mapped uniquely"
        )
        migrated_added[preserved_id] = preserved
        preserved_unmapped.append(old_id)

    def mapping_priority(old_id: str, target_id: str) -> tuple[Any, ...]:
        reviewed_row = {
            **old_rows[old_id],
            **(
                review["row_overrides"][old_id].get("changes")
                or {}
            ),
        }
        score = _review_mapping_score(
            reviewed_row,
            new_rows_by_id[target_id],
        )
        return (
            old_id == target_id,
            *(score or (0, 0, 0.0)),
        )

    for old_id, override in review.get("row_overrides", {}).items():
        old_row = old_rows.get(old_id)
        if old_row is None:
            raise ValueError(
                f"reviewed row is absent from the old result: {old_id}"
            )
        target_id = old_id if old_id in new_ids else None
        rebased_override = _rebase_review_override(old_row, override)
        if target_id is None:
            reviewed_row = {
                **old_row,
                **(override.get("changes") or {}),
            }
            target_id = _map_reviewed_row(reviewed_row, new_rows)
        if target_id is None:
            preserve_override(old_id, override)
            continue
        if target_id in migrated_overrides:
            incumbent_id = migrated_override_sources[target_id]
            incoming_wins = mapping_priority(
                old_id,
                target_id,
            ) > mapping_priority(
                incumbent_id,
                target_id,
            )
            preferred = (
                rebased_override
                if incoming_wins
                else migrated_overrides[target_id]
            )
            other = (
                migrated_overrides[target_id]
                if incoming_wins
                else rebased_override
            )
            merged = _merge_compatible_review_overrides(
                preferred,
                other,
                target_row=new_rows_by_id[target_id],
            )
            if merged is not None:
                losing_members = (
                    migrated_override_members[target_id]
                    if incoming_wins
                    else [old_id]
                )
                subsumed_overrides.extend(losing_members)
                migrated_overrides[target_id] = merged
                if incoming_wins:
                    migrated_override_sources[target_id] = old_id
                    migrated_override_members[target_id] = [
                        *migrated_override_members[target_id],
                        old_id,
                    ]
                else:
                    migrated_override_members[target_id].append(old_id)
                continue
            if incoming_wins:
                for losing_id in migrated_override_members[target_id]:
                    preserve_override(
                        losing_id,
                        review["row_overrides"][losing_id],
                    )
            else:
                preserve_override(old_id, override)
                continue
        migrated_overrides[target_id] = rebased_override
        migrated_override_sources[target_id] = old_id
        migrated_override_members[target_id] = [old_id]

    issue_probe = {**review, "issue_overrides": {}}
    new_issue_ids = {
        str(issue["id"]) for issue in structural_issues(new_result, issue_probe)
    }
    old_issue_overrides = review.get("issue_overrides", {})
    missing_issues = set(old_issue_overrides) - new_issue_ids
    active_issue_overrides = {
        issue_id: override
        for issue_id, override in old_issue_overrides.items()
        if issue_id in new_issue_ids
    }

    revision = int(review.get("revision") or 0) + 1
    migrated = json.loads(json.dumps(review))
    migrated["revision"] = revision
    migrated["updated_at"] = utc_now()
    migrated["row_overrides"] = migrated_overrides
    migrated["added_rows"] = migrated_added
    migrated["issue_overrides"] = active_issue_overrides
    archived = migrated.setdefault("archived_issue_overrides", {})
    archived.update(
        {issue_id: old_issue_overrides[issue_id] for issue_id in sorted(missing_issues)}
    )
    migrated["approval"] = None
    migrated.setdefault("events", []).append(
        {
            "revision": revision,
            "action": "document_reprocessed",
            "target_id": job_id,
            "reviewer": "system-reprocessor",
            "reason": "Machine extraction upgraded with preserved review history",
            "changes": {
                "old_rows": len(old_result.get("rows", [])),
                "new_rows": len(new_result.get("rows", [])),
                "mapped_row_overrides": len(migrated_overrides),
                "preserved_unmapped_row_overrides": preserved_unmapped,
                "subsumed_row_overrides": sorted(set(subsumed_overrides)),
                "archived_issue_overrides": sorted(missing_issues),
            },
            "created_at": utc_now(),
        }
    )
    return migrated


def _validate_result(
    source: Path,
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    artifact_root: Path,
) -> None:
    if new_result.get("document_id") != old_result.get("document_id"):
        raise ValueError("document identity changed")
    if new_result.get("source_sha256") != _file_digest(source):
        raise ValueError("source hash changed")
    if int(new_result.get("pages") or 0) != int(old_result.get("pages") or 0):
        raise ValueError("page count changed")
    assets = new_result.get("page_assets", [])
    if len(assets) != int(new_result.get("pages") or 0):
        raise ValueError("page inventory is incomplete")
    page_numbers = [
        asset.get("page_number") if isinstance(asset, dict) else None
        for asset in assets
    ]
    if (
        any(type(page_number) is not int for page_number in page_numbers)
        or page_numbers != list(range(1, int(new_result.get("pages") or 0) + 1))
    ):
        raise ValueError("page inventory has invalid page numbers")
    for asset in assets:
        path = (artifact_root / str(asset["relative_path"])).resolve()
        if artifact_root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"page artifact is missing: {asset['relative_path']}")
        if _file_digest(path) != str(asset["artifact_sha256"]):
            raise ValueError(f"page artifact hash changed: {asset['relative_path']}")
    for row in new_result.get("rows", []):
        fields = row.get("field_evidence") or {}
        if not row.get("description") or "description" not in fields:
            raise ValueError(f"row lacks grounded description: {row.get('id')}")
        if row.get("role") == "informational":
            if row.get("net_amount") is not None:
                raise ValueError(f"informational row has an amount: {row.get('id')}")
            if not {"service_date", "request_no", "service_code", "hsn_code"}.intersection(
                fields
            ):
                raise ValueError(f"informational row lacks typed evidence: {row.get('id')}")
        elif row.get("role") in {"detail", "refund", "category_rollup"}:
            if row.get("net_amount") is None or "amount" not in fields:
                raise ValueError(f"billable row lacks grounded amount: {row.get('id')}")

    source_payload = new_result.get("source_tables")
    if new_result.get("rows") and not source_payload:
        raise ValueError("canonical rows require validated source tables")
    try:
        source_tables = tuple(
            SourceTable.model_validate(table) for table in source_payload or []
        )
    except ValidationError as error:
        raise ValueError(f"source tables failed grounding validation: {error}") from error

    canonical_rows = {
        str(row["id"]): row for row in new_result.get("rows", [])
    }
    linked_ids: set[str] = set()
    evidence_fields = {
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
    numeric_fields = {
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }
    for table in source_tables:
        for source_row in table.rows:
            cells = {cell.column_id: cell for cell in source_row.cells}
            if source_row.canonical_row_id is None:
                financial_values = tuple(
                    (str(column.canonical_field), parsed)
                    for column in table.columns
                    if column.canonical_field in {"net_amount", "gross_amount"}
                    and (raw_value := cells[column.id].raw_value)
                    and raw_value.strip()
                    and (parsed := parse_decimal(raw_value)) is not None
                )
                if financial_values and not _unlinked_financial_row_is_explained(
                    table=table,
                    source_tables=source_tables,
                    source_row=source_row,
                    cells=cells,
                    financial_values=financial_values,
                    canonical_rows=canonical_rows,
                    result=new_result,
                ):
                    raise ValueError(
                        "unlinked source row contains a mapped financial value: "
                        f"{source_row.id}"
                    )
                continue
            canonical = canonical_rows.get(source_row.canonical_row_id)
            if canonical is None:
                raise ValueError(
                    f"source row links unknown canonical row: {source_row.canonical_row_id}"
                )
            if source_row.canonical_row_id in linked_ids:
                raise ValueError(
                    f"canonical row has multiple source links: {source_row.canonical_row_id}"
                )
            linked_ids.add(source_row.canonical_row_id)
            for column in table.columns:
                field = column.canonical_field
                if field is None:
                    continue
                cell = cells[column.id]
                canonical_value = canonical.get(field)
                printed_present = bool(cell.raw_value and cell.raw_value.strip())
                canonical_present = canonical_value is not None and (
                    not isinstance(canonical_value, str) or bool(canonical_value.strip())
                )
                evidence_field = evidence_fields[field]
                field_token_ids = {
                    str(token_id)
                    for item in (canonical.get("field_evidence") or {}).get(
                        evidence_field, []
                    )
                    for token_id in item.get("token_ids") or []
                }
                if printed_present and not canonical_present:
                    raise ValueError(
                        f"{field} has a printed value but is missing canonical value "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if canonical_present and not printed_present:
                    serial_description_is_grounded = bool(
                        field == "description"
                        and "missing_printed_description"
                        in (canonical.get("validation_flags") or [])
                        and re.fullmatch(r"\d+[.)]?", str(canonical_value).strip())
                        and field_token_ids
                        and any(
                            column_candidate.canonical_field is None
                            and _normalized(column_candidate.label)
                            in {"#", "s no", "serial no", "sr n", "sr no"}
                            and (serial_cell := cells[column_candidate.id]).raw_value
                            and serial_cell.raw_value.strip()
                            == str(canonical_value).strip()
                            and field_token_ids.issubset(
                                {
                                    token_id
                                    for item in serial_cell.evidence
                                    for token_id in item.token_ids
                                }
                            )
                            for column_candidate in table.columns
                        )
                    )
                    if serial_description_is_grounded:
                        continue
                    raise ValueError(
                        f"{field} has a canonical value but is missing printed value "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if not printed_present:
                    continue
                cell_token_ids = {
                    token_id
                    for item in cell.evidence
                    for token_id in item.token_ids
                }
                evidence_matches = field_token_ids.issubset(cell_token_ids)
                if not field_token_ids or not evidence_matches:
                    raise ValueError(
                        f"{field} evidence is not in its mapped source cell "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if field in numeric_fields:
                    printed_value = (
                        parse_quantity(cell.raw_value or "")
                        if field == "quantity"
                        else parse_decimal(cell.raw_value or "")
                    )
                    canonical_value = parse_decimal(str(canonical[field]))
                    if printed_value is None or printed_value != canonical_value:
                        raise ValueError(
                            f"{field} value does not match its mapped source cell "
                            f"for canonical row {source_row.canonical_row_id}"
                        )
                elif field != "description":
                    value_matches = _normalized(cell.raw_value) == _normalized(
                        canonical_value
                    )
                    if field == "service_date_raw":
                        printed_date_iso = _printed_service_date_iso(
                            cell.raw_value or "",
                            canonical.get("description"),
                        )
                        value_matches = value_matches or bool(
                            canonical.get("service_date_iso")
                            and printed_date_iso == canonical.get("service_date_iso")
                        )
                    if not value_matches:
                        raise ValueError(
                            f"{field} value does not match its mapped source cell "
                            f"for canonical row {source_row.canonical_row_id}"
                        )
    for row_id, row_payload in canonical_rows.items():
        if (
            "ocr_spatial_graph" in (row_payload.get("source_routes") or [])
            and row_id not in linked_ids
        ):
            raise ValueError(f"canonical OCR row lacks a source-table link: {row_id}")


def _snapshot_job(
    *,
    store: JobStore,
    job_id: str,
) -> JobSnapshot:
    job_dir = store.job_dir(job_id)
    source = job_dir / "source.pdf"
    result_path = job_dir / "result.json"
    artifact_root = job_dir / "artifacts"
    with store.job_lock(job_id, exclusive=False):
        store._require_stable_workspace(job_id)
        state = store.read(job_id)
        if state.get("status") != "complete":
            raise ValueError("only complete jobs can be reprocessed")
        if not source.is_file() or not result_path.is_file() or not artifact_root.is_dir():
            raise ValueError("source, result, or artifacts are missing")
        old_result = json.loads(result_path.read_text())
        review = store._read_review_unlocked(job_id)
        marker = _review_marker(job_dir)
        source_sha256 = _file_digest(source)
    return JobSnapshot(
        job_id=job_id,
        source=source,
        artifact_root=artifact_root,
        source_sha256=source_sha256,
        source_name=str(
            old_result.get("source_name")
            or state.get("original_name")
            or source.name
        ),
        old_result=old_result,
        review=review,
        review_marker=marker,
    )


def _hardlink_or_copy2(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _prepare_source_group(
    *,
    store: JobStore,
    snapshots: list[JobSnapshot],
    stage_root: Path,
    extractor: Any,
) -> list[PreparedJob]:
    ordered = sorted(snapshots, key=lambda item: item.job_id)
    representative = ordered[0]
    representative_stage = stage_root / representative.job_id
    if representative_stage.exists():
        raise ValueError(
            f"staging directory already exists: {representative_stage}"
        )
    representative_stage.mkdir(parents=True, mode=0o700)
    with store.job_lock(representative.job_id, exclusive=False):
        store._require_stable_workspace(representative.job_id)
        if (
            _review_marker(store.job_dir(representative.job_id))
            != representative.review_marker
        ):
            raise ValueError("a review changed while staging; no results were applied")
        shutil.copytree(
            representative.artifact_root,
            representative_stage / "artifacts",
        )
    extracted_result = extractor.extract(
        representative.source,
        representative_stage / "artifacts",
    )

    prepared: list[PreparedJob] = []
    for snapshot in ordered:
        stage_dir = stage_root / snapshot.job_id
        if snapshot.job_id != representative.job_id:
            if stage_dir.exists():
                raise ValueError(f"staging directory already exists: {stage_dir}")
            stage_dir.mkdir(parents=True, mode=0o700)
            shutil.copytree(
                representative_stage / "artifacts",
                stage_dir / "artifacts",
                copy_function=_hardlink_or_copy2,
            )
        new_result = json.loads(json.dumps(extracted_result))
        new_result["source_name"] = snapshot.source_name
        _validate_result(
            snapshot.source,
            snapshot.old_result,
            new_result,
            stage_dir / "artifacts",
        )
        migrated_review = _migrate_review(
            snapshot.job_id,
            snapshot.old_result,
            new_result,
            snapshot.review,
        )
        _atomic_json(stage_dir / "result.json", new_result)
        _atomic_json(stage_dir / "review.json", migrated_review)
        prepared.append(
            PreparedJob(
                job_id=snapshot.job_id,
                stage_dir=stage_dir,
                old_result=snapshot.old_result,
                new_result=new_result,
                migrated_review=migrated_review,
                review_marker=snapshot.review_marker,
            )
        )
    return prepared


def _cutover_job(
    *,
    store: JobStore,
    prepared: PreparedJob,
    backup_root: Path,
    commit_marker: Path,
) -> Path:
    job_dir = store.job_dir(prepared.job_id)
    backup_dir = backup_root / prepared.job_id
    journal_path = job_dir / ".cutover.json"

    def restore_path(name: str) -> None:
        live = job_dir / name
        backup = backup_dir / name
        staged = prepared.stage_dir / name
        if not backup.exists():
            return
        if live.exists():
            if not staged.exists():
                live.replace(staged)
            elif live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
        backup.replace(live)

    if backup_dir.exists():
        raise ValueError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, mode=0o700)
    _atomic_json(
        journal_path,
        {
            "version": "job_cutover_v1",
            "job_id": prepared.job_id,
            "stage_dir": str(prepared.stage_dir),
            "backup_dir": str(backup_dir),
            "commit_marker": str(commit_marker),
            "started_at": utc_now(),
        },
    )
    try:
        shutil.copy2(job_dir / "state.json", backup_dir / "state.json")
        if (job_dir / "review.json").is_file():
            shutil.copy2(job_dir / "review.json", backup_dir / "review.json")

        (job_dir / "artifacts").replace(backup_dir / "artifacts")
        (prepared.stage_dir / "artifacts").replace(job_dir / "artifacts")
        (job_dir / "result.json").replace(backup_dir / "result.json")
        (prepared.stage_dir / "result.json").replace(job_dir / "result.json")
        _atomic_json(job_dir / "review.json", prepared.migrated_review)
        hospital = prepared.new_result.get("hospital") or {}
        store.update(
            prepared.job_id,
            status="complete",
            page=int(prepared.new_result.get("pages") or 0),
            pages=int(prepared.new_result.get("pages") or 0),
            row_count=len(prepared.new_result.get("rows", [])),
            hospital_name=hospital.get("name"),
            hospital_confidence=hospital.get("confidence"),
            error=None,
            reprocessed_at=utc_now(),
        )
    except BaseException:
        try:
            restore_path("result.json")
            restore_path("artifacts")
            _atomic_json(
                job_dir / "state.json",
                json.loads((backup_dir / "state.json").read_text()),
            )
            if (backup_dir / "review.json").is_file():
                _atomic_json(
                    job_dir / "review.json",
                    json.loads((backup_dir / "review.json").read_text()),
                )
            elif (job_dir / "review.json").is_file():
                (job_dir / "review.json").unlink()
            journal_path.unlink(missing_ok=True)
            shutil.rmtree(backup_dir, ignore_errors=True)
        except BaseException as recovery_error:
            raise RuntimeError(
                f"cutover recovery required for job {prepared.job_id}"
            ) from recovery_error
        raise
    return backup_dir


def _rollback_cutover_job(
    *,
    store: JobStore,
    prepared: PreparedJob,
    backup_dir: Path,
) -> None:
    job_dir = store.job_dir(prepared.job_id)

    def restore_path(name: str) -> None:
        live = job_dir / name
        backup = backup_dir / name
        staged = prepared.stage_dir / name
        if live.exists():
            if not staged.exists():
                live.replace(staged)
            elif live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
        backup.replace(live)

    restore_path("result.json")
    restore_path("artifacts")
    _atomic_json(
        job_dir / "state.json",
        json.loads((backup_dir / "state.json").read_text()),
    )
    if (backup_dir / "review.json").is_file():
        _atomic_json(
            job_dir / "review.json",
            json.loads((backup_dir / "review.json").read_text()),
        )
    elif (job_dir / "review.json").is_file():
        (job_dir / "review.json").unlink()
    (job_dir / ".cutover.json").unlink(missing_ok=True)
    shutil.rmtree(backup_dir, ignore_errors=True)


def reprocess_jobs(
    *,
    root: Path,
    job_ids: list[str] | None = None,
    apply: bool = False,
    stage_root: Path | None = None,
    backup_root: Path | None = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    extractor: Any | None = None,
) -> dict[str, Any]:
    store = JobStore(root)
    selected = job_ids or [
        str(state["id"]) for state in store.states() if state.get("status") == "complete"
    ]
    selected = sorted(set(selected))
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "selected": len(selected),
        "would_reprocess": 0,
        "reprocessed": 0,
        "documents": [],
    }
    for job_id in selected:
        state = store.read(job_id)
        if state.get("status") != "complete":
            raise ValueError(f"job is not complete: {job_id}")
        summary["documents"].append(
            {
                "job_id": job_id,
                "source_name": state.get("original_name"),
                "rows_before": state.get("row_count"),
            }
        )
    if not apply:
        summary["would_reprocess"] = len(selected)
        return summary
    raise ValueError(
        "direct apply is disabled; use the two-phase visual audit workflow: "
        "stage_reprocess_jobs(), record visual-audit.json, then "
        "apply_staged_jobs()"
    )


def stage_reprocess_jobs(
    *,
    root: Path,
    job_ids: list[str] | None = None,
    stage_root: Path | None = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    extractor: Any | None = None,
) -> dict[str, Any]:
    """Build and validate replacement results without mutating live jobs."""
    store = JobStore(root)
    if extractor is None and is_gpu_device(paddle_device):
        if _gpu_inference_locks_pid != os.getpid():
            _reset_gpu_inference_locks_after_fork()
        lock_path = store.inference_lock_path.resolve()
        with _gpu_inference_locks_guard:
            retained_lock = _gpu_inference_locks.get(lock_path)
            if retained_lock is None or retained_lock.closed:
                _gpu_inference_locks[lock_path] = store.acquire_inference_lock()
        return _stage_reprocess_jobs(
            store=store,
            job_ids=job_ids,
            stage_root=stage_root,
            extractor=OfflineExtractor(
                vl_url,
                paddle_device=paddle_device,
                vl_device=vl_device,
            ),
        )
    return _stage_reprocess_jobs(
        store=store,
        job_ids=job_ids,
        stage_root=stage_root,
        extractor=(
            extractor
            if extractor is not None
            else OfflineExtractor(
                vl_url,
                paddle_device=paddle_device,
                vl_device=vl_device,
            )
        ),
    )


def _stage_reprocess_jobs(
    *,
    store: JobStore,
    job_ids: list[str] | None,
    stage_root: Path | None,
    extractor: Any,
) -> dict[str, Any]:
    selected = sorted(
        set(
            job_ids
            or [
                str(state["id"])
                for state in store.states()
                if state.get("status") == "complete"
            ]
        )
    )
    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    staging = (stage_root or store.jobs_root / ".reprocess-staging") / timestamp
    staging.mkdir(parents=True, mode=0o700)
    snapshots = [
        _snapshot_job(
            store=store,
            job_id=job_id,
        )
        for job_id in selected
    ]
    snapshots_by_source: dict[str, list[JobSnapshot]] = {}
    for snapshot in snapshots:
        snapshots_by_source.setdefault(snapshot.source_sha256, []).append(snapshot)
    prepared_by_id: dict[str, PreparedJob] = {}
    for source_sha256 in sorted(snapshots_by_source):
        for item in _prepare_source_group(
            store=store,
            snapshots=snapshots_by_source[source_sha256],
            stage_root=staging,
            extractor=extractor,
        ):
            prepared_by_id[item.job_id] = item
    prepared = [prepared_by_id[job_id] for job_id in selected]
    if any(
        _review_marker(store.job_dir(item.job_id)) != item.review_marker
        for item in prepared
    ):
        raise ValueError("a review changed while staging; no results were applied")
    documents = [
        {
            "job_id": item.job_id,
            "source_name": item.old_result.get("source_name"),
            "rows_before": len(item.old_result.get("rows", [])),
            "rows_after": len(item.new_result.get("rows", [])),
            "review_marker": _marker_payload(item.review_marker),
            "source_sha256": item.new_result.get("source_sha256"),
            "staged_payload_sha256": _tree_digest(item.stage_dir),
        }
        for item in prepared
    ]
    sources = []
    for source_sha256 in sorted(snapshots_by_source):
        group = sorted(
            snapshots_by_source[source_sha256],
            key=lambda item: item.job_id,
        )
        representative = prepared_by_id[group[0].job_id]
        sources.append(
            {
                "source_sha256": source_sha256,
                "representative_job_id": group[0].job_id,
                "job_ids": [item.job_id for item in group],
                "page_count": int(representative.new_result.get("pages") or 0),
                "source_names": sorted({item.source_name for item in group}),
                "extraction_sha256": _extraction_digest(
                    representative.stage_dir
                ),
            }
        )
    sealed_at = utc_now()
    _atomic_json(
        staging / "manifest.json",
        {
            "version": "reprocess_stage_v2",
            "created_at": sealed_at,
            "sealed_at": sealed_at,
            "documents": documents,
            "sources": sources,
        },
    )
    return {
        "mode": "stage",
        "selected": len(selected),
        "staged": len(prepared),
        "staging_root": str(staging),
        "documents": documents,
    }


VISUAL_AUDIT_CHECKS = {
    "hospital_identity",
    "printed_columns",
    "row_order_and_count",
    "cell_values",
    "explicit_totals",
    "non_ledger_exclusion",
}


def _explicit_utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp is not UTC")
    return parsed


def _stage_source_inventory(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], datetime]:
    if manifest.get("version") != "reprocess_stage_v2":
        raise ValueError("unsupported or incomplete reprocess stage")
    documents = manifest.get("documents")
    sources = manifest.get("sources")
    if not isinstance(documents, list) or not isinstance(sources, list):
        raise ValueError("unsupported or incomplete reprocess stage")
    try:
        sealed_at = _explicit_utc_timestamp(manifest.get("sealed_at"))
    except ValueError as error:
        raise ValueError("unsupported or incomplete reprocess stage") from error

    document_ids = [
        str(document.get("job_id"))
        for document in documents
        if isinstance(document, dict) and document.get("job_id") is not None
    ]
    if len(document_ids) != len(documents) or len(set(document_ids)) != len(
        document_ids
    ):
        raise ValueError("unsupported or incomplete reprocess stage")
    for document in documents:
        if (
            not isinstance(document.get("source_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", document["source_sha256"]) is None
            or not isinstance(document.get("staged_payload_sha256"), str)
            or re.fullmatch(
                r"[a-f0-9]{64}", document["staged_payload_sha256"]
            )
            is None
        ):
            raise ValueError("unsupported or incomplete reprocess stage")

    source_groups: dict[str, dict[str, Any]] = {}
    job_sources: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("unsupported or incomplete reprocess stage")
        source_sha256 = source.get("source_sha256")
        job_ids = source.get("job_ids")
        representative_job_id = source.get("representative_job_id")
        page_count = source.get("page_count")
        source_names = source.get("source_names")
        extraction_sha256 = source.get("extraction_sha256")
        if (
            not isinstance(source_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None
            or source_sha256 in source_groups
            or not isinstance(job_ids, list)
            or not job_ids
            or any(not isinstance(job_id, str) or not job_id for job_id in job_ids)
            or len(set(job_ids)) != len(job_ids)
            or representative_job_id != sorted(job_ids)[0]
            or not isinstance(page_count, int)
            or isinstance(page_count, bool)
            or page_count < 1
            or not isinstance(source_names, list)
            or not source_names
            or any(not isinstance(name, str) or not name.strip() for name in source_names)
            or not isinstance(extraction_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", extraction_sha256) is None
        ):
            raise ValueError("unsupported or incomplete reprocess stage")
        source_groups[source_sha256] = source
        for job_id in job_ids:
            if job_id in job_sources:
                raise ValueError("unsupported or incomplete reprocess stage")
            job_sources[job_id] = source_sha256
    if sorted(document_ids) != sorted(job_sources):
        raise ValueError("unsupported or incomplete reprocess stage")
    if any(
        document["source_sha256"] != job_sources[str(document["job_id"])]
        for document in documents
    ):
        raise ValueError("unsupported or incomplete reprocess stage")
    return source_groups, job_sources, sealed_at


def _validate_visual_audit(
    *,
    stage_batch: Path,
    source_groups: dict[str, dict[str, Any]],
    sealed_at: datetime,
) -> str:
    audit_path = stage_batch / "visual-audit.json"
    if not audit_path.is_file():
        raise ValueError("visual audit is missing")
    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("visual audit is unreadable") from error
    if (
        not isinstance(audit, dict)
        or audit.get("version") != "reprocess_visual_audit_v1"
        or not isinstance(audit.get("sources"), list)
    ):
        raise ValueError("unsupported or incomplete visual audit")

    audited_sources = audit["sources"]
    audited_sha256s: list[str] = []
    for source in audited_sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("source_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", source["source_sha256"]) is None
        ):
            raise ValueError("visual audit source inventory does not match stage")
        audited_sha256s.append(source["source_sha256"])
    if (
        len(audited_sha256s) != len(source_groups)
        or len(set(audited_sha256s)) != len(audited_sha256s)
        or set(audited_sha256s) != set(source_groups)
    ):
        raise ValueError("visual audit source inventory does not match stage")

    for source in audited_sources:
        source_sha256 = str(source["source_sha256"])
        reviewer = source.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"visual audit reviewer is missing for {source_sha256}")
        reviewed_at = source.get("reviewed_at")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise ValueError(f"visual audit reviewed_at is missing for {source_sha256}")
        try:
            reviewed_timestamp = _explicit_utc_timestamp(reviewed_at)
        except ValueError as error:
            raise ValueError(
                f"visual audit reviewed_at is invalid for {source_sha256}"
            ) from error
        if reviewed_timestamp < sealed_at:
            raise ValueError(
                f"visual audit reviewed_at predates the sealed stage for {source_sha256}"
            )

        pages = source.get("pages")
        page_count = int(source_groups[source_sha256]["page_count"])
        page_numbers = (
            [
                page.get("page_number") if isinstance(page, dict) else None
                for page in pages
            ]
            if isinstance(pages, list)
            else []
        )
        if (
            not isinstance(pages, list)
            or any(type(page_number) is not int for page_number in page_numbers)
            or page_numbers != list(range(1, page_count + 1))
        ):
            raise ValueError(
                f"visual audit page inventory is incomplete for {source_sha256}"
            )
        for page in pages:
            page_number = int(page["page_number"])
            if page.get("status") != "pass":
                raise ValueError(
                    f"visual audit page {page_number} did not pass for {source_sha256}"
                )
            if not isinstance(page.get("notes"), str):
                raise ValueError(
                    f"visual audit page {page_number} notes are invalid "
                    f"for {source_sha256}"
                )

        checks = source.get("checks")
        if not isinstance(checks, dict) or set(checks) != VISUAL_AUDIT_CHECKS:
            raise ValueError(
                f"visual audit check inventory is incomplete for {source_sha256}"
            )
        for check in sorted(VISUAL_AUDIT_CHECKS):
            if checks[check] != "pass":
                raise ValueError(
                    f"visual audit check {check} did not pass for {source_sha256}"
                )
    return hashlib.sha256(audit_bytes).hexdigest()


def _validate_staged_review_migration(
    *,
    prepared: PreparedJob,
    current_review: dict[str, Any],
) -> PreparedJob:
    expected = _migrate_review(
        prepared.job_id,
        prepared.old_result,
        prepared.new_result,
        current_review,
    )
    staged = prepared.migrated_review
    try:
        staged_updated_at = staged["updated_at"]
        staged_event_created_at = staged["events"][-1]["created_at"]
        _explicit_utc_timestamp(staged_updated_at)
        _explicit_utc_timestamp(staged_event_created_at)
        expected["updated_at"] = staged_updated_at
        expected["events"][-1]["created_at"] = staged_event_created_at
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"staged review migration is invalid for {prepared.job_id}"
        ) from error
    if expected != staged:
        raise ValueError(
            f"staged review migration does not match live review for {prepared.job_id}"
        )
    return replace(prepared, migrated_review=expected)


def _require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _file_digest(path) != expected:
        raise ValueError(f"{label} changed after visual audit")


def _require_staged_payload(stage_dir: Path, expected: str) -> None:
    if _tree_digest(stage_dir) != expected:
        raise ValueError(f"staged payload digest changed for {stage_dir.name}")


def _require_direct_child(path: Path, parent: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"staged payload root is a symbolic link: {path}")
    if not path.is_dir() or path.resolve().parent != parent.resolve():
        raise ValueError(f"staged payload is outside staging: {path}")


def _locate_staged_payload(
    *,
    stage_batch: Path,
    claimed_root: Path,
    job_id: str,
) -> Path:
    original = stage_batch / job_id
    claimed = claimed_root / job_id
    original_present = original.exists() or original.is_symlink()
    claimed_present = claimed.exists() or claimed.is_symlink()
    if original_present == claimed_present:
        raise ValueError(f"staged payload inventory is ambiguous for {job_id}")
    path = original if original_present else claimed
    parent = stage_batch if original_present else claimed_root
    _require_direct_child(path, parent)
    return path


def _claim_staged_payloads(
    *,
    prepared: list[PreparedJob],
    stage_batch: Path,
    claimed_root: Path,
) -> list[PreparedJob]:
    if claimed_root.is_symlink():
        raise ValueError("apply-owned staging root is a symbolic link")
    claimed_root.mkdir(mode=0o700, exist_ok=True)
    _require_direct_child(claimed_root, stage_batch)
    claimed_items: list[PreparedJob] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for item in prepared:
            target = claimed_root / item.job_id
            if item.stage_dir.parent == claimed_root:
                _require_direct_child(item.stage_dir, claimed_root)
                claimed_items.append(item)
                continue
            _require_direct_child(item.stage_dir, stage_batch)
            if target.exists() or target.is_symlink():
                raise ValueError(
                    f"apply-owned staged payload already exists for {item.job_id}"
                )
            item.stage_dir.replace(target)
            moved.append((target, item.stage_dir))
            claimed_items.append(replace(item, stage_dir=target))
    except BaseException:
        for claimed, original in reversed(moved):
            if claimed.exists() and not original.exists():
                claimed.replace(original)
        if claimed_root.is_dir() and not any(claimed_root.iterdir()):
            claimed_root.rmdir()
        raise
    return claimed_items


def _release_claimed_payload_paths(
    *,
    job_ids: list[str],
    stage_batch: Path,
    claimed_root: Path,
) -> None:
    if not _claim_namespace_is_safe(
        job_ids=job_ids,
        stage_batch=stage_batch,
        claimed_root=claimed_root,
    ):
        raise ValueError("apply-owned staging namespace is unsafe")
    for job_id in job_ids:
        claimed = claimed_root / job_id
        if not claimed.exists():
            continue
        original = stage_batch / job_id
        if original.exists() or original.is_symlink():
            raise RuntimeError(
                f"cannot release apply-owned staged payload for {job_id}"
            )
        claimed.replace(original)
    if claimed_root.is_dir() and not any(claimed_root.iterdir()):
        claimed_root.rmdir()


def _claim_namespace_is_safe(
    *,
    job_ids: list[str],
    stage_batch: Path,
    claimed_root: Path,
) -> bool:
    if claimed_root.is_symlink():
        return False
    if not claimed_root.exists():
        return True
    if (
        not claimed_root.is_dir()
        or claimed_root.resolve().parent != stage_batch.resolve()
    ):
        return False
    for job_id in job_ids:
        claimed = claimed_root / job_id
        if not claimed.exists() and not claimed.is_symlink():
            continue
        if (
            claimed.is_symlink()
            or not claimed.is_dir()
            or claimed.resolve().parent != claimed_root.resolve()
        ):
            return False
    return True


def _release_claims_if_jobs_are_stable(
    *,
    store: JobStore,
    job_ids: list[str],
    stage_batch: Path,
    claimed_root: Path,
) -> bool:
    if not _claim_namespace_is_safe(
        job_ids=job_ids,
        stage_batch=stage_batch,
        claimed_root=claimed_root,
    ):
        return False
    if any((store.job_dir(job_id) / ".cutover.json").is_file() for job_id in job_ids):
        return False
    _release_claimed_payload_paths(
        job_ids=job_ids,
        stage_batch=stage_batch,
        claimed_root=claimed_root,
    )
    return True


def apply_staged_jobs(
    *,
    root: Path,
    stage_batch: Path,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically cut over a validated batch after rechecking review markers."""
    store = JobStore(root)
    if stage_batch.is_symlink() or not stage_batch.is_dir():
        raise ValueError("staging batch is missing or is a symbolic link")
    manifest_path = stage_batch / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    claimed_root = stage_batch / f".apply-{manifest_digest}"
    source_groups, job_sources, sealed_at = _stage_source_inventory(manifest)
    stage_job_ids = [str(document["job_id"]) for document in manifest["documents"]]
    audit_path = stage_batch / "visual-audit.json"
    expected_payloads = {
        str(document["job_id"]): str(document["staged_payload_sha256"])
        for document in manifest["documents"]
    }
    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    backups = (backup_root or store.jobs_root / ".reprocess-backups") / timestamp
    commit_marker = backups / ".committed.json"
    documents: list[dict[str, Any]] = []
    completed: list[tuple[PreparedJob, Path]] = []
    prepared: list[PreparedJob] = []
    with ExitStack() as locks:
        for job_id in sorted(stage_job_ids):
            locks.enter_context(store.job_lock(job_id, exclusive=True))
        try:
            for job_id in stage_job_ids:
                store._require_stable_workspace(job_id)
            if not _claim_namespace_is_safe(
                job_ids=stage_job_ids,
                stage_batch=stage_batch,
                claimed_root=claimed_root,
            ):
                raise ValueError("apply-owned staging namespace is unsafe")
            audit_digest = _validate_visual_audit(
                stage_batch=stage_batch,
                source_groups=source_groups,
                sealed_at=sealed_at,
            )
            _require_digest(manifest_path, manifest_digest, "stage manifest")
            _require_digest(audit_path, audit_digest, "visual audit")
            stage_dirs: dict[str, Path] = {}
            for document in manifest["documents"]:
                job_id = str(document["job_id"])
                job_dir = store.job_dir(job_id)
                stage_dir = _locate_staged_payload(
                    stage_batch=stage_batch,
                    claimed_root=claimed_root,
                    job_id=job_id,
                )
                stage_dirs[job_id] = stage_dir
                _require_staged_payload(
                    stage_dir,
                    expected_payloads[job_id],
                )
                marker = _marker_from_payload(document["review_marker"])
                if _review_marker(job_dir) != marker:
                    raise ValueError(
                        "review changed after staging; no results were applied"
                    )
                old_result = json.loads((job_dir / "result.json").read_text())
                new_result = json.loads((stage_dir / "result.json").read_text())
                migrated_review = json.loads(
                    (stage_dir / "review.json").read_text()
                )
                source_sha256 = _file_digest(job_dir / "source.pdf")
                if (
                    source_sha256 != job_sources[job_id]
                    or document.get("source_sha256") != source_sha256
                ):
                    raise ValueError(
                        "staged source group does not match live source"
                    )
                page_count = int(source_groups[source_sha256]["page_count"])
                if int(new_result.get("pages") or 0) != page_count:
                    raise ValueError(
                        "staged page count does not match source group "
                        f"for {job_id}"
                    )
                _validate_result(
                    job_dir / "source.pdf",
                    old_result,
                    new_result,
                    stage_dir / "artifacts",
                )
                prepared.append(
                    PreparedJob(
                        job_id=job_id,
                        stage_dir=stage_dir,
                        old_result=old_result,
                        new_result=new_result,
                        migrated_review=migrated_review,
                        review_marker=marker,
                    )
                )
            for source_sha256, source_group in source_groups.items():
                representative_stage = stage_dirs[
                    str(source_group["representative_job_id"])
                ]
                if (
                    _extraction_digest(representative_stage)
                    != source_group["extraction_sha256"]
                ):
                    raise ValueError(
                        f"staged extraction digest changed for {source_sha256}"
                    )
            prepared = _claim_staged_payloads(
                prepared=prepared,
                stage_batch=stage_batch,
                claimed_root=claimed_root,
            )
            _require_digest(manifest_path, manifest_digest, "stage manifest")
            _require_digest(audit_path, audit_digest, "visual audit")
            verified_prepared: list[PreparedJob] = []
            for item in prepared:
                _require_staged_payload(
                    item.stage_dir,
                    expected_payloads[item.job_id],
                )
                _validate_result(
                    store.job_dir(item.job_id) / "source.pdf",
                    item.old_result,
                    item.new_result,
                    item.stage_dir / "artifacts",
                )
                verified_prepared.append(
                    _validate_staged_review_migration(
                        prepared=item,
                        current_review=store._read_review_unlocked(item.job_id),
                    )
                )
            prepared = verified_prepared
            backups.mkdir(parents=True, mode=0o700)
            for item in prepared:
                _require_digest(manifest_path, manifest_digest, "stage manifest")
                _require_digest(audit_path, audit_digest, "visual audit")
                _require_staged_payload(
                    item.stage_dir,
                    expected_payloads[item.job_id],
                )
                backup = _cutover_job(
                    store=store,
                    prepared=item,
                    backup_root=backups,
                    commit_marker=commit_marker,
                )
                completed.append((item, backup))
                documents.append(
                    {
                        "job_id": item.job_id,
                        "source_name": item.old_result.get("source_name"),
                        "rows_before": len(item.old_result.get("rows", [])),
                        "rows_after": len(item.new_result.get("rows", [])),
                        "backup": str(backup),
                        "deployed_review_marker": _marker_payload(
                            _review_marker(store.job_dir(item.job_id))
                        ),
                    }
                )
            _atomic_json(
                commit_marker,
                {
                    "version": "reprocess_batch_commit_v2",
                    "committed_at": utc_now(),
                    "job_ids": [item.job_id for item in prepared],
                    "documents": [
                        {
                            "job_id": document["job_id"],
                            "deployed_review_marker": document[
                                "deployed_review_marker"
                            ],
                        }
                        for document in documents
                    ],
                },
            )
        except BaseException:
            try:
                for item, backup in reversed(completed):
                    _rollback_cutover_job(
                        store=store,
                        prepared=item,
                        backup_dir=backup,
                    )
            except BaseException:
                raise
            _release_claims_if_jobs_are_stable(
                store=store,
                job_ids=stage_job_ids,
                stage_batch=stage_batch,
                claimed_root=claimed_root,
            )
            raise
        for item in prepared:
            (store.job_dir(item.job_id) / ".cutover.json").unlink(missing_ok=True)
        _release_claims_if_jobs_are_stable(
            store=store,
            job_ids=stage_job_ids,
            stage_batch=stage_batch,
            claimed_root=claimed_root,
        )
    return {
        "reprocessed": len(prepared),
        "documents": documents,
        "staging_root": str(stage_batch),
        "backup_root": str(backups),
    }


def rollback_jobs(*, root: Path, backup_batch: Path) -> dict[str, Any]:
    store = JobStore(root)
    backup_jobs = sorted(path for path in backup_batch.iterdir() if path.is_dir())
    commit_path = backup_batch / ".committed.json"
    if not commit_path.is_file():
        raise ValueError("backup batch has no commit marker")
    commit = json.loads(commit_path.read_text())
    commit_version = commit.get("version")
    if commit_version not in {"reprocess_batch_commit_v1", "reprocess_batch_commit_v2"}:
        raise ValueError("unsupported backup batch commit marker")
    expected_markers = {
        str(document["job_id"]): _marker_from_payload(
            document["deployed_review_marker"]
        )
        for document in commit.get("documents", [])
        if document.get("deployed_review_marker") is not None
    }
    expected_ids = sorted(str(job_id) for job_id in commit.get("job_ids", []))
    if expected_ids != [backup.name for backup in backup_jobs]:
        raise ValueError("backup batch document inventory does not match its commit marker")

    for backup in backup_jobs:
        if (
            not (backup / "result.json").is_file()
            or not (backup / "artifacts").is_dir()
            or not (backup / "state.json").is_file()
        ):
            raise ValueError(f"backup is incomplete: {backup}")
        store.read(backup.name)

    review_markers: dict[str, tuple[bool, str | None]] = {}
    for backup in backup_jobs:
        job_id = backup.name
        with store.job_lock(job_id, exclusive=False):
            store._require_stable_workspace(job_id)
            current_review = store._read_review_unlocked(job_id)
            current_marker = _review_marker(store.job_dir(job_id))
            if commit_version == "reprocess_batch_commit_v2":
                if expected_markers.get(job_id) != current_marker:
                    raise ValueError(
                        "review changed after deployment; rollback aborted"
                    )
            else:
                backup_review = (
                    json.loads((backup / "review.json").read_text())
                    if (backup / "review.json").is_file()
                    else store.empty_review()
                )
                if not _legacy_rollback_review_is_safe(
                    job_id=job_id,
                    backup_review=backup_review,
                    current_review=current_review,
                ):
                    raise ValueError(
                        "review changed after deployment; rollback aborted"
                    )
            review_markers[job_id] = current_marker

    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    displaced_root = (
        store.jobs_root / ".reprocess-rollback-current" / timestamp
    )
    rollback_marker = displaced_root / ".rollback.json"
    rollback_commit = displaced_root / ".committed.json"
    rollback_jobs = [
        RollbackJob(
            job_id=backup.name,
            backup_dir=backup,
            displaced_dir=displaced_root / backup.name,
            review_marker=review_markers[backup.name],
        )
        for backup in backup_jobs
    ]

    with ExitStack() as locks:
        for item in rollback_jobs:
            locks.enter_context(store.job_lock(item.job_id, exclusive=True))
        for item in rollback_jobs:
            store._require_stable_workspace(item.job_id)
            if _review_marker(store.job_dir(item.job_id)) != item.review_marker:
                raise ValueError(
                    "review changed during rollback; no documents were restored"
                )

        displaced_root.mkdir(parents=True, mode=0o700)
        _atomic_json(
            rollback_marker,
            {
                "version": "reprocess_rollback_batch_v1",
                "started_at": utc_now(),
                "backup_batch": str(backup_batch),
                "job_ids": [item.job_id for item in rollback_jobs],
                "review_markers": {
                    item.job_id: _marker_payload(item.review_marker)
                    for item in rollback_jobs
                },
            },
        )
        try:
            for item in rollback_jobs:
                job_dir = store.job_dir(item.job_id)
                item.displaced_dir.mkdir(mode=0o700)
                shutil.copy2(
                    job_dir / "state.json",
                    item.displaced_dir / "state.json",
                )
                if (job_dir / "review.json").is_file():
                    shutil.copy2(
                        job_dir / "review.json",
                        item.displaced_dir / "review.json",
                    )
            for item in rollback_jobs:
                _atomic_json(
                    store.job_dir(item.job_id) / ".cutover.json",
                    {
                        "version": "job_rollback_v1",
                        "job_id": item.job_id,
                        "backup_dir": str(item.backup_dir),
                        "displaced_dir": str(item.displaced_dir),
                        "commit_marker": str(rollback_commit),
                        "started_at": utc_now(),
                    },
                )
            for item in rollback_jobs:
                job_dir = store.job_dir(item.job_id)
                (job_dir / "artifacts").replace(
                    item.displaced_dir / "artifacts"
                )
                (job_dir / "result.json").replace(
                    item.displaced_dir / "result.json"
                )
                (item.backup_dir / "artifacts").replace(
                    job_dir / "artifacts"
                )
                (item.backup_dir / "result.json").replace(
                    job_dir / "result.json"
                )
                _atomic_json(
                    job_dir / "state.json",
                    json.loads((item.backup_dir / "state.json").read_text()),
                )
                if (item.backup_dir / "review.json").is_file():
                    _atomic_json(
                        job_dir / "review.json",
                        json.loads(
                            (item.backup_dir / "review.json").read_text()
                        ),
                    )
                elif (job_dir / "review.json").is_file():
                    (job_dir / "review.json").unlink()
            _atomic_json(
                rollback_commit,
                {
                    "version": "reprocess_rollback_commit_v1",
                    "committed_at": utc_now(),
                    "job_ids": [item.job_id for item in rollback_jobs],
                },
            )
        except BaseException:
            try:
                for item in reversed(rollback_jobs):
                    store._restore_rollback_job_unlocked(
                        item.job_id,
                        {
                            "version": "job_rollback_v1",
                            "job_id": item.job_id,
                            "backup_dir": str(item.backup_dir),
                            "displaced_dir": str(item.displaced_dir),
                            "commit_marker": str(rollback_commit),
                        },
                    )
                for item in rollback_jobs:
                    (store.job_dir(item.job_id) / ".cutover.json").unlink(
                        missing_ok=True
                    )
                shutil.rmtree(displaced_root)
            except BaseException as recovery_error:
                raise RuntimeError(
                    "manual rollback recovery is required before jobs can be read"
                ) from recovery_error
            raise

        for item in rollback_jobs:
            (store.job_dir(item.job_id) / ".cutover.json").unlink(missing_ok=True)

    documents = [
        {
            "job_id": item.job_id,
            "displaced_result": str(item.displaced_dir),
        }
        for item in rollback_jobs
    ]
    return {
        "restored": len(documents),
        "documents": documents,
        "displaced_root": str(displaced_root),
    }


@app.callback()
def run(
    root: Annotated[Path, typer.Option(file_okay=False, resolve_path=True)],
    job_id: Annotated[list[str] | None, typer.Option("--job-id")] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Stage, validate, and replace completed results."),
    ] = False,
    stage_only: Annotated[
        bool,
        typer.Option("--stage-only", help="Stage and validate results without cutover."),
    ] = False,
    apply_staged: Annotated[
        Path | None,
        typer.Option(
            "--apply-staged",
            exists=True,
            file_okay=False,
            help="Cut over a previously validated staging batch.",
        ),
    ] = None,
    stage_root: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    backup_root: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    rollback_from: Annotated[
        Path | None,
        typer.Option(
            "--rollback-from",
            exists=True,
            file_okay=False,
            help="Restore every job from a previous backup batch.",
        ),
    ] = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
) -> None:
    if rollback_from is not None:
        typer.echo(
            json.dumps(
                rollback_jobs(root=root, backup_batch=rollback_from),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if apply_staged is not None:
        if apply or stage_only or job_id:
            raise typer.BadParameter(
                "--apply-staged cannot be combined with job selection or staging"
            )
        typer.echo(
            json.dumps(
                apply_staged_jobs(
                    root=root,
                    stage_batch=apply_staged,
                    backup_root=backup_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if stage_only:
        if apply:
            raise typer.BadParameter("--stage-only and --apply are mutually exclusive")
        typer.echo(
            json.dumps(
                stage_reprocess_jobs(
                    root=root,
                    job_ids=job_id,
                    stage_root=stage_root,
                    vl_url=vl_url,
                    paddle_device=paddle_device,
                    vl_device=vl_device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = reprocess_jobs(
        root=root,
        job_ids=job_id,
        apply=apply,
        stage_root=stage_root,
        backup_root=backup_root,
        vl_url=vl_url,
        paddle_device=paddle_device,
        vl_device=vl_device,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
