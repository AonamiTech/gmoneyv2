"""Per-table finishing step for the TeleOCR (and TeleOCR + Gemini) table readers.

The offline extractor calls :func:`finish_reader_table` once the TeleOCR reply for a table
has been parsed.  It dedupes a missed-table guard read, runs the optional Gemini second
read and consensus, grounds rows to PP-OCR tokens, and publishes canonical rows.  Rows
that cannot be grounded are not dropped silently: they are returned for a blocking
review issue that carries their printed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from gmoney.contracts.evidence import OcrToken
from gmoney.contracts.extraction import CanonicalRow, ReviewDisposition, RowRole, TableType
from gmoney.extraction.canonicalize import (
    BILLABLE_ROLES,
    canonicalize_rows,
    is_publishable_aligned_row,
)
from gmoney.extraction.reader_consensus import (
    PageBudget,
    TableReader,
    gemini_candidates,
    needs_reviewer,
    reconcile_readers,
    second_read_table,
)
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import align_candidate_rows
from gmoney.extraction.teleocr_reader import drop_rows_already_read
from gmoney.inference.redaction import redact_crop

READER_MODES = ("heuristic", "teleocr", "teleocr_gemini")


@dataclass
class ReaderTableOutcome:
    rows: tuple[CanonicalRow, ...]
    ungrounded: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: dict[str, Any] = field(default_factory=dict)
    gemini_calls: int = 0
    gemini_cost_usd: Decimal = Decimal(0)
    gemini_input_sha256: str | None = None


def _text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _apply_table_schema(
    candidates: tuple[CandidateLedgerRow, ...], table_type: TableType | None
) -> tuple[CandidateLedgerRow, ...]:
    if table_type is None:
        return candidates
    summary = table_type in {TableType.CATEGORY_SUMMARY, TableType.PACKAGE_SUMMARY}
    return tuple(
        replace(
            candidate,
            role=(
                RowRole.CATEGORY_ROLLUP
                if summary and candidate.role is RowRole.DETAIL
                else candidate.role
            ),
            table_type=table_type,
        )
        for candidate in candidates
    )


def finish_reader_table(
    *,
    document_id: str,
    page_number: int,
    table_id: str,
    page_artifact_sha256: str,
    candidates: tuple[CandidateLedgerRow, ...],
    tokens: tuple[OcrToken, ...],
    starting_order: int,
    table_type: TableType | None = None,
    existing_page_rows: tuple[tuple[object, object], ...] = (),
    is_guard: bool = False,
    gemini_reader: TableReader | None = None,
    gemini_allowed: bool = False,
    budget: PageBudget | None = None,
    crop_path: Path | None = None,
    crop_tokens: tuple[OcrToken, ...] = (),
    crop_size: tuple[int, int] | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
    page_size: tuple[int, int] | None = None,
    redaction_output: Path | None = None,
) -> ReaderTableOutcome:
    diagnostic: dict[str, Any] = {"reader_candidate_count": len(candidates)}
    if is_guard:
        before = len(candidates)
        candidates = drop_rows_already_read(candidates, existing_page_rows)
        diagnostic["reader_full_page_guard"] = {
            "candidates": before,
            "already_read": before - len(candidates),
        }
    outcome = ReaderTableOutcome(rows=(), diagnostic=diagnostic)
    if gemini_reader is not None:
        block_reason: str | None = None
        if not gemini_allowed:
            block_reason = "gemini_not_allowed_in_this_stage"
        elif is_guard:
            block_reason = "guard_band_not_sent"
        elif None in (budget, crop_path, crop_size, crop_box, page_size, redaction_output):
            block_reason = "crop_unavailable"
        if block_reason is None:
            redaction = redact_crop(crop_path, redaction_output, crop_tokens, (0, 0, *crop_size))
            if not redaction.safe:
                block_reason = "redaction_blocked"
            else:
                try:
                    response, block_reason = second_read_table(
                        gemini_reader,
                        redacted_crop=redaction.path,
                        crop_box=crop_box,
                        page_width=page_size[0],
                        page_height=page_size[1],
                        budget=budget,
                    )
                except Exception as error:  # a failed second read flags, never blocks rows
                    response, block_reason = None, f"provider_error:{type(error).__name__}"
                    outcome.gemini_calls += 1
                if response is not None:
                    outcome.gemini_calls += 1
                    outcome.gemini_cost_usd += response.measured_cost_usd
                    outcome.gemini_input_sha256 = redaction.artifact_sha256
                    consensus = reconcile_readers(candidates, gemini_candidates(response))
                    candidates = consensus.rows
                    diagnostic["reader_consensus"] = consensus.summary()
                    if response.rejected_reasons:
                        diagnostic["gemini_reader_rejected"] = list(response.rejected_reasons)
        if block_reason is not None:
            diagnostic["gemini_reader_block_reason"] = block_reason
            candidates = tuple(
                replace(row, validation_flags=(*row.validation_flags, "gemini_not_read"))
                if row.role in {RowRole.DETAIL, RowRole.REFUND}
                else row
                for row in candidates
            )
        diagnostic["gemini_reader_calls"] = outcome.gemini_calls
        diagnostic["gemini_reader_cost_usd"] = _text(outcome.gemini_cost_usd)
    candidates = _apply_table_schema(candidates, table_type)
    aligned = align_candidate_rows(candidates, tokens)
    publishable = tuple(row for row in aligned if is_publishable_aligned_row(row))
    for row in aligned:
        if row in publishable or row.candidate.role not in BILLABLE_ROLES:
            continue
        outcome.ungrounded.append(
            {
                "description": row.candidate.description,
                "amount": _text(row.candidate.amount),
                "section": row.candidate.section,
                "role": row.candidate.role.value,
                "source_route": row.candidate.source_route,
                "grounded_fields": sorted(row.field_token_ids),
            }
        )
    rows = canonicalize_rows(
        document_id,
        page_number,
        table_id,
        page_artifact_sha256,
        publishable,
        starting_order=starting_order,
    )
    outcome.rows = tuple(
        row.model_copy(update={"review_disposition": ReviewDisposition.PENDING})
        if needs_reviewer(row.validation_flags)
        else row
        for row in rows
    )
    if outcome.ungrounded:
        diagnostic["reader_ungrounded_rows"] = outcome.ungrounded
    diagnostic["reader_published_rows"] = len(outcome.rows)
    return outcome
