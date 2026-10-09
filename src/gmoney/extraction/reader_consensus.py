"""Gemini as an independent second reader of table crops, and cell-level consensus.

Rules (``teleocr_gemini`` mode):

* amounts agree -> accept; disagree -> choose the value that makes the table's printed
  total reconcile; if neither or both do, keep TeleOCR's value and flag the cell with both;
* descriptions are compared normalized (case, whitespace, punctuation); on disagreement
  TeleOCR's text is kept, the field flagged, and Gemini's alternative stored for review;
* a row read by only one reader is kept and flagged.

Only redacted table crops are sent, never a full page, under a per-page cost cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from gmoney.contracts.extraction import RowRole
from gmoney.extraction.reconciliation import match_outcome
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.typed_values import parse_decimal
from gmoney.inference.gemini import TableReaderRow, TableReadResponse

GEMINI_ROUTE = "gemini_table_reader"
FULL_PAGE_AREA_SHARE = 0.85
# Conservative per-call token estimate used before the first measured call on a page.
ESTIMATED_INPUT_TOKENS = 1800
ESTIMATED_OUTPUT_TOKENS = 2500
PENDING_FLAGS = frozenset(
    {
        "reader_amount_disagreement",
        "reader_description_disagreement",
        "reader_only_teleocr",
        "reader_only_gemini",
    }
)


class TableReader(Protocol):
    model: str

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal: ...

    def read_table(self, crop_bytes: bytes, mime_type: str = "image/png") -> TableReadResponse: ...


@dataclass
class PageBudget:
    """Per-page Gemini cost cap (INR converted to USD with the configured rate)."""

    limit_usd: Decimal
    spent_usd: Decimal = Decimal(0)
    calls: int = 0
    blocked: int = 0
    largest_call_usd: Decimal = Decimal(0)

    @classmethod
    def from_inr(cls, limit_inr: float, inr_per_usd: float) -> PageBudget:
        return cls(limit_usd=Decimal(str(limit_inr)) / Decimal(str(inr_per_usd)))

    def allows(self, reader: TableReader) -> bool:
        expected = max(
            self.largest_call_usd,
            reader.estimated_cost_usd(ESTIMATED_INPUT_TOKENS, ESTIMATED_OUTPUT_TOKENS),
        )
        return self.spent_usd + expected <= self.limit_usd

    def record(self, cost_usd: Decimal) -> None:
        self.calls += 1
        self.spent_usd += cost_usd
        self.largest_call_usd = max(self.largest_call_usd, cost_usd)


def is_full_page_crop(
    crop_box: tuple[int, int, int, int], page_width: int, page_height: int
) -> bool:
    left, top, right, bottom = crop_box
    area = max(0, right - left) * max(0, bottom - top)
    return area >= FULL_PAGE_AREA_SHARE * page_width * page_height


def second_read_table(
    reader: TableReader,
    *,
    redacted_crop: Path,
    crop_box: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
    budget: PageBudget,
) -> tuple[TableReadResponse | None, str | None]:
    """Send one redacted table crop to Gemini, or say why it was not sent."""
    if is_full_page_crop(crop_box, page_width, page_height):
        return None, "full_page_crop_not_sent"
    if not budget.allows(reader):
        budget.blocked += 1
        return None, "page_cost_cap_reached"
    response = reader.read_table(redacted_crop.read_bytes(), "image/png")
    budget.record(response.measured_cost_usd)
    return response, None


def gemini_candidates(response: TableReadResponse) -> tuple[CandidateLedgerRow, ...]:
    rows: list[CandidateLedgerRow] = []
    for index, row in enumerate(response.rows):
        rows.append(_candidate(index, row))
    return tuple(rows)


def _candidate(index: int, row: TableReaderRow) -> CandidateLedgerRow:
    amount = parse_decimal(row.amount)
    role = RowRole.DETAIL
    if row.is_total:
        role = RowRole.SECTION_TOTAL
    elif row.is_return:
        # Printed amount kept, as for TeleOCR rows; the refund role carries the sign.
        role = RowRole.REFUND
    return CandidateLedgerRow(
        source_row=index,
        role=role,
        cells=tuple(
            value or ""
            for value in (row.description, row.quantity, row.unit_price, row.discount, row.amount)
        ),
        section=row.section,
        description=(row.description or "").strip() or None,
        quantity=parse_decimal(row.quantity),
        rate=parse_decimal(row.unit_price),
        discount=parse_decimal(row.discount),
        amount=amount,
        source_route=GEMINI_ROUTE,
    )


def normalized_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _similarity(left: CandidateLedgerRow, right: CandidateLedgerRow) -> float:
    a, b = normalized_text(left.description), normalized_text(right.description)
    text = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
    same_amount = left.amount is not None and left.amount == right.amount
    return text + (0.5 if same_amount else 0.0)


def align_reader_rows(
    primary: tuple[CandidateLedgerRow, ...],
    secondary: tuple[CandidateLedgerRow, ...],
    *,
    threshold: float = 0.75,
) -> list[tuple[int | None, int | None]]:
    """Order-preserving pairing that maximises total description/amount similarity."""
    rows, columns = len(primary), len(secondary)
    scores = [[_similarity(left, right) for right in secondary] for left in primary]
    best = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            paired = scores[i][j] + best[i + 1][j + 1] if scores[i][j] >= threshold else -1.0
            best[i][j] = max(paired, best[i + 1][j], best[i][j + 1])
    pairs: list[tuple[int | None, int | None]] = []
    i = j = 0
    while i < rows and j < columns:
        if scores[i][j] >= threshold and best[i][j] == scores[i][j] + best[i + 1][j + 1]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            pairs.append((i, None))
            i += 1
        else:
            pairs.append((None, j))
            j += 1
    pairs.extend((index, None) for index in range(i, rows))
    pairs.extend((None, index) for index in range(j, columns))
    return pairs


@dataclass
class ConsensusResult:
    rows: tuple[CandidateLedgerRow, ...]
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "disagreements": list(self.disagreements)}


def _text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _matches_any(total: Decimal, targets: tuple[Decimal, ...]) -> bool:
    return any(match_outcome(target, total) != "fail" for target in targets)


def _flag(row: CandidateLedgerRow, *flags: str) -> CandidateLedgerRow:
    return replace(row, validation_flags=tuple(dict.fromkeys((*row.validation_flags, *flags))))


def reconcile_readers(
    teleocr_rows: tuple[CandidateLedgerRow, ...],
    gemini_rows: tuple[CandidateLedgerRow, ...],
    *,
    printed_totals: tuple[Decimal, ...] = (),
) -> ConsensusResult:
    """Cell-level consensus between the two readers' charge rows for one table."""
    charge_roles = {RowRole.DETAIL, RowRole.REFUND}
    primary = tuple(row for row in teleocr_rows if row.role in charge_roles)
    secondary = tuple(row for row in gemini_rows if row.role in charge_roles)
    passthrough = tuple(row for row in teleocr_rows if row.role not in charge_roles)
    totals = tuple(
        dict.fromkeys(
            (
                *printed_totals,
                *(
                    abs(row.amount)
                    for row in (*teleocr_rows, *gemini_rows)
                    if row.role in {RowRole.SECTION_TOTAL, RowRole.DOCUMENT_TOTAL}
                    and row.amount is not None
                ),
            )
        )
    )
    pairs = align_reader_rows(primary, secondary)
    base_sum = sum((row.amount for row in primary if row.amount is not None), Decimal(0))
    counts = {
        "rows_compared": 0,
        "rows_agreed": 0,
        "amount_disagreements": 0,
        "amount_resolved_by_total": 0,
        "description_disagreements": 0,
        "only_teleocr": 0,
        "only_gemini": 0,
    }
    disagreements: list[dict[str, Any]] = []
    output: list[CandidateLedgerRow] = []
    for tele_index, gemini_index in pairs:
        if tele_index is None:
            row = _flag(secondary[gemini_index], "reader_only_gemini")
            counts["only_gemini"] += 1
            disagreements.append(
                {"field": "row", "teleocr": None, "gemini": row.description, "resolution": "kept"}
            )
            output.append(row)
            continue
        tele = primary[tele_index]
        if gemini_index is None:
            counts["only_teleocr"] += 1
            disagreements.append(
                {
                    "field": "row",
                    "source_row": tele.source_row,
                    "teleocr": tele.description,
                    "gemini": None,
                    "resolution": "kept",
                }
            )
            output.append(_flag(tele, "reader_only_teleocr"))
            continue
        gemini = secondary[gemini_index]
        counts["rows_compared"] += 1
        row = tele
        flags: list[str] = []
        if tele.amount != gemini.amount:
            tele_ok = bool(totals) and _matches_any(base_sum, totals)
            gemini_sum = (
                base_sum - (tele.amount or Decimal(0)) + (gemini.amount or Decimal(0))
                if gemini.amount is not None
                else None
            )
            gemini_ok = gemini_sum is not None and _matches_any(gemini_sum, totals)
            if gemini_ok and not tele_ok:
                row = replace(row, amount=gemini.amount)
                base_sum = gemini_sum
                resolution = "gemini_reconciles"
                flags.append("reader_amount_resolved_by_total")
                counts["amount_resolved_by_total"] += 1
            elif tele_ok and not gemini_ok:
                resolution = "teleocr_reconciles"
                flags.append("reader_amount_resolved_by_total")
                counts["amount_resolved_by_total"] += 1
            else:
                resolution = "unresolved"
                flags.append("reader_amount_disagreement")
                counts["amount_disagreements"] += 1
            disagreements.append(
                {
                    "field": "amount",
                    "source_row": tele.source_row,
                    "description": tele.description,
                    "teleocr": _text(tele.amount),
                    "gemini": _text(gemini.amount),
                    "resolution": resolution,
                }
            )
        if normalized_text(tele.description) != normalized_text(gemini.description):
            flags.append("reader_description_disagreement")
            counts["description_disagreements"] += 1
            disagreements.append(
                {
                    "field": "description",
                    "source_row": tele.source_row,
                    "teleocr": tele.description,
                    "gemini": gemini.description,
                    "resolution": "teleocr_kept",
                }
            )
        if not flags:
            flags.append("reader_agree")
            counts["rows_agreed"] += 1
        output.append(_flag(row, *flags))
    return ConsensusResult(
        rows=(*output, *passthrough),
        disagreements=disagreements,
        counts=counts,
    )


def needs_reviewer(row_flags: tuple[str, ...] | list[str]) -> bool:
    return bool(PENDING_FLAGS.intersection(row_flags))
