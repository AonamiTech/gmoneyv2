from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import DocumentTotal, EvidenceRef
from gmoney.extraction.typed_values import parse_decimal

DOCUMENT_TOTAL_VERSION = "document_total_v1"

FINAL_LABELS: tuple[tuple[str, str, int], ...] = (
    ("net bill amount", "Net Bill Amount", 4),
    ("final bill amount", "Final Bill Amount", 3),
    ("total bill amount", "Total Bill Amount", 2),
    ("grand total", "Grand Total", 1),
)

EXCLUDED_LABEL_TERMS = (
    "sub total",
    "subtotal",
    "gross amount",
    "gross bill",
    "discount",
    "round off",
    "rounding",
    "cgst",
    "sgst",
    "igst",
    "tax amount",
    "advance",
    "deposit",
    "amount paid",
    "amount received",
    "amount refunded",
    "balance",
    "payer",
    "patient",
    "company",
    "receipt",
    "refund",
)

MONEY_FRAGMENT = re.compile(
    r"(?:₹|inr|rs\.?)?\s*[+-]?(?:\d[\d,]*)(?:\.\d{1,4})?(?:\s*(?:cr|dr)\.?)?",
    re.IGNORECASE,
)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _bounds(token: OcrToken) -> tuple[float, float, float, float]:
    xs = [point.x for point in token.polygon.points]
    ys = [point.y for point in token.polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def _center_x(token: OcrToken) -> float:
    left, _, right, _ = _bounds(token)
    return (left + right) / 2


def _center_y(token: OcrToken) -> float:
    _, top, _, bottom = _bounds(token)
    return (top + bottom) / 2


def _height(token: OcrToken) -> float:
    _, top, _, bottom = _bounds(token)
    return max(1.0, bottom - top)


@dataclass(frozen=True)
class _Line:
    tokens: tuple[OcrToken, ...]

    @property
    def text(self) -> str:
        return " ".join(token.text.strip() for token in self.tokens if token.text.strip())

    @property
    def center_y(self) -> float:
        return median(_center_y(token) for token in self.tokens)


@dataclass(frozen=True)
class DocumentTotalCandidate:
    total: DocumentTotal
    label_priority: int
    vertical_position: float

    @property
    def rank(self) -> tuple[int, int, float, float]:
        return (
            self.label_priority,
            self.total.page_number,
            self.vertical_position,
            self.total.confidence,
        )


def _lines(tokens: tuple[OcrToken, ...]) -> tuple[_Line, ...]:
    if not tokens:
        return ()
    tolerance = max(5.0, median(_height(token) for token in tokens) * 0.65)
    grouped: list[list[OcrToken]] = []
    for token in sorted(tokens, key=lambda item: (_center_y(item), _center_x(item))):
        if not grouped:
            grouped.append([token])
            continue
        current_y = median(_center_y(item) for item in grouped[-1])
        if abs(_center_y(token) - current_y) <= tolerance:
            grouped[-1].append(token)
        else:
            grouped.append([token])
    return tuple(_Line(tuple(sorted(group, key=_center_x))) for group in grouped)


def _label(line: _Line) -> tuple[str, int] | None:
    normalized = _normalize(line.text)
    if any(term in normalized for term in EXCLUDED_LABEL_TERMS):
        return None
    return next(
        ((display, priority) for term, display, priority in FINAL_LABELS if term in normalized),
        None,
    )


def _money_values(line: _Line) -> list[tuple[OcrToken, str, Decimal]]:
    values: list[tuple[OcrToken, str, Decimal]] = []
    for token in line.tokens:
        direct = parse_decimal(token.text)
        if direct is not None:
            values.append((token, token.text.strip(), direct))
            continue
        for match in MONEY_FRAGMENT.finditer(token.text):
            raw = match.group().strip()
            parsed = parse_decimal(raw)
            if parsed is not None:
                values.append((token, raw, parsed))
    return values


def _evidence(
    label_line: _Line,
    amount_line: _Line,
    amount_token: OcrToken,
) -> tuple[EvidenceRef, float]:
    tokens_by_id: dict[str, OcrToken] = {}
    for token in (
        *label_line.tokens,
        *(amount_line.tokens if amount_line is not label_line else ()),
        amount_token,
    ):
        tokens_by_id.setdefault(token.token_id, token)
    tokens = tuple(tokens_by_id.values())
    boxes = [_bounds(token) for token in tokens]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return (
        EvidenceRef(
            page_number=amount_token.page_number,
            polygon=Polygon(
                points=(
                    Point(x=left, y=top),
                    Point(x=right, y=top),
                    Point(x=right, y=bottom),
                    Point(x=left, y=bottom),
                )
            ),
            artifact_sha256=amount_token.artifact_sha256,
            token_ids=tuple(token.token_id for token in tokens),
        ),
        sum(token.confidence for token in tokens) / len(tokens),
    )


def extract_document_total_candidates(
    tokens: tuple[OcrToken, ...],
) -> tuple[DocumentTotalCandidate, ...]:
    lines = _lines(tokens)
    if not lines:
        return ()
    line_height = median(_height(token) for token in tokens)
    candidates: list[DocumentTotalCandidate] = []
    for index, line in enumerate(lines):
        detected = _label(line)
        if detected is None:
            continue
        label, priority = detected
        amount_line = line
        values = _money_values(line)
        if not values and index + 1 < len(lines):
            following = lines[index + 1]
            if (
                following.center_y - line.center_y <= line_height * 2.2
                and _label(following) is None
            ):
                amount_line = following
                values = _money_values(following)
        if not values:
            continue
        amount_token, amount_raw, amount = max(values, key=lambda value: _center_x(value[0]))
        evidence, confidence = _evidence(line, amount_line, amount_token)
        total = DocumentTotal(
            amount_raw=amount_raw,
            amount=amount,
            label=label,
            page_number=amount_token.page_number,
            evidence=evidence,
            confidence=confidence,
        )
        candidates.append(
            DocumentTotalCandidate(
                total=total,
                label_priority=priority,
                vertical_position=amount_line.center_y,
            )
        )
    return tuple(candidates)


def select_document_total(
    candidates: tuple[DocumentTotalCandidate, ...] | list[DocumentTotalCandidate],
) -> DocumentTotal | None:
    return max(candidates, key=lambda candidate: candidate.rank).total if candidates else None
