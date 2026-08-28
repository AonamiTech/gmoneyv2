from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import (
    DocumentTotal,
    DocumentTotalKind,
    DocumentTotalScope,
    EvidenceRef,
)
from gmoney.extraction.typed_values import parse_decimal

DOCUMENT_TOTAL_VERSION = "document_total_v3"
DOCUMENT_TOTALS_VERSION = "document_totals_v2"

# term, display label, priority, kind, default scope, requires summary context
FINAL_LABELS: tuple[
    tuple[str, str, int, DocumentTotalKind, DocumentTotalScope, bool], ...
] = (
    (
        "total bill amount",
        "Total Bill Amount",
        100,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "net bill amount",
        "Net Bill Amount",
        98,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "net medical amount",
        "Net Medical Amount",
        97,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "final bill amount",
        "Final Bill Amount",
        96,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "total gross bill value",
        "Total Gross Bill Value",
        94,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "net amount incl tax",
        "Net Amount (Incl. Tax)",
        90,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        True,
    ),
    (
        "net amount ind tax",
        "Net Amount (Incl. Tax)",
        90,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        True,
    ),
    (
        "grand total",
        "Grand Total",
        86,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        False,
    ),
    (
        "total payable amount",
        "Total Payable Amount",
        76,
        DocumentTotalKind.PAYABLE_TOTAL,
        DocumentTotalScope.SETTLEMENT,
        True,
    ),
    (
        "net patient payable amount",
        "Net Patient Payable Amount",
        74,
        DocumentTotalKind.PAYABLE_TOTAL,
        DocumentTotalScope.SETTLEMENT,
        True,
    ),
    (
        "net patient payable amt",
        "Net Patient Payable Amount",
        74,
        DocumentTotalKind.PAYABLE_TOTAL,
        DocumentTotalScope.SETTLEMENT,
        True,
    ),
    (
        "net payable",
        "Net Payable",
        72,
        DocumentTotalKind.PAYABLE_TOTAL,
        DocumentTotalScope.SETTLEMENT,
        True,
    ),
    (
        "total amount",
        "Total Amount",
        66,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        True,
    ),
    (
        "gross amount",
        "Gross Amount",
        64,
        DocumentTotalKind.GROSS_TOTAL,
        DocumentTotalScope.DOCUMENT,
        True,
    ),
    (
        "net amount",
        "Net Amount",
        62,
        DocumentTotalKind.BILL_TOTAL,
        DocumentTotalScope.DOCUMENT,
        True,
    ),
    (
        "amount to be received",
        "Amount To Be Received",
        40,
        DocumentTotalKind.SETTLEMENT_TOTAL,
        DocumentTotalScope.PAYMENT,
        True,
    ),
    (
        "amount to be recelved",
        "Amount To Be Received",
        40,
        DocumentTotalKind.SETTLEMENT_TOTAL,
        DocumentTotalScope.PAYMENT,
        True,
    ),
    (
        "amount to be receive",
        "Amount To Be Received",
        40,
        DocumentTotalKind.SETTLEMENT_TOTAL,
        DocumentTotalScope.PAYMENT,
        True,
    ),
)

EXCLUDED_LABEL_TERMS = (
    "discount",
    "rounding",
    "cgst",
    "sgst",
    "igst",
    "tax amount",
    "advance",
    "deposit",
    "amount paid",
    "amount refunded",
    "balance due",
    "receipt",
    "refund",
)

SUMMARY_CONTEXT_TERMS = (
    "gross amount",
    "net amount",
    "total amount",
    "total payable",
    "net payable",
    "amount to be received",
    "amount to be recelved",
    "advance amount",
    "paid amount",
    "balance amount",
    "roundoff amount",
    "round off amount",
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
    local_context: str = ""

    @property
    def rank(self) -> tuple[int, int, Decimal, float, int, float]:
        scope_priority = {
            DocumentTotalScope.DOCUMENT: 4,
            DocumentTotalScope.SETTLEMENT: 3,
            DocumentTotalScope.SECTION: 2,
            DocumentTotalScope.PAYMENT: 1,
        }[self.total.scope]
        return (
            self.label_priority,
            scope_priority,
            abs(self.total.amount),
            self.total.confidence,
            self.total.page_number,
            self.vertical_position,
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


def _label(
    line: _Line,
    local_text: str,
) -> tuple[str, int, DocumentTotalKind, DocumentTotalScope, bool] | None:
    normalized = _normalize(line.text)
    if normalized.startswith(("total for", "sub total", "subtotal")) or normalized.startswith(
        ("patient grand total", "company grand total", "payer grand total")
    ):
        return None
    detected = next((spec for spec in FINAL_LABELS if spec[0] in normalized), None)
    if detected is None or any(term in normalized for term in EXCLUDED_LABEL_TERMS):
        return None
    term, display, priority, kind, scope, requires_summary = detected
    if term == "grand total":
        pharmacy_markers = {
            "pharmacy",
            "medicine",
            "drug",
            "batch",
            "expiry",
            "mrp",
            "cgst",
            "sgst",
        }
        pharmacy_marker_count = len(
            pharmacy_markers.intersection(local_text.split())
        )
        invoice_context = "invoice" in local_text and pharmacy_marker_count >= 2
        pharmacy_table_context = pharmacy_marker_count >= 3
        if (
            "pharmacy detailed bill" in local_text
            or invoice_context
            or pharmacy_table_context
        ):
            scope = DocumentTotalScope.SECTION
        if any(
            marker in local_text
            for marker in (
                "receipt no",
                "receipt number",
                "charges towards",
                "payment receipt",
            )
        ):
            scope = DocumentTotalScope.SECTION
        if any(
            marker in local_text
            for marker in (
                "category summary",
                "package summary",
                "charge summary",
                "department summary",
            )
        ):
            scope = DocumentTotalScope.SECTION
    return display, priority, kind, scope, requires_summary


def _has_summary_context(lines: tuple[_Line, ...], index: int) -> bool:
    window = " ".join(
        _normalize(line.text)
        for line in lines[max(0, index - 5) : min(len(lines), index + 6)]
    )
    return sum(term in window for term in SUMMARY_CONTEXT_TERMS) >= 2


def _standalone_money_line(line: _Line) -> bool:
    stripped = re.sub(r"\b(?:inr|rs|cr|dr|rupees?)\.?\b", "", line.text, flags=re.I)
    return bool(_money_values(line)) and not re.search(r"[A-Za-z]", stripped)


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
        context_start = max(0, index - 5)
        context_end = min(len(lines), index + 4)
        local_text = _normalize(
            " ".join(candidate.text for candidate in lines[context_start:context_end])
        )
        detected = _label(line, local_text)
        if detected is None:
            continue
        label, priority, kind, scope, requires_summary = detected
        if requires_summary and not _has_summary_context(lines, index):
            continue
        amount_line = line
        values = _money_values(line)
        if not values and index + 1 < len(lines):
            following = lines[index + 1]
            if (
                following.center_y - line.center_y <= line_height * 2.2
                and _label(following, local_text) is None
                and _standalone_money_line(following)
            ):
                amount_line = following
                values = _money_values(following)
        if not values:
            continue
        amount_token, amount_raw, amount = max(values, key=lambda value: _center_x(value[0]))
        evidence, confidence = _evidence(line, amount_line, amount_token)
        if label == "Grand Total" and any(
            marker in local_text
            for marker in (
                "receipt",
                "charges towards",
                "collection",
                "category summary",
                "charge summary",
                "package summary",
            )
        ):
            scope = DocumentTotalScope.SECTION
        context_kind = {
            DocumentTotalScope.DOCUMENT: "document_final",
            DocumentTotalScope.SECTION: "section",
            DocumentTotalScope.SETTLEMENT: "settlement",
            DocumentTotalScope.PAYMENT: "payment",
        }[scope]
        if scope is DocumentTotalScope.SECTION:
            if any(
                marker in local_text
                for marker in ("pharmacy", "medicine", "drug", "batch", "expiry", "mrp")
            ):
                context_kind = "pharmacy"
            elif any(
                marker in local_text
                for marker in ("receipt", "charges towards", "collection")
            ):
                context_kind = "receipt"
            else:
                context_kind = "category"
        context_id = f"p{amount_token.page_number}:page-summary:{context_kind}:o1"
        total = DocumentTotal(
            amount_raw=amount_raw,
            amount=amount,
            label=label,
            kind=kind,
            scope=scope,
            page_number=amount_token.page_number,
            evidence=evidence,
            confidence=confidence,
            context_id=context_id,
            context_kind=context_kind,
        )
        candidates.append(
            DocumentTotalCandidate(
                total=total,
                label_priority=priority,
                vertical_position=amount_line.center_y,
                local_context=local_text,
            )
        )
    return tuple(candidates)


def assign_document_total_contexts(
    candidates: tuple[DocumentTotalCandidate, ...] | list[DocumentTotalCandidate],
    diagnostics: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> tuple[DocumentTotalCandidate, ...]:
    """Attach stable table/invoice context identities after layout is known."""
    regions = tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.get("table_id")
        and isinstance(diagnostic.get("box"), (tuple, list))
        and len(diagnostic["box"]) == 4
    )
    classified: list[tuple[DocumentTotalCandidate, str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.total.page_number, item.vertical_position),
    ):
        points = candidate.total.evidence.polygon.points
        center_x = sum(point.x for point in points) / len(points)
        center_y = sum(point.y for point in points) / len(points)
        region = next(
            (
                diagnostic
                for diagnostic in regions
                if diagnostic.get("page_number") == candidate.total.page_number
                and float(diagnostic["box"][0]) <= center_x <= float(diagnostic["box"][2])
                and float(diagnostic["box"][1]) <= center_y <= float(diagnostic["box"][3])
            ),
            None,
        )
        identity = str((region or {}).get("table_id") or "page-summary")
        table_type = str((region or {}).get("table_type") or "")
        context = _normalize(candidate.local_context)
        if table_type == "payment":
            kind = "payment"
        elif table_type == "pharmacy" or any(
            marker in context
            for marker in ("pharmacy", "medicine", "drug", "batch", "expiry", "mrp")
        ):
            kind = "pharmacy"
        elif any(marker in context for marker in ("receipt", "charges towards", "collection")):
            kind = "receipt"
        elif table_type in {"category_summary", "package_summary"} or any(
            marker in context
            for marker in ("category summary", "package summary", "charge summary")
        ):
            kind = "category"
        elif candidate.total.scope is DocumentTotalScope.SETTLEMENT or any(
            marker in context for marker in ("settlement", "claim approved", "tpa amount")
        ):
            kind = "settlement"
        elif candidate.total.scope is DocumentTotalScope.PAYMENT:
            kind = "payment"
        else:
            kind = "document_final"
        classified.append((candidate, identity, kind))

    ordinal_state: dict[tuple[int, str, str], tuple[int, float, float]] = {}
    output: list[DocumentTotalCandidate] = []
    for candidate, identity, kind in classified:
        key = (candidate.total.page_number, identity, kind)
        points = candidate.total.evidence.polygon.points
        top = min(point.y for point in points)
        bottom = max(point.y for point in points)
        height = max(1.0, bottom - top)
        previous = ordinal_state.get(key)
        if previous is None:
            ordinal = 1
        else:
            previous_ordinal, previous_bottom, previous_height = previous
            ordinal = previous_ordinal + int(
                top - previous_bottom > max(80.0, previous_height * 4.0, height * 4.0)
            )
        ordinal_state[key] = (ordinal, bottom, height)
        scope = candidate.total.scope
        if kind in {"category", "pharmacy", "receipt"}:
            scope = DocumentTotalScope.SECTION
        elif kind == "payment":
            scope = DocumentTotalScope.PAYMENT
        elif kind == "settlement":
            scope = DocumentTotalScope.SETTLEMENT
        output.append(
            DocumentTotalCandidate(
                total=candidate.total.model_copy(
                    update={
                        "scope": scope,
                        "context_kind": kind,
                        "context_id": (
                            f"p{candidate.total.page_number}:{identity}:{kind}:o{ordinal}"
                        ),
                    }
                ),
                label_priority=candidate.label_priority,
                vertical_position=candidate.vertical_position,
                local_context=candidate.local_context,
            )
        )
    return tuple(output)


def select_document_total(
    candidates: tuple[DocumentTotalCandidate, ...] | list[DocumentTotalCandidate],
) -> DocumentTotal | None:
    document_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.total.scope is DocumentTotalScope.DOCUMENT
        and candidate.total.context_kind == "document_final"
        and candidate.total.context_id
    )
    contexts = {candidate.total.context_id for candidate in document_candidates}
    if len(contexts) != 1:
        return None
    return max(document_candidates, key=lambda candidate: candidate.rank).total


def select_document_totals(
    candidates: tuple[DocumentTotalCandidate, ...] | list[DocumentTotalCandidate],
) -> tuple[DocumentTotal, ...]:
    selected: list[DocumentTotalCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (item.total.page_number, item.vertical_position, -item.label_priority),
    ):
        key = (
            candidate.total.page_number,
            candidate.total.label,
            candidate.total.amount,
            candidate.total.kind,
            candidate.total.scope,
            candidate.total.evidence.token_ids,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    return tuple(candidate.total for candidate in selected)
