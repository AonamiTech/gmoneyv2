from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from difflib import SequenceMatcher
from statistics import median

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import (
    EvidenceRef,
    RowRole,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    TableType,
)
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow
from gmoney.extraction.typed_values import parse_decimal

HEADER_TERMS: dict[str, tuple[str, ...]] = {
    "serial": ("sr no", "s no", "serial no", "#"),
    "description": (
        "description",
        "particular",
        "particulars",
        "service name",
        "item name",
        "itemname",
        "product name",
        "productname",
        "procedure name",
        "test name",
        "investigation",
        "services",
        "department",
    ),
    "service_date": (
        "service date",
        "bill date",
        "billdate",
        "date time",
        "date/time",
        "date",
    ),
    "request_no": ("request no", "requisition", "ref no", "bill number"),
    "service_code": ("service code", "code"),
    "hsn_code": ("hsn code", "hsn", "sac code", "sac"),
    "company": ("company", "comp"),
    "batch": ("batch no", "bath no", "batch"),
    "expiry": ("expiry", "exp"),
    "quantity": ("quantity", "qty", "nos", "unit days", "units"),
    "rate": (
        "amount rs",
        "unit price",
        "unitprice",
        "unit rate",
        "unitrate",
        "rate",
    ),
    "gross_amount": ("gross amount", "service amount", "service amt"),
    "discount": ("discount", "disc amt", "disc"),
    "amount": (
        "net amount",
        "total amount",
        "line total",
        "amount",
        "amoun",
        "amour",
        "total",
    ),
}

DATE_VALUE = (
    r"\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}"
    r"|[-\s](?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[-\s]\d{2,4})"
)
DATE_PREFIX = re.compile(
    rf"^\s*(?P<date>{DATE_VALUE})(?:\s*\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?\s*[-:]?\s*",
    re.IGNORECASE,
)
DATE_SPAN = re.compile(
    rf"{DATE_VALUE}(?:\s*\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?",
    re.IGNORECASE,
)
DATE_RANGE_SUFFIX = re.compile(
    rf"\s*-\s*{DATE_VALUE}(?:\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?"
    rf"\s+to\s+(?:{DATE_VALUE}|\d{{1,2}})(?:\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?\s*$",
    re.IGNORECASE,
)
REQUEST_PREFIX = re.compile(r"^[A-Z][A-Z0-9-]{2,}/[A-Z0-9-]+\s*", re.IGNORECASE)
BATCH_SUFFIX = re.compile(
    r"\s*(?:\[?\s*(?:B\.?\s*No|Batch|Exp(?:iry)?\s*Date)\s*[:.-].*)$",
    re.IGNORECASE,
)
LEADING_BATCH_FRAGMENT = re.compile(
    r"^\s*\[?\s*(?:B\.?\s*No|Batch|Exp(?:iry)?\s*Date)\s*[:.-][^\]]*\]\s*",
    re.IGNORECASE,
)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9#]+", " ", str(value or "").casefold()).strip()


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


def _cell_reading_order(
    tokens: tuple[OcrToken, ...] | list[OcrToken],
) -> tuple[OcrToken, ...]:
    """Read a cell by local text bands instead of globally sorting it by x."""
    if not tokens:
        return ()
    tolerance = max(2.0, median(_height(token) for token in tokens) * 0.35)
    bands: list[list[OcrToken]] = []
    for token in sorted(tokens, key=lambda item: (_center_y(item), _center_x(item))):
        band = next(
            (
                candidate
                for candidate in reversed(bands)
                if abs(_center_y(token) - median(_center_y(item) for item in candidate))
                <= tolerance
            ),
            None,
        )
        if band is None:
            bands.append([token])
        else:
            band.append(token)
    return tuple(
        token
        for band in bands
        for token in sorted(band, key=lambda item: (_center_x(item), _center_y(item)))
    )


def tokens_in_box(
    tokens: tuple[OcrToken, ...],
    box: tuple[int, int, int, int],
) -> tuple[OcrToken, ...]:
    left, top, right, bottom = box
    return tuple(
        token
        for token in tokens
        if left <= _center_x(token) <= right and top <= _center_y(token) <= bottom
    )


@dataclass(frozen=True)
class OcrLine:
    tokens: tuple[OcrToken, ...]

    @property
    def center_y(self) -> float:
        return median(_center_y(token) for token in self.tokens)

    @property
    def text(self) -> str:
        return " ".join(token.text.strip() for token in self.tokens if token.text.strip())


@dataclass(frozen=True)
class TableSchemaState:
    source_page: int
    source_table: str
    table_type: TableType
    column_centers: dict[str, float]
    confidence: float
    header_token_ids: tuple[str, ...]
    orientation: str = "upright"


@dataclass(frozen=True)
class ReconstructionResult:
    rows: tuple[AlignedLedgerRow, ...]
    schema: TableSchemaState | None
    diagnostics: dict[str, object]
    source_tables: tuple[SourceTable, ...] = ()


@dataclass(frozen=True)
class HeaderBlock:
    start: int
    end: int
    roles: dict[str, OcrToken]


def _lines(tokens: tuple[OcrToken, ...]) -> tuple[OcrLine, ...]:
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
    return tuple(OcrLine(tuple(sorted(group, key=_center_x))) for group in grouped)


def _normalize_orientation(
    tokens: tuple[OcrToken, ...],
    box: tuple[int, int, int, int],
) -> tuple[tuple[OcrToken, ...], tuple[int, int, int, int], str, float]:
    """Rotate sideways OCR geometry into reading order without changing token IDs."""
    if not tokens:
        return tokens, box, "upright", 0.0

    def deskew(
        candidate: tuple[OcrToken, ...],
        *,
        normalize_origin: bool,
    ) -> tuple[tuple[OcrToken, ...], float]:
        slopes: list[float] = []
        for token in candidate:
            points = token.polygon.points
            edges = [(points[index], points[(index + 1) % 4]) for index in range(4)]
            start, end = max(
                edges,
                key=lambda edge: (edge[1].x - edge[0].x) ** 2 + (edge[1].y - edge[0].y) ** 2,
            )
            if abs(end.x - start.x) <= 5:
                continue
            slope = (end.y - start.y) / (end.x - start.x)
            if abs(slope) <= 0.25:
                slopes.append(slope)
        slope = median(slopes) if slopes else 0.0
        anchor_x = min(point.x for token in candidate for point in token.polygon.points)
        minimum_y = min(
            point.y - slope * (point.x - anchor_x)
            for token in candidate
            for point in token.polygon.points
        )
        y_offset = -minimum_y if normalize_origin else max(0.0, -minimum_y)
        transformed = tuple(
            token.model_copy(
                update={
                    "polygon": Polygon(
                        points=tuple(
                            Point(
                                x=point.x,
                                y=(point.y - slope * (point.x - anchor_x) + y_offset),
                            )
                            for point in token.polygon.points
                        )
                    )
                }
            )
            for token in candidate
        )
        return transformed, slope

    vertical_fraction = sum(
        (_bounds(token)[3] - _bounds(token)[1])
        > 1.3 * max(1.0, _bounds(token)[2] - _bounds(token)[0])
        for token in tokens
    ) / len(tokens)
    if vertical_fraction < 0.65:
        deskewed, slope = deskew(tokens, normalize_origin=False)
        return deskewed, box, "upright", slope

    points = [point for token in tokens for point in token.polygon.points]
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)

    def rotate(clockwise: bool) -> tuple[OcrToken, ...]:
        return tuple(
            token.model_copy(
                update={
                    "polygon": Polygon(
                        points=tuple(
                            (
                                Point(x=max_y - point.y, y=point.x - min_x)
                                if clockwise
                                else Point(x=point.y - min_y, y=max_x - point.x)
                            )
                            for point in token.polygon.points
                        )
                    )
                }
            )
            for token in tokens
        )

    width = max_y - min_y

    def reading_order_score(candidate: tuple[OcrToken, ...]) -> tuple[int, int]:
        lines = _lines(candidate)
        centers = _stable_numeric_centers(lines, 0, width)
        if not centers:
            return (0, 0)
        amount_center = centers[-1]
        currency_lane = 0
        for line in lines:
            pair = _closest_numeric(_numeric_tokens(line), amount_center, 0, width, set())
            if pair is None:
                continue
            token, _ = pair
            relative_x = _center_x(token) / width
            if abs(relative_x - amount_center) <= 0.06 and re.search(
                r"\d[\d,]*\.\d{2}\b", token.text
            ):
                currency_lane += 1
        header_strength = max((len(_header_roles(line)) for line in lines), default=0)
        return (currency_lane, header_strength)

    clockwise_tokens, clockwise_slope = deskew(rotate(True), normalize_origin=True)
    counterclockwise_tokens, counterclockwise_slope = deskew(rotate(False), normalize_origin=True)
    if reading_order_score(clockwise_tokens) >= reading_order_score(counterclockwise_tokens):
        rotated = clockwise_tokens
        orientation = "clockwise_90"
        slope = clockwise_slope
    else:
        rotated = counterclockwise_tokens
        orientation = "counterclockwise_90"
        slope = counterclockwise_slope
    normalized_height = max(point.y for token in rotated for point in token.polygon.points)
    return rotated, (0, 0, round(width), round(normalized_height)), orientation, slope


def _virtual_horizontal_token(
    token: OcrToken,
    start: int,
    end: int,
    text_length: int,
) -> OcrToken:
    left, top, right, bottom = _bounds(token)
    span_left = left + (right - left) * start / max(1, text_length)
    span_right = left + (right - left) * end / max(1, text_length)
    return token.model_copy(
        update={
            "polygon": Polygon(
                points=(
                    Point(x=span_left, y=top),
                    Point(x=span_right, y=top),
                    Point(x=span_right, y=bottom),
                    Point(x=span_left, y=bottom),
                )
            )
        }
    )


def _header_roles(line: OcrLine) -> dict[str, OcrToken]:
    roles: dict[str, OcrToken] = {}
    for token in line.tokens:
        normalized = _normalize(token.text)
        matches: dict[str, tuple[int, int, int]] = {}
        for role, terms in HEADER_TERMS.items():
            for term in terms:
                normalized_term = _normalize(term)
                if not normalized_term:
                    continue
                if len(normalized_term) <= 4 and " " not in normalized_term:
                    match = re.search(
                        rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
                        normalized,
                    )
                    start = match.start() if match else -1
                else:
                    start = normalized.find(normalized_term)
                if normalized_term and start >= 0:
                    candidate = (len(normalized_term), start, start + len(normalized_term))
                    if role not in matches or candidate > matches[role]:
                        matches[role] = candidate
        compound = len(matches) > 1
        for role, (_, start, end) in matches.items():
            if role == "service_code" and "hsn_code" in matches:
                continue
            role_token = (
                _virtual_horizontal_token(token, start, end, len(normalized)) if compound else token
            )
            if role != "amount" or role not in roles or _center_x(token) > _center_x(roles[role]):
                roles[role] = role_token
    rate = roles.get("rate")
    amount = roles.get("amount")
    if rate is not None and amount is not None and rate.token_id == amount.token_id:
        roles.pop("rate")
    return roles


def _is_header_token(token: OcrToken) -> bool:
    normalized = _normalize(token.text)
    if any(normalized == _normalize(term) for terms in HEADER_TERMS.values() for term in terms):
        return True
    if len(_header_roles(OcrLine((token,)))) > 1:
        return True
    if " " not in normalized:
        return False
    terms = sorted(
        {
            _normalize(term)
            for role_terms in HEADER_TERMS.values()
            for term in role_terms
            if _normalize(term)
        },
        key=len,
        reverse=True,
    )
    remainder = normalized
    matched = False
    for term in terms:
        if term in remainder:
            remainder = remainder.replace(term, " ")
            matched = True
    return matched and not re.sub(r"[^a-z0-9]+", "", remainder)


def _valid_header(roles: dict[str, OcrToken]) -> bool:
    return "description" in roles and bool(
        {"amount", "rate", "gross_amount", "quantity"} & roles.keys()
    )


def _valid_header_line(line: OcrLine, roles: dict[str, OcrToken]) -> bool:
    """Reject ledger totals that happen to contain header vocabulary."""
    normalized = _normalize(" ".join(token.text for token in line.tokens))
    return _valid_header(roles) and not normalized.startswith(
        ("total for", "sub total", "subtotal", "grand total")
    )


def _merge_header_roles(
    current: dict[str, OcrToken], incoming: dict[str, OcrToken]
) -> dict[str, OcrToken]:
    merged = dict(current)
    for role, token in incoming.items():
        existing = merged.get(role)
        if existing is None or (role == "amount" and _center_x(token) > _center_x(existing)):
            merged[role] = token
    return merged


def _header_blocks(lines: tuple[OcrLine, ...]) -> tuple[HeaderBlock, ...]:
    """Recognize headers split by OCR over as many as three adjacent lines."""
    fragment_words = {
        "amount",
        "bill",
        "code",
        "date",
        "description",
        "disc",
        "discount",
        "gross",
        "hsn",
        "item",
        "name",
        "net",
        "no",
        "particulars",
        "price",
        "product",
        "procedure",
        "qty",
        "quantity",
        "rate",
        "ref",
        "request",
        "sac",
        "serial",
        "service",
        "sr",
        "test",
        "time",
        "total",
        "unit",
    }
    candidates: list[HeaderBlock] = []
    for start in range(len(lines)):
        roles: dict[str, OcrToken] = {}
        start_words = set(re.findall(r"[a-z]+", _normalize(lines[start].text)))
        if not _header_roles(lines[start]) and not (start_words and start_words <= fragment_words):
            continue
        for end in range(start, min(len(lines), start + 3)):
            roles = _merge_header_roles(roles, _header_roles(lines[end]))
            if not _valid_header(roles):
                continue
            normalized = _normalize(" ".join(line.text for line in lines[start : end + 1]))
            if normalized.startswith(("total for", "sub total", "subtotal", "grand total")):
                continue
            candidates.append(HeaderBlock(start=start, end=end, roles=dict(roles)))
            break
    selected: list[HeaderBlock] = []
    for candidate in candidates:
        if selected and candidate.start <= selected[-1].end:
            continue
        selected.append(candidate)
    return tuple(selected)


def _column_centers(
    roles: dict[str, OcrToken],
    *,
    left: float,
    width: float,
) -> dict[str, float]:
    return {role: (_center_x(token) - left) / width for role, token in roles.items()}


def _printed_header_centers(
    lines: tuple[OcrLine, ...],
    block: HeaderBlock,
    *,
    left: float,
    width: float,
) -> tuple[float, ...]:
    header_tokens = tuple(
        token
        for line in lines[block.start : block.end + 1]
        for token in line.tokens
        if token.text.strip()
    )
    recognized_ids = {token.token_id for token in block.roles.values()}
    centers = [_center_x(token) for token in block.roles.values()]
    centers.extend(
        _center_x(token)
        for token in header_tokens
        if token.token_id not in recognized_ids
        and parse_decimal(token.text) is None
        and not re.fullmatch(r"\d+[.)]?", token.text.strip())
    )
    return tuple(sorted((center - left) / width for center in centers))


SOURCE_CANONICAL_FIELDS: dict[str, str | None] = {
    "serial": None,
    "description": "description",
    "service_date": "service_date_raw",
    "request_no": "request_no",
    "service_code": "service_code",
    "hsn_code": "hsn_code",
    "company": None,
    "batch": None,
    "expiry": None,
    "quantity": "quantity",
    "rate": "unit_price",
    "gross_amount": "gross_amount",
    "discount": "discount",
    "amount": "net_amount",
}


def _source_label(role: str, token: OcrToken, shared_token: bool) -> str:
    raw = re.sub(r"\s+", " ", token.text).strip()
    normalized = _normalize(raw)
    if not shared_token:
        return raw
    if role == "rate" and "amount rs" in normalized:
        return "Amount Rs."
    if role == "quantity" and "unit days" in normalized:
        return "Unit/Days"
    return {
        "serial": "Sr. No.",
        "description": "Particular",
        "service_date": "Date",
        "request_no": "Request No.",
        "service_code": "Service Code",
        "hsn_code": "HSN Code",
        "company": "Company",
        "batch": "Batch",
        "expiry": "Expiry",
        "quantity": "Quantity",
        "rate": "Rate",
        "gross_amount": "Gross Amount",
        "discount": "Discount",
        "amount": "Amount",
    }[role]


def _source_evidence(
    tokens: tuple[OcrToken, ...],
    original_by_id: dict[str, OcrToken],
    table_id: str,
) -> tuple[EvidenceRef, ...]:
    originals = tuple(
        original_by_id[token.token_id] for token in tokens if token.token_id in original_by_id
    )
    if not originals:
        return ()
    boxes = [_bounds(token) for token in originals]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    first = originals[0]
    return (
        EvidenceRef(
            page_number=first.page_number,
            table_id=table_id,
            polygon=Polygon(
                points=(
                    Point(x=left, y=top),
                    Point(x=right, y=top),
                    Point(x=right, y=bottom),
                    Point(x=left, y=bottom),
                )
            ),
            artifact_sha256=first.artifact_sha256,
            token_ids=tuple(dict.fromkeys(token.token_id for token in originals)),
        ),
    )


def _source_columns(
    lines: tuple[OcrLine, ...],
    block: HeaderBlock,
    original_by_id: dict[str, OcrToken],
    table_id: str,
) -> tuple[tuple[SourceColumn, ...], tuple[float, ...]]:
    header_tokens = tuple(
        token
        for line in lines[block.start : block.end + 1]
        for token in line.tokens
        if token.text.strip()
    )
    role_token_counts: dict[str, int] = {}
    for token in block.roles.values():
        role_token_counts[token.token_id] = role_token_counts.get(token.token_id, 0) + 1

    entries: list[tuple[float, str, str | None, OcrToken]] = []
    recognized_ids: set[str] = set()
    for role, token in block.roles.items():
        recognized_ids.add(token.token_id)
        entries.append(
            (
                _center_x(token),
                _source_label(role, token, role_token_counts[token.token_id] > 1),
                SOURCE_CANONICAL_FIELDS.get(role),
                token,
            )
        )
    for token in header_tokens:
        if token.token_id not in recognized_ids:
            if block.roles and (
                parse_decimal(token.text) is not None
                or re.fullmatch(r"\d+[.)]?", token.text.strip())
            ):
                continue
            entries.append((_center_x(token), token.text.strip(), None, token))
    entries.sort(key=lambda entry: entry[0])

    columns: list[SourceColumn] = []
    centers: list[float] = []
    for order, (center, label, canonical_field, token) in enumerate(entries):
        columns.append(
            SourceColumn(
                id=f"c{order + 1}",
                label=label,
                order=order,
                canonical_field=canonical_field,
                evidence=_source_evidence((token,), original_by_id, table_id),
            )
        )
        centers.append(center)
    return tuple(columns), tuple(centers)


def _source_rows(
    lines: tuple[OcrLine, ...],
    *,
    start: int,
    end: int,
    source_table_id: str,
    columns: tuple[SourceColumn, ...],
    centers: tuple[float, ...],
    original_by_id: dict[str, OcrToken],
    table_id: str,
) -> tuple[SourceRow, ...]:
    if not columns:
        return ()
    output: list[SourceRow] = []
    description_index = next(
        (
            index
            for index, column in enumerate(columns)
            if column.canonical_field == "description"
        ),
        None,
    )
    for line in lines[start:end]:
        buckets: list[list[OcrToken]] = [[] for _ in columns]
        for token in line.tokens:
            column = min(
                range(len(centers)),
                key=lambda index: abs(_center_x(token) - centers[index]),
            )
            buckets[column].append(token)
        if not any(buckets):
            continue
        cells: list[SourceCell] = []
        for column, bucket in zip(columns, buckets, strict=True):
            ordered = _cell_reading_order(bucket)
            raw_value = " ".join(token.text.strip() for token in ordered if token.text.strip())
            cells.append(
                SourceCell(
                    column_id=column.id,
                    raw_value=raw_value or None,
                    evidence=_source_evidence(ordered, original_by_id, table_id),
                    validation_flags=(() if raw_value else ("empty_cell",)),
                )
            )
        if not any(cell.raw_value for cell in cells):
            continue
        populated = tuple(index for index, cell in enumerate(cells) if cell.raw_value)
        if (
            output
            and description_index is not None
            and populated == (description_index,)
        ):
            previous_cells = list(output[-1].cells)
            previous_description = previous_cells[description_index]
            continuation = cells[description_index]
            if (
                previous_description.raw_value
                and continuation.raw_value
                and not _is_payment_footer_description(continuation.raw_value)
                and _is_description_continuation(
                    continuation.raw_value,
                    previous_description.raw_value,
                )
            ):
                previous_cells[description_index] = previous_description.model_copy(
                    update={
                        "raw_value": (
                            f"{previous_description.raw_value} {continuation.raw_value}"
                        ),
                        "evidence": (
                            *previous_description.evidence,
                            *continuation.evidence,
                        ),
                    }
                )
                output[-1] = output[-1].model_copy(update={"cells": tuple(previous_cells)})
                continue
        output.append(
            SourceRow(
                id=f"{source_table_id}-r{len(output) + 1}",
                order=len(output),
                cells=tuple(cells),
            )
        )
    return tuple(output)


def _build_source_tables(
    lines: tuple[OcrLine, ...],
    *,
    primary: HeaderBlock,
    repeated: tuple[HeaderBlock, ...],
    original_by_id: dict[str, OcrToken],
    page_number: int,
    table_id: str,
    table_type: TableType,
) -> tuple[SourceTable, ...]:
    blocks = (primary, *repeated)
    output: list[SourceTable] = []
    for segment, block in enumerate(blocks, start=1):
        source_table_id = f"{table_id}-s{segment}"
        columns, centers = _source_columns(lines, block, original_by_id, table_id)
        next_start = blocks[segment].start if segment < len(blocks) else len(lines)
        rows = _source_rows(
            lines,
            start=block.end + 1,
            end=next_start,
            source_table_id=source_table_id,
            columns=columns,
            centers=centers,
            original_by_id=original_by_id,
            table_id=table_id,
        )
        if not columns or not rows:
            continue
        output.append(
            SourceTable(
                id=source_table_id,
                page_number=page_number,
                table_id=table_id,
                table_type=table_type,
                columns=columns,
                rows=rows,
                validation_flags=(
                    ("unmapped_columns",)
                    if any(
                        column.canonical_field is None
                        and _normalize(column.label)
                        not in {"sr no", "sr n", "s no", "serial no", "no", "#"}
                        for column in columns
                    )
                    else ()
                ),
            )
        )
    return tuple(output)


def _raw_source_header(lines: tuple[OcrLine, ...], width: float) -> HeaderBlock | None:
    """Find a non-canonical header from its geometry and following numeric rows."""
    for index, line in enumerate(lines[:-1]):
        header_tokens = tuple(token for token in line.tokens if token.text.strip())
        if len(header_tokens) < 2 or any(
            parse_decimal(token.text) is not None for token in header_tokens
        ):
            continue
        following = lines[index + 1 : min(len(lines), index + 4)]
        candidates = [
            candidate
            for candidate in following
            if len(candidate.tokens) >= 2 and len(_numeric_tokens(candidate)) >= 1
        ]
        if not candidates:
            continue
        aligned = max(
            sum(
                min(abs(_center_x(header) - _center_x(value)) for value in candidate.tokens)
                <= width * 0.08
                for header in header_tokens
            )
            for candidate in candidates
        )
        if aligned >= 2:
            return HeaderBlock(start=index, end=index, roles={})
    return None


def _synthetic_source_table(
    lines: tuple[OcrLine, ...],
    *,
    original_by_id: dict[str, OcrToken],
    page_number: int,
    table_id: str,
    table_type: TableType,
    width: float,
) -> tuple[SourceTable, ...]:
    indexed_lines = tuple(
        (index, line)
        for index, line in enumerate(lines)
        if any(token.text.strip() for token in line.tokens)
    )
    candidate_lines = tuple(
        (index, line)
        for index, line in indexed_lines
        if len(tuple(token for token in line.tokens if token.text.strip())) >= 2
        and _numeric_tokens(line)
    )
    if not candidate_lines:
        return ()
    first_data_index = candidate_lines[0][0]
    data_lines = tuple(line for index, line in indexed_lines if index >= first_data_index)
    centers = [
        _center_x(token)
        for token in candidate_lines[0][1].tokens
        if token.text.strip()
    ]
    if len(centers) < 2:
        return ()
    for _, line in candidate_lines[1:]:
        for token in line.tokens:
            if not token.text.strip():
                continue
            center = _center_x(token)
            nearest = min(range(len(centers)), key=lambda index: abs(centers[index] - center))
            if abs(centers[nearest] - center) > width * 0.08:
                centers.append(center)
    centers.sort()
    columns = tuple(
        SourceColumn(
            id=f"c{index + 1}",
            label=f"Column {index + 1}",
            order=index,
            validation_flags=("synthetic_header",),
        )
        for index in range(len(centers))
    )
    source_table_id = f"{table_id}-s1"
    rows = _source_rows(
        data_lines,
        start=0,
        end=len(data_lines),
        source_table_id=source_table_id,
        columns=columns,
        centers=tuple(centers),
        original_by_id=original_by_id,
        table_id=table_id,
    )
    if not rows:
        return ()
    return (
        SourceTable(
            id=source_table_id,
            page_number=page_number,
            table_id=table_id,
            table_type=table_type,
            columns=columns,
            rows=rows,
            validation_flags=("synthetic_headers", "unmapped_columns"),
        ),
    )


def _description_lane(
    centers: dict[str, float],
    stable_centers: tuple[float, ...],
    table_type: TableType,
) -> tuple[float, float | None]:
    description_center = centers.get("description")
    financial_centers = tuple(
        center
        for center in stable_centers
        if (center >= 0.55 if description_center is None else center > description_center + 0.04)
    )
    labeled_field_centers = tuple(
        centers[role]
        for role in (
            "request_no",
            "service_code",
            "hsn_code",
            "service_date",
            "rate",
            "quantity",
            "gross_amount",
            "discount",
            "amount",
        )
        if role in centers
        and (description_center is None or centers[role] > description_center + 0.04)
    )
    first_numeric_center = (
        min(labeled_field_centers)
        if labeled_field_centers
        else (min(financial_centers) if financial_centers else centers.get("amount"))
    )
    description_min = 0.0
    description_max = first_numeric_center
    if table_type is TableType.PHARMACY:
        description_min = min(stable_centers, default=0.02) + 0.015
        description_max = min(
            centers.get("company", 1.0) - 0.03,
            centers.get("batch", 1.0) - 0.05,
            first_numeric_center or 1.0,
        )
    return description_min, description_max


def _description_cell_boundaries(
    centers: dict[str, float],
    printed_header_centers: tuple[float, ...],
) -> tuple[float | None, float | None]:
    description_center = centers.get("description")
    previous_printed_center = max(
        (
            center
            for center in printed_header_centers
            if description_center is not None and center < description_center - 0.04
        ),
        default=None,
    )
    next_printed_center = min(
        (
            center
            for center in printed_header_centers
            if description_center is not None and center > description_center + 0.04
        ),
        default=None,
    )
    return (
        (
            (previous_printed_center + description_center) / 2
            if description_center is not None and previous_printed_center is not None
            else None
        ),
        (
            (description_center + next_printed_center) / 2
            if description_center is not None and next_printed_center is not None
            else None
        ),
    )


def _numeric_tokens(line: OcrLine) -> list[tuple[OcrToken, Decimal]]:
    output: list[tuple[OcrToken, Decimal]] = []
    for token in line.tokens:
        value = parse_decimal(token.text)
        if value is not None:
            output.append((token, value))
            continue
        expiry_quantity = re.search(
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/\d{4}\s*"
            r"(?P<quantity>[+-]?\d+(?:\.\d{1,4})?)\s*$",
            token.text,
            re.IGNORECASE,
        )
        if expiry_quantity is not None:
            parsed = parse_decimal(expiry_quantity.group("quantity"))
            if parsed is not None:
                output.append(
                    (
                        _virtual_horizontal_token(
                            token,
                            expiry_quantity.start("quantity"),
                            expiry_quantity.end("quantity"),
                            len(token.text),
                        ),
                        parsed,
                    )
                )
                continue
        non_currency_text = re.sub(
            r"\b(?:inr|rs|rupees?)\.?\b",
            "",
            token.text,
            flags=re.IGNORECASE,
        )
        if re.search(r"[A-Za-z]", non_currency_text):
            continue
        date_spans = tuple((match.start(), match.end()) for match in DATE_SPAN.finditer(token.text))
        concatenated = re.fullmatch(
            r"(?P<first>[+-]?\d[\d,]*\.\d{2})(?P<second>\d+(?:\.\d{1,4})?)",
            token.text.strip(),
        )
        if concatenated is not None:
            spans = tuple(
                (
                    concatenated.group(name),
                    concatenated.start(name),
                    concatenated.end(name),
                )
                for name in ("first", "second")
            )
        else:
            currency_parts = tuple(re.finditer(r"[+-]?\d[\d,]*\.\d{2}", token.text))
            parts = (
                currency_parts
                if len(currency_parts) >= 2
                else tuple(re.finditer(r"[+-]?\d[\d,]*(?:\.\d{1,4})?", token.text))
            )
            spans = tuple((part.group(), part.start(), part.end()) for part in parts)
        if date_spans:
            spans = tuple(
                span
                for span in spans
                if not any(span[1] < end and span[2] > start for start, end in date_spans)
            )
        if len(spans) < 2:
            continue
        left, top, right, bottom = _bounds(token)
        text_length = max(1, len(token.text))
        for text, start, end in spans:
            parsed = parse_decimal(text)
            if parsed is None:
                continue
            # OCR occasionally merges adjacent numeric cells (for example,
            # discount and line total) into one token. Approximate each text
            # span's horizontal geometry so the rightmost printed value keeps
            # its column identity. The original token ID is deliberately kept
            # and resolves back to the untouched source polygon for evidence.
            span_left = left + (right - left) * start / text_length
            span_right = left + (right - left) * end / text_length
            virtual = token.model_copy(
                update={
                    "polygon": Polygon(
                        points=(
                            Point(x=span_left, y=top),
                            Point(x=span_right, y=top),
                            Point(x=span_right, y=bottom),
                            Point(x=span_left, y=bottom),
                        )
                    )
                }
            )
            output.append((virtual, parsed))
    return output


def _stable_numeric_centers(
    lines: tuple[OcrLine, ...], left: float, width: float
) -> tuple[float, ...]:
    centers: list[float] = []
    for line in lines:
        centers.extend((_center_x(token) - left) / width for token, _ in _numeric_tokens(line))
    clusters: list[list[float]] = []
    for center in sorted(centers):
        matching = next(
            (cluster for cluster in clusters if abs(median(cluster) - center) <= 0.025), None
        )
        if matching is None:
            clusters.append([center])
        else:
            matching.append(center)
    minimum_support = max(2, round(len(lines) * 0.15))
    return tuple(median(cluster) for cluster in clusters if len(cluster) >= minimum_support)


def _classify_table(
    text: str,
    has_serial: bool,
    row_count: int,
    *,
    header_text: str = "",
    has_ledger_header: bool = False,
    zero_tail_summary: bool = False,
) -> TableType:
    normalized = _normalize(text)
    normalized_header = _normalize(header_text)
    metadata_hits = sum(
        term in normalized
        for term in (
            "uhid",
            "ip no",
            "original bill no",
            "contact no",
            "date of admission",
            "d o a",
            "sponsor",
            "billing category",
            "mobile no",
            "uid",
            "aadhaar",
            "aadhar",
            "policy no",
            "policy number",
            "pin code",
            "pincode",
            "bill no",
            "bill date time",
            "relation",
            "admitting doctor",
            "discharge",
        )
    )
    if metadata_hits >= 2 and row_count <= 24 and not has_serial and not has_ledger_header:
        return TableType.METADATA
    if "break up of charges" in normalized or "package break up" in normalized:
        return TableType.PACKAGE_SUMMARY
    if row_count <= 24 and all(
        term in normalized
        for term in (
            "service name",
            "bill amount",
            "discount amount",
            "total amount",
            "total bill amount",
        )
    ):
        return TableType.CATEGORY_SUMMARY
    if (
        all(
            term in normalized
            for term in ("service name", "patient amount", "company amount", "total amount")
        )
        or all(term in normalized for term in ("service name", "patient", "company", "grand total"))
        or zero_tail_summary
        or any(
            term in normalized
            for term in ("bill summary", "summary of charges", "department amount")
        )
        or ("department" in normalized and "amount" in normalized and row_count <= 30)
    ):
        return TableType.CATEGORY_SUMMARY
    if any(
        term in normalized
        for term in (
            "batch",
            "bath no",
            "expiry",
            "cash memo",
            "medical general stores",
        )
    ) or ("m r p" in normalized and "exp" in normalized):
        return TableType.PHARMACY
    if (
        row_count <= 24
        and not has_ledger_header
        and any(
            term in normalized
            for term in (
                "payer received",
                "payer receivable",
                "patient received",
                "patient balance",
            )
        )
    ):
        return TableType.PAYMENT
    if any(
        term in normalized
        for term in ("payment mode", "amount paid", "amount received", "receipt ref")
    ):
        return TableType.PAYMENT
    laboratory_terms = ("test name", "pathology", "laboratory", "investigation")
    if any(term in normalized_header for term in laboratory_terms) or sum(
        normalized.count(term) for term in laboratory_terms
    ) >= 2:
        return TableType.LABORATORY
    if has_serial and row_count <= 20 and "particular" in normalized:
        return TableType.CATEGORY_SUMMARY
    return TableType.ITEM_LEDGER


def row_category(description: str, table_type: TableType) -> str | None:
    normalized = _normalize(description)
    if "package" in normalized:
        return "package"
    if table_type is TableType.PHARMACY or any(
        term in normalized for term in ("pharmacy", "medicine", "consumable")
    ):
        return "pharmacy"
    if table_type is TableType.LABORATORY or any(
        term in normalized for term in ("laboratory", "pathology", "investigation", "lab charges")
    ):
        return "laboratory"
    mappings = {
        "bed": ("bed charge", "room charge", "accommodation"),
        "nursing": ("nursing",),
        "doctor": ("doctor", "surgeon", "consultation", "round charge", "visit charge"),
        "ot": ("o t charge", "ot charge", "theatre"),
        "anaesthesia": ("anaesth", "anesth"),
        "package": ("package",),
        "service": ("service",),
    }
    return next(
        (
            category
            for category, terms in mappings.items()
            if any(term in normalized for term in terms)
        ),
        None,
    )


def _clean_description(text: str) -> tuple[str, str | None, str | None]:
    raw = re.sub(r"\s+", " ", text).strip(" -:")
    date_match = DATE_PREFIX.match(raw)
    service_date = date_match.group("date").strip(" -:") if date_match else None
    if date_match:
        raw = raw[date_match.end() :]
    request_match = REQUEST_PREFIX.match(raw)
    request_no = request_match.group(0).strip() if request_match else None
    if request_match:
        raw = raw[request_match.end() :].lstrip(" -:")
    raw = LEADING_BATCH_FRAGMENT.sub("", raw)
    raw = BATCH_SUFFIX.sub("", raw)
    raw = DATE_RANGE_SUFFIX.sub("", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" -:[]")
    return raw, service_date, request_no


def _is_metadata_description(description: str) -> bool:
    normalized = _normalize(description)
    number_words = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "twenty",
        "thirty",
        "fourty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "lakh",
        "crore",
    }
    return (
        any(
            term in normalized
            for term in (
                "uhid",
                "original bill no",
                "mobile no bill no",
                "bill date time",
                "contact no d o a",
                "sponsor billing category",
                "admitting doctor",
                "relation self",
                "amount in words",
                "in words",
                "print date time",
                "receipt details",
                "doctor appointments",
                "free home sample",
                "book appointment",
                "call for appointment",
                "customer care",
                "helpline",
                "website",
                "email",
                "e mail",
                "phone no",
                "mobile no",
                "uid",
                "aadhaar",
                "aadhar",
                "policy no",
                "policy number",
                "pin code",
                "pincode",
                "credit fro",
            )
        )
        or normalized.startswith("rupees in")
        or normalized
        in {
            "cgst",
            "sgst",
            "net bill",
            "upi",
            "expdate",
            "exp date",
            "uid aadhaar",
            "uid aadhar",
            "policy no",
            "policy number",
            "pin code",
            "pincode",
        }
        or normalized.endswith(" admission")
        or normalized.endswith(" discharge")
        or ("date time" in normalized and "page" in normalized)
        or normalized.endswith("only cgst")
        or normalized.endswith("only sgst")
        or (normalized.endswith(" only") and bool(number_words.intersection(normalized.split())))
    )


def _is_total_description(normalized: str) -> bool:
    return (
        normalized in {"total", "totals", "tota", "sub total", "subtotal"}
        or any(
            term in normalized
            for term in (
                "gross bill amount",
                "bill round off",
                "round off amount",
                "net bill amount",
                "total gross bill value",
                "total payable amount",
                "net patient payable amt",
                "net patient payable amount",
                "net payable",
                "gross amount",
                "amount to be recelved",
                "amount to be receive",
                "company credit limit",
            )
        )
        or normalized.startswith(
            (
                "sub total",
                "subtotal",
                "total for",
                "grand total",
                "total bill amount",
                "total discount amount",
                "net amount",
                "net tpa corporate amount",
                "bill amount",
                "amount paid",
                "amount to be received",
                "balance amount",
                "balance due",
                "payer amount",
                "payer received",
                "payer receivable",
                "patient amount",
                "patient received",
                "patient balance",
                "cgst amount",
                "sgst amount",
                "item issue total",
                "item issues total",
                "item return total",
                "item returns total",
                "patient wise total",
                "company wise total",
            )
        )
    )


def _is_structural_total_line(line: OcrLine) -> bool:
    normalized = _normalize(line.text)
    if _is_total_description(normalized):
        return True
    if not line.tokens:
        return False
    leading = _normalize(line.tokens[0].text)
    if leading not in {"total", "sub total", "subtotal"}:
        return False
    return not any(re.search(r"[a-z]", _normalize(token.text)) for token in line.tokens[1:])


def _is_description_continuation(text: str, previous: str) -> bool:
    stripped = text.strip()
    return bool(
        stripped.startswith(("(", ")", "[", "]", "/", "-"))
        or previous.count("(") > previous.count(")")
        or previous.rstrip().endswith(("/", "-", "(", "["))
    )


def _is_payment_footer_description(text: str) -> bool:
    words = _normalize(text).split()
    if not words:
        return False
    if words[0] == "amount":
        return bool({"paid", "received"} & set(words[1:]))
    subject = words[0].removesuffix("s")
    if subject not in {"payment", "receipt", "settlement"}:
        return False
    qualifiers = set(words[1:])
    return bool(
        {
            "breakup",
            "detail",
            "details",
            "history",
            "information",
            "mode",
            "status",
            "summary",
        }
        & qualifiers
    ) or {"break", "up"} <= qualifiers


def _clip_token_to_lane(
    token: OcrToken,
    minimum: float,
    maximum: float | None,
    left: float,
    width: float,
) -> OcrToken | None:
    token_left, _, token_right, _ = _bounds(token)
    relative_left = (token_left - left) / width
    relative_right = (token_right - left) / width
    effective_maximum = maximum if maximum is not None else 1.0
    if relative_right <= minimum or relative_left >= effective_maximum:
        return None
    if relative_left >= minimum and relative_right <= effective_maximum:
        return token
    span = max(1e-6, relative_right - relative_left)
    start_fraction = max(0.0, (minimum - relative_left) / span)
    end_fraction = min(1.0, (effective_maximum - relative_left) / span)
    text_length = len(token.text)
    start = min(text_length, round(text_length * start_fraction))
    end = min(text_length, round(text_length * end_fraction))
    clipped = token.text[start:end].strip(" -:[]")
    return token.model_copy(update={"text": clipped}) if clipped else None


def _closest_field_token(
    line: OcrLine,
    target: float | None,
    left: float,
    width: float,
    *,
    role: str,
) -> OcrToken | None:
    if target is None:
        return None

    def valid(token: OcrToken) -> bool:
        text = token.text.strip()
        if not text or _is_header_token(token):
            return False
        if role == "service_date":
            return DATE_SPAN.search(text) is not None
        if role == "service_code":
            return bool(re.fullmatch(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9./-]{3,30}", text))
        if role == "hsn_code":
            return DATE_SPAN.search(text) is None and bool(
                re.fullmatch(r"[A-Za-z0-9./-]{3,30}", text)
            )
        if role == "request_no":
            return bool(re.fullmatch(r"[A-Za-z0-9./-]{3,60}", text))
        return False

    candidates = [token for token in line.tokens if valid(token)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda token: abs(((_center_x(token) - left) / width) - target))
    return selected if abs(((_center_x(selected) - left) / width) - target) <= 0.08 else None


def _structured_text_fields(
    line: OcrLine,
    column_centers: dict[str, float],
    left: float,
    width: float,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    values: dict[str, str] = {}
    evidence: dict[str, tuple[str, ...]] = {}
    for role in ("service_date", "request_no", "service_code", "hsn_code"):
        token = _closest_field_token(
            line,
            column_centers.get(role),
            left,
            width,
            role=role,
        )
        if token is None:
            continue
        raw = re.sub(r"\s+", " ", token.text).strip()
        values[role] = raw
        evidence[role] = (token.token_id,)
    return values, evidence


def _closest_numeric(
    numeric: list[tuple[OcrToken, Decimal]],
    target: float | None,
    left: float,
    width: float,
    used: set[tuple[str, float, Decimal]],
) -> tuple[OcrToken, Decimal] | None:
    available = [
        (token, value) for token, value in numeric if _numeric_identity(token, value) not in used
    ]
    if not available:
        return None
    if target is None:
        return max(available, key=lambda item: _center_x(item[0]))
    return min(available, key=lambda item: abs(((_center_x(item[0]) - left) / width) - target))


def _numeric_identity(token: OcrToken, value: Decimal) -> tuple[str, float, Decimal]:
    """Identify a numeric span while retaining its original OCR evidence ID."""
    return token.token_id, round(_center_x(token), 6), value


def reconstruct_ocr_rows(
    tokens: tuple[OcrToken, ...],
    *,
    page_number: int,
    table_id: str,
    box: tuple[int, int, int, int],
    prior_schemas: tuple[TableSchemaState, ...] = (),
) -> ReconstructionResult:
    original_scoped = tokens_in_box(tokens, box)
    original_by_id = {token.token_id: token for token in original_scoped}
    scoped, geometry_box, orientation, deskew_slope = _normalize_orientation(original_scoped, box)
    left, _, right, _ = geometry_box
    width = max(1.0, right - left)
    lines = _lines(scoped)
    if not lines:
        return ReconstructionResult((), None, {"ocr_line_count": 0, "ocr_row_count": 0})

    header_candidates = [(index, _header_roles(line)) for index, line in enumerate(lines)]
    merged_headers = _header_blocks(lines)
    if merged_headers:
        primary_header = merged_headers[0]
        header_start = primary_header.start
        header_index = primary_header.end
        header_roles = primary_header.roles
    else:
        header_start = 0
        header_index, header_roles = max(
            header_candidates,
            key=lambda item: (len(item[1]), -item[0]),
        )
    header_valid = _valid_header(header_roles)
    primary_header = HeaderBlock(
        start=header_start,
        end=header_index,
        roles=header_roles,
    )
    data_lines = lines[header_index + 1 :] if header_valid else lines
    repeated_header_blocks = tuple(
        block for block in merged_headers if header_valid and block.start > header_index
    )
    if header_valid and not merged_headers:
        repeated_header_blocks = tuple(
            HeaderBlock(start=index, end=index, roles=roles)
            for index, roles in header_candidates
            if index > header_index and _valid_header_line(lines[index], roles)
        )
    repeated_header_indexes = tuple(block.start for block in repeated_header_blocks)
    repeated_header_by_line = {
        index: block
        for block in repeated_header_blocks
        for index in range(block.start, block.end + 1)
    }
    first_segment_end = repeated_header_indexes[0] if repeated_header_indexes else len(lines)
    first_segment_lines = (
        lines[header_index + 1 : first_segment_end] if header_valid else data_lines
    )
    stable_centers = _stable_numeric_centers(first_segment_lines, left, width)

    detected_centers = _column_centers(header_roles, left=left, width=width)
    if not header_valid:
        detected_centers = {
            role: center
            for role, center in detected_centers.items()
            if role in {"company", "batch", "expiry"}
        }
    column_centers: dict[str, float] = dict(detected_centers)
    inherited = False
    if not header_valid and prior_schemas:
        compatible = [
            schema
            for schema in prior_schemas
            if page_number - schema.source_page <= 2
            and schema.orientation == orientation
            and "amount" in schema.column_centers
            and (
                not stable_centers
                or min(abs(schema.column_centers["amount"] - center) for center in stable_centers)
                <= 0.05
            )
        ]
        if compatible:
            selected = max(compatible, key=lambda schema: (schema.source_page, schema.confidence))
            column_centers = dict(selected.column_centers)
            column_centers.update(detected_centers)
            inherited = True

    if stable_centers:
        labeled_amount = column_centers.get("amount")
        if labeled_amount is None:
            column_centers["amount"] = stable_centers[-1]
        else:
            nearby = min(stable_centers, key=lambda center: abs(center - labeled_amount))
            if abs(nearby - labeled_amount) <= 0.06:
                column_centers["amount"] = nearby

    summary_lines = [values for line in data_lines if len(values := _numeric_tokens(line)) >= 3]
    zero_tail_summary = bool(
        len(summary_lines) >= 5
        and sum(values[-1][1] == 0 for values in summary_lines) / len(summary_lines) >= 0.7
        and sum(any(value != 0 for _, value in values[:-1]) for values in summary_lines)
        / len(summary_lines)
        >= 0.7
    )
    table_text = " ".join(line.text for line in lines)
    header_text = " ".join(
        line.text
        for line in (lines[header_start : header_index + 1] if header_valid else ())
    )
    table_type = _classify_table(
        table_text,
        header_valid and "serial" in header_roles,
        len(data_lines),
        header_text=header_text,
        has_ledger_header=header_valid,
        zero_tail_summary=zero_tail_summary,
    )
    has_tax_columns = bool(re.search(r"\b(?:gst|tax)\b", _normalize(table_text)))
    header_ids = tuple(
        token.token_id
        for line in (lines[header_start : header_index + 1] if header_valid else ())
        for token in line.tokens
    )
    schema = None
    if "amount" in column_centers:
        schema = TableSchemaState(
            source_page=page_number,
            source_table=table_id,
            table_type=table_type,
            column_centers=column_centers,
            confidence=0.95 if header_valid else (0.8 if inherited else 0.65),
            header_token_ids=header_ids,
            orientation=orientation,
        )

    amount_center = column_centers.get("amount")
    description_header_centers = (
        _printed_header_centers(
            lines,
            primary_header,
            left=left,
            width=width,
        )
        if header_valid
        else ()
    )
    description_min, description_max = _description_lane(
        column_centers,
        stable_centers,
        table_type,
    )
    description_cell_min, description_cell_max = _description_cell_boundaries(
        column_centers,
        description_header_centers,
    )
    aligned: list[AlignedLedgerRow] = []
    aligned_description_raw: list[str] = []
    pending_description_tokens: list[OcrToken] = []
    pending_service_date: str | None = None
    pending_service_date_ids: tuple[str, ...] = ()
    current_section: str | None = None

    def pending_description_is_proven_continuation(
        continuation_tokens: list[OcrToken],
        continuation_raw: str,
    ) -> bool:
        if not aligned or _is_payment_footer_description(continuation_raw):
            return False
        description_center = column_centers.get("description")
        if description_center is None:
            return False
        continuation_left = min(
            (_bounds(token)[0] - left) / width for token in continuation_tokens
        )
        anchored = (
            description_min - 0.03
            <= continuation_left
            <= description_center + 0.08
        )
        if not anchored:
            return False
        if _is_description_continuation(
            continuation_raw,
            aligned_description_raw[-1],
        ):
            return True
        previous_tokens = [
            original_by_id[token_id]
            for token_id in aligned[-1].field_token_ids.get("description", ())
            if token_id in original_by_id
        ]
        if not previous_tokens:
            return False
        previous_left = min(
            (_bounds(token)[0] - left) / width for token in previous_tokens
        )
        return abs(continuation_left - previous_left) <= 0.03

    def extend_previous_description(
        continuation_tokens: list[OcrToken], continuation_raw: str
    ) -> None:
        previous = aligned[-1]
        combined_raw = f"{aligned_description_raw[-1]} {continuation_raw}".strip()
        combined_description, _, _ = _clean_description(combined_raw)
        continuation_ids = tuple(token.token_id for token in continuation_tokens)
        description_ids = tuple(
            dict.fromkeys((*previous.field_token_ids.get("description", ()), *continuation_ids))
        )
        continuation_boxes = [_bounds(original_by_id[token_id]) for token_id in continuation_ids]
        previous_box = previous.evidence_box
        assert previous_box is not None
        combined_box = (
            min(previous_box[0], *(item[0] for item in continuation_boxes)),
            min(previous_box[1], *(item[1] for item in continuation_boxes)),
            max(previous_box[2], *(item[2] for item in continuation_boxes)),
            max(previous_box[3], *(item[3] for item in continuation_boxes)),
        )
        aligned[-1] = replace(
            previous,
            candidate=replace(
                previous.candidate,
                description=combined_description,
            ),
            field_token_ids={
                **previous.field_token_ids,
                "description": description_ids,
            },
            evidence_token_ids=tuple(
                dict.fromkeys((*previous.evidence_token_ids, *continuation_ids))
            ),
            evidence_box=combined_box,
        )
        aligned_description_raw[-1] = combined_raw

    def append_aligned(
        *,
        candidate: CandidateLedgerRow,
        field_tokens: dict[str, tuple[str, ...]],
        raw_description: str,
    ) -> None:
        evidence_tokens = tuple(
            dict.fromkeys(token_id for ids in field_tokens.values() for token_id in ids)
        )
        selected_tokens = [original_by_id[token_id] for token_id in evidence_tokens]
        boxes = [_bounds(token) for token in selected_tokens]
        evidence_box = (
            min(item[0] for item in boxes),
            min(item[1] for item in boxes),
            max(item[2] for item in boxes),
            max(item[3] for item in boxes),
        )
        aligned.append(
            AlignedLedgerRow(
                candidate=candidate,
                field_token_ids=field_tokens,
                evidence_token_ids=evidence_tokens,
                evidence_box=evidence_box,
                grounding_ratio=1.0,
                source_routes=("ocr_spatial_graph",),
            )
        )
        aligned_description_raw.append(raw_description)

    for source_row, line in enumerate(data_lines, start=(header_index + 1 if header_valid else 0)):
        repeated_block = repeated_header_by_line.get(source_row)
        if repeated_block is not None:
            if source_row != repeated_block.end:
                continue
            repeated_roles = repeated_block.roles
            next_header = next(
                (index for index in repeated_header_indexes if index > source_row),
                len(lines),
            )
            segment_stable_centers = _stable_numeric_centers(
                lines[source_row + 1 : next_header],
                left,
                width,
            )
            column_centers = _column_centers(repeated_roles, left=left, width=width)
            if segment_stable_centers:
                labeled_amount = column_centers.get("amount")
                if labeled_amount is None:
                    column_centers["amount"] = segment_stable_centers[-1]
                else:
                    nearby = min(
                        segment_stable_centers,
                        key=lambda center: abs(center - labeled_amount),
                    )
                    if abs(nearby - labeled_amount) <= 0.06:
                        column_centers["amount"] = nearby
            stable_centers = segment_stable_centers
            amount_center = column_centers.get("amount")
            description_header_centers = _printed_header_centers(
                lines,
                repeated_block,
                left=left,
                width=width,
            )
            description_min, description_max = _description_lane(
                column_centers,
                stable_centers,
                table_type,
            )
            description_cell_min, description_cell_max = _description_cell_boundaries(
                column_centers,
                description_header_centers,
            )
            continue
        numeric = _numeric_tokens(line)
        amount_pair = _closest_numeric(numeric, amount_center, left, width, set())
        amount_token = amount_pair[0] if amount_pair else None
        amount = amount_pair[1] if amount_pair else None
        amount_in_lane = bool(
            amount_token is not None
            and (
                amount_center is None
                or abs(((_center_x(amount_token) - left) / width) - amount_center) <= 0.06
            )
        )
        structured_values, structured_evidence = _structured_text_fields(
            line,
            column_centers,
            left,
            width,
        )
        if pending_service_date:
            current_date = structured_values.get("service_date")
            same_date = bool(
                current_date
                and re.sub(r"\s+", "", current_date) == re.sub(r"\s+", "", pending_service_date)
            )
            structured_values["service_date"] = (
                pending_service_date
                if not current_date or same_date
                else f"{pending_service_date} {current_date}"
            )
            structured_evidence["service_date"] = tuple(
                dict.fromkeys(
                    (
                        *pending_service_date_ids,
                        *structured_evidence.get("service_date", ()),
                    )
                )
            )
            pending_service_date = None
            pending_service_date_ids = ()
        reserved_ids = {
            token_id for token_ids in structured_evidence.values() for token_id in token_ids
        }
        if amount_token is not None:
            reserved_ids.add(amount_token.token_id)

        line_description_tokens: list[OcrToken] = []
        for token in line.tokens:
            if token.token_id in reserved_ids:
                continue
            if _is_header_token(token):
                continue
            relative_center = (_center_x(token) - left) / width
            if description_cell_min is not None and relative_center < description_cell_min:
                continue
            if description_cell_max is not None and relative_center >= description_cell_max:
                continue
            description_token = _clip_token_to_lane(
                token,
                description_min,
                description_max - 0.03 if description_max is not None else None,
                left,
                width,
            )
            if description_token is None:
                continue
            normalized = _normalize(description_token.text)
            if parse_decimal(description_token.text) is not None:
                continue
            if normalized and not all(
                character.isdigit() for character in normalized.replace(" ", "")
            ):
                line_description_tokens.append(description_token)
        line_description_tokens = list(_cell_reading_order(line_description_tokens))

        if _is_structural_total_line(line):
            if pending_description_tokens and aligned:
                continuation_raw = " ".join(
                    token.text for token in pending_description_tokens
                )
                continuation_text, _, _ = _clean_description(continuation_raw)
                if (
                    continuation_text
                    and pending_description_is_proven_continuation(
                        pending_description_tokens,
                        continuation_raw,
                    )
                ):
                    extend_previous_description(
                        pending_description_tokens,
                        continuation_raw,
                    )
            pending_description_tokens = []
            continue

        if not amount_in_lane:
            if (
                (line_description_tokens or pending_description_tokens)
                and structured_values
                and table_type not in {TableType.METADATA, TableType.PAYMENT}
            ):
                if line_description_tokens and pending_description_tokens:
                    pending_text, _, _ = _clean_description(
                        " ".join(token.text for token in pending_description_tokens)
                    )
                    current_section = pending_text or current_section
                    pending_description_tokens = []
                elif pending_description_tokens:
                    line_description_tokens = pending_description_tokens
                    pending_description_tokens = []
                description_text = " ".join(token.text for token in line_description_tokens)
                description, embedded_date, embedded_request = _clean_description(description_text)
                if description and len(re.sub(r"[^a-z0-9]", "", _normalize(description))) >= 3:
                    field_tokens = {
                        "description": tuple(token.token_id for token in line_description_tokens),
                        **structured_evidence,
                    }
                    service_date = structured_values.get("service_date") or embedded_date
                    request_no = structured_values.get("request_no") or embedded_request
                    if description.startswith("/") and request_no:
                        description = " ".join(
                            part
                            for part in (
                                current_section,
                                f"{request_no}{description}",
                            )
                            if part
                        )
                    candidate = CandidateLedgerRow(
                        source_row=source_row,
                        role=RowRole.INFORMATIONAL,
                        cells=tuple(token.text for token in line.tokens),
                        section=current_section,
                        description=description,
                        service_date=service_date,
                        request_no=request_no,
                        service_code=structured_values.get("service_code"),
                        hsn_code=structured_values.get("hsn_code"),
                        amount=None,
                        table_type=table_type,
                        category=(
                            row_category(description, table_type)
                            or row_category(current_section or "", table_type)
                        ),
                        source_route="ocr_spatial_graph",
                    )
                    append_aligned(
                        candidate=candidate,
                        field_tokens=field_tokens,
                        raw_description=description_text,
                    )
                continue
            if (
                not line_description_tokens
                and structured_values.get("service_date")
                and set(structured_values) == {"service_date"}
            ):
                pending_service_date = structured_values["service_date"]
                pending_service_date_ids = structured_evidence["service_date"]
                continue
            # Defer deciding whether a text-only line is a section label or a
            # description split from its numeric row until the following line.
            if line_description_tokens:
                continuation_raw = " ".join(
                    token.text for token in line_description_tokens
                )
                continuation_text, _, _ = _clean_description(continuation_raw)
                if (
                    aligned
                    and continuation_text
                    and not _is_payment_footer_description(continuation_raw)
                    and _is_description_continuation(
                        continuation_raw,
                        aligned_description_raw[-1],
                    )
                ):
                    extend_previous_description(line_description_tokens, continuation_raw)
                    pending_description_tokens = []
                else:
                    pending_description_tokens = line_description_tokens
            continue

        description_tokens = line_description_tokens
        if pending_description_tokens:
            pending_text, _, _ = _clean_description(
                " ".join(token.text for token in pending_description_tokens)
            )
            if description_tokens and orientation != "upright" and table_type is TableType.PHARMACY:
                description_tokens = [*pending_description_tokens, *description_tokens]
            elif description_tokens:
                current_section = pending_text or current_section
            else:
                description_tokens = pending_description_tokens
            pending_description_tokens = []

        description_text = " ".join(token.text for token in description_tokens)
        description, embedded_date, embedded_request = _clean_description(description_text)
        service_date = structured_values.get("service_date") or embedded_date
        request_no = structured_values.get("request_no") or embedded_request
        if not description:
            continue
        normalized_description = _normalize(description)
        if len(re.sub(r"[^a-z0-9]", "", normalized_description)) < 3:
            continue
        if _is_total_description(normalized_description):
            continue

        assert amount_token is not None and amount is not None
        used = {_numeric_identity(amount_token, amount)}
        field_tokens: dict[str, tuple[str, ...]] = {
            "description": tuple(token.token_id for token in description_tokens),
            "amount": (amount_token.token_id,),
            **structured_evidence,
        }
        if service_date and "service_date" not in field_tokens:
            date_tokens = tuple(
                token.token_id
                for token in description_tokens
                if DATE_PREFIX.match(token.text.strip())
            )
            if date_tokens:
                field_tokens["service_date"] = date_tokens
        if request_no and "request_no" not in field_tokens:
            request_tokens = tuple(
                token.token_id
                for token in description_tokens
                if request_no.casefold() in token.text.casefold()
            )
            if request_tokens:
                field_tokens["request_no"] = request_tokens
        values: dict[str, Decimal | None] = {
            "quantity": None,
            "rate": None,
            "gross_amount": None,
            "discount": None,
        }
        for role in ("rate", "quantity", "gross_amount", "discount"):
            target = column_centers.get(role)
            if target is None:
                continue
            pair = _closest_numeric(numeric, target, left, width, used)
            if pair is None:
                continue
            token, value = pair
            if abs(((_center_x(token) - left) / width) - target) > 0.06:
                continue
            used.add(_numeric_identity(token, value))
            field_tokens[role] = (token.token_id,)
            values[role] = value

        normalized_description = _normalize(description)
        if _is_metadata_description(description):
            role = RowRole.UNRESOLVED
        elif (
            table_type is TableType.PAYMENT
            or normalized_description
            in {
                "advance",
                "ip advance",
                "deposit",
            }
            or any(
                term in normalized_description
                for term in (
                    "advance received",
                    "amount received",
                    "amount refunded",
                    "payment mode",
                    "receipt ref",
                )
            )
        ):
            role = RowRole.PAYMENT
        elif table_type is TableType.METADATA:
            role = RowRole.UNRESOLVED
        elif table_type in {TableType.CATEGORY_SUMMARY, TableType.PACKAGE_SUMMARY}:
            role = RowRole.CATEGORY_ROLLUP
        else:
            role = RowRole.DETAIL
        if amount < 0 and role is RowRole.DETAIL:
            role = RowRole.REFUND
        validation_flags: list[str] = []
        if "quantity" in column_centers and values["quantity"] is None:
            validation_flags.append("missing_labeled_quantity")
        if "rate" in column_centers and values["rate"] is None:
            validation_flags.append("missing_labeled_unit_price")
        if values["quantity"] is not None and values["rate"] is not None and not has_tax_columns:
            expected_amount = values["quantity"] * values["rate"]
            if values["discount"] is not None:
                expected_amount -= values["discount"]
            if abs(expected_amount - amount) > Decimal("0.01"):
                validation_flags.append("line_arithmetic_mismatch")
        elif values["gross_amount"] is not None and not has_tax_columns:
            expected_amount = values["gross_amount"]
            if values["discount"] is not None:
                expected_amount -= values["discount"]
            if abs(expected_amount - amount) > Decimal("0.01"):
                validation_flags.append("line_arithmetic_mismatch")
        candidate = CandidateLedgerRow(
            source_row=source_row,
            role=role,
            cells=tuple(token.text for token in line.tokens),
            section=current_section,
            description=description,
            service_date=service_date,
            request_no=request_no,
            service_code=structured_values.get("service_code"),
            hsn_code=structured_values.get("hsn_code"),
            quantity=values["quantity"],
            rate=values["rate"],
            gross_amount=values["gross_amount"],
            discount=values["discount"],
            amount=amount,
            table_type=table_type,
            category=(
                row_category(description, table_type)
                or row_category(current_section or "", table_type)
            ),
            source_route="ocr_spatial_graph",
            validation_flags=tuple(validation_flags),
        )
        append_aligned(
            candidate=candidate,
            field_tokens=field_tokens,
            raw_description=description_text,
        )

    source_header = primary_header if header_valid else _raw_source_header(lines, width)
    source_tables = (
        _build_source_tables(
            lines,
            primary=source_header,
            repeated=repeated_header_blocks if header_valid else (),
            original_by_id=original_by_id,
            page_number=page_number,
            table_id=table_id,
            table_type=table_type,
        )
        if source_header is not None
        else _synthetic_source_table(
            lines,
            original_by_id=original_by_id,
            page_number=page_number,
            table_id=table_id,
            table_type=table_type,
            width=width,
        )
    )
    return ReconstructionResult(
        rows=tuple(aligned),
        schema=schema,
        diagnostics={
            "ocr_token_count": len(scoped),
            "ocr_line_count": len(lines),
            "ocr_row_count": len(aligned),
            "header_found": header_valid,
            "schema_inherited": inherited,
            "orientation": orientation,
            "deskew_slope": deskew_slope,
            "table_type": table_type.value,
            "column_centers": column_centers,
            "header_segments": 1 + len(repeated_header_indexes) if header_valid else 0,
        },
        source_tables=source_tables,
    )


def fuse_provider_descriptions(
    ocr_rows: tuple[AlignedLedgerRow, ...],
    provider_rows: tuple[CandidateLedgerRow, ...],
) -> tuple[AlignedLedgerRow, ...]:
    """Use structured provider text only when the OCR row independently supports it.

    Printed OCR remains authoritative for geometry and amount. Provider text can
    repair recognition noise on rotated/skewed descriptions, but cannot create a
    row or overwrite the printed amount.
    """
    proposals = tuple(
        row
        for row in provider_rows
        if row.description
        and row.amount is not None
        and row.role in {RowRole.DETAIL, RowRole.REFUND}
    )
    if not ocr_rows or not proposals:
        return ocr_rows

    exact_amount_fraction = sum(
        left.candidate.amount == right.amount
        for left, right in zip(ocr_rows, proposals, strict=False)
    ) / max(len(ocr_rows), len(proposals))
    sequence_aligned = abs(len(ocr_rows) - len(proposals)) <= 1 and exact_amount_fraction >= 0.5
    used: set[int] = set()
    output: list[AlignedLedgerRow] = []
    for index, aligned in enumerate(ocr_rows):
        source_description = aligned.candidate.description or ""
        match_index: int | None = None
        match_score = 0.0
        if sequence_aligned and index < len(proposals):
            match_index = index
        else:
            for proposal_index, proposal in enumerate(proposals):
                if proposal_index in used or abs(proposal_index - index) > 3:
                    continue
                similarity = SequenceMatcher(
                    None,
                    _normalize(source_description),
                    _normalize(proposal.description),
                ).ratio()
                amount_bonus = 1.0 if aligned.candidate.amount == proposal.amount else 0.0
                score = amount_bonus + similarity - abs(proposal_index - index) * 0.05
                if score > match_score:
                    match_score = score
                    match_index = proposal_index
        if match_index is None:
            output.append(aligned)
            continue
        proposal = proposals[match_index]
        similarity = SequenceMatcher(
            None,
            _normalize(source_description),
            _normalize(proposal.description),
        ).ratio()
        if similarity < 0.25 and not (
            sequence_aligned and aligned.candidate.amount == proposal.amount
        ):
            output.append(aligned)
            continue
        used.add(match_index)
        description = proposal.description or source_description
        candidate = replace(
            aligned.candidate,
            description=description,
            category=(
                row_category(description, aligned.candidate.table_type)
                or aligned.candidate.category
            ),
            source_route="ocr_spatial_graph+provider_otsl",
        )
        output.append(
            replace(
                aligned,
                candidate=candidate,
                source_routes=("ocr_spatial_graph", "provider_otsl"),
            )
        )
    return tuple(output)
