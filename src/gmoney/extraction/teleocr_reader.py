"""Turn TeleOCR OTSL table reads into candidate ledger rows.

TeleOCR (and other VLM readers) return a faithful cell grid, but a bill grid still needs
plain-code interpretation before it becomes ledger rows:

* column roles by header meaning, with the description column being the item-name text
  column, never a code column (``SER0943548`` vs ``GRAM STAIN``);
* multi-line descriptions merged (``Advamab`` + ``100mg Inj``);
* section headings and ``Issued Date`` group rows become section/service-date context;
* rows under a ``Return``/``Returns`` heading become negative refunds;
* printed totals, settlement lines, and document identifiers (bill/IP/UHID numbers) are
  never emitted as charges.

The functions here are pure so they can be tested with recorded model replies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from statistics import median

from gmoney.contracts.evidence import OcrToken
from gmoney.contracts.extraction import RowRole
from gmoney.extraction.document_total import FINAL_LABELS
from gmoney.extraction.otsl import OtslCell, OtslTable, parse_otsl, split_otsl_tables
from gmoney.extraction.reconciliation import is_settlement_label
from gmoney.extraction.rows import ROLE_TERMS, CandidateLedgerRow
from gmoney.extraction.total_labels import is_total_label, normalized_label
from gmoney.extraction.typed_values import parse_decimal

READER_ROUTE = "teleocr_otsl"
# Header vocabulary in addition to rows.ROLE_TERMS (which the heuristic path keeps unchanged).
READER_ROLE_TERMS: dict[str, tuple[str, ...]] = {
    **ROLE_TERMS,
    "description": (
        *ROLE_TERMS["description"],
        "procedure name",
        "procedure",
        "drug name",
        "medicine name",
        "medicine",
        "test description",
        "service description",
        "charge description",
        "name",
    ),
    "quantity": (*ROLE_TERMS["quantity"], "no of days", "days"),
    "rate": (*ROLE_TERMS["rate"], "mrp", "price", "unit cost"),
    "amount": (*ROLE_TERMS["amount"], "amt", "value", "net amt", "amount rs"),
    "service_code": (*ROLE_TERMS["service_code"], "imr", "item code", "test code"),
    "service_date": (*ROLE_TERMS["service_date"], "issued date", "issue date"),
    "serial": ("sr no", "s no", "sl no", "sno", "#"),
    "batch": ("batch", "batch no", "expiry", "exp"),
}
CODE_PATTERN = re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9/_.-]{3,}$")
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}[- ][A-Za-z]{3}[- ]\d{2,4}\b"
)
DATE_GROUP_PATTERN = re.compile(r"^(?:issued?|issue|service|bill|order)?\s*date\b", re.IGNORECASE)
# Applied to normalized labels ("Bill No.:" -> "bill no").
# The whole label must be the identifier ("bill no", "uhid"), so "Patient ID Band" or
# "IPD Registration" stay charges.
IDENTIFIER_LABEL = re.compile(
    r"(?:uhid|mrn|gstin|(?:bill|invoice|receipt|ip|ipd|op|reg|registration|admission|"
    r"patient|policy|claim|card|pan|phone|mobile|page|uhid|mrn)\s*(?:no|nos|number|id))"
)
# Printed bill/gross/payable total labels from the document-total detector.
DOCUMENT_TOTAL_TERMS = tuple(sorted({spec[0] for spec in FINAL_LABELS}))
RETURN_WORDS_SET = frozenset({"return", "returns", "returned"})
TOTAL_QUALIFIER_WORDS = frozenset(
    {
        "item",
        "items",
        "page",
        "grand",
        "net",
        "gross",
        "final",
        "lab",
        "drug",
        "drugs",
        "category",
        "section",
        "package",
        "department",
        "ward",
        "icu",
        "ot",
        "sub",
    }
)
TOTAL_TAIL_WORDS = frozenset(
    {"amount", "amt", "value", "rs", "inr", "bill", "of", "the", "for", "charges", "charge"}
)
SECTION_RETURN = re.compile(r"\breturn(?:s|ed)?\b", re.IGNORECASE)
SECTION_WORDS = (
    "bed",
    "charges",
    "consultation",
    "consumable",
    "doctor",
    "investigation",
    "issue",
    "laboratory",
    "medicine",
    "nursing",
    "pathology",
    "pharmacy",
    "procedure",
    "radiology",
    "return",
    "room",
    "service",
    "surgery",
    "visit",
)
NUMERIC_ROLES = ("quantity", "rate", "gross_amount", "discount", "amount")
AMOUNT_TOKEN = re.compile(r"^\(?-?(?:₹|rs\.?)?\s*\d{1,3}(?:,?\d{2,3})*\.\d{2}\)?$", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_number(text: str) -> bool:
    return bool(text) and parse_decimal(text) is not None and not DATE_PATTERN.search(text)


def _header_role(text: str) -> str | None:
    normalized = normalized_label(text)
    if not normalized:
        return None
    best: tuple[int, str] | None = None
    for role, terms in READER_ROLE_TERMS.items():
        for term in terms:
            term_text = normalized_label(term) or term
            if term_text == "#" and text.strip() != "#":
                continue
            words = normalized.split()
            if (" " in term_text and term_text in normalized) or any(
                word == term_text or (len(term_text) >= 4 and word.startswith(term_text))
                for word in words
            ):
                score = len(term_text)
                if best is None or score > best[0]:
                    best = (score, role)
    if best and best[1] == "description" and "code" in normalized.split():
        return "service_code"
    return best[1] if best else None


def infer_reader_columns(header: tuple[OtslCell, ...]) -> dict[str, int]:
    columns: dict[str, int] = {}
    for index, cell in enumerate(header):
        role = _header_role(cell.text)
        if role is None:
            continue
        if role == "amount" and "amount" in columns:
            # Bills print "Amount" (rate) before "Total"/"Net Amount"; keep the right-most.
            columns.setdefault("rate", columns["amount"])
            columns["amount"] = index
            continue
        columns.setdefault(role, index)
    return columns


def _header_index(table: OtslTable) -> int | None:
    best: tuple[int, int] | None = None
    for index, row in enumerate(table.rows[:6]):
        if any(_is_number(cell.text) for cell in row):
            continue
        columns = infer_reader_columns(row)
        score = len(set(columns) & {"description", *NUMERIC_ROLES})
        if "description" in columns and score >= 2 and (best is None or score > best[0]):
            best = (score, index)
    return best[1] if best else None


def _looks_like_code(text: str) -> bool:
    return bool(CODE_PATTERN.fullmatch(text.strip())) and " " not in text.strip()


def _looks_like_name(text: str) -> bool:
    letters = sum(character.isalpha() for character in text)
    return letters >= 3 and not _looks_like_code(text) and not _is_number(text)


def _fix_description_column(columns: dict[str, int], rows: list[tuple[str, ...]]) -> dict[str, int]:
    """Prefer the item-name column over a code column mislabelled as the description."""
    width = max((len(row) for row in rows), default=0)
    if not rows or width == 0:
        return columns

    def share(column: int, predicate) -> float:
        values = [row[column] for row in rows if column < len(row) and row[column]]
        return sum(predicate(value) for value in values) / len(values) if values else 0.0

    description = columns.get("description")
    used = {index for role, index in columns.items() if role != "description"}
    candidates = [index for index in range(width) if index not in used]
    named = max(candidates, key=lambda index: share(index, _looks_like_name), default=None)
    if named is None or share(named, _looks_like_name) < 0.5:
        return columns
    if description is None or (
        share(description, _looks_like_code) >= 0.5 and named != description
    ):
        updated = dict(columns)
        if description is not None:
            updated.setdefault("service_code", description)
        updated["description"] = named
        return updated
    return columns


def _cells(row: tuple[OtslCell, ...]) -> tuple[str, ...]:
    return tuple(_clean(cell.text) for cell in row)


def _spans_columns(row: tuple[OtslCell, ...]) -> bool:
    return any(cell.is_span_marker for cell in row)


@dataclass
class _Context:
    section: str | None = None
    service_date: str | None = None
    returns: bool = False


def _value(cells: tuple[str, ...], columns: dict[str, int], role: str) -> str | None:
    index = columns.get(role)
    if index is None or index >= len(cells):
        return None
    return cells[index] or None


def _is_suffix_fragment(text: str) -> bool:
    """``300 Mcg`` / ``100mg`` / ``(Long CT)`` continue the previous item's name."""
    stripped = text.strip()
    return bool(stripped) and (
        stripped[0].isdigit() or stripped[0] in "(-/&+," or stripped[0].islower()
    )


def _is_section_heading(text: str, *, spanned: bool) -> bool:
    normalized = normalized_label(text)
    if not normalized or is_total_label(normalized):
        return False
    if spanned:
        return True
    words = normalized.split()
    return (
        len(words) <= 4
        and any(word.startswith(SECTION_WORDS) for word in words)
        and (
            text.isupper()
            or any(word in {"charges", "return", "returns", "issues"} for word in words)
        )
    )


def _row_label(cells: tuple[str, ...]) -> str:
    return normalized_label(" ".join(cell for cell in cells if cell and not _is_number(cell)))


def _is_identifier_row(label: str, numbers: list[str]) -> bool:
    """Header/footer identifiers such as ``Bill No: 4172`` are never charges."""
    if not IDENTIFIER_LABEL.fullmatch(label):
        return False
    return all("." not in value for value in numbers)


def _is_total_row_label(
    label: str,
    *,
    priced_line: bool = False,
    shape_known: bool = True,
    closes_open_rows: bool = False,
) -> bool:
    """A printed total, never an item whose name merely contains "total".

    A label ending in "total" is a section total when its qualifier words are section
    words, when its amount equals the charge rows printed above it since the previous
    total (``closes_open_rows``), or, in a table with quantity/rate columns
    (``shape_known``), when the row prints neither (``priced_line`` False).  So
    "Bilirubin Total 1 x 250.00" or an amount-only "Bilirubin Total 250.00" under other
    tests stays a charge, while "Blood Bank Total" closing its rows is a total.
    """
    if is_total_label(label) or any(
        label == term or label.startswith(f"{term} ") for term in DOCUMENT_TOTAL_TERMS
    ):
        return True
    words = label.split()
    if not words:
        return False

    def qualifiers(rest: list[str]) -> bool:
        # "Pharmacy Total", "Item Issues Total" qualify; "Bilirubin Total" is a lab test.
        return all(
            word in TOTAL_TAIL_WORDS
            or word in TOTAL_QUALIFIER_WORDS
            or word.startswith(SECTION_WORDS)
            for word in rest
        )

    if words[-1] in {"total", "totals", "subtotal"}:
        return (
            qualifiers(words[:-2] if words[-2:-1] == ["sub"] else words[:-1])
            or closes_open_rows
            or (shape_known and not priced_line)
        )
    if RETURN_WORDS_SET & set(words) and set(words) <= RETURN_WORDS_SET | {"amount", "value"}:
        return len(words) > 1  # "Return Amount", "Returns Value"

    if words[0] in {"total", "subtotal"} or words[:2] == ["sub", "total"]:
        return qualifiers(words[2:] if words[:2] == ["sub", "total"] else words[1:])
    return False


def _classify_numeric_row(
    cells: tuple[str, ...],
    description: str | None,
    context: _Context,
    *,
    priced_line: bool = False,
    shape_known: bool = True,
    closes_open_rows: bool = False,
) -> tuple[RowRole, tuple[str, ...]]:
    label = _row_label(cells)
    numbers = [cell for cell in cells if _is_number(cell)]
    if _is_identifier_row(label, numbers):
        return RowRole.METADATA, ("reader_identifier_row",)
    if (
        label
        and is_settlement_label(label)
        and (not description or normalized_label(description) == label)
    ):
        return RowRole.PAYMENT, ()
    if label and _is_total_row_label(
        label,
        priced_line=priced_line,
        shape_known=shape_known,
        closes_open_rows=closes_open_rows,
    ):
        return (
            RowRole.DOCUMENT_TOTAL
            if is_total_label(label) and "grand" in label
            else RowRole.SECTION_TOTAL
        ), ()
    if not description:
        return RowRole.METADATA, ("reader_row_without_description",)
    if context.returns:
        return RowRole.REFUND, ("reader_return_section",)
    return RowRole.DETAIL, ()


def _heading_only(table: OtslTable) -> str | None:
    """A split-off chunk that only carries a section heading (e.g. ``Bed Procedure``)."""
    texts = [cell for row in table.rows for cell in _cells(row) if cell]
    if not texts or any(_is_number(text) for text in texts) or _header_index(table) is not None:
        return None
    text = " ".join(texts)
    return text if len(table.rows) <= 2 and len(text) <= 80 else None


def _table_rows(
    table: OtslTable,
    *,
    section: str | None = None,
    inherited_header: tuple[str, ...] | None = None,
) -> tuple[CandidateLedgerRow, ...]:
    header_index = _header_index(table)
    inherited = False
    if header_index is None and inherited_header and len(inherited_header) == table.column_count:
        # A headerless tile/continuation read reuses the header of the first read.
        header = tuple(
            OtslCell(text=text, row=0, column=index) for index, text in enumerate(inherited_header)
        )
        table = OtslTable(rows=(header, *table.rows))
        header_index, inherited = 0, True
    if header_index is None:
        return ()
    columns = infer_reader_columns(table.rows[header_index])
    data = [
        (index, _cells(row), row) for index, row in enumerate(table.rows) if index > header_index
    ]
    columns = _fix_description_column(columns, [cells for _, cells, _ in data])
    description_column = columns["description"]
    context = _Context(section=section, returns=bool(section and SECTION_RETURN.search(section)))
    for row in table.rows[:header_index]:
        text = " ".join(cell for cell in _cells(row) if cell)
        if text and _is_section_heading(text, spanned=_spans_columns(row) or len(row) == 1):
            context.section = text
            context.returns = bool(SECTION_RETURN.search(text))
    output: list[CandidateLedgerRow] = []
    # Charge amounts printed since the previous total (refunds negative).
    open_charges: list[Decimal] = []
    pending_prefix: list[str] = []
    for position, (source_row, cells, raw) in enumerate(data):
        populated = [(index, cell) for index, cell in enumerate(cells) if cell]
        if not populated:
            continue
        has_number = any(
            _is_number(cell)
            for index, cell in populated
            if index != description_column or len(populated) > 1
        )
        text = " ".join(cell for _, cell in populated)
        if not has_number:
            only_text = len(populated) == 1
            if DATE_GROUP_PATTERN.match(text) or (only_text and DATE_PATTERN.fullmatch(text)):
                match = DATE_PATTERN.search(text)
                context.service_date = match.group(0) if match else context.service_date
                continue
            words = normalized_label(text).split()
            ends_returns = bool(
                context.returns
                and only_text
                and 0 < len(words) <= 4
                and any(word.startswith(SECTION_WORDS) for word in words)
                and not SECTION_RETURN.search(text)
            )
            if ends_returns or _is_section_heading(text, spanned=_spans_columns(raw)):
                # Any section heading ends a returns section ("Consumables" after returns).
                context.section = text
                context.returns = bool(SECTION_RETURN.search(text))
                pending_prefix.clear()
                continue
            if only_text and populated[0][0] == description_column:
                following = data[position + 1][1] if position + 1 < len(data) else ()
                if (
                    output
                    and _is_suffix_fragment(text)
                    and output[-1].role in {RowRole.DETAIL, RowRole.REFUND}
                ):
                    previous = output[-1]
                    output[-1] = replace(
                        previous,
                        description=_clean(f"{previous.description} {text}"),
                        validation_flags=tuple(
                            dict.fromkeys((*previous.validation_flags, "reader_multiline_merged"))
                        ),
                    )
                    continue
                if any(_is_number(cell) for cell in following):
                    pending_prefix.append(text)
                    continue
            # Unpriced text rows (package inclusions, notes) stay informational.
            output.append(
                CandidateLedgerRow(
                    source_row=source_row,
                    role=RowRole.INFORMATIONAL,
                    cells=cells,
                    section=context.section,
                    description=_value(cells, columns, "description") or text,
                    service_date=_value(cells, columns, "service_date") or context.service_date,
                    service_code=_value(cells, columns, "service_code"),
                    source_route=READER_ROUTE,
                )
            )
            continue
        description = _value(cells, columns, "description")
        if pending_prefix:
            description = _clean(" ".join([*pending_prefix, description or ""])) or None
            flags: tuple[str, ...] = ("reader_multiline_merged",)
            pending_prefix.clear()
        else:
            flags = ()
        quantity = parse_decimal(_value(cells, columns, "quantity"))
        rate = parse_decimal(_value(cells, columns, "rate"))
        gross_amount = parse_decimal(_value(cells, columns, "gross_amount"))
        discount = parse_decimal(_value(cells, columns, "discount"))
        amount = parse_decimal(_value(cells, columns, "amount"))
        if amount is None and "amount" not in columns:
            # No amount column: the right-most number that is not a count, code or date.
            excluded = {
                columns[role]
                for role in (
                    "quantity",
                    "service_code",
                    "hsn_code",
                    "request_no",
                    "serial",
                    "batch",
                )
                if role in columns
            }
            numeric = [
                parse_decimal(cell)
                for index, cell in enumerate(cells)
                if index not in excluded and _is_number(cell)
            ]
            amount = numeric[-1] if numeric else None
        role, role_flags = _classify_numeric_row(
            cells,
            description,
            context,
            priced_line=quantity is not None or rate is not None,
            shape_known="quantity" in columns or "rate" in columns,
            closes_open_rows=bool(
                open_charges and amount is not None and abs(amount) == abs(sum(open_charges))
            ),
        )
        # Refunds keep the printed (usually positive) amount so they ground to the printed
        # token; the refund role carries the sign (reconciliation counts refunds negative).
        if role is RowRole.DETAIL and amount is None:
            role, role_flags = RowRole.METADATA, ("reader_row_without_amount",)
        row_label = description
        if role in {RowRole.SECTION_TOTAL, RowRole.DOCUMENT_TOTAL, RowRole.PAYMENT}:
            printed_label = " ".join(cell for cell in cells if cell and not _is_number(cell))
            row_label = _clean(printed_label) or description
        output.append(
            CandidateLedgerRow(
                source_row=source_row,
                role=role,
                cells=cells,
                section=context.section,
                description=row_label,
                service_date=_value(cells, columns, "service_date") or context.service_date,
                request_no=_value(cells, columns, "request_no"),
                service_code=_value(cells, columns, "service_code"),
                hsn_code=_value(cells, columns, "hsn_code"),
                quantity=quantity,
                rate=rate,
                gross_amount=gross_amount,
                discount=discount,
                amount=amount,
                source_route=READER_ROUTE,
                validation_flags=tuple(
                    dict.fromkeys(
                        (*flags, *role_flags, *(("reader_inherited_header",) if inherited else ()))
                    )
                ),
            )
        )
        if role in {RowRole.SECTION_TOTAL, RowRole.DOCUMENT_TOTAL}:
            open_charges.clear()
        elif role in {RowRole.DETAIL, RowRole.REFUND} and amount is not None:
            open_charges.append(-abs(amount) if role is RowRole.REFUND else amount)
    return tuple(output)


def read_otsl_rows(
    content: str, *, inherited_header: tuple[str, ...] | None = None
) -> tuple[CandidateLedgerRow, ...]:
    """Candidate rows from one TeleOCR reply (which may contain several OTSL tables)."""
    content = content.split("<|im_end|>", 1)[0]
    rows: list[CandidateLedgerRow] = []
    section: str | None = None
    for table in split_otsl_tables(parse_otsl(content)):
        heading = _heading_only(table)
        if heading is not None:
            section = heading
            continue
        rows.extend(_table_rows(table, section=section, inherited_header=inherited_header))
        section = None
    return tuple(rows)


def otsl_header(content: str) -> tuple[str, ...] | None:
    """Header cells of the first table in a reply, for headerless tile reads."""
    content = content.split("<|im_end|>", 1)[0]
    for table in split_otsl_tables(parse_otsl(content)):
        index = _header_index(table)
        if index is not None:
            return tuple(cell.text for cell in table.rows[index])
    return None


def charge_rows(rows: Iterable[CandidateLedgerRow]) -> tuple[CandidateLedgerRow, ...]:
    return tuple(row for row in rows if row.role in {RowRole.DETAIL, RowRole.REFUND})


def printed_total_rows(rows: Iterable[CandidateLedgerRow]) -> tuple[CandidateLedgerRow, ...]:
    return tuple(row for row in rows if row.role in {RowRole.SECTION_TOTAL, RowRole.DOCUMENT_TOTAL})


# Missed-table guard -------------------------------------------------------------------


def _token_center(token: OcrToken) -> tuple[float, float]:
    xs = [point.x for point in token.polygon.points]
    ys = [point.y for point in token.polygon.points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def is_amount_token(text: str) -> bool:
    value = text.strip()
    return bool(AMOUNT_TOKEN.fullmatch(value)) and parse_decimal(value) not in (None, Decimal(0))


def uncovered_financial_tokens(
    tokens: Iterable[OcrToken], boxes: Iterable[tuple[int, int, int, int]]
) -> tuple[OcrToken, ...]:
    boxes = tuple(boxes)
    uncovered = []
    for token in tokens:
        if not is_amount_token(token.text):
            continue
        x, y = _token_center(token)
        if not any(left <= x <= right and top <= y <= bottom for left, top, right, bottom in boxes):
            uncovered.append(token)
    return tuple(uncovered)


def needs_full_page_read(
    tokens: Iterable[OcrToken],
    boxes: Iterable[tuple[int, int, int, int]],
    *,
    minimum_tokens: int = 2,
) -> bool:
    """True when amount-like PP-OCR tokens lie outside every detected table box."""
    return len(uncovered_financial_tokens(tokens, boxes)) >= minimum_tokens


def uncovered_band(
    tokens: Iterable[OcrToken],
    boxes: Iterable[tuple[int, int, int, int]],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Full-width page band around the uncovered amounts, used as the guard's evidence crop."""
    tokens = tuple(tokens)
    uncovered = uncovered_financial_tokens(tokens, boxes)
    if not uncovered:
        return None
    heights = [
        max(point.y for point in token.polygon.points)
        - min(point.y for point in token.polygon.points)
        for token in tokens
    ]
    pad = max(12.0, (median(heights) if heights else 12.0) * 4)
    top = min(min(point.y for point in token.polygon.points) for token in uncovered) - pad
    bottom = max(max(point.y for point in token.polygon.points) for token in uncovered) + pad
    return 0, max(0, round(top)), width, min(height, round(bottom))


def _identity(
    description: object, amount: object, service_date: object = None
) -> tuple[str, Decimal | None, str]:
    value = parse_decimal(str(amount)) if amount is not None else None
    return (
        normalized_label(description),
        abs(value) if value is not None else None,
        normalized_label(service_date),
    )


def drop_rows_already_read(
    candidates: Iterable[CandidateLedgerRow],
    existing: Iterable[tuple[object, ...]],
) -> tuple[CandidateLedgerRow, ...]:
    """Dedupe a re-read (full-page guard, overlapping tile) against rows already read.

    ``existing`` holds ``(description, amount)`` or ``(description, amount, date)``; with a
    date, a same-named charge on another day is kept.
    """
    existing = list(existing)
    seen = {_identity(*item) for item in existing}
    dated = any(len(item) > 2 for item in existing)
    return tuple(
        row
        for row in candidates
        if row.role not in {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}
        or _identity(row.description, row.amount, row.service_date if dated else None) not in seen
    )
