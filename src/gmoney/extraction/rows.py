from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from gmoney.contracts.extraction import RowRole, TableType
from gmoney.extraction.otsl import OtslCell, OtslTable
from gmoney.extraction.typed_values import parse_decimal

ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "description": (
        "description",
        "particular",
        "service name",
        "item name",
        "product name",
        "productname",
        "item",
        "services",
        "test name",
        "investigation",
    ),
    "quantity": ("quantity", "qty", "nos", "units", "unit/days"),
    "rate": ("rate", "unit price", "unit rate"),
    "gross_amount": ("gross amount", "service amount", "service amt"),
    "discount": ("discount", "disco", "disc amt"),
    "amount": ("total amount", "net amount", "payable", "bill amount", "amount", "total"),
    "service_date": ("service date", "bill date", "billdate", "date/time", "date"),
    "request_no": ("request no", "requisition", "ref no", "bill number"),
    "service_code": ("service code", "code"),
    "hsn_code": ("hsn", "sac"),
}


@dataclass(frozen=True)
class CandidateLedgerRow:
    source_row: int
    role: RowRole
    cells: tuple[str, ...]
    section: str | None = None
    description: str | None = None
    service_date: str | None = None
    request_no: str | None = None
    service_code: str | None = None
    hsn_code: str | None = None
    quantity: Decimal | None = None
    rate: Decimal | None = None
    gross_amount: Decimal | None = None
    discount: Decimal | None = None
    amount: Decimal | None = None
    amount_derived: bool = False
    table_type: TableType = TableType.UNKNOWN
    category: str | None = None
    source_route: str = "provider_otsl"
    validation_flags: tuple[str, ...] = ()


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _compound_header_columns(header: tuple[OtslCell, ...]) -> dict[str, int]:
    nonempty = [cell for cell in header if cell.text.strip()]
    if len(nonempty) != 1 or len(header) < 3:
        return {}
    tokens = re.findall(r"#|[a-z0-9]+", nonempty[0].text.casefold())
    columns: dict[str, int] = {}
    column = 0
    index = 0
    phrases: tuple[tuple[tuple[str, ...], str | None], ...] = (
        (("service", "name"), "description"),
        (("item", "name"), "description"),
        (("total", "amount"), "amount"),
        (("net", "amount"), "amount"),
        (("unit", "price"), "rate"),
        (("unit", "rate"), "rate"),
        (("qty", "days"), "quantity"),
        (("sr", "no"), None),
    )
    single_roles = {
        "particular": "description",
        "particulars": "description",
        "description": "description",
        "item": "description",
        "qty": "quantity",
        "quantity": "quantity",
        "rate": "rate",
        "amount": "amount",
        "discount": "discount",
    }
    while index < len(tokens):
        matched = False
        for phrase, role in phrases:
            if tuple(tokens[index : index + len(phrase)]) != phrase:
                continue
            if role:
                columns[role] = column
            index += len(phrase)
            column += 1
            matched = True
            break
        if matched:
            continue
        token = tokens[index]
        role = single_roles.get(token)
        if role:
            columns[role] = column
        index += 1
        column += 1
    if "description" in columns and (
        {"rate", "gross_amount", "amount", "quantity"} & columns.keys()
    ):
        return columns
    return {}


def infer_columns(header: tuple[OtslCell, ...]) -> dict[str, int]:
    compound = _compound_header_columns(header)
    if compound:
        return compound
    columns: dict[str, int] = {}
    for index, cell in enumerate(header):
        normalized = _normalized(cell.text)
        best: tuple[int, str] | None = None
        for role, terms in ROLE_TERMS.items():
            for term in terms:
                normalized_term = _normalized(term)
                if normalized_term in normalized:
                    score = len(normalized_term)
                    if best is None or score > best[0]:
                        best = (score, role)
        if best:
            role = best[1]
            if role != "amount" or role not in columns or index > columns[role]:
                columns[role] = index

    # Bills commonly label the unit-rate column "Amount Rs." and the extended
    # line value "Total".  Once a distinct right-hand total is present, retain
    # the earlier generic amount column as the rate instead of discarding it.
    amount_index = columns.get("amount")
    if "rate" not in columns and amount_index is not None:
        for index, cell in enumerate(header[:amount_index]):
            normalized = _normalized(cell.text)
            if normalized == "amount" or normalized.startswith("amount rs"):
                columns["rate"] = index
                break
    return columns


def _row_role(cells: tuple[str, ...], description: str | None) -> RowRole:
    text = _normalized(" ".join(cells))
    if any(term in text for term in ("refund", "credit note", "return")):
        return RowRole.REFUND
    if any(term in text for term in ("advance received", "payment mode", "receipt ref")):
        return RowRole.PAYMENT
    if any(term in text for term in ("deposit", "advance deposit")):
        return RowRole.DEPOSIT
    if any(term in text for term in ("grand total", "net bill amount", "total bill amount")):
        return RowRole.DOCUMENT_TOTAL
    normalized_description = _normalized(description or "")
    if normalized_description in {"discount", "discounts"}:
        return RowRole.SECTION_TOTAL
    if normalized_description == "total" or normalized_description.startswith(
        ("sub total", "subtotal")
    ):
        return RowRole.SECTION_TOTAL
    return RowRole.DETAIL


def _value(cells: tuple[str, ...], columns: dict[str, int], role: str) -> str | None:
    index = columns.get(role)
    if index is None or index >= len(cells):
        return None
    return cells[index].strip() or None


def _merge_continuation_rows(
    rows: tuple[tuple[OtslCell, ...], ...],
    columns: dict[str, int],
) -> tuple[tuple[str, ...], ...]:
    """Join a description-only row to its immediately following numeric row."""
    merged: list[tuple[str, ...]] = []
    index = 0
    numeric_roles = sorted(
        (
            (column, role)
            for role in ("quantity", "rate", "gross_amount", "discount", "amount")
            if (column := columns.get(role)) is not None
        ),
        key=lambda item: item[0],
    )
    description_column = columns["description"]
    while index < len(rows):
        cells = tuple(cell.text.strip() for cell in rows[index])
        description = cells[description_column] if description_column < len(cells) else ""
        current_numbers = [parse_decimal(cell) for cell in cells]
        if (
            description
            and not any(value is not None for value in current_numbers)
            and index + 1 < len(rows)
        ):
            following = tuple(cell.text.strip() for cell in rows[index + 1])
            following_description = (
                following[description_column] if description_column < len(following) else ""
            )
            values = [cell for cell in following if parse_decimal(cell) is not None]
            if not following_description and values and numeric_roles:
                combined = list(cells)
                if len(values) >= len(numeric_roles):
                    assignments = zip(numeric_roles, values[-len(numeric_roles) :], strict=True)
                elif len(values) == 1:
                    amount_role = next(
                        (item for item in reversed(numeric_roles) if item[1] == "amount"),
                        numeric_roles[-1],
                    )
                    assignments = ((amount_role, values[0]),)
                else:
                    quantity_role = next(
                        (item for item in numeric_roles if item[1] == "quantity"),
                        None,
                    )
                    tail_roles = [item for item in numeric_roles if item != quantity_role]
                    selected_roles = ([quantity_role] if quantity_role else []) + tail_roles[
                        -(len(values) - (1 if quantity_role else 0)) :
                    ]
                    assignments = zip(selected_roles, values, strict=True)
                for (column, _), value in assignments:
                    combined[column] = value
                merged.append(tuple(combined))
                index += 2
                continue
        merged.append(cells)
        index += 1
    return tuple(merged)


def extract_candidate_rows(table: OtslTable) -> tuple[CandidateLedgerRow, ...]:
    if len(table.rows) < 2:
        return ()
    header_index = max(
        range(len(table.rows)), key=lambda index: len(infer_columns(table.rows[index]))
    )
    columns = infer_columns(table.rows[header_index])
    if "description" not in columns:
        return ()
    rows: list[CandidateLedgerRow] = []
    data_rows = _merge_continuation_rows(table.rows[header_index + 1 :], columns)
    for source_row, cells in enumerate(data_rows, header_index + 1):
        description = _value(cells, columns, "description")
        quantity = parse_decimal(_value(cells, columns, "quantity"))
        rate = parse_decimal(_value(cells, columns, "rate"))
        gross_amount = parse_decimal(_value(cells, columns, "gross_amount"))
        discount = parse_decimal(_value(cells, columns, "discount"))
        amount_text = _value(cells, columns, "amount")
        amount = parse_decimal(amount_text)
        amount_derived = False
        if amount is None and quantity is not None and rate is not None and "amount" in columns:
            amount = quantity * rate
            amount_derived = True
        elif amount is None:
            numeric = [(index, parse_decimal(cell)) for index, cell in enumerate(cells)]
            numeric = [(index, value) for index, value in numeric if value is not None]
            amount_text = cells[numeric[-1][0]] if numeric else None
            amount = parse_decimal(amount_text)
        role = _row_role(cells, description)
        if not description and amount is None:
            continue
        if role is RowRole.DETAIL and amount is None:
            role = RowRole.METADATA
        rows.append(
            CandidateLedgerRow(
                source_row=source_row,
                role=role,
                cells=cells,
                description=description,
                service_date=_value(cells, columns, "service_date"),
                request_no=_value(cells, columns, "request_no"),
                service_code=_value(cells, columns, "service_code"),
                hsn_code=_value(cells, columns, "hsn_code"),
                quantity=quantity,
                rate=rate,
                gross_amount=gross_amount,
                discount=discount,
                amount=amount,
                amount_derived=amount_derived,
            )
        )
    return tuple(rows)
