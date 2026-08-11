from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

CURRENCY = re.compile(r"(?:₹|inr|rs\.?|rupees?)", re.IGNORECASE)
NUMERIC = re.compile(r"^[+-]?\d+(?:\.\d{1,4})?$")
DAY_QUANTITY = re.compile(
    r"\s*(?P<quantity>[+-]?\d[\d,]*(?:\.\d{1,4})?)\s*"
    r"(?:days?|day\s*\(s\))\.?\s*",
    re.IGNORECASE,
)
MAX_ABSOLUTE = Decimal("999999999999.9999")
DATE_FRAGMENT = re.compile(
    r"\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}|[-\s](?:Jan(?:uary)?|Feb(?:ruary)?|"
    r"Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-\s]\d{2,4})",
    re.IGNORECASE,
)

SERVICE_DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%d %B %Y",
)


def parse_decimal(value: object) -> Decimal | None:
    text = str(value or "").strip()
    if not text or any(character in text for character in ("/", ":", "@")):
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if re.search(r"\bCR\.?$", text, re.IGNORECASE):
        negative = True
        text = re.sub(r"\bCR\.?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDR\.?$", "", text, flags=re.IGNORECASE)
    text = CURRENCY.sub("", text)
    text = text.replace(",", "").replace(" ", "")
    if text and re.fullmatch(r"[0-9OoIl.+-]+", text):
        text = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    if not NUMERIC.fullmatch(text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if negative:
        number = -abs(number)
    if abs(number) > MAX_ABSOLUTE:
        return None
    return number


def parse_quantity(value: object) -> Decimal | None:
    text = str(value or "").strip()
    duration = DAY_QUANTITY.fullmatch(text)
    if duration is not None:
        text = duration.group("quantity")
    elif re.fullmatch(r"[+-]?\d+\.", text):
        text = text[:-1]
    return parse_decimal(text)


def parse_service_date(value: object) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -:")
    if not text:
        return None
    matches = tuple(DATE_FRAGMENT.finditer(text))
    if len(matches) != 1 or matches[0].start() != 0:
        return None
    remainder = text[matches[0].end() :].strip()
    time_match = (
        re.fullmatch(
            r"[,;.\-]?\s*(?P<clock>\d{1,2}:\d{2}(?::\d{2})?)"
            r"\s*(?P<meridiem>am|pm)?",
            remainder,
            re.IGNORECASE,
        )
        if remainder
        else None
    )
    if remainder and time_match is None:
        return None
    if time_match is not None:
        clock = time_match.group("clock")
        meridiem = time_match.group("meridiem")
        parsed_time = f"{clock} {meridiem}" if meridiem else clock
        try:
            datetime.strptime(
                parsed_time,
                (
                    ("%I:%M:%S %p" if clock.count(":") == 2 else "%I:%M %p")
                    if meridiem
                    else ("%H:%M:%S" if clock.count(":") == 2 else "%H:%M")
                ),
            )
        except ValueError:
            return None
    text = matches[0].group()
    for date_format in SERVICE_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return None


STRUCTURED_TEXT_PATTERNS = {
    "service_code": re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9./-]{3,30}"),
    "hsn_code": re.compile(r"[A-Za-z0-9./-]{3,30}"),
    "request_no": re.compile(r"[A-Za-z0-9./-]{3,60}"),
}


def structured_field_value_is_valid(field: str, value: str) -> bool:
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return False
    if field == "service_date":
        return DATE_FRAGMENT.search(text) is not None
    if field in STRUCTURED_TEXT_PATTERNS:
        if field == "hsn_code" and DATE_FRAGMENT.search(text):
            return False
        return STRUCTURED_TEXT_PATTERNS[field].fullmatch(text) is not None
    if field == "section":
        return parse_decimal(text) is None and len(text) <= 120
    if field == "description":
        return parse_decimal(text) is None and len(text) <= 500
    return False


def parse_alias_field_value(
    canonical_field: str, raw_value: str
) -> tuple[dict[str, Any], str, str] | None:
    """Parse a grounded source cell into canonical review changes."""
    value = re.sub(r"\s+", " ", raw_value).strip()
    if not value:
        return None
    if canonical_field in {
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }:
        parsed = parse_quantity(value) if canonical_field == "quantity" else parse_decimal(value)
        if parsed is None:
            return None
        rendered = format(parsed, "f")
        raw_field = {
            "quantity": "quantity_raw",
            "unit_price": "unit_price_raw",
            "gross_amount": "gross_amount_raw",
            "discount": "discount_raw",
            "net_amount": "net_amount_raw",
        }[canonical_field]
        evidence_field = {
            "unit_price": "rate",
            "net_amount": "amount",
        }.get(canonical_field, canonical_field)
        return (
            {canonical_field: rendered, raw_field: value},
            canonical_field,
            evidence_field,
        )
    if canonical_field == "service_date":
        parsed_date = parse_service_date(value)
        if parsed_date is None and not structured_field_value_is_valid(canonical_field, value):
            return None
        if parsed_date is None:
            return (
                {"service_date_raw": value, "service_date_iso": None},
                "service_date_raw",
                "service_date",
            )
        return (
            {"service_date_raw": value, "service_date_iso": parsed_date},
            "service_date_iso",
            "service_date",
        )
    if not structured_field_value_is_valid(canonical_field, value):
        return None
    return ({canonical_field: value}, canonical_field, canonical_field)
