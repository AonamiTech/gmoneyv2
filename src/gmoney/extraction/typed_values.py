from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

CURRENCY = re.compile(r"(?:₹|inr|rs\.?|rupees?)", re.IGNORECASE)
NUMERIC = re.compile(r"^[+-]?\d+(?:\.\d{1,4})?$")
MAX_ABSOLUTE = Decimal("999999999999.9999")

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


def parse_service_date(value: object) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -:")
    if not text:
        return None
    for date_format in SERVICE_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return None
