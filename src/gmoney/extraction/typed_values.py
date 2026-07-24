from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

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
    text = str(value or "")
    duration = DAY_QUANTITY.fullmatch(text)
    return parse_decimal(duration.group("quantity") if duration is not None else text)


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
            r"[,;-]?\s*(?P<clock>\d{1,2}:\d{2}(?::\d{2})?)"
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
