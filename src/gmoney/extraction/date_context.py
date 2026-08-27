from __future__ import annotations

import re

from gmoney.extraction.typed_values import parse_service_date

DATE_FRAGMENT = re.compile(
    r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\d)"
)

_NON_SERVICE_MARKERS = (
    "admission date",
    "admitted on",
    "batch",
    "bill date",
    "date of admission",
    "date of birth",
    "date of discharge",
    "discharge date",
    "dob",
    "exp date",
    "expdate",
    "expiry",
    "manufacturing date",
    "medicine",
    "mfg date",
    "print date",
    "print time",
    "receipt date",
)


def normalized_date_context(*values: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        " ".join(str(value or "") for value in values).casefold(),
    ).strip()


def is_service_date_context(
    raw_value: object,
    *,
    column_label: object = None,
    description: object = None,
) -> bool:
    context = normalized_date_context(column_label, raw_value, description)
    if not context or any(marker in context for marker in _NON_SERVICE_MARKERS):
        return False
    return re.search(r"(?:^|\s)(?:exp|mfg)(?:\s|$|date\b)", context) is None


def service_date_from_context(
    raw_value: object,
    *,
    column_label: object = None,
    description: object = None,
) -> tuple[str, str] | None:
    raw = re.sub(r"\s+", " ", str(raw_value or "")).strip()
    if not raw or not is_service_date_context(
        raw,
        column_label=column_label,
        description=description,
    ):
        return None
    direct = parse_service_date(raw)
    if direct is not None:
        return raw, direct
    matches = tuple(DATE_FRAGMENT.finditer(raw))
    if len(matches) != 1:
        return None
    parsed = parse_service_date(matches[0].group(0))
    return (matches[0].group(0), parsed) if parsed is not None else None
