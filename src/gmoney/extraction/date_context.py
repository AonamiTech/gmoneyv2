from __future__ import annotations

import re

from gmoney.extraction.typed_values import parse_service_date

DATE_FRAGMENT = re.compile(
    r"(?<!\d)(?:\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|"
    r"\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})(?!\d)"
)
PRINTED_DATE_REQUEST_SUFFIX = re.compile(
    r"\s*[-:]?\s*[A-Z][A-Z0-9-]{2,}/[A-Z0-9-]+"
    r"(?:\s+(?P<bleed>[A-Z]{1,2}))?\s*$",
    re.IGNORECASE,
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
    canonical_field: object = None,
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
    canonical_field: object = None,
    description: object = None,
) -> tuple[str, str] | None:
    raw = re.sub(r"\s+", " ", str(raw_value or "")).strip()
    normalized_label = normalized_date_context(column_label)
    direct = parse_service_date(raw)
    explicitly_mapped = str(canonical_field or "") in {
        "service_date",
        "service_date_raw",
    }
    if direct is not None and (explicitly_mapped or normalized_label in {
        "date",
        "date time",
        "dos",
        "dt",
        "service date",
        "service date time",
        "service dt",
    }):
        # A date-only value in an explicitly mapped service-date lane is local
        # evidence. Product words elsewhere in the row must not turn it into an
        # expiry/batch date; those markers still apply to embedded dates.
        return raw, direct
    if not raw or not is_service_date_context(
        raw,
        column_label=column_label,
        canonical_field=canonical_field,
        description=description,
    ):
        return None
    if direct is not None:
        return raw, direct
    request_suffix = PRINTED_DATE_REQUEST_SUFFIX.search(raw)
    if request_suffix is not None:
        parsed_prefix = parse_service_date(raw[: request_suffix.start()])
        bleed = normalized_date_context(request_suffix.group("bleed"))
        first_description_word = next(
            iter(normalized_date_context(description).split()), ""
        )
        if parsed_prefix is not None and (
            not bleed
            or first_description_word.startswith(bleed)
            or first_description_word.endswith(bleed)
        ):
            date_match = DATE_FRAGMENT.search(raw)
            return (
                date_match.group(0) if date_match is not None else raw,
                parsed_prefix,
            )
        return None
    matches = tuple(DATE_FRAGMENT.finditer(raw))
    if len(matches) != 1:
        return None
    match = matches[0]
    suffix = raw[match.end() :].strip(" ()[]{}:;,-")
    if suffix:
        return None
    parsed = parse_service_date(match.group(0))
    return (matches[0].group(0), parsed) if parsed is not None else None
