from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, median

from gmoney.contracts.evidence import OcrToken

ORGANIZATION_MARKERS = (
    "hospital",
    "hospitals",
    "medical centre",
    "medical center",
    "medical college",
    "cancer centre",
    "cancer center",
    "research institute",
    "healthcare centre",
    "healthcare center",
    "nursing home",
    "clinic",
)
REJECT_MARKERS = (
    "details of insured person",
    "hospitalized",
    "hospitalization",
    "claim form",
    "patient name",
    "invoice no",
    "invoice date",
    "bill no",
    "bill date",
    "credit invoice",
    "final bill",
    "address",
    "phone",
    "mobile",
    "email",
    "e mail",
    "website",
    "gst",
    "uhid",
    "ip no",
    "date of admission",
)
STOP_MARKERS = (
    " final bill",
    " credit invoice",
    " bill of supply",
    " address :",
    " address:",
    " phone no",
    " phone :",
    " phone:",
    " gstin",
    " gst no",
    " patient name",
    " invoice no",
)
TAGLINE_MARKERS = (
    "clinical excellence",
    "your health is our responsibility",
    "redefining health",
)
RUN_DATE_PREFIX = re.compile(
    r"^\s*run\s*date\s*:\s*\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}:\d{2}\s*",
    re.IGNORECASE,
)
UNIT_PREFIX = re.compile(r"^\s*\(?\s*a\s+unit\s+of\s+", re.IGNORECASE)
UNIT_CLAUSE = re.compile(r"\(?\s*a\s+unit\s+of\s+", re.IGNORECASE)


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


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _contains_normalized_phrase(value: str, phrase: str) -> bool:
    return f" {phrase} " in f" {value} "


@dataclass(frozen=True)
class HeaderLine:
    tokens: tuple[OcrToken, ...]

    @property
    def text(self) -> str:
        return " ".join(token.text.strip() for token in self.tokens if token.text.strip())

    @property
    def center_y(self) -> float:
        return median(_center_y(token) for token in self.tokens)


def _lines(tokens: tuple[OcrToken, ...]) -> tuple[HeaderLine, ...]:
    if not tokens:
        return ()
    tolerance = max(7.0, median(_height(token) for token in tokens) * 0.72)
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
    return tuple(HeaderLine(tuple(sorted(group, key=_center_x))) for group in grouped)


def _clean_name(value: str) -> str:
    value = RUN_DATE_PREFIX.sub("", value)
    value = re.sub(r"\bNABH\b", "", value, flags=re.IGNORECASE)
    if re.search(r"\bhospital", value, re.IGNORECASE):
        value = re.sub(r"^.*?\b(?:clinical\s+)?excellence\b\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\d{1,3}\s+(?=[A-Z]{3})", "", value)
    normalized = value.casefold()
    stop_indexes = [normalized.find(marker) for marker in STOP_MARKERS if marker in normalized]
    if stop_indexes:
        value = value[: min(stop_indexes)]
    unit_match = UNIT_CLAUSE.search(value)
    if unit_match:
        brand = value[: unit_match.start()].strip(" -|(),")
        operator = value[unit_match.end() :].split(")", 1)[0].strip(" -|(),")
        brand_normalized = _normalize(brand)
        value = (
            brand
            if any(
                _contains_normalized_phrase(brand_normalized, marker)
                for marker in ORGANIZATION_MARKERS
            )
            else operator
        )
    value = UNIT_PREFIX.sub("", value)
    value = re.sub(r"\bredefining\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(\w+)(?:\s+\1)+\s+", r"\1 ", value, flags=re.IGNORECASE)
    hospital_match = re.search(r"\b(\w+)\s+(hospitals?)\b", value, re.IGNORECASE)
    if hospital_match:
        brand_word = hospital_match.group(1)
        earlier = [
            match.start()
            for match in re.finditer(
                rf"\b{re.escape(brand_word)}\b",
                value[: hospital_match.start()],
                re.IGNORECASE,
            )
        ]
        if earlier and brand_word.isupper():
            value = value[hospital_match.start(1) :]
    if re.search(r"\bhospital\b", value, re.IGNORECASE):
        value = re.sub(r"\s+HOSPITALS\s*$", "", value)
    value = re.sub(
        r"^(\w+)(.+\bhospitals?)\s+\1$",
        lambda match: f"{match.group(1)}{match.group(2)}",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" -|(),")
    if value.isupper():
        value = value.title()
        value = re.sub(r"\b(Of|And|The)\b", lambda match: match.group(0).lower(), value)
    value = re.sub(r"\bPvt\s+Ltd\b", "Pvt. Ltd.", value, flags=re.IGNORECASE)
    return value[:180].strip()


def _candidate_score(text: str, tokens: tuple[OcrToken, ...], page_height: int) -> float:
    normalized = _normalize(text)
    organization_markers = tuple(
        marker
        for marker in ORGANIZATION_MARKERS
        if _contains_normalized_phrase(normalized, marker)
    )
    if not normalized or not organization_markers:
        return -1.0
    if any(marker in normalized for marker in REJECT_MARKERS):
        return -1.0
    if len(re.sub(r"[^a-z]", "", normalized)) < 8:
        return -1.0
    digit_fraction = sum(character.isdigit() for character in text) / max(1, len(text))
    if digit_fraction > 0.2:
        return -1.0
    top = min(_bounds(token)[1] for token in tokens)
    confidence = mean(token.confidence for token in tokens)
    marker_strength = max(len(marker) for marker in organization_markers) / 20
    tagline_penalty = 1.25 if any(marker in normalized for marker in TAGLINE_MARKERS) else 0.0
    length_penalty = max(0, len(text) - 30) * 0.005
    return (
        4.0
        + marker_strength
        + confidence
        + max(0.0, 1.0 - top / (page_height * 0.35))
        - tagline_penalty
        - length_penalty
    )


def detect_hospital(
    tokens: tuple[OcrToken, ...],
    *,
    page_width: int,
    page_height: int,
) -> dict[str, object] | None:
    """Return a conservative, OCR-evidenced hospital identity from page-one headers."""
    header_tokens = tuple(
        token
        for token in tokens
        if token.text.strip()
        and token.confidence >= 0.35
        and _bounds(token)[1] <= page_height * 0.34
    )
    lines = _lines(header_tokens)
    candidates: list[tuple[float, str, tuple[OcrToken, ...]]] = []
    for index, line in enumerate(lines):
        spans = [(line.text, line.tokens)]
        if index > 0:
            previous = lines[index - 1]
            if line.center_y - previous.center_y <= page_height * 0.06:
                spans.append((f"{previous.text} {line.text}", (*previous.tokens, *line.tokens)))
        if index + 1 < len(lines):
            following = lines[index + 1]
            if following.center_y - line.center_y <= page_height * 0.06:
                spans.append((f"{line.text} {following.text}", (*line.tokens, *following.tokens)))
        for text, span_tokens in spans:
            cleaned = _clean_name(text)
            score = _candidate_score(cleaned, span_tokens, page_height)
            if score >= 0:
                candidates.append((score, cleaned, span_tokens))
    if not candidates:
        return None
    score, name, selected = max(candidates, key=lambda item: (item[0], len(item[1])))
    left = min(_bounds(token)[0] for token in selected)
    top = min(_bounds(token)[1] for token in selected)
    right = max(_bounds(token)[2] for token in selected)
    bottom = max(_bounds(token)[3] for token in selected)
    artifact_sha256 = selected[0].artifact_sha256
    confidence = min(0.99, max(0.5, (score - 3.5) / 3.5))
    return {
        "name": name,
        "confidence": round(confidence, 4),
        "source": "machine",
        "page_number": 1,
        "evidence": {
            "page_number": 1,
            "table_id": None,
            "polygon": {
                "points": [
                    {"x": max(0.0, left), "y": max(0.0, top)},
                    {"x": min(float(page_width), right), "y": max(0.0, top)},
                    {"x": min(float(page_width), right), "y": min(float(page_height), bottom)},
                    {"x": max(0.0, left), "y": min(float(page_height), bottom)},
                ]
            },
            "artifact_sha256": artifact_sha256,
            "token_ids": [token.token_id for token in selected],
        },
    }
