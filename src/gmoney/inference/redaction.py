from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.evaluation.corpus import sha256_file

SENSITIVE_LABELS = re.compile(
    r"\b(patient|patient name|uhid|mrn|ipid|ip no|mobile|phone|address|email|dob|date of birth|"
    r"aadhaar|aadhar|pan no|policy no|claim no)\b",
    re.IGNORECASE,
)
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)")
LONG_IDENTIFIER = re.compile(r"(?<!\d)\d{8,16}(?!\d)")


@dataclass(frozen=True)
class RedactionResult:
    path: Path
    artifact_sha256: str
    masked_token_ids: tuple[str, ...]
    tokens: tuple[OcrToken, ...]
    safe: bool
    reasons: tuple[str, ...]


def _bounds(token: OcrToken) -> tuple[float, float, float, float]:
    xs = [point.x for point in token.polygon.points]
    ys = [point.y for point in token.polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def _center_y(token: OcrToken) -> float:
    _, top, _, bottom = _bounds(token)
    return (top + bottom) / 2


def _height(token: OcrToken) -> float:
    _, top, _, bottom = _bounds(token)
    return bottom - top


def _crop_token(token: OcrToken, left: float, top: float) -> OcrToken:
    return token.model_copy(
        update={
            "polygon": Polygon(
                points=tuple(
                    Point(x=max(0, point.x - left), y=max(0, point.y - top))
                    for point in token.polygon.points
                )
            )
        }
    )


def _direct_sensitive(text: str) -> bool:
    return bool(EMAIL.search(text) or PHONE.search(text) or LONG_IDENTIFIER.search(text))


def redact_crop(
    source: Path,
    output: Path,
    tokens: tuple[OcrToken, ...],
    crop_box: tuple[int, int, int, int],
) -> RedactionResult:
    """Mask high-risk identifiers and return only sanitized crop-local OCR tokens."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read crop for redaction: {source}")
    left, top, right, bottom = crop_box
    scoped = tuple(
        token
        for token in tokens
        if (_bounds(token)[0] < right and _bounds(token)[2] > left)
        and (_bounds(token)[1] < bottom and _bounds(token)[3] > top)
    )
    if not scoped:
        return RedactionResult(output, "0" * 64, (), (), False, ("no_ocr_coverage",))

    heights = [_height(token) for token in scoped]
    line_tolerance = max(6.0, median(heights) * 0.75)
    masked: set[str] = set()
    for token in scoped:
        if SENSITIVE_LABELS.search(token.text) or _direct_sensitive(token.text):
            masked.add(token.token_id)
        if not SENSITIVE_LABELS.search(token.text):
            continue
        token_left, _, token_right, _ = _bounds(token)
        for candidate in scoped:
            candidate_left, _, _, _ = _bounds(candidate)
            vertical_delta = _center_y(candidate) - _center_y(token)
            same_line = abs(vertical_delta) <= line_tolerance
            following_value_line = (
                0 < vertical_delta <= line_tolerance * 2.5
                and candidate_left >= token_left - image.shape[1] * 0.05
            )
            if (
                candidate.token_id != token.token_id
                and (same_line or following_value_line)
                and candidate_left >= token_left
                and candidate_left <= token_right + image.shape[1] * 0.45
            ):
                masked.add(candidate.token_id)

    sanitized: list[OcrToken] = []
    for token in scoped:
        local = _crop_token(token, left, top)
        if token.token_id in masked:
            points = np.array(
                [
                    [
                        [
                            max(0, min(image.shape[1] - 1, round(point.x))),
                            max(0, min(image.shape[0] - 1, round(point.y))),
                        ]
                        for point in local.polygon.points
                    ]
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(image, points, color=(0, 0, 0))
            local = local.model_copy(update={"text": "[REDACTED]"})
        sanitized.append(local)

    surviving_sensitive = [
        token.token_id
        for token in sanitized
        if token.token_id not in masked
        and (SENSITIVE_LABELS.search(token.text) or _direct_sensitive(token.text))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write redacted crop: {temporary}")
    temporary.replace(output)
    reasons = ("surviving_sensitive_tokens",) if surviving_sensitive else ()
    return RedactionResult(
        path=output,
        artifact_sha256=sha256_file(output),
        masked_token_ids=tuple(sorted(masked)),
        tokens=tuple(sanitized),
        safe=not surviving_sensitive,
        reasons=reasons,
    )
