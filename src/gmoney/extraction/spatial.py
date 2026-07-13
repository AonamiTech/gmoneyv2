from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from gmoney.contracts.evidence import OcrToken
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.typed_values import parse_decimal


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


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


def _text_score(expected: str, actual: str) -> float:
    left, right = _normalize(expected), _normalize(actual)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right)) * 0.9
    return SequenceMatcher(None, left, right).ratio()


@dataclass(frozen=True)
class AlignedLedgerRow:
    candidate: CandidateLedgerRow
    field_token_ids: dict[str, tuple[str, ...]]
    evidence_token_ids: tuple[str, ...]
    evidence_box: tuple[float, float, float, float] | None
    grounding_ratio: float


def _description_anchor(
    description: str,
    tokens: tuple[OcrToken, ...],
    minimum_y: float,
) -> tuple[int, float] | None:
    scored: list[tuple[float, float, int]] = []
    heights = [_height(token) for token in tokens]
    tolerance = (median(heights) if heights else 10) * 0.75
    for index, token in enumerate(tokens):
        center_y = _center_y(token)
        if center_y < minimum_y - tolerance:
            continue
        score = _text_score(description, token.text)
        if score >= 0.5:
            scored.append((-score, center_y, index))
    if not scored:
        return None
    _, center_y, index = min(scored)
    return index, center_y


def _numeric_token(
    value,
    tokens: tuple[OcrToken, ...],
    anchor_y: float,
    used: set[int],
) -> int | None:
    if value is None:
        return None
    candidates: list[tuple[float, int]] = []
    for index, token in enumerate(tokens):
        if index in used or parse_decimal(token.text) != value:
            continue
        tolerance = max(12.0, _height(token) * 1.5)
        distance = abs(_center_y(token) - anchor_y)
        if distance <= tolerance:
            candidates.append((distance, index))
    return min(candidates)[1] if candidates else None


def align_candidate_rows(
    candidates: tuple[CandidateLedgerRow, ...],
    tokens: tuple[OcrToken, ...],
) -> tuple[AlignedLedgerRow, ...]:
    aligned: list[AlignedLedgerRow] = []
    minimum_y = 0.0
    used_description_tokens: set[int] = set()
    for candidate in candidates:
        fields: dict[str, tuple[str, ...]] = {}
        selected: set[int] = set()
        anchor = None
        if candidate.description:
            available = tuple(
                token for index, token in enumerate(tokens) if index not in used_description_tokens
            )
            available_indexes = [
                index for index in range(len(tokens)) if index not in used_description_tokens
            ]
            available_anchor = _description_anchor(candidate.description, available, minimum_y)
            if available_anchor:
                local_index, anchor_y = available_anchor
                token_index = available_indexes[local_index]
                anchor = (token_index, anchor_y)
                selected.add(token_index)
                used_description_tokens.add(token_index)
                fields["description"] = (tokens[token_index].token_id,)
                minimum_y = anchor_y
        if anchor:
            _, anchor_y = anchor
            for field, value in (
                ("quantity", candidate.quantity),
                ("rate", candidate.rate),
                ("discount", candidate.discount),
                ("amount", candidate.amount),
            ):
                token_index = _numeric_token(value, tokens, anchor_y, selected)
                if token_index is not None:
                    selected.add(token_index)
                    fields[field] = (tokens[token_index].token_id,)
        evidence_box = None
        if selected:
            boxes = [_bounds(tokens[index]) for index in selected]
            evidence_box = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
        expected_fields = 1 + sum(
            value is not None
            for value in (candidate.quantity, candidate.rate, candidate.discount, candidate.amount)
        )
        aligned.append(
            AlignedLedgerRow(
                candidate=candidate,
                field_token_ids=fields,
                evidence_token_ids=tuple(tokens[index].token_id for index in sorted(selected)),
                evidence_box=evidence_box,
                grounding_ratio=len(fields) / expected_fields,
            )
        )
    return tuple(aligned)
