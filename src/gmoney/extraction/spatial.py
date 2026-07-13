from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
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
    source_routes: tuple[str, ...] = ("provider_otsl",)


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
    minimum_x: float = 0.0,
) -> int | None:
    if value is None:
        return None
    candidates: list[tuple[float, int]] = []
    for index, token in enumerate(tokens):
        if index in used or value not in _numeric_values(token.text):
            continue
        tolerance = max(8.0, _height(token) * 0.6)
        distance = abs(_center_y(token) - anchor_y)
        left, _, _, _ = _bounds(token)
        if distance <= tolerance and left >= minimum_x - 5:
            candidates.append((distance + left / 1_000_000, index))
    return min(candidates)[1] if candidates else None


def _numeric_values(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    whole = parse_decimal(text)
    if whole is not None:
        values.append(whole)
    parts = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    if len(parts) >= 2:
        values.extend(value for part in parts if (value := parse_decimal(part)) is not None)
    return tuple(dict.fromkeys(values))


def _derived_amount_token(
    tokens: tuple[OcrToken, ...],
    anchor_y: float,
    description_index: int,
    quantity_index: int | None,
    rate_index: int | None,
) -> tuple[int, Decimal] | None:
    if quantity_index is not None:
        quantity_values = _numeric_values(tokens[quantity_index].text)
        if len(quantity_values) >= 2:
            return quantity_index, quantity_values[-1]
        _, _, after_x, _ = _bounds(tokens[quantity_index])
    elif rate_index is not None:
        _, _, after_x, _ = _bounds(tokens[rate_index])
    else:
        _, _, after_x, _ = _bounds(tokens[description_index])

    candidates: list[tuple[float, int, Decimal]] = []
    for index, token in enumerate(tokens):
        values = _numeric_values(token.text)
        if not values:
            continue
        tolerance = max(8.0, _height(token) * 0.6)
        if abs(_center_y(token) - anchor_y) > tolerance:
            continue
        left, _, _, _ = _bounds(token)
        if left < after_x - 5:
            continue
        candidates.append((left, index, values[-1]))
    if not candidates:
        return None
    _, index, value = min(candidates)
    return index, value


def align_candidate_rows(
    candidates: tuple[CandidateLedgerRow, ...],
    tokens: tuple[OcrToken, ...],
) -> tuple[AlignedLedgerRow, ...]:
    aligned: list[AlignedLedgerRow] = []
    minimum_y = 0.0
    used_description_tokens: set[int] = set()
    for candidate in candidates:
        aligned_candidate = candidate
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
            description_index, anchor_y = anchor
            _, _, description_right, _ = _bounds(tokens[description_index])
            minimum_x = description_right
            for field, value in (
                ("rate", aligned_candidate.rate),
                ("quantity", aligned_candidate.quantity),
                ("discount", aligned_candidate.discount),
            ):
                token_index = _numeric_token(
                    value,
                    tokens,
                    anchor_y,
                    selected,
                    minimum_x=minimum_x,
                )
                if token_index is not None:
                    selected.add(token_index)
                    fields[field] = (tokens[token_index].token_id,)
                    if field in {"rate", "quantity"}:
                        _, _, minimum_x, _ = _bounds(tokens[token_index])
            if candidate.amount_derived:
                quantity_id = fields.get("quantity", (None,))[0]
                rate_id = fields.get("rate", (None,))[0]
                quantity_index = next(
                    (i for i, token in enumerate(tokens) if token.token_id == quantity_id),
                    None,
                )
                rate_index = next(
                    (i for i, token in enumerate(tokens) if token.token_id == rate_id),
                    None,
                )
                derived = _derived_amount_token(
                    tokens,
                    anchor_y,
                    description_index,
                    quantity_index,
                    rate_index,
                )
                if derived:
                    token_index, value = derived
                    aligned_candidate = replace(candidate, amount=value)
                    selected.add(token_index)
                    fields["amount"] = (tokens[token_index].token_id,)
            else:
                token_index = _numeric_token(
                    aligned_candidate.amount,
                    tokens,
                    anchor_y,
                    selected,
                    minimum_x=minimum_x,
                )
                if token_index is not None:
                    selected.add(token_index)
                    fields["amount"] = (tokens[token_index].token_id,)
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
            for value in (
                aligned_candidate.quantity,
                aligned_candidate.rate,
                aligned_candidate.discount,
                aligned_candidate.amount,
            )
        )
        aligned.append(
            AlignedLedgerRow(
                candidate=aligned_candidate,
                field_token_ids=fields,
                evidence_token_ids=tuple(tokens[index].token_id for index in sorted(selected)),
                evidence_box=evidence_box,
                grounding_ratio=len(fields) / expected_fields,
                source_routes=(aligned_candidate.source_route,),
            )
        )
    return tuple(aligned)
