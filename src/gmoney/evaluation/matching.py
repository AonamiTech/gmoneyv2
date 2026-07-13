from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Protocol


class RowLike(Protocol):
    page_number: int
    table_id: str | None
    row_order: int | None
    description: str | None
    request_no: str | None
    amount: Decimal | None


@dataclass(frozen=True)
class RowView:
    page_number: int
    table_id: str | None
    row_order: int | None
    description: str | None
    request_no: str | None
    amount: Decimal | None


def normalize_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(left: object, right: object) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def decimal_value(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def amount_matches(left: Decimal | None, right: Decimal | None) -> bool:
    return left is not None and right is not None and abs(left - right) <= Decimal("0.01")


def greedy_one_to_one(
    gold: Sequence[RowView],
    actual: Sequence[RowView],
    score,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold_row in enumerate(gold):
        for actual_index, actual_row in enumerate(actual):
            candidate_score = score(gold_row, actual_row)
            if candidate_score is not None:
                candidates.append((candidate_score, gold_index, actual_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gold: set[int] = set()
    matched_actual: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gold_index, actual_index in candidates:
        if gold_index in matched_gold or actual_index in matched_actual:
            continue
        matched_gold.add(gold_index)
        matched_actual.add(actual_index)
        matches.append((gold_index, actual_index))
    return sorted(matches)
