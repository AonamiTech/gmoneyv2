from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

from gmoney.evaluation.matching import (
    RowView,
    amount_matches,
    greedy_one_to_one,
    normalize_text,
    similarity,
)


@dataclass(frozen=True)
class MetricResult:
    metric_version: str
    gold_rows: int
    actual_rows: int
    matched_rows: int
    precision: float
    recall: float
    f1: float
    false_negative_indexes: tuple[int, ...]
    false_positive_indexes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _result(
    version: str,
    gold_count: int,
    actual_count: int,
    matches: list[tuple[int, int]],
) -> MetricResult:
    matched = len(matches)
    precision = matched / actual_count if actual_count else (1.0 if gold_count == 0 else 0.0)
    recall = matched / gold_count if gold_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_gold = {left for left, _ in matches}
    matched_actual = {right for _, right in matches}
    return MetricResult(
        metric_version=version,
        gold_rows=gold_count,
        actual_rows=actual_count,
        matched_rows=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        false_negative_indexes=tuple(i for i in range(gold_count) if i not in matched_gold),
        false_positive_indexes=tuple(i for i in range(actual_count) if i not in matched_actual),
    )


def legacy_metric_v1(gold: Sequence[RowView], actual: Sequence[RowView]) -> MetricResult:
    def score(left: RowView, right: RowView) -> float | None:
        if left.page_number != right.page_number or not amount_matches(left.amount, right.amount):
            return None
        description_score = similarity(left.description, right.description)
        request_match = bool(
            normalize_text(left.request_no)
            and normalize_text(left.request_no) == normalize_text(right.request_no)
        )
        if description_score < 0.35 and not request_match:
            return None
        return description_score + (0.2 if request_match else 0.0)

    matches = greedy_one_to_one(gold, actual, score)
    return _result("legacy_metric_v1", len(gold), len(actual), matches)


def canonical_metric_v2(gold: Sequence[RowView], actual: Sequence[RowView]) -> MetricResult:
    def score(left: RowView, right: RowView) -> float | None:
        if left.page_number != right.page_number:
            return None
        if left.table_id and right.table_id and left.table_id != right.table_id:
            return None
        description_score = similarity(left.description, right.description)
        request_match = bool(
            normalize_text(left.request_no)
            and normalize_text(left.request_no) == normalize_text(right.request_no)
        )
        if not amount_matches(left.amount, right.amount):
            return None
        if description_score < 0.55 and not request_match:
            return None
        order_score = 0.0
        if left.row_order is not None and right.row_order is not None:
            distance = abs(left.row_order - right.row_order)
            order_score = max(0.0, 0.2 - min(distance, 4) * 0.05)
        table_score = 0.15 if left.table_id and left.table_id == right.table_id else 0.0
        return description_score + order_score + table_score + (0.2 if request_match else 0.0)

    matches = greedy_one_to_one(gold, actual, score)
    return _result("canonical_metric_v2", len(gold), len(actual), matches)


def row_view(payload: dict[str, object], index: int) -> RowView:
    amount = payload.get("amount", payload.get("net_amount", payload.get("gross_amount")))
    return RowView(
        page_number=int(payload.get("page_number", payload.get("page", 0)) or 0),
        table_id=str(payload["table_id"]) if payload.get("table_id") else None,
        row_order=int(payload["row_order"]) if payload.get("row_order") is not None else index,
        description=str(payload.get("description") or payload.get("raw_text") or ""),
        request_no=str(payload["request_no"]) if payload.get("request_no") else None,
        amount=Decimal(str(amount)) if amount is not None else None,
    )
