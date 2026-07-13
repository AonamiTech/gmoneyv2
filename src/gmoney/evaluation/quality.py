from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

from gmoney.evaluation.matching import RowView, amount_matches, similarity
from gmoney.evaluation.metrics import canonical_metric_v2, legacy_metric_v1


@dataclass(frozen=True)
class Phase2QualityResult:
    legacy_precision: float
    legacy_recall: float
    legacy_f1: float
    canonical_precision: float
    canonical_recall: float
    canonical_f1: float
    amount_accuracy: float
    amount_compared_rows: int
    negative_recall: float
    accepted_ungrounded_rows: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _description_matches(
    gold: tuple[RowView, ...],
    actual: tuple[RowView, ...],
) -> list[tuple[int, int]]:
    """Match row structure monotonically before scoring the amount field.

    Description-only greedy matching crosses identical repeated services and can
    make correct amounts appear wrong. Hospital ledgers have a meaningful print
    order, so each page is aligned as a maximum-cardinality monotonic sequence.
    Amount is intentionally absent from this step.
    """
    output: list[tuple[int, int]] = []
    pages = sorted({row.page_number for row in (*gold, *actual)})
    for page_number in pages:
        gold_indexes = [index for index, row in enumerate(gold) if row.page_number == page_number]
        actual_indexes = [
            index for index, row in enumerate(actual) if row.page_number == page_number
        ]
        rows = len(gold_indexes)
        columns = len(actual_indexes)
        scores: list[list[tuple[int, float]]] = [
            [(0, 0.0) for _ in range(columns + 1)] for _ in range(rows + 1)
        ]
        actions = [["" for _ in range(columns + 1)] for _ in range(rows + 1)]
        for row_index in range(1, rows + 1):
            actions[row_index][0] = "skip_gold"
        for column_index in range(1, columns + 1):
            actions[0][column_index] = "skip_actual"
        for row_index in range(1, rows + 1):
            for column_index in range(1, columns + 1):
                choices = [
                    (scores[row_index - 1][column_index], "skip_gold"),
                    (scores[row_index][column_index - 1], "skip_actual"),
                ]
                gold_index = gold_indexes[row_index - 1]
                actual_index = actual_indexes[column_index - 1]
                description_score = similarity(
                    gold[gold_index].description,
                    actual[actual_index].description,
                )
                if description_score >= 0.55:
                    previous_count, previous_score = scores[row_index - 1][column_index - 1]
                    choices.append(
                        (
                            (previous_count + 1, previous_score + description_score),
                            "match",
                        )
                    )
                scores[row_index][column_index], actions[row_index][column_index] = max(
                    choices,
                    key=lambda choice: (choice[0][0], choice[0][1], choice[1] == "match"),
                )

        page_matches: list[tuple[int, int]] = []
        row_index = rows
        column_index = columns
        while row_index or column_index:
            action = actions[row_index][column_index]
            if action == "match":
                page_matches.append((gold_indexes[row_index - 1], actual_indexes[column_index - 1]))
                row_index -= 1
                column_index -= 1
            elif action == "skip_gold":
                row_index -= 1
            else:
                column_index -= 1
        output.extend(reversed(page_matches))
    return output


def evaluate_phase2_quality(
    gold: tuple[RowView, ...],
    actual: tuple[RowView, ...],
    *,
    accepted_ungrounded_rows: int = 0,
    row_threshold: float = 0.85,
    amount_threshold: float = 0.95,
) -> Phase2QualityResult:
    legacy = legacy_metric_v1(gold, actual)
    canonical = canonical_metric_v2(gold, actual)
    description_matches = _description_matches(gold, actual)
    correct_amounts = sum(
        amount_matches(gold[left].amount, actual[right].amount)
        for left, right in description_matches
    )
    amount_accuracy = correct_amounts / len(description_matches) if description_matches else 0.0

    negative_gold = tuple(row for row in gold if (row.amount or Decimal("0")) < 0)
    negative_actual = tuple(row for row in actual if (row.amount or Decimal("0")) < 0)
    negative_recall = legacy_metric_v1(negative_gold, negative_actual).recall
    passed = bool(
        min(
            legacy.precision,
            legacy.recall,
            legacy.f1,
            canonical.precision,
            canonical.recall,
            canonical.f1,
        )
        >= row_threshold
        and amount_accuracy >= amount_threshold
        and negative_recall == 1.0
        and accepted_ungrounded_rows == 0
    )
    return Phase2QualityResult(
        legacy_precision=legacy.precision,
        legacy_recall=legacy.recall,
        legacy_f1=legacy.f1,
        canonical_precision=canonical.precision,
        canonical_recall=canonical.recall,
        canonical_f1=canonical.f1,
        amount_accuracy=amount_accuracy,
        amount_compared_rows=len(description_matches),
        negative_recall=negative_recall,
        accepted_ungrounded_rows=accepted_ungrounded_rows,
        passed=passed,
    )
