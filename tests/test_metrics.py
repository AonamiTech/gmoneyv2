import json
from pathlib import Path

from gmoney.evaluation.metrics import canonical_metric_v2, legacy_metric_v1, row_view

FIXTURES = Path(__file__).parent / "fixtures"


def load_rows(name: str) -> list:
    payload = json.loads((FIXTURES / name).read_text())
    return [row_view(row, index) for index, row in enumerate(payload["rows"])]


def test_frozen_metrics_are_deterministic_and_perfect_for_exact_rows() -> None:
    gold = load_rows("gold_rows.json")
    actual = load_rows("actual_rows.json")
    for metric in (legacy_metric_v1, canonical_metric_v2):
        first = metric(gold, actual)
        second = metric(gold, actual)
        assert first == second
        assert first.precision == first.recall == first.f1 == 1.0


def test_one_to_one_matching_preserves_legitimate_duplicate_accounting() -> None:
    gold = [row_view({"page_number": 1, "description": "Needle", "amount": 5}, i) for i in range(2)]
    actual = [row_view({"page_number": 1, "description": "Needle", "amount": 5}, 0)]
    result = canonical_metric_v2(gold, actual)
    assert result.matched_rows == 1
    assert result.precision == 1.0
    assert result.recall == 0.5


def test_wrong_page_does_not_match() -> None:
    gold = [row_view({"page_number": 1, "description": "Needle", "amount": 5}, 0)]
    actual = [row_view({"page_number": 2, "description": "Needle", "amount": 5}, 0)]
    assert canonical_metric_v2(gold, actual).matched_rows == 0


def test_frozen_legacy_baseline_counts_reproduce_reported_metrics() -> None:
    baseline_path = Path(__file__).parents[1] / "corpus" / "baselines.json"
    baseline = json.loads(baseline_path.read_text())["baselines"][0]
    precision = baseline["matched_rows"] / baseline["visible_extracted_rows"]
    recall = baseline["matched_rows"] / baseline["gold_rows"]
    assert round(precision, 4) == baseline["row_precision"]
    assert round(recall, 4) == baseline["row_recall"]
