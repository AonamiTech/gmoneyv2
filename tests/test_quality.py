from decimal import Decimal

from gmoney.evaluation.matching import RowView
from gmoney.evaluation.quality import evaluate_phase2_quality


def view(index: int, description: str, amount: str) -> RowView:
    return RowView(
        page_number=1,
        table_id=None,
        row_order=index,
        description=description,
        request_no=None,
        amount=Decimal(amount),
    )


def test_phase2_gate_requires_row_and_amount_quality() -> None:
    gold = tuple(view(index, f"row {index}", str(index + 1)) for index in range(20))
    assert evaluate_phase2_quality(gold, gold).passed is True

    wrong_amount = (*gold[:-1], view(19, "row 19", "999"))
    result = evaluate_phase2_quality(gold, wrong_amount)
    assert result.canonical_recall == 0.95
    assert result.amount_accuracy == 0.95
    assert result.passed is True


def test_phase2_gate_rejects_accepted_ungrounded_rows() -> None:
    rows = (view(0, "refund", "-10"),)
    result = evaluate_phase2_quality(rows, rows, accepted_ungrounded_rows=1)
    assert result.negative_recall == 1
    assert result.passed is False


def test_amount_accuracy_aligns_repeated_descriptions_in_print_order() -> None:
    gold = (
        view(0, "ECG-BED SIDE", "700"),
        view(1, "ECG-BED SIDE", "500"),
        view(2, "GRBS", "250"),
        view(3, "GRBS", "150"),
    )
    actual = (
        view(0, "ECG-BED SIDE watermark", "700"),
        view(1, "ECG-BED SIDE", "500"),
        view(2, "GRBS watermark", "250"),
        view(3, "GRBS", "150"),
    )
    assert evaluate_phase2_quality(gold, actual).amount_accuracy == 1.0
