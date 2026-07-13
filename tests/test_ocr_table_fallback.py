import pytest

from gmoney.inference.ocr_table_fallback import propose_tables_from_ocr


def test_aligned_borderless_numeric_rows_produce_table_region() -> None:
    boxes = [
        [10, 10, 80, 25],
        [120, 10, 170, 25],
        [200, 10, 250, 25],
        [10, 35, 80, 50],
        [120, 35, 170, 50],
        [200, 35, 250, 50],
        [10, 60, 80, 75],
        [120, 60, 170, 75],
        [200, 60, 250, 75],
    ]
    texts = ["Medicine", "2", "100.00", "Nursing", "1", "500", "Doctor", "1", "2000"]
    proposals = propose_tables_from_ocr(boxes, texts)
    assert len(proposals) == 1
    assert proposals[0].row_count == 3
    assert proposals[0].box == (10.0, 10.0, 250.0, 75.0)
    assert proposals[0].confidence >= 0.7


def test_narrative_text_does_not_become_table() -> None:
    boxes = [[10, 10, 80, 25], [90, 10, 160, 25], [170, 10, 250, 25]]
    assert propose_tables_from_ocr(boxes, ["Patient", "was", "discharged"]) == []


def test_mismatched_ocr_arrays_are_rejected() -> None:
    with pytest.raises(ValueError):
        propose_tables_from_ocr([[0, 0, 1, 1]], [])
