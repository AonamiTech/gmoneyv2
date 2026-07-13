from decimal import Decimal

from gmoney.evaluation.matching import RowView
from gmoney.evaluation.phase3 import Phase3Document, evaluate_phase3_quality


def rows(count: int, *, negative: bool = False) -> tuple[RowView, ...]:
    return tuple(
        RowView(
            page_number=1,
            table_id=None,
            row_order=index,
            description=f"Service {index}",
            request_no=None,
            amount=Decimal("-1.00") if negative and index == 0 else Decimal(index + 1),
        )
        for index in range(count)
    )


def test_phase3_gate_passes_complete_exact_cohorts() -> None:
    documents = []
    for index in range(10):
        document_rows = rows(2, negative=index == 0)
        documents.append(
            Phase3Document(
                document_id=f"unseen-{index}",
                hospital_id=f"UH{index}",
                cohort="unseen",
                route="local_recovery",
                gold=document_rows,
                actual=document_rows,
            )
        )
    for index in range(10):
        document_rows = rows(20, negative=index == 0)
        documents.append(
            Phase3Document(
                document_id=f"known-{index}",
                hospital_id="KNOWN",
                cohort="known_active",
                route="profile_fast",
                gold=document_rows,
                actual=document_rows,
                ordinary_active=True,
            )
        )
    result = evaluate_phase3_quality(tuple(documents), bootstrap_samples=50)
    assert result.passed
    assert not result.blocking_reasons
    assert result.active_zero_gemini_fraction == 1


def test_phase3_gate_blocks_missing_profile_and_unseen_hospitals() -> None:
    document_rows = rows(2)
    result = evaluate_phase3_quality(
        (
            Phase3Document(
                document_id="only",
                hospital_id="H1",
                cohort="unseen",
                route="local",
                gold=document_rows,
                actual=document_rows,
            ),
        ),
        bootstrap_samples=10,
    )
    assert not result.passed
    assert "unseen_10_hospital_gate_missing" in result.blocking_reasons
    assert "known_active_profile_gate_missing" in result.blocking_reasons
