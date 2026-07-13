from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace

from gmoney.evaluation.matching import RowView
from gmoney.evaluation.metrics import canonical_metric_v2, legacy_metric_v1
from gmoney.evaluation.quality import Phase2QualityResult, evaluate_phase2_quality


@dataclass(frozen=True)
class Phase3Document:
    document_id: str
    hospital_id: str | None
    cohort: str
    route: str
    gold: tuple[RowView, ...]
    actual: tuple[RowView, ...]
    accepted_ungrounded_rows: int = 0
    privacy_failures: int = 0
    gemini_calls: int = 0
    ordinary_active: bool = False


@dataclass(frozen=True)
class ConfidenceBounds:
    legacy_precision_lower: float
    legacy_recall_lower: float
    canonical_precision_lower: float
    canonical_recall_lower: float


@dataclass(frozen=True)
class CohortResult:
    cohort: str
    documents: int
    quality: Phase2QualityResult

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort": self.cohort,
            "documents": self.documents,
            "quality": self.quality.to_dict(),
        }


@dataclass(frozen=True)
class Phase3QualityResult:
    aggregate: Phase2QualityResult
    cohorts: tuple[CohortResult, ...]
    confidence_bounds: ConfidenceBounds
    active_zero_gemini_fraction: float
    privacy_failures: int
    blocking_reasons: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate.to_dict(),
            "cohorts": [item.to_dict() for item in self.cohorts],
            "confidence_bounds": asdict(self.confidence_bounds),
            "active_zero_gemini_fraction": self.active_zero_gemini_fraction,
            "privacy_failures": self.privacy_failures,
            "blocking_reasons": self.blocking_reasons,
            "passed": self.passed,
        }


def _combine(
    documents: tuple[Phase3Document, ...],
) -> tuple[tuple[RowView, ...], tuple[RowView, ...]]:
    gold: list[RowView] = []
    actual: list[RowView] = []
    for index, document in enumerate(documents, 1):
        offset = index * 10_000
        gold.extend(replace(row, page_number=row.page_number + offset) for row in document.gold)
        actual.extend(replace(row, page_number=row.page_number + offset) for row in document.actual)
    return tuple(gold), tuple(actual)


def _quality(documents: tuple[Phase3Document, ...]) -> Phase2QualityResult:
    gold, actual = _combine(documents)
    return evaluate_phase2_quality(
        gold,
        actual,
        accepted_ungrounded_rows=sum(item.accepted_ungrounded_rows for item in documents),
    )


def _lower(values: list[float], percentile: float = 0.025) -> float:
    if not values:
        return 0.0
    values.sort()
    return values[min(len(values) - 1, int(len(values) * percentile))]


def bootstrap_confidence_bounds(
    documents: tuple[Phase3Document, ...],
    *,
    samples: int = 1_000,
    seed: str = "gmoney-phase3-bootstrap-v1",
) -> ConfidenceBounds:
    if not documents:
        return ConfidenceBounds(0, 0, 0, 0)
    rng = random.Random(seed)
    legacy_precision: list[float] = []
    legacy_recall: list[float] = []
    canonical_precision: list[float] = []
    canonical_recall: list[float] = []
    for _ in range(samples):
        selected = tuple(rng.choice(documents) for _ in documents)
        gold, actual = _combine(selected)
        legacy = legacy_metric_v1(gold, actual)
        canonical = canonical_metric_v2(gold, actual)
        legacy_precision.append(legacy.precision)
        legacy_recall.append(legacy.recall)
        canonical_precision.append(canonical.precision)
        canonical_recall.append(canonical.recall)
    return ConfidenceBounds(
        legacy_precision_lower=_lower(legacy_precision),
        legacy_recall_lower=_lower(legacy_recall),
        canonical_precision_lower=_lower(canonical_precision),
        canonical_recall_lower=_lower(canonical_recall),
    )


def evaluate_phase3_quality(
    documents: tuple[Phase3Document, ...],
    *,
    bootstrap_samples: int = 1_000,
) -> Phase3QualityResult:
    if not documents:
        raise ValueError("Phase 3 evaluation needs documents")
    aggregate = _quality(documents)
    cohort_names = sorted({item.cohort for item in documents})
    cohorts = tuple(
        CohortResult(
            cohort=name,
            documents=len(selected := tuple(item for item in documents if item.cohort == name)),
            quality=_quality(selected),
        )
        for name in cohort_names
    )
    by_name = {item.cohort: item for item in cohorts}
    bounds = bootstrap_confidence_bounds(documents, samples=bootstrap_samples)
    active = tuple(item for item in documents if item.ordinary_active)
    active_zero_gemini = (
        sum(item.gemini_calls == 0 for item in active) / len(active) if active else 0.0
    )
    blocking: list[str] = []
    aggregate_rows = (
        aggregate.legacy_precision,
        aggregate.legacy_recall,
        aggregate.legacy_f1,
        aggregate.canonical_precision,
        aggregate.canonical_recall,
        aggregate.canonical_f1,
    )
    if min(aggregate_rows) < 0.9:
        blocking.append("aggregate_row_metrics_below_90_percent")
    if aggregate.amount_accuracy < 0.95:
        blocking.append("aggregate_amount_accuracy_below_95_percent")
    if aggregate.negative_recall < 1:
        blocking.append("negative_recall_below_100_percent")
    if aggregate.accepted_ungrounded_rows:
        blocking.append("accepted_ungrounded_rows")
    privacy_failures = sum(item.privacy_failures for item in documents)
    if privacy_failures:
        blocking.append("privacy_failures")
    unseen_documents = tuple(item for item in documents if item.cohort == "unseen")
    unseen_hospitals = {item.hospital_id for item in unseen_documents if item.hospital_id}
    unseen = by_name.get("unseen")
    if unseen is None or len(unseen_hospitals) < 10:
        blocking.append("unseen_10_hospital_gate_missing")
    elif min(
        unseen.quality.legacy_precision,
        unseen.quality.legacy_recall,
        unseen.quality.canonical_precision,
        unseen.quality.canonical_recall,
    ) < 0.85:
        blocking.append("unseen_row_metrics_below_85_percent")
    known_documents = tuple(item for item in documents if item.cohort == "known_active")
    known_gold_rows = sum(len(item.gold) for item in known_documents)
    known = by_name.get("known_active")
    if known is None or known.documents < 10 or known_gold_rows < 200:
        blocking.append("known_active_profile_gate_missing")
    elif min(
        known.quality.legacy_precision,
        known.quality.legacy_recall,
        known.quality.legacy_f1,
        known.quality.canonical_precision,
        known.quality.canonical_recall,
        known.quality.canonical_f1,
    ) < 0.95:
        blocking.append("known_active_row_metrics_below_95_percent")
    elif known.quality.amount_accuracy < 0.95:
        blocking.append("known_active_amount_accuracy_below_95_percent")
    elif known.quality.negative_recall < 1:
        blocking.append("known_active_negative_recall_below_100_percent")
    if min(asdict(bounds).values()) <= 0.85:
        blocking.append("aggregate_confidence_lower_bound_not_above_85_percent")
    if not active or active_zero_gemini < 0.95:
        blocking.append("active_zero_gemini_fraction_below_95_percent")
    return Phase3QualityResult(
        aggregate=aggregate,
        cohorts=cohorts,
        confidence_bounds=bounds,
        active_zero_gemini_fraction=active_zero_gemini,
        privacy_failures=privacy_failures,
        blocking_reasons=tuple(blocking),
        passed=not blocking,
    )
