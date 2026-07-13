from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from gmoney.contracts.phase3 import (
    DriftDecision,
    DriftObservation,
    LayoutProfile,
    ProfileEvent,
    ProfileLifecycle,
)

ALLOWED_TRANSITIONS = {
    ProfileLifecycle.CANDIDATE: {ProfileLifecycle.SHADOW, ProfileLifecycle.ARCHIVED},
    ProfileLifecycle.SHADOW: {ProfileLifecycle.ACTIVE, ProfileLifecycle.ARCHIVED},
    ProfileLifecycle.ACTIVE: {ProfileLifecycle.DRIFTED, ProfileLifecycle.ARCHIVED},
    ProfileLifecycle.DRIFTED: {ProfileLifecycle.SHADOW, ProfileLifecycle.ARCHIVED},
    ProfileLifecycle.ARCHIVED: set(),
}


def activation_failures(profile: LayoutProfile) -> tuple[str, ...]:
    metrics = profile.metrics
    failures: list[str] = []
    if metrics is None:
        return ("missing_metrics",)
    if metrics.holdout_bills < 10:
        failures.append("holdout_bills_below_10")
    if metrics.holdout_rows < 200:
        failures.append("holdout_rows_below_200")
    if min(metrics.precision, metrics.recall, metrics.f1) < 0.95:
        failures.append("row_metrics_below_95_percent")
    if min(
        metrics.precision_ci_lower,
        metrics.recall_ci_lower,
        metrics.f1_ci_lower,
    ) <= 0.85:
        failures.append("confidence_lower_bound_not_above_85_percent")
    if metrics.amount_accuracy < 0.95:
        failures.append("amount_accuracy_below_95_percent")
    if metrics.negative_recall < 1:
        failures.append("negative_recall_below_100_percent")
    if metrics.accepted_ungrounded_rows:
        failures.append("accepted_ungrounded_rows")
    if metrics.incomplete_pages:
        failures.append("incomplete_pages")
    if metrics.identity_ambiguities:
        failures.append("identity_ambiguities")
    if metrics.unsafe_numeric_failures:
        failures.append("unsafe_numeric_failures")
    if metrics.supported_field_failures:
        failures.append("supported_field_gate_failures")
    if "mixed_capture" in profile.quality_flags and metrics.capture_conditions < 2:
        failures.append("mixed_capture_conditions_below_2")
    if not profile.construction_dataset_ids or not profile.holdout_dataset_ids:
        failures.append("missing_dataset_provenance")
    if set(profile.construction_dataset_ids) & set(profile.holdout_dataset_ids):
        failures.append("construction_holdout_leakage")
    if not profile.calibration_version:
        failures.append("missing_calibration_artifact")
    if not {"description", "amount"}.issubset(profile.supported_fields):
        failures.append("missing_core_supported_fields")
    return tuple(failures)


def transition_profile(
    profile: LayoutProfile,
    to_state: ProfileLifecycle,
    reason: str,
    *,
    now: datetime | None = None,
) -> tuple[LayoutProfile, ProfileEvent]:
    if to_state not in ALLOWED_TRANSITIONS[profile.lifecycle]:
        raise ValueError(f"invalid profile transition: {profile.lifecycle} -> {to_state}")
    if to_state is ProfileLifecycle.ACTIVE:
        failures = activation_failures(profile)
        if failures:
            raise ValueError(f"profile activation gate failed: {', '.join(failures)}")
    event = ProfileEvent(
        profile_key=profile.profile_key,
        profile_version=profile.profile_version,
        from_state=profile.lifecycle,
        to_state=to_state,
        reason=reason,
        occurred_at=now or datetime.now(UTC),
    )
    return profile.model_copy(update={"lifecycle": to_state}), event


def deterministic_shadow_sample(document_sha256: str, profile_key: str, percent: int = 5) -> bool:
    if not 0 <= percent <= 100:
        raise ValueError("sample percent must be between 0 and 100")
    digest = hashlib.sha256(f"{document_sha256}:{profile_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < percent


def evaluate_drift(
    observations: Sequence[DriftObservation],
    *,
    window_size: int = 20,
    breach_rate: float = 0.25,
) -> DriftDecision:
    if not observations:
        raise ValueError("drift evaluation needs observations")
    latest = observations[-1]
    if any(item.hard_incompatibility for item in observations[-window_size:]):
        return DriftDecision(
            profile_key=latest.profile_key,
            profile_version=latest.profile_version,
            drifted=True,
            immediate=True,
            breached_windows=1,
            reasons=("hard_incompatibility",),
        )
    if len(observations) < window_size * 2:
        return DriftDecision(
            profile_key=latest.profile_key,
            profile_version=latest.profile_version,
            drifted=False,
            reasons=("insufficient_window",),
        )
    breached = 0
    reasons: list[str] = []
    for start in (len(observations) - window_size * 2, len(observations) - window_size):
        window = observations[start : start + window_size]
        failures = sum(item.escalated or item.heavy_disagreement for item in window)
        if failures / window_size >= breach_rate:
            breached += 1
    if breached == 2:
        reasons.append("two_consecutive_drift_windows")
    return DriftDecision(
        profile_key=latest.profile_key,
        profile_version=latest.profile_version,
        drifted=breached == 2,
        breached_windows=breached,
        reasons=tuple(reasons),
    )


def rollback_profile(
    profiles: Sequence[LayoutProfile], profile_key: str
) -> tuple[LayoutProfile, LayoutProfile]:
    versions = sorted(
        (item for item in profiles if item.profile_key == profile_key),
        key=lambda item: item.profile_version,
        reverse=True,
    )
    current = next((item for item in versions if item.lifecycle is ProfileLifecycle.ACTIVE), None)
    previous = next(
        (
            item
            for item in versions
            if item is not current
            and item.metrics is not None
            and not activation_failures(item)
            and item.lifecycle in {ProfileLifecycle.ARCHIVED, ProfileLifecycle.DRIFTED}
        ),
        None,
    )
    if current is None or previous is None:
        raise ValueError("rollback requires an active version and an earlier passing version")
    return (
        current.model_copy(update={"lifecycle": ProfileLifecycle.ARCHIVED}),
        previous.model_copy(update={"lifecycle": ProfileLifecycle.ACTIVE}),
    )
