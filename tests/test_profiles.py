from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import (
    DriftObservation,
    LayoutObservation,
    LayoutProfile,
    ProfileLifecycle,
    ProfileMetrics,
)
from gmoney.profiles.construction import build_profile
from gmoney.profiles.lifecycle import (
    activation_failures,
    deterministic_shadow_sample,
    evaluate_drift,
    rollback_profile,
    transition_profile,
)
from gmoney.profiles.matching import match_profile
from gmoney.profiles.repository import JsonProfileRepository


def passing_metrics() -> ProfileMetrics:
    return ProfileMetrics(
        holdout_bills=10,
        holdout_rows=200,
        precision=0.97,
        recall=0.96,
        f1=0.965,
        amount_accuracy=0.99,
        precision_ci_lower=0.9,
        recall_ci_lower=0.89,
        f1_ci_lower=0.895,
        negative_recall=1,
    )


def profile(
    *,
    key: str = "h1-itemized",
    version: int = 1,
    lifecycle: ProfileLifecycle = ProfileLifecycle.CANDIDATE,
    hospital_id: str = "H1",
) -> LayoutProfile:
    return LayoutProfile(
        contract_version="layout_profile_v1",
        profile_key=key,
        profile_version=version,
        lifecycle=lifecycle,
        hospital_id=hospital_id,
        page_type=PageType.ITEMIZED_CHARGES,
        table_type=TableType.ITEM_LEDGER,
        page_aspect_ratio=0.7,
        table_box=(0.05, 0.15, 0.95, 0.9),
        header_tokens=("Service Name", "Amount"),
        stable_anchors=("service",),
        column_centers={"description": 0.25, "amount": 0.88},
        column_tolerances={"description": 0.08, "amount": 0.04},
        supported_fields=("description", "amount"),
        construction_dataset_ids=("construction-1",),
        holdout_dataset_ids=("holdout-1",),
        metrics=passing_metrics(),
        calibration_version="isotonic-v1",
    )


def observation(hospital_id: str = "H1") -> LayoutObservation:
    return LayoutObservation(
        document_id="document-1",
        hospital_id=hospital_id,
        page_number=1,
        page_type=PageType.ITEMIZED_CHARGES,
        table_type=TableType.ITEM_LEDGER,
        page_aspect_ratio=0.7,
        table_box=(0.05, 0.15, 0.95, 0.9),
        header_tokens=("Service Name", "Amount"),
        column_centers={"description": 0.25, "amount": 0.88},
    )


def test_profile_activation_enforces_full_gate() -> None:
    candidate = profile()
    shadow, first = transition_profile(
        candidate,
        ProfileLifecycle.SHADOW,
        "construction complete",
        now=datetime(2026, 7, 13, tzinfo=UTC),
    )
    active, second = transition_profile(shadow, ProfileLifecycle.ACTIVE, "holdout passed")
    assert active.lifecycle is ProfileLifecycle.ACTIVE
    assert first.from_state is ProfileLifecycle.CANDIDATE
    assert second.to_state is ProfileLifecycle.ACTIVE

    unsafe = shadow.model_copy(
        update={"metrics": passing_metrics().model_copy(update={"holdout_bills": 9})}
    )
    assert "holdout_bills_below_10" in activation_failures(unsafe)
    with pytest.raises(ValueError, match="activation gate failed"):
        transition_profile(unsafe, ProfileLifecycle.ACTIVE, "unsafe override")


def test_profile_matching_requires_active_threshold_and_margin() -> None:
    active = profile(lifecycle=ProfileLifecycle.ACTIVE)
    result = match_profile((active,), observation())
    assert result.selected
    assert result.profile_key == active.profile_key

    shadow = active.model_copy(update={"lifecycle": ProfileLifecycle.SHADOW})
    assert not match_profile((shadow,), observation()).selected
    assert match_profile((shadow,), observation(), include_shadow=True).selected

    duplicate = active.model_copy(update={"profile_key": "duplicate"})
    ambiguous = match_profile((active, duplicate), observation())
    assert not ambiguous.selected
    assert "below_candidate_margin" in ambiguous.reasons


def test_headerless_known_profile_can_match_on_trusted_identity_and_geometry() -> None:
    active = profile(lifecycle=ProfileLifecycle.ACTIVE).model_copy(
        update={"header_tokens": (), "stable_anchors": (), "column_centers": {}}
    )
    headerless = observation().model_copy(update={"header_tokens": (), "column_centers": {}})
    result = match_profile((active,), headerless)
    assert result.selected
    assert result.score == 1


def test_construct_profile_uses_independent_median_observations() -> None:
    first = observation().model_copy(update={"document_id": "one"})
    second = observation().model_copy(
        update={
            "document_id": "two",
            "table_box": (0.07, 0.17, 0.93, 0.88),
            "column_centers": {"description": 0.27, "amount": 0.86},
        }
    )
    built = build_profile(
        (first, second),
        profile_key="constructed",
        profile_version=1,
        hospital_id="H1",
        construction_dataset_ids=("construction-set",),
    )
    assert built.lifecycle is ProfileLifecycle.CANDIDATE
    assert built.table_box == pytest.approx((0.06, 0.16, 0.94, 0.89))
    assert built.column_centers["amount"] == pytest.approx(0.87)
    assert {"description", "amount"}.issubset(built.supported_fields)

    with pytest.raises(ValueError, match="independent documents"):
        build_profile(
            (first, first),
            profile_key="invalid",
            profile_version=1,
            hospital_id="H1",
            construction_dataset_ids=("construction-set",),
        )


def test_profile_repository_is_versioned_and_atomic(tmp_path) -> None:
    repo = JsonProfileRepository(tmp_path / "profiles.json")
    item = profile()
    repo.add_profile(item)
    assert repo.get(item.profile_key, 1) == item
    with pytest.raises(ValueError, match="already exists"):
        repo.add_profile(item)
    shadow, event = transition_profile(item, ProfileLifecycle.SHADOW, "ready")
    repo.replace_profile(shadow, event)
    assert repo.get(item.profile_key).lifecycle is ProfileLifecycle.SHADOW
    assert repo.events() == (event,)

    active = profile(version=2, lifecycle=ProfileLifecycle.ACTIVE)
    repo.add_profile(active)
    with pytest.raises(ValueError, match="only one active version"):
        repo.add_profile(profile(version=3, lifecycle=ProfileLifecycle.ACTIVE))
    assert len(repo.list_profiles()) == 2


def test_profile_drift_sampling_pause_and_rollback() -> None:
    assert deterministic_shadow_sample("a" * 64, "profile", percent=100)
    assert not deterministic_shadow_sample("a" * 64, "profile", percent=0)
    sampled = sum(
        deterministic_shadow_sample(f"{index:064x}", "profile") for index in range(10_000)
    )
    assert 400 <= sampled <= 600
    observations = tuple(
        DriftObservation(
            profile_key="profile",
            profile_version=2,
            document_id=f"d-{index}",
            match_score=0.9,
            escalated=index % 4 == 0,
        )
        for index in range(40)
    )
    decision = evaluate_drift(observations)
    assert decision.drifted
    assert decision.breached_windows == 2

    immediate = evaluate_drift(
        (observations[0].model_copy(update={"hard_incompatibility": True}),)
    )
    assert immediate.drifted and immediate.immediate

    previous = profile(version=1, lifecycle=ProfileLifecycle.ARCHIVED)
    current = profile(version=2, lifecycle=ProfileLifecycle.ACTIVE)
    archived, restored = rollback_profile((previous, current), current.profile_key)
    assert archived.lifecycle is ProfileLifecycle.ARCHIVED
    assert restored.lifecycle is ProfileLifecycle.ACTIVE
    assert restored.profile_version == 1


def test_mixed_capture_profile_needs_two_holdout_conditions() -> None:
    candidate = profile(lifecycle=ProfileLifecycle.SHADOW).model_copy(
        update={"quality_flags": ("mixed_capture",)}
    )
    assert "mixed_capture_conditions_below_2" in activation_failures(candidate)


def test_profile_match_scales_to_20_000_variants() -> None:
    base = profile(lifecycle=ProfileLifecycle.ACTIVE)
    profiles = tuple(
        base.model_copy(
            update={"profile_key": f"profile-{index}", "hospital_id": f"H{index}"}
        )
        for index in range(20_000)
    )
    result = match_profile(profiles, observation("H19999"))
    assert result.selected
    assert result.profile_key == "profile-19999"
