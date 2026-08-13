from __future__ import annotations

import json
import multiprocessing
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import (
    DriftObservation,
    LayoutObservation,
    LayoutProfile,
    ProfileLifecycle,
    ProfileMetrics,
)
from gmoney.demo.alias_transactions import (
    ALIAS_JOURNAL_VERSION,
    AliasTransactionCoordinator,
    _digest,
)
from gmoney.demo.store import JobStore
from gmoney.profiles.aliases import (
    LEGACY_ALIAS_REGISTRY_VERSION,
    AliasRegistryUnavailable,
    JsonAliasRepository,
    empty_alias_registry,
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
from gmoney.profiles.repository import (
    LEGACY_PROFILE_REGISTRY_VERSION,
    PROFILE_REGISTRY_VERSION,
    HospitalIdentityConflict,
    JsonProfileRepository,
    ProfileMutationCoordinationRequired,
    ProfileRegistryFormatError,
    ProfileRegistryRevisionConflict,
    ProfileRegistryUnavailable,
    active_hospital_identities,
)


def _replace_profile_while_locked(
    registry_value: str,
    lock_value: str,
    sender: Connection,
) -> None:
    repository = JsonProfileRepository(
        Path(registry_value),
        Path(lock_value),
    )
    current = repository.get("h1-itemized", 1)
    sender.send("ready")
    repository.replace_profile(
        current.model_copy(update={"hospital_name": "Renamed Hospital"})
    )
    sender.send("written")
    sender.close()


def _coordinated_profile_rename_process(
    root_value: str,
    profile_value: str,
    profile_lock_value: str,
    alias_value: str,
    start_event: object,
    sender: Connection,
) -> None:
    repository = JsonProfileRepository(Path(profile_value), Path(profile_lock_value))
    coordinator = AliasTransactionCoordinator(JobStore(Path(root_value)), Path(alias_value))
    start_event.wait()
    current = repository.snapshot().profiles[0]
    updated = coordinator.mutate_profile_registry(
        repository,
        1,
        lambda snapshot: repository.replace_in_snapshot(
            snapshot,
            (current.model_copy(update={"hospital_name": "Renamed Hospital"}),),
        ),
    )
    sender.send(("profile", updated.revision))
    sender.close()


def _coordinated_alias_mutation_process(
    root_value: str,
    alias_value: str,
    start_event: object,
    sender: Connection,
) -> None:
    coordinator = AliasTransactionCoordinator(JobStore(Path(root_value)), Path(alias_value))
    start_event.wait()

    def mutation(registry: dict[str, object]) -> dict[str, object]:
        events = list(registry["events"])
        events.append(
            {
                "action": "concurrent_alias_audit",
                "reviewer": "process-test",
                "reason": "Exercise the shared process lock",
                "created_at": "2026-08-13T00:00:00Z",
            }
        )
        registry["events"] = events
        return registry

    updated = coordinator.mutate_registry(0, mutation)
    sender.send(("alias", updated["revision"]))
    sender.close()


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


def test_profile_repository_reads_v1_and_upgrades_on_first_write(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    legacy = {
        "registry_version": LEGACY_PROFILE_REGISTRY_VERSION,
        "profiles": [profile().model_dump(mode="json")],
        "events": [],
    }
    path.write_text(json.dumps(legacy))
    repo = JsonProfileRepository(path)

    observed = repo.snapshot()
    assert observed.registry_version == LEGACY_PROFILE_REGISTRY_VERSION
    assert observed.revision == 0

    repo.add_profile(profile(key="h1-pharmacy", version=1))
    upgraded = json.loads(path.read_text())
    assert upgraded["registry_version"] == PROFILE_REGISTRY_VERSION
    assert upgraded["revision"] == 1


@pytest.mark.parametrize("revision", [False, "3", 3.0, -1])
def test_v2_profile_registry_requires_an_exact_non_negative_integer_revision(
    tmp_path: Path,
    revision: object,
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "registry_version": PROFILE_REGISTRY_VERSION,
                "revision": revision,
                "profiles": [],
                "events": [],
            }
        )
    )

    with pytest.raises(ProfileRegistryUnavailable):
        JsonProfileRepository(path).snapshot()


def test_v2_profile_registry_requires_the_revision_field(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "registry_version": PROFILE_REGISTRY_VERSION,
                "profiles": [],
                "events": [],
            }
        )
    )

    with pytest.raises(ProfileRegistryUnavailable):
        JsonProfileRepository(path).snapshot()


@pytest.mark.parametrize("revision", [False, "0", 0.0, 1])
def test_v1_profile_registry_rejects_non_exact_legacy_zero_revisions(
    tmp_path: Path,
    revision: object,
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "registry_version": LEGACY_PROFILE_REGISTRY_VERSION,
                "revision": revision,
                "profiles": [],
                "events": [],
            }
        )
    )

    with pytest.raises(ProfileRegistryUnavailable):
        JsonProfileRepository(path).snapshot()


def test_coordinated_profile_write_rejects_cross_registry_identity_conflict(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    repository = JsonProfileRepository(profile_path)
    current = profile(lifecycle=ProfileLifecycle.ACTIVE).model_copy(
        update={"hospital_name": "Original Hospital"}
    )
    repository.add_profile(current)
    alias_path = tmp_path / "alias-registry.json"
    aliases = empty_alias_registry()
    aliases.update(
        revision=1,
        hospitals=[
            {
                "hospital_id": "H2",
                "hospital_name": "Taken Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Taken Hospital",
                        "normalized_name": "taken hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "test-reviewer",
                        "reason": "Verified hospital identity",
                        "created_at": "2026-08-12T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
    )
    JsonAliasRepository(alias_path)._write_unlocked(aliases)
    coordinator = AliasTransactionCoordinator(JobStore(tmp_path), alias_path)
    before = profile_path.read_bytes()

    with pytest.raises(HospitalIdentityConflict) as conflict:
        coordinator.mutate_profile_registry(
            repository,
            1,
            lambda snapshot: repository.replace_in_snapshot(
                snapshot,
                (current.model_copy(update={"hospital_name": "Taken Hospital"}),),
            ),
        )

    assert conflict.value.owner_ids == ["H1", "H2"]
    assert profile_path.read_bytes() == before
    assert repository.snapshot().revision == 1
    profiles, observed_aliases, identities = coordinator.identity_snapshots(repository)
    assert profiles.revision == 1
    assert observed_aliases["revision"] == 1
    assert identities["H1"]["hospital_name"] == "Original Hospital"


def test_rejected_profile_mutation_does_not_migrate_legacy_alias_registry(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    repository = JsonProfileRepository(profile_path)
    current = profile(lifecycle=ProfileLifecycle.ACTIVE).model_copy(
        update={"hospital_name": "Original Hospital"}
    )
    repository.add_profile(current)
    alias_path = tmp_path / "alias-registry.json"
    aliases = empty_alias_registry()
    aliases.update(
        registry_version=LEGACY_ALIAS_REGISTRY_VERSION,
        revision=0,
        hospitals=[
            {
                "hospital_id": "H2",
                "hospital_name": "Taken Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Taken Hospital",
                        "normalized_name": "taken hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "reviewer",
                        "reason": "Verified identity",
                        "created_at": "2026-08-13T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-13T00:00:00Z",
                "updated_at": "2026-08-13T00:00:00Z",
            }
        ],
    )
    alias_path.write_text(json.dumps(aliases))
    coordinator = AliasTransactionCoordinator(JobStore(tmp_path), alias_path)
    before = {
        profile_path: profile_path.read_bytes(),
        alias_path: alias_path.read_bytes(),
    }

    with pytest.raises(HospitalIdentityConflict):
        coordinator.mutate_profile_registry(
            repository,
            1,
            lambda snapshot: repository.replace_in_snapshot(
                snapshot,
                (current.model_copy(update={"hospital_name": "Taken Hospital"}),),
            ),
        )

    assert {path: path.read_bytes() for path in before} == before

    updated = coordinator.mutate_profile_registry(
        repository,
        1,
        lambda snapshot: repository.replace_in_snapshot(
            snapshot,
            (current.model_copy(update={"hospital_name": "Available Hospital"}),),
        ),
    )
    assert updated.revision == 2
    assert JsonAliasRepository(alias_path).read()["registry_version"] == (
        "hospital_alias_registry_v3"
    )


def test_profile_conflict_in_pending_recovery_projection_changes_no_file(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    repository = JsonProfileRepository(profile_path)
    current = profile(lifecycle=ProfileLifecycle.ACTIVE).model_copy(
        update={"hospital_name": "Original Hospital"}
    )
    repository.add_profile(current)
    store = JobStore(tmp_path)
    state = store.create("pending-alias.pdf")
    job_id = str(state["id"])
    base_review = store.empty_review()
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_aliases = empty_alias_registry()
    target_aliases = json.loads(json.dumps(base_aliases))
    target_aliases.update(
        revision=1,
        hospitals=[
            {
                "hospital_id": "H2",
                "hospital_name": "Taken Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Taken Hospital",
                        "normalized_name": "taken hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "reviewer",
                        "reason": "Verified identity",
                        "created_at": "2026-08-13T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-13T00:00:00Z",
                "updated_at": "2026-08-13T00:00:00Z",
            }
        ],
        events=[
            {
                "action": "hospital_linked",
                "reviewer": "reviewer",
                "reason": "Pending identity cutover",
                "created_at": "2026-08-13T00:00:00Z",
            }
        ],
    )
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps(base_aliases))
    review_path = store.job_dir(job_id) / "review.json"
    review_path.write_text(json.dumps(base_review))
    journal_path = store.job_dir(job_id) / ".alias-operation.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": ALIAS_JOURNAL_VERSION,
                "job_id": job_id,
                "base_review": base_review,
                "target_review": target_review,
                "base_registry": base_aliases,
                "target_registry": target_aliases,
                "base_review_sha256": _digest(base_review),
                "target_review_sha256": _digest(target_review),
                "base_registry_sha256": _digest(base_aliases),
                "target_registry_sha256": _digest(target_aliases),
            }
        )
    )
    tracked = (profile_path, alias_path, review_path, journal_path)
    before = {path: path.read_bytes() for path in tracked}

    with pytest.raises(HospitalIdentityConflict):
        AliasTransactionCoordinator(store, alias_path).mutate_profile_registry(
            repository,
            1,
            lambda snapshot: repository.replace_in_snapshot(
                snapshot,
                (current.model_copy(update={"hospital_name": "Taken Hospital"}),),
            ),
        )

    assert {path: path.read_bytes() for path in tracked} == before


def test_profile_mutation_translates_malformed_pending_journal(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    repository = JsonProfileRepository(profile_path)
    repository.add_profile(profile(lifecycle=ProfileLifecycle.ACTIVE))
    store = JobStore(tmp_path)
    state = store.create("malformed-journal.pdf")
    journal_path = store.job_dir(str(state["id"])) / ".alias-operation.json"
    journal_path.write_text("{not-json")
    alias_path = tmp_path / "aliases.json"
    tracked = (profile_path, journal_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked}

    with pytest.raises(AliasRegistryUnavailable):
        AliasTransactionCoordinator(store, alias_path).mutate_profile_registry(
            repository,
            1,
            lambda snapshot: snapshot,
        )

    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked
    } == before
    assert not alias_path.exists()


@pytest.mark.parametrize("journal_kind", ("directory", "symlink"))
def test_recovery_and_projection_reject_non_regular_journals(
    tmp_path: Path,
    journal_kind: str,
) -> None:
    store = JobStore(tmp_path)
    state = store.create("invalid-journal.pdf")
    journal_path = store.job_dir(str(state["id"])) / ".alias-operation.json"
    if journal_kind == "directory":
        journal_path.mkdir()
    else:
        journal_path.symlink_to(tmp_path / "missing-journal.json")
    coordinator = AliasTransactionCoordinator(store, tmp_path / "aliases.json")

    with (
        pytest.raises(AliasRegistryUnavailable, match="not a regular file"),
        coordinator.repository.lock(exclusive=False),
        coordinator._projection_unlocked(),
    ):
        pass
    with pytest.raises(AliasRegistryUnavailable, match="not a regular file"):
        coordinator.recover_all()


@pytest.mark.parametrize("registry_kind", ("directory", "broken_symlink"))
def test_alias_registry_rejects_non_regular_filesystem_objects(
    tmp_path: Path,
    registry_kind: str,
) -> None:
    registry_path = tmp_path / "aliases.json"
    if registry_kind == "directory":
        registry_path.mkdir()
    else:
        registry_path.symlink_to(tmp_path / "missing-aliases.json")
    coordinator = AliasTransactionCoordinator(JobStore(tmp_path / "jobs"), registry_path)

    with (
        pytest.raises(AliasRegistryUnavailable, match="not a regular file"),
        coordinator.repository.lock(exclusive=False),
        coordinator._projection_unlocked(),
    ):
        pass
    with pytest.raises(AliasRegistryUnavailable, match="not a regular file"):
        coordinator.recover_all()


def test_profile_repository_rejects_stale_writes(tmp_path) -> None:
    repo = JsonProfileRepository(tmp_path / "profiles.json")
    item = profile()
    repo.add_profile(item)

    with pytest.raises(ProfileRegistryRevisionConflict) as conflict:
        repo.replace_profile(
            item.model_copy(update={"hospital_name": "Current Hospital"}),
            expected_revision=0,
        )

    assert conflict.value.current_revision == 1
    assert repo.snapshot().revision == 1


def test_direct_profile_write_is_disabled_when_alias_registry_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GMONEY_ALIAS_REGISTRY", str(tmp_path / "aliases.json"))
    repository = JsonProfileRepository(tmp_path / "profiles.json")

    with pytest.raises(ProfileMutationCoordinationRequired):
        repository.add_profile(profile())

    assert not repository.path.exists()


def test_active_profile_identity_validation_and_stable_display(tmp_path) -> None:
    repo = JsonProfileRepository(tmp_path / "profiles.json")
    repo.add_profile(
        profile(key="z-pharmacy", lifecycle=ProfileLifecycle.ACTIVE).model_copy(
            update={"hospital_name": "MACHINE-HOSPITAL"}
        )
    )
    repo.add_profile(
        profile(key="a-items", lifecycle=ProfileLifecycle.ACTIVE).model_copy(
            update={"hospital_name": "Machine Hospital"}
        )
    )

    identities = active_hospital_identities(repo.snapshot())
    assert identities["H1"] == {
        "hospital_id": "H1",
        "hospital_name": "Machine Hospital",
        "normalized_name": "machine hospital",
        "active_profile_count": 2,
    }

    with pytest.raises(ProfileRegistryFormatError, match="disagree"):
        repo.add_profile(
            profile(key="other-name", lifecycle=ProfileLifecycle.ACTIVE).model_copy(
                update={"hospital_name": "Different Medical Center"}
            )
        )
    with pytest.raises(ProfileRegistryFormatError, match="multiple hospitals"):
        repo.add_profile(
            profile(
                key="other-owner",
                lifecycle=ProfileLifecycle.ACTIVE,
                hospital_id="H2",
            ).model_copy(update={"hospital_name": "Machine Hospital"})
        )


def test_nameless_active_profile_does_not_claim_a_canonical_name(tmp_path) -> None:
    repo = JsonProfileRepository(tmp_path / "profiles.json")
    repo.add_profile(profile(lifecycle=ProfileLifecycle.ACTIVE))

    assert active_hospital_identities(repo.snapshot())["H1"]["hospital_name"] is None


def test_profile_writer_waits_for_shared_snapshot_lock(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    lock_path = tmp_path / "profiles.lock"
    repo = JsonProfileRepository(path, lock_path)
    repo.add_profile(profile())
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_replace_profile_while_locked,
        args=(str(path), str(lock_path), sender),
    )

    with repo.lock(exclusive=False):
        process.start()
        sender.close()
        assert receiver.poll(5)
        assert receiver.recv() == "ready"
        assert not receiver.poll(0.2)

    assert receiver.poll(5)
    assert receiver.recv() == "written"
    process.join(timeout=5)
    receiver.close()
    assert process.exitcode == 0


def test_profile_and_alias_process_mutations_serialize_without_deadlock(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_lock_path = tmp_path / "profiles.lock"
    alias_path = tmp_path / "aliases.json"
    repository = JsonProfileRepository(profile_path, profile_lock_path)
    repository.add_profile(
        profile(lifecycle=ProfileLifecycle.ACTIVE).model_copy(
            update={"hospital_name": "Original Hospital"}
        )
    )
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    processes = (
        context.Process(
            target=_coordinated_profile_rename_process,
            args=(
                str(tmp_path),
                str(profile_path),
                str(profile_lock_path),
                str(alias_path),
                start_event,
                sender,
            ),
        ),
        context.Process(
            target=_coordinated_alias_mutation_process,
            args=(str(tmp_path), str(alias_path), start_event, sender),
        ),
    )
    for process in processes:
        process.start()
    sender.close()
    start_event.set()
    try:
        assert receiver.poll(8)
        first = receiver.recv()
        assert receiver.poll(8)
        second = receiver.recv()
    finally:
        receiver.close()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert {first, second} == {("profile", 2), ("alias", 1)}
    profiles, aliases, identities = AliasTransactionCoordinator(
        JobStore(tmp_path), alias_path
    ).identity_snapshots(repository)
    assert profiles.revision == 2
    assert aliases["revision"] == 1
    assert identities["H1"]["hospital_name"] == "Renamed Hospital"


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
