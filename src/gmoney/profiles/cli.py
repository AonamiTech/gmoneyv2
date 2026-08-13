from __future__ import annotations

import json
import os
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import (
    DriftObservation,
    LayoutObservation,
    LayoutProfile,
    ProfileEvent,
    ProfileLifecycle,
    ProfileRegistrySnapshot,
)
from gmoney.demo.alias_transactions import AliasTransactionCoordinator
from gmoney.demo.store import JobStore
from gmoney.profiles.aliases import AliasRegistryUnavailable
from gmoney.profiles.construction import build_profile
from gmoney.profiles.lifecycle import evaluate_drift, rollback_profile, transition_profile
from gmoney.profiles.matching import match_profile
from gmoney.profiles.repository import (
    HospitalIdentityConflict,
    JsonProfileRepository,
    ProfileRegistryUnavailable,
)
from gmoney.release import build_revision

app = typer.Typer(no_args_is_help=True)


@app.callback()
def validate_release_manifest() -> None:
    build_revision()


def _writer(
    registry: Path,
    alias_registry: Path,
    jobs_root: Path,
) -> tuple[JsonProfileRepository, AliasTransactionCoordinator]:
    if jobs_root.name != "jobs":
        raise typer.BadParameter("--jobs-root must identify the runtime jobs directory")
    repository = JsonProfileRepository(registry)
    coordinator = AliasTransactionCoordinator(JobStore(jobs_root.parent), alias_registry)
    return repository, coordinator


def _add_coordinated(
    registry: Path,
    alias_registry: Path,
    jobs_root: Path,
    profile: LayoutProfile,
) -> None:
    repository, coordinator = _writer(registry, alias_registry, jobs_root)
    coordinator.mutate_profile_registry(
        repository,
        None,
        lambda snapshot: repository.add_to_snapshot(snapshot, profile),
    )


@app.command("benchmark")
def benchmark(
    variants: Annotated[int, typer.Option(min=1, max=100_000)] = 20_000,
    queries: Annotated[int, typer.Option(min=1, max=10_000)] = 100,
) -> None:
    started = time.perf_counter()
    base = LayoutProfile(
        contract_version="layout_profile_v1",
        profile_key="profile-0",
        profile_version=1,
        lifecycle=ProfileLifecycle.ACTIVE,
        hospital_id="H0",
        page_type=PageType.ITEMIZED_CHARGES,
        table_type=TableType.ITEM_LEDGER,
        page_aspect_ratio=0.7,
        table_box=(0.05, 0.15, 0.95, 0.9),
        header_tokens=("description", "amount"),
        column_centers={"description": 0.25, "amount": 0.88},
        column_tolerances={"description": 0.08, "amount": 0.04},
        supported_fields=("description", "amount"),
        construction_dataset_ids=("benchmark",),
    )
    profiles = tuple(
        base.model_copy(
            update={"profile_key": f"profile-{index}", "hospital_id": f"H{index}"}
        )
        for index in range(variants)
    )
    build_seconds = time.perf_counter() - started
    observation = LayoutObservation(
        document_id="benchmark",
        hospital_id=f"H{variants - 1}",
        page_number=1,
        page_type=PageType.ITEMIZED_CHARGES,
        table_type=TableType.ITEM_LEDGER,
        page_aspect_ratio=0.7,
        table_box=(0.05, 0.15, 0.95, 0.9),
        header_tokens=("description", "amount"),
        column_centers={"description": 0.25, "amount": 0.88},
    )
    match_started = time.perf_counter()
    result = None
    for _ in range(queries):
        result = match_profile(profiles, observation)
    match_seconds = time.perf_counter() - match_started
    assert result is not None
    typer.echo(
        json.dumps(
            {
                "benchmark_version": "phase3_profile_scale_v1",
                "variants": variants,
                "queries": queries,
                "build_seconds": build_seconds,
                "mean_match_ms": match_seconds * 1000 / queries,
                "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "selected": result.selected,
                "profile_key": result.profile_key,
            },
            sort_keys=True,
        )
    )


@app.command("construct")
def construct(
    registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
    profile_version: int,
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
    hospital_id: str | None = None,
    hospital_name: str | None = None,
    global_family: str | None = None,
) -> None:
    payload = json.loads(observations.read_text())
    items = tuple(LayoutObservation.model_validate(item) for item in payload["observations"])
    profile = build_profile(
        items,
        profile_key=profile_key,
        profile_version=profile_version,
        hospital_id=hospital_id,
        hospital_name=hospital_name,
        global_family=global_family,
        construction_dataset_ids=tuple(payload["construction_dataset_ids"]),
    )
    _add_coordinated(registry, alias_registry, jobs_root, profile)
    typer.echo(json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("add")
def add(
    registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    item = LayoutProfile.model_validate_json(profile.read_text())
    _add_coordinated(registry, alias_registry, jobs_root, item)
    typer.echo(f"added {item.profile_key}@{item.profile_version}")


@app.command("list")
def list_profiles(registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    payload = [
        item.model_dump(mode="json")
        for item in JsonProfileRepository(registry).list_profiles()
    ]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("match")
def match(
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    observation: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    include_shadow: bool = False,
) -> None:
    repo = JsonProfileRepository(registry)
    result = match_profile(
        repo.list_profiles(),
        LayoutObservation.model_validate_json(observation.read_text()),
        include_shadow=include_shadow,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("transition")
def transition(
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    profile_key: str,
    profile_version: int,
    to_state: ProfileLifecycle,
    reason: str,
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    repo, coordinator = _writer(registry, alias_registry, jobs_root)
    current_lifecycle: dict[str, ProfileLifecycle] = {}

    def mutation(snapshot: ProfileRegistrySnapshot) -> ProfileRegistrySnapshot:
        current = next(
            (
                item
                for item in snapshot.profiles
                if item.profile_key == profile_key
                and item.profile_version == profile_version
            ),
            None,
        )
        if current is None:
            raise KeyError(f"unknown profile: {profile_key}@{profile_version}")
        current_lifecycle["value"] = current.lifecycle
        updated, event = transition_profile(current, to_state, reason)
        replacements = [updated]
        events = [event]
        if to_state is ProfileLifecycle.ACTIVE:
            active = next(
                (
                    item
                    for item in snapshot.profiles
                    if item.profile_key == profile_key
                    and item.profile_version != profile_version
                    and item.lifecycle is ProfileLifecycle.ACTIVE
                ),
                None,
            )
            if active is not None:
                archived, archive_event = transition_profile(
                    active,
                    ProfileLifecycle.ARCHIVED,
                    f"superseded by version {profile_version}",
                )
                replacements.append(archived)
                events.append(archive_event)
                replacements[0] = updated.model_copy(
                    update={"supersedes_version": active.profile_version}
                )
        return repo.replace_in_snapshot(snapshot, tuple(replacements), tuple(events))

    coordinator.mutate_profile_registry(
        repo,
        None,
        mutation,
    )
    typer.echo(
        f"{profile_key}@{profile_version}: {current_lifecycle['value']} -> {to_state}"
    )


@app.command("rollback")
def rollback(
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    profile_key: str,
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    repo, coordinator = _writer(registry, alias_registry, jobs_root)
    restored_version: dict[str, int] = {}

    def mutation(snapshot: ProfileRegistrySnapshot) -> ProfileRegistrySnapshot:
        current, previous = rollback_profile(snapshot.profiles, profile_key)
        restored_version["value"] = previous.profile_version
        previous_before = next(
            item
            for item in snapshot.profiles
            if item.profile_key == previous.profile_key
            and item.profile_version == previous.profile_version
        )
        now = datetime.now(UTC)
        events = (
            ProfileEvent(
                profile_key=current.profile_key,
                profile_version=current.profile_version,
                from_state=ProfileLifecycle.ACTIVE,
                to_state=ProfileLifecycle.ARCHIVED,
                reason=f"rollback to version {previous.profile_version}",
                occurred_at=now,
            ),
            ProfileEvent(
                profile_key=previous.profile_key,
                profile_version=previous.profile_version,
                from_state=previous_before.lifecycle,
                to_state=ProfileLifecycle.ACTIVE,
                reason=f"rollback from version {current.profile_version}",
                occurred_at=now,
            ),
        )
        return repo.replace_in_snapshot(snapshot, (current, previous), events)

    coordinator.mutate_profile_registry(
        repo,
        None,
        mutation,
    )
    typer.echo(f"rolled back {profile_key} to version {restored_version['value']}")


@app.command("drift")
def drift(
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
    profile_version: int,
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    payload = json.loads(observations.read_text())
    decision = evaluate_drift(
        tuple(DriftObservation.model_validate(item) for item in payload["observations"])
    )
    if decision.drifted:
        repo, coordinator = _writer(registry, alias_registry, jobs_root)

        def mutation(snapshot: ProfileRegistrySnapshot) -> ProfileRegistrySnapshot:
            current = next(
                (
                    item
                    for item in snapshot.profiles
                    if item.profile_key == profile_key
                    and item.profile_version == profile_version
                ),
                None,
            )
            if current is None:
                raise KeyError(f"unknown profile: {profile_key}@{profile_version}")
            updated, event = transition_profile(
                current,
                ProfileLifecycle.DRIFTED,
                ",".join(decision.reasons) or "drift gate",
            )
            return repo.replace_in_snapshot(snapshot, (updated,), (event,))

        coordinator.mutate_profile_registry(
            repo,
            None,
            mutation,
        )
    typer.echo(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("validate")
def validate_registries(
    registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    repository, coordinator = _writer(registry, alias_registry, jobs_root)
    try:
        profiles, projection, identities = coordinator.inspect_identity_snapshots(
            repository
        )
    except AliasRegistryUnavailable:
        typer.echo(json.dumps({"status": "invalid", "code": "alias_registry_unavailable"}))
        raise typer.Exit(1) from None
    except ProfileRegistryUnavailable:
        typer.echo(
            json.dumps({"status": "invalid", "code": "profile_registry_unavailable"})
        )
        raise typer.Exit(1) from None
    except HospitalIdentityConflict as error:
        typer.echo(
            json.dumps(
                {
                    "status": "invalid",
                    "code": "hospital_identity_conflict",
                    "owner_ids": error.owner_ids,
                },
                sort_keys=True,
            )
        )
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "profile_revision": profiles.revision,
                "alias_registry_persisted_exists": projection.persisted_exists,
                "alias_registry_version": projection.persisted["registry_version"],
                "alias_registry_revision": projection.persisted["revision"],
                "projected_alias_registry_version": projection.projected[
                    "registry_version"
                ],
                "projected_alias_registry_revision": projection.projected["revision"],
                "pending_journal_count": projection.pending_journal_count,
                "recovery_required": projection.recovery_required,
                "migration_required": projection.migration_required,
                "active_hospital_count": len(identities),
            },
            sort_keys=True,
        )
    )


@app.command("check-access")
def check_access(
    registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_PROFILE_REGISTRY"),
    ],
    alias_registry: Annotated[
        Path,
        typer.Option(dir_okay=False, envvar="GMONEY_ALIAS_REGISTRY"),
    ],
    jobs_root: Annotated[
        Path,
        typer.Option(file_okay=False, envvar="GMONEY_JOBS_ROOT"),
    ],
) -> None:
    repository, coordinator = _writer(registry, alias_registry, jobs_root)
    directories = (registry.parent, alias_registry.parent, jobs_root)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, probe = tempfile.mkstemp(prefix=".gmoney-access-", dir=directory)
        try:
            os.write(descriptor, b"profile-admin-access-check\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            Path(probe).unlink(missing_ok=True)
    with repository.lock(exclusive=True), coordinator.repository.lock(exclusive=True):
        pass
    typer.echo("profile administration paths and locks are writable")


if __name__ == "__main__":
    app()
