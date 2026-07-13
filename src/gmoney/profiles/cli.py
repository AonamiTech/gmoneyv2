from __future__ import annotations

import json
import resource
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
)
from gmoney.profiles.construction import build_profile
from gmoney.profiles.lifecycle import evaluate_drift, rollback_profile, transition_profile
from gmoney.profiles.matching import match_profile
from gmoney.profiles.repository import JsonProfileRepository

app = typer.Typer(no_args_is_help=True)


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
    registry: Annotated[Path, typer.Option(dir_okay=False)],
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
    profile_version: int,
    hospital_id: str | None = None,
    global_family: str | None = None,
) -> None:
    payload = json.loads(observations.read_text())
    items = tuple(LayoutObservation.model_validate(item) for item in payload["observations"])
    profile = build_profile(
        items,
        profile_key=profile_key,
        profile_version=profile_version,
        hospital_id=hospital_id,
        global_family=global_family,
        construction_dataset_ids=tuple(payload["construction_dataset_ids"]),
    )
    JsonProfileRepository(registry).add_profile(profile)
    typer.echo(json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("add")
def add(
    registry: Annotated[Path, typer.Option(dir_okay=False)],
    profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    item = LayoutProfile.model_validate_json(profile.read_text())
    JsonProfileRepository(registry).add_profile(item)
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
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
    profile_version: int,
    to_state: ProfileLifecycle,
    reason: str,
) -> None:
    repo = JsonProfileRepository(registry)
    current = repo.get(profile_key, profile_version)
    updated, event = transition_profile(current, to_state, reason)
    replacements = [updated]
    events = [event]
    if to_state is ProfileLifecycle.ACTIVE:
        active = next(
            (
                item
                for item in repo.list_profiles()
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
            updated = updated.model_copy(update={"supersedes_version": active.profile_version})
            replacements[0] = updated
    repo.replace_profiles(tuple(replacements), tuple(events))
    typer.echo(f"{profile_key}@{profile_version}: {current.lifecycle} -> {to_state}")


@app.command("rollback")
def rollback(
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
) -> None:
    repo = JsonProfileRepository(registry)
    current, previous = rollback_profile(repo.list_profiles(), profile_key)
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
            from_state=repo.get(
                previous.profile_key, previous.profile_version
            ).lifecycle,
            to_state=ProfileLifecycle.ACTIVE,
            reason=f"rollback from version {current.profile_version}",
            occurred_at=now,
        ),
    )
    repo.replace_profiles((current, previous), events)
    typer.echo(f"rolled back {profile_key} to version {previous.profile_version}")


@app.command("drift")
def drift(
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile_key: str,
    profile_version: int,
) -> None:
    payload = json.loads(observations.read_text())
    decision = evaluate_drift(
        tuple(DriftObservation.model_validate(item) for item in payload["observations"])
    )
    if decision.drifted:
        repo = JsonProfileRepository(registry)
        current = repo.get(profile_key, profile_version)
        updated, event = transition_profile(
            current,
            ProfileLifecycle.DRIFTED,
            ",".join(decision.reasons) or "drift gate",
        )
        repo.replace_profile(updated, event)
    typer.echo(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
