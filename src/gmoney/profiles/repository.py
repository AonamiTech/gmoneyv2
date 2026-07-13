from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from gmoney.contracts.phase3 import (
    LayoutProfile,
    ProfileEvent,
    ProfileLifecycle,
    ProfileRegistrySnapshot,
)


class ProfileRepository(Protocol):
    def list_profiles(self) -> tuple[LayoutProfile, ...]: ...

    def add_profile(self, profile: LayoutProfile) -> None: ...

    def replace_profile(
        self, profile: LayoutProfile, event: ProfileEvent | None = None
    ) -> None: ...

    def replace_profiles(
        self,
        profiles: tuple[LayoutProfile, ...],
        events: tuple[ProfileEvent, ...] = (),
    ) -> None: ...

    def events(self) -> tuple[ProfileEvent, ...]: ...


class JsonProfileRepository:
    """Atomic file-backed registry used by the Phase 3 offline pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> ProfileRegistrySnapshot:
        if not self.path.exists():
            return ProfileRegistrySnapshot()
        return ProfileRegistrySnapshot.model_validate_json(self.path.read_text())

    def _write(self, snapshot: ProfileRegistrySnapshot) -> None:
        snapshot = ProfileRegistrySnapshot.model_validate(snapshot.model_dump())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(self.path)

    def list_profiles(self) -> tuple[LayoutProfile, ...]:
        return self._load().profiles

    def events(self) -> tuple[ProfileEvent, ...]:
        return self._load().events

    def add_profile(self, profile: LayoutProfile) -> None:
        snapshot = self._load()
        identity = (profile.profile_key, profile.profile_version)
        if any((item.profile_key, item.profile_version) == identity for item in snapshot.profiles):
            raise ValueError(
                "profile version already exists: "
                f"{profile.profile_key}@{profile.profile_version}"
            )
        self._write(
            snapshot.model_copy(
                update={
                    "profiles": tuple(
                        sorted(
                            (*snapshot.profiles, profile),
                            key=lambda item: (item.profile_key, item.profile_version),
                        )
                    )
                }
            )
        )

    def replace_profile(self, profile: LayoutProfile, event: ProfileEvent | None = None) -> None:
        self.replace_profiles((profile,), (event,) if event else ())

    def replace_profiles(
        self,
        profiles: tuple[LayoutProfile, ...],
        events: tuple[ProfileEvent, ...] = (),
    ) -> None:
        snapshot = self._load()
        replacements = {
            (profile.profile_key, profile.profile_version): profile for profile in profiles
        }
        if len(replacements) != len(profiles):
            raise ValueError("duplicate profile replacement identity")
        replaced: set[tuple[str, int]] = set()
        updated: list[LayoutProfile] = []
        for item in snapshot.profiles:
            identity = (item.profile_key, item.profile_version)
            if identity in replacements:
                updated.append(replacements[identity])
                replaced.add(identity)
            else:
                updated.append(item)
        missing = replacements.keys() - replaced
        if missing:
            rendered = ", ".join(f"{key}@{version}" for key, version in sorted(missing))
            raise KeyError(f"unknown profile versions: {rendered}")
        self._write(
            snapshot.model_copy(
                update={
                    "profiles": tuple(updated),
                    "events": (*snapshot.events, *events),
                }
            )
        )

    def get(self, profile_key: str, profile_version: int | None = None) -> LayoutProfile:
        matches = [item for item in self.list_profiles() if item.profile_key == profile_key]
        if profile_version is not None:
            matches = [item for item in matches if item.profile_version == profile_version]
        if not matches:
            suffix = f"@{profile_version}" if profile_version is not None else ""
            raise KeyError(f"unknown profile: {profile_key}{suffix}")
        return max(matches, key=lambda item: item.profile_version)

    def active_profiles(self) -> tuple[LayoutProfile, ...]:
        return tuple(
            item for item in self.list_profiles() if item.lifecycle is ProfileLifecycle.ACTIVE
        )
