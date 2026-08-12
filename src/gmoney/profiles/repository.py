from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_SH, flock
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from gmoney.contracts.phase3 import (
    LayoutProfile,
    ProfileEvent,
    ProfileLifecycle,
    ProfileRegistrySnapshot,
)
from gmoney.normalization import normalize_hospital_name
from gmoney.profiles.aliases import JsonAliasRepository, durable_json_replace

LEGACY_PROFILE_REGISTRY_VERSION = "profile_registry_v1"
PROFILE_REGISTRY_VERSION = "profile_registry_v2"
PROFILE_STORAGE_ERRORS = (
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    ValidationError,
)

class ProfileRegistryUnavailable(RuntimeError):
    """The persisted profile registry cannot be read or trusted."""


class ProfileRegistryRevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(f"profile registry revision is {current_revision}")
        self.current_revision = current_revision


class ProfileRegistryFormatError(ValueError):
    """Profile identities do not satisfy the active ownership contract."""


class HospitalIdentityConflict(RuntimeError):
    def __init__(self, owner_ids: set[str]) -> None:
        super().__init__("hospital identity belongs to multiple hospitals")
        self.owner_ids = sorted(owner_ids)


class ProfileRepository(Protocol):
    def snapshot(self) -> ProfileRegistrySnapshot: ...

    def list_profiles(self) -> tuple[LayoutProfile, ...]: ...

    def add_profile(self, profile: LayoutProfile) -> None: ...

    def replace_profile(
        self,
        profile: LayoutProfile,
        event: ProfileEvent | None = None,
        *,
        expected_revision: int | None = None,
    ) -> None: ...

    def replace_profiles(
        self,
        profiles: tuple[LayoutProfile, ...],
        events: tuple[ProfileEvent, ...] = (),
        *,
        expected_revision: int | None = None,
    ) -> None: ...

    def events(self) -> tuple[ProfileEvent, ...]: ...


def active_hospital_identities(
    snapshot: ProfileRegistrySnapshot,
) -> dict[str, dict[str, Any]]:
    """Return deterministic active hospital identities after validating ownership."""
    profiles_by_hospital: dict[str, list[LayoutProfile]] = {}
    for profile in snapshot.profiles:
        if profile.lifecycle is not ProfileLifecycle.ACTIVE or not profile.hospital_id:
            continue
        profiles_by_hospital.setdefault(str(profile.hospital_id), []).append(profile)

    identities: dict[str, dict[str, Any]] = {}
    normalized_owners: dict[str, str] = {}
    for hospital_id, profiles in sorted(profiles_by_hospital.items()):
        ordered = sorted(
            profiles,
            key=lambda item: (item.profile_key, item.profile_version),
        )
        named = [
            (profile, str(profile.hospital_name).strip())
            for profile in ordered
            if profile.hospital_name and str(profile.hospital_name).strip()
        ]
        normalized_names = {
            normalize_hospital_name(name) for _, name in named
        }
        if "" in normalized_names or len(normalized_names) > 1:
            raise ProfileRegistryFormatError(
                f"active profiles disagree on the name for hospital {hospital_id}"
            )
        hospital_name = named[0][1] if named else None
        normalized_name = next(iter(normalized_names), None)
        if normalized_name is not None:
            owner = normalized_owners.setdefault(normalized_name, hospital_id)
            if owner != hospital_id:
                raise ProfileRegistryFormatError(
                    "active profile name belongs to multiple hospitals"
                )
        identities[hospital_id] = {
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "normalized_name": normalized_name,
            "active_profile_count": len(ordered),
        }
    return identities


def profile_hospital_name_owners(
    identities: dict[str, dict[str, Any]],
    name: str | None,
) -> set[str]:
    normalized = normalize_hospital_name(name or "")
    if not normalized:
        return set()
    return {
        hospital_id
        for hospital_id, identity in identities.items()
        if identity.get("normalized_name") == normalized
    }


def combined_hospital_name_owners(
    identities: dict[str, dict[str, Any]],
    aliases: dict[str, Any],
    name: str | None,
) -> set[str]:
    return profile_hospital_name_owners(
        identities,
        name,
    ) | JsonAliasRepository.hospital_name_owners(aliases, name)


def validate_combined_hospital_identities(
    identities: dict[str, dict[str, Any]],
    aliases: dict[str, Any],
) -> None:
    normalized_owners: dict[str, set[str]] = {}
    for hospital_id, identity in identities.items():
        normalized = identity.get("normalized_name")
        if normalized:
            normalized_owners.setdefault(str(normalized), set()).add(hospital_id)
    for hospital in aliases["hospitals"]:
        hospital_id = str(hospital["hospital_id"])
        names = [hospital.get("hospital_name")]
        names.extend(
            variant.get("display_name")
            for variant in hospital.get("name_variants", [])
        )
        for name in names:
            normalized = normalize_hospital_name(str(name or ""))
            if normalized:
                normalized_owners.setdefault(normalized, set()).add(hospital_id)
    for owners in normalized_owners.values():
        if len(owners) > 1:
            raise HospitalIdentityConflict(owners)


class JsonProfileRepository:
    """Versioned, locked, durable file-backed profile registry."""

    def __init__(self, path: Path, lock_path: Path | None = None) -> None:
        self.path = path
        configured_lock = os.environ.get("GMONEY_PROFILE_REGISTRY_LOCK")
        self.lock_path = lock_path or (
            Path(configured_lock)
            if configured_lock
            else path.with_suffix(f"{path.suffix}.lock")
        )

    @contextmanager
    def lock(self, *, exclusive: bool) -> Iterator[None]:
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = self.lock_path.open("a+")
            try:
                flock(lock.fileno(), LOCK_EX if exclusive else LOCK_SH)
            except OSError:
                lock.close()
                raise
        except OSError as error:
            raise ProfileRegistryUnavailable("profile registry lock is unavailable") from error
        try:
            yield
        finally:
            lock.close()

    def _load_unlocked(self) -> ProfileRegistrySnapshot:
        if not self.path.exists():
            return ProfileRegistrySnapshot()
        payload = json.loads(self.path.read_text())
        version = payload.get("registry_version") if isinstance(payload, dict) else None
        if version not in {LEGACY_PROFILE_REGISTRY_VERSION, PROFILE_REGISTRY_VERSION}:
            raise ProfileRegistryFormatError("unsupported profile registry")
        if version == LEGACY_PROFILE_REGISTRY_VERSION:
            if payload.get("revision", 0) != 0:
                raise ProfileRegistryFormatError("legacy profile registry has a revision")
            payload = {**payload, "revision": 0}
        snapshot = ProfileRegistrySnapshot.model_validate(payload)
        active_hospital_identities(snapshot)
        return snapshot

    @contextmanager
    def locked_snapshot(self) -> Iterator[ProfileRegistrySnapshot]:
        with self.lock(exclusive=False):
            try:
                snapshot = self._load_unlocked()
            except ProfileRegistryUnavailable:
                raise
            except (*PROFILE_STORAGE_ERRORS, ProfileRegistryFormatError) as error:
                raise ProfileRegistryUnavailable(
                    "profile registry is unavailable"
                ) from error
            yield snapshot

    def snapshot(self) -> ProfileRegistrySnapshot:
        with self.locked_snapshot() as snapshot:
            return snapshot.model_copy(deep=True)

    def _write_unlocked(self, snapshot: ProfileRegistrySnapshot) -> None:
        validated = ProfileRegistrySnapshot.model_validate(snapshot.model_dump())
        if validated.registry_version != PROFILE_REGISTRY_VERSION:
            raise ProfileRegistryFormatError("unsupported writable profile registry")
        active_hospital_identities(validated)
        try:
            durable_json_replace(
                self.path,
                validated.model_dump(mode="json"),
                suffix="profile-registry.tmp",
            )
        except PROFILE_STORAGE_ERRORS as error:
            raise ProfileRegistryUnavailable("profile registry write failed") from error

    def _mutate(
        self,
        mutation: Callable[[ProfileRegistrySnapshot], ProfileRegistrySnapshot],
        *,
        expected_revision: int | None = None,
    ) -> ProfileRegistrySnapshot:
        with self.lock(exclusive=True):
            try:
                current = self._load_unlocked()
            except (*PROFILE_STORAGE_ERRORS, ProfileRegistryFormatError) as error:
                raise ProfileRegistryUnavailable("profile registry is unavailable") from error
            if expected_revision is not None and current.revision != expected_revision:
                raise ProfileRegistryRevisionConflict(current.revision)
            updated = mutation(current.model_copy(deep=True)).model_copy(
                update={
                    "registry_version": PROFILE_REGISTRY_VERSION,
                    "revision": current.revision + 1,
                }
            )
            self._write_unlocked(updated)
            return updated

    def list_profiles(self) -> tuple[LayoutProfile, ...]:
        return self.snapshot().profiles

    def events(self) -> tuple[ProfileEvent, ...]:
        return self.snapshot().events

    def add_profile(self, profile: LayoutProfile) -> None:
        def mutation(snapshot: ProfileRegistrySnapshot) -> ProfileRegistrySnapshot:
            identity = (profile.profile_key, profile.profile_version)
            if any(
                (item.profile_key, item.profile_version) == identity
                for item in snapshot.profiles
            ):
                raise ValueError(
                    "profile version already exists: "
                    f"{profile.profile_key}@{profile.profile_version}"
                )
            return snapshot.model_copy(
                update={
                    "profiles": tuple(
                        sorted(
                            (*snapshot.profiles, profile),
                            key=lambda item: (item.profile_key, item.profile_version),
                        )
                    )
                }
            )

        self._mutate(mutation)

    def replace_profile(
        self,
        profile: LayoutProfile,
        event: ProfileEvent | None = None,
        *,
        expected_revision: int | None = None,
    ) -> None:
        self.replace_profiles(
            (profile,),
            (event,) if event else (),
            expected_revision=expected_revision,
        )

    def replace_profiles(
        self,
        profiles: tuple[LayoutProfile, ...],
        events: tuple[ProfileEvent, ...] = (),
        *,
        expected_revision: int | None = None,
    ) -> None:
        def mutation(snapshot: ProfileRegistrySnapshot) -> ProfileRegistrySnapshot:
            replacements = {
                (profile.profile_key, profile.profile_version): profile
                for profile in profiles
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
                rendered = ", ".join(
                    f"{key}@{version}" for key, version in sorted(missing)
                )
                raise KeyError(f"unknown profile versions: {rendered}")
            return snapshot.model_copy(
                update={
                    "profiles": tuple(updated),
                    "events": (*snapshot.events, *events),
                }
            )

        self._mutate(mutation, expected_revision=expected_revision)

    def get(self, profile_key: str, profile_version: int | None = None) -> LayoutProfile:
        matches = [
            item for item in self.list_profiles() if item.profile_key == profile_key
        ]
        if profile_version is not None:
            matches = [
                item for item in matches if item.profile_version == profile_version
            ]
        if not matches:
            suffix = f"@{profile_version}" if profile_version is not None else ""
            raise KeyError(f"unknown profile: {profile_key}{suffix}")
        return max(matches, key=lambda item: item.profile_version)

    def active_profiles(self) -> tuple[LayoutProfile, ...]:
        return tuple(
            item
            for item in self.list_profiles()
            if item.lifecycle is ProfileLifecycle.ACTIVE
        )
