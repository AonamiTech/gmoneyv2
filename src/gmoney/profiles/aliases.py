from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_SH, flock
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from gmoney.normalization import (
    HEADER_NORMALIZER_VERSION,
    HOSPITAL_NORMALIZER_VERSION,
    normalize_header,
    normalize_hospital_name,
)

LEGACY_ALIAS_REGISTRY_VERSION = "hospital_alias_registry_v2"
ALIAS_REGISTRY_VERSION = "hospital_alias_registry_v3"

ALIAS_CANONICAL_FIELDS = frozenset(
    {
        "description",
        "section",
        "service_date",
        "request_no",
        "service_code",
        "hsn_code",
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }
)

CANONICAL_TO_HEADER_ROLE = {
    "description": "description",
    "section": "section",
    "service_date": "service_date",
    "request_no": "request_no",
    "service_code": "service_code",
    "hsn_code": "hsn_code",
    "quantity": "quantity",
    "unit_price": "rate",
    "gross_amount": "gross_amount",
    "discount": "discount",
    "net_amount": "amount",
}

CANONICAL_TO_SOURCE_FIELD = {
    "service_date": "service_date_raw",
    **{field: field for field in ALIAS_CANONICAL_FIELDS if field != "service_date"},
}


# Kept as a compatibility import for callers outside the demo API.
normalize_alias = normalize_header


def empty_alias_registry() -> dict[str, Any]:
    return {
        "registry_version": ALIAS_REGISTRY_VERSION,
        "revision": 0,
        "header_normalizer_version": HEADER_NORMALIZER_VERSION,
        "hospital_normalizer_version": HOSPITAL_NORMALIZER_VERSION,
        "hospitals": [],
        "aliases": [],
        "events": [],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json_replace(path: Path, payload: dict[str, Any], *, suffix: str = "tmp") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{suffix}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def durable_unlink(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


class AliasRegistryRevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(f"alias registry revision is {current_revision}")
        self.current_revision = current_revision


class AliasRegistryUnavailable(RuntimeError):
    """The registry or an interrupted cross-file operation cannot be trusted."""


class AliasRegistryFormatError(ValueError):
    """Persisted alias registry content does not satisfy a supported schema."""


def persisted_regular_file_exists(path: Path, *, context: str) -> bool:
    """Return false only for genuine absence; reject every other filesystem type."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AliasRegistryUnavailable(f"{context} is unavailable") from error
    if not stat.S_ISREG(status.st_mode):
        raise AliasRegistryUnavailable(f"{context} is not a regular file")
    return True


def _required_text(payload: dict[str, Any], field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AliasRegistryFormatError(f"{context} has an invalid {field}")
    return value.strip()


def _required_timestamp(payload: dict[str, Any], field: str, context: str) -> str:
    value = _required_text(payload, field, context)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AliasRegistryFormatError(f"{context} has an invalid {field}") from error
    if parsed.tzinfo is None:
        raise AliasRegistryFormatError(f"{context} has an invalid {field}")
    return value


class JsonAliasRepository:
    """Versioned, durable registry shared by the demo API and extraction workers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def _validate_contents(
        self,
        payload: dict[str, Any],
        *,
        require_event_reviewer: bool,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AliasRegistryFormatError("hospital alias registry is not an object")
        if payload.get("header_normalizer_version") != HEADER_NORMALIZER_VERSION:
            raise AliasRegistryFormatError("unsupported header normalizer")
        if payload.get("hospital_normalizer_version") != HOSPITAL_NORMALIZER_VERSION:
            raise AliasRegistryFormatError("unsupported hospital-name normalizer")
        if type(payload.get("revision")) is not int or payload["revision"] < 0:
            raise AliasRegistryFormatError("invalid hospital alias registry revision")
        hospitals = payload.get("hospitals")
        aliases = payload.get("aliases")
        events = payload.get("events")
        if not all(isinstance(item, list) for item in (hospitals, aliases, events)):
            raise AliasRegistryFormatError("invalid hospital alias registry collections")
        if not all(isinstance(item, dict) for item in (*hospitals, *aliases, *events)):
            raise AliasRegistryFormatError("invalid hospital alias registry entry")

        hospital_ids = {
            _required_text(item, "hospital_id", "hospital registry entry")
            for item in hospitals
        }
        if "" in hospital_ids or len(hospital_ids) != len(hospitals):
            raise AliasRegistryFormatError(
                "hospital alias registry has duplicate hospital IDs"
            )
        name_owners: dict[str, str] = {}
        for hospital in hospitals:
            hospital_id = _required_text(hospital, "hospital_id", "hospital registry entry")
            hospital_name = _required_text(
                hospital, "hospital_name", "hospital registry entry"
            )
            canonical_normalized = normalize_hospital_name(hospital_name)
            if not canonical_normalized:
                raise AliasRegistryFormatError(
                    "hospital alias registry has an invalid hospital name"
                )
            owner = name_owners.setdefault(canonical_normalized, hospital_id)
            if owner != hospital_id:
                raise AliasRegistryFormatError(
                    "hospital name belongs to multiple hospitals"
                )
            _required_timestamp(hospital, "created_at", "hospital registry entry")
            _required_timestamp(hospital, "updated_at", "hospital registry entry")
            origins = hospital.get("origins")
            if (
                not isinstance(origins, list)
                or not origins
                or not all(isinstance(origin, str) for origin in origins)
                or len(set(origins)) != len(origins)
                or not set(origins)
                <= {
                    "profile",
                    "reviewer_alias",
                }
            ):
                raise AliasRegistryFormatError("hospital alias registry has invalid origins")
            variants = hospital.get("name_variants")
            if (
                not isinstance(variants, list)
                or not variants
                or not all(isinstance(variant, dict) for variant in variants)
            ):
                raise AliasRegistryFormatError(
                    "hospital alias registry has no verified name variants"
                )
            seen: set[str] = set()
            for variant in variants:
                display = _required_text(variant, "display_name", "hospital name variant")
                normalized = _required_text(
                    variant, "normalized_name", "hospital name variant"
                )
                if not display or normalize_hospital_name(display) != normalized:
                    raise AliasRegistryFormatError(
                        "hospital alias registry has an invalid name variant"
                    )
                if variant.get("verified") is not True or normalized in seen:
                    raise AliasRegistryFormatError(
                        "hospital alias registry has duplicate/unverified variants"
                    )
                _required_text(variant, "source_document_id", "hospital name variant")
                _required_text(variant, "reviewer", "hospital name variant")
                _required_text(variant, "reason", "hospital name variant")
                _required_timestamp(variant, "created_at", "hospital name variant")
                owner = name_owners.setdefault(normalized, hospital_id)
                if owner != hospital_id:
                    raise AliasRegistryFormatError(
                        "hospital name variant belongs to multiple hospitals"
                    )
                seen.add(normalized)

        alias_ids = {
            _required_text(item, "alias_id", "hospital alias") for item in aliases
        }
        if "" in alias_ids or len(alias_ids) != len(aliases):
            raise AliasRegistryFormatError("hospital alias registry has duplicate alias IDs")
        alias_keys = {
            (
                _required_text(item, "hospital_id", "hospital alias"),
                _required_text(item, "normalized_label", "hospital alias"),
            )
            for item in aliases
        }
        if len(alias_keys) != len(aliases):
            raise AliasRegistryFormatError(
                "hospital alias registry has duplicate normalized labels"
            )
        for alias in aliases:
            if alias.get("hospital_id") not in hospital_ids:
                raise AliasRegistryFormatError(
                    "hospital alias references an unknown hospital"
                )
            if alias.get("canonical_field") not in ALIAS_CANONICAL_FIELDS:
                raise AliasRegistryFormatError(
                    "hospital alias references an unsupported field"
                )
            source_label = _required_text(alias, "source_label", "hospital alias")
            if normalize_header(source_label) != alias.get(
                "normalized_label"
            ):
                raise AliasRegistryFormatError(
                    "hospital alias has an invalid normalized label"
                )
            if type(alias.get("active")) is not bool:
                raise AliasRegistryFormatError("hospital alias has invalid active state")
            _required_text(alias, "reason", "hospital alias")
            _required_timestamp(alias, "created_at", "hospital alias")
            _required_timestamp(alias, "updated_at", "hospital alias")
        for event in events:
            _required_text(event, "action", "hospital alias audit event")
            if require_event_reviewer or "reviewer" in event:
                _required_text(event, "reviewer", "hospital alias audit event")
            _required_text(event, "reason", "hospital alias audit event")
            _required_timestamp(event, "created_at", "hospital alias audit event")
        return payload

    def _validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get(
            "registry_version"
        ) != ALIAS_REGISTRY_VERSION:
            raise AliasRegistryFormatError("unsupported hospital alias registry")
        return self._validate_contents(payload, require_event_reviewer=True)

    def _validate_supported(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AliasRegistryFormatError("hospital alias registry is not an object")
        version = payload.get("registry_version")
        if version == ALIAS_REGISTRY_VERSION:
            return self._validate_contents(payload, require_event_reviewer=True)
        if version == LEGACY_ALIAS_REGISTRY_VERSION:
            return self._validate_contents(payload, require_event_reviewer=False)
        raise AliasRegistryFormatError("unsupported hospital alias registry")

    def _read_supported_unlocked(self) -> dict[str, Any]:
        if not persisted_regular_file_exists(
            self.path,
            context="hospital alias registry",
        ):
            return empty_alias_registry()
        return self._validate_supported(json.loads(self.path.read_text()))

    def _read_unlocked(self) -> dict[str, Any]:
        if not persisted_regular_file_exists(
            self.path,
            context="hospital alias registry",
        ):
            return empty_alias_registry()
        return self._validate(json.loads(self.path.read_text()))

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        durable_json_replace(self.path, self._validate(payload), suffix="registry.tmp")

    def _write_supported_unlocked(self, payload: dict[str, Any]) -> None:
        durable_json_replace(
            self.path,
            self._validate_supported(payload),
            suffix="registry.tmp",
        )

    def _migrate_unlocked(self) -> dict[str, Any]:
        current = self._read_supported_unlocked()
        if current["registry_version"] == ALIAS_REGISTRY_VERSION:
            return current
        migrated = self._migrate_image(current)
        self._write_unlocked(migrated)
        return migrated

    def _migrate_image(self, current: dict[str, Any]) -> dict[str, Any]:
        """Return the strict registry image without changing persisted state."""
        self._validate_supported(current)
        if current["registry_version"] == ALIAS_REGISTRY_VERSION:
            return json.loads(json.dumps(current))
        migrated = json.loads(json.dumps(current))
        for event in migrated["events"]:
            event.setdefault("reviewer", "legacy-unknown")
        migrated["registry_version"] = ALIAS_REGISTRY_VERSION
        migrated["revision"] = current["revision"] + 1
        migrated["events"].append(
            {
                "action": "alias_registry_migrated",
                "from_registry_version": LEGACY_ALIAS_REGISTRY_VERSION,
                "to_registry_version": ALIAS_REGISTRY_VERSION,
                "reviewer": "system:migration",
                "reason": "Migrated the alias registry to the strict v3 schema",
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        )
        return self._validate(migrated)

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
            raise AliasRegistryUnavailable("alias registry lock is unavailable") from error
        try:
            yield
        finally:
            lock.close()

    @contextmanager
    def locked(self, *, exclusive: bool) -> Iterator[dict[str, Any]]:
        with self.lock(exclusive=exclusive):
            yield self._read_supported_unlocked()

    def read(self) -> dict[str, Any]:
        with self.locked(exclusive=True):
            return json.loads(json.dumps(self._migrate_unlocked()))

    def mutate(
        self,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        with self.locked(exclusive=True):
            current = self._migrate_unlocked()
            if current["revision"] != expected_revision:
                raise AliasRegistryRevisionConflict(current["revision"])
            updated = mutation(json.loads(json.dumps(current)))
            updated["revision"] = current["revision"] + 1
            self._write_unlocked(updated)
            return updated

    @staticmethod
    def hospital_id_for_name(name: str) -> str:
        normalized = normalize_hospital_name(name)
        return str(uuid5(NAMESPACE_URL, f"gmoney:hospital:{normalized}"))

    @staticmethod
    def hospital_name_owners(
        snapshot: dict[str, Any], name: str | None
    ) -> set[str]:
        normalized = normalize_hospital_name(name or "")
        if not normalized:
            return set()
        return {
            str(item["hospital_id"])
            for item in snapshot["hospitals"]
            if normalize_hospital_name(str(item.get("hospital_name") or ""))
            == normalized
            or any(
                variant.get("normalized_name") == normalized
                for variant in item.get("name_variants", [])
            )
        }

    @staticmethod
    def resolve_hospital(snapshot: dict[str, Any], name: str | None) -> str | None:
        matches = JsonAliasRepository.hospital_name_owners(snapshot, name)
        return next(iter(matches)) if len(matches) == 1 else None

    @staticmethod
    def active_aliases(
        snapshot: dict[str, Any], hospital_id: str | None
    ) -> tuple[dict[str, Any], ...]:
        if not hospital_id:
            return ()
        return tuple(
            alias
            for alias in snapshot["aliases"]
            if alias["hospital_id"] == hospital_id and alias.get("active") is True
        )

    @staticmethod
    def new_alias_id() -> str:
        return str(uuid4())
