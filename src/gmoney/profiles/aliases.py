from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_SH, flock
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

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
    **{
        field: field
        for field in ALIAS_CANONICAL_FIELDS
        if field != "service_date"
    },
}


def normalize_alias(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def empty_alias_registry() -> dict[str, Any]:
    return {
        "registry_version": "hospital_alias_registry_v1",
        "revision": 0,
        "hospitals": [],
        "aliases": [],
        "events": [],
    }


class AliasRegistryRevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(f"alias registry revision is {current_revision}")
        self.current_revision = current_revision


class JsonAliasRepository:
    """Small atomic registry shared by the demo API and extraction workers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def _validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("registry_version") != "hospital_alias_registry_v1":
            raise ValueError("unsupported hospital alias registry")
        if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
            raise ValueError("invalid hospital alias registry revision")
        hospitals = payload.get("hospitals")
        aliases = payload.get("aliases")
        events = payload.get("events")
        if (
            not isinstance(hospitals, list)
            or not isinstance(aliases, list)
            or not isinstance(events, list)
        ):
            raise ValueError("invalid hospital alias registry collections")
        hospital_ids = {str(item.get("hospital_id") or "") for item in hospitals}
        if "" in hospital_ids or len(hospital_ids) != len(hospitals):
            raise ValueError("hospital alias registry has duplicate hospital IDs")
        alias_ids = {str(item.get("alias_id") or "") for item in aliases}
        if "" in alias_ids or len(alias_ids) != len(aliases):
            raise ValueError("hospital alias registry has duplicate alias IDs")
        alias_keys = {
            (str(item.get("hospital_id") or ""), str(item.get("normalized_label") or ""))
            for item in aliases
        }
        if len(alias_keys) != len(aliases):
            raise ValueError("hospital alias registry has duplicate normalized labels")
        for alias in aliases:
            if alias.get("hospital_id") not in hospital_ids:
                raise ValueError("hospital alias references an unknown hospital")
            if alias.get("canonical_field") not in ALIAS_CANONICAL_FIELDS:
                raise ValueError("hospital alias references an unsupported field")
            if normalize_alias(str(alias.get("source_label") or "")) != alias.get(
                "normalized_label"
            ):
                raise ValueError("hospital alias has an invalid normalized label")
        return payload

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return empty_alias_registry()
        return self._validate(json.loads(self.path.read_text()))

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        validated = self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    @contextmanager
    def locked(self, *, exclusive: bool) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            flock(lock.fileno(), LOCK_EX if exclusive else LOCK_SH)
            yield self._read_unlocked()

    def read(self) -> dict[str, Any]:
        with self.locked(exclusive=False) as payload:
            return json.loads(json.dumps(payload))

    def mutate(
        self,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        with self.locked(exclusive=True) as current:
            if current["revision"] != expected_revision:
                raise AliasRegistryRevisionConflict(current["revision"])
            updated = mutation(json.loads(json.dumps(current)))
            updated["revision"] = current["revision"] + 1
            self._write_unlocked(updated)
            return updated

    @staticmethod
    def hospital_id_for_name(name: str) -> str:
        normalized = normalize_alias(name)
        return str(uuid5(NAMESPACE_URL, f"gmoney:hospital:{normalized}"))

    @staticmethod
    def resolve_hospital(snapshot: dict[str, Any], name: str | None) -> str | None:
        normalized = normalize_alias(name or "")
        if not normalized:
            return None
        matches = [
            item["hospital_id"]
            for item in snapshot["hospitals"]
            if normalized in item.get("normalized_names", [])
        ]
        return matches[0] if len(set(matches)) == 1 else None

    @staticmethod
    def active_aliases(
        snapshot: dict[str, Any], hospital_id: str | None
    ) -> tuple[dict[str, Any], ...]:
        if not hospital_id:
            return ()
        return tuple(
            alias
            for alias in snapshot["aliases"]
            if alias["hospital_id"] == hospital_id and alias.get("active", True)
        )

    @staticmethod
    def new_alias_id() -> str:
        return str(uuid4())
