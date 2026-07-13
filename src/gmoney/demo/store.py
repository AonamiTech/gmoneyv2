from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

ACTIVE_STATUSES = {"queued", "processing"}
TERMINAL_STATUSES = {"complete", "failed"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs_root = root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        try:
            canonical = str(UUID(job_id))
        except ValueError as error:
            raise KeyError(job_id) from error
        return self.jobs_root / canonical

    def create(self, original_name: str) -> dict[str, Any]:
        job_id = str(uuid4())
        directory = self.jobs_root / job_id
        directory.mkdir(mode=0o700)
        state = {
            "id": job_id,
            "status": "uploading",
            "original_name": Path(original_name).name[:200],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "page": 0,
            "pages": None,
            "row_count": None,
            "error": None,
        }
        self.write(job_id, state)
        return state

    def read(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "state.json"
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text())

    def write(self, job_id: str, state: dict[str, Any]) -> None:
        directory = self.job_dir(job_id)
        state = {**state, "updated_at": utc_now()}
        temporary = directory / f"state.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(directory / "state.json")

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        state = self.read(job_id)
        state.update(changes)
        self.write(job_id, state)
        return state

    def states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for state_path in self.jobs_root.glob("*/state.json"):
            try:
                states.append(json.loads(state_path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return states

    def active_count(self) -> int:
        return sum(state.get("status") in ACTIVE_STATUSES for state in self.states())

    def queued(self) -> list[dict[str, Any]]:
        return sorted(
            (state for state in self.states() if state.get("status") == "queued"),
            key=lambda state: state["created_at"],
        )

    def recover(self) -> None:
        for state in self.states():
            if state.get("status") == "processing":
                state.update(status="queued", error=None, page=0)
                self.write(state["id"], state)

    def delete(self, job_id: str) -> None:
        state = self.read(job_id)
        if state.get("status") in ACTIVE_STATUSES:
            raise RuntimeError("active_job")
        shutil.rmtree(self.job_dir(job_id))

    def cleanup(self, retention_hours: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        removed = 0
        for state in self.states():
            if state.get("status") in ACTIVE_STATUSES:
                continue
            updated = datetime.fromisoformat(state["updated_at"].replace("Z", "+00:00"))
            if updated < cutoff:
                shutil.rmtree(self.job_dir(state["id"]), ignore_errors=True)
                removed += 1
        return removed
