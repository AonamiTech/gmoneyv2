from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

ACTIVE_STATUSES = {"uploading", "queued", "processing"}
TERMINAL_STATUSES = {"complete", "failed"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class ReviewRevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__("review_revision_conflict")
        self.current_revision = current_revision


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
            "hospital_name": None,
            "hospital_confidence": None,
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

    @staticmethod
    def empty_review() -> dict[str, Any]:
        return {
            "revision": 0,
            "updated_at": utc_now(),
            "row_overrides": {},
            "added_rows": {},
            "issue_overrides": {},
            "document_overrides": {},
            "events": [],
            "approval": None,
        }

    def read_review(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "review.json"
        if not path.is_file():
            return self.empty_review()
        review = json.loads(path.read_text())
        for key, value in self.empty_review().items():
            review.setdefault(key, value)
        return review

    @contextmanager
    def _review_lock(self, job_id: str) -> Iterator[None]:
        lock_path = self.job_dir(job_id) / ".review.lock"
        with lock_path.open("a+") as lock:
            flock(lock.fileno(), LOCK_EX)
            try:
                yield
            finally:
                flock(lock.fileno(), LOCK_UN)

    def mutate_review(
        self,
        job_id: str,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        with self._review_lock(job_id):
            current = self.read_review(job_id)
            if current["revision"] != expected_revision:
                raise ReviewRevisionConflict(current["revision"])
            updated = mutation(json.loads(json.dumps(current)))
            updated["revision"] = current["revision"] + 1
            updated["updated_at"] = utc_now()
            path = self.job_dir(job_id) / "review.json"
            temporary = path.with_name(f"review.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
            return updated

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

    def last_activity(self, job_id: str, state: dict[str, Any] | None = None) -> datetime:
        current = state or self.read(job_id)
        updated = parse_utc(str(current["updated_at"]))
        review_path = self.job_dir(job_id) / "review.json"
        if review_path.is_file():
            try:
                review_updated = parse_utc(str(json.loads(review_path.read_text())["updated_at"]))
                updated = max(updated, review_updated)
            except (KeyError, OSError, json.JSONDecodeError, ValueError):
                pass
        return updated

    def expires_at(
        self,
        job_id: str,
        retention_hours: int,
        state: dict[str, Any] | None = None,
    ) -> str | None:
        current = state or self.read(job_id)
        if current.get("status") not in TERMINAL_STATUSES:
            return None
        return utc_text(self.last_activity(job_id, current) + timedelta(hours=retention_hours))

    def delete(self, job_id: str) -> None:
        state = self.read(job_id)
        if state.get("status") in ACTIVE_STATUSES:
            raise RuntimeError("active_job")
        shutil.rmtree(self.job_dir(job_id))

    def cleanup(self, retention_hours: int) -> int:
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        removed = 0
        for state in self.states():
            if state.get("status") in ACTIVE_STATUSES:
                continue
            updated = self.last_activity(state["id"], state)
            if updated < cutoff:
                shutil.rmtree(self.job_dir(state["id"]), ignore_errors=True)
                removed += 1
        return removed
