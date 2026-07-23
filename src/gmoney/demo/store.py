from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from fcntl import LOCK_EX, LOCK_SH, LOCK_UN, flock
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


class JobTransactionError(RuntimeError):
    pass


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

    def _read_review_unlocked(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "review.json"
        if not path.is_file():
            return self.empty_review()
        review = json.loads(path.read_text())
        for key, value in self.empty_review().items():
            review.setdefault(key, value)
        return review

    @contextmanager
    def job_lock(self, job_id: str, *, exclusive: bool) -> Iterator[None]:
        lock_path = self.job_dir(job_id) / ".job.lock"
        with lock_path.open("a+") as lock:
            flock(lock.fileno(), LOCK_EX if exclusive else LOCK_SH)
            try:
                yield
            finally:
                flock(lock.fileno(), LOCK_UN)

    def _require_stable_workspace(self, job_id: str) -> None:
        if (self.job_dir(job_id) / ".cutover.json").is_file():
            raise JobTransactionError("job_cutover_recovery_required")

    def read_review(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=False):
            self._require_stable_workspace(job_id)
            return self._read_review_unlocked(job_id)

    def read_result(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=False):
            self._require_stable_workspace(job_id)
            path = self.job_dir(job_id) / "result.json"
            if not path.is_file():
                raise JobTransactionError("extraction_result_unavailable")
            return json.loads(path.read_text())

    def read_workspace(
        self, job_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        with self.job_lock(job_id, exclusive=False):
            self._require_stable_workspace(job_id)
            state = self.read(job_id)
            result_path = self.job_dir(job_id) / "result.json"
            if not result_path.is_file():
                raise JobTransactionError("extraction_result_unavailable")
            result = json.loads(result_path.read_text())
            review = self._read_review_unlocked(job_id)
            return state, result, review

    def read_page_bytes(
        self, job_id: str, page_number: int
    ) -> tuple[dict[str, Any], bytes]:
        with self.job_lock(job_id, exclusive=False):
            self._require_stable_workspace(job_id)
            state = self.read(job_id)
            result_path = self.job_dir(job_id) / "result.json"
            if not result_path.is_file():
                raise JobTransactionError("extraction_result_unavailable")
            result = json.loads(result_path.read_text())
            asset = next(
                (
                    item
                    for item in result.get("page_assets", [])
                    if item.get("page_number") == page_number
                ),
                None,
            )
            if asset is None:
                raise JobTransactionError("page_not_found")
            artifact_root = (self.job_dir(job_id) / "artifacts").resolve()
            path = (artifact_root / str(asset["relative_path"])).resolve()
            if artifact_root not in path.parents or not path.is_file():
                raise JobTransactionError("page_not_found")
            return state, path.read_bytes()

    def mutate_review(
        self,
        job_id: str,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._require_stable_workspace(job_id)
            current = self._read_review_unlocked(job_id)
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

    @staticmethod
    def _restore_payload(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.recovery.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def _recover_cutover_unlocked(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        journal_path = job_dir / ".cutover.json"
        if not journal_path.is_file():
            return
        journal = json.loads(journal_path.read_text())
        if (
            journal.get("version") != "job_cutover_v1"
            or journal.get("job_id") != job_id
        ):
            raise JobTransactionError("unsupported_cutover_journal")
        stage_dir = Path(str(journal["stage_dir"]))
        backup_dir = Path(str(journal["backup_dir"]))
        if stage_dir.name != job_id or backup_dir.name != job_id:
            raise JobTransactionError("invalid_cutover_journal_paths")
        commit_marker = (
            Path(str(journal["commit_marker"]))
            if journal.get("commit_marker")
            else None
        )
        if commit_marker is not None and commit_marker.is_file():
            journal_path.unlink()
            return

        def restore_path(name: str) -> None:
            live = job_dir / name
            backup = backup_dir / name
            staged = stage_dir / name
            if not backup.exists():
                return
            if live.exists():
                if not staged.exists():
                    live.replace(staged)
                elif live.is_dir():
                    shutil.rmtree(live)
                else:
                    live.unlink()
            backup.replace(live)

        restore_path("result.json")
        restore_path("artifacts")
        state_backup = backup_dir / "state.json"
        if state_backup.is_file():
            self._restore_payload(
                job_dir / "state.json",
                json.loads(state_backup.read_text()),
            )
        review_backup = backup_dir / "review.json"
        if review_backup.is_file():
            self._restore_payload(
                job_dir / "review.json",
                json.loads(review_backup.read_text()),
            )
        elif (job_dir / "review.json").is_file():
            (job_dir / "review.json").unlink()
        journal_path.unlink()
        shutil.rmtree(backup_dir, ignore_errors=True)

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
        for journal_path in self.jobs_root.glob("*/.cutover.json"):
            job_id = journal_path.parent.name
            with self.job_lock(job_id, exclusive=True):
                self._recover_cutover_unlocked(job_id)
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
