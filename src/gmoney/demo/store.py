from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID, uuid4

ACTIVE_STATUSES = {"uploading", "queued", "processing"}
TERMINAL_STATUSES = {"complete", "needs_review", "failed"}
ABORTABLE_STATUSES = {"queued", "processing"}
CANCELLING_STATUS = "cancelling"


def is_gpu_device(value: str) -> bool:
    return value.strip().partition(":")[0].casefold() == "gpu"


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

    @property
    def inference_lock_path(self) -> Path:
        return self.jobs_root / ".gpu-inference.lock"

    def acquire_inference_lock(
        self,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> TextIO:
        lock = self.inference_lock_path.open("a+")
        try:
            while True:
                if cancel_requested is not None and cancel_requested():
                    raise JobTransactionError("job_abort_requested")
                try:
                    flock(lock.fileno(), LOCK_EX | LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            if cancel_requested is not None and cancel_requested():
                flock(lock.fileno(), LOCK_UN)
                raise JobTransactionError("job_abort_requested")
        except BaseException:
            lock.close()
            raise
        return lock

    @contextmanager
    def inference_lock(self) -> Iterator[None]:
        lock = self.acquire_inference_lock()
        try:
            yield
        finally:
            flock(lock.fileno(), LOCK_UN)
            lock.close()

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
        self._write_state_unlocked(job_id, state)
        return state

    def _read_state_unlocked(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "state.json"
        if not (path.exists() or path.is_symlink()):
            raise KeyError(job_id)
        self._require_regular_file(path, "invalid_live_state")
        return json.loads(path.read_text())

    def read(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            return self._read_state_unlocked(job_id)

    def _write_state_unlocked(self, job_id: str, state: dict[str, Any]) -> None:
        directory = self.job_dir(job_id)
        state = {**state, "updated_at": utc_now()}
        temporary = directory / f"state.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(directory / "state.json")

    def write(self, job_id: str, state: dict[str, Any]) -> None:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._write_state_unlocked(job_id, state)

    def _durable_bytes(self, path: Path, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            self._publication_checkpoint(f"before_file_fsync:{path.name}")
            os.fsync(output.fileno())
            self._publication_checkpoint(f"after_file_fsync:{path.name}")

    def _durable_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._durable_bytes(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _publication_checkpoint(self, phase: str) -> None:
        """Crash-injection boundary used by process-level transaction tests."""

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            state.update(changes)
            self._write_state_unlocked(job_id, state)
            return state

    @property
    def abort_marker_name(self) -> str:
        return ".abort-requested"

    def abort_requested(self, job_id: str) -> bool:
        try:
            with self.job_lock(job_id, exclusive=True):
                self._recover_publication_unlocked(job_id)
                directory = self.job_dir(job_id)
                return bool(
                    (directory / self.abort_marker_name).is_file()
                    or self._read_state_unlocked(job_id).get("status")
                    == CANCELLING_STATUS
                )
        except (FileNotFoundError, KeyError):
            return True

    def request_abort(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            current = state.get("status")
            if current == CANCELLING_STATUS:
                return state
            if current not in ABORTABLE_STATUSES:
                raise RuntimeError("job_not_abortable")
            marker = self.job_dir(job_id) / self.abort_marker_name
            marker.touch(mode=0o600, exist_ok=True)
            state.update(status=CANCELLING_STATUS, error=None)
            self._write_state_unlocked(job_id, state)
            return state

    def claim_queued(
        self,
        job_id: str,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        if cancel_requested is not None and cancel_requested():
            return None
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if (
                state.get("status") != "queued"
                or (self.job_dir(job_id) / self.abort_marker_name).is_file()
            ):
                return None
            state.update(status="processing", page=0, error=None)
            self._write_state_unlocked(job_id, state)
            return state

    def requeue_claimed(self, job_id: str) -> bool:
        """Return a claimed job to the queue before it is submitted to a runner."""
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if (
                state.get("status") != "processing"
                or (self.job_dir(job_id) / self.abort_marker_name).is_file()
            ):
                return False
            state.update(status="queued", page=0, error=None)
            self._write_state_unlocked(job_id, state)
            return True

    def fail_queued(
        self,
        job_id: str,
        error: str,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> bool:
        """Fail a queued job without exposing a transient processing state."""
        if cancel_requested is not None and cancel_requested():
            return False
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if state.get("status") != "queued":
                return False
            state.update(status="failed", error=error)
            self._write_state_unlocked(job_id, state)
            return True

    def update_processing_progress(self, job_id: str, page: int, pages: int) -> bool:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if (
                state.get("status") != "processing"
                or (self.job_dir(job_id) / self.abort_marker_name).is_file()
            ):
                return False
            state.update(page=page, pages=pages)
            self._write_state_unlocked(job_id, state)
            return True

    def publish_processing_outcome(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        status: str,
        **changes: Any,
    ) -> bool:
        """Journal and publish a validated result/state pair under one job lock."""
        if status not in {"complete", "needs_review"}:
            raise ValueError("processing outcome must be complete or needs_review")
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            directory = self.job_dir(job_id)
            if (
                state.get("status") != "processing"
                or (directory / self.abort_marker_name).is_file()
            ):
                return False
            paths = {
                "journal": directory / ".publish-operation.json",
                "target_result": directory / ".publish-result.json",
                "target_state": directory / ".publish-state.json",
                "base_result": directory / ".publish-base-result.json",
                "base_state": directory / ".publish-base-state.json",
            }
            if any(path.exists() or path.is_symlink() for path in paths.values()):
                raise JobTransactionError("publication_recovery_required")
            target_state = {
                **state,
                "status": status,
                "error": None,
                **changes,
                "updated_at": utc_now(),
            }
            self._publication_checkpoint("before_target_result_staged")
            self._durable_json(paths["target_result"], result)
            self._publication_checkpoint("after_target_result_staged")
            self._publication_checkpoint("before_target_state_staged")
            self._durable_json(paths["target_state"], target_state)
            self._publication_checkpoint("after_target_state_staged")
            self._publication_checkpoint("before_base_state_staged")
            live_state = directory / "state.json"
            self._require_regular_file(live_state, "invalid_live_state")
            self._durable_bytes(paths["base_state"], live_state.read_bytes())
            self._publication_checkpoint("after_base_state_staged")
            live_result = directory / "result.json"
            if live_result.exists() or live_result.is_symlink():
                self._require_regular_file(live_result, "invalid_live_result")
                base_result_exists = True
            else:
                base_result_exists = False
            if base_result_exists:
                self._publication_checkpoint("before_base_result_staged")
                self._durable_bytes(paths["base_result"], live_result.read_bytes())
                self._publication_checkpoint("after_base_result_staged")
            journal = {
                "version": "job_publication_v2",
                "job_id": job_id,
                "base_result_exists": base_result_exists,
                "base_result_sha256": (
                    self._file_sha256(paths["base_result"])
                    if base_result_exists
                    else None
                ),
                "base_state_sha256": self._file_sha256(paths["base_state"]),
                "target_result_sha256": self._file_sha256(paths["target_result"]),
                "target_state_sha256": self._file_sha256(paths["target_state"]),
                "created_at": utc_now(),
            }
            self._publication_checkpoint("before_journal_staged")
            self._durable_json(paths["journal"], journal)
            self._publication_checkpoint("before_journal_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_journal_directory_fsync")
            self._publication_checkpoint("after_journal_staged")
            self._publication_checkpoint("journal_persisted")
            self._publication_checkpoint("before_result_replaced")
            paths["target_result"].replace(live_result)
            self._publication_checkpoint("before_result_replace_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_result_replace_directory_fsync")
            self._publication_checkpoint("after_result_replaced")
            self._publication_checkpoint("result_replaced")
            self._publication_checkpoint("before_state_replaced")
            paths["target_state"].replace(directory / "state.json")
            self._publication_checkpoint("before_state_replace_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_state_replace_directory_fsync")
            self._publication_checkpoint("after_state_replaced")
            self._publication_checkpoint("state_replaced")
            self._cleanup_publication_unlocked(directory)
            return True

    def fail_processing(self, job_id: str, error: str) -> bool:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if (
                state.get("status") != "processing"
                or (self.job_dir(job_id) / self.abort_marker_name).is_file()
            ):
                return False
            state.update(status="failed", error=error)
            self._write_state_unlocked(job_id, state)
            return True

    def publish_maintenance_result(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        expected_result_sha256: str,
        status: str,
        **changes: Any,
    ) -> bool:
        """CAS-publish a validated reprojection of a completed result."""

        if status not in {"complete", "needs_review"}:
            raise ValueError("maintenance outcome must be complete or needs_review")
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            state = self._read_state_unlocked(job_id)
            if state.get("status") not in {"complete", "needs_review"}:
                return False
            directory = self.job_dir(job_id)
            live_result = directory / "result.json"
            self._require_regular_file(live_result, "invalid_live_result")
            if self._file_sha256(live_result) != expected_result_sha256:
                raise JobTransactionError("maintenance_result_changed")
            paths = {
                "journal": directory / ".publish-operation.json",
                "target_result": directory / ".publish-result.json",
                "target_state": directory / ".publish-state.json",
                "base_result": directory / ".publish-base-result.json",
                "base_state": directory / ".publish-base-state.json",
            }
            if any(path.exists() or path.is_symlink() for path in paths.values()):
                raise JobTransactionError("publication_recovery_required")
            target_state = {
                **state,
                "status": status,
                "error": None,
                **changes,
                "updated_at": utc_now(),
            }
            self._publication_checkpoint("before_target_result_staged")
            self._durable_json(paths["target_result"], result)
            self._publication_checkpoint("after_target_result_staged")
            self._publication_checkpoint("before_target_state_staged")
            self._durable_json(paths["target_state"], target_state)
            self._publication_checkpoint("after_target_state_staged")
            self._publication_checkpoint("before_base_state_staged")
            self._require_regular_file(directory / "state.json", "invalid_live_state")
            self._durable_bytes(
                paths["base_state"], (directory / "state.json").read_bytes()
            )
            self._publication_checkpoint("after_base_state_staged")
            self._publication_checkpoint("before_base_result_staged")
            self._durable_bytes(paths["base_result"], live_result.read_bytes())
            self._publication_checkpoint("after_base_result_staged")
            journal = {
                "version": "job_publication_v2",
                "job_id": job_id,
                "base_result_exists": True,
                "base_result_sha256": self._file_sha256(paths["base_result"]),
                "base_state_sha256": self._file_sha256(paths["base_state"]),
                "target_result_sha256": self._file_sha256(paths["target_result"]),
                "target_state_sha256": self._file_sha256(paths["target_state"]),
                "created_at": utc_now(),
            }
            self._publication_checkpoint("before_journal_staged")
            self._durable_json(paths["journal"], journal)
            self._publication_checkpoint("before_journal_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_journal_directory_fsync")
            self._publication_checkpoint("after_journal_staged")
            self._publication_checkpoint("journal_persisted")
            self._publication_checkpoint("before_result_replaced")
            paths["target_result"].replace(live_result)
            self._publication_checkpoint("before_result_replace_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_result_replace_directory_fsync")
            self._publication_checkpoint("after_result_replaced")
            self._publication_checkpoint("result_replaced")
            self._publication_checkpoint("before_state_replaced")
            paths["target_state"].replace(directory / "state.json")
            self._publication_checkpoint("before_state_replace_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_state_replace_directory_fsync")
            self._publication_checkpoint("after_state_replaced")
            self._publication_checkpoint("state_replaced")
            self._cleanup_publication_unlocked(directory)
            return True

    def finalize_abort(self, job_id: str) -> bool:
        directory = self.job_dir(job_id)
        if not directory.is_dir():
            return False
        tombstone = self.jobs_root / f".{job_id}.aborted-{uuid4()}"
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            state = self._read_state_unlocked(job_id)
            if state.get("status") != CANCELLING_STATUS:
                return False
            directory.replace(tombstone)
        shutil.rmtree(tombstone, ignore_errors=True)
        return True

    @staticmethod
    def empty_review() -> dict[str, Any]:
        return {
            "revision": 0,
            "updated_at": utc_now(),
            "row_overrides": {},
            "added_rows": {},
            "column_mappings": {},
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
        directory = self.job_dir(job_id)
        publication_artifacts = (
            directory / ".publish-operation.json",
            *self._publication_stage_paths(directory),
        )
        if any(path.exists() or path.is_symlink() for path in publication_artifacts):
            raise JobTransactionError("publication_recovery_required")
        if (self.job_dir(job_id) / ".cutover.json").is_file():
            raise JobTransactionError("job_cutover_recovery_required")
        alias_journal = self.job_dir(job_id) / ".alias-operation.json"
        if alias_journal.exists() or alias_journal.is_symlink():
            raise JobTransactionError("alias_operation_recovery_required")

    def _cleanup_publication_unlocked(self, directory: Path) -> None:
        for name in (
            ".publish-result.json",
            ".publish-state.json",
            ".publish-base-result.json",
            ".publish-base-state.json",
        ):
            path = directory / name
            if path.exists() or path.is_symlink():
                self._require_regular_file(path, "invalid_publication_stage")
                self._publication_checkpoint(f"before_cleanup:{name}")
                path.unlink()
                self._publication_checkpoint(f"after_cleanup:{name}")
        self._publication_checkpoint("before_cleanup_stage_directory_fsync")
        self._fsync_directory(directory)
        self._publication_checkpoint("after_cleanup_stage_directory_fsync")
        if any(
            path.exists() or path.is_symlink()
            for path in self._publication_stage_paths(directory)
        ):
            raise JobTransactionError("publication_stage_cleanup_incomplete")
        journal = directory / ".publish-operation.json"
        if journal.exists() or journal.is_symlink():
            self._require_regular_file(journal, "invalid_publication_journal")
            self._publication_checkpoint("before_cleanup:.publish-operation.json")
            journal.unlink()
            self._publication_checkpoint("before_cleanup_journal_directory_fsync")
            self._fsync_directory(directory)
            self._publication_checkpoint("after_cleanup_journal_directory_fsync")
            self._publication_checkpoint("after_cleanup:.publish-operation.json")

    @staticmethod
    def _publication_stage_paths(directory: Path) -> tuple[Path, ...]:
        return tuple(
            directory / name
            for name in (
                ".publish-result.json",
                ".publish-state.json",
                ".publish-base-result.json",
                ".publish-base-state.json",
            )
        )

    def _discard_orphan_publication_stages_unlocked(self, directory: Path) -> None:
        stages = tuple(
            path
            for path in self._publication_stage_paths(directory)
            if path.exists() or path.is_symlink()
        )
        if not stages:
            return
        if any(not path.is_file() or path.is_symlink() for path in stages):
            raise JobTransactionError("invalid_publication_stage")
        base_state = directory / ".publish-base-state.json"
        live_state = directory / "state.json"
        if base_state.is_file() and (
            not live_state.is_file()
            or self._file_sha256(base_state) != self._file_sha256(live_state)
        ):
            raise JobTransactionError("orphan_publication_base_mismatch")
        # The protocol never replaces a live file before the journal is
        # durable. With no journal and an unchanged staged base, these files
        # are an uncommitted operation and are safe to discard.
        for path in stages:
            self._publication_checkpoint(f"before_orphan_cleanup:{path.name}")
            path.unlink()
            self._publication_checkpoint(f"after_orphan_cleanup:{path.name}")
        self._publication_checkpoint("before_orphan_cleanup_directory_fsync")
        self._fsync_directory(directory)
        self._publication_checkpoint("after_orphan_cleanup_directory_fsync")

    @staticmethod
    def _require_regular_file(path: Path, error: str) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as missing:
            raise JobTransactionError(error) from missing
        if not stat.S_ISREG(mode):
            raise JobTransactionError(error)

    def _recover_publication_unlocked(self, job_id: str) -> None:
        directory = self.job_dir(job_id)
        journal_path = directory / ".publish-operation.json"
        if not journal_path.exists() and not journal_path.is_symlink():
            self._discard_orphan_publication_stages_unlocked(directory)
            return
        self._require_regular_file(journal_path, "invalid_publication_journal")
        try:
            journal = json.loads(journal_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise JobTransactionError("invalid_publication_journal") from error
        if (
            journal.get("version") not in {"job_publication_v1", "job_publication_v2"}
            or journal.get("job_id") != job_id
            or type(journal.get("base_result_exists")) is not bool
        ):
            raise JobTransactionError("unsupported_publication_journal")
        target_result = directory / ".publish-result.json"
        target_state = directory / ".publish-state.json"
        live_result = directory / "result.json"
        live_state = directory / "state.json"
        self._require_regular_file(live_state, "invalid_live_state")
        if live_result.exists() or live_result.is_symlink():
            self._require_regular_file(live_result, "invalid_live_result")

        def matches(path: Path, expected: object) -> bool:
            if not isinstance(expected, str):
                return False
            if not (path.exists() or path.is_symlink()):
                return False
            self._require_regular_file(path, "invalid_publication_file")
            return self._file_sha256(path) == expected

        result_digest = journal.get("target_result_sha256")
        state_digest = journal.get("target_state_sha256")
        base_result_digest = journal.get("base_result_sha256")
        base_state_digest = journal.get("base_state_sha256")
        if journal.get("version") == "job_publication_v2" and (
            not isinstance(state_digest, str)
            or not isinstance(result_digest, str)
            or not isinstance(base_state_digest, str)
            or (
                journal["base_result_exists"]
                and not isinstance(base_result_digest, str)
            )
            or (
                not journal["base_result_exists"]
                and base_result_digest is not None
            )
        ):
            raise JobTransactionError("unsupported_publication_journal")
        aborting = (directory / self.abort_marker_name).is_file()
        if aborting:
            if journal.get("version") == "job_publication_v2" and not (
                matches(live_state, base_state_digest)
                or matches(live_state, state_digest)
            ):
                raise JobTransactionError("publication_live_state_digest_mismatch")
            base_result = directory / ".publish-base-result.json"
            if journal["base_result_exists"]:
                if not (
                    matches(base_result, base_result_digest)
                    if journal.get("version") == "job_publication_v2"
                    else base_result.is_file() and not base_result.is_symlink()
                ):
                    raise JobTransactionError("publication_base_result_missing")
                base_result.replace(live_result)
            elif live_result.exists() or live_result.is_symlink():
                if not matches(live_result, result_digest):
                    raise JobTransactionError("publication_live_result_digest_mismatch")
                live_result.unlink()
            base_state = directory / ".publish-base-state.json"
            if not (
                matches(base_state, base_state_digest)
                if journal.get("version") == "job_publication_v2"
                else base_state.is_file() and not base_state.is_symlink()
            ):
                raise JobTransactionError("publication_base_state_missing")
            base_state.replace(live_state)
            self._cleanup_publication_unlocked(directory)
            return

        if not matches(live_result, result_digest):
            if journal.get("version") == "job_publication_v2":
                live_is_base = (
                    matches(live_result, base_result_digest)
                    if journal["base_result_exists"]
                    else not (live_result.exists() or live_result.is_symlink())
                )
                if not live_is_base:
                    raise JobTransactionError("publication_live_result_digest_mismatch")
            if not matches(target_result, result_digest):
                raise JobTransactionError("publication_target_result_missing")
            target_result.replace(live_result)
            self._fsync_directory(directory)
        if not matches(live_state, state_digest):
            if (
                journal.get("version") == "job_publication_v2"
                and not matches(live_state, base_state_digest)
            ):
                raise JobTransactionError("publication_live_state_digest_mismatch")
            if not matches(target_state, state_digest):
                raise JobTransactionError("publication_target_state_missing")
            target_state.replace(live_state)
            self._fsync_directory(directory)
        self._cleanup_publication_unlocked(directory)

    def read_review(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            return self._read_review_unlocked(job_id)

    def read_result(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            path = self.job_dir(job_id) / "result.json"
            if not (path.exists() or path.is_symlink()):
                raise JobTransactionError("extraction_result_unavailable")
            self._require_regular_file(path, "invalid_live_result")
            return json.loads(path.read_text())

    @contextmanager
    def locked_workspace(
        self, job_id: str
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            state = self._read_state_unlocked(job_id)
            result_path = self.job_dir(job_id) / "result.json"
            if not (result_path.exists() or result_path.is_symlink()):
                raise JobTransactionError("extraction_result_unavailable")
            self._require_regular_file(result_path, "invalid_live_result")
            result = json.loads(result_path.read_text())
            review = self._read_review_unlocked(job_id)
            yield state, result, review

    def read_workspace(
        self, job_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        with self.locked_workspace(job_id) as workspace:
            return workspace

    def read_page_bytes(
        self, job_id: str, page_number: int
    ) -> tuple[dict[str, Any], bytes]:
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            state = self._read_state_unlocked(job_id)
            result_path = self.job_dir(job_id) / "result.json"
            if not (result_path.exists() or result_path.is_symlink()):
                raise JobTransactionError("extraction_result_unavailable")
            self._require_regular_file(result_path, "invalid_live_result")
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
            self._recover_publication_unlocked(job_id)
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

    def _restore_rollback_job_unlocked(
        self,
        job_id: str,
        journal: dict[str, Any],
    ) -> None:
        if (
            journal.get("version") != "job_rollback_v1"
            or journal.get("job_id") != job_id
        ):
            raise JobTransactionError("unsupported_cutover_journal")
        job_dir = self.job_dir(job_id)
        backup_dir = Path(str(journal["backup_dir"]))
        displaced_dir = Path(str(journal["displaced_dir"]))
        commit_marker = Path(str(journal["commit_marker"]))
        if (
            backup_dir.name != job_id
            or displaced_dir.name != job_id
            or commit_marker.parent != displaced_dir.parent
        ):
            raise JobTransactionError("invalid_cutover_journal_paths")

        for name in ("result.json", "artifacts"):
            live = job_dir / name
            backup = backup_dir / name
            displaced = displaced_dir / name
            if not backup.exists() and live.exists():
                live.replace(backup)
            if displaced.exists():
                if live.is_dir():
                    shutil.rmtree(live)
                elif live.exists():
                    live.unlink()
                displaced.replace(live)

        state_snapshot = displaced_dir / "state.json"
        if state_snapshot.is_file():
            self._restore_payload(
                job_dir / "state.json",
                json.loads(state_snapshot.read_text()),
            )
        review_snapshot = displaced_dir / "review.json"
        if review_snapshot.is_file():
            self._restore_payload(
                job_dir / "review.json",
                json.loads(review_snapshot.read_text()),
            )
        elif (job_dir / "review.json").is_file():
            (job_dir / "review.json").unlink()

    def _recover_rollback_batch(
        self,
        journals: list[tuple[str, Path, dict[str, Any]]],
    ) -> None:
        ordered = sorted(journals, key=lambda item: item[0])
        commit_markers = {
            str(journal.get("commit_marker")) for _, _, journal in ordered
        }
        if len(commit_markers) != 1:
            raise JobTransactionError("inconsistent_rollback_batch")
        commit_marker = Path(commit_markers.pop())
        with ExitStack() as locks:
            for job_id, _, _ in ordered:
                locks.enter_context(self.job_lock(job_id, exclusive=True))
            for job_id, journal_path, journal in ordered:
                current = json.loads(journal_path.read_text())
                if current != journal:
                    raise JobTransactionError("cutover_journal_changed")
                if (
                    current.get("version") != "job_rollback_v1"
                    or current.get("job_id") != job_id
                ):
                    raise JobTransactionError("unsupported_cutover_journal")
            if commit_marker.is_file():
                for _, journal_path, _ in ordered:
                    journal_path.unlink()
                return
            for job_id, _, journal in reversed(ordered):
                self._restore_rollback_job_unlocked(job_id, journal)
            for _, journal_path, _ in ordered:
                journal_path.unlink()
            shutil.rmtree(commit_marker.parent, ignore_errors=True)

    def states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for directory in self.jobs_root.iterdir():
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                job_id = str(UUID(directory.name))
                with self.job_lock(job_id, exclusive=True):
                    self._recover_publication_unlocked(job_id)
                    states.append(self._read_state_unlocked(job_id))
            except (
                KeyError,
                OSError,
                ValueError,
                json.JSONDecodeError,
                JobTransactionError,
            ):
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
        for directory in self.jobs_root.iterdir():
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if not any(
                path.exists() or path.is_symlink()
                for path in (
                    directory / ".publish-operation.json",
                    *self._publication_stage_paths(directory),
                )
            ):
                continue
            job_id = directory.name
            with self.job_lock(job_id, exclusive=True):
                self._recover_publication_unlocked(job_id)
        rollback_batches: dict[
            str, list[tuple[str, Path, dict[str, Any]]]
        ] = {}
        for journal_path in self.jobs_root.glob("*/.cutover.json"):
            job_id = journal_path.parent.name
            journal = json.loads(journal_path.read_text())
            if journal.get("version") == "job_rollback_v1":
                rollback_batches.setdefault(
                    str(journal.get("commit_marker")),
                    [],
                ).append((job_id, journal_path, journal))
                continue
            with self.job_lock(job_id, exclusive=True):
                self._recover_cutover_unlocked(job_id)
        for journals in rollback_batches.values():
            self._recover_rollback_batch(journals)
        for state in self.states():
            if state.get("status") == CANCELLING_STATUS:
                self.finalize_abort(state["id"])
                continue
            if state.get("status") == "processing":
                with self.job_lock(state["id"], exclusive=True):
                    self._recover_publication_unlocked(state["id"])
                    current = self._read_state_unlocked(state["id"])
                    if current.get("status") == "processing":
                        current.update(status="queued", error=None, page=0)
                        self._write_state_unlocked(state["id"], current)

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
        with self.job_lock(job_id, exclusive=True):
            self._recover_publication_unlocked(job_id)
            self._require_stable_workspace(job_id)
            state = self._read_state_unlocked(job_id)
            if state.get("status") in ACTIVE_STATUSES:
                raise RuntimeError("active_job")
            shutil.rmtree(self.job_dir(job_id))

    def cleanup(self, retention_hours: int) -> int:
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        removed = 0
        for state in self.states():
            try:
                with self.job_lock(state["id"], exclusive=True):
                    try:
                        self._recover_publication_unlocked(state["id"])
                        self._require_stable_workspace(state["id"])
                    except JobTransactionError:
                        continue
                    current = self._read_state_unlocked(state["id"])
                    if current.get("status") in ACTIVE_STATUSES:
                        continue
                    updated = self.last_activity(state["id"], current)
                    if updated < cutoff:
                        shutil.rmtree(self.job_dir(state["id"]), ignore_errors=True)
                        removed += 1
            except FileNotFoundError:
                continue
        return removed
