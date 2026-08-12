from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from gmoney.demo.store import (
    ACTIVE_STATUSES,
    JobStore,
    JobTransactionError,
    ReviewRevisionConflict,
    utc_now,
)
from gmoney.profiles.aliases import (
    AliasRegistryFormatError,
    AliasRegistryRevisionConflict,
    AliasRegistryUnavailable,
    JsonAliasRepository,
    durable_json_replace,
    durable_unlink,
)

T = TypeVar("T")
ALIAS_JOURNAL = ".alias-operation.json"
ALIAS_JOURNAL_VERSION = "alias_operation_v2"
ALIAS_STORAGE_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    TypeError,
    UnicodeError,
    json.JSONDecodeError,
    AliasRegistryFormatError,
)


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class AliasTransactionCoordinator:
    """Coordinates the global alias registry with per-document review state."""

    def __init__(self, store: JobStore, registry_path: Path) -> None:
        self.store = store
        self.repository = JsonAliasRepository(registry_path)

    def journal_path(self, job_id: str) -> Path:
        return self.store.job_dir(job_id) / ALIAS_JOURNAL

    @staticmethod
    def _assert_image(journal: dict[str, Any], name: str) -> dict[str, Any]:
        payload = journal.get(name)
        if not isinstance(payload, dict) or _digest(payload) != journal.get(f"{name}_sha256"):
            raise AliasRegistryUnavailable(f"invalid {name} in alias operation journal")
        return payload

    def _recover_unlocked(self, job_id: str) -> None:
        journal_path = self.journal_path(job_id)
        if not journal_path.is_file():
            return
        try:
            journal = json.loads(journal_path.read_text())
            if journal.get("version") != ALIAS_JOURNAL_VERSION or journal.get("job_id") != job_id:
                raise AliasRegistryUnavailable("unsupported alias operation journal")
            base_review = self._assert_image(journal, "base_review")
            target_review = self._assert_image(journal, "target_review")
            base_registry = self._assert_image(journal, "base_registry")
            target_registry = self._assert_image(journal, "target_registry")
            self.repository._validate(base_registry)
            self.repository._validate(target_registry)
            if target_review.get("revision") != base_review.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid review revision in alias journal")
            if target_registry.get("revision") != base_registry.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid registry revision in alias journal")

            review_path = self.store.job_dir(job_id) / "review.json"
            current_review = (
                self.store._read_review_unlocked(job_id) if review_path.is_file() else base_review
            )
            current_registry = self.repository._read_unlocked()
            if _digest(current_review) not in {_digest(base_review), _digest(target_review)}:
                raise AliasRegistryUnavailable("newer review state blocks alias recovery")
            if _digest(current_registry) not in {
                _digest(base_registry),
                _digest(target_registry),
            }:
                raise AliasRegistryUnavailable("newer registry state blocks alias recovery")

            self.repository._write_unlocked(target_registry)
            durable_json_replace(
                self.store.job_dir(job_id) / "review.json",
                target_review,
                suffix="review.recovery.tmp",
            )
            durable_unlink(journal_path)
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias operation recovery failed") from error

    def recover_job(self, job_id: str) -> None:
        if not self.journal_path(job_id).is_file():
            return
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias operation recovery failed") from error

    def recover_all(self) -> None:
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry recovery failed") from error

    def _recover_all_unlocked(self) -> None:
        """Recover every journal while the caller continuously holds the registry lock."""
        journals = sorted(self.store.jobs_root.glob(f"*/{ALIAS_JOURNAL}"))
        for journal_path in journals:
            job_id = journal_path.parent.name
            with self.store.job_lock(job_id, exclusive=True):
                self._recover_unlocked(job_id)

    def registry_snapshot(self) -> dict[str, Any]:
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
                return json.loads(json.dumps(self.repository._read_unlocked()))
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry is unavailable") from error

    def mutate_registry(
        self,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Recover, compare, and mutate without releasing the registry lock."""
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
                current = self.repository._read_unlocked()
                if current["revision"] != expected_revision:
                    raise AliasRegistryRevisionConflict(current["revision"])
                updated = mutation(json.loads(json.dumps(current)))
                updated["revision"] = current["revision"] + 1
                self.repository._write_unlocked(updated)
                return updated
        except AliasRegistryRevisionConflict:
            raise
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry mutation failed") from error

    def run_job_operation(self, job_id: str, operation: Callable[[], T]) -> T:
        for attempt in range(2):
            try:
                return operation()
            except JobTransactionError as error:
                if str(error) != "alias_operation_recovery_required" or attempt:
                    raise
                self.recover_job(job_id)
        raise AliasRegistryUnavailable("alias operation recovery retry failed")

    def mutate_review(
        self,
        job_id: str,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        return self.run_job_operation(
            job_id,
            lambda: self.store.mutate_review(job_id, expected_revision, mutation),
        )

    def mutate_review_and_registry(
        self,
        job_id: str,
        expected_review_revision: int,
        expected_registry_revision: int,
        mutation: Callable[
            [dict[str, Any], dict[str, Any], dict[str, Any]],
            tuple[dict[str, Any], dict[str, Any]],
        ],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
                with self.store.job_lock(job_id, exclusive=True):
                    self.store._require_stable_workspace(job_id)
                    registry = self.repository._read_unlocked()
                    review = self.store._read_review_unlocked(job_id)
                    review_path = self.store.job_dir(job_id) / "review.json"
                    if not review_path.is_file():
                        durable_json_replace(review_path, review, suffix="review.base.tmp")
                    result_path = self.store.job_dir(job_id) / "result.json"
                    if not result_path.is_file():
                        raise JobTransactionError("extraction_result_unavailable")
                    result = json.loads(result_path.read_text())
                    if registry["revision"] != expected_registry_revision:
                        raise AliasRegistryRevisionConflict(registry["revision"])
                    if review["revision"] != expected_review_revision:
                        raise ReviewRevisionConflict(review["revision"])
                    updated_review, updated_registry = mutation(
                        json.loads(json.dumps(review)),
                        json.loads(json.dumps(registry)),
                        result,
                    )
                    updated_review["revision"] = review["revision"] + 1
                    updated_review["updated_at"] = utc_now()
                    updated_registry["revision"] = registry["revision"] + 1
                    self.repository._validate(updated_registry)
                    journal = {
                        "version": ALIAS_JOURNAL_VERSION,
                        "job_id": job_id,
                        "base_review": review,
                        "target_review": updated_review,
                        "base_registry": registry,
                        "target_registry": updated_registry,
                    }
                    journal.update(
                        {
                            f"{name}_sha256": _digest(payload)
                            for name, payload in tuple(journal.items())
                            if name
                            in {
                                "base_review",
                                "target_review",
                                "base_registry",
                                "target_registry",
                            }
                        }
                    )
                    durable_json_replace(
                        self.journal_path(job_id), journal, suffix="journal.tmp"
                    )
                    self.repository._write_unlocked(updated_registry)
                    durable_json_replace(
                        self.store.job_dir(job_id) / "review.json",
                        updated_review,
                        suffix="review.tmp",
                    )
                    durable_unlink(self.journal_path(job_id))
                    return updated_review, updated_registry
        except (AliasRegistryRevisionConflict, ReviewRevisionConflict):
            raise
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry mutation failed") from error

    def delete_job(self, job_id: str) -> None:
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
                with self.store.job_lock(job_id, exclusive=True):
                    state = self.store.read(job_id)
                    if state.get("status") in ACTIVE_STATUSES:
                        raise RuntimeError("active_job")
                    shutil.rmtree(self.store.job_dir(job_id))
        except (AliasRegistryUnavailable, RuntimeError):
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry deletion failed") from error

    def cleanup(self, retention_hours: int) -> int:
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        removed = 0
        try:
            with self.repository.locked(exclusive=True):
                self._recover_all_unlocked()
                for observed in self.store.states():
                    job_id = str(observed["id"])
                    with self.store.job_lock(job_id, exclusive=True):
                        try:
                            state = self.store.read(job_id)
                        except KeyError:
                            continue
                        if state.get("status") in ACTIVE_STATUSES:
                            continue
                        if self.store.last_activity(job_id, state) >= cutoff:
                            continue
                        shutil.rmtree(self.store.job_dir(job_id))
                        removed += 1
            return removed
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry cleanup failed") from error
