from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from gmoney.contracts.phase3 import ProfileRegistrySnapshot
from gmoney.demo.store import (
    ACTIVE_STATUSES,
    JobStore,
    JobTransactionError,
    ReviewRevisionConflict,
    utc_now,
)
from gmoney.profiles.aliases import (
    ALIAS_REGISTRY_VERSION,
    AliasRegistryFormatError,
    AliasRegistryRevisionConflict,
    AliasRegistryUnavailable,
    JsonAliasRepository,
    durable_json_replace,
    durable_unlink,
)
from gmoney.profiles.repository import (
    PROFILE_STORAGE_ERRORS,
    JsonProfileRepository,
    ProfileRegistryFormatError,
    ProfileRegistryRevisionConflict,
    ProfileRegistryUnavailable,
    active_hospital_identities,
    validate_combined_hospital_identities,
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


@dataclass(frozen=True)
class AliasRegistryProjection:
    persisted: dict[str, Any]
    projected: dict[str, Any]
    pending_journal_count: int
    recovery_required: bool
    migration_required: bool
    journal_paths: tuple[Path, ...]


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
    def _storage(operation: Callable[[], T], message: str) -> T:
        try:
            return operation()
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable(message) from error

    @staticmethod
    def _assert_image(journal: dict[str, Any], name: str) -> dict[str, Any]:
        payload = journal.get(name)
        if not isinstance(payload, dict) or _digest(payload) != journal.get(f"{name}_sha256"):
            raise AliasRegistryUnavailable(f"invalid {name} in alias operation journal")
        return payload

    @staticmethod
    def _is_empty_registry_image(payload: dict[str, Any]) -> bool:
        return (
            payload.get("revision") == 0
            and payload.get("hospitals") == []
            and payload.get("aliases") == []
            and payload.get("events") == []
        )

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
            self.repository._validate_supported(base_registry)
            self.repository._validate_supported(target_registry)
            if target_review.get("revision") != base_review.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid review revision in alias journal")
            if target_registry.get("revision") != base_registry.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid registry revision in alias journal")

            review_path = self.store.job_dir(job_id) / "review.json"
            current_review = (
                self.store._read_review_unlocked(job_id) if review_path.is_file() else base_review
            )
            if self.repository.path.is_file():
                current_registry = self.repository._read_supported_unlocked()
            elif self._is_empty_registry_image(base_registry):
                current_registry = json.loads(json.dumps(base_registry))
            else:
                raise AliasRegistryUnavailable(
                    "missing registry cannot recover a non-empty alias base"
                )
            if _digest(current_review) not in {_digest(base_review), _digest(target_review)}:
                raise AliasRegistryUnavailable("newer review state blocks alias recovery")
            if _digest(current_registry) not in {
                _digest(base_registry),
                _digest(target_registry),
            }:
                raise AliasRegistryUnavailable("newer registry state blocks alias recovery")

            self.repository._write_supported_unlocked(target_registry)
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

    def _project_unlocked(self, journal_path: Path, registry: dict[str, Any]) -> dict[str, Any]:
        """Validate one journal and return its target without writing any file."""
        job_id = journal_path.parent.name
        try:
            journal = json.loads(journal_path.read_text())
            if journal.get("version") != ALIAS_JOURNAL_VERSION or journal.get("job_id") != job_id:
                raise AliasRegistryUnavailable("unsupported alias operation journal")
            base_review = self._assert_image(journal, "base_review")
            target_review = self._assert_image(journal, "target_review")
            base_registry = self._assert_image(journal, "base_registry")
            target_registry = self._assert_image(journal, "target_registry")
            self.repository._validate_supported(base_registry)
            self.repository._validate_supported(target_registry)
            if target_review.get("revision") != base_review.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid review revision in alias journal")
            if target_registry.get("revision") != base_registry.get("revision", -1) + 1:
                raise AliasRegistryUnavailable("invalid registry revision in alias journal")

            review_path = journal_path.parent / "review.json"
            current_review = (
                self.store._read_review_unlocked(job_id)
                if review_path.is_file()
                else base_review
            )
            if _digest(current_review) not in {_digest(base_review), _digest(target_review)}:
                raise AliasRegistryUnavailable("newer review state blocks alias recovery")
            if _digest(registry) not in {_digest(base_registry), _digest(target_registry)}:
                raise AliasRegistryUnavailable("newer registry state blocks alias recovery")
            return json.loads(json.dumps(target_registry))
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias operation projection failed") from error

    @contextmanager
    def _projection_unlocked(self, *, exclusive: bool = False) -> Any:
        """Hold every affected job lock and expose a read-only recovered projection."""
        journals = sorted(self.store.jobs_root.glob(f"*/{ALIAS_JOURNAL}"))
        with ExitStack() as locks:
            for journal_path in journals:
                locks.enter_context(
                    self.store.job_lock(journal_path.parent.name, exclusive=exclusive)
                )
            persisted = self.repository._read_supported_unlocked()
            projected = json.loads(json.dumps(persisted))
            registry_exists = self.repository.path.is_file()
            for journal_path in journals:
                journal = json.loads(journal_path.read_text())
                base_registry = self._assert_image(journal, "base_registry")
                if (
                    not registry_exists
                    and self._is_empty_registry_image(base_registry)
                    and self._is_empty_registry_image(projected)
                ):
                    projected = json.loads(json.dumps(base_registry))
                projected = self._project_unlocked(journal_path, projected)
                registry_exists = True
            migration_required = projected.get("registry_version") != ALIAS_REGISTRY_VERSION
            if migration_required:
                projected = self.repository._migrate_image(projected)
            yield AliasRegistryProjection(
                persisted=json.loads(json.dumps(persisted)),
                projected=json.loads(json.dumps(projected)),
                pending_journal_count=len(journals),
                recovery_required=bool(journals),
                migration_required=migration_required,
                journal_paths=tuple(journals),
            )

    def recover_job(self, job_id: str) -> None:
        try:
            with self.repository.lock(exclusive=True):
                self._recover_all_unlocked()
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias operation recovery failed") from error

    def recover_all(self) -> None:
        try:
            with self.repository.lock(exclusive=True):
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
        self.repository._migrate_unlocked()

    def registry_snapshot(self) -> dict[str, Any]:
        try:
            with self.repository.lock(exclusive=True):
                self._recover_all_unlocked()
                return json.loads(json.dumps(self.repository._read_unlocked()))
        except AliasRegistryUnavailable:
            raise
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry is unavailable") from error

    def identity_snapshots(
        self,
        profile_repository: JsonProfileRepository,
    ) -> tuple[ProfileRegistrySnapshot, dict[str, Any], dict[str, dict[str, Any]]]:
        """Read one validated profile/alias identity view under the global lock order."""
        with profile_repository.locked_snapshot() as profile_snapshot:
            aliases = self.registry_snapshot()
            identities = active_hospital_identities(profile_snapshot)
            validate_combined_hospital_identities(identities, aliases)
            return (
                profile_snapshot.model_copy(deep=True),
                json.loads(json.dumps(aliases)),
                json.loads(json.dumps(identities)),
            )

    def inspect_identity_snapshots(
        self,
        profile_repository: JsonProfileRepository,
    ) -> tuple[
        ProfileRegistrySnapshot,
        AliasRegistryProjection,
        dict[str, dict[str, Any]],
    ]:
        """Validate the projected runtime identity view without recovery or migration."""
        try:
            with (
                profile_repository.lock(exclusive=False),
                self.repository.lock(exclusive=False),
                self._projection_unlocked() as projection,
            ):
                profiles = profile_repository._load_unlocked()
                identities = active_hospital_identities(profiles)
                validate_combined_hospital_identities(identities, projection.projected)
                return (
                    profiles.model_copy(deep=True),
                    projection,
                    json.loads(json.dumps(identities)),
                )
        except (AliasRegistryUnavailable, ProfileRegistryUnavailable):
            raise
        except (*PROFILE_STORAGE_ERRORS, ProfileRegistryFormatError) as error:
            raise ProfileRegistryUnavailable("profile registry is unavailable") from error
        except ALIAS_STORAGE_ERRORS as error:
            raise AliasRegistryUnavailable("alias registry inspection failed") from error

    def mutate_profile_registry(
        self,
        profile_repository: JsonProfileRepository,
        expected_profile_revision: int | None,
        mutation: Callable[[ProfileRegistrySnapshot], ProfileRegistrySnapshot],
    ) -> ProfileRegistrySnapshot:
        """Validate aliases and profiles together before committing a profile write."""
        with (
            profile_repository.lock(exclusive=True),
            self.repository.lock(exclusive=True),
        ):
            try:
                current = profile_repository._load_unlocked()
            except ProfileRegistryUnavailable:
                raise
            except (*PROFILE_STORAGE_ERRORS, ProfileRegistryFormatError) as error:
                raise ProfileRegistryUnavailable(
                    "profile registry is unavailable"
                ) from error
            if (
                expected_profile_revision is not None
                and current.revision != expected_profile_revision
            ):
                raise ProfileRegistryRevisionConflict(current.revision)
            updated = profile_repository._updated_snapshot_unlocked(
                current,
                mutation,
            )
            identities = active_hospital_identities(updated)
            with self._projection_unlocked(exclusive=True) as projection:
                validate_combined_hospital_identities(identities, projection.projected)
                for journal_path in projection.journal_paths:
                    self._storage(
                        lambda path=journal_path: self._recover_unlocked(path.parent.name),
                        "alias registry recovery failed",
                    )
                self._storage(self.repository._migrate_unlocked, "alias registry migration failed")
            aliases = self._storage(self.repository._read_unlocked, "alias registry read failed")
            validate_combined_hospital_identities(identities, aliases)
            profile_repository._write_unlocked(updated)
            return updated.model_copy(deep=True)

    def mutate_registry(
        self,
        expected_revision: int,
        mutation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Recover, compare, and mutate without releasing the registry lock."""
        with self.repository.lock(exclusive=True):
            self._storage(
                self._recover_all_unlocked,
                "alias registry recovery failed",
            )
            current = self._storage(
                self.repository._read_unlocked,
                "alias registry read failed",
            )
            if current["revision"] != expected_revision:
                raise AliasRegistryRevisionConflict(current["revision"])
            updated = mutation(json.loads(json.dumps(current)))
            updated["revision"] = current["revision"] + 1
            self.repository._validate(updated)
            self._storage(
                lambda: self.repository._write_unlocked(updated),
                "alias registry write failed",
            )
            return updated

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
        with self.repository.lock(exclusive=True):
            self._storage(
                self._recover_all_unlocked,
                "alias registry recovery failed",
            )
            with self.store.job_lock(job_id, exclusive=True):
                self.store._require_stable_workspace(job_id)
                registry = self._storage(
                    self.repository._read_unlocked,
                    "alias registry read failed",
                )
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

                def persist_cutover() -> None:
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

                self._storage(persist_cutover, "alias registry mutation failed")
                return updated_review, updated_registry

    def mutate_review_and_registry_with_profiles(
        self,
        job_id: str,
        expected_review_revision: int,
        expected_registry_revision: int,
        profile_repository: JsonProfileRepository,
        expected_profile_revision: int,
        mutation: Callable[
            [dict[str, Any], dict[str, Any], dict[str, Any], Any],
            tuple[dict[str, Any], dict[str, Any]],
        ],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Hold the profile snapshot stable through alias/review cutover."""
        with profile_repository.locked_snapshot() as profile_snapshot:
            if profile_snapshot.revision != expected_profile_revision:
                raise ProfileRegistryRevisionConflict(profile_snapshot.revision)

            def coordinated_mutation(
                review: dict[str, Any],
                registry: dict[str, Any],
                result: dict[str, Any],
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                return mutation(review, registry, result, profile_snapshot)

            return self.mutate_review_and_registry(
                job_id,
                expected_review_revision,
                expected_registry_revision,
                coordinated_mutation,
            )

    def delete_job(self, job_id: str) -> None:
        try:
            with self.repository.lock(exclusive=True):
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
            with self.repository.lock(exclusive=True):
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
