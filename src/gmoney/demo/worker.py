from __future__ import annotations

import json
import logging
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from gmoney.contracts.phase3 import ProfileRegistrySnapshot
from gmoney.demo.alias_transactions import AliasTransactionCoordinator
from gmoney.demo.store import TERMINAL_STATUSES, JobStore, is_gpu_device
from gmoney.profiles.aliases import AliasRegistryUnavailable, durable_json_replace
from gmoney.profiles.repository import (
    HospitalIdentityConflict,
    JsonProfileRepository,
    ProfileRegistryUnavailable,
    active_hospital_identities,
)
from gmoney.release import build_revision

_extractor: Any = None
_gpu_inference_lock: TextIO | None = None
ALIAS_REGISTRY_RETRY_SECONDS = 5.0
WORKER_STATUS_INTERVAL_SECONDS = 10.0
logger = logging.getLogger(__name__)
SEMANTIC_VALIDATION_VERSION = "semantic_result_validation_v1"


def _validation_issue(error: ValueError) -> dict[str, Any]:
    message = re.sub(r"\s+", " ", str(error)).strip() or "semantic validation failed"
    stable_codes = (
        ("document identity changed", "document_identity_changed"),
        ("source hash changed", "source_hash_changed"),
        ("page count changed", "page_count_changed"),
        ("page inventory is incomplete", "page_inventory_incomplete"),
        ("page inventory has invalid", "page_inventory_invalid"),
        ("page artifact is missing", "page_artifact_missing"),
        ("page artifact hash changed", "page_artifact_hash_changed"),
        ("row lacks grounded description", "ungrounded_description"),
        ("informational row has an amount", "informational_row_has_amount"),
        ("informational row lacks typed evidence", "informational_row_untyped"),
        ("billable row lacks grounded amount", "ungrounded_billable_amount"),
        ("canonical rows require validated source tables", "source_tables_missing"),
        ("source tables failed grounding validation", "source_tables_ungrounded"),
        ("unlinked source row contains a mapped financial value", "unlinked_financial_source_row"),
        ("source row links unknown canonical row", "unknown_canonical_source_link"),
        ("canonical row has multiple source links", "duplicate_canonical_source_link"),
        ("has a printed value but is missing canonical value", "printed_value_missing_canonical"),
        ("has a canonical value but is missing printed value", "canonical_value_missing_printed"),
        ("evidence is not in its mapped source cell", "mapped_cell_evidence_mismatch"),
        ("value does not match its mapped source cell", "mapped_cell_value_mismatch"),
        (
            "service_date_raw lacks matching grounded source evidence",
            "service_date_evidence_mismatch",
        ),
        ("canonical OCR row lacks a source-table link", "canonical_source_link_missing"),
    )
    normalized = message.casefold()
    code = next(
        (candidate for marker, candidate in stable_codes if marker.casefold() in normalized),
        "semantic_validation_failed",
    )
    return {
        "code": code,
        "message": message,
        "severity": "error",
    }


def _semantic_validation(
    source: Path,
    artifact_root: Path,
    result: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    # Lightweight worker fakes and legacy imports do not emit the offline
    # extraction contract. Production OfflineExtractor results always do.
    if not result.get("output_version"):
        return "complete", {
            "validation_version": SEMANTIC_VALIDATION_VERSION,
            "status": "not_applicable",
            "issues": [],
        }
    from gmoney.demo.reprocess import validate_result

    try:
        validate_result(source, result, result, artifact_root)
    except ValueError as error:
        return "needs_review", {
            "validation_version": SEMANTIC_VALIDATION_VERSION,
            "status": "needs_review",
            "issues": [_validation_issue(error)],
        }
    return "complete", {
        "validation_version": SEMANTIC_VALIDATION_VERSION,
        "status": "passed",
        "issues": [],
    }


@dataclass(frozen=True)
class RuntimeIdentitySnapshot:
    aliases: dict[str, Any] | None
    profiles: ProfileRegistrySnapshot | None
    identities: dict[str, dict[str, Any]] | None


def _executor_options(
    concurrency: int,
    paddle_device: str,
) -> dict[str, int]:
    options = {"max_workers": concurrency}
    if is_gpu_device(paddle_device):
        options["max_tasks_per_child"] = 1
    return options


def _extract_and_publish(
    *,
    store: JobStore,
    job_id: str,
    extractor: Any,
    alias_snapshot: dict[str, Any] | None = None,
    profile_identities: dict[str, dict[str, Any]] | None = None,
    profile_snapshot: ProfileRegistrySnapshot | None = None,
) -> dict[str, Any] | None:
    directory = store.job_dir(job_id)
    state = store.read(job_id)

    def progress(page: int, pages: int) -> None:
        if not store.update_processing_progress(job_id, page, pages):
            raise ExtractionAborted

    def should_abort() -> bool:
        return store.abort_requested(job_id)

    from gmoney.extraction.offline import ExtractionAborted

    try:
        extraction_options: dict[str, Any] = {"should_abort": should_abort}
        if alias_snapshot is not None:
            extraction_options["alias_snapshot"] = alias_snapshot
        if profile_identities is not None:
            extraction_options["profile_identities"] = profile_identities
        if profile_snapshot is not None:
            extraction_options["profiles"] = profile_snapshot.profiles
            extraction_options["profile_registry_revision"] = profile_snapshot.revision
        result = extractor.extract(
            directory / "source.pdf",
            directory / "artifacts",
            progress,
            **extraction_options,
        )
    except ExtractionAborted:
        return None
    result["source_name"] = state["original_name"]
    result["worker_release_revision"] = build_revision()
    outcome, validation = _semantic_validation(
        directory / "source.pdf",
        directory / "artifacts",
        result,
    )
    result["semantic_validation"] = validation
    hospital = result.get("hospital") or {}
    summary = {
        "row_count": len(result["rows"]),
        "hospital_name": hospital.get("name"),
        "hospital_confidence": hospital.get("confidence"),
    }
    if validation["status"] != "not_applicable":
        summary.update(
            validation_status=validation["status"],
            validation_issue_count=len(validation["issues"]),
            validation_issue_codes=[
                issue["code"] for issue in validation["issues"]
            ],
        )
    if not store.publish_processing_outcome(
        job_id,
        result,
        status=outcome,
        **summary,
    ):
        return None
    return summary


def _run_job(
    root_value: str,
    job_id: str,
    vl_url: str,
    identity_snapshot: RuntimeIdentitySnapshot | None = None,
) -> dict[str, Any] | None:
    global _extractor, _gpu_inference_lock
    from gmoney.extraction.offline import OfflineExtractor

    store = JobStore(Path(root_value))
    state = store.read(job_id)
    if state.get("status") == "queued":
        if store.claim_queued(job_id) is None:
            return None
    elif state.get("status") != "processing":
        return None
    paddle_device = os.environ.get("GMONEY_PADDLE_DEVICE", "cpu")
    vl_device = os.environ.get("GMONEY_VL_DEVICE", "cpu")
    alias_registry_value = os.environ.get("GMONEY_ALIAS_REGISTRY")
    alias_registry = Path(alias_registry_value) if alias_registry_value else None
    profile_registry_value = os.environ.get("GMONEY_PROFILE_REGISTRY")
    profile_registry = Path(profile_registry_value) if profile_registry_value else None
    if identity_snapshot is None:
        identity_snapshot = _runtime_identity_snapshots(
            AliasTransactionCoordinator(store, alias_registry)
            if alias_registry is not None
            else None,
            JsonProfileRepository(profile_registry)
            if profile_registry is not None
            else None,
        )
    extractor_options: dict[str, Any] = {
        "paddle_device": paddle_device,
        "vl_device": vl_device,
    }
    if alias_registry is not None:
        extractor_options["alias_registry"] = alias_registry
    if is_gpu_device(paddle_device):
        _gpu_inference_lock = store.acquire_inference_lock(lambda: store.abort_requested(job_id))
        return _extract_and_publish(
            store=store,
            job_id=job_id,
            extractor=OfflineExtractor(
                vl_url,
                **extractor_options,
            ),
            alias_snapshot=identity_snapshot.aliases,
            profile_identities=identity_snapshot.identities,
            profile_snapshot=identity_snapshot.profiles,
        )

    if _extractor is None:
        _extractor = OfflineExtractor(
            vl_url,
            **extractor_options,
        )
    return _extract_and_publish(
        store=store,
        job_id=job_id,
        extractor=_extractor,
        alias_snapshot=identity_snapshot.aliases,
        profile_identities=identity_snapshot.identities,
        profile_snapshot=identity_snapshot.profiles,
    )


def _fail_queued_for_registry_outage(
    store: JobStore,
    error: str,
    stop_requested: Callable[[], bool] = lambda: False,
) -> int:
    failed = 0
    for state in store.queued():
        if stop_requested():
            break
        try:
            if store.fail_queued(
                str(state["id"]),
                error,
                stop_requested,
            ):
                failed += 1
        except KeyError:
            continue
    return failed


def _fail_queued_for_alias_outage(
    store: JobStore,
    stop_requested: Callable[[], bool] = lambda: False,
) -> int:
    return _fail_queued_for_registry_outage(
        store,
        "alias_registry_unavailable",
        stop_requested,
    )


def _fail_queued_for_profile_outage(
    store: JobStore,
    stop_requested: Callable[[], bool] = lambda: False,
) -> int:
    return _fail_queued_for_registry_outage(
        store,
        "profile_registry_unavailable",
        stop_requested,
    )


def _runtime_identity_snapshots(
    alias_coordinator: AliasTransactionCoordinator | None,
    profile_repository: JsonProfileRepository | None,
) -> RuntimeIdentitySnapshot:
    if profile_repository is not None and alias_coordinator is not None:
        profiles, aliases, identities = alias_coordinator.identity_snapshots(
            profile_repository
        )
        return RuntimeIdentitySnapshot(aliases, profiles, identities)
    if profile_repository is not None:
        with profile_repository.locked_snapshot() as profile_snapshot:
            identities = active_hospital_identities(profile_snapshot)
            return RuntimeIdentitySnapshot(
                None,
                profile_snapshot.model_copy(deep=True),
                identities,
            )
    aliases = (
        alias_coordinator.registry_snapshot()
        if alias_coordinator is not None
        else None
    )
    return RuntimeIdentitySnapshot(aliases, None, None)


def _probe_runtime_registries(
    alias_coordinator: AliasTransactionCoordinator | None,
    profile_repository: JsonProfileRepository | None,
    store: JobStore,
    stop_requested: Callable[[], bool] = lambda: False,
) -> tuple[str | None, RuntimeIdentitySnapshot | None]:
    try:
        snapshot = _runtime_identity_snapshots(alias_coordinator, profile_repository)
    except ProfileRegistryUnavailable:
        error = "profile_registry_unavailable"
    except AliasRegistryUnavailable:
        error = "alias_registry_unavailable"
    except HospitalIdentityConflict:
        error = "hospital_identity_conflict"
    else:
        return None, snapshot
    _fail_queued_for_registry_outage(store, error, stop_requested)
    return error, None


def _probe_alias_registry(
    coordinator: AliasTransactionCoordinator,
    store: JobStore,
    stop_requested: Callable[[], bool] = lambda: False,
) -> bool:
    try:
        coordinator.registry_snapshot()
    except AliasRegistryUnavailable:
        _fail_queued_for_alias_outage(store, stop_requested)
        return False
    return True


def _probe_profile_registry(
    repository: JsonProfileRepository,
    store: JobStore,
    stop_requested: Callable[[], bool] = lambda: False,
) -> bool:
    try:
        repository.snapshot()
    except ProfileRegistryUnavailable:
        _fail_queued_for_profile_outage(store, stop_requested)
        return False
    return True


def _cleanup_jobs(
    store: JobStore,
    coordinator: AliasTransactionCoordinator | None,
    retention_hours: int,
    stop_requested: Callable[[], bool] = lambda: False,
) -> bool:
    try:
        if coordinator is not None:
            coordinator.cleanup(retention_hours)
        else:
            store.cleanup(retention_hours)
    except AliasRegistryUnavailable:
        _fail_queued_for_alias_outage(store, stop_requested)
        return False
    return True


def run_worker_loop(
    *,
    root: Path,
    vl_url: str,
    paddle_device: str,
    concurrency: int,
    retention_hours: int,
    alias_registry: Path | None,
    profile_registry: Path | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
    job_runner: Callable[
        [str, str, str, RuntimeIdentitySnapshot],
        dict[str, Any] | None,
    ] = _run_job,
    coordinator_factory: Callable[[JobStore, Path], AliasTransactionCoordinator] = (
        AliasTransactionCoordinator
    ),
    alias_retry_seconds: float = ALIAS_REGISTRY_RETRY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    worker_status_path: Path | None = None,
    identity_captured: Callable[[RuntimeIdentitySnapshot], None] | None = None,
) -> None:
    store = JobStore(root)
    store.recover()
    alias_coordinator = (
        coordinator_factory(store, alias_registry) if alias_registry is not None else None
    )
    profile_repository = (
        JsonProfileRepository(profile_registry) if profile_registry is not None else None
    )
    futures: dict[Future[dict[str, Any] | None], str] = {}
    last_cleanup = 0.0
    runtime_registries_available = (
        alias_coordinator is None and profile_repository is None
    )
    registry_error: str | None = None
    next_registry_probe = 0.0
    draining = False
    fatal_error: Exception | None = None
    pending_identity_snapshot: RuntimeIdentitySnapshot | None = None
    release_revision = build_revision()
    worker_started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    last_worker_status = float("-inf")
    logger.info(
        json.dumps(
            {
                "event": "worker_started",
                "release_revision": release_revision,
                "paddle_device": paddle_device,
                "concurrency": concurrency,
            },
            sort_keys=True,
        )
    )

    def publish_worker_status(now: float, status: str = "running") -> None:
        nonlocal last_worker_status
        if worker_status_path is None:
            return
        durable_json_replace(
            worker_status_path,
            {
                "status_version": "worker_status_v1",
                "status": status,
                "release_revision": release_revision,
                "started_at": worker_started_at,
                "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "pid": os.getpid(),
                "paddle_device": paddle_device,
                "concurrency": concurrency,
            },
            suffix="worker-status.tmp",
        )
        last_worker_status = now

    with executor_factory(**_executor_options(concurrency, paddle_device)) as executor:
        while True:
            now = clock()
            if now - last_worker_status >= WORKER_STATUS_INTERVAL_SECONDS:
                publish_worker_status(now)
            if stop_requested():
                draining = True
            for future, job_id in list(futures.items()):
                if not future.done():
                    continue
                try:
                    summary = future.result()
                    current_status = store.read(job_id).get("status")
                    if summary is None or (
                        current_status not in TERMINAL_STATUSES
                        and not store.finish_processing(job_id, **summary)
                    ):
                        store.finalize_abort(job_id)
                except Exception as error:  # noqa: BLE001 - boundary records sanitized failure
                    try:
                        failure = (
                            "alias_registry_unavailable"
                            if isinstance(error, AliasRegistryUnavailable)
                            else "profile_registry_unavailable"
                            if isinstance(error, ProfileRegistryUnavailable)
                            else "hospital_identity_conflict"
                            if isinstance(error, HospitalIdentityConflict)
                            else type(error).__name__
                        )
                        if not store.fail_processing(job_id, failure):
                            store.finalize_abort(job_id)
                    except KeyError:
                        pass
                del futures[future]

            running = set(futures.values())
            for state in store.states():
                if state.get("status") == "cancelling" and state["id"] not in running:
                    store.finalize_abort(state["id"])

            if draining:
                if not futures:
                    if fatal_error is not None:
                        raise fatal_error
                    break
                sleeper(0.5)
                continue

            if alias_coordinator is not None or profile_repository is not None:
                cleanup_due = now - last_cleanup >= 300
                should_probe = (
                    not runtime_registries_available
                    or bool(store.queued())
                    or cleanup_due
                )
                if should_probe and now >= next_registry_probe:
                    registry_error, observed_snapshot = _probe_runtime_registries(
                        alias_coordinator,
                        profile_repository,
                        store,
                        stop_requested,
                    )
                    runtime_registries_available = registry_error is None
                    pending_identity_snapshot = (
                        observed_snapshot
                        if runtime_registries_available and store.queued()
                        else None
                    )
                    next_registry_probe = (
                        0.0
                        if runtime_registries_available
                        else now + alias_retry_seconds
                    )
                elif not runtime_registries_available and registry_error is not None:
                    _fail_queued_for_registry_outage(
                        store,
                        registry_error,
                        stop_requested,
                    )
            if stop_requested():
                draining = True
                continue

            if runtime_registries_available:
                available = concurrency - len(futures)
                for state in store.queued()[:available]:
                    if stop_requested():
                        draining = True
                        break
                    job_id = state["id"]
                    identity_snapshot = pending_identity_snapshot
                    pending_identity_snapshot = None
                    if identity_snapshot is None:
                        try:
                            identity_snapshot = _runtime_identity_snapshots(
                                alias_coordinator,
                                profile_repository,
                            )
                        except ProfileRegistryUnavailable:
                            registry_error = "profile_registry_unavailable"
                        except AliasRegistryUnavailable:
                            registry_error = "alias_registry_unavailable"
                        except HospitalIdentityConflict:
                            registry_error = "hospital_identity_conflict"
                        else:
                            registry_error = None
                        if registry_error is not None:
                            runtime_registries_available = False
                            next_registry_probe = now + alias_retry_seconds
                            _fail_queued_for_registry_outage(
                                store,
                                registry_error,
                                stop_requested,
                            )
                            break
                    if identity_captured is not None:
                        identity_captured(identity_snapshot)
                    try:
                        claimed = store.claim_queued(job_id, stop_requested)
                    except KeyError:
                        continue
                    if claimed is None:
                        if stop_requested():
                            draining = True
                            break
                        continue
                    if stop_requested():
                        store.requeue_claimed(job_id)
                        draining = True
                        break
                    try:
                        future = executor.submit(
                            job_runner,
                            str(root),
                            job_id,
                            vl_url,
                            identity_snapshot,
                        )
                    except Exception as error:
                        if (
                            not store.requeue_claimed(job_id)
                            and store.abort_requested(job_id)
                        ):
                            store.finalize_abort(job_id)
                        fatal_error = error
                        draining = True
                        break
                    if stop_requested():
                        draining = True
                        if future.cancel():
                            store.requeue_claimed(job_id)
                            break
                    futures[future] = job_id
                    logger.info(
                        json.dumps(
                            {
                                "event": "worker_job_submitted",
                                "job_id": job_id,
                                "release_revision": release_revision,
                                "profile_registry_revision": (
                                    identity_snapshot.profiles.revision
                                    if identity_snapshot.profiles is not None
                                    else None
                                ),
                                "alias_registry_revision": (
                                    identity_snapshot.aliases.get("revision")
                                    if identity_snapshot.aliases is not None
                                    else None
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                    if draining:
                        break

            if draining:
                continue

            if now - last_cleanup >= 300:
                cleanup_succeeded = True
                if runtime_registries_available:
                    cleanup_succeeded = _cleanup_jobs(
                        store,
                        alias_coordinator,
                        retention_hours,
                        stop_requested,
                    )
                last_cleanup = now
                if not cleanup_succeeded:
                    runtime_registries_available = False
                    registry_error = "alias_registry_unavailable"
                    next_registry_probe = now + alias_retry_seconds
            sleeper(0.5)
    publish_worker_status(clock(), "stopped")


def shutdown_event() -> threading.Event:
    event = threading.Event()

    def request_shutdown(signum: int, frame: Any) -> None:
        del signum, frame
        event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    return event


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_revision()
    alias_registry_value = os.environ.get("GMONEY_ALIAS_REGISTRY")
    profile_registry_value = os.environ.get("GMONEY_PROFILE_REGISTRY")
    stop_event = shutdown_event()
    run_worker_loop(
        root=Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo")),
        vl_url=os.environ.get("GMONEY_VL_URL", "http://paddleocr-vl:8111"),
        paddle_device=os.environ.get("GMONEY_PADDLE_DEVICE", "cpu"),
        concurrency=int(os.environ.get("GMONEY_WORKER_CONCURRENCY", "3")),
        retention_hours=int(os.environ.get("GMONEY_RETENTION_HOURS", "720")),
        alias_registry=(Path(alias_registry_value) if alias_registry_value else None),
        profile_registry=(
            Path(profile_registry_value) if profile_registry_value else None
        ),
        worker_status_path=(
            Path(value)
            if (value := os.environ.get("GMONEY_WORKER_STATUS_PATH"))
            else None
        ),
        stop_requested=stop_event.is_set,
    )


if __name__ == "__main__":
    main()
