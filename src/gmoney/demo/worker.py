from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, TextIO

from gmoney.demo.alias_transactions import AliasTransactionCoordinator
from gmoney.demo.store import JobStore, is_gpu_device
from gmoney.profiles.aliases import AliasRegistryUnavailable

_extractor: Any = None
_gpu_inference_lock: TextIO | None = None
ALIAS_REGISTRY_RETRY_SECONDS = 5.0


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
        result = extractor.extract(
            directory / "source.pdf",
            directory / "artifacts",
            progress,
            **extraction_options,
        )
    except ExtractionAborted:
        return None
    result["source_name"] = state["original_name"]
    if not store.publish_processing_result(job_id, result):
        return None
    hospital = result.get("hospital") or {}
    return {
        "row_count": len(result["rows"]),
        "hospital_name": hospital.get("name"),
        "hospital_confidence": hospital.get("confidence"),
    }


def _run_job(root_value: str, job_id: str, vl_url: str) -> dict[str, Any] | None:
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
    alias_snapshot = (
        AliasTransactionCoordinator(store, alias_registry).registry_snapshot()
        if alias_registry is not None
        else None
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
            alias_snapshot=alias_snapshot,
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
        alias_snapshot=alias_snapshot,
    )


def _fail_queued_for_alias_outage(store: JobStore) -> int:
    failed = 0
    for state in store.queued():
        try:
            if store.fail_queued(str(state["id"]), "alias_registry_unavailable"):
                failed += 1
        except KeyError:
            continue
    return failed


def _probe_alias_registry(
    coordinator: AliasTransactionCoordinator,
    store: JobStore,
) -> bool:
    try:
        coordinator.registry_snapshot()
    except AliasRegistryUnavailable:
        _fail_queued_for_alias_outage(store)
        return False
    return True


def _cleanup_jobs(
    store: JobStore,
    coordinator: AliasTransactionCoordinator | None,
    retention_hours: int,
) -> bool:
    try:
        if coordinator is not None:
            coordinator.cleanup(retention_hours)
        else:
            store.cleanup(retention_hours)
    except AliasRegistryUnavailable:
        _fail_queued_for_alias_outage(store)
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
    stop_requested: Callable[[], bool] = lambda: False,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
    job_runner: Callable[[str, str, str], dict[str, Any] | None] = _run_job,
    coordinator_factory: Callable[[JobStore, Path], AliasTransactionCoordinator] = (
        AliasTransactionCoordinator
    ),
    alias_retry_seconds: float = ALIAS_REGISTRY_RETRY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    store = JobStore(root)
    store.recover()
    alias_coordinator = (
        coordinator_factory(store, alias_registry) if alias_registry is not None else None
    )
    futures: dict[Future[dict[str, Any] | None], str] = {}
    last_cleanup = 0.0
    alias_registry_available = alias_coordinator is None
    next_alias_probe = 0.0
    with executor_factory(**_executor_options(concurrency, paddle_device)) as executor:
        while not stop_requested():
            for future, job_id in list(futures.items()):
                if not future.done():
                    continue
                try:
                    summary = future.result()
                    if summary is None or not store.finish_processing(job_id, **summary):
                        store.finalize_abort(job_id)
                except Exception as error:  # noqa: BLE001 - boundary records sanitized failure
                    try:
                        failure = (
                            "alias_registry_unavailable"
                            if isinstance(error, AliasRegistryUnavailable)
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

            now = clock()
            if alias_coordinator is not None:
                cleanup_due = now - last_cleanup >= 300
                should_probe = (
                    not alias_registry_available
                    or bool(store.queued())
                    or cleanup_due
                )
                if should_probe and now >= next_alias_probe:
                    alias_registry_available = _probe_alias_registry(
                        alias_coordinator,
                        store,
                    )
                    next_alias_probe = (
                        0.0
                        if alias_registry_available
                        else now + alias_retry_seconds
                    )
                elif not alias_registry_available:
                    _fail_queued_for_alias_outage(store)

            if alias_registry_available:
                available = concurrency - len(futures)
                for state in store.queued()[:available]:
                    job_id = state["id"]
                    try:
                        claimed = store.claim_queued(job_id)
                    except KeyError:
                        continue
                    if claimed is None:
                        continue
                    future = executor.submit(job_runner, str(root), job_id, vl_url)
                    futures[future] = job_id

            if now - last_cleanup >= 300:
                cleanup_succeeded = True
                if alias_coordinator is None or alias_registry_available:
                    cleanup_succeeded = _cleanup_jobs(
                        store,
                        alias_coordinator,
                        retention_hours,
                    )
                last_cleanup = now
                if not cleanup_succeeded:
                    alias_registry_available = False
                    next_alias_probe = now + alias_retry_seconds
            sleeper(0.5)


def main() -> None:
    alias_registry_value = os.environ.get("GMONEY_ALIAS_REGISTRY")
    run_worker_loop(
        root=Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo")),
        vl_url=os.environ.get("GMONEY_VL_URL", "http://paddleocr-vl:8111"),
        paddle_device=os.environ.get("GMONEY_PADDLE_DEVICE", "cpu"),
        concurrency=int(os.environ.get("GMONEY_WORKER_CONCURRENCY", "3")),
        retention_hours=int(os.environ.get("GMONEY_RETENTION_HOURS", "720")),
        alias_registry=(Path(alias_registry_value) if alias_registry_value else None),
    )


if __name__ == "__main__":
    main()
