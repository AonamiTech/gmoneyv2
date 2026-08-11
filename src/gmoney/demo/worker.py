from __future__ import annotations

import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, TextIO

from gmoney.demo.store import JobStore, is_gpu_device

_extractor: Any = None
_gpu_inference_lock: TextIO | None = None


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
        result = extractor.extract(
            directory / "source.pdf",
            directory / "artifacts",
            progress,
            should_abort=should_abort,
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
    extractor_options: dict[str, Any] = {
        "paddle_device": paddle_device,
        "vl_device": vl_device,
    }
    if alias_registry is not None:
        extractor_options["alias_registry"] = alias_registry
    if is_gpu_device(paddle_device):
        _gpu_inference_lock = store.acquire_inference_lock(
            lambda: store.abort_requested(job_id)
        )
        return _extract_and_publish(
            store=store,
            job_id=job_id,
            extractor=OfflineExtractor(
                vl_url,
                **extractor_options,
            ),
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
    )


def main() -> None:
    root = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
    vl_url = os.environ.get("GMONEY_VL_URL", "http://paddleocr-vl:8111")
    paddle_device = os.environ.get("GMONEY_PADDLE_DEVICE", "cpu")
    concurrency = int(os.environ.get("GMONEY_WORKER_CONCURRENCY", "3"))
    retention_hours = int(os.environ.get("GMONEY_RETENTION_HOURS", "720"))
    store = JobStore(root)
    store.recover()
    futures: dict[Future[dict[str, Any] | None], str] = {}
    last_cleanup = 0.0
    with ProcessPoolExecutor(
        **_executor_options(concurrency, paddle_device)
    ) as executor:
        while True:
            for future, job_id in list(futures.items()):
                if not future.done():
                    continue
                try:
                    summary = future.result()
                    if summary is None or not store.finish_processing(job_id, **summary):
                        store.finalize_abort(job_id)
                except Exception as error:  # noqa: BLE001 - boundary records sanitized failure
                    try:
                        if not store.fail_processing(job_id, type(error).__name__):
                            store.finalize_abort(job_id)
                    except KeyError:
                        pass
                del futures[future]

            running = set(futures.values())
            for state in store.states():
                if state.get("status") == "cancelling" and state["id"] not in running:
                    store.finalize_abort(state["id"])

            available = concurrency - len(futures)
            for state in store.queued()[:available]:
                job_id = state["id"]
                try:
                    claimed = store.claim_queued(job_id)
                except KeyError:
                    continue
                if claimed is None:
                    continue
                future = executor.submit(_run_job, str(root), job_id, vl_url)
                futures[future] = job_id

            now = time.monotonic()
            if now - last_cleanup >= 300:
                store.cleanup(retention_hours)
                last_cleanup = now
            time.sleep(0.5)


if __name__ == "__main__":
    main()
