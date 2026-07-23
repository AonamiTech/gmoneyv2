from __future__ import annotations

import json
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
) -> dict[str, Any]:
    directory = store.job_dir(job_id)
    state = store.read(job_id)

    def progress(page: int, pages: int) -> None:
        store.update(job_id, status="processing", page=page, pages=pages)

    result = extractor.extract(
        directory / "source.pdf",
        directory / "artifacts",
        progress,
    )
    result["source_name"] = state["original_name"]
    result_path = directory / "result.json"
    temporary = directory / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    hospital = result.get("hospital") or {}
    return {
        "row_count": len(result["rows"]),
        "hospital_name": hospital.get("name"),
        "hospital_confidence": hospital.get("confidence"),
    }


def _run_job(root_value: str, job_id: str, vl_url: str) -> dict[str, Any]:
    global _extractor, _gpu_inference_lock
    from gmoney.extraction.offline import OfflineExtractor

    store = JobStore(Path(root_value))
    paddle_device = os.environ.get("GMONEY_PADDLE_DEVICE", "cpu")
    vl_device = os.environ.get("GMONEY_VL_DEVICE", "cpu")
    if is_gpu_device(paddle_device):
        _gpu_inference_lock = store.acquire_inference_lock()
        return _extract_and_publish(
            store=store,
            job_id=job_id,
            extractor=OfflineExtractor(
                vl_url,
                paddle_device=paddle_device,
                vl_device=vl_device,
            ),
        )

    if _extractor is None:
        _extractor = OfflineExtractor(
            vl_url,
            paddle_device=paddle_device,
            vl_device=vl_device,
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
    futures: dict[Future[dict[str, Any]], str] = {}
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
                    store.update(job_id, status="complete", error=None, **summary)
                except Exception as error:  # noqa: BLE001 - boundary records sanitized failure
                    store.update(job_id, status="failed", error=type(error).__name__)
                del futures[future]

            available = concurrency - len(futures)
            for state in store.queued()[:available]:
                job_id = state["id"]
                store.update(job_id, status="processing", page=0, error=None)
                future = executor.submit(_run_job, str(root), job_id, vl_url)
                futures[future] = job_id

            now = time.monotonic()
            if now - last_cleanup >= 300:
                store.cleanup(retention_hours)
                last_cleanup = now
            time.sleep(0.5)


if __name__ == "__main__":
    main()
