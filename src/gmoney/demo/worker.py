from __future__ import annotations

import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from gmoney.demo.store import JobStore

_extractor: Any = None


def _run_job(root_value: str, job_id: str, vl_url: str) -> int:
    global _extractor
    from gmoney.extraction.offline import OfflineExtractor

    store = JobStore(Path(root_value))
    if _extractor is None:
        _extractor = OfflineExtractor(vl_url)
    directory = store.job_dir(job_id)

    def progress(page: int, pages: int) -> None:
        store.update(job_id, status="processing", page=page, pages=pages)

    result = _extractor.extract(directory / "source.pdf", directory / "artifacts", progress)
    result_path = directory / "result.json"
    temporary = directory / "result.tmp"
    import json

    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return len(result["rows"])


def main() -> None:
    root = Path(os.environ.get("GMONEY_DEMO_ROOT", "/tmp/gmoney-v2-demo"))
    vl_url = os.environ.get("GMONEY_VL_URL", "http://paddleocr-vl:8111")
    concurrency = int(os.environ.get("GMONEY_WORKER_CONCURRENCY", "3"))
    retention_hours = int(os.environ.get("GMONEY_RETENTION_HOURS", "6"))
    store = JobStore(root)
    store.recover()
    futures: dict[Future[int], str] = {}
    last_cleanup = 0.0
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        while True:
            for future, job_id in list(futures.items()):
                if not future.done():
                    continue
                try:
                    row_count = future.result()
                    store.update(job_id, status="complete", row_count=row_count, error=None)
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
