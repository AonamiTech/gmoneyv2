from __future__ import annotations

import json
import multiprocessing
import threading
import time
from collections.abc import Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from gmoney.demo import worker as worker_module
from gmoney.demo.store import JobStore, JobTransactionError
from gmoney.extraction.offline import ExtractionAborted
from gmoney.profiles.aliases import AliasRegistryUnavailable


@pytest.fixture(autouse=True)
def reset_worker_extractor() -> Iterator[None]:
    worker_module._extractor = None
    worker_module._gpu_inference_lock = None
    yield
    worker_module._extractor = None
    gpu_lock = worker_module._gpu_inference_lock
    if gpu_lock is not None:
        gpu_lock.close()
    worker_module._gpu_inference_lock = None


def _report_inference_lock_entry(root_value: str, sender: Connection) -> None:
    store = JobStore(Path(root_value))
    sender.send(("ready", str(store.inference_lock_path)))
    with store.inference_lock():
        sender.send(("entered", str(store.inference_lock_path)))
    sender.close()


def _hold_inference_lock(root_value: str, sender: Connection) -> None:
    store = JobStore(Path(root_value))
    with store.inference_lock():
        sender.send("entered")
        while True:
            time.sleep(1)


def _run_gpu_job_then_wait(
    root_value: str,
    job_id: str,
    sender: Connection,
) -> None:
    worker_module._run_job(root_value, job_id, "http://vl.test")
    sender.send("task_returned")
    while True:
        time.sleep(1)


def _run_failing_gpu_job_then_wait(
    root_value: str,
    job_id: str,
    sender: Connection,
) -> None:
    try:
        worker_module._run_job(root_value, job_id, "http://vl.test")
    except RuntimeError as error:
        sender.send(("task_failed", str(error)))
    else:
        sender.send(("task_succeeded", None))
    while True:
        time.sleep(1)


def _assert_process_can_enter(root: Path) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_report_inference_lock_entry,
        args=(str(root), sender),
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(5)
        assert receiver.recv() == (
            "ready",
            str(JobStore(root).jobs_root / ".gpu-inference.lock"),
        )
        assert receiver.poll(5)
        assert receiver.recv() == (
            "entered",
            str(JobStore(root).jobs_root / ".gpu-inference.lock"),
        )
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_inference_lock_blocks_a_second_process(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)

    with store.inference_lock():
        process = context.Process(
            target=_report_inference_lock_entry,
            args=(str(tmp_path), sender),
        )
        process.start()
        sender.close()
        try:
            assert receiver.poll(5)
            assert receiver.recv() == (
                "ready",
                str(store.jobs_root / ".gpu-inference.lock"),
            )
            assert not receiver.poll(0.25)
        except BaseException:
            process.terminate()
            process.join(timeout=5)
            raise

    try:
        assert receiver.poll(5)
        assert receiver.recv() == (
            "entered",
            str(store.jobs_root / ".gpu-inference.lock"),
        )
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_inference_lock_releases_after_exception(tmp_path: Path) -> None:
    store = JobStore(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="extract failed",
    ), store.inference_lock():
        raise RuntimeError("extract failed")

    _assert_process_can_enter(tmp_path)


def test_inference_lock_releases_when_process_exits(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_hold_inference_lock,
        args=(str(tmp_path), sender),
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(5)
        assert receiver.recv() == "entered"
    finally:
        receiver.close()
        process.terminate()
        process.join(timeout=5)
    assert not process.is_alive()

    _assert_process_can_enter(tmp_path)


def _create_worker_job(store: JobStore, source_name: str) -> str:
    state = store.create(source_name)
    (store.job_dir(state["id"]) / "source.pdf").write_bytes(b"%PDF-worker-test")
    store.update(state["id"], status="queued")
    return str(state["id"])


def test_gpu_job_constructs_extractor_inside_lock_and_publishes_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "GPU bill.pdf")
    result_path = store.job_dir(job_id) / "result.json"
    events: list[tuple[str, Any]] = []

    class TrackingLock:
        closed = False

        def close(self) -> None:
            self.closed = True

    tracking_lock = TrackingLock()

    def tracking_inference_lock(
        locked_store: JobStore,
        cancel_requested: Any = None,
    ) -> TrackingLock:
        events.append(("lock_acquired", locked_store.inference_lock_path))
        return tracking_lock

    class FakeOfflineExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            assert events == [("lock_acquired", store.inference_lock_path)]
            events.append(
                (
                    "constructed",
                    (vl_url, paddle_device, vl_device),
                )
            )

        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any = None,
        ) -> dict[str, Any]:
            assert events[-1][0] == "constructed"
            assert source == store.job_dir(job_id) / "source.pdf"
            assert artifact_root == store.job_dir(job_id) / "artifacts"
            events.append(("extracted", source))
            return {"rows": [], "hospital": None}

    monkeypatch.setattr(
        JobStore,
        "acquire_inference_lock",
        tracking_inference_lock,
        raising=False,
    )
    monkeypatch.setattr(
        "gmoney.extraction.offline.OfflineExtractor",
        FakeOfflineExtractor,
    )
    monkeypatch.setenv("GMONEY_PADDLE_DEVICE", "gpu:0")
    monkeypatch.setenv("GMONEY_VL_DEVICE", "cuda:0")

    assert worker_module._run_job(str(tmp_path), job_id, "http://vl.test") == {
        "row_count": 0,
        "hospital_name": None,
        "hospital_confidence": None,
    }

    assert events == [
        ("lock_acquired", store.inference_lock_path),
        ("constructed", ("http://vl.test", "gpu:0", "cuda:0")),
        ("extracted", store.job_dir(job_id) / "source.pdf"),
    ]
    assert worker_module._gpu_inference_lock is tracking_lock
    assert not tracking_lock.closed
    assert json.loads(result_path.read_text())["source_name"] == "GPU bill.pdf"


def test_gpu_job_keeps_lock_until_the_one_task_child_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "GPU lifetime bill.pdf")

    class FakeOfflineExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any = None,
        ) -> dict[str, Any]:
            return {"rows": [], "hospital": None}

    monkeypatch.setattr(
        "gmoney.extraction.offline.OfflineExtractor",
        FakeOfflineExtractor,
    )
    monkeypatch.setenv("GMONEY_PADDLE_DEVICE", "gpu:0")
    monkeypatch.setenv("GMONEY_VL_DEVICE", "cuda:0")

    holder_context = multiprocessing.get_context("fork")
    holder_receiver, holder_sender = holder_context.Pipe(duplex=False)
    holder = holder_context.Process(
        target=_run_gpu_job_then_wait,
        args=(str(tmp_path), job_id, holder_sender),
    )
    holder.start()
    holder_sender.close()
    contender = None
    contender_receiver = None
    try:
        assert holder_receiver.poll(5)
        assert holder_receiver.recv() == "task_returned"

        contender_context = multiprocessing.get_context("spawn")
        contender_receiver, contender_sender = contender_context.Pipe(duplex=False)
        contender = contender_context.Process(
            target=_report_inference_lock_entry,
            args=(str(tmp_path), contender_sender),
        )
        contender.start()
        contender_sender.close()
        assert contender_receiver.poll(5)
        assert contender_receiver.recv()[0] == "ready"
        assert not contender_receiver.poll(0.25)

        holder.terminate()
        holder.join(timeout=5)
        assert not holder.is_alive()

        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "entered",
            str(store.inference_lock_path),
        )
    finally:
        holder_receiver.close()
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if contender_receiver is not None:
            contender_receiver.close()
        if contender is not None:
            contender.join(timeout=5)
            if contender.is_alive():
                contender.terminate()
                contender.join(timeout=5)

    assert holder.exitcode is not None
    assert contender is not None
    assert contender.exitcode == 0


def test_failed_gpu_job_keeps_lock_until_the_one_task_child_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Failed GPU lifetime bill.pdf")

    class FailingOfflineExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            pass

        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any = None,
        ) -> dict[str, Any]:
            raise RuntimeError("gpu extraction failed")

    monkeypatch.setattr(
        "gmoney.extraction.offline.OfflineExtractor",
        FailingOfflineExtractor,
    )
    monkeypatch.setenv("GMONEY_PADDLE_DEVICE", "gpu:0")
    monkeypatch.setenv("GMONEY_VL_DEVICE", "cuda:0")

    holder_context = multiprocessing.get_context("fork")
    holder_receiver, holder_sender = holder_context.Pipe(duplex=False)
    holder = holder_context.Process(
        target=_run_failing_gpu_job_then_wait,
        args=(str(tmp_path), job_id, holder_sender),
    )
    holder.start()
    holder_sender.close()
    contender = None
    contender_receiver = None
    try:
        assert holder_receiver.poll(5)
        assert holder_receiver.recv() == (
            "task_failed",
            "gpu extraction failed",
        )

        contender_context = multiprocessing.get_context("spawn")
        contender_receiver, contender_sender = contender_context.Pipe(duplex=False)
        contender = contender_context.Process(
            target=_report_inference_lock_entry,
            args=(str(tmp_path), contender_sender),
        )
        contender.start()
        contender_sender.close()
        assert contender_receiver.poll(5)
        assert contender_receiver.recv()[0] == "ready"
        assert not contender_receiver.poll(0.25)

        holder.terminate()
        holder.join(timeout=5)
        assert not holder.is_alive()

        assert contender_receiver.poll(5)
        assert contender_receiver.recv() == (
            "entered",
            str(store.inference_lock_path),
        )
    finally:
        holder_receiver.close()
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if contender_receiver is not None:
            contender_receiver.close()
        if contender is not None:
            contender.join(timeout=5)
            if contender.is_alive():
                contender.terminate()
                contender.join(timeout=5)

    assert holder.exitcode is not None
    assert contender is not None
    assert contender.exitcode == 0


def test_cpu_jobs_reuse_one_extractor_without_inference_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    first_id = _create_worker_job(store, "First CPU bill.pdf")
    second_id = _create_worker_job(store, "Second CPU bill.pdf")
    constructions: list[tuple[str, str, str]] = []
    extractions: list[Path] = []

    class FakeOfflineExtractor:
        def __init__(
            self,
            vl_url: str,
            *,
            paddle_device: str,
            vl_device: str,
        ) -> None:
            constructions.append((vl_url, paddle_device, vl_device))

        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any = None,
        ) -> dict[str, Any]:
            extractions.append(source)
            return {"rows": [], "hospital": None}

    def fail_if_locked(store: JobStore) -> Iterator[None]:
        raise AssertionError(
            f"CPU extraction unexpectedly locked {store.inference_lock_path}"
        )

    monkeypatch.setattr(JobStore, "inference_lock", fail_if_locked)
    monkeypatch.setattr(
        "gmoney.extraction.offline.OfflineExtractor",
        FakeOfflineExtractor,
    )
    monkeypatch.setenv("GMONEY_PADDLE_DEVICE", "cpu")
    monkeypatch.setenv("GMONEY_VL_DEVICE", "cpu")

    worker_module._run_job(str(tmp_path), first_id, "http://vl.test")
    worker_module._run_job(str(tmp_path), second_id, "http://vl.test")

    assert constructions == [("http://vl.test", "cpu", "cpu")]
    assert extractions == [
        store.job_dir(first_id) / "source.pdf",
        store.job_dir(second_id) / "source.pdf",
    ]


def test_processing_abort_prevents_result_publication(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Abort me.pdf")
    started = threading.Event()
    resume = threading.Event()
    outcome: list[dict[str, Any] | None] = []

    class BlockingExtractor:
        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any,
        ) -> dict[str, Any]:
            started.set()
            assert resume.wait(timeout=2)
            if should_abort():
                raise ExtractionAborted
            return {"rows": [{"id": "too-late"}], "hospital": None}

    def run() -> None:
        outcome.append(
            worker_module._extract_and_publish(
                store=store,
                job_id=job_id,
                extractor=BlockingExtractor(),
            )
        )

    assert store.claim_queued(job_id) is not None
    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=2)
    store.request_abort(job_id)
    resume.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert outcome == [None]
    assert not (store.job_dir(job_id) / "result.json").exists()
    assert store.read(job_id)["status"] == "cancelling"
    assert store.finalize_abort(job_id)


def test_worker_recovery_deletes_interrupted_abort(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Interrupted abort.pdf")
    assert store.claim_queued(job_id) is not None
    store.request_abort(job_id)

    store.recover()

    assert not store.job_dir(job_id).exists()


def test_abort_interrupts_wait_for_gpu_inference_lock(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Waiting for GPU.pdf")
    assert store.claim_queued(job_id) is not None
    outcome: list[str] = []

    def wait_for_lock() -> None:
        try:
            store.acquire_inference_lock(lambda: store.abort_requested(job_id))
        except JobTransactionError as error:
            outcome.append(str(error))

    with store.inference_lock():
        thread = threading.Thread(target=wait_for_lock)
        thread.start()
        time.sleep(0.15)
        store.request_abort(job_id)
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert outcome == ["job_abort_requested"]


def test_gpu_pool_exits_each_child_after_one_task() -> None:
    assert worker_module._executor_options(2, "gpu:0") == {
        "max_workers": 2,
        "max_tasks_per_child": 1,
    }


def test_cpu_pool_keeps_reusable_children() -> None:
    assert worker_module._executor_options(2, "cpu") == {"max_workers": 2}


def test_worker_fails_closed_before_ocr_when_alias_registry_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Invalid alias registry.pdf")
    registry = tmp_path / "alias-registry.json"
    registry.write_text('{"registry_version":"unsupported"}')
    monkeypatch.setenv("GMONEY_ALIAS_REGISTRY", str(registry))

    with pytest.raises(AliasRegistryUnavailable):
        worker_module._run_job(str(tmp_path), job_id, "http://vl.test")

    assert store.read(job_id)["status"] == "processing"
    assert not (store.job_dir(job_id) / "result.json").exists()


def test_registry_probe_fails_all_queued_jobs_and_recovers_without_worker_exit(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    first_id = _create_worker_job(store, "First queued bill.pdf")
    second_id = _create_worker_job(store, "Second queued bill.pdf")
    registry = tmp_path / "alias-registry.json"
    registry.write_text('{"registry_version":"unsupported"}')
    coordinator = worker_module.AliasTransactionCoordinator(store, registry)

    assert worker_module._probe_alias_registry(coordinator, store) is False
    for job_id in (first_id, second_id):
        state = store.read(job_id)
        assert state["status"] == "failed"
        assert state["error"] == "alias_registry_unavailable"

    registry.unlink()
    recovered_id = _create_worker_job(store, "Bill after registry repair.pdf")
    assert worker_module._probe_alias_registry(coordinator, store) is True
    assert store.read(recovered_id)["status"] == "queued"


def test_cleanup_registry_failure_fails_queue_without_escaping(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Queued during cleanup failure.pdf")

    class BrokenCoordinator:
        def cleanup(self, retention_hours: int) -> int:
            assert retention_hours == 720
            raise AliasRegistryUnavailable("corrupt journal")

    assert worker_module._cleanup_jobs(store, BrokenCoordinator(), 720) is False
    state = store.read(job_id)
    assert state["status"] == "failed"
    assert state["error"] == "alias_registry_unavailable"
