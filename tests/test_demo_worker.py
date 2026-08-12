from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import LayoutProfile, ProfileLifecycle
from gmoney.demo import worker as worker_module
from gmoney.demo.store import JobStore, JobTransactionError
from gmoney.extraction.offline import ExtractionAborted
from gmoney.profiles.aliases import (
    AliasRegistryUnavailable,
    JsonAliasRepository,
    empty_alias_registry,
)
from gmoney.profiles.repository import (
    JsonProfileRepository,
    combined_hospital_name_owners,
)


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


def _controlled_worker_job(
    root_value: str,
    job_id: str,
    vl_url: str,
) -> dict[str, Any]:
    assert vl_url == "http://vl.test"
    store = JobStore(Path(root_value))
    directory = store.job_dir(job_id)
    (directory / ".test-runner-started").touch()
    hold = directory / ".test-hold"
    while hold.exists():
        time.sleep(0.02)
    return {"row_count": 1, "hospital_name": "Test Hospital"}


def _run_controlled_worker_loop(
    root_value: str,
    registry_value: str,
    stop_event: Any,
) -> None:
    worker_module.run_worker_loop(
        root=Path(root_value),
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=Path(registry_value),
        stop_requested=stop_event.is_set,
        executor_factory=ThreadPoolExecutor,
        job_runner=_controlled_worker_job,
    )


def _run_signal_controlled_worker_loop(root_value: str, registry_value: str) -> None:
    stop_event = worker_module.shutdown_event()
    _run_controlled_worker_loop(root_value, registry_value, stop_event)


def _run_registry_retry_timing_worker(
    root_value: str,
    registry_value: str,
    stop_event: Any,
    sender: Connection,
) -> None:
    class UnavailableCoordinator:
        def registry_snapshot(self) -> None:
            sender.send(time.monotonic())
            raise AliasRegistryUnavailable("registry unavailable")

    def coordinator_factory(store: JobStore, path: Path) -> UnavailableCoordinator:
        return UnavailableCoordinator()

    worker_module.run_worker_loop(
        root=Path(root_value),
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=Path(registry_value),
        stop_requested=stop_event.is_set,
        executor_factory=ThreadPoolExecutor,
        coordinator_factory=coordinator_factory,
    )


def _wait_until(predicate: Any, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for worker state")


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


def test_worker_uses_current_profile_name_to_resolve_existing_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Renamed hospital bill.pdf")
    profile_path = tmp_path / "profiles.json"
    JsonProfileRepository(profile_path).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="renamed-hospital-items",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="hospital-1",
            hospital_name="New Machine Medical Center",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    alias_path = tmp_path / "aliases.json"
    aliases = empty_alias_registry()
    aliases.update(
        revision=1,
        hospitals=[
            {
                "hospital_id": "hospital-1",
                "hospital_name": "Old Machine Hospital",
                "origins": ["profile", "reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Old Machine Hospital",
                        "normalized_name": "old machine hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "test-reviewer",
                        "reason": "Verified historical hospital name",
                        "created_at": "2026-08-12T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        aliases=[
            {
                "alias_id": "alias-1",
                "hospital_id": "hospital-1",
                "source_label": "Amount Rs",
                "normalized_label": "amount rs",
                "canonical_field": "net_amount",
                "active": True,
                "reason": "Verified printed amount header",
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
        events=[
            {
                "action": "column_alias_applied",
                "reviewer": "test-reviewer",
                "reason": "Verified printed amount header",
                "created_at": "2026-08-12T00:00:00Z",
            }
        ],
    )
    JsonAliasRepository(alias_path)._write_unlocked(aliases)

    class FakeOfflineExtractor:
        def __init__(self, vl_url: str, **options: Any) -> None:
            assert vl_url == "http://vl.test"

        def extract(
            self,
            source: Path,
            artifact_root: Path,
            progress: Any,
            *,
            should_abort: Any,
            alias_snapshot: dict[str, Any],
            profile_identities: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
            assert combined_hospital_name_owners(
                profile_identities,
                alias_snapshot,
                "New Machine Medical Center",
            ) == {"hospital-1"}
            assert JsonAliasRepository.active_aliases(
                alias_snapshot,
                "hospital-1",
            )[0]["normalized_label"] == "amount rs"
            return {"rows": [], "hospital": None}

    monkeypatch.setattr(
        "gmoney.extraction.offline.OfflineExtractor",
        FakeOfflineExtractor,
    )
    monkeypatch.setenv("GMONEY_PROFILE_REGISTRY", str(profile_path))
    monkeypatch.setenv("GMONEY_ALIAS_REGISTRY", str(alias_path))

    summary = worker_module._run_job(str(tmp_path), job_id, "http://vl.test")

    assert summary is not None
    assert summary["row_count"] == 0


def test_worker_fails_queued_bill_on_combined_hospital_identity_conflict(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Ambiguous hospital bill.pdf")
    profile_path = tmp_path / "profiles.json"
    JsonProfileRepository(profile_path).add_profile(
        LayoutProfile(
            contract_version="layout_profile_v1",
            profile_key="conflicting-profile",
            profile_version=1,
            lifecycle=ProfileLifecycle.ACTIVE,
            hospital_id="hospital-2",
            hospital_name="Machine Hospital",
            page_type=PageType.ITEMIZED_CHARGES,
            table_type=TableType.ITEM_LEDGER,
            page_aspect_ratio=0.7,
            table_box=(0.05, 0.15, 0.95, 0.9),
            supported_fields=("description", "amount"),
            construction_dataset_ids=("training",),
        )
    )
    alias_path = tmp_path / "aliases.json"
    aliases = empty_alias_registry()
    aliases.update(
        revision=1,
        hospitals=[
            {
                "hospital_id": "hospital-1",
                "hospital_name": "Machine Hospital",
                "origins": ["reviewer_alias"],
                "name_variants": [
                    {
                        "display_name": "Machine Hospital",
                        "normalized_name": "machine hospital",
                        "verified": True,
                        "source_document_id": "d" * 64,
                        "reviewer": "test-reviewer",
                        "reason": "Verified hospital identity",
                        "created_at": "2026-08-12T00:00:00Z",
                    }
                ],
                "created_at": "2026-08-12T00:00:00Z",
                "updated_at": "2026-08-12T00:00:00Z",
            }
        ],
    )
    JsonAliasRepository(alias_path)._write_unlocked(aliases)
    current_time = 0.0
    runner_calls: list[str] = []

    def sleeper(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    worker_module.run_worker_loop(
        root=tmp_path,
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=alias_path,
        profile_registry=profile_path,
        stop_requested=lambda: current_time > 0.5,
        executor_factory=ThreadPoolExecutor,
        job_runner=lambda root, claimed, url: runner_calls.append(claimed),
        clock=lambda: current_time,
        sleeper=sleeper,
    )

    assert runner_calls == []
    assert store.read(job_id)["status"] == "failed"
    assert store.read(job_id)["error"] == "hospital_identity_conflict"


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


@pytest.mark.parametrize("operation", ("probe", "cleanup"))
def test_registry_failure_during_shutdown_preserves_queued_jobs(
    tmp_path: Path,
    operation: str,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, f"Queued during {operation} shutdown.pdf")
    stop_event = threading.Event()

    class StoppingCoordinator:
        def registry_snapshot(self) -> None:
            stop_event.set()
            raise AliasRegistryUnavailable("registry failed during shutdown")

        def cleanup(self, retention_hours: int) -> int:
            stop_event.set()
            raise AliasRegistryUnavailable("cleanup failed during shutdown")

    coordinator = StoppingCoordinator()
    if operation == "probe":
        assert (
            worker_module._probe_alias_registry(
                coordinator,
                store,
                stop_event.is_set,
            )
            is False
        )
    else:
        assert (
            worker_module._cleanup_jobs(
                store,
                coordinator,
                720,
                stop_event.is_set,
            )
            is False
        )

    assert store.read(job_id)["status"] == "queued"


def test_shutdown_stops_alias_outage_failure_between_queued_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    first_id = _create_worker_job(store, "First outage bill.pdf")
    second_id = _create_worker_job(store, "Second outage bill.pdf")
    stop_event = threading.Event()
    original_fail = JobStore.fail_queued
    failures = 0

    def stopping_failure(
        job_store: JobStore,
        job_id: str,
        error: str,
        cancel_requested: Any = None,
    ) -> bool:
        nonlocal failures
        failed = original_fail(job_store, job_id, error, cancel_requested)
        failures += int(failed)
        if failed:
            stop_event.set()
        return failed

    monkeypatch.setattr(JobStore, "fail_queued", stopping_failure)

    assert (
        worker_module._fail_queued_for_alias_outage(store, stop_event.is_set) == 1
    )
    assert store.read(first_id)["status"] == "failed"
    assert store.read(second_id)["status"] == "queued"
    assert failures == 1


def test_shutdown_stops_alias_outage_failure_inside_job_lock(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Outage bill at shutdown boundary.pdf")
    checks = 0

    def stop_during_failure() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    assert worker_module._fail_queued_for_alias_outage(store, stop_during_failure) == 0
    assert store.read(job_id)["status"] == "queued"


def test_worker_loop_uses_five_second_registry_retry_interval(tmp_path: Path) -> None:
    observed_probes: list[float] = []
    current_time = 0.0

    class UnavailableCoordinator:
        def registry_snapshot(self) -> None:
            observed_probes.append(current_time)
            raise AliasRegistryUnavailable("registry unavailable")

    def coordinator_factory(store: JobStore, path: Path) -> UnavailableCoordinator:
        assert path == tmp_path / "alias-registry.json"
        return UnavailableCoordinator()

    def clock() -> float:
        return current_time

    def sleeper(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    worker_module.run_worker_loop(
        root=tmp_path,
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=tmp_path / "alias-registry.json",
        stop_requested=lambda: current_time > 5.0,
        executor_factory=ThreadPoolExecutor,
        coordinator_factory=coordinator_factory,
        clock=clock,
        sleeper=sleeper,
    )

    assert observed_probes == [0.0, 5.0]


def test_worker_recovers_profile_registry_after_five_seconds(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    outage_id = _create_worker_job(store, "Queued during profile outage.pdf")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text('{"registry_version":"unsupported"}')
    current_time = 0.0
    recovered_id: str | None = None

    def clock() -> float:
        return current_time

    def sleeper(seconds: float) -> None:
        nonlocal current_time, recovered_id
        current_time += seconds
        if current_time == 5.0:
            profile_path.unlink()
            recovered_id = _create_worker_job(store, "Queued after profile recovery.pdf")

    worker_module.run_worker_loop(
        root=tmp_path,
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=None,
        profile_registry=profile_path,
        stop_requested=lambda: current_time > 6.0,
        executor_factory=ThreadPoolExecutor,
        job_runner=lambda *args: {"row_count": 3},
        clock=clock,
        sleeper=sleeper,
    )

    assert store.read(outage_id)["status"] == "failed"
    assert store.read(outage_id)["error"] == "profile_registry_unavailable"
    assert recovered_id is not None
    assert store.read(recovered_id)["status"] == "complete"


def test_worker_does_not_claim_when_probe_observes_shutdown(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Queued before shutdown probe.pdf")
    stop_event = threading.Event()
    runner_calls: list[str] = []

    class StoppingCoordinator:
        def registry_snapshot(self) -> None:
            stop_event.set()
            raise AliasRegistryUnavailable("registry failed during shutdown")

    def coordinator_factory(store: JobStore, path: Path) -> StoppingCoordinator:
        return StoppingCoordinator()

    def runner(root_value: str, claimed_id: str, vl_url: str) -> dict[str, Any]:
        runner_calls.append(claimed_id)
        return {"row_count": 0}

    worker_module.run_worker_loop(
        root=tmp_path,
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=tmp_path / "alias-registry.json",
        stop_requested=stop_event.is_set,
        executor_factory=ThreadPoolExecutor,
        job_runner=runner,
        coordinator_factory=coordinator_factory,
    )

    assert store.read(job_id)["status"] == "queued"
    assert runner_calls == []


def test_worker_requeues_claim_when_shutdown_arrives_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Claim interrupted by shutdown.pdf")
    stop_event = threading.Event()
    runner_calls: list[str] = []
    original_claim = JobStore.claim_queued

    def stopping_claim(
        claimed_store: JobStore,
        claimed_id: str,
        cancel_requested: Any = None,
    ) -> dict[str, Any] | None:
        claimed = original_claim(claimed_store, claimed_id, cancel_requested)
        if claimed is not None:
            stop_event.set()
        return claimed

    def runner(root_value: str, claimed_id: str, vl_url: str) -> dict[str, Any]:
        runner_calls.append(claimed_id)
        return {"row_count": 0}

    monkeypatch.setattr(JobStore, "claim_queued", stopping_claim)

    worker_module.run_worker_loop(
        root=tmp_path,
        vl_url="http://vl.test",
        paddle_device="cpu",
        concurrency=1,
        retention_hours=720,
        alias_registry=None,
        stop_requested=stop_event.is_set,
        executor_factory=ThreadPoolExecutor,
        job_runner=runner,
    )

    assert store.read(job_id)["status"] == "queued"
    assert runner_calls == []


def test_executor_submission_failure_requeues_claim_and_exits_worker(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id = _create_worker_job(store, "Executor rejected bill.pdf")

    class RejectingExecutor:
        def __init__(self, **options: Any) -> None:
            assert options["max_workers"] == 1

        def __enter__(self) -> RejectingExecutor:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def submit(self, *args: Any) -> None:
            raise RuntimeError("executor rejected submission")

    with pytest.raises(RuntimeError, match="executor rejected submission"):
        worker_module.run_worker_loop(
            root=tmp_path,
            vl_url="http://vl.test",
            paddle_device="cpu",
            concurrency=1,
            retention_hours=720,
            alias_registry=None,
            executor_factory=RejectingExecutor,
        )

    assert store.read(job_id)["status"] == "queued"


def test_submission_failure_drains_previously_submitted_jobs_before_exit(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    first_id = _create_worker_job(store, "Submitted before pool failure.pdf")
    second_id = _create_worker_job(store, "Rejected by pool.pdf")
    completed: list[str] = []

    def runner(root_value: str, job_id: str, vl_url: str) -> dict[str, Any]:
        completed.append(job_id)
        return {"row_count": 7, "hospital_name": "Machine Hospital"}

    class PartiallyRejectingExecutor:
        def __init__(self, **options: Any) -> None:
            assert options["max_workers"] == 2
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.submissions = 0

        def __enter__(self) -> PartiallyRejectingExecutor:
            return self

        def __exit__(self, *args: Any) -> None:
            self.pool.shutdown(wait=True)

        def submit(self, *args: Any):
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError("executor rejected second submission")
            return self.pool.submit(*args)

    with pytest.raises(RuntimeError, match="executor rejected second submission"):
        worker_module.run_worker_loop(
            root=tmp_path,
            vl_url="http://vl.test",
            paddle_device="cpu",
            concurrency=2,
            retention_hours=720,
            alias_registry=None,
            executor_factory=PartiallyRejectingExecutor,
            job_runner=runner,
        )

    assert completed == [first_id]
    assert store.read(first_id)["status"] == "complete"
    assert store.read(first_id)["row_count"] == 7
    assert store.read(second_id)["status"] == "queued"


def test_spawned_worker_retries_unavailable_registry_after_five_seconds(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_registry_retry_timing_worker,
        args=(
            str(tmp_path),
            str(tmp_path / "alias-registry.json"),
            stop_event,
            sender,
        ),
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(8)
        first = receiver.recv()
        assert receiver.poll(8)
        second = receiver.recv()
        assert 5.0 <= second - first < 6.0
    finally:
        stop_event.set()
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_sigterm_drains_running_job_without_claiming_new_work(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    registry = tmp_path / "alias-registry.json"
    running_id = _create_worker_job(store, "Running during shutdown.pdf")
    hold = store.job_dir(running_id) / ".test-hold"
    hold.touch()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_signal_controlled_worker_loop,
        args=(str(tmp_path), str(registry)),
    )
    process.start()
    try:
        _wait_until(
            lambda: (store.job_dir(running_id) / ".test-runner-started").exists()
            and store.read(running_id)["status"] == "processing",
            timeout=30,
        )
        os.kill(process.pid, signal.SIGTERM)
        time.sleep(0.75)
        assert process.is_alive()
        queued_id = _create_worker_job(store, "Queued during shutdown.pdf")
        time.sleep(0.75)
        assert store.read(queued_id)["status"] == "queued"

        hold.unlink()
        process.join(timeout=8)

        assert not process.is_alive()
        assert process.exitcode == 0
        assert store.read(running_id)["status"] == "complete"
        assert store.read(queued_id)["status"] == "queued"
    finally:
        hold.unlink(missing_ok=True)
        if process.is_alive():
            os.kill(process.pid, signal.SIGTERM)
            process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_worker_process_survives_registry_outage_and_recovers_later_job(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    registry = tmp_path / "alias-registry.json"
    first_id = _create_worker_job(store, "Already running.pdf")
    first_hold = store.job_dir(first_id) / ".test-hold"
    first_hold.touch()
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    process = context.Process(
        target=_run_controlled_worker_loop,
        args=(str(tmp_path), str(registry), stop_event),
    )
    process.start()
    try:
        _wait_until(
            lambda: (store.job_dir(first_id) / ".test-runner-started").exists()
            and store.read(first_id)["status"] == "processing",
            timeout=30,
        )

        registry.write_text('{"registry_version":"unsupported"}')
        outage_id = _create_worker_job(store, "Queued during outage.pdf")
        _wait_until(lambda: store.read(outage_id)["status"] == "failed")
        assert store.read(outage_id)["error"] == "alias_registry_unavailable"
        assert process.is_alive()

        first_hold.unlink()
        _wait_until(lambda: store.read(first_id)["status"] == "complete")
        assert process.is_alive()

        newly_queued_id = _create_worker_job(store, "Newly queued during outage.pdf")
        _wait_until(lambda: store.read(newly_queued_id)["status"] == "failed")
        assert store.read(newly_queued_id)["error"] == "alias_registry_unavailable"

        registry.unlink()
        time.sleep(worker_module.ALIAS_REGISTRY_RETRY_SECONDS + 0.75)
        recovered_id = _create_worker_job(store, "Processed after repair.pdf")
        _wait_until(lambda: store.read(recovered_id)["status"] == "complete")
        assert process.is_alive()
    finally:
        stop_event.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0
