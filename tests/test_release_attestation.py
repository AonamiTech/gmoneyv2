from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_release_attestation.py"
SPEC = importlib.util.spec_from_file_location("verify_release_attestation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_release_manifest = MODULE._release_manifest
_require_revision = MODULE._require_revision
_worker_heartbeat = MODULE._worker_heartbeat


def test_release_attestation_accepts_strict_manifest() -> None:
    revision = "a" * 40

    assert _release_manifest(
        json.dumps(
            {
                "manifest_version": "gmoney_release_v1",
                "revision": revision,
                "revision_required": True,
            }
        )
    )["revision"] == revision


@pytest.mark.parametrize("revision", ("unknown", "A" * 40, "short"))
def test_release_attestation_rejects_non_release_revisions(revision: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _require_revision(revision)


def test_release_attestation_requires_enforced_manifest() -> None:
    with pytest.raises(RuntimeError, match="invalid production release manifest"):
        _release_manifest(
            json.dumps(
                {
                    "manifest_version": "gmoney_release_v1",
                    "revision": "a" * 40,
                    "revision_required": False,
                }
            )
        )


def test_release_attestation_compares_labels_manifests_and_service_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40

    def run(command: list[str]) -> str:
        if "ps" in command:
            return f"container-{command[-1]}"
        if command[:2] == ["docker", "inspect"]:
            return json.dumps([{"Image": command[-1].replace("container", "image")}])
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps(
                [{"Config": {"Labels": {MODULE.RELEASE_LABEL: revision}}}]
            )
        if "exec" in command:
            return json.dumps(
                {
                    "manifest_version": "gmoney_release_v1",
                    "revision": revision,
                    "revision_required": True,
                }
            )
        raise AssertionError(command)

    def json_url(url: str, timeout: float) -> dict[str, object]:
        assert timeout == 2
        if url.endswith("/build.json"):
            return {"release_revision": revision}
        return {
            "status": "ready",
            "release_revision": revision,
            "worker_release_revision": revision,
            "worker_status_updated_at": "2026-08-13T00:00:00Z",
            "worker_status_max_age_seconds": 30,
            "worker_status_future_skew_seconds": 5,
            "profile_revision": 2,
            "alias_registry_revision": 3,
            "release_consistent": True,
        }

    monkeypatch.setattr(MODULE, "_run", run)
    monkeypatch.setattr(MODULE, "_json_url", json_url)

    record = MODULE.verify(
        SimpleNamespace(
            compose_file=[Path("compose.demo.yaml"), Path("compose.gpu.yaml")],
            expected_revision=revision,
            base_url="http://demo.test/",
            timeout=2,
        ),
        now=datetime(2026, 8, 13, tzinfo=UTC),
    )

    assert record["expected_revision"] == revision
    assert record["reported_revisions"] == {
        "api": revision,
        "frontend": revision,
        "worker": revision,
    }
    assert set(record["services"]) == {"api", "frontend", "worker"}
    assert record["worker_status_age_seconds"] == 0
    assert record["worker_status_max_age_seconds"] == 30
    assert record["worker_status_future_skew_seconds"] == 5


@pytest.mark.parametrize(
    ("timestamp", "message"),
    (
        ("2026-08-12T23:59:29Z", "stale"),
        ("2026-08-13T00:00:06Z", "future"),
        ("2026-08-13T00:00:00", "timezone"),
        ("not-a-time", "invalid"),
    ),
)
def test_worker_heartbeat_rejects_unattestable_timestamps(
    timestamp: str,
    message: str,
) -> None:
    readiness = {
        "worker_status_updated_at": timestamp,
        "worker_status_max_age_seconds": 30,
        "worker_status_future_skew_seconds": 5,
    }

    with pytest.raises(RuntimeError, match=message):
        _worker_heartbeat(readiness, now=datetime(2026, 8, 13, tzinfo=UTC))


@pytest.mark.parametrize(
    "updated_at",
    (
        datetime(2026, 8, 12, 23, 59, 30, tzinfo=UTC),
        datetime(2026, 8, 13, 0, 0, 5, tzinfo=UTC),
    ),
)
def test_worker_heartbeat_accepts_exact_age_boundaries(updated_at: datetime) -> None:
    heartbeat = _worker_heartbeat(
        {
            "worker_status_updated_at": updated_at.isoformat(),
            "worker_status_max_age_seconds": 30,
            "worker_status_future_skew_seconds": 5,
        },
        now=datetime(2026, 8, 13, tzinfo=UTC),
    )

    assert -5 <= heartbeat["age_seconds"] <= 30


@pytest.mark.parametrize(
    ("max_age", "future_skew"),
    ((True, 5), (0, 5), (30, False), (30, -1)),
)
def test_worker_heartbeat_rejects_invalid_limits(
    max_age: object,
    future_skew: object,
) -> None:
    with pytest.raises(RuntimeError, match="heartbeat"):
        _worker_heartbeat(
            {
                "worker_status_updated_at": datetime.now(UTC).isoformat(),
                "worker_status_max_age_seconds": max_age,
                "worker_status_future_skew_seconds": future_skew,
            }
        )
