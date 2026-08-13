#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RELEASE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RELEASE_LABEL = "org.opencontainers.image.revision"
SERVICES = ("api", "frontend", "worker")


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _json_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return payload


def _release_manifest(payload: str) -> dict[str, Any]:
    manifest = json.loads(payload)
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != "gmoney_release_v1"
        or manifest.get("revision_required") is not True
        or not RELEASE_PATTERN.fullmatch(str(manifest.get("revision") or ""))
    ):
        raise RuntimeError("container has an invalid production release manifest")
    return manifest


def _require_revision(value: str) -> str:
    if not RELEASE_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a full lowercase 40-character Git SHA")
    return value


def _worker_heartbeat(
    readiness: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, float | int]:
    timestamp = readiness.get("worker_status_updated_at")
    max_age = readiness.get("worker_status_max_age_seconds")
    future_skew = readiness.get("worker_status_future_skew_seconds")
    if not isinstance(timestamp, str):
        raise RuntimeError("worker heartbeat timestamp is missing")
    if type(max_age) is not int or max_age <= 0:
        raise RuntimeError("worker heartbeat maximum age is invalid")
    if type(future_skew) is not int or future_skew < 0:
        raise RuntimeError("worker heartbeat future skew is invalid")
    try:
        updated_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("worker heartbeat timestamp is invalid") from error
    if updated_at.tzinfo is None:
        raise RuntimeError("worker heartbeat timestamp must include a timezone")
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise RuntimeError("attestation clock must include a timezone")
    age = (checked_at.astimezone(UTC) - updated_at.astimezone(UTC)).total_seconds()
    if age > max_age:
        raise RuntimeError("worker heartbeat is stale")
    if age < -future_skew:
        raise RuntimeError("worker heartbeat is too far in the future")
    return {
        "age_seconds": age,
        "max_age_seconds": max_age,
        "future_skew_seconds": future_skew,
    }


def verify(
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    compose = ["docker", "compose"]
    for path in args.compose_file:
        compose.extend(("-f", str(path)))

    services: dict[str, dict[str, Any]] = {}
    for service in SERVICES:
        container_id = _run([*compose, "ps", "-q", service])
        if not container_id:
            raise RuntimeError(f"{service} container is not running")
        container = json.loads(_run(["docker", "inspect", container_id]))[0]
        image_id = str(container["Image"])
        image = json.loads(_run(["docker", "image", "inspect", image_id]))[0]
        label_revision = str(image["Config"]["Labels"].get(RELEASE_LABEL) or "")
        manifest = _release_manifest(
            _run([*compose, "exec", "-T", service, "cat", "/etc/gmoney/release.json"])
        )
        if label_revision != args.expected_revision:
            raise RuntimeError(f"{service} image label does not match expected revision")
        if manifest["revision"] != label_revision:
            raise RuntimeError(f"{service} manifest does not match its image label")
        services[service] = {
            "container_id": container_id,
            "image_id": image_id,
            "label_revision": label_revision,
            "manifest_revision": manifest["revision"],
        }

    base_url = args.base_url.rstrip("/")
    readiness = _json_url(f"{base_url}/api/v2/health/ready", args.timeout)
    frontend = _json_url(f"{base_url}/build.json", args.timeout)
    reported = {
        "api": readiness.get("release_revision"),
        "worker": readiness.get("worker_release_revision"),
        "frontend": frontend.get("release_revision"),
    }
    if any(value != args.expected_revision for value in reported.values()):
        raise RuntimeError("service-reported revisions do not match the expected revision")
    if readiness.get("status") != "ready" or readiness.get("release_consistent") is not True:
        raise RuntimeError("release readiness is not consistent")
    heartbeat = _worker_heartbeat(readiness, now=now)

    return {
        "attestation_version": "gmoney_release_attestation_v1",
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "expected_revision": args.expected_revision,
        "services": services,
        "reported_revisions": reported,
        "worker_status_updated_at": readiness.get("worker_status_updated_at"),
        "worker_status_age_seconds": heartbeat["age_seconds"],
        "worker_status_max_age_seconds": heartbeat["max_age_seconds"],
        "worker_status_future_skew_seconds": heartbeat["future_skew_seconds"],
        "profile_revision": readiness.get("profile_revision"),
        "alias_registry_revision": readiness.get("alias_registry_revision"),
        "release_consistent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a running immutable GMoney release")
    parser.add_argument("--expected-revision", required=True, type=_require_revision)
    parser.add_argument(
        "--compose-file",
        action="append",
        type=Path,
        default=[],
        help="Compose file; repeat in merge order",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3100")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.compose_file:
        args.compose_file = [Path("compose.demo.yaml"), Path("compose.gpu.yaml")]
    record = verify(args)
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
