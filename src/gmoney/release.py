from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNKNOWN_BUILD_REVISION = "unknown"
RELEASE_MANIFEST_PATH = Path("/etc/gmoney/release.json")
RELEASE_MANIFEST_VERSION = "gmoney_release_v1"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReleaseManifest:
    revision: str
    revision_required: bool
    baked: bool


def _validated_manifest(payload: Any, *, baked: bool) -> ReleaseManifest:
    if not isinstance(payload, dict):
        raise RuntimeError("GMoney release manifest is not an object")
    if payload.get("manifest_version") != RELEASE_MANIFEST_VERSION:
        raise RuntimeError("GMoney release manifest version is unsupported")
    revision = payload.get("revision")
    required = payload.get("revision_required")
    if not isinstance(revision, str) or (
        revision != UNKNOWN_BUILD_REVISION and not _COMMIT_PATTERN.fullmatch(revision)
    ):
        raise RuntimeError("GMoney release revision is invalid")
    if type(required) is not bool:
        raise RuntimeError("GMoney release revision policy is invalid")
    if required and revision == UNKNOWN_BUILD_REVISION:
        raise RuntimeError("GMoney release revision is required in production")
    return ReleaseManifest(revision=revision, revision_required=required, baked=baked)


def release_manifest(
    *,
    manifest_path: Path | None = None,
    required: bool | None = None,
) -> ReleaseManifest:
    path = manifest_path or RELEASE_MANIFEST_PATH
    baked_root = path.parent
    try:
        root_status = baked_root.lstat()
    except FileNotFoundError:
        root_status = None
    except OSError as error:
        raise RuntimeError("GMoney baked release location is unavailable") from error

    if root_status is not None:
        if not stat.S_ISDIR(root_status.st_mode):
            raise RuntimeError("GMoney baked release location is invalid")
        try:
            manifest_status = path.lstat()
            if not stat.S_ISREG(manifest_status.st_mode):
                raise RuntimeError("GMoney release manifest is not a regular file")
            manifest = _validated_manifest(json.loads(path.read_text()), baked=True)
        except FileNotFoundError as error:
            raise RuntimeError("GMoney release manifest is missing") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GMoney release manifest is unavailable") from error
        if required is True and manifest.revision == UNKNOWN_BUILD_REVISION:
            raise RuntimeError("GMoney release revision is required in production")
        return manifest

    # Environment fallback is intentionally limited to source/development runs where
    # the entire baked-release location is absent.
    revision = os.environ.get("GMONEY_BUILD_REVISION", UNKNOWN_BUILD_REVISION)
    environment_required = os.environ.get("GMONEY_REQUIRE_BUILD_REVISION", "0")
    if environment_required not in {"0", "1"}:
        raise RuntimeError("GMONEY_REQUIRE_BUILD_REVISION must be 0 or 1")
    return _validated_manifest(
        {
            "manifest_version": RELEASE_MANIFEST_VERSION,
            "revision": revision,
            "revision_required": (
                required if required is not None else environment_required == "1"
            ),
        },
        baked=False,
    )


def build_revision(
    *,
    required: bool | None = None,
    manifest_path: Path | None = None,
) -> str:
    return release_manifest(
        manifest_path=manifest_path,
        required=required,
    ).revision
