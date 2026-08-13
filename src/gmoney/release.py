from __future__ import annotations

import os
import re

UNKNOWN_BUILD_REVISION = "unknown"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def build_revision(*, required: bool | None = None) -> str:
    revision = os.environ.get("GMONEY_BUILD_REVISION", UNKNOWN_BUILD_REVISION)
    if required is None:
        required = os.environ.get("GMONEY_REQUIRE_BUILD_REVISION", "0") == "1"
    if revision != UNKNOWN_BUILD_REVISION and not _COMMIT_PATTERN.fullmatch(revision):
        raise RuntimeError("GMONEY_BUILD_REVISION must be a full 40-character Git SHA")
    if required and revision == UNKNOWN_BUILD_REVISION:
        raise RuntimeError("GMONEY_BUILD_REVISION is required in production")
    return revision
