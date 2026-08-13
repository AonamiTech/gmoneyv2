from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmoney.release import RELEASE_MANIFEST_VERSION, build_revision, release_manifest


def test_build_revision_accepts_full_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = "1a" * 20
    monkeypatch.setenv("GMONEY_BUILD_REVISION", revision)

    assert build_revision(required=True) == revision


@pytest.mark.parametrize("revision", ("short", "g" * 40, "A" * 40, " ", ""))
def test_build_revision_rejects_invalid_values(
    revision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GMONEY_BUILD_REVISION", revision)

    with pytest.raises(RuntimeError, match="release revision is invalid"):
        build_revision(required=True)


def test_unknown_revision_is_allowed_only_for_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GMONEY_BUILD_REVISION", raising=False)

    assert build_revision(required=False) == "unknown"
    with pytest.raises(RuntimeError, match="required in production"):
        build_revision(required=True)


def write_manifest(path: Path, revision: str, required: object) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_version": RELEASE_MANIFEST_VERSION,
                "revision": revision,
                "revision_required": required,
            }
        )
    )


def test_baked_manifest_ignores_runtime_revision_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "release.json"
    baked_revision = "a" * 40
    write_manifest(path, baked_revision, True)
    monkeypatch.setenv("GMONEY_BUILD_REVISION", "b" * 40)
    monkeypatch.setenv("GMONEY_REQUIRE_BUILD_REVISION", "0")

    manifest = release_manifest(manifest_path=path)

    assert manifest.revision == baked_revision
    assert manifest.revision_required is True
    assert manifest.baked is True


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ("not-json", "unavailable"),
        ({"manifest_version": "other"}, "unsupported"),
        (
            {
                "manifest_version": RELEASE_MANIFEST_VERSION,
                "revision": "unknown",
                "revision_required": True,
            },
            "required in production",
        ),
        (
            {
                "manifest_version": RELEASE_MANIFEST_VERSION,
                "revision": "a" * 40,
                "revision_required": "true",
            },
            "policy is invalid",
        ),
    ),
)
def test_baked_manifest_fails_closed(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "release.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))

    with pytest.raises(RuntimeError, match=message):
        release_manifest(manifest_path=path)


def test_release_environment_fallback_requires_absent_baked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setenv("GMONEY_BUILD_REVISION", revision)
    path = tmp_path / "absent" / "release.json"

    manifest = release_manifest(manifest_path=path, required=True)

    assert manifest.revision == revision
    assert manifest.baked is False


@pytest.mark.parametrize("manifest_kind", ("missing", "directory", "symlink"))
def test_baked_root_requires_real_regular_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_kind: str,
) -> None:
    baked_root = tmp_path / "gmoney"
    baked_root.mkdir()
    path = baked_root / "release.json"
    if manifest_kind == "directory":
        path.mkdir()
    elif manifest_kind == "symlink":
        target = tmp_path / "valid-release.json"
        write_manifest(target, "a" * 40, True)
        path.symlink_to(target)
    monkeypatch.setenv("GMONEY_BUILD_REVISION", "b" * 40)

    with pytest.raises(RuntimeError, match="manifest"):
        release_manifest(manifest_path=path)


def test_invalid_baked_root_object_fails_closed(tmp_path: Path) -> None:
    baked_root = tmp_path / "gmoney"
    baked_root.write_text("not a directory")

    with pytest.raises(RuntimeError, match="location is invalid"):
        release_manifest(manifest_path=baked_root / "release.json")
