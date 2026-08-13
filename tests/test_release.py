from __future__ import annotations

import pytest

from gmoney.release import build_revision


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

    with pytest.raises(RuntimeError, match="GMONEY_BUILD_REVISION"):
        build_revision(required=True)


def test_unknown_revision_is_allowed_only_for_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GMONEY_BUILD_REVISION", raising=False)

    assert build_revision(required=False) == "unknown"
    with pytest.raises(RuntimeError, match="required in production"):
        build_revision(required=True)
