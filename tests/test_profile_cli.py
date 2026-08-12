from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import LayoutProfile, ProfileLifecycle
from gmoney.profiles.cli import app


def active_profile() -> LayoutProfile:
    return LayoutProfile(
        contract_version="layout_profile_v1",
        profile_key="admin-profile",
        profile_version=1,
        lifecycle=ProfileLifecycle.ACTIVE,
        hospital_id="hospital-1",
        hospital_name="Admin Hospital",
        page_type=PageType.ITEMIZED_CHARGES,
        table_type=TableType.ITEM_LEDGER,
        page_aspect_ratio=0.7,
        table_box=(0.05, 0.15, 0.95, 0.9),
        supported_fields=("description", "amount"),
        construction_dataset_ids=("training",),
    )


def admin_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "GMONEY_PROFILE_REGISTRY": str(tmp_path / "profiles" / "registry.json"),
        "GMONEY_PROFILE_REGISTRY_LOCK": str(tmp_path / "config" / "profiles.lock"),
        "GMONEY_ALIAS_REGISTRY": str(tmp_path / "config" / "aliases.json"),
        "GMONEY_JOBS_ROOT": str(tmp_path / "jobs"),
    }


def test_admin_cli_add_validate_and_access_check_share_runtime_paths(
    tmp_path: Path,
) -> None:
    profile_file = tmp_path / "new-profile.json"
    profile_file.write_text(active_profile().model_dump_json())
    environment = admin_environment(tmp_path)
    runner = CliRunner()

    added = runner.invoke(
        app,
        ["add", "--profile", str(profile_file)],
        env=environment,
    )
    validated = runner.invoke(app, ["validate"], env=environment)
    checked = runner.invoke(app, ["check-access"], env=environment)

    assert added.exit_code == 0, added.output
    assert "added admin-profile@1" in added.output
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output) == {
        "active_hospital_count": 1,
        "alias_registry_revision": 0,
        "profile_revision": 1,
        "status": "valid",
    }
    assert checked.exit_code == 0, checked.output
    assert "paths and locks are writable" in checked.output
    assert not list(tmp_path.rglob(".gmoney-access-*"))
