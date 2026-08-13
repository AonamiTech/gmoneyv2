from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gmoney.contracts.extraction import PageType, TableType
from gmoney.contracts.phase3 import LayoutProfile, ProfileLifecycle
from gmoney.demo.alias_transactions import ALIAS_JOURNAL_VERSION, _digest
from gmoney.demo.store import JobStore
from gmoney.profiles.aliases import LEGACY_ALIAS_REGISTRY_VERSION, empty_alias_registry
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
        "alias_registry_version": "hospital_alias_registry_v3",
        "migration_required": False,
        "pending_journal_count": 0,
        "profile_revision": 1,
        "projected_alias_registry_revision": 0,
        "projected_alias_registry_version": "hospital_alias_registry_v3",
        "recovery_required": False,
        "status": "valid",
    }
    assert checked.exit_code == 0, checked.output
    assert "paths and locks are writable" in checked.output
    assert not list(tmp_path.rglob(".gmoney-access-*"))


def test_validate_and_check_access_do_not_recover_or_migrate(
    tmp_path: Path,
) -> None:
    environment = admin_environment(tmp_path)
    store = JobStore(tmp_path)
    state = store.create("pending.pdf")
    job_id = str(state["id"])
    base_review = store.empty_review()
    target_review = json.loads(json.dumps(base_review))
    target_review["revision"] = 1
    base_registry = empty_alias_registry()
    base_registry["registry_version"] = LEGACY_ALIAS_REGISTRY_VERSION
    target_registry = json.loads(json.dumps(base_registry))
    target_registry["revision"] = 1
    target_registry["events"].append(
        {
            "action": "legacy_pending_operation",
            "reason": "Test read-only projection",
            "created_at": "2026-08-13T00:00:00Z",
        }
    )
    alias_path = Path(environment["GMONEY_ALIAS_REGISTRY"])
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(json.dumps(base_registry))
    review_path = store.job_dir(job_id) / "review.json"
    review_path.write_text(json.dumps(base_review))
    journal_path = store.job_dir(job_id) / ".alias-operation.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": ALIAS_JOURNAL_VERSION,
                "job_id": job_id,
                "base_review": base_review,
                "target_review": target_review,
                "base_registry": base_registry,
                "target_registry": target_registry,
                "base_review_sha256": _digest(base_review),
                "target_review_sha256": _digest(target_review),
                "base_registry_sha256": _digest(base_registry),
                "target_registry_sha256": _digest(target_registry),
            }
        )
    )
    tracked = (alias_path, review_path, journal_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked}
    runner = CliRunner()

    validated = runner.invoke(app, ["validate"], env=environment)
    checked = runner.invoke(app, ["check-access"], env=environment)

    assert validated.exit_code == 0, validated.output
    payload = json.loads(validated.output)
    assert payload["alias_registry_version"] == LEGACY_ALIAS_REGISTRY_VERSION
    assert payload["alias_registry_revision"] == 0
    assert payload["projected_alias_registry_version"] == "hospital_alias_registry_v3"
    assert payload["projected_alias_registry_revision"] == 2
    assert payload["pending_journal_count"] == 1
    assert payload["recovery_required"] is True
    assert payload["migration_required"] is True
    assert checked.exit_code == 0, checked.output
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked
    } == before
    assert not list(tmp_path.rglob(".gmoney-access-*"))
