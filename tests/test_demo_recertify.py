from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gmoney.demo import recertify as recertify_module
from gmoney.demo.recertify import apply_recertification, plan_recertification
from gmoney.demo.store import JobStore
from gmoney.extraction import validation as validation_module
from gmoney.extraction.validation import ValidationReport, ValidationStatus


def _legacy_v5_job(root: Path, *, recovered: bool = False) -> tuple[JobStore, str]:
    store = JobStore(root)
    state = store.create("legacy-v5.pdf")
    job_id = state["id"]
    source = b"%PDF-legacy-v5"
    (store.job_dir(job_id) / "source.pdf").write_bytes(source)
    result = {
        "output_version": "offline_accuracy_spine_v5",
        "contract_revision": 2,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "rows": [],
        "recovery": {"attempted": recovered, "targets": []},
        "semantic_validation": {
            "validation_version": "legacy-validation",
            "status": "passed",
            "issues": [],
        },
    }
    (store.job_dir(job_id) / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (store.job_dir(job_id) / "validation.json").write_text(
        json.dumps(result["semantic_validation"], indent=2, sort_keys=True) + "\n"
    )
    review = store.empty_review()
    review["revision"] = 4
    review["approval"] = {"status": "approved"}
    (store.job_dir(job_id) / "review.json").write_text(json.dumps(review))
    store.update(job_id, status="complete", certification=None)
    return store, job_id


def _passing(*_args: object) -> ValidationReport:
    return ValidationReport(status=ValidationStatus.PASSED, issues=())


def test_recertification_dry_run_is_read_only_and_apply_clears_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _legacy_v5_job(tmp_path)
    monkeypatch.setattr(recertify_module, "validate_extraction_result", _passing)
    monkeypatch.setattr(validation_module, "validate_extraction_result", _passing)
    tracked = {
        name: (store.job_dir(job_id) / name).read_bytes()
        for name in ("state.json", "result.json", "validation.json", "review.json")
    }

    plan = plan_recertification(tmp_path)

    assert plan["jobs"][0]["eligibility"] == "eligible"
    assert all(
        (store.job_dir(job_id) / name).read_bytes() == content for name, content in tracked.items()
    )

    report = apply_recertification(tmp_path, plan)

    assert report["jobs"][0]["status"] == "recertified"
    assert store.read(job_id)["_certification_valid"] is True
    review = store.read_review(job_id)
    assert review["approval"] is None
    assert review["revision"] == 5


def test_revision_two_recovery_requires_full_reprocessing(
    tmp_path: Path,
) -> None:
    _store, job_id = _legacy_v5_job(tmp_path, recovered=True)

    plan = plan_recertification(tmp_path, {job_id})

    assert plan["jobs"][0]["eligibility"] == "reprocess_required"
    assert plan["jobs"][0]["reason"] == ("revision_2_recovery_audit_incomplete")
