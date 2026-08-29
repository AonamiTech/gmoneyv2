from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import pytest
from pydantic import ValidationError

from gmoney.demo.store import JobStore
from gmoney.evaluation import release_gate
from gmoney.evaluation.release_gate import (
    CorpusManifest,
    evaluate_manifest,
    seal_manifest,
)
from gmoney.extraction import validation as validation_module
from gmoney.extraction.validation import ValidationReport, ValidationStatus

REVISION = "1" * 40
IMAGE = "sha256:" + "2" * 64
IMAGES = {"api": IMAGE, "frontend": IMAGE, "worker": IMAGE}


def _pdf(path: Path, label: str) -> None:
    document = fitz.open()
    page = document.new_page(width=100, height=100)
    page.insert_text((10, 20), label)
    document.save(path)
    document.close()


def _published_job(data_root: Path, source: Path) -> None:
    store = JobStore(data_root)
    state = store.create(source.name)
    (store.job_dir(state["id"]) / "source.pdf").write_bytes(source.read_bytes())
    store.update(state["id"], status="queued")
    assert store.claim_queued(state["id"]) is not None
    assert store.publish_processing_outcome(
        state["id"],
        {
            "output_version": "offline_accuracy_spine_v5",
            "contract_revision": 2,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "worker_release_revision": REVISION,
            "rows": [],
        },
    )


@pytest.fixture()
def small_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, tuple[Path, int]], Path, dict[str, dict[str, object]]]:
    monkeypatch.setattr(
        release_gate,
        "RELEASE_COHORT_COUNTS",
        {"production14": 1, "passing36": 1, "staging159": 1},
    )
    monkeypatch.setattr(
        validation_module,
        "validate_extraction_result",
        lambda *_: ValidationReport(status=ValidationStatus.PASSED, issues=()),
    )
    cohorts: dict[str, tuple[Path, int]] = {}
    data_root = tmp_path / "jobs"
    for ordinal, name in enumerate(("production14", "passing36", "staging159")):
        root = tmp_path / name
        root.mkdir()
        source = root / f"private-client-name-{ordinal}.pdf"
        _pdf(source, f"bill-{ordinal}")
        cohorts[name] = (root, 1)
        _published_job(data_root, source)
    inventory = release_gate._workspace_inventory(data_root, REVISION)
    gold = {
        digest: matches[0]["gold"].model_dump(mode="json") for digest, matches in inventory.items()
    }
    return cohorts, data_root, gold


def test_strict_manifest_and_exact_gold_pass_when_attestation_matches(
    small_gate: tuple[dict[str, tuple[Path, int]], Path, dict[str, dict[str, object]]],
) -> None:
    cohorts, data_root, gold = small_gate
    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    encoded = json.dumps(manifest)

    assert manifest["manifest_version"] == "gmoney_release_corpus_v2"
    assert "private-client-name" not in encoded
    report = evaluate_manifest(
        manifest,
        {name: root for name, (root, _count) in cohorts.items()},
        data_root,
        release_revision=REVISION,
        image_digests=IMAGES,
    )
    assert report["passed"] is True


def test_manifest_rejects_unknown_gold_fields_and_duplicate_hashes(
    small_gate: tuple[dict[str, tuple[Path, int]], Path, dict[str, dict[str, object]]],
) -> None:
    cohorts, data_root, gold = small_gate
    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    manifest["cohorts"]["production14"]["documents"][0]["gold"]["totlas_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(manifest)

    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    document = manifest["cohorts"]["production14"]["documents"][0]
    manifest["cohorts"]["production14"]["required_count"] = 2
    manifest["cohorts"]["production14"]["documents"] = [document, document]
    with pytest.raises(ValidationError, match="duplicate source hashes"):
        CorpusManifest.model_validate(manifest)


def test_passing_baseline_and_attestation_are_mandatory(
    small_gate: tuple[dict[str, tuple[Path, int]], Path, dict[str, dict[str, object]]],
) -> None:
    cohorts, data_root, gold = small_gate
    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    manifest["cohorts"]["passing36"]["documents"][0].pop("baseline")
    with pytest.raises(ValidationError, match="baseline snapshot"):
        CorpusManifest.model_validate(manifest)

    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    with pytest.raises(ValueError, match="image digest"):
        evaluate_manifest(
            manifest,
            {name: root for name, (root, _count) in cohorts.items()},
            data_root,
            release_revision=REVISION,
            image_digests={
                **IMAGES,
                "worker": "sha256:" + "3" * 64,
            },
        )


def test_exact_row_or_evidence_fingerprint_regression_fails_gate(
    small_gate: tuple[dict[str, tuple[Path, int]], Path, dict[str, dict[str, object]]],
) -> None:
    cohorts, data_root, gold = small_gate
    manifest = seal_manifest(
        cohorts,
        gold=gold,
        release_revision=REVISION,
        image_digests=IMAGES,
        baseline_data_root=data_root,
    )
    manifest["cohorts"]["staging159"]["documents"][0]["gold"]["canonical_rows_sha256"] = "f" * 64

    report = evaluate_manifest(
        manifest,
        {name: root for name, (root, _count) in cohorts.items()},
        data_root,
        release_revision=REVISION,
        image_digests=IMAGES,
    )

    assert report["passed"] is False
    failures = report["cohorts"]["staging159"]["documents"][0]["failures"]
    assert "canonical_rows_sha256_mismatch" in failures
