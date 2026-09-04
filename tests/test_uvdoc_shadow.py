from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gmoney.contracts.v6 import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    DenseBackwardGridMapping,
    ExtractionResultV6,
    IdentityMapping,
    PageArtifact,
    UvdocShadowRun,
)
from gmoney.evaluation.corpus import sha256_file
from gmoney.evaluation.uvdoc import UvdocAccuracyReport, evaluate_uvdoc_gate
from gmoney.geometry.dense import write_dense_grid
from gmoney.inference.uvdoc import (
    PaddleUvdocAdapter,
    UvdocPreregistration,
    load_preregistration,
    replay_uvdoc,
)


def _identity_grid(height: int, width: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    return np.stack((grid_x, grid_y), axis=-1)


def _preregistration() -> UvdocPreregistration:
    return UvdocPreregistration(
        release_corpus_manifest_sha256="a" * 64,
        audited_gold_sha256="b" * 64,
        evaluator_name="gmoney-m4",
        evaluator_version="v1",
        evaluator_sha256="c" * 64,
        model_sha256="1" * 64,
        model_config_sha256="2" * 64,
        adapter_config_sha256="3" * 64,
        targets=(
            {
                "source_sha256": "d" * 64,
                "page_number": 1,
                "cohort": "curved",
                "failure_ids": ("missing-row-1",),
            },
            {
                "source_sha256": "e" * 64,
                "page_number": 2,
                "cohort": "flat",
            },
        ),
    )


def test_preregistration_is_strict_unique_and_hash_bound(tmp_path: Path) -> None:
    preregistration = _preregistration()
    path = tmp_path / "uvdoc.json"
    path.write_text(preregistration.model_dump_json())
    loaded, digest = load_preregistration(path)
    assert loaded == preregistration
    assert digest == sha256_file(path)
    assert loaded.eligible("d" * 64, 1)
    assert not loaded.eligible("d" * 64, 2)

    with pytest.raises(ValueError, match="unique"):
        UvdocPreregistration(
            **preregistration.model_dump(exclude={"targets"}),
            targets=(preregistration.targets[0], preregistration.targets[0]),
        )


def test_replay_uvdoc_identity_preserves_rgb_conversion() -> None:
    parent_bgr = np.asarray(
        [[[1, 2, 3], [10, 20, 30]], [[40, 50, 60], [70, 80, 90]]], dtype=np.uint8
    )
    replayed = replay_uvdoc(parent_bgr, _identity_grid(2, 2))
    assert np.array_equal(replayed, parent_bgr[..., ::-1])


def test_adapter_captures_and_replays_exact_grid(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "upsample_size": [712, 488],
                "upsample_mode": "bilinear",
                "out_point_positions2D": [[128, 32], [32, 2]],
            }
        )
    )
    (model_dir / "model.safetensors").write_bytes(b"fixture")
    model = SimpleNamespace(eval=lambda: None)
    adapter = PaddleUvdocAdapter(model_dir, model=model)
    parent_bgr = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    input_path = tmp_path / "input.png"
    assert cv2.imwrite(str(input_path), parent_bgr)
    grid = _identity_grid(4, 5)
    monkeypatch.setattr(
        adapter,
        "_forward",
        lambda batch: (replay_uvdoc(parent_bgr, grid), grid),
    )

    run = adapter.predict(input_path, tmp_path / "artifacts", page_number=1)
    assert run.status == "valid"
    assert run.reproduction_max_error_by_channel == (0, 0, 0)
    assert run.grid_shape == (4, 5, 2)
    assert run.transform_metrics["foldover_count"] == 0


def test_v6_uvdoc_shadow_requires_unselected_dense_lineage(tmp_path: Path) -> None:
    source = ArtifactRef(
        artifact_kind=ArtifactKind.SOURCE_RAW,
        image_sha256="1" * 64,
        artifact_relative_path="page.png",
        width=4,
        height=4,
        producer="fixture",
        producer_version="v1",
        configuration_sha256="2" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )
    oriented = ArtifactRef(
        artifact_kind=ArtifactKind.ORIENTED_RAW,
        image_sha256="1" * 64,
        artifact_relative_path="page.png",
        width=4,
        height=4,
        parent_artifact_id=source.artifact_id,
        producer="fixture",
        producer_version="v1",
        configuration_sha256="3" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )
    stored = write_dense_grid(tmp_path, "uvdoc/grid.npz", _identity_grid(4, 4))
    uvdoc = ArtifactRef(
        artifact_kind=ArtifactKind.UVDOC,
        image_sha256="4" * 64,
        artifact_relative_path="uvdoc/page.png",
        width=4,
        height=4,
        parent_artifact_id=oriented.artifact_id,
        producer="fixture",
        producer_version="v1",
        configuration_sha256="5" * 64,
        child_to_parent_mapping=DenseBackwardGridMapping(
            grid_relative_path=stored.relative_path,
            grid_sha256=stored.sha256,
            grid_shape=stored.shape,
            child_width=4,
            child_height=4,
            parent_width=4,
            parent_height=4,
            padding_mode="zeros",
        ),
    )
    enhanced = ArtifactRef(
        artifact_kind=ArtifactKind.UVDOC_ENHANCED,
        image_sha256="6" * 64,
        artifact_relative_path="uvdoc/enhanced.png",
        width=4,
        height=4,
        parent_artifact_id=uvdoc.artifact_id,
        producer="fixture",
        producer_version="v1",
        configuration_sha256="7" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )
    pages = (
        PageArtifact(artifact=source, page_number=1, dpi=300, role="SOURCE_RAW"),
        PageArtifact(artifact=oriented, page_number=1, dpi=300, role="ORIENTED_RAW"),
        PageArtifact(artifact=uvdoc, page_number=1, dpi=300, role="CANDIDATE"),
        PageArtifact(artifact=enhanced, page_number=1, dpi=300, role="CANDIDATE"),
    )
    run = UvdocShadowRun(
        page_number=1,
        status="valid",
        reason_code="uvdoc_shadow_valid",
        input_artifact_id=oriented.artifact_id,
        uvdoc_artifact_id=uvdoc.artifact_id,
        enhanced_artifact_id=enhanced.artifact_id,
        model_sha256="8" * 64,
        model_config_sha256="9" * 64,
        adapter_config_sha256="a" * 64,
        paddle_version="3.2.2",
        paddleocr_version="3.7.0",
        paddlex_version="3.7.2",
        reproduction_max_error_by_channel=(0, 1, 0),
        transform_metrics={
            "mean_displacement_px": 0.0,
            "max_displacement_px": 0.0,
            "local_scale_p05": 1.0,
            "local_scale_p50": 1.0,
            "local_scale_p95": 1.0,
            "anisotropy_p95": 1.0,
            "jacobian_determinant_p05": 1.0,
            "jacobian_determinant_p50": 1.0,
            "jacobian_determinant_p95": 1.0,
            "foldover_count": 0,
            "out_of_bounds_rate": 0.0,
        },
    )
    payload = dict(
        document_id="fixture",
        source_sha256="b" * 64,
        source_name="fixture.pdf",
        pages=1,
        artifact_manifest=ArtifactManifest(artifacts=(source, oriented, uvdoc, enhanced)),
        page_artifacts=pages,
        uvdoc_shadow_runs=(run,),
    )
    assert ExtractionResultV6(**payload).uvdoc_shadow_runs == (run,)
    selected_pages = (*pages[:-2], pages[-2].model_copy(update={"selected": True}), pages[-1])
    with pytest.raises(ValueError, match="unselected"):
        ExtractionResultV6(**{**payload, "page_artifacts": selected_pages})


def test_uvdoc_gate_holds_without_authoritative_accuracy(monkeypatch, tmp_path: Path) -> None:
    preregistration = _preregistration()
    monkeypatch.setattr(
        "gmoney.evaluation.uvdoc._shadow_inventory",
        lambda root: {
            "d" * 64: [{"uvdoc_shadow_runs": [{"page_number": 1, "status": "valid"}]}],
            "e" * 64: [{"uvdoc_shadow_runs": [{"page_number": 2, "status": "valid"}]}],
        },
    )
    report = evaluate_uvdoc_gate(preregistration, tmp_path, None)
    assert report["status"] == "hold"
    assert report["accuracy_failures"] == ["authoritative_accuracy_report_missing"]


def test_uvdoc_gate_uses_primary_branch_and_flat_non_regression(
    monkeypatch, tmp_path: Path
) -> None:
    preregistration = _preregistration()
    monkeypatch.setattr(
        "gmoney.evaluation.uvdoc._shadow_inventory",
        lambda root: {
            "d" * 64: [{"uvdoc_shadow_runs": [{"page_number": 1, "status": "valid"}]}],
            "e" * 64: [{"uvdoc_shadow_runs": [{"page_number": 2, "status": "valid"}]}],
        },
    )
    accuracy = UvdocAccuracyReport(
        release_corpus_manifest_sha256=preregistration.release_corpus_manifest_sha256,
        audited_gold_sha256=preregistration.audited_gold_sha256,
        evaluator_name=preregistration.evaluator_name,
        evaluator_version=preregistration.evaluator_version,
        evaluator_sha256=preregistration.evaluator_sha256,
        model_sha256=preregistration.model_sha256,
        model_config_sha256=preregistration.model_config_sha256,
        adapter_config_sha256=preregistration.adapter_config_sha256,
        observations=(
            {
                "source_sha256": "d" * 64,
                "page_number": 1,
                "cohort": "curved",
                "branch": "UVDOC",
                "improved_failure_ids": ("missing-row-1",),
            },
            {
                "source_sha256": "e" * 64,
                "page_number": 2,
                "cohort": "flat",
                "branch": "UVDOC",
            },
        ),
    )
    report = evaluate_uvdoc_gate(preregistration, tmp_path, accuracy)
    assert report["status"] == "shadow"
    assert report["improved_curved_failure_ids"] == ["missing-row-1"]

    regressed = accuracy.model_copy(
        update={
            "observations": (
                accuracy.observations[0],
                accuracy.observations[1].model_copy(
                    update={"new_critical_error_ids": ("amount-2",)}
                ),
            )
        }
    )
    report = evaluate_uvdoc_gate(preregistration, tmp_path, regressed)
    assert report["status"] == "hold"
    assert "new_critical_error" in report["accuracy_failures"]
