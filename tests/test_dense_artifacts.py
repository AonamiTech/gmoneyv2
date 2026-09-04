from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest
from pydantic import ValidationError

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.v6 import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    CanonicalTableArtifact,
    DenseBackwardGridMapping,
    EvidenceRefV2,
    ExtractionResultV6,
    HomographyMapping,
    IdentityMapping,
    PageArtifact,
)
from gmoney.demo.recertify import apply_recertification, plan_recertification
from gmoney.demo.review import create_evidence_bundle
from gmoney.demo.store import JobStore, JobTransactionError
from gmoney.extraction.validation import validate_extraction_result
from gmoney.geometry.artifacts import map_points_to_source, map_polygon_to_source
from gmoney.geometry.dense import (
    DenseGridError,
    DenseGridResolver,
    analyze_dense_grid,
    load_dense_grid,
    render_dense_overlay,
    write_dense_grid,
)


def _identity_grid(height: int, width: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy), axis=-1)


def _mapping(
    root: Path,
    grid: np.ndarray,
    *,
    child_width: int = 101,
    child_height: int = 101,
    parent_width: int = 101,
    parent_height: int = 101,
    relative_path: str = "grids/grid.npz",
) -> DenseBackwardGridMapping:
    stored = write_dense_grid(root, relative_path, grid)
    return DenseBackwardGridMapping(
        grid_relative_path=stored.relative_path,
        grid_sha256=stored.sha256,
        grid_shape=stored.shape,
        child_width=child_width,
        child_height=child_height,
        parent_width=parent_width,
        parent_height=parent_height,
        padding_mode="border",
    )


def _artifact(
    kind: ArtifactKind,
    image: str,
    *,
    parent: str | None = None,
    mapping: IdentityMapping | HomographyMapping | DenseBackwardGridMapping | None = None,
    width: int = 101,
    height: int = 101,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=kind,
        image_sha256=image * 64,
        artifact_relative_path=f"{image}.png",
        width=width,
        height=height,
        parent_artifact_id=parent,
        producer="test",
        producer_version="1",
        configuration_sha256="c" * 64,
        child_to_parent_mapping=mapping or IdentityMapping(),
    )


def test_dense_grid_storage_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    grid = _identity_grid(8, 12)
    first = write_dense_grid(tmp_path, "grids/identity.npz", grid)
    first_bytes = (tmp_path / first.relative_path).read_bytes()
    second = write_dense_grid(tmp_path, "grids/identity.npz", np.asfortranarray(grid))

    assert first == second
    assert hashlib.sha256(first_bytes).hexdigest() == first.sha256
    with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
        assert archive.namelist() == ["grid.npy"]
        assert archive.infolist()[0].date_time == (1980, 1, 1, 0, 0, 0)
    loaded = load_dense_grid(tmp_path, _mapping(tmp_path, grid, relative_path="copy.npz"))
    assert loaded.dtype == np.dtype("<f4")
    assert loaded.flags.c_contiguous
    assert not loaded.flags.writeable
    np.testing.assert_array_equal(loaded, grid)


def test_dense_grid_serialization_is_deterministic_across_processes(tmp_path: Path) -> None:
    script = """
import hashlib
import sys
from pathlib import Path
import numpy as np
from gmoney.geometry.dense import write_dense_grid
size = 16
x = np.linspace(-1, 1, size, dtype=np.float32)
y = np.linspace(-1, 1, size, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
stored = write_dense_grid(Path(sys.argv[1]), 'grid.npz', np.stack((xx, yy), axis=-1))
print(stored.sha256)
"""
    digests = [
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / f"process-{index}")],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for index in range(2)
    ]
    assert len(set(digests)) == 1


def test_dense_contract_rejects_invalid_paths_shapes_and_dimensions() -> None:
    base = {
        "grid_relative_path": "grids/grid.npz",
        "grid_sha256": "a" * 64,
        "grid_shape": (2, 2, 2),
        "child_width": 2,
        "child_height": 2,
        "parent_width": 2,
        "parent_height": 2,
        "padding_mode": "border",
    }
    for change in (
        {"grid_relative_path": "../grid.npz"},
        {"grid_relative_path": "/grid.npz"},
        {"grid_relative_path": "grid.npy"},
        {"grid_shape": (1, 2, 2)},
        {"grid_shape": (2, 2, 3)},
        {"child_width": 1},
        {"parent_height": 1},
    ):
        with pytest.raises(ValidationError):
            DenseBackwardGridMapping(**{**base, **change})

    dense = DenseBackwardGridMapping(**base)
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    wrong_child = _artifact(
        ArtifactKind.UVDOC,
        "b",
        parent=source.artifact_id,
        mapping=dense,
        width=101,
        height=101,
    )
    with pytest.raises(ValidationError, match="child dimensions"):
        ArtifactManifest(artifacts=(source, wrong_child))


def test_dense_identity_and_mixed_chain_map_points(tmp_path: Path) -> None:
    dense = _mapping(tmp_path, _identity_grid(11, 11))
    source = _artifact(ArtifactKind.SOURCE_RAW, "a", width=121, height=121)
    projective = _artifact(
        ArtifactKind.PROJECTIVE,
        "b",
        parent=source.artifact_id,
        mapping=HomographyMapping(
            child_to_parent_matrix=((1, 0, 10), (0, 1, 15), (0, 0, 1))
        ),
    )
    dense_child = _artifact(
        ArtifactKind.UVDOC,
        "d",
        parent=projective.artifact_id,
        mapping=dense,
    )
    manifest = ArtifactManifest(artifacts=(source, projective, dense_child))
    resolver = DenseGridResolver(tmp_path)

    np.testing.assert_allclose(
        map_points_to_source(
            manifest,
            dense_child.artifact_id,
            ((0, 0), (50, 75), (100, 100)),
            dense_grid_resolver=resolver,
        ),
        ((10.0, 15.0), (60.0, 90.0), (110.0, 115.0)),
    )


def test_smooth_dense_warp_adaptively_maps_polygon_and_reports_diagnostics(
    tmp_path: Path,
) -> None:
    size = 101
    x = np.linspace(0.0, size - 1, 11)
    y = np.linspace(0.0, size - 1, 11)
    xx, yy = np.meshgrid(x, y)
    warped_x = xx + 5.0 * np.sin(np.pi * xx / (size - 1)) * np.sin(
        np.pi * yy / (size - 1)
    )
    grid = np.stack(
        (2.0 * warped_x / (size - 1) - 1.0, 2.0 * yy / (size - 1) - 1.0),
        axis=-1,
    ).astype(np.float32)
    dense = _mapping(tmp_path, grid)
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    child = _artifact(ArtifactKind.UVDOC, "b", parent=source.artifact_id, mapping=dense)
    manifest = ArtifactManifest(artifacts=(source, child))
    polygon = Polygon(
        points=(
            Point(x=10, y=10),
            Point(x=90, y=10),
            Point(x=90, y=90),
            Point(x=10, y=90),
        )
    )

    mapped = map_polygon_to_source(
        manifest,
        child.artifact_id,
        polygon,
        dense_grid_resolver=DenseGridResolver(tmp_path),
    )
    diagnostics = analyze_dense_grid(dense, load_dense_grid(tmp_path, dense))

    assert len(mapped.points) > len(polygon.points)
    assert diagnostics.foldover_count == 0
    assert diagnostics.out_of_bounds_rate == 0
    assert diagnostics.max_displacement_px == pytest.approx(5.0, abs=0.01)

    sample = tuple((float(value), float(value)) for value in np.linspace(0, 100, 101))
    actual = map_points_to_source(
        manifest,
        child.artifact_id,
        sample,
        dense_grid_resolver=DenseGridResolver(tmp_path),
    )
    expected = tuple(
        (
            value + 5.0 * np.sin(np.pi * value / 100) ** 2,
            value,
        )
        for value in np.linspace(0, 100, 101)
    )
    assert (
        max(
            np.hypot(x - ex, y - ey)
            for (x, y), (ex, ey) in zip(actual, expected, strict=True)
        )
        < 2
    )


def test_dense_overlay_is_deterministic(tmp_path: Path) -> None:
    grid = _identity_grid(8, 8)
    mapping = _mapping(tmp_path, grid)
    polygons = (((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),)
    first = render_dense_overlay(mapping, grid, polygons)
    second = render_dense_overlay(mapping, grid, polygons)
    assert first == second
    decoded = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (101, 101, 3)


def test_dense_loader_rejects_missing_tampered_and_symlinked_files(tmp_path: Path) -> None:
    grid = _identity_grid(3, 3)
    mapping = _mapping(tmp_path, grid)
    target = tmp_path / mapping.grid_relative_path
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(DenseGridError, match="digest") as tampered:
        load_dense_grid(tmp_path, mapping)
    assert tampered.value.code == "v6_dense_grid_digest_mismatch"

    target.unlink()
    with pytest.raises(DenseGridError, match="missing") as missing:
        load_dense_grid(tmp_path, mapping)
    assert missing.value.code == "v6_dense_grid_missing"

    outside = tmp_path / "outside.npz"
    write_dense_grid(tmp_path, "outside.npz", grid)
    target.symlink_to(outside)
    with pytest.raises(DenseGridError, match="symlink") as symlinked:
        load_dense_grid(tmp_path, mapping)
    assert symlinked.value.code == "v6_dense_grid_path_invalid"


@pytest.mark.parametrize(
    ("array", "expected_code"),
    (
        (np.zeros((3, 3, 2), dtype=np.float64), "v6_dense_grid_metadata_mismatch"),
        (
            np.full((3, 3, 2), np.nan, dtype=np.float32),
            "v6_dense_grid_non_finite",
        ),
    ),
)
def test_dense_loader_rejects_invalid_member_arrays(
    tmp_path: Path,
    array: np.ndarray,
    expected_code: str,
) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, grid=array)
    content = stream.getvalue()
    target = tmp_path / "grid.npz"
    target.write_bytes(content)
    mapping = DenseBackwardGridMapping(
        grid_relative_path="grid.npz",
        grid_sha256=hashlib.sha256(content).hexdigest(),
        grid_shape=(3, 3, 2),
        child_width=3,
        child_height=3,
        parent_width=3,
        parent_height=3,
        padding_mode="border",
    )
    with pytest.raises(DenseGridError) as failure:
        load_dense_grid(tmp_path, mapping)
    assert failure.value.code == expected_code


def test_dense_loader_rejects_extra_archive_member(tmp_path: Path) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, grid=_identity_grid(3, 3), extra=np.zeros(1))
    content = stream.getvalue()
    (tmp_path / "grid.npz").write_bytes(content)
    mapping = DenseBackwardGridMapping(
        grid_relative_path="grid.npz",
        grid_sha256=hashlib.sha256(content).hexdigest(),
        grid_shape=(3, 3, 2),
        child_width=3,
        child_height=3,
        parent_width=3,
        parent_height=3,
        padding_mode="border",
    )
    with pytest.raises(DenseGridError) as failure:
        load_dense_grid(tmp_path, mapping)
    assert failure.value.code == "v6_dense_grid_archive_invalid"


def test_dense_loader_rejects_corrupt_archive_with_matching_digest(tmp_path: Path) -> None:
    content = b"not-a-zip-archive"
    (tmp_path / "grid.npz").write_bytes(content)
    mapping = DenseBackwardGridMapping(
        grid_relative_path="grid.npz",
        grid_sha256=hashlib.sha256(content).hexdigest(),
        grid_shape=(3, 3, 2),
        child_width=3,
        child_height=3,
        parent_width=3,
        parent_height=3,
        padding_mode="border",
    )
    with pytest.raises(DenseGridError) as failure:
        load_dense_grid(tmp_path, mapping)
    assert failure.value.code == "v6_dense_grid_archive_invalid"


def test_dense_loader_rejects_oversized_npy_header_before_allocation(tmp_path: Path) -> None:
    npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        npy,
        {
            "descr": "<f4",
            "fortran_order": False,
            "shape": (100_000, 100_000, 2),
        },
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("grid.npy", npy.getvalue())
    content = archive.getvalue()
    (tmp_path / "grid.npz").write_bytes(content)
    mapping = DenseBackwardGridMapping(
        grid_relative_path="grid.npz",
        grid_sha256=hashlib.sha256(content).hexdigest(),
        grid_shape=(100_000, 100_000, 2),
        child_width=3,
        child_height=3,
        parent_width=3,
        parent_height=3,
        padding_mode="border",
    )
    with pytest.raises(DenseGridError, match="size limit") as failure:
        load_dense_grid(tmp_path, mapping)
    assert failure.value.code == "v6_dense_grid_archive_invalid"


def test_dense_diagnostics_detect_foldover_and_bounds(tmp_path: Path) -> None:
    folded = _identity_grid(3, 3)
    folded[:, :, 0] *= -1
    mapping = _mapping(tmp_path, folded)
    diagnostics = analyze_dense_grid(mapping, load_dense_grid(tmp_path, mapping))
    assert diagnostics.foldover_count > 0

    outside = _identity_grid(3, 3)
    outside[1, 1, 0] = 1.5
    outside_mapping = _mapping(tmp_path, outside, relative_path="outside-grid.npz")
    outside_diagnostics = analyze_dense_grid(
        outside_mapping, load_dense_grid(tmp_path, outside_mapping)
    )
    assert outside_diagnostics.out_of_bounds_rate > 0


def test_dense_polygon_rejects_orientation_intersection_and_budget(tmp_path: Path) -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    folded = _identity_grid(3, 3)
    folded[..., 0] *= -1
    folded_mapping = _mapping(tmp_path, folded)
    folded_child = _artifact(
        ArtifactKind.UVDOC,
        "b",
        parent=source.artifact_id,
        mapping=folded_mapping,
    )
    folded_manifest = ArtifactManifest(artifacts=(source, folded_child))
    rectangle = Polygon(
        points=(
            Point(x=10, y=10),
            Point(x=90, y=10),
            Point(x=90, y=90),
            Point(x=10, y=90),
        )
    )
    with pytest.raises(DenseGridError) as orientation:
        map_polygon_to_source(
            folded_manifest,
            folded_child.artifact_id,
            rectangle,
            dense_grid_resolver=DenseGridResolver(tmp_path),
        )
    assert orientation.value.code == "v6_dense_polygon_orientation_invalid"

    identity_mapping = _mapping(
        tmp_path,
        _identity_grid(3, 3),
        relative_path="identity.npz",
    )
    identity_child = _artifact(
        ArtifactKind.UVDOC,
        "c",
        parent=source.artifact_id,
        mapping=identity_mapping,
    )
    identity_manifest = ArtifactManifest(artifacts=(source, identity_child))
    crossing = Polygon(
        points=(
            Point(x=10, y=10),
            Point(x=90, y=90),
            Point(x=10, y=90),
            Point(x=90, y=10),
            Point(x=50, y=95),
        )
    )
    with pytest.raises(DenseGridError) as intersection:
        map_polygon_to_source(
            identity_manifest,
            identity_child.artifact_id,
            crossing,
            dense_grid_resolver=DenseGridResolver(tmp_path),
        )
    assert intersection.value.code == "v6_dense_polygon_self_intersection"

    curved = _identity_grid(3, 3)
    curved[1, 1, 0] += 0.1
    curved_mapping = _mapping(tmp_path, curved, relative_path="curved.npz")
    curved_child = _artifact(
        ArtifactKind.UVDOC,
        "d",
        parent=source.artifact_id,
        mapping=curved_mapping,
    )
    curved_manifest = ArtifactManifest(artifacts=(source, curved_child))
    with pytest.raises(DenseGridError) as budget:
        map_polygon_to_source(
            curved_manifest,
            curved_child.artifact_id,
            rectangle,
            dense_grid_resolver=DenseGridResolver(tmp_path),
            chord_tolerance_px=0.01,
            max_depth=0,
        )
    assert budget.value.code == "v6_dense_polygon_budget_exhausted"


def _dense_publication_fixture(
    root: Path,
    grid: np.ndarray,
    *,
    include_geometry: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    source_pdf = root / "source.pdf"
    document = fitz.open()
    document.new_page(width=101, height=101)
    document.save(source_pdf)
    document.close()
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    ok, encoded = cv2.imencode(".png", np.zeros((101, 101, 3), dtype=np.uint8))
    assert ok
    image = encoded.tobytes()
    (artifact_root / "page.png").write_bytes(image)
    image_sha = hashlib.sha256(image).hexdigest()
    dense = _mapping(artifact_root, grid)
    source = ArtifactRef(
        artifact_kind=ArtifactKind.SOURCE_RAW,
        image_sha256=image_sha,
        artifact_relative_path="page.png",
        width=101,
        height=101,
        producer="test",
        producer_version="1",
        configuration_sha256="a" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )
    oriented = ArtifactRef(
        artifact_kind=ArtifactKind.ORIENTED_RAW,
        image_sha256=image_sha,
        artifact_relative_path="page.png",
        width=101,
        height=101,
        parent_artifact_id=source.artifact_id,
        producer="test",
        producer_version="1",
        configuration_sha256="b" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )
    dense_page = ArtifactRef(
        artifact_kind=ArtifactKind.UVDOC,
        image_sha256=image_sha,
        artifact_relative_path="page.png",
        width=101,
        height=101,
        parent_artifact_id=oriented.artifact_id,
        producer="test",
        producer_version="1",
        configuration_sha256="c" * 64,
        child_to_parent_mapping=dense,
    )
    artifacts = [source, oriented, dense_page]
    tables: tuple[CanonicalTableArtifact, ...] = ()
    evidence: tuple[EvidenceRefV2, ...] = ()
    if include_geometry:
        table_artifact = ArtifactRef(
            artifact_kind=ArtifactKind.TABLE_CROP,
            image_sha256=image_sha,
            artifact_relative_path="page.png",
            width=101,
            height=101,
            parent_artifact_id=dense_page.artifact_id,
            producer="test",
            producer_version="1",
            configuration_sha256="d" * 64,
            child_to_parent_mapping=IdentityMapping(),
        )
        artifacts.append(table_artifact)
        manifest = ArtifactManifest(artifacts=tuple(artifacts))
        polygon = Polygon(
            points=(
                Point(x=10, y=10),
                Point(x=90, y=10),
                Point(x=90, y=90),
                Point(x=10, y=90),
            )
        )
        source_polygon = map_polygon_to_source(
            manifest,
            dense_page.artifact_id,
            polygon,
            dense_grid_resolver=DenseGridResolver(artifact_root),
        )
        tables = (
            CanonicalTableArtifact(
                artifact=table_artifact,
                page_number=1,
                logical_table_id="p1-t1",
                page_artifact_id=dense_page.artifact_id,
                crop_polygon_in_page_artifact=polygon,
                crop_polygon_in_source_raw=source_polygon,
            ),
        )
        evidence = (
            EvidenceRefV2(
                artifact_id=dense_page.artifact_id,
                artifact_sha256=dense_page.image_sha256,
                canonical_polygon=polygon,
                source_page_polygon=source_polygon,
                source_page_number=1,
                source_page_artifact_id=source.artifact_id,
                extractor="test",
                model_name="test",
                model_version="1",
                recognition_variant="dense",
            ),
        )
    else:
        manifest = ArtifactManifest(artifacts=tuple(artifacts))
    result = ExtractionResultV6(
        document_id="doc",
        source_sha256=hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        source_name="source.pdf",
        pages=1,
        artifact_manifest=manifest,
        page_artifacts=(
            PageArtifact(
                artifact=source,
                page_number=1,
                dpi=72,
                role="SOURCE_RAW",
            ),
            PageArtifact(
                artifact=oriented,
                page_number=1,
                dpi=72,
                role="ORIENTED_RAW",
            ),
            PageArtifact(
                artifact=dense_page,
                page_number=1,
                dpi=72,
                role="CANDIDATE",
                selected=True,
            ),
        ),
        canonical_table_artifacts=tables,
        evidence=evidence,
        worker_release_revision="e" * 40,
    ).model_dump(mode="json")
    return source_pdf, artifact_root, result


def test_dense_v6_publication_passes_and_grid_tampering_fails(tmp_path: Path) -> None:
    source, artifact_root, result = _dense_publication_fixture(
        tmp_path, _identity_grid(11, 11)
    )
    report = validate_extraction_result(source, result, artifact_root)
    assert report.status == "passed"
    assert report.validation_version == "extraction_validation_v6_r2"

    grid_path = artifact_root / "grids/grid.npz"
    grid_path.write_bytes(grid_path.read_bytes() + b"tampered")
    failed = validate_extraction_result(source, result, artifact_root)
    assert failed.status == "failed"
    assert "v6_dense_grid_digest_mismatch" in {issue.code for issue in failed.issues}


def test_dense_v6_publication_rejects_foldover_and_out_of_bounds(tmp_path: Path) -> None:
    folded = _identity_grid(3, 3)
    folded[..., 0] *= -1
    source, artifact_root, result = _dense_publication_fixture(
        tmp_path / "folded", folded, include_geometry=False
    )
    report = validate_extraction_result(source, result, artifact_root)
    assert "v6_dense_grid_jacobian_invalid" in {issue.code for issue in report.issues}

    outside = _identity_grid(3, 3)
    outside[1, 1, 0] = 1.5
    source, artifact_root, result = _dense_publication_fixture(
        tmp_path / "outside", outside, include_geometry=False
    )
    report = validate_extraction_result(source, result, artifact_root)
    assert "v6_dense_mapping_out_of_bounds" in {issue.code for issue in report.issues}


def test_dense_grid_is_bound_into_v3_certification_inventory(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    source, artifact_root, result = _dense_publication_fixture(
        fixture_root, _identity_grid(11, 11)
    )
    store = JobStore(tmp_path / "runtime")
    state = store.create("source.pdf")
    job_id = state["id"]
    job_root = store.job_dir(job_id)
    shutil.copy2(source, job_root / "source.pdf")
    shutil.copytree(artifact_root, job_root / "artifacts")
    result["document_id"] = job_id
    store.update(job_id, status="queued")
    assert store.claim_queued(job_id) is not None
    assert store.publish_processing_outcome(job_id, result)

    published = store.read(job_id)
    inventory, digest = store._artifact_inventory_unlocked(job_id, store.read_result(job_id))
    assert published["_certification_valid"] is True
    assert published["certification"]["certification_version"] == "job_certification_v3"
    assert published["certification"]["artifact_inventory_sha256"] == digest
    assert {item["relative_path"] for item in inventory} == {
        "page.png",
        "grids/grid.npz",
    }
    plan = plan_recertification(store.root, {job_id})
    assert plan["jobs"][0]["eligibility"] == "eligible"
    recertification = apply_recertification(store.root, plan)
    assert recertification["jobs"][0]["status"] == "recertified"
    published = store.read(job_id)
    assert published["_certification_valid"] is True
    bundle = create_evidence_bundle(
        store,
        job_id,
        published,
        store.read_result(job_id),
        store.empty_review(),
    )
    with zipfile.ZipFile(bundle) as archive:
        assert "artifacts/grids/grid.npz" in archive.namelist()

    grid_path = job_root / "artifacts/grids/grid.npz"
    grid_path.write_bytes(grid_path.read_bytes() + b"tampered")
    assert store.read(job_id)["_certification_valid"] is False
    with pytest.raises(JobTransactionError, match="digest"):
        store._artifact_inventory_unlocked(job_id, store.read_result(job_id))
    with pytest.raises(JobTransactionError, match="digest"):
        create_evidence_bundle(
            store,
            job_id,
            store.read(job_id),
            store.read_result(job_id),
            store.empty_review(),
        )
