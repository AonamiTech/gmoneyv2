from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from gmoney.contracts.evidence import PageAsset, PageQuality, PageQualityFlag
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.crop import (
    color_overlay_suppressed_variant,
    crop_region,
    render_pdf_region,
    resize_region,
)
from gmoney.geometry.normalize import normalize_page, normalize_quadrilateral_region
from gmoney.geometry.preprocess import (
    detect_page_quadrilateral,
    detect_table_quadrilateral,
    prepare_page_candidates,
)
from gmoney.geometry.quality import estimate_skew
from gmoney.geometry.render import render_pdf
from gmoney.geometry.transform import (
    apply_matrix,
    compose,
    identity,
    invert,
    right_angle_rotation,
    rotation,
    translation,
)


def test_transform_round_trip_is_subpixel() -> None:
    matrix = compose(translation(15, -3), rotation(7.5, 100, 200), translation(-2, 9))
    points = ((0.0, 0.0), (50.5, 22.25), (199.0, 399.0), (120.0, 80.0))
    transformed = apply_matrix(matrix, points)
    restored = apply_matrix(invert(matrix), transformed)
    for expected, actual in zip(points, restored, strict=True):
        assert np.linalg.norm(np.asarray(expected) - np.asarray(actual)) < 1e-6
    assert compose(identity(), identity()) == identity()


def test_render_is_immutable_and_idempotent(tmp_path: Path) -> None:
    pdf = tmp_path / "bill.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 50), "DESCRIPTION  QTY  RATE  AMOUNT")
    page.insert_text((30, 80), "Complete Blood Count  1  50.00  50.00")
    document.save(pdf)
    document.close()
    original_hash = sha256_file(pdf)

    first = render_pdf(pdf, tmp_path / "artifacts", dpi=144)
    second = render_pdf(pdf, tmp_path / "artifacts", dpi=144)

    assert sha256_file(pdf) == original_hash
    assert first.document_sha256 == second.document_sha256 == original_hash
    assert first.pages[0].artifact_sha256 == second.pages[0].artifact_sha256
    assert first.pages[0].width == 600
    assert first.pages[0].height == 800


def test_skew_estimator_finds_tilted_horizontal_rows() -> None:
    image = np.full((500, 800), 255, dtype=np.uint8)
    for y in range(80, 450, 50):
        cv2.line(image, (40, y), (760, y), 0, 3)
    transform = cv2.getRotationMatrix2D((400, 250), 4.0, 1.0)
    tilted = cv2.warpAffine(image, transform, (800, 500), borderValue=255)
    assert abs(abs(estimate_skew(tilted)) - 4.0) < 0.5


def test_normalization_persists_inverse_mapping(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    image = np.full((300, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 30), (180, 270), (0, 0, 0), 2)
    assert cv2.imwrite(str(source), image)
    result = normalize_page(
        source,
        tmp_path / "normalized.png",
        page_number=1,
        dpi=300,
        rotation_correction_degrees=3.0,
        perspective_source=((5, 8), (194, 2), (198, 295), (3, 290)),
    )
    points = ((25.0, 35.0), (170.0, 35.0), (170.0, 260.0), (25.0, 260.0))
    derived = apply_matrix(result.transform.forward_matrix, points)
    restored = apply_matrix(result.transform.inverse_matrix, derived)
    for expected, actual in zip(points, restored, strict=True):
        assert np.linalg.norm(np.asarray(expected) - np.asarray(actual)) <= 2.0
    assert result.output_path.exists()


def test_crop_retains_inverse_page_mapping(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    image = np.full((300, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    result = crop_region(source, tmp_path / "crop.png", 1, (20, 30, 180, 270))
    page_points = ((20.0, 30.0), (180.0, 270.0), (50.0, 60.0))
    crop_points = apply_matrix(result.transform.forward_matrix, page_points)
    assert crop_points[0] == (0.0, 0.0)
    restored = apply_matrix(result.transform.inverse_matrix, crop_points)
    assert restored == page_points


def test_canonical_recovery_resize_retains_inverse_crop_mapping(tmp_path: Path) -> None:
    source = tmp_path / "crop.png"
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    result = resize_region(source, tmp_path / "resized.png", 1, 2.0)
    original = ((0.0, 0.0), (119.0, 79.0), (30.0, 20.0))
    resized = apply_matrix(result.transform.forward_matrix, original)
    assert resized[0] == (0.0, 0.0)
    assert resized[1] == (239.0, 159.0)
    restored = apply_matrix(result.transform.inverse_matrix, resized)
    for expected, actual in zip(original, restored, strict=True):
        assert actual == pytest.approx(expected)


def test_canonical_recovery_resize_binds_rounded_output_endpoints(tmp_path: Path) -> None:
    source = tmp_path / "crop.png"
    image = np.full((732, 2306, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    result = resize_region(source, tmp_path / "resized.png", 1, 4 / 3)

    assert (result.transform.derived_width, result.transform.derived_height) == (3075, 976)
    child_endpoints = ((0.0, 0.0), (3074.0, 975.0))
    restored = apply_matrix(result.transform.inverse_matrix, child_endpoints)
    for expected, actual in zip(((0.0, 0.0), (2305.0, 731.0)), restored, strict=True):
        assert actual == pytest.approx(expected)


def test_canonical_recovery_identity_resize_preserves_pixel_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "crop.png"
    image = np.full((7, 11, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    result = resize_region(source, tmp_path / "resized.png", 1, 1.0)

    assert result.transform.forward_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_canonical_recovery_resize_rejects_expanded_one_pixel_axis(tmp_path: Path) -> None:
    source = tmp_path / "crop.png"
    image = np.full((7, 1, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    with pytest.raises(ValueError, match="one-pixel axes"):
        resize_region(source, tmp_path / "resized.png", 1, 2.0)


def test_color_overlay_suppression_fades_colored_marks_and_preserves_dark_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "colored.png"
    image = np.asarray(
        (
            ((10, 10, 10), (120, 50, 200)),
            ((255, 255, 255), (40, 180, 80)),
        ),
        dtype=np.uint8,
    )
    assert cv2.imwrite(str(source), image)

    result = color_overlay_suppressed_variant(
        source,
        tmp_path / "overlay-suppressed.png",
    )

    recovered = cv2.imread(str(result.output_path), cv2.IMREAD_GRAYSCALE)
    assert recovered is not None
    assert recovered.tolist() == [[10, 200], [255, 180]]
    assert result.artifact_sha256 == sha256_file(result.output_path)


def test_400_dpi_region_rerender_round_trips_to_300_dpi_page(tmp_path: Path) -> None:
    pdf = tmp_path / "bill.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 50), "DESCRIPTION AMOUNT")
    document.save(pdf)
    document.close()
    result = render_pdf_region(
        pdf,
        tmp_path / "region.png",
        1,
        (100, 150, 600, 900),
    )
    page_points = ((100.0, 150.0), (600.0, 900.0), (250.0, 300.0))
    crop_points = apply_matrix(result.transform.forward_matrix, page_points)
    assert crop_points[0] == pytest.approx((0, 0))
    restored = apply_matrix(result.transform.inverse_matrix, crop_points)
    for expected, actual in zip(page_points, restored, strict=True):
        assert actual == pytest.approx(expected)
    child_endpoints = (
        (0.0, 0.0),
        (result.transform.derived_width - 1.0, result.transform.derived_height - 1.0),
    )
    parent_endpoints = apply_matrix(result.transform.inverse_matrix, child_endpoints)
    assert parent_endpoints[0] == pytest.approx((100.0, 150.0))
    assert parent_endpoints[1] == pytest.approx((599.0, 899.0))


def test_all_right_angle_orientations_round_trip_and_swap_dimensions() -> None:
    points = ((0.0, 0.0), (199.0, 0.0), (199.0, 299.0), (0.0, 299.0))
    cases = (
        (0, (200, 300)),
        (90, (300, 200)),
        (180, (200, 300)),
        (270, (300, 200)),
    )
    for degrees, expected_size in cases:
        matrix, width, height = right_angle_rotation(degrees, 200, 300)
        assert (width, height) == expected_size
        restored = apply_matrix(invert(matrix), apply_matrix(matrix, points))
        assert restored == points


def test_normalization_applies_page_orientation_before_deskew(tmp_path: Path) -> None:
    source = tmp_path / "portrait.png"
    image = np.full((300, 200, 3), 255, dtype=np.uint8)
    cv2.circle(image, (20, 20), 5, (0, 0, 0), -1)
    assert cv2.imwrite(str(source), image)
    result = normalize_page(
        source,
        tmp_path / "landscape.png",
        page_number=1,
        dpi=300,
        orientation_correction_degrees=90,
        rotation_correction_degrees=2.0,
    )
    assert result.transform.derived_width == 300
    assert result.transform.derived_height == 200
    assert result.transform.operations == ("orientation:90", "rotate:2.000000")


def test_page_quadrilateral_detector_is_conservative_and_reports_keystone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "photo.png"
    image = np.zeros((1000, 800, 3), dtype=np.uint8)
    physical_page = np.asarray(((35, 45), (770, 80), (720, 960), (60, 915)))
    cv2.fillConvexPoly(image, physical_page, (255, 255, 255))
    cv2.polylines(image, (physical_page,), True, (20, 20, 20), 5)
    assert cv2.imwrite(str(path), image)

    points, confidence, keystone = detect_page_quadrilateral(path)

    assert points is not None
    assert confidence >= 0.85
    assert keystone >= 0.02


def test_table_quadrilateral_detector_rectifies_divergent_ledger_rules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "perspective-table.png"
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    top_left = (0, 150)
    top_right = (1199, 200)
    bottom_left = (0, 700)
    bottom_right = (1199, 660)
    cv2.line(image, top_left, top_right, (0, 0, 0), 5)
    cv2.line(image, bottom_left, bottom_right, (0, 0, 0), 5)
    cv2.line(image, top_left, bottom_left, (0, 0, 0), 5)
    cv2.line(image, top_right, bottom_right, (0, 0, 0), 5)
    assert cv2.imwrite(str(source), image)

    points, confidence, divergence = detect_table_quadrilateral(source)

    assert points is not None
    assert confidence >= 0.80
    assert divergence >= 0.75
    result = normalize_quadrilateral_region(
        source,
        tmp_path / "rectified-table.png",
        1,
        points,
    )
    restored = apply_matrix(
        result.transform.inverse_matrix,
        apply_matrix(result.transform.forward_matrix, points),
    )
    for expected, actual in zip(points, restored, strict=True):
        assert actual == pytest.approx(expected)
    assert result.transform.operations == ("perspective_crop",)


def test_camera_candidate_composes_400dpi_transform_back_to_raw_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=144, height=192)
    page.insert_text((12, 30), "DESCRIPTION AMOUNT")
    document.save(source)
    document.close()

    artifact_root = tmp_path / "artifacts"
    raw_path = artifact_root / "pages" / "raw.png"
    raw_path.parent.mkdir(parents=True)
    raw = np.full((800, 600, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(raw_path), raw)
    raw_hash = sha256_file(raw_path)
    page_asset = PageAsset(
        document_sha256=sha256_file(source),
        page_number=1,
        artifact_sha256=raw_hash,
        relative_path="raw.png",
        width=600,
        height=800,
        dpi=300,
        renderer="test",
        renderer_version="1",
    )
    quality = PageQuality(
        page_number=1,
        artifact_sha256=raw_hash,
        width=600,
        height=800,
        dpi=300,
        mean_luminance=246,
        contrast_stddev=40,
        laplacian_variance=100,
        edge_density=0.05,
        estimated_skew_degrees=0,
        flags=(PageQualityFlag.OVEREXPOSED,),
    )

    candidates = prepare_page_candidates(
        source_pdf=source,
        raw_path=raw_path,
        raw_relative_path="pages/raw.png",
        page=page_asset,
        quality=quality,
        artifact_root=artifact_root,
        orientation_degrees=0,
        orientation_confidence=1,
    )

    camera = next(item for item in candidates if item.contract.variant.value == "camera_400")
    raw_points = ((0.0, 0.0), (600.0, 800.0), (120.0, 240.0))
    restored = apply_matrix(
        camera.contract.transform.inverse_matrix,
        apply_matrix(camera.contract.transform.forward_matrix, raw_points),
    )
    for expected, actual in zip(raw_points, restored, strict=True):
        assert actual == pytest.approx(expected)
    assert camera.path.exists()
    assert not (camera.path.parent / "page-400.png").exists()
    assert not (camera.path.parent / "geometry-400.png").exists()
