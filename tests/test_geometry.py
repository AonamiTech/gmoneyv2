from pathlib import Path

import cv2
import fitz
import numpy as np

from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.crop import crop_region
from gmoney.geometry.normalize import normalize_page
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
