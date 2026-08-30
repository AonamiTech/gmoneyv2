from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.evidence import (
    PageAsset,
    PageQuality,
    PageQualityFlag,
    PreprocessingCandidate,
    PreprocessingVariant,
    TransformChain,
)
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.crop import render_pdf_region
from gmoney.geometry.normalize import normalize_page
from gmoney.geometry.quality import assess_quality
from gmoney.geometry.transform import apply_matrix, compose, identity, invert

CAMERA_PREPROCESSING_VERSION = "camera_preprocessing_v1"
ORIENTATION_CONFIDENCE_THRESHOLD = 0.90
MIN_DESKEW_DEGREES = 0.75
MAX_DESKEW_DEGREES = 15.0
PERSPECTIVE_CONFIDENCE_THRESHOLD = 0.85
MIN_KEYSTONE_SCORE = 0.02


@dataclass(frozen=True)
class PreparedPageCandidate:
    path: Path
    contract: PreprocessingCandidate


def _identity_transform(page: PageAsset) -> TransformChain:
    matrix = identity()
    return TransformChain(
        page_number=page.page_number,
        source_width=page.width,
        source_height=page.height,
        derived_width=page.width,
        derived_height=page.height,
        forward_matrix=matrix,
        inverse_matrix=matrix,
        operations=(),
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def _corner_angle_score(points: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-9:
            return 0.0
        cosine = abs(float(np.dot(previous, following) / denominator))
        scores.append(max(0.0, 1.0 - cosine))
    return float(sum(scores) / len(scores))


def detect_page_quadrilateral(
    image_path: Path,
) -> tuple[tuple[tuple[float, float], ...] | None, float, float]:
    """Return a conservative physical-page quad in source-image coordinates."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read page image: {image_path}")
    height, width = image.shape
    scale = min(1.0, 1600.0 / max(height, width))
    resized = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(resized.shape[0] * resized.shape[1])
    best: tuple[np.ndarray, float, float] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        area_ratio = float(cv2.contourArea(polygon)) / image_area
        if not 0.50 <= area_ratio <= 0.995:
            continue
        ordered = _order_quad(polygon.reshape(4, 2).astype(np.float32))
        angle_score = _corner_angle_score(ordered)
        area_score = min(1.0, max(0.0, (area_ratio - 0.50) / 0.30))
        confidence = 0.65 * area_score + 0.35 * angle_score
        top = float(np.linalg.norm(ordered[1] - ordered[0]))
        bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
        left = float(np.linalg.norm(ordered[3] - ordered[0]))
        right = float(np.linalg.norm(ordered[2] - ordered[1]))
        keystone = max(
            abs(top - bottom) / max(top, bottom, 1.0),
            abs(left - right) / max(left, right, 1.0),
        )
        if best is None or confidence > best[1]:
            best = (ordered, confidence, keystone)
    if best is None:
        return None, 0.0, 0.0
    points, confidence, keystone = best
    restored = tuple((float(x / scale), float(y / scale)) for x, y in points)
    return restored, confidence, keystone


def _enhance_camera_page(source: Path, output: Path) -> str:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read page image: {source}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    enhanced = cv2.cvtColor(
        cv2.merge((luminance, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), enhanced):
        raise RuntimeError(f"failed to write camera-enhanced page: {temporary}")
    temporary.replace(output)
    return sha256_file(output)


def _combined_transform(
    source_to_intermediate: TransformChain,
    intermediate_to_derived: TransformChain,
    *,
    operations: tuple[str, ...] = (),
) -> TransformChain:
    forward = compose(
        source_to_intermediate.forward_matrix,
        intermediate_to_derived.forward_matrix,
    )
    return TransformChain(
        page_number=source_to_intermediate.page_number,
        source_width=source_to_intermediate.source_width,
        source_height=source_to_intermediate.source_height,
        derived_width=intermediate_to_derived.derived_width,
        derived_height=intermediate_to_derived.derived_height,
        forward_matrix=forward,
        inverse_matrix=invert(forward),
        operations=(
            *source_to_intermediate.operations,
            *intermediate_to_derived.operations,
            *operations,
        ),
    )


def _raw_candidate(
    page: PageAsset,
    quality: PageQuality,
    raw_path: Path,
    raw_relative_path: str,
) -> PreparedPageCandidate:
    return PreparedPageCandidate(
        path=raw_path,
        contract=PreprocessingCandidate(
            variant=PreprocessingVariant.RAW,
            artifact_sha256=page.artifact_sha256,
            artifact_relative_path=raw_relative_path,
            width=page.width,
            height=page.height,
            dpi=page.dpi,
            transform=_identity_transform(page),
            quality=quality,
            route_reasons=(),
        ),
    )


def prepare_page_candidates(
    *,
    source_pdf: Path,
    raw_path: Path,
    raw_relative_path: str,
    page: PageAsset,
    quality: PageQuality,
    artifact_root: Path,
    orientation_degrees: int,
    orientation_confidence: float,
) -> tuple[PreparedPageCandidate, ...]:
    """Create a bounded, content-addressed set of reversible page candidates."""

    candidates = [_raw_candidate(page, quality, raw_path, raw_relative_path)]
    route_reasons: list[str] = []
    orientation = (
        orientation_degrees if orientation_confidence >= ORIENTATION_CONFIDENCE_THRESHOLD else 0
    )
    if orientation:
        route_reasons.append(f"orientation:{orientation}")
    skew = quality.estimated_skew_degrees
    rotation = -skew if MIN_DESKEW_DEGREES <= abs(skew) <= MAX_DESKEW_DEGREES else 0.0
    if rotation:
        route_reasons.append("skew")
    quad, perspective_confidence, keystone = detect_page_quadrilateral(raw_path)
    perspective = (
        quad
        if perspective_confidence >= PERSPECTIVE_CONFIDENCE_THRESHOLD
        and keystone >= MIN_KEYSTONE_SCORE
        else None
    )
    if perspective is not None:
        route_reasons.append("perspective")

    destination = (
        artifact_root
        / "preprocessing"
        / page.artifact_sha256
        / CAMERA_PREPROCESSING_VERSION
        / f"page-{page.page_number:04d}"
    )
    geometry_result = None
    if route_reasons:
        geometry_result = normalize_page(
            raw_path,
            destination / "geometry-300.png",
            page_number=page.page_number,
            dpi=page.dpi,
            orientation_correction_degrees=orientation,
            rotation_correction_degrees=rotation,
            perspective_source=perspective,
        )
        geometry_quality = assess_quality(
            geometry_result.output_path,
            page.page_number,
            page.dpi,
        )
        candidates.append(
            PreparedPageCandidate(
                path=geometry_result.output_path,
                contract=PreprocessingCandidate(
                    variant=PreprocessingVariant.GEOMETRY_300,
                    artifact_sha256=geometry_result.artifact_sha256,
                    artifact_relative_path=str(
                        geometry_result.output_path.resolve().relative_to(artifact_root.resolve())
                    ),
                    width=geometry_result.transform.derived_width,
                    height=geometry_result.transform.derived_height,
                    dpi=page.dpi,
                    transform=geometry_result.transform,
                    quality=geometry_quality,
                    route_reasons=tuple(route_reasons),
                ),
            )
        )

    flags = set(quality.flags)
    needs_camera_variant = bool(
        PageQualityFlag.BLUR in flags
        or PageQualityFlag.LOW_CONTRAST in flags
        or PageQualityFlag.UNDEREXPOSED in flags
        or PageQualityFlag.OVEREXPOSED in flags
        or PageQualityFlag.LOW_EDGE_DENSITY in flags
    )
    if needs_camera_variant:
        high_resolution = render_pdf_region(
            source_pdf,
            destination / "page-400.png",
            page.page_number,
            (0, 0, page.width, page.height),
            source_dpi=page.dpi,
            output_dpi=400,
            source_size=(page.width, page.height),
        )
        scaled_perspective = (
            apply_matrix(high_resolution.transform.forward_matrix, perspective)
            if perspective is not None
            else None
        )
        normalized_400 = normalize_page(
            high_resolution.output_path,
            destination / "geometry-400.png",
            page_number=page.page_number,
            dpi=400,
            orientation_correction_degrees=orientation,
            rotation_correction_degrees=rotation,
            perspective_source=scaled_perspective,
        )
        camera_path = destination / "camera-400.png"
        camera_sha256 = _enhance_camera_page(normalized_400.output_path, camera_path)
        camera_transform = _combined_transform(
            high_resolution.transform,
            normalized_400.transform,
            operations=("clahe:2.0,8x8", "unsharp:1.0,0.5"),
        )
        camera_quality = assess_quality(camera_path, page.page_number, 400)
        high_resolution.output_path.unlink(missing_ok=True)
        normalized_400.output_path.unlink(missing_ok=True)
        candidates.append(
            PreparedPageCandidate(
                path=camera_path,
                contract=PreprocessingCandidate(
                    variant=PreprocessingVariant.CAMERA_400,
                    artifact_sha256=camera_sha256,
                    artifact_relative_path=str(
                        camera_path.resolve().relative_to(artifact_root.resolve())
                    ),
                    width=camera_transform.derived_width,
                    height=camera_transform.derived_height,
                    dpi=400,
                    transform=camera_transform,
                    quality=camera_quality,
                    route_reasons=tuple(
                        dict.fromkeys(
                            (
                                *route_reasons,
                                *(
                                    flag.value
                                    for flag in quality.flags
                                    if flag is not PageQualityFlag.ORIENTATION_UNVERIFIED
                                ),
                                "render_dpi:400",
                                "clahe",
                                "unsharp",
                            )
                        )
                    ),
                ),
            )
        )
    return tuple(candidates)
