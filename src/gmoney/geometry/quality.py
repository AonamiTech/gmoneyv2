from __future__ import annotations

from statistics import median

import cv2
import numpy as np

from gmoney.contracts.evidence import PageQuality, PageQualityFlag
from gmoney.evaluation.corpus import sha256_file


def estimate_skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minimum_length = max(40, min(gray.shape[:2]) // 5)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=50,
        minLineLength=minimum_length,
        maxLineGap=15,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 15:
            angles.append(angle)
    return float(median(angles)) if angles else 0.0


def assess_quality(image_path, page_number: int, dpi: int) -> PageQuality:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read page image: {image_path}")
    mean = float(image.mean())
    contrast = float(image.std())
    laplacian = float(cv2.Laplacian(image, cv2.CV_64F).var())
    edges = cv2.Canny(image, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    skew = estimate_skew(image)
    flags: list[PageQualityFlag] = [PageQualityFlag.ORIENTATION_UNVERIFIED]
    if laplacian < 50:
        flags.append(PageQualityFlag.BLUR)
    if contrast < 25:
        flags.append(PageQualityFlag.LOW_CONTRAST)
    if mean < 40:
        flags.append(PageQualityFlag.UNDEREXPOSED)
    if mean > 245:
        flags.append(PageQualityFlag.OVEREXPOSED)
    if edge_density < 0.005:
        flags.append(PageQualityFlag.LOW_EDGE_DENSITY)
    if abs(skew) > 0.5:
        flags.append(PageQualityFlag.SKEWED)
    height, width = image.shape
    return PageQuality(
        page_number=page_number,
        artifact_sha256=sha256_file(image_path),
        width=width,
        height=height,
        dpi=dpi,
        mean_luminance=mean,
        contrast_stddev=contrast,
        laplacian_variance=laplacian,
        edge_density=edge_density,
        estimated_skew_degrees=skew,
        flags=tuple(flags),
    )

