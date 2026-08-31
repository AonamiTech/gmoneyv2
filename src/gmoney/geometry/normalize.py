from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.evidence import TransformChain
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.transform import (
    Matrix,
    apply_matrix,
    compose,
    identity,
    invert,
    right_angle_rotation,
    rotation,
)


@dataclass(frozen=True)
class NormalizationResult:
    output_path: Path
    artifact_sha256: str
    transform: TransformChain


def normalize_quadrilateral_region(
    source: Path,
    output: Path,
    page_number: int,
    corners: tuple[tuple[float, float], ...],
) -> NormalizationResult:
    """Rectify one ordered source quadrilateral into a tightly cropped image."""
    if len(corners) != 4:
        raise ValueError("quadrilateral must contain four ordered corners")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read page image: {source}")
    height, width = image.shape[:2]
    source_points = np.asarray(corners, dtype=np.float32)
    top_left, top_right, bottom_right, bottom_left = source_points
    derived_width = max(
        1,
        round(
            max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
        ),
    )
    derived_height = max(
        1,
        round(
            max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )
        ),
    )
    destination_points = np.asarray(
        (
            (0, 0),
            (derived_width - 1, 0),
            (derived_width - 1, derived_height - 1),
            (0, derived_height - 1),
        ),
        dtype=np.float32,
    )
    perspective = cv2.getPerspectiveTransform(source_points, destination_points)
    forward: Matrix = tuple(tuple(float(value) for value in row) for row in perspective)
    normalized = cv2.warpPerspective(
        image,
        perspective,
        (derived_width, derived_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), normalized):
        raise RuntimeError(f"failed to write normalized region: {temporary}")
    temporary.replace(output)
    transform = TransformChain(
        page_number=page_number,
        source_width=width,
        source_height=height,
        derived_width=derived_width,
        derived_height=derived_height,
        forward_matrix=forward,
        inverse_matrix=invert(forward),
        operations=("perspective_crop",),
    )
    return NormalizationResult(
        output_path=output,
        artifact_sha256=sha256_file(output),
        transform=transform,
    )


def normalize_page(
    source: Path,
    output: Path,
    page_number: int,
    dpi: int,
    orientation_correction_degrees: int = 0,
    rotation_correction_degrees: float = 0.0,
    perspective_source: tuple[tuple[float, float], ...] | None = None,
) -> NormalizationResult:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read page image: {source}")
    height, width = image.shape[:2]
    forward: Matrix = identity()
    operations: list[str] = []
    normalized = image
    derived_width, derived_height = width, height

    orientation, derived_width, derived_height = right_angle_rotation(
        orientation_correction_degrees,
        width,
        height,
    )
    if orientation_correction_degrees % 360:
        forward = orientation
        normalized = cv2.warpPerspective(
            normalized,
            np.asarray(orientation, dtype=np.float64),
            (derived_width, derived_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        operations.append(f"orientation:{orientation_correction_degrees % 360}")

    if abs(rotation_correction_degrees) > 1e-9:
        deskew = rotation(
            rotation_correction_degrees,
            derived_width / 2,
            derived_height / 2,
        )
        forward = compose(forward, deskew)
        normalized = cv2.warpPerspective(
            normalized,
            np.asarray(deskew, dtype=np.float64),
            (derived_width, derived_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        operations.append(f"rotate:{rotation_correction_degrees:.6f}")

    if perspective_source is not None:
        if len(perspective_source) != 4:
            raise ValueError("perspective source must contain four ordered corners")
        # The public contract expresses perspective corners in the original
        # rendered-page coordinate space.  Orientation and fine rotation have
        # already changed the active image, so project the corners through the
        # accumulated transform before estimating the final homography.
        source_points = np.asarray(
            apply_matrix(forward, perspective_source),
            dtype=np.float32,
        )
        destination_points = np.asarray(
            (
                (0, 0),
                (derived_width - 1, 0),
                (derived_width - 1, derived_height - 1),
                (0, derived_height - 1),
            ),
            dtype=np.float32,
        )
        perspective = cv2.getPerspectiveTransform(source_points, destination_points)
        forward_array = perspective @ np.asarray(forward, dtype=np.float64)
        forward = tuple(tuple(float(value) for value in row) for row in forward_array)
        normalized = cv2.warpPerspective(
            normalized,
            perspective,
            (derived_width, derived_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        operations.append("perspective")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), normalized):
        raise RuntimeError(f"failed to write normalized page: {temporary}")
    temporary.replace(output)
    transform = TransformChain(
        page_number=page_number,
        source_width=width,
        source_height=height,
        derived_width=derived_width,
        derived_height=derived_height,
        forward_matrix=forward,
        inverse_matrix=invert(forward),
        operations=tuple(operations),
    )
    return NormalizationResult(
        output_path=output,
        artifact_sha256=sha256_file(output),
        transform=transform,
    )
