from __future__ import annotations

from collections.abc import Iterable
from math import cos, radians, sin

import numpy as np

Matrix = tuple[tuple[float, float, float], ...]


def _array(matrix: Matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("transform matrix must be 3x3")
    return value


def _matrix(value: np.ndarray) -> Matrix:
    return tuple(tuple(float(cell) for cell in row) for row in value)


def identity() -> Matrix:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def invert(matrix: Matrix) -> Matrix:
    return _matrix(np.linalg.inv(_array(matrix)))


def compose(*matrices: Matrix) -> Matrix:
    """Compose source-to-destination matrices in application order."""
    result = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        result = _array(matrix) @ result
    return _matrix(result)


def translation(x: float, y: float) -> Matrix:
    return ((1.0, 0.0, x), (0.0, 1.0, y), (0.0, 0.0, 1.0))


def rotation(degrees: float, center_x: float, center_y: float) -> Matrix:
    angle = radians(degrees)
    rotate_origin: Matrix = (
        (cos(angle), -sin(angle), 0.0),
        (sin(angle), cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return compose(
        translation(-center_x, -center_y),
        rotate_origin,
        translation(center_x, center_y),
    )


def right_angle_rotation(degrees: int, width: int, height: int) -> tuple[Matrix, int, int]:
    normalized = degrees % 360
    if normalized == 0:
        return identity(), width, height
    if normalized == 90:
        return (
            ((0.0, -1.0, height - 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            height,
            width,
        )
    if normalized == 180:
        return (
            ((-1.0, 0.0, width - 1.0), (0.0, -1.0, height - 1.0), (0.0, 0.0, 1.0)),
            width,
            height,
        )
    if normalized == 270:
        return (
            ((0.0, 1.0, 0.0), (-1.0, 0.0, width - 1.0), (0.0, 0.0, 1.0)),
            height,
            width,
        )
    raise ValueError("orientation correction must be 0, 90, 180, or 270 degrees")


def apply_matrix(
    matrix: Matrix,
    points: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    transform = _array(matrix)
    output: list[tuple[float, float]] = []
    for x, y in points:
        projected = transform @ np.array((x, y, 1.0), dtype=np.float64)
        if abs(projected[2]) < 1e-12:
            raise ValueError("point maps to infinity")
        output.append((float(projected[0] / projected[2]), float(projected[1] / projected[2])))
    return tuple(output)
