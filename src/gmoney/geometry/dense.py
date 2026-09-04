"""Deterministic storage, loading, and diagnostics for V6 dense mappings."""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from gmoney.contracts.v6 import DenseBackwardGridMapping

DENSE_GRID_MEMBER: Final = "grid.npy"
MAX_DENSE_GRID_BYTES: Final = 128 * 1024 * 1024
_NPY_HEADER_ALLOWANCE: Final = 1024 * 1024


class DenseGridError(ValueError):
    """A fail-closed dense-grid error with a stable publication code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StoredDenseGrid:
    relative_path: str
    sha256: str
    shape: tuple[int, int, int]
    size_bytes: int


@dataclass(frozen=True)
class DenseGridDiagnostics:
    mean_displacement_px: float
    max_displacement_px: float
    local_scale_p05: float
    local_scale_p50: float
    local_scale_p95: float
    anisotropy_p95: float
    jacobian_determinant_p05: float
    jacobian_determinant_p50: float
    jacobian_determinant_p95: float
    foldover_count: int
    out_of_bounds_rate: float


def _relative_path(value: str | Path) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "\\" in str(value)
        or relative.suffix != ".npz"
    ):
        raise DenseGridError(
            "v6_dense_grid_path_invalid",
            "dense-grid path must be a contained relative .npz path",
        )
    return relative


def _require_directory(path: Path, code: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise DenseGridError(code, "dense-grid directory is unavailable") from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise DenseGridError(code, "dense-grid directory may not be a symlink")


def _contained_file(root: Path, relative: Path) -> Path:
    _require_directory(root, "v6_dense_grid_path_invalid")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        _require_directory(current, "v6_dense_grid_path_invalid")
    target = root / relative
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError as error:
        raise DenseGridError("v6_dense_grid_missing", "dense-grid file is missing") from error
    except OSError as error:
        raise DenseGridError(
            "v6_dense_grid_path_invalid", "dense-grid file is unavailable"
        ) from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise DenseGridError(
            "v6_dense_grid_path_invalid",
            "dense-grid path must identify a regular non-symlink file",
        )
    return target


def _canonical_array(grid: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(grid, dtype=np.dtype("<f4"), order="C")
    if array.ndim != 3 or array.shape[2] != 2 or min(array.shape[:2]) < 2:
        raise DenseGridError(
            "v6_dense_grid_metadata_mismatch",
            "dense grid must have shape (height>=2, width>=2, 2)",
        )
    if array.nbytes > MAX_DENSE_GRID_BYTES:
        raise DenseGridError("v6_dense_grid_archive_invalid", "dense grid exceeds size limit")
    if not np.isfinite(array).all():
        raise DenseGridError("v6_dense_grid_non_finite", "dense grid contains non-finite values")
    return np.ascontiguousarray(array)


def _encoded_grid(grid: NDArray[np.generic]) -> tuple[bytes, NDArray[np.float32]]:
    array = _canonical_array(grid)
    stream = io.BytesIO()
    np.savez_compressed(stream, grid=array)
    return stream.getvalue(), array


def write_dense_grid(
    artifact_root: Path,
    relative_path: str | Path,
    grid: NDArray[np.generic],
) -> StoredDenseGrid:
    """Write a deterministic, single-member dense-grid archive atomically."""

    relative = _relative_path(relative_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _require_directory(artifact_root, "v6_dense_grid_path_invalid")
    current = artifact_root
    for part in relative.parts[:-1]:
        current /= part
        if current.exists():
            _require_directory(current, "v6_dense_grid_path_invalid")
        else:
            current.mkdir()
            _require_directory(current, "v6_dense_grid_path_invalid")

    encoded, array = _encoded_grid(grid)
    digest = hashlib.sha256(encoded).hexdigest()
    target = artifact_root / relative
    if target.exists() or target.is_symlink():
        existing = _contained_file(artifact_root, relative)
        if existing.read_bytes() != encoded:
            raise DenseGridError(
                "v6_dense_grid_digest_mismatch",
                "existing dense-grid path contains different bytes",
            )
        return StoredDenseGrid(relative.as_posix(), digest, tuple(array.shape), len(encoded))

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return StoredDenseGrid(relative.as_posix(), digest, tuple(array.shape), len(encoded))


def _validate_loaded_grid(
    mapping: DenseBackwardGridMapping,
    array: NDArray[np.generic],
) -> NDArray[np.float32]:
    if array.dtype != np.dtype("<f4") or not array.flags.c_contiguous:
        raise DenseGridError(
            "v6_dense_grid_metadata_mismatch",
            "dense grid must be C-contiguous little-endian float32",
        )
    if tuple(array.shape) != mapping.grid_shape:
        raise DenseGridError(
            "v6_dense_grid_metadata_mismatch",
            "dense-grid shape differs from mapping metadata",
        )
    if array.nbytes > MAX_DENSE_GRID_BYTES:
        raise DenseGridError("v6_dense_grid_archive_invalid", "dense grid exceeds size limit")
    if not np.isfinite(array).all():
        raise DenseGridError("v6_dense_grid_non_finite", "dense grid contains non-finite values")
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def load_dense_grid(
    artifact_root: Path,
    mapping: DenseBackwardGridMapping,
) -> NDArray[np.float32]:
    """Load and strictly verify the archive referenced by a dense mapping."""

    relative = _relative_path(mapping.grid_relative_path)
    target = _contained_file(artifact_root, relative)
    try:
        content = target.read_bytes()
    except OSError as error:
        raise DenseGridError("v6_dense_grid_missing", "dense-grid file is unreadable") from error
    if len(content) > MAX_DENSE_GRID_BYTES + _NPY_HEADER_ALLOWANCE:
        raise DenseGridError(
            "v6_dense_grid_archive_invalid", "dense-grid archive exceeds size limit"
        )
    if hashlib.sha256(content).hexdigest() != mapping.grid_sha256:
        raise DenseGridError(
            "v6_dense_grid_digest_mismatch",
            "dense-grid bytes do not match mapping digest",
        )
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != DENSE_GRID_MEMBER:
                raise DenseGridError(
                    "v6_dense_grid_archive_invalid",
                    "dense-grid archive must contain only grid.npy",
                )
            if members[0].file_size > MAX_DENSE_GRID_BYTES + _NPY_HEADER_ALLOWANCE:
                raise DenseGridError(
                    "v6_dense_grid_archive_invalid",
                    "dense-grid member exceeds size limit",
                )
        with np.load(io.BytesIO(content), allow_pickle=False) as payload:
            if payload.files != ["grid"]:
                raise DenseGridError(
                    "v6_dense_grid_archive_invalid",
                    "dense-grid archive has an unexpected member",
                )
            array = payload["grid"]
    except DenseGridError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise DenseGridError(
            "v6_dense_grid_archive_invalid",
            "dense-grid archive cannot be decoded",
        ) from error
    return _validate_loaded_grid(mapping, array)


class DenseGridResolver:
    """Explicit artifact-root resolver that caches verified grids by mapping hash."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self._cache: dict[str, NDArray[np.float32]] = {}

    def __call__(self, mapping: DenseBackwardGridMapping) -> NDArray[np.float32]:
        cached = self._cache.get(mapping.mapping_sha256)
        if cached is None:
            cached = load_dense_grid(self.artifact_root, mapping)
            self._cache[mapping.mapping_sha256] = cached
        return _validate_loaded_grid(mapping, cached)


def dense_grid_parent_pixels(
    mapping: DenseBackwardGridMapping,
    grid: NDArray[np.generic],
) -> NDArray[np.float64]:
    array = _validate_loaded_grid(mapping, grid)
    output = array.astype(np.float64, copy=True)
    output[..., 0] = (output[..., 0] + 1.0) * 0.5 * (mapping.parent_width - 1)
    output[..., 1] = (output[..., 1] + 1.0) * 0.5 * (mapping.parent_height - 1)
    return output


def map_dense_points(
    mapping: DenseBackwardGridMapping,
    grid: NDArray[np.generic],
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    parent = dense_grid_parent_pixels(mapping, grid)
    grid_height, grid_width, _ = parent.shape
    mapped: list[tuple[float, float]] = []
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise DenseGridError(
                "v6_dense_grid_non_finite", "dense mapping received non-finite coordinates"
            )
        if x < 0 or y < 0 or x > mapping.child_width - 1 or y > mapping.child_height - 1:
            raise DenseGridError(
                "v6_dense_mapping_out_of_bounds",
                "dense mapping received a point outside the child artifact",
            )
        gx = x * (grid_width - 1) / (mapping.child_width - 1)
        gy = y * (grid_height - 1) / (mapping.child_height - 1)
        x0, y0 = int(math.floor(gx)), int(math.floor(gy))
        x1, y1 = min(x0 + 1, grid_width - 1), min(y0 + 1, grid_height - 1)
        wx, wy = gx - x0, gy - y0
        value = (
            parent[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + parent[y0, x1] * wx * (1.0 - wy)
            + parent[y1, x0] * (1.0 - wx) * wy
            + parent[y1, x1] * wx * wy
        )
        px, py = float(value[0]), float(value[1])
        if (
            px < 0
            or py < 0
            or px > mapping.parent_width - 1
            or py > mapping.parent_height - 1
        ):
            raise DenseGridError(
                "v6_dense_mapping_out_of_bounds",
                "dense mapping produced a point outside the parent artifact",
            )
        mapped.append((px, py))
    return tuple(mapped)


def _percentile(values: NDArray[np.float64], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def analyze_dense_grid(
    mapping: DenseBackwardGridMapping,
    grid: NDArray[np.generic],
) -> DenseGridDiagnostics:
    parent = dense_grid_parent_pixels(mapping, grid)
    grid_height, grid_width, _ = parent.shape
    child_x = np.linspace(0.0, mapping.child_width - 1, grid_width)
    child_y = np.linspace(0.0, mapping.child_height - 1, grid_height)
    expected_x, expected_y = np.meshgrid(child_x, child_y)
    displacement = np.hypot(parent[..., 0] - expected_x, parent[..., 1] - expected_y)
    spacing_x = (mapping.child_width - 1) / (grid_width - 1)
    spacing_y = (mapping.child_height - 1) / (grid_height - 1)
    top_left = parent[:-1, :-1]
    top_right = parent[:-1, 1:]
    bottom_left = parent[1:, :-1]
    bottom_right = parent[1:, 1:]
    bilinear = bottom_right - top_right - bottom_left + top_left
    cell_count = (grid_height - 1) * (grid_width - 1)
    determinants = np.empty(cell_count * 4, dtype=np.float64)
    local_scales = np.empty(cell_count * 4, dtype=np.float64)
    anisotropies = np.empty(cell_count * 4, dtype=np.float64)
    for index, (u, v) in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))):
        derivative_x = (top_right - top_left + v * bilinear) / spacing_x
        derivative_y = (bottom_left - top_left + u * bilinear) / spacing_y
        determinant = (
            derivative_x[..., 0] * derivative_y[..., 1]
            - derivative_x[..., 1] * derivative_y[..., 0]
        ).ravel()
        x_norm = np.sum(derivative_x * derivative_x, axis=-1).ravel()
        y_norm = np.sum(derivative_y * derivative_y, axis=-1).ravel()
        dot_product = np.sum(derivative_x * derivative_y, axis=-1).ravel()
        gram_trace = x_norm + y_norm
        discriminant = np.sqrt(
            np.maximum((x_norm - y_norm) ** 2 + 4 * dot_product**2, 0.0)
        )
        largest = np.sqrt(np.maximum((gram_trace + discriminant) * 0.5, 0.0))
        smallest = np.sqrt(np.maximum((gram_trace - discriminant) * 0.5, 0.0))
        start, end = index * cell_count, (index + 1) * cell_count
        determinants[start:end] = determinant
        local_scales[start:end] = np.sqrt(np.maximum(determinant, 0.0))
        anisotropies[start:end] = np.divide(
            largest,
            smallest,
            out=np.full_like(smallest, np.finfo(np.float64).max),
            where=smallest > 1e-12,
        )
    out_of_bounds = (
        (parent[..., 0] < 0)
        | (parent[..., 1] < 0)
        | (parent[..., 0] > mapping.parent_width - 1)
        | (parent[..., 1] > mapping.parent_height - 1)
    )
    return DenseGridDiagnostics(
        mean_displacement_px=float(np.mean(displacement)),
        max_displacement_px=float(np.max(displacement)),
        local_scale_p05=_percentile(local_scales, 5),
        local_scale_p50=_percentile(local_scales, 50),
        local_scale_p95=_percentile(local_scales, 95),
        anisotropy_p95=_percentile(anisotropies, 95),
        jacobian_determinant_p05=_percentile(determinants, 5),
        jacobian_determinant_p50=_percentile(determinants, 50),
        jacobian_determinant_p95=_percentile(determinants, 95),
        foldover_count=int(np.count_nonzero(determinants <= 0)),
        out_of_bounds_rate=float(np.mean(out_of_bounds)),
    )


def render_dense_overlay(
    mapping: DenseBackwardGridMapping,
    grid: NDArray[np.generic],
    canonical_polygons: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> bytes:
    """Render a deterministic parent-space mesh and polygon diagnostic PNG."""

    parent = dense_grid_parent_pixels(mapping, grid)
    canvas = np.full((mapping.parent_height, mapping.parent_width, 3), 255, dtype=np.uint8)

    def integer_points(points: NDArray[np.float64]) -> NDArray[np.int32]:
        rounded = np.rint(points).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, mapping.parent_width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, mapping.parent_height - 1)
        return rounded.reshape((-1, 1, 2))

    row_indexes = np.unique(
        np.linspace(0, parent.shape[0] - 1, min(33, parent.shape[0])).astype(int)
    )
    column_indexes = np.unique(
        np.linspace(0, parent.shape[1] - 1, min(33, parent.shape[1])).astype(int)
    )
    for index in row_indexes:
        cv2.polylines(canvas, [integer_points(parent[index])], False, (210, 210, 210), 1)
    for index in column_indexes:
        cv2.polylines(canvas, [integer_points(parent[:, index])], False, (210, 210, 210), 1)

    for polygon in canonical_polygons:
        child = np.asarray(polygon, dtype=np.float64)
        displayed = child.copy()
        displayed[:, 0] *= (mapping.parent_width - 1) / (mapping.child_width - 1)
        displayed[:, 1] *= (mapping.parent_height - 1) / (mapping.child_height - 1)
        mapped = np.asarray(map_dense_points(mapping, grid, tuple(polygon)), dtype=np.float64)
        cv2.polylines(canvas, [integer_points(displayed)], True, (255, 128, 0), 1)
        cv2.polylines(canvas, [integer_points(mapped)], True, (0, 0, 220), 1)

    ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise DenseGridError("v6_dense_grid_archive_invalid", "dense overlay encoding failed")
    return encoded.tobytes()


__all__ = [
    "DENSE_GRID_MEMBER",
    "MAX_DENSE_GRID_BYTES",
    "DenseGridDiagnostics",
    "DenseGridError",
    "DenseGridResolver",
    "StoredDenseGrid",
    "analyze_dense_grid",
    "dense_grid_parent_pixels",
    "load_dense_grid",
    "map_dense_points",
    "render_dense_overlay",
    "write_dense_grid",
]
