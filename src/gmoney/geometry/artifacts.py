"""Coordinate traversal and validation for the V6 artifact graph."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import hypot, isfinite

import numpy as np
from numpy.typing import NDArray

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.v6 import (
    ArtifactManifest,
    ArtifactRef,
    DenseBackwardGridMapping,
    HomographyMapping,
    IdentityMapping,
)
from gmoney.geometry.dense import DenseGridError, map_dense_points
from gmoney.geometry.transform import apply_matrix, invert

DenseGridLookup = Callable[[DenseBackwardGridMapping], NDArray[np.float32]]


def artifact_chain(manifest: ArtifactManifest, artifact_id: str) -> tuple[ArtifactRef, ...]:
    """Return an artifact and its parents, ending at SOURCE_RAW."""

    by_id = manifest.by_id()
    current = by_id.get(artifact_id)
    if current is None:
        raise ValueError(f"artifact {artifact_id!r} is not in the manifest")
    chain: list[ArtifactRef] = []
    seen: set[str] = set()
    while True:
        if current.artifact_id in seen:
            raise ValueError("artifact graph contains a cycle")
        seen.add(current.artifact_id)
        chain.append(current)
        if current.parent_artifact_id is None:
            return tuple(chain)
        parent = by_id.get(current.parent_artifact_id)
        if parent is None:
            raise ValueError("artifact parent is missing from manifest")
        current = parent


def map_points_to_source(
    manifest: ArtifactManifest,
    artifact_id: str,
    points: Iterable[tuple[float, float]],
    *,
    dense_grid_resolver: DenseGridLookup | None = None,
) -> tuple[tuple[float, float], ...]:
    """Map points from an artifact through child-to-parent mappings to SOURCE_RAW."""

    mapped = tuple((float(x), float(y)) for x, y in points)
    for artifact in artifact_chain(manifest, artifact_id)[:-1]:
        mapping = artifact.child_to_parent_mapping
        if isinstance(mapping, IdentityMapping):
            continue
        if isinstance(mapping, DenseBackwardGridMapping):
            if dense_grid_resolver is None:
                raise DenseGridError(
                    "v6_dense_grid_missing", "dense traversal requires an explicit grid resolver"
                )
            mapped = map_dense_points(mapping, dense_grid_resolver(mapping), mapped)
            continue
        assert isinstance(mapping, HomographyMapping)
        mapped = apply_matrix(mapping.child_to_parent_matrix, mapped)
    if any(not isfinite(value) for point in mapped for value in point):
        raise ValueError("artifact mapping produced non-finite coordinates")
    return mapped


def map_polygon_to_source(
    manifest: ArtifactManifest,
    artifact_id: str,
    polygon: Polygon,
    *,
    dense_grid_resolver: DenseGridLookup | None = None,
    chord_tolerance_px: float = 1.0,
    max_depth: int = 12,
    max_points: int = 4096,
) -> Polygon:
    if chord_tolerance_px <= 0 or max_depth < 0 or max_points < 3:
        raise ValueError("adaptive polygon limits are invalid")
    canonical = tuple((point.x, point.y) for point in polygon.points)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return map_points_to_source(
            manifest,
            artifact_id,
            (point,),
            dense_grid_resolver=dense_grid_resolver,
        )[0]

    projected: list[tuple[float, float]] = [project(canonical[0])]

    def edge(
        start: tuple[float, float],
        end: tuple[float, float],
        mapped_start: tuple[float, float],
        mapped_end: tuple[float, float],
        depth: int,
    ) -> list[tuple[float, float]]:
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        mapped_midpoint = project(midpoint)
        chord_midpoint = (
            (mapped_start[0] + mapped_end[0]) * 0.5,
            (mapped_start[1] + mapped_end[1]) * 0.5,
        )
        error = hypot(
            mapped_midpoint[0] - chord_midpoint[0],
            mapped_midpoint[1] - chord_midpoint[1],
        )
        if error <= chord_tolerance_px:
            return [mapped_end]
        if depth >= max_depth:
            raise DenseGridError(
                "v6_dense_polygon_budget_exhausted",
                "adaptive polygon mapping did not converge",
            )
        left = edge(start, midpoint, mapped_start, mapped_midpoint, depth + 1)
        right = edge(midpoint, end, mapped_midpoint, mapped_end, depth + 1)
        return [*left, *right]

    for index, start in enumerate(canonical):
        end = canonical[(index + 1) % len(canonical)]
        mapped_start = projected[-1]
        mapped_end = projected[0] if index == len(canonical) - 1 else project(end)
        additions = edge(start, end, mapped_start, mapped_end, 0)
        if index == len(canonical) - 1:
            additions = additions[:-1]
        projected.extend(additions)
        if len(projected) > max_points:
            raise DenseGridError(
                "v6_dense_polygon_budget_exhausted",
                "adaptive polygon mapping exceeded its point budget",
            )

    def signed_area(points: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> float:
        return 0.5 * sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
        )

    def orientation(
        a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> bool:
        values = (
            orientation(a, b, c),
            orientation(a, b, d),
            orientation(c, d, a),
            orientation(c, d, b),
        )
        if values[0] * values[1] < 0 and values[2] * values[3] < 0:
            return True

        def on_segment(
            start: tuple[float, float],
            point: tuple[float, float],
            end: tuple[float, float],
        ) -> bool:
            return (
                min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
                and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
            )

        epsilon = 1e-9
        return any(
            abs(value) <= epsilon and on_segment(start, point, end)
            for value, start, point, end in (
                (values[0], a, c, b),
                (values[1], a, d, b),
                (values[2], c, a, d),
                (values[3], c, b, d),
            )
        )

    canonical_area, projected_area = signed_area(canonical), signed_area(projected)
    if canonical_area == 0 or projected_area == 0 or canonical_area * projected_area < 0:
        raise DenseGridError(
            "v6_dense_polygon_orientation_invalid", "mapped polygon reverses orientation"
        )
    count = len(projected)
    for first in range(count):
        a, b = projected[first], projected[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count} or (second + 1) % count == first:
                continue
            c, d = projected[second], projected[(second + 1) % count]
            if intersects(a, b, c, d):
                raise DenseGridError(
                    "v6_dense_polygon_self_intersection", "mapped polygon self-intersects"
                )
    source = artifact_chain(manifest, artifact_id)[-1]
    if any(
        x < 0 or y < 0 or x > source.width - 1 or y > source.height - 1
        for x, y in projected
    ):
        raise DenseGridError(
            "v6_dense_mapping_out_of_bounds", "mapped polygon exceeds source bounds"
        )
    return Polygon(points=tuple(Point(x=x, y=y) for x, y in projected))


def max_round_trip_error(
    matrix: tuple[tuple[float, float, float], ...],
    points: Iterable[tuple[float, float]],
) -> float:
    """Return the greatest forward/inverse point error for a homography."""

    source = tuple(points)
    restored = apply_matrix(invert(matrix), apply_matrix(matrix, source))
    return max(
        (
            hypot(expected[0] - actual[0], expected[1] - actual[1])
            for expected, actual in zip(source, restored, strict=True)
        ),
        default=0.0,
    )


def round_trip_within_tolerance(
    matrix: tuple[tuple[float, float, float], ...],
    points: Iterable[tuple[float, float]],
    tolerance_px: float = 2.0,
) -> bool:
    if tolerance_px < 0:
        raise ValueError("round-trip tolerance must be non-negative")
    return max_round_trip_error(matrix, points) <= tolerance_px


__all__ = [
    "artifact_chain",
    "map_points_to_source",
    "map_polygon_to_source",
    "max_round_trip_error",
    "round_trip_within_tolerance",
]
