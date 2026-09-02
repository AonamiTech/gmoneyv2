"""Coordinate traversal and validation for the V6 artifact graph."""

from __future__ import annotations

from collections.abc import Iterable
from math import hypot, isfinite

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.v6 import (
    ArtifactManifest,
    ArtifactRef,
    DenseBackwardGridMapping,
    HomographyMapping,
    IdentityMapping,
)
from gmoney.geometry.transform import apply_matrix, invert


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
) -> tuple[tuple[float, float], ...]:
    """Map points from an artifact through child-to-parent mappings to SOURCE_RAW."""

    mapped = tuple((float(x), float(y)) for x, y in points)
    for artifact in artifact_chain(manifest, artifact_id)[:-1]:
        mapping = artifact.child_to_parent_mapping
        if isinstance(mapping, IdentityMapping):
            continue
        if isinstance(mapping, DenseBackwardGridMapping):
            raise ValueError("dense backward-grid traversal is reserved for M3")
        assert isinstance(mapping, HomographyMapping)
        mapped = apply_matrix(mapping.child_to_parent_matrix, mapped)
    if any(not isfinite(value) for point in mapped for value in point):
        raise ValueError("artifact mapping produced non-finite coordinates")
    return mapped


def map_polygon_to_source(
    manifest: ArtifactManifest,
    artifact_id: str,
    polygon: Polygon,
) -> Polygon:
    points = map_points_to_source(
        manifest,
        artifact_id,
        ((point.x, point.y) for point in polygon.points),
    )
    if any(x < 0 or y < 0 for x, y in points):
        raise ValueError("artifact mapping produced out-of-bounds negative coordinates")
    return Polygon(points=tuple(Point(x=x, y=y) for x, y in points))


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
