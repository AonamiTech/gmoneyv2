import pytest
from hypothesis import given
from hypothesis import strategies as st
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
    TokenManifestEntryV2,
    canonical_json,
    canonical_sha256,
)
from gmoney.geometry.artifacts import (
    artifact_chain,
    map_points_to_source,
    max_round_trip_error,
    round_trip_within_tolerance,
)
from gmoney.geometry.transform import apply_matrix, invert, right_angle_rotation


def _artifact(
    kind: ArtifactKind,
    image: str,
    *,
    parent: str | None = None,
    mapping: IdentityMapping | HomographyMapping | None = None,
    width: int = 100,
    height: int = 100,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=kind,
        image_sha256=image * 64,
        artifact_relative_path=f"artifacts/{image}.png",
        width=width,
        height=height,
        parent_artifact_id=parent,
        producer="test",
        producer_version="1",
        configuration_sha256="c" * 64,
        child_to_parent_mapping=mapping or IdentityMapping(),
    )


def test_artifact_and_mapping_ids_are_canonical_and_deterministic() -> None:
    mapping = HomographyMapping(child_to_parent_matrix=((1, 0, 5), (0, 1, 7), (0, 0, 1)))
    assert mapping.mapping_sha256 == canonical_sha256(
        {
            "mapping_type": "HOMOGRAPHY",
            "child_to_parent_matrix": mapping.child_to_parent_matrix,
        }
    )
    first = _artifact(ArtifactKind.SOURCE_RAW, "a")
    second = _artifact(ArtifactKind.SOURCE_RAW, "a")
    assert first.artifact_id == second.artifact_id


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"not_a_number": float("nan")})


def test_source_and_oriented_lineage_rules_are_strict() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a", width=120, height=80)
    oriented = _artifact(
        ArtifactKind.ORIENTED_RAW,
        "b",
        parent=source.artifact_id,
        width=source.width,
        height=source.height,
    )
    manifest = ArtifactManifest(artifacts=(source, oriented))
    assert artifact_chain(manifest, oriented.artifact_id) == (oriented, source)

    rotated = _artifact(
        ArtifactKind.ORIENTED_RAW,
        "c",
        parent=source.artifact_id,
        width=source.height,
        height=source.width,
        mapping=HomographyMapping(
            child_to_parent_matrix=invert(
                right_angle_rotation(90, source.width, source.height)[0]
            )
        ),
    )
    ArtifactManifest(artifacts=(source, rotated))

    wrong_dimensions = _artifact(
        ArtifactKind.ORIENTED_RAW,
        "e",
        parent=source.artifact_id,
        width=79,
        height=120,
        mapping=HomographyMapping(
            child_to_parent_matrix=invert(
                right_angle_rotation(90, source.width, source.height)[0]
            )
        ),
    )
    with pytest.raises(ValidationError, match="90/180/270"):
        ArtifactManifest(artifacts=(source, wrong_dimensions))

    with pytest.raises(ValidationError, match="SOURCE_RAW artifacts must be graph roots"):
        _artifact(ArtifactKind.SOURCE_RAW, "d", parent=source.artifact_id)


@pytest.mark.parametrize("degrees", (90, 270))
def test_rectangular_oriented_inverse_maps_pixel_corners_exactly(degrees: int) -> None:
    width, height = 120, 80
    forward, child_width, child_height = right_angle_rotation(degrees, width, height)
    source_corners = ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
    oriented_corners = apply_matrix(forward, source_corners)
    restored = apply_matrix(invert(forward), oriented_corners)
    assert restored == source_corners
    source = _artifact(ArtifactKind.SOURCE_RAW, "a", width=width, height=height)
    oriented = _artifact(
        ArtifactKind.ORIENTED_RAW,
        "b",
        parent=source.artifact_id,
        width=child_width,
        height=child_height,
        mapping=HomographyMapping(child_to_parent_matrix=invert(forward)),
    )
    ArtifactManifest(artifacts=(source, oriented))


def test_graph_traversal_maps_child_coordinates_to_source() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    crop = _artifact(
        ArtifactKind.TABLE_CROP,
        "b",
        parent=source.artifact_id,
        mapping=HomographyMapping(child_to_parent_matrix=((1, 0, 10), (0, 1, 20), (0, 0, 1))),
        width=20,
        height=20,
    )
    manifest = ArtifactManifest(artifacts=(source, crop))
    assert map_points_to_source(manifest, crop.artifact_id, ((0, 0), (2, 3))) == (
        (10.0, 20.0),
        (12.0, 23.0),
    )


def test_homography_requires_finite_invertible_orientation_preserving_matrix() -> None:
    with pytest.raises(ValidationError, match="invertible"):
        HomographyMapping(child_to_parent_matrix=((1, 0, 0), (0, 0, 0), (0, 0, 1)))
    with pytest.raises(ValidationError, match="orientation"):
        HomographyMapping(child_to_parent_matrix=((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValidationError, match="Out of range float values"):
        HomographyMapping(child_to_parent_matrix=((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1)))


def test_round_trip_helper_enforces_two_pixel_contract() -> None:
    matrix = ((1.0, 0, 10), (0, 1.0, 20), (0, 0, 1.0))
    points = ((0.0, 0.0), (50.0, 75.0))
    assert max_round_trip_error(matrix, points) < 1e-9
    assert round_trip_within_tolerance(matrix, points)


@given(x=st.floats(min_value=0, max_value=1000), y=st.floats(min_value=0, max_value=1000))
def test_translation_round_trip_property(x: float, y: float) -> None:
    matrix = ((1.0, 0, 13.5), (0, 1.0, 8.25), (0, 0, 1.0))
    assert round_trip_within_tolerance(matrix, ((x, y),), tolerance_px=1e-8)


def test_v6_rejects_dense_mapping_but_schema_can_parse_reserved_tag() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    dense_payload = {
        "mapping_type": "DENSE_BACKWARD_GRID",
        "grid_relative_path": "grids/a.npy",
        "grid_sha256": "d" * 64,
        "grid_dtype": "float32",
        "grid_shape": (10, 10, 2),
        "child_width": 100,
        "child_height": 100,
        "parent_width": 100,
        "parent_height": 100,
        "coordinate_domain": "normalized_minus_one_to_one",
        "interpolation": "bilinear",
        "align_corners": True,
        "padding_mode": "border",
    }
    dense = DenseBackwardGridMapping(
        **dense_payload,
        mapping_sha256=canonical_sha256(dense_payload),
    )
    child = _artifact(
        ArtifactKind.PROJECTIVE,
        "b",
        parent=source.artifact_id,
        mapping=dense,
    )
    manifest = ArtifactManifest(artifacts=(source, child))
    with pytest.raises(ValidationError, match="reserved for M3"):
        ExtractionResultV6(
            document_id="doc",
            source_sha256="e" * 64,
            source_name="bill.pdf",
            pages=1,
            artifact_manifest=manifest,
            page_artifacts=(PageArtifact(artifact=source, page_number=1, dpi=300),),
        )


def test_v6_envelope_binds_page_and_canonical_table_artifacts() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    table = _artifact(
        ArtifactKind.TABLE_CROP,
        "b",
        parent=source.artifact_id,
        mapping=HomographyMapping(child_to_parent_matrix=((1, 0, 10), (0, 1, 10), (0, 0, 1))),
        width=50,
        height=50,
    )
    manifest = ArtifactManifest(artifacts=(source, table))
    assert manifest.manifest_version == "artifact_manifest_v1"
    polygon = Polygon(
        points=(Point(x=0, y=0), Point(x=40, y=0), Point(x=40, y=40), Point(x=0, y=40))
    )
    envelope = ExtractionResultV6(
        document_id="doc",
        source_sha256="e" * 64,
        source_name="bill.pdf",
        pages=1,
        artifact_manifest=manifest,
        page_artifacts=(PageArtifact(artifact=source, page_number=1, dpi=300),),
        canonical_table_artifacts=(
            CanonicalTableArtifact(
                artifact=table,
                logical_table_id="p1-t1",
                page_artifact_id=source.artifact_id,
                crop_polygon_in_page_artifact=polygon,
                crop_polygon_in_source_raw=polygon,
            ),
        ),
        evidence=(
            EvidenceRefV2(
                artifact_id=table.artifact_id,
                artifact_sha256=table.image_sha256,
                canonical_polygon=polygon,
                source_page_polygon=polygon,
                source_page_number=1,
                source_page_artifact_id=source.artifact_id,
                extractor="test",
                model_name="test-model",
                model_version="1",
                recognition_variant="canonical",
            ),
        ),
    )
    assert envelope.output_version == "offline_accuracy_spine_v6"


def test_v6_rejects_embedded_artifact_drift_and_table_parent_mismatch() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    other_source = _artifact(ArtifactKind.SOURCE_RAW, "c")
    table = _artifact(
        ArtifactKind.TABLE_CROP,
        "b",
        parent=source.artifact_id,
        mapping=HomographyMapping(child_to_parent_matrix=((1, 0, 10), (0, 1, 10), (0, 0, 1))),
        width=50,
        height=50,
    )
    manifest = ArtifactManifest(artifacts=(source, other_source, table))
    polygon = Polygon(
        points=(Point(x=0, y=0), Point(x=40, y=0), Point(x=40, y=40), Point(x=0, y=40))
    )
    table_record = CanonicalTableArtifact(
        artifact=table,
        logical_table_id="p1-t1",
        page_artifact_id=other_source.artifact_id,
        crop_polygon_in_page_artifact=polygon,
        crop_polygon_in_source_raw=polygon,
    )
    with pytest.raises(ValidationError, match="parent differs"):
        ExtractionResultV6(
            document_id="doc",
            source_sha256="e" * 64,
            source_name="bill.pdf",
            pages=1,
            artifact_manifest=manifest,
            page_artifacts=(
                PageArtifact(artifact=source, page_number=1, dpi=300),
                PageArtifact(artifact=other_source, page_number=2, dpi=300),
            ),
            canonical_table_artifacts=(table_record,),
        )

    drifted_source = source.model_copy(update={"producer_version": "tampered"})
    drifted_page = PageArtifact.model_construct(
        artifact=drifted_source,
        page_number=1,
        dpi=300,
        quality_metrics={},
        route_reasons=(),
        transform_metrics={},
    )
    with pytest.raises(ValidationError, match="embedded page artifact differs"):
        ExtractionResultV6(
            document_id="doc",
            source_sha256="e" * 64,
            source_name="bill.pdf",
            pages=1,
            artifact_manifest=ArtifactManifest(artifacts=(source,)),
            page_artifacts=(drifted_page,),
        )


def test_v2_tokens_are_canonical_first() -> None:
    token = TokenManifestEntryV2(
        token_id="t1",
        page_number=1,
        text="Amount",
        canonical_polygon=Polygon(points=(Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=5))),
        source_page_polygon=Polygon(
            points=(Point(x=20, y=30), Point(x=30, y=30), Point(x=30, y=35))
        ),
        source_page_artifact_id="a" * 64,
        artifact_id="b" * 64,
        artifact_sha256="c" * 64,
        artifact_relative_path="tables/t1.png",
        confidence=0.9,
    )
    assert token.canonical_polygon.points[0].x == 0
    with pytest.raises(ValidationError):
        TokenManifestEntryV2.model_validate(
            {
                **token.model_dump(mode="json"),
                "polygon": token.canonical_polygon.model_dump(mode="json"),
            }
        )
