"""Strict V6 artifact, evidence, and envelope contracts.

The V5 contracts deliberately remain unchanged.  V6 makes the image used by an
adapter a first-class, content-addressed node in an immutable coordinate graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AliasChoices, Field, TypeAdapter, field_validator, model_validator

from gmoney.contracts.common import ContractModel
from gmoney.contracts.evidence import PageAsset, PagePreprocessingRecord, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    DocumentTotal,
    ExtractionDiagnostic,
    ProviderUsage,
    RawTotalCandidate,
    ReceiptDuplicatePair,
    RecoveryMetadata,
    SourceCell,
    SourceColumn,
    SourceRow,
    SourceTable,
    SuppressedSourceTable,
)

SHA256_PATTERN = r"^[a-f0-9]{64}$"
Matrix = tuple[tuple[float, float, float], ...]


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for content hashes."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite_matrix(matrix: Matrix) -> None:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("homography must be a 3x3 matrix")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("homography must contain only finite values")


def _determinant(matrix: Matrix) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_homography(matrix: Matrix) -> None:
    _finite_matrix(matrix)
    determinant = _determinant(matrix)
    if abs(determinant) <= 1e-12:
        raise ValueError("homography must be invertible")
    if determinant < 0:
        raise ValueError("homography must preserve orientation")


class ArtifactKind(StrEnum):
    SOURCE_RAW = "SOURCE_RAW"
    ORIENTED_RAW = "ORIENTED_RAW"
    PROJECTIVE = "PROJECTIVE"
    PROJECTIVE_ENHANCED = "PROJECTIVE_ENHANCED"
    UVDOC = "UVDOC"
    UVDOC_ENHANCED = "UVDOC_ENHANCED"
    TABLE_CROP = "TABLE_CROP"
    CELL_CROP = "CELL_CROP"


class MappingType(StrEnum):
    IDENTITY = "IDENTITY"
    HOMOGRAPHY = "HOMOGRAPHY"
    DENSE_BACKWARD_GRID = "DENSE_BACKWARD_GRID"


class IdentityMapping(ContractModel):
    mapping_type: Literal[MappingType.IDENTITY] = MappingType.IDENTITY
    mapping_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_mapping_hash(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("mapping_sha256"):
            data["mapping_sha256"] = canonical_sha256({"mapping_type": MappingType.IDENTITY.value})
        return data

    @model_validator(mode="after")
    def verify_mapping_hash(self) -> IdentityMapping:
        expected = canonical_sha256({"mapping_type": self.mapping_type.value})
        if self.mapping_sha256 != expected:
            raise ValueError("identity mapping hash does not match canonical mapping")
        return self


class HomographyMapping(ContractModel):
    mapping_type: Literal[MappingType.HOMOGRAPHY] = MappingType.HOMOGRAPHY
    child_to_parent_matrix: Matrix
    mapping_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_mapping_hash(cls, value: Any) -> Any:
        data = dict(value)
        matrix = data.get("child_to_parent_matrix")
        if not data.get("mapping_sha256") and matrix is not None:
            matrix = tuple(tuple(float(cell) for cell in row) for row in matrix)
            data["child_to_parent_matrix"] = matrix
            data["mapping_sha256"] = canonical_sha256(
                {
                    "mapping_type": MappingType.HOMOGRAPHY.value,
                    "child_to_parent_matrix": matrix,
                }
            )
        return data

    @model_validator(mode="after")
    def validate_matrix_and_hash(self) -> HomographyMapping:
        _validate_homography(self.child_to_parent_matrix)
        expected = canonical_sha256(
            {
                "mapping_type": self.mapping_type.value,
                "child_to_parent_matrix": self.child_to_parent_matrix,
            }
        )
        if self.mapping_sha256 != expected:
            raise ValueError("homography mapping hash does not match canonical mapping")
        return self


class DenseBackwardGridMapping(ContractModel):
    """A content-addressed child-to-parent normalized coordinate grid."""

    mapping_type: Literal[MappingType.DENSE_BACKWARD_GRID] = MappingType.DENSE_BACKWARD_GRID
    grid_relative_path: str = Field(min_length=1)
    grid_sha256: str = Field(pattern=SHA256_PATTERN)
    grid_dtype: Literal["float32"] = "float32"
    grid_shape: tuple[int, int, int] = Field(min_length=3, max_length=3)
    child_width: int = Field(ge=2)
    child_height: int = Field(ge=2)
    parent_width: int = Field(ge=2)
    parent_height: int = Field(ge=2)
    coordinate_domain: Literal["normalized_minus_one_to_one"] = "normalized_minus_one_to_one"
    interpolation: Literal["bilinear"] = "bilinear"
    align_corners: Literal[True] = True
    padding_mode: Literal["zeros", "border", "reflection"]
    mapping_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @field_validator("grid_relative_path")
    @classmethod
    def require_contained_npz_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or path.suffix != ".npz":
            raise ValueError("dense grid path must be a contained relative .npz path")
        return value

    @field_validator("grid_shape")
    @classmethod
    def require_positive_shape(cls, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        if shape[0] < 2 or shape[1] < 2 or shape[2] != 2:
            raise ValueError("dense grid shape must be (height>=2, width>=2, 2)")
        return shape

    @model_validator(mode="before")
    @classmethod
    def fill_mapping_hash(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("mapping_sha256"):
            data.setdefault("mapping_type", MappingType.DENSE_BACKWARD_GRID)
            data.setdefault("grid_dtype", "float32")
            data.setdefault("coordinate_domain", "normalized_minus_one_to_one")
            data.setdefault("interpolation", "bilinear")
            data.setdefault("align_corners", True)
            payload = dict(data)
            payload.pop("mapping_sha256", None)
            data["mapping_sha256"] = canonical_sha256(payload)
        return data

    @model_validator(mode="after")
    def verify_mapping_hash(self) -> DenseBackwardGridMapping:
        payload = self.model_dump(mode="json", exclude={"mapping_sha256"})
        if self.mapping_sha256 != canonical_sha256(payload):
            raise ValueError("dense mapping hash does not match canonical mapping")
        return self


ArtifactMapping: TypeAlias = Annotated[
    IdentityMapping | HomographyMapping | DenseBackwardGridMapping,
    Field(discriminator="mapping_type"),
]
_mapping_adapter = TypeAdapter(ArtifactMapping)


def artifact_id_for(
    *,
    artifact_kind: ArtifactKind,
    image_sha256: str,
    parent_artifact_id: str | None,
    producer: str,
    producer_version: str,
    configuration_sha256: str,
    mapping_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "artifact_kind": artifact_kind.value,
            "image_sha256": image_sha256,
            "parent_artifact_id": parent_artifact_id,
            "producer": producer,
            "producer_version": producer_version,
            "configuration_sha256": configuration_sha256,
            "mapping_sha256": mapping_sha256,
        }
    )


def _is_right_angle_homography(
    matrix: Matrix,
    parent_width: int,
    parent_height: int,
    child_width: int,
    child_height: int,
) -> bool:
    """Check a child-to-parent matrix against the three pixel-grid rotations."""

    w, h = float(parent_width), float(parent_height)
    expected = (
        ((0.0, 1.0, 0.0), (-1.0, 0.0, h - 1.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, w - 1.0), (0.0, -1.0, h - 1.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0, w - 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    dimensions = ((h, w), (w, h), (h, w))
    return any(
        (child_width, child_height) == size
        and all(
            abs(actual - wanted) <= 1e-6
            for actual, wanted in zip(
                (cell for row in matrix for cell in row),
                (cell for row in candidate for cell in row),
                strict=True,
            )
        )
        for candidate, size in zip(expected, dimensions, strict=True)
    )


class ArtifactRef(ContractModel):
    artifact_id: str = Field(default="", pattern=SHA256_PATTERN)
    artifact_kind: ArtifactKind
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_relative_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    parent_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    child_to_parent_mapping: ArtifactMapping

    @model_validator(mode="before")
    @classmethod
    def fill_artifact_id(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("artifact_id") and data.get("child_to_parent_mapping") is not None:
            mapping = _mapping_adapter.validate_python(data["child_to_parent_mapping"])
            data["artifact_id"] = artifact_id_for(
                artifact_kind=ArtifactKind(data["artifact_kind"]),
                image_sha256=data["image_sha256"],
                parent_artifact_id=data.get("parent_artifact_id"),
                producer=data["producer"],
                producer_version=data["producer_version"],
                configuration_sha256=data["configuration_sha256"],
                mapping_sha256=mapping.mapping_sha256,
            )
        return data

    @model_validator(mode="after")
    def validate_identity_and_id(self) -> ArtifactRef:
        if self.artifact_kind is ArtifactKind.SOURCE_RAW:
            if self.parent_artifact_id is not None:
                raise ValueError("SOURCE_RAW artifacts must be graph roots")
            if not isinstance(self.child_to_parent_mapping, IdentityMapping):
                raise ValueError("SOURCE_RAW artifacts require an identity mapping")
        elif self.parent_artifact_id is None:
            raise ValueError("non-SOURCE_RAW artifacts require a parent artifact")
        expected = artifact_id_for(
            artifact_kind=self.artifact_kind,
            image_sha256=self.image_sha256,
            parent_artifact_id=self.parent_artifact_id,
            producer=self.producer,
            producer_version=self.producer_version,
            configuration_sha256=self.configuration_sha256,
            mapping_sha256=self.child_to_parent_mapping.mapping_sha256,
        )
        if self.artifact_id != expected:
            raise ValueError("artifact ID does not match canonical artifact identity")
        return self


class ArtifactManifest(ContractModel):
    manifest_version: Literal["artifact_manifest_v1"] = "artifact_manifest_v1"
    artifacts: tuple[ArtifactRef, ...]
    manifest_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_manifest_hash(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("manifest_sha256") and data.get("artifacts") is not None:
            artifacts = tuple(
                item if isinstance(item, ArtifactRef) else ArtifactRef.model_validate(item)
                for item in data["artifacts"]
            )
            data["manifest_sha256"] = canonical_sha256(
                {
                    "manifest_version": data.get("manifest_version", "artifact_manifest_v1"),
                    "artifacts": [item.model_dump(mode="json") for item in artifacts],
                }
            )
        return data

    @model_validator(mode="after")
    def validate_graph(self) -> ArtifactManifest:
        by_id = {item.artifact_id: item for item in self.artifacts}
        if len(by_id) != len(self.artifacts):
            raise ValueError("artifact IDs must be unique")
        roots = tuple(item for item in self.artifacts if item.parent_artifact_id is None)
        if not roots or any(item.artifact_kind is not ArtifactKind.SOURCE_RAW for item in roots):
            raise ValueError("only SOURCE_RAW artifacts may be graph roots")
        for artifact in self.artifacts:
            parent_id = artifact.parent_artifact_id
            if parent_id is not None and parent_id not in by_id:
                raise ValueError("artifact parent is missing from manifest")
            if artifact.artifact_kind is ArtifactKind.ORIENTED_RAW and (
                parent_id is None or by_id[parent_id].artifact_kind is not ArtifactKind.SOURCE_RAW
            ):
                raise ValueError("ORIENTED_RAW must be a direct child of SOURCE_RAW")
            if artifact.artifact_kind is ArtifactKind.ORIENTED_RAW:
                parent = by_id[parent_id]
                mapping = artifact.child_to_parent_mapping
                if isinstance(mapping, IdentityMapping):
                    if (artifact.width, artifact.height) != (parent.width, parent.height):
                        raise ValueError("identity ORIENTED_RAW must preserve source dimensions")
                elif not isinstance(mapping, HomographyMapping) or not _is_right_angle_homography(
                    mapping.child_to_parent_matrix,
                    parent.width,
                    parent.height,
                    artifact.width,
                    artifact.height,
                ):
                    raise ValueError(
                        "ORIENTED_RAW homography must represent a 90/180/270-degree rotation"
                    )
            if (
                isinstance(artifact.child_to_parent_mapping, IdentityMapping)
                and parent_id is not None
            ):
                parent = by_id[parent_id]
                if (artifact.width, artifact.height) != (parent.width, parent.height):
                    raise ValueError("identity child dimensions must equal parent dimensions")
            if isinstance(artifact.child_to_parent_mapping, DenseBackwardGridMapping):
                mapping = artifact.child_to_parent_mapping
                parent = by_id[parent_id]
                if (mapping.child_width, mapping.child_height) != (
                    artifact.width,
                    artifact.height,
                ):
                    raise ValueError("dense mapping child dimensions differ from artifact")
                if (mapping.parent_width, mapping.parent_height) != (
                    parent.width,
                    parent.height,
                ):
                    raise ValueError("dense mapping parent dimensions differ from parent artifact")
        for artifact in self.artifacts:
            seen: set[str] = set()
            current = artifact
            while current.parent_artifact_id is not None:
                if current.artifact_id in seen:
                    raise ValueError("artifact graph contains a cycle")
                seen.add(current.artifact_id)
                current = by_id[current.parent_artifact_id]
        expected = canonical_sha256(
            {
                "manifest_version": self.manifest_version,
                "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
            }
        )
        if self.manifest_sha256 != expected:
            raise ValueError("artifact manifest hash does not match canonical manifest")
        return self

    def by_id(self) -> dict[str, ArtifactRef]:
        return {item.artifact_id: item for item in self.artifacts}


class PageArtifact(ContractModel):
    artifact: ArtifactRef
    page_number: int = Field(ge=1)
    dpi: int = Field(gt=0)
    role: Literal["SOURCE_RAW", "ORIENTED_RAW", "CANDIDATE"] | None = None
    selected: bool = False
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    route_reasons: tuple[str, ...] = ()
    transform_metrics: dict[str, Any] = Field(default_factory=dict)


class TableCandidateVariant(StrEnum):
    """Page-artifact variants that may be compared for one logical table.

    These values intentionally mirror :class:`ArtifactKind`.  Keeping a
    table-selection variant separate from the generic artifact enum prevents a
    selection record from silently referring to a table crop, cell crop, or
    another non-page artifact.  ``_missing_`` accepts the lower-case spelling
    used by older extraction telemetry while serializing to the canonical V6
    spelling.
    """

    SOURCE_RAW = "source_raw"
    ORIENTED_RAW = "oriented_raw"
    PROJECTIVE = "projective"
    PROJECTIVE_ENHANCED = "projective_enhanced"
    UVDOC = "uvdoc"
    UVDOC_ENHANCED = "uvdoc_enhanced"

    @classmethod
    def _missing_(cls, value: object) -> TableCandidateVariant | None:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class TableMatchDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


class TableSelectionDecision(StrEnum):
    SELECTED = "selected"
    ABSTAINED = "abstained"
    AMBIGUOUS = "ambiguous"
    NO_CANDIDATE = "no_candidate"


def _require_non_blank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must be non-blank")
    return value


class TableMatchEdge(ContractModel):
    """One deterministic source-space proposal matching observation.

    Match edges are shadow telemetry.  They may mention a UVDoc candidate,
    including a hypothetical winner, but they never carry an authority bit and
    therefore cannot publish evidence or switch a canonical table on their own.
    """

    edge_id: str = Field(
        default="",
        validation_alias=AliasChoices("edge_id", "match_edge_id"),
    )
    logical_table_id: str = Field(min_length=1)
    page_number: int = Field(default=1, ge=1)
    anchor_proposal_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "anchor_proposal_id",
            "source_proposal_id",
            "left_proposal_id",
        ),
    )
    candidate_proposal_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "candidate_proposal_id",
            "derivative_proposal_id",
            "right_proposal_id",
        ),
    )
    anchor_artifact_id: str = Field(
        pattern=SHA256_PATTERN,
        validation_alias=AliasChoices("anchor_artifact_id", "source_artifact_id"),
    )
    candidate_artifact_id: str = Field(
        pattern=SHA256_PATTERN,
        validation_alias=AliasChoices("candidate_artifact_id", "artifact_id"),
    )
    candidate_variant: TableCandidateVariant = Field(
        validation_alias=AliasChoices("candidate_variant", "variant")
    )
    match_score: float = Field(
        ge=0,
        le=1,
        validation_alias=AliasChoices("match_score", "score"),
    )
    overlap_score: float | None = Field(default=None, ge=0, le=1)
    center_distance_score: float | None = Field(default=None, ge=0, le=1)
    reading_order_score: float | None = Field(default=None, ge=0, le=1)
    table_type_score: float | None = Field(default=None, ge=0, le=1)
    header_similarity_score: float | None = Field(default=None, ge=0, le=1)
    decision: TableMatchDecision = Field(
        default=TableMatchDecision.REJECTED,
        validation_alias=AliasChoices("decision", "disposition", "status"),
    )
    ambiguous: bool = False
    ambiguity_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_edge_id(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("edge_id") and not data.get("match_edge_id"):
            logical_table_id = data.get("logical_table_id", "")
            anchor = (
                data.get("anchor_proposal_id")
                or data.get("source_proposal_id")
                or data.get("left_proposal_id", "")
            )
            candidate = (
                data.get("candidate_proposal_id")
                or data.get("derivative_proposal_id")
                or data.get("right_proposal_id", "")
            )
            data["edge_id"] = canonical_sha256(
                {
                    "contract": "table_match_edge_v1",
                    "logical_table_id": logical_table_id,
                    "anchor_proposal_id": anchor,
                    "candidate_proposal_id": candidate,
                }
            )
        return data

    @model_validator(mode="after")
    def validate_edge(self) -> TableMatchEdge:
        _require_non_blank(self.logical_table_id, "logical table ID")
        _require_non_blank(self.anchor_proposal_id, "anchor proposal ID")
        _require_non_blank(self.candidate_proposal_id, "candidate proposal ID")
        if self.anchor_proposal_id == self.candidate_proposal_id:
            raise ValueError("table match edge cannot match a proposal to itself")
        if self.decision is TableMatchDecision.AMBIGUOUS:
            if not self.ambiguous:
                raise ValueError("ambiguous match edges must set ambiguous=true")
            if not self.ambiguity_reason or not self.ambiguity_reason.strip():
                raise ValueError("ambiguous match edges require an ambiguity reason")
        elif self.ambiguous:
            raise ValueError("only ambiguous match edges may set ambiguous=true")
        elif self.ambiguity_reason is not None:
            raise ValueError("non-ambiguous match edges cannot carry an ambiguity reason")
        return self


class TableCandidateScore(ContractModel):
    """Stage-1/2 score for one whole-table candidate artifact."""

    evaluation_id: str = Field(
        default="",
        validation_alias=AliasChoices("evaluation_id", "candidate_score_id"),
    )
    logical_table_id: str = Field(min_length=1)
    page_number: int = Field(default=1, ge=1)
    candidate_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("candidate_id", "candidate_proposal_id"),
    )
    candidate_artifact_id: str = Field(
        pattern=SHA256_PATTERN,
        validation_alias=AliasChoices("candidate_artifact_id", "artifact_id"),
    )
    candidate_variant: TableCandidateVariant = Field(
        validation_alias=AliasChoices("candidate_variant", "variant")
    )
    score: float = Field(
        ge=0,
        le=1,
        validation_alias=AliasChoices("score", "candidate_score", "stage2_score"),
    )
    rank: int = Field(default=1, ge=1)
    eligible: bool = True
    selected: bool = Field(
        default=False,
        validation_alias=AliasChoices("selected", "is_selected"),
    )
    evaluation_status: Literal["eligible", "rejected", "selected", "ambiguous"] = "eligible"
    rejection_reason: str | None = None
    score_components: dict[str, float] = Field(default_factory=dict)
    whole_table: Literal[True] = True

    @model_validator(mode="before")
    @classmethod
    def fill_evaluation_id(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("evaluation_id") and not data.get("candidate_score_id"):
            data["evaluation_id"] = canonical_sha256(
                {
                    "contract": "table_candidate_score_v1",
                    "logical_table_id": data.get("logical_table_id", ""),
                    "candidate_id": data.get("candidate_id")
                    or data.get("candidate_proposal_id", ""),
                    "candidate_artifact_id": data.get("candidate_artifact_id")
                    or data.get("artifact_id", ""),
                }
            )
        return data

    @model_validator(mode="after")
    def validate_score(self) -> TableCandidateScore:
        _require_non_blank(self.logical_table_id, "logical table ID")
        _require_non_blank(self.candidate_id, "candidate ID")
        if self.whole_table is not True:
            raise ValueError("candidate evaluations must have whole_table=true")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in self.score_components.values()
        ):
            raise ValueError("candidate score components must be finite values in [0, 1]")
        if self.selected and (
            not self.eligible or self.evaluation_status not in {"eligible", "selected"}
        ):
            raise ValueError("selected candidate must be eligible")
        if self.evaluation_status == "selected" and not self.selected:
            raise ValueError("selected candidate evaluation must set selected=true")
        if self.evaluation_status == "rejected" and self.eligible:
            raise ValueError("rejected candidate evaluation cannot remain eligible")
        if self.rejection_reason is not None and not self.rejection_reason.strip():
            raise ValueError("candidate rejection reason must be non-blank")
        return self


class LogicalTableSelection(ContractModel):
    """A whole-table decision and its nested M5 shadow observations."""

    logical_table_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    match_edges: tuple[TableMatchEdge, ...] = ()
    candidate_evaluations: tuple[TableCandidateScore, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "candidate_evaluations",
            "candidate_scores",
        ),
    )
    decision: TableSelectionDecision = Field(
        default=TableSelectionDecision.NO_CANDIDATE,
        validation_alias=AliasChoices("decision", "status"),
    )
    selected_candidate_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("selected_candidate_id", "winner_candidate_id"),
    )
    selected_candidate_variant: TableCandidateVariant | None = Field(
        default=None,
        validation_alias=AliasChoices("selected_candidate_variant", "winner_variant"),
    )
    selected_artifact_id: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
        validation_alias=AliasChoices(
            "selected_artifact_id",
            "selected_candidate_artifact_id",
            "winner_artifact_id",
        ),
    )
    ambiguous: bool = False
    ambiguity_reason: str | None = None
    whole_table: Literal[True] = True

    @model_validator(mode="after")
    def validate_selection(self) -> LogicalTableSelection:
        _require_non_blank(self.logical_table_id, "logical table ID")
        if self.whole_table is not True:
            raise ValueError("table selection decisions must have whole_table=true")
        if len({edge.edge_id for edge in self.match_edges}) != len(self.match_edges):
            raise ValueError("match edge IDs must be unique within a logical table")
        if len({item.evaluation_id for item in self.candidate_evaluations}) != len(
            self.candidate_evaluations
        ):
            raise ValueError("candidate evaluation IDs must be unique within a logical table")
        for edge in self.match_edges:
            if (
                edge.logical_table_id != self.logical_table_id
                or edge.page_number != self.page_number
            ):
                raise ValueError("match edge references another logical table")
        for evaluation in self.candidate_evaluations:
            if (
                evaluation.logical_table_id != self.logical_table_id
                or evaluation.page_number != self.page_number
            ):
                raise ValueError("candidate evaluation references another logical table")
        selected = tuple(item for item in self.candidate_evaluations if item.selected)
        if self.decision is TableSelectionDecision.SELECTED:
            if self.ambiguous or self.ambiguity_reason is not None:
                raise ValueError("selected table cannot be ambiguous")
            if len(selected) != 1:
                raise ValueError("selected table requires exactly one selected candidate")
            winner = selected[0]
            if self.selected_candidate_id != winner.candidate_id:
                raise ValueError("selected candidate ID does not match candidate evaluation")
            if self.selected_candidate_variant is not winner.candidate_variant:
                raise ValueError("selected candidate variant does not match candidate evaluation")
            if self.selected_artifact_id != winner.candidate_artifact_id:
                raise ValueError("selected artifact does not match candidate evaluation")
        else:
            if selected:
                raise ValueError("abstained or ambiguous tables cannot select a candidate")
            if any(
                value is not None
                for value in (
                    self.selected_candidate_id,
                    self.selected_candidate_variant,
                    self.selected_artifact_id,
                )
            ):
                raise ValueError("non-selected table decisions cannot name a winner")
        if self.decision is TableSelectionDecision.AMBIGUOUS:
            if not self.ambiguous:
                raise ValueError("ambiguous table decisions must set ambiguous=true")
            if not self.ambiguity_reason or not self.ambiguity_reason.strip():
                raise ValueError("ambiguous table decisions require an ambiguity reason")
        elif self.ambiguous:
            raise ValueError("only ambiguous table decisions may set ambiguous=true")
        elif self.ambiguity_reason is not None:
            raise ValueError("non-ambiguous table decisions cannot carry an ambiguity reason")
        return self

    @property
    def candidate_scores(self) -> tuple[TableCandidateScore, ...]:
        """Compatibility view for callers that call evaluations ``scores``."""

        return self.candidate_evaluations


class TableSelectionRun(ContractModel):
    """One deterministic M5 run, with observations grouped by logical table."""

    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "selection_run_id"),
    )
    run_version: Literal["table_selection_v1"] = "table_selection_v1"
    document_id: str | None = None
    source_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    mode: Literal["off", "shadow", "enabled"] = "shadow"
    logical_tables: tuple[LogicalTableSelection, ...] = Field(
        default=(),
        validation_alias=AliasChoices("logical_tables", "tables"),
    )
    # The single-table form is accepted for early M5 producers.  New writers
    # should use ``logical_tables`` so edges and evaluations stay nested.
    logical_table_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    match_edges: tuple[TableMatchEdge, ...] = ()
    candidate_evaluations: tuple[TableCandidateScore, ...] = Field(
        default=(),
        validation_alias=AliasChoices("candidate_evaluations", "candidate_scores"),
    )

    @model_validator(mode="before")
    @classmethod
    def fill_run_id(cls, value: Any) -> Any:
        data = dict(value)
        if not data.get("run_id") and not data.get("selection_run_id"):
            data["run_id"] = canonical_sha256(
                {
                    "contract": "table_selection_run_v1",
                    "document_id": data.get("document_id"),
                    "source_sha256": data.get("source_sha256"),
                    "logical_table_id": data.get("logical_table_id"),
                }
            )
        return data

    @model_validator(mode="after")
    def validate_run(self) -> TableSelectionRun:
        if self.document_id is not None:
            _require_non_blank(self.document_id, "document ID")
        if self.logical_tables and (
            self.logical_table_id is not None
            or self.page_number is not None
            or self.match_edges
            or self.candidate_evaluations
        ):
            raise ValueError("nested and single-table selection run forms cannot be mixed")
        if self.logical_tables:
            logical_ids = tuple(item.logical_table_id for item in self.logical_tables)
            if len(set(logical_ids)) != len(logical_ids):
                raise ValueError("selection run logical table IDs must be unique")
        else:
            if self.logical_table_id is None or self.page_number is None:
                raise ValueError("selection run requires logical table records")
            LogicalTableSelection(
                logical_table_id=self.logical_table_id,
                page_number=self.page_number,
                match_edges=self.match_edges,
                candidate_evaluations=self.candidate_evaluations,
            )
        if self.mode == "off":
            raise ValueError("table selection runs cannot be emitted when mode is off")
        if self.mode == "enabled" and any(
            table.decision is TableSelectionDecision.SELECTED
            and table.selected_candidate_variant
            in {TableCandidateVariant.UVDOC, TableCandidateVariant.UVDOC_ENHANCED}
            for table in self.tables
        ):
            raise ValueError("UVDoc shadow candidates cannot become authoritative in enabled mode")
        return self

    @property
    def tables(self) -> tuple[LogicalTableSelection, ...]:
        """Compatibility view for callers that use ``tables`` as the field name."""

        if self.logical_tables:
            return self.logical_tables
        assert self.logical_table_id is not None and self.page_number is not None
        return (
            LogicalTableSelection(
                logical_table_id=self.logical_table_id,
                page_number=self.page_number,
                match_edges=self.match_edges,
                candidate_evaluations=self.candidate_evaluations,
            ),
        )

    @property
    def candidate_scores(self) -> tuple[TableCandidateScore, ...]:
        """Compatibility view for the legacy score terminology."""

        if self.logical_tables:
            return tuple(
                item for table in self.logical_tables for item in table.candidate_evaluations
            )
        return self.candidate_evaluations


# The V6 envelope is already versioned, while a few integrations imported
# explicit ``V1`` names for the first M5 draft.  Keep those names as aliases so
# old shadow readers do not need a migration just to read a result.
TableMatchEdgeV1 = TableMatchEdge
TableCandidateScoreV1 = TableCandidateScore
TableCandidateEvaluation = TableCandidateScore
TableCandidateEvaluationV1 = TableCandidateScore
TableSelectionTable = LogicalTableSelection
TableSelectionRunV1 = TableSelectionRun


class M5ShadowReconstructionV1(ContractModel):
    """Complete reconstruction retained for evaluation-only M5 scoring."""

    reconstruction_version: Literal["m5_reconstruction_v1"] = "m5_reconstruction_v1"
    proposal_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    variant: TableCandidateVariant
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_box: tuple[float, float, float, float]
    candidate_box: tuple[int, int, int, int] | None = None
    schema: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    source_tables: tuple[dict[str, Any], ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    reconstruction_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_reconstruction_hash(cls, value: Any) -> Any:
        data = dict(value)
        data.setdefault("reconstruction_version", "m5_reconstruction_v1")
        data.setdefault("candidate_box", None)
        data.setdefault("schema", None)
        data.setdefault("diagnostics", {})
        data.setdefault("source_tables", ())
        data.setdefault("rows", ())
        if data.get("source_box") is not None:
            data["source_box"] = tuple(float(value) for value in data["source_box"])
        if data.get("candidate_box") is not None:
            data["candidate_box"] = tuple(int(value) for value in data["candidate_box"])
        if not data.get("reconstruction_sha256"):
            payload = {
                key: item for key, item in data.items() if key != "reconstruction_sha256"
            }
            data["reconstruction_sha256"] = canonical_sha256(payload)
        return data

    @model_validator(mode="after")
    def validate_reconstruction(self) -> M5ShadowReconstructionV1:
        left, top, right, bottom = self.source_box
        if any(not math.isfinite(value) for value in self.source_box):
            raise ValueError("M5 reconstruction source box must be finite")
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("M5 reconstruction source box is invalid")
        if self.candidate_box is not None:
            candidate_left, candidate_top, candidate_right, candidate_bottom = self.candidate_box
            if (
                candidate_left < 0
                or candidate_top < 0
                or candidate_right <= candidate_left
                or candidate_bottom <= candidate_top
            ):
                raise ValueError("M5 reconstruction candidate box is invalid")
        payload = self.model_dump(mode="json", exclude={"reconstruction_sha256"})
        expected = canonical_sha256(payload)
        if self.reconstruction_sha256 != expected:
            raise ValueError("M5 reconstruction digest does not match content")
        return self


class M5ShadowProposalV1(ContractModel):
    """Source/artifact lineage for one M5 proposal."""

    proposal_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    variant: TableCandidateVariant
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_box: tuple[float, float, float, float]
    candidate_box: tuple[int, int, int, int] | None = None
    reading_order: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_proposal(self) -> M5ShadowProposalV1:
        left, top, right, bottom = self.source_box
        if any(not math.isfinite(value) for value in self.source_box):
            raise ValueError("M5 proposal source box must be finite")
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("M5 proposal source box is invalid")
        if self.candidate_box is not None:
            candidate_left, candidate_top, candidate_right, candidate_bottom = self.candidate_box
            if (
                candidate_left < 0
                or candidate_top < 0
                or candidate_right <= candidate_left
                or candidate_bottom <= candidate_top
            ):
                raise ValueError("M5 proposal candidate box is invalid")
        return self


class M5ShadowTableProjectionV1(ContractModel):
    """One whole-table M5 decision and its evaluation reconstruction."""

    logical_table_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    anchor_proposal_id: str | None = None
    proposal_ids: tuple[str, ...] = ()
    proposals: tuple[M5ShadowProposalV1, ...] = ()
    baseline_proposal_id: str | None = None
    selected_proposal_id: str | None = None
    baseline_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    selected_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    baseline_variant: TableCandidateVariant | None = None
    selected_variant: TableCandidateVariant | None = None
    decision: Literal["selected", "baseline_fallback", "abstained"]
    ambiguous: bool = False
    ambiguity_reason: str | None = None
    selected_metrics: dict[str, Any] = Field(default_factory=dict)
    candidate_ranking: tuple[dict[str, Any], ...] = ()
    baseline_reconstruction: M5ShadowReconstructionV1 | None = None
    selected_reconstruction: M5ShadowReconstructionV1 | None = None

    @model_validator(mode="after")
    def validate_table(self) -> M5ShadowTableProjectionV1:
        if len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError("M5 table proposal IDs must be unique")
        proposal_map = {item.proposal_id: item for item in self.proposals}
        if set(proposal_map) != set(self.proposal_ids):
            raise ValueError("M5 table proposal lineage is incomplete")
        if any(item.source_sha256 != self.source_sha256 for item in self.proposals):
            raise ValueError("M5 proposal source hash differs from table source")
        if any(item.page_number != self.page_number for item in self.proposals):
            raise ValueError("M5 proposal page differs from table page")
        if self.anchor_proposal_id is not None and self.anchor_proposal_id not in proposal_map:
            raise ValueError("M5 anchor proposal is missing from lineage")
        if self.baseline_proposal_id is not None and self.baseline_proposal_id not in proposal_map:
            raise ValueError("M5 baseline proposal is missing from lineage")
        if self.selected_proposal_id is not None and self.selected_proposal_id not in proposal_map:
            raise ValueError("M5 selected proposal is missing from lineage")
        if self.baseline_reconstruction is not None:
            if self.baseline_proposal_id != self.baseline_reconstruction.proposal_id:
                raise ValueError("M5 baseline reconstruction does not match proposal")
            if self.baseline_artifact_sha256 != self.baseline_reconstruction.artifact_sha256:
                raise ValueError("M5 baseline artifact does not match reconstruction")
            if self.baseline_variant is not self.baseline_reconstruction.variant:
                raise ValueError("M5 baseline variant does not match reconstruction")
        if self.selected_reconstruction is not None:
            if self.selected_proposal_id != self.selected_reconstruction.proposal_id:
                raise ValueError("M5 selected reconstruction does not match proposal")
            if self.selected_artifact_sha256 != self.selected_reconstruction.artifact_sha256:
                raise ValueError("M5 selected artifact does not match reconstruction")
            if self.selected_variant is not self.selected_reconstruction.variant:
                raise ValueError("M5 selected variant does not match reconstruction")
        if self.decision == "selected":
            if (
                self.ambiguous
                or self.selected_proposal_id is None
                or self.selected_reconstruction is None
            ):
                raise ValueError("selected M5 table requires a complete winner")
        elif self.decision == "baseline_fallback":
            if not self.ambiguous or not self.ambiguity_reason or not self.ambiguity_reason.strip():
                raise ValueError("M5 baseline fallback requires an ambiguity reason")
            if self.baseline_proposal_id != self.selected_proposal_id:
                raise ValueError("M5 ambiguity fallback must select the baseline proposal")
            if self.selected_reconstruction is None:
                raise ValueError("M5 ambiguity fallback requires baseline reconstruction")
        else:
            if self.ambiguous or self.selected_proposal_id is not None:
                raise ValueError("abstained M5 table cannot name an ambiguous winner")
            if self.selected_reconstruction is not None:
                raise ValueError("abstained M5 table cannot carry selected reconstruction")
        if self.ambiguous != (self.decision == "baseline_fallback"):
            raise ValueError("M5 ambiguity state does not match decision")
        return self


class M5ShadowPageProjectionV1(ContractModel):
    """One page's shadow run, including failed runs for omission detection."""

    page_number: int = Field(ge=1)
    status: Literal["complete", "failed"]
    reason: str | None = None
    tables: tuple[M5ShadowTableProjectionV1, ...] = ()

    @model_validator(mode="after")
    def validate_page(self) -> M5ShadowPageProjectionV1:
        if len({item.logical_table_id for item in self.tables}) != len(self.tables):
            raise ValueError("M5 logical table IDs must be unique per page")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed M5 page runs require a reason")
        if self.status == "complete" and self.reason is not None:
            raise ValueError("complete M5 page runs cannot carry a failure reason")
        return self


class M5ShadowProjectionV1(ContractModel):
    """Content-addressed, non-canonical M5 projection sidecar."""

    projection_version: Literal["m5_shadow_projection_v1"] = "m5_shadow_projection_v1"
    document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    mode: Literal["shadow"] = "shadow"
    authority: Literal["non_canonical"] = "non_canonical"
    canonical_publication: Literal[False] = False
    expected_page_count: int | None = Field(default=None, ge=1)
    pages: tuple[M5ShadowPageProjectionV1, ...] = ()
    projection_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_projection_hash(cls, value: Any) -> Any:
        data = dict(value)
        data.setdefault("projection_version", "m5_shadow_projection_v1")
        data.setdefault("mode", "shadow")
        data.setdefault("authority", "non_canonical")
        data.setdefault("canonical_publication", False)
        data.setdefault("expected_page_count", None)
        data.setdefault("pages", ())
        data["pages"] = tuple(
            M5ShadowPageProjectionV1.model_validate(item).model_dump(mode="json")
            for item in data["pages"]
        )
        if not data.get("projection_sha256"):
            payload = {key: item for key, item in data.items() if key != "projection_sha256"}
            data["projection_sha256"] = canonical_sha256(payload)
        return data

    @model_validator(mode="after")
    def validate_projection(self) -> M5ShadowProjectionV1:
        page_numbers = tuple(item.page_number for item in self.pages)
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("M5 projection pages must be unique")
        if self.expected_page_count is not None and page_numbers != tuple(
            range(1, self.expected_page_count + 1)
        ):
            raise ValueError("M5 projection pages are incomplete or out of order")
        logical_ids = tuple(
            table.logical_table_id for page in self.pages for table in page.tables
        )
        if len(set(logical_ids)) != len(logical_ids):
            raise ValueError("M5 projection logical table IDs must be unique")
        for page in self.pages:
            for table in page.tables:
                if table.source_sha256 != self.source_sha256:
                    raise ValueError("M5 table source hash differs from projection")
        payload = self.model_dump(mode="json", exclude={"projection_sha256"})
        expected = canonical_sha256(payload)
        if self.projection_sha256 != expected:
            raise ValueError("M5 projection digest does not match content")
        return self

    @property
    def tables(self) -> tuple[M5ShadowTableProjectionV1, ...]:
        """Return deterministic table order independent of page nesting."""

        return tuple(
            table
            for page in self.pages
            for table in sorted(page.tables, key=lambda item: item.logical_table_id)
        )

    def authority_document(self) -> dict[str, Any]:
        """Project selected source tables for read-only authority scoring."""

        source_tables = [
            source_table
            for table in self.tables
            if table.selected_reconstruction is not None
            for source_table in table.selected_reconstruction.source_tables
        ]
        return {
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "source_name": self.source_name,
            "tables": source_tables,
        }


class UvdocShadowStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    FAILED = "failed"
    INELIGIBLE = "ineligible"


class UvdocShadowRun(ContractModel):
    """Deterministic audit record for one page-level UVDoc shadow attempt."""

    page_number: int = Field(ge=1)
    status: UvdocShadowStatus
    reason_code: str = Field(min_length=1)
    input_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    uvdoc_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    enhanced_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    adapter_config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    paddle_version: str | None = None
    paddleocr_version: str | None = None
    paddlex_version: str | None = None
    reproduction_max_error_by_channel: tuple[int, int, int] | None = None
    transform_metrics: dict[str, Any] = Field(default_factory=dict)
    probe_metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_valid_artifacts(self) -> UvdocShadowRun:
        artifact_ids = (self.input_artifact_id, self.uvdoc_artifact_id, self.enhanced_artifact_id)
        identities = (self.model_sha256, self.model_config_sha256, self.adapter_config_sha256)
        if self.status is UvdocShadowStatus.VALID:
            if any(value is None for value in (*artifact_ids, *identities)):
                raise ValueError("valid UVDoc shadow runs require artifacts and model identities")
            if any(
                value is None
                for value in (
                    self.paddle_version,
                    self.paddleocr_version,
                    self.paddlex_version,
                )
            ):
                raise ValueError("valid UVDoc shadow runs require package identities")
            if self.reproduction_max_error_by_channel is None:
                raise ValueError("valid UVDoc shadow runs require reproduction errors")
            if any(not 0 <= value <= 1 for value in self.reproduction_max_error_by_channel):
                raise ValueError("UVDoc reproduction error exceeds one value per channel")
            required_metrics = {
                "mean_displacement_px",
                "max_displacement_px",
                "local_scale_p05",
                "local_scale_p50",
                "local_scale_p95",
                "anisotropy_p95",
                "jacobian_determinant_p05",
                "jacobian_determinant_p50",
                "jacobian_determinant_p95",
                "foldover_count",
                "out_of_bounds_rate",
            }
            if set(self.transform_metrics) != required_metrics:
                raise ValueError("valid UVDoc shadow runs require complete transform metrics")
            if self.transform_metrics["foldover_count"] != 0:
                raise ValueError("valid UVDoc shadow runs cannot contain fold-over")
            if self.transform_metrics["out_of_bounds_rate"] > 0.005:
                raise ValueError("valid UVDoc shadow run exceeds out-of-bounds limit")
        elif self.uvdoc_artifact_id is not None or self.enhanced_artifact_id is not None:
            raise ValueError("invalid UVDoc shadow runs may not publish candidate artifacts")
        return self


class CanonicalTableArtifact(ContractModel):
    artifact: ArtifactRef
    page_number: int = Field(default=1, ge=1)
    logical_table_id: str = Field(min_length=1)
    page_artifact_id: str = Field(pattern=SHA256_PATTERN)
    crop_polygon_in_page_artifact: Polygon
    crop_polygon_in_source_raw: Polygon
    context_margin: tuple[int, int, int, int] = (0, 0, 0, 0)
    table_type_hint: str | None = None

    @model_validator(mode="after")
    def require_table_crop(self) -> CanonicalTableArtifact:
        if self.artifact.artifact_kind is not ArtifactKind.TABLE_CROP:
            raise ValueError("canonical table artifact must reference a TABLE_CROP")
        if any(value < 0 for value in self.context_margin):
            raise ValueError("context margin must be non-negative")
        return self


class EvidenceRefV2(ContractModel):
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_polygon: Polygon
    source_page_polygon: Polygon
    source_page_number: int = Field(ge=1)
    source_page_artifact_id: str = Field(pattern=SHA256_PATTERN)
    ocr_token_ids: tuple[str, ...] = ()
    extractor: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    recognition_variant: str = Field(min_length=1)

    @field_validator("ocr_token_ids")
    @classmethod
    def require_non_blank_token_ids(cls, token_ids: tuple[str, ...]) -> tuple[str, ...]:
        if any(not token_id.strip() for token_id in token_ids):
            raise ValueError("evidence token IDs must be non-blank strings")
        return token_ids


class SourceColumnV2(SourceColumn):
    """Printed column with V2 artifact evidence."""

    evidence: tuple[EvidenceRefV2, ...] = ()

    @model_validator(mode="after")
    def require_grounded_or_synthetic_header(self) -> SourceColumnV2:
        grounded = any(item.ocr_token_ids for item in self.evidence)
        synthetic = "synthetic_header" in self.validation_flags
        if not grounded and not synthetic:
            raise ValueError("source header requires grounded OCR evidence")
        if grounded and synthetic:
            raise ValueError("grounded source header cannot be marked synthetic")
        return self


class SourceCellV2(SourceCell):
    """Printed cell with V2 artifact evidence."""

    evidence: tuple[EvidenceRefV2, ...] = ()

    @model_validator(mode="after")
    def require_grounded_value(self) -> SourceCellV2:
        if (
            self.raw_value
            and self.raw_value.strip()
            and not any(item.ocr_token_ids for item in self.evidence)
        ):
            raise ValueError("non-empty source cell requires grounded OCR evidence")
        return self


class SourceRowV2(SourceRow):
    cells: tuple[SourceCellV2, ...]


class SourceTableV2(SourceTable):
    columns: tuple[SourceColumnV2, ...]
    rows: tuple[SourceRowV2, ...]


class DocumentTotalV2(DocumentTotal):
    evidence: EvidenceRefV2


class RawTotalCandidateV2(RawTotalCandidate):
    total: DocumentTotalV2
    context_evidence: tuple[EvidenceRefV2, ...]


class CanonicalRowV2(CanonicalRow):
    evidence: tuple[EvidenceRefV2, ...]
    field_evidence: dict[str, tuple[EvidenceRefV2, ...]]


class TokenManifestEntryV2(ContractModel):
    token_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    table_ids: tuple[str, ...] = ()
    text: str
    canonical_polygon: Polygon
    source_page_polygon: Polygon
    source_page_artifact_id: str = Field(pattern=SHA256_PATTERN)
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_relative_path: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    parent_token_id: str | None = None
    parent_token_ids: tuple[str, ...] = ()
    parent_character_spans: tuple[tuple[int, int], ...] = ()
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)
    fragment_role: str | None = None

    @model_validator(mode="after")
    def require_consistent_fragment(self) -> TokenManifestEntryV2:
        if self.parent_token_id and self.parent_token_ids:
            raise ValueError("token provenance must use one parent representation")
        if self.parent_token_id is not None:
            if self.character_start is None or self.character_end is None or not self.fragment_role:
                raise ValueError("fragment token metadata must be complete")
            if self.character_end <= self.character_start:
                raise ValueError("fragment token range must be non-empty")
        if self.parent_token_ids:
            if len(self.parent_token_ids) != len(self.parent_character_spans):
                raise ValueError("composite token parents and spans must align")
            if not self.fragment_role or any(
                end <= start for start, end in self.parent_character_spans
            ):
                raise ValueError("composite fragments require role and non-empty spans")
        return self


class TableAdapterInputV2(ContractModel):
    page_number: int = Field(ge=1)
    logical_table_id: str = Field(min_length=1)
    input_artifact_id: str = Field(pattern=SHA256_PATTERN)
    input_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_name: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    latency_ms: int = Field(ge=0)
    cache_hit: bool = False
    accepted: bool = False
    stage: str = Field(min_length=1)
    recognition_variant: str = Field(min_length=1)


class ExtractionResultV6(ContractModel):
    """Complete V6 extraction envelope with strict business payload types."""

    output_version: Literal["offline_accuracy_spine_v6"] = "offline_accuracy_spine_v6"
    contract_revision: Literal[6] = 6
    document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_name: str = Field(min_length=1)
    pages: int = Field(ge=1)
    artifact_manifest: ArtifactManifest
    page_artifacts: tuple[PageArtifact, ...]
    uvdoc_shadow_runs: tuple[UvdocShadowRun, ...] = ()
    # M5 records are deliberately optional and default-empty so an old V6
    # payload remains byte-for-byte readable when revalidated.  The nested
    # records are observations; authoritative V6 fields below still own
    # canonical crops, tokens, evidence, and rows.
    table_match_edges: tuple[TableMatchEdge, ...] = Field(
        default=(),
        validation_alias=AliasChoices("table_match_edges", "m5_match_edges"),
    )
    table_candidate_scores: tuple[TableCandidateScore, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "table_candidate_scores",
            "table_candidate_evaluations",
            "m5_candidate_scores",
        ),
    )
    table_selection_runs: tuple[TableSelectionRun, ...] = Field(
        default=(),
        validation_alias=AliasChoices("table_selection_runs", "m5_selection_runs"),
    )
    canonical_table_artifacts: tuple[CanonicalTableArtifact, ...] = ()
    token_manifest: tuple[TokenManifestEntryV2, ...] = ()
    evidence: tuple[EvidenceRefV2, ...] = ()
    adapter_inputs: tuple[TableAdapterInputV2, ...] = ()
    document_total_version: str = "document_total_v2"
    document_totals_version: str = "document_totals_v2"
    document_total: DocumentTotalV2 | None = None
    document_totals: tuple[DocumentTotalV2, ...] = ()
    raw_total_candidates: tuple[RawTotalCandidateV2, ...] = ()
    hospital_id: str | None = None
    hospital: dict[str, Any] | None = None
    alias_registry_revision: int | None = None
    profile_registry_revision: int | None = None
    applied_alias_ids: tuple[str, ...] = ()
    page_assets: tuple[PageAsset, ...] = ()
    page_preprocessing: tuple[PagePreprocessingRecord, ...] = ()
    source_tables: tuple[SourceTableV2, ...] = ()
    suppressed_repeated_source_tables: tuple[SuppressedSourceTable, ...] = ()
    rows: tuple[CanonicalRowV2, ...] = ()
    receipt_duplicate_pairs: tuple[ReceiptDuplicatePair, ...] = ()
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()
    provider_usage: ProviderUsage | None = None
    recovery: RecoveryMetadata | None = None
    worker_release_revision: str | None = None
    semantic_validation: dict[str, Any] | None = None
    validation_recovery_attempted: bool | None = None

    @property
    def table_candidate_evaluations(self) -> tuple[TableCandidateScore, ...]:
        """Compatibility alias for the score records' descriptive name."""

        return self.table_candidate_scores

    @property
    def m5_match_edges(self) -> tuple[TableMatchEdge, ...]:
        return self.table_match_edges

    @property
    def m5_candidate_scores(self) -> tuple[TableCandidateScore, ...]:
        return self.table_candidate_scores

    @property
    def m5_selection_runs(self) -> tuple[TableSelectionRun, ...]:
        return self.table_selection_runs

    @model_validator(mode="after")
    def validate_v6_references(self) -> ExtractionResultV6:
        manifest = self.artifact_manifest.by_id()
        page_artifact_ids = [page.artifact.artifact_id for page in self.page_artifacts]
        if len(page_artifact_ids) != len(set(page_artifact_ids)):
            raise ValueError("page artifact wrappers must reference unique artifacts")
        for page in self.page_artifacts:
            expected = manifest.get(page.artifact.artifact_id)
            if expected is None:
                raise ValueError("page artifact is missing from artifact manifest")
            if page.artifact != expected:
                raise ValueError("embedded page artifact differs from artifact manifest")
        page_wrapper_by_id = {page.artifact.artifact_id: page for page in self.page_artifacts}
        if len({run.page_number for run in self.uvdoc_shadow_runs}) != len(self.uvdoc_shadow_runs):
            raise ValueError("UVDoc shadow runs must be unique by page")
        for run in self.uvdoc_shadow_runs:
            if run.status is not UvdocShadowStatus.VALID:
                continue
            input_artifact = manifest.get(run.input_artifact_id or "")
            uvdoc_artifact = manifest.get(run.uvdoc_artifact_id or "")
            enhanced_artifact = manifest.get(run.enhanced_artifact_id or "")
            if (
                input_artifact is None
                or input_artifact.artifact_kind is not ArtifactKind.ORIENTED_RAW
            ):
                raise ValueError("UVDoc shadow input must be an ORIENTED_RAW artifact")
            if (
                uvdoc_artifact is None
                or uvdoc_artifact.artifact_kind is not ArtifactKind.UVDOC
                or uvdoc_artifact.parent_artifact_id != input_artifact.artifact_id
                or not isinstance(uvdoc_artifact.child_to_parent_mapping, DenseBackwardGridMapping)
            ):
                raise ValueError("UVDoc shadow artifact has invalid dense lineage")
            if (
                enhanced_artifact is None
                or enhanced_artifact.artifact_kind is not ArtifactKind.UVDOC_ENHANCED
                or enhanced_artifact.parent_artifact_id != uvdoc_artifact.artifact_id
                or not isinstance(enhanced_artifact.child_to_parent_mapping, IdentityMapping)
            ):
                raise ValueError("enhanced UVDoc artifact has invalid identity lineage")
            for artifact in (uvdoc_artifact, enhanced_artifact):
                wrapper = page_wrapper_by_id.get(artifact.artifact_id)
                if wrapper is None or wrapper.selected or wrapper.role != "CANDIDATE":
                    raise ValueError("UVDoc shadow artifacts must be unselected page candidates")
        shadow_artifact_ids = {
            artifact_id
            for run in self.uvdoc_shadow_runs
            for artifact_id in (run.uvdoc_artifact_id, run.enhanced_artifact_id)
            if artifact_id is not None
        }
        self._validate_m5_records(manifest, shadow_artifact_ids)
        for table in self.canonical_table_artifacts:
            expected = manifest.get(table.artifact.artifact_id)
            if expected is None:
                raise ValueError("canonical table artifact is missing from artifact manifest")
            if table.artifact != expected:
                raise ValueError("embedded table artifact differs from artifact manifest")
            page = manifest.get(table.page_artifact_id)
            if page is None:
                raise ValueError("canonical table page artifact is missing from artifact manifest")
            if table.artifact.parent_artifact_id != table.page_artifact_id:
                raise ValueError("canonical table artifact parent differs from page artifact")
            if table.page_artifact_id in shadow_artifact_ids:
                raise ValueError("UVDoc shadow artifacts cannot own canonical tables")
        token_by_id = {item.token_id: item for item in self.token_manifest}
        if any(item.artifact_id not in manifest for item in self.token_manifest):
            raise ValueError("token artifact is missing from artifact manifest")
        if any(item.artifact_id in shadow_artifact_ids for item in self.token_manifest):
            raise ValueError("UVDoc shadow artifacts cannot own authoritative tokens")
        if len({item.token_id for item in self.token_manifest}) != len(self.token_manifest):
            raise ValueError("V6 token IDs must be unique")
        if any(item.input_artifact_id not in manifest for item in self.adapter_inputs):
            raise ValueError("adapter input artifact is missing from artifact manifest")
        if any(item.input_artifact_id in shadow_artifact_ids for item in self.adapter_inputs):
            raise ValueError("UVDoc shadow artifacts cannot be authoritative adapter inputs")
        table_ids = {(item.page_number, item.table_id) for item in self.source_tables}
        crop_by_key = {
            (item.page_number, item.logical_table_id): item
            for item in self.canonical_table_artifacts
        }
        if len(crop_by_key) != len(self.canonical_table_artifacts):
            raise ValueError("canonical table artifact identities must be unique")
        crop_ids = set(crop_by_key)
        if not table_ids.issubset(crop_ids):
            raise ValueError("every V6 source table requires a canonical table artifact")
        for adapter in self.adapter_inputs:
            table = crop_by_key.get((adapter.page_number, adapter.logical_table_id))
            if table is None:
                raise ValueError("adapter input references an unknown logical table")
            artifact = manifest[adapter.input_artifact_id]
            if artifact.image_sha256 != adapter.input_artifact_sha256:
                raise ValueError("adapter input hash does not match its artifact")
            while artifact.artifact_id != table.artifact.artifact_id:
                if artifact.parent_artifact_id is None:
                    raise ValueError("adapter input is outside its canonical table lineage")
                artifact = manifest[artifact.parent_artifact_id]

        referenced: set[str] = set(page_artifact_ids)
        referenced.update(item.artifact.artifact_id for item in self.canonical_table_artifacts)
        referenced.update(item.page_artifact_id for item in self.canonical_table_artifacts)
        referenced.update(item.artifact_id for item in self.token_manifest)
        referenced.update(item.source_page_artifact_id for item in self.token_manifest)
        referenced.update(item.input_artifact_id for item in self.adapter_inputs)
        # M5 observations are not authoritative, but their artifact references
        # still keep the candidate nodes in the manifest closure.  UVDoc IDs
        # are therefore retained for audit while the checks above prevent them
        # from owning any canonical output.
        referenced.update(item.anchor_artifact_id for item in self.table_match_edges)
        referenced.update(item.candidate_artifact_id for item in self.table_match_edges)
        referenced.update(item.candidate_artifact_id for item in self.table_candidate_scores)
        for run in self.table_selection_runs:
            for table in run.tables:
                referenced.update(edge.anchor_artifact_id for edge in table.match_edges)
                referenced.update(edge.candidate_artifact_id for edge in table.match_edges)
                referenced.update(
                    item.candidate_artifact_id for item in table.candidate_evaluations
                )

        evidence_items: list[EvidenceRefV2] = list(self.evidence)
        for row in self.rows:
            evidence_items.extend(row.evidence)
            evidence_items.extend(item for values in row.field_evidence.values() for item in values)
        for table in self.source_tables:
            evidence_items.extend(item for column in table.columns for item in column.evidence)
            evidence_items.extend(
                item for row in table.rows for cell in row.cells for item in cell.evidence
            )
        if self.document_total is not None:
            evidence_items.append(self.document_total.evidence)
        for candidate in self.raw_total_candidates:
            evidence_items.append(candidate.total.evidence)
            evidence_items.extend(candidate.context_evidence)
        for evidence in evidence_items:
            if evidence.artifact_id in shadow_artifact_ids:
                raise ValueError("UVDoc shadow artifacts cannot own authoritative evidence")
            referenced.add(evidence.artifact_id)
            referenced.add(evidence.source_page_artifact_id)
            artifact = manifest.get(evidence.artifact_id)
            if artifact is None or evidence.artifact_sha256 != artifact.image_sha256:
                raise ValueError("evidence artifact hash does not match manifest")
            for token_id in evidence.ocr_token_ids:
                token = token_by_id.get(token_id)
                if token is None:
                    raise ValueError("evidence references an unknown OCR token")
                if token.artifact_id != evidence.artifact_id:
                    raise ValueError("evidence token belongs to another artifact")
        for token in self.token_manifest:
            artifact = manifest.get(token.artifact_id)
            if artifact is None or token.artifact_sha256 != artifact.image_sha256:
                raise ValueError("token artifact hash does not match manifest")
            if token.artifact_relative_path != artifact.artifact_relative_path:
                raise ValueError("token artifact path does not match manifest")

        closure = set(referenced)
        pending = list(referenced)
        while pending:
            artifact_id = pending.pop()
            artifact = manifest.get(artifact_id)
            if artifact is None:
                continue
            parent_id = artifact.parent_artifact_id
            if parent_id is not None and parent_id not in closure:
                closure.add(parent_id)
                pending.append(parent_id)
        if closure != set(manifest):
            raise ValueError("artifact manifest contains unreferenced orphan nodes")
        return self

    def _validate_m5_records(
        self,
        manifest: dict[str, ArtifactRef],
        shadow_artifact_ids: set[str],
    ) -> None:
        """Validate M5 observation references without promoting shadow data.

        This runs inside the V6 envelope validator so an M5 record cannot be
        parsed successfully in isolation and then become authoritative merely
        because it is copied into a result payload.
        """

        expected_kinds = {
            variant: ArtifactKind(variant.value.upper()) for variant in TableCandidateVariant
        }
        known_logical_tables = {
            (item.page_number, item.logical_table_id) for item in self.canonical_table_artifacts
        }
        known_logical_tables.update(
            (item.page_number, item.table_id) for item in self.source_tables
        )
        # A standalone shadow fixture may contain only M5 records.  Once an
        # authoritative table is declared, however, every M5 reference must
        # point at one of those declared logical tables.
        enforce_logical_reference = bool(known_logical_tables)

        def variant_for_artifact(artifact_id: str) -> TableCandidateVariant:
            artifact = manifest.get(artifact_id)
            if artifact is None:
                raise ValueError("M5 record references an unknown artifact")
            try:
                return TableCandidateVariant(artifact.artifact_kind.value)
            except ValueError as error:
                raise ValueError(
                    "M5 match edge artifact must be a page candidate variant"
                ) from error

        def check_artifact(
            artifact_id: str,
            *,
            variant: TableCandidateVariant,
            page_number: int,
            selected: bool,
        ) -> None:
            artifact = manifest.get(artifact_id)
            if artifact is None:
                raise ValueError("M5 record references an unknown artifact")
            if artifact.artifact_kind is not expected_kinds[variant]:
                raise ValueError("M5 candidate variant does not match artifact kind")
            if page_number > self.pages:
                raise ValueError("M5 record page number exceeds envelope pages")

        def check_logical(page_number: int, logical_table_id: str) -> None:
            if enforce_logical_reference and (
                page_number,
                logical_table_id,
            ) not in known_logical_tables:
                raise ValueError("M5 record references an unknown logical table")

        edge_ids: set[str] = set()
        for edge in self.table_match_edges:
            if edge.edge_id in edge_ids:
                raise ValueError("M5 match edge IDs must be unique")
            edge_ids.add(edge.edge_id)
            check_logical(edge.page_number, edge.logical_table_id)
            check_artifact(
                edge.anchor_artifact_id,
                variant=variant_for_artifact(edge.anchor_artifact_id),
                page_number=edge.page_number,
                selected=False,
            )
            check_artifact(
                edge.candidate_artifact_id,
                variant=edge.candidate_variant,
                page_number=edge.page_number,
                selected=False,
            )

        score_ids: set[str] = set()
        for score in self.table_candidate_scores:
            if score.evaluation_id in score_ids:
                raise ValueError("M5 candidate evaluation IDs must be unique")
            score_ids.add(score.evaluation_id)
            check_logical(score.page_number, score.logical_table_id)
            check_artifact(
                score.candidate_artifact_id,
                variant=score.candidate_variant,
                page_number=score.page_number,
                selected=score.selected,
            )

        run_ids: set[str] = set()
        for run in self.table_selection_runs:
            if run.run_id in run_ids:
                raise ValueError("M5 selection run IDs must be unique")
            run_ids.add(run.run_id)
            if run.source_sha256 is not None and run.source_sha256 != self.source_sha256:
                raise ValueError("M5 selection run source hash differs from envelope")
            for table in run.tables:
                # Shadow runs introduce the stable source-derived M5 identity
                # alongside legacy sequential table IDs.  Only an enabled run
                # may claim that its logical ID already owns authoritative V6
                # output.
                if run.mode == "enabled":
                    check_logical(table.page_number, table.logical_table_id)
                for edge in table.match_edges:
                    if run.mode == "enabled":
                        check_logical(edge.page_number, edge.logical_table_id)
                    check_artifact(
                        edge.anchor_artifact_id,
                        variant=variant_for_artifact(edge.anchor_artifact_id),
                        page_number=edge.page_number,
                        selected=False,
                    )
                    check_artifact(
                        edge.candidate_artifact_id,
                        variant=edge.candidate_variant,
                        page_number=edge.page_number,
                        selected=False,
                    )
                for score in table.candidate_evaluations:
                    if run.mode == "enabled":
                        check_logical(score.page_number, score.logical_table_id)
                    check_artifact(
                        score.candidate_artifact_id,
                        variant=score.candidate_variant,
                        page_number=score.page_number,
                        selected=score.selected,
                    )


__all__ = [
    "ArtifactKind",
    "MappingType",
    "ArtifactMapping",
    "IdentityMapping",
    "HomographyMapping",
    "DenseBackwardGridMapping",
    "ArtifactRef",
    "ArtifactManifest",
    "PageArtifact",
    "TableCandidateVariant",
    "TableMatchDecision",
    "TableSelectionDecision",
    "TableMatchEdge",
    "TableMatchEdgeV1",
    "TableCandidateScore",
    "TableCandidateScoreV1",
    "TableCandidateEvaluation",
    "TableCandidateEvaluationV1",
    "LogicalTableSelection",
    "TableSelectionTable",
    "TableSelectionRun",
    "TableSelectionRunV1",
    "UvdocShadowRun",
    "UvdocShadowStatus",
    "CanonicalTableArtifact",
    "EvidenceRefV2",
    "TokenManifestEntryV2",
    "TableAdapterInputV2",
    "SourceColumnV2",
    "SourceCellV2",
    "SourceRowV2",
    "SourceTableV2",
    "DocumentTotalV2",
    "RawTotalCandidateV2",
    "CanonicalRowV2",
    "ExtractionResultV6",
    "canonical_json",
    "canonical_sha256",
    "artifact_id_for",
]
