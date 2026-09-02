"""Strict V6 artifact, evidence, and envelope contracts.

The V5 contracts deliberately remain unchanged.  V6 makes the image used by an
adapter a first-class, content-addressed node in an immutable coordinate graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, TypeAdapter, field_validator, model_validator

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
    """Reserved M3 mapping schema; V6 envelopes reject this mapping type."""

    mapping_type: Literal[MappingType.DENSE_BACKWARD_GRID] = MappingType.DENSE_BACKWARD_GRID
    grid_relative_path: str = Field(min_length=1)
    grid_sha256: str = Field(pattern=SHA256_PATTERN)
    grid_dtype: Literal["float32"] = "float32"
    grid_shape: tuple[int, int, int] = Field(min_length=3, max_length=3)
    child_width: int = Field(gt=0)
    child_height: int = Field(gt=0)
    parent_width: int = Field(gt=0)
    parent_height: int = Field(gt=0)
    coordinate_domain: Literal["normalized_minus_one_to_one"] = "normalized_minus_one_to_one"
    interpolation: Literal["bilinear"] = "bilinear"
    align_corners: Literal[True] = True
    padding_mode: Literal["zeros", "border", "reflection"]
    mapping_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @field_validator("grid_shape")
    @classmethod
    def require_positive_shape(cls, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(value <= 0 for value in shape):
            raise ValueError("dense grid dimensions must be positive")
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
        if self.raw_value and self.raw_value.strip() and not any(
            item.ocr_token_ids for item in self.evidence
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

    @model_validator(mode="after")
    def validate_v6_references(self) -> ExtractionResultV6:
        manifest = self.artifact_manifest.by_id()
        if any(
            isinstance(item.child_to_parent_mapping, DenseBackwardGridMapping)
            for item in self.artifact_manifest.artifacts
        ):
            raise ValueError("dense backward-grid mappings are reserved for M3")
        page_artifact_ids = [page.artifact.artifact_id for page in self.page_artifacts]
        if len(page_artifact_ids) != len(set(page_artifact_ids)):
            raise ValueError("page artifact wrappers must reference unique artifacts")
        page_numbers = {item.page_number for item in self.page_artifacts}
        if any(item.role is not None for item in self.page_artifacts) and page_numbers != set(
            range(1, self.pages + 1)
        ):
            raise ValueError("page artifact numbers must cover the declared page range")
        for page in self.page_artifacts:
            expected = manifest.get(page.artifact.artifact_id)
            if expected is None:
                raise ValueError("page artifact is missing from artifact manifest")
            if page.artifact != expected:
                raise ValueError("embedded page artifact differs from artifact manifest")
        if self.page_artifacts and any(item.role is not None for item in self.page_artifacts):
            for page_number in sorted({item.page_number for item in self.page_artifacts}):
                page_items = tuple(
                    item for item in self.page_artifacts if item.page_number == page_number
                )
                if sum(item.role == "SOURCE_RAW" for item in page_items) != 1:
                    raise ValueError("each page requires exactly one SOURCE_RAW inventory item")
                if sum(item.role == "ORIENTED_RAW" for item in page_items) != 1:
                    raise ValueError("each page requires exactly one ORIENTED_RAW inventory item")
                if sum(item.selected for item in page_items) != 1:
                    raise ValueError("each page requires exactly one selected candidate")
                source = next(item for item in page_items if item.role == "SOURCE_RAW")
                oriented = next(item for item in page_items if item.role == "ORIENTED_RAW")
                if oriented.artifact.parent_artifact_id != source.artifact.artifact_id:
                    raise ValueError("ORIENTED_RAW must directly reference its page SOURCE_RAW")
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
        token_by_id = {item.token_id: item for item in self.token_manifest}
        if any(item.artifact_id not in manifest for item in self.token_manifest):
            raise ValueError("token artifact is missing from artifact manifest")
        if len({item.token_id for item in self.token_manifest}) != len(self.token_manifest):
            raise ValueError("V6 token IDs must be unique")
        if any(item.input_artifact_id not in manifest for item in self.adapter_inputs):
            raise ValueError("adapter input artifact is missing from artifact manifest")
        table_ids = {(item.page_number, item.table_id) for item in self.source_tables}
        crop_ids = {
            (item.page_number, item.logical_table_id)
            for item in self.canonical_table_artifacts
        }
        if not table_ids.issubset(crop_ids):
            raise ValueError("every V6 source table requires a canonical table artifact")

        referenced: set[str] = set(page_artifact_ids)
        referenced.update(item.artifact.artifact_id for item in self.canonical_table_artifacts)
        referenced.update(item.page_artifact_id for item in self.canonical_table_artifacts)
        referenced.update(item.artifact_id for item in self.token_manifest)
        referenced.update(item.source_page_artifact_id for item in self.token_manifest)
        referenced.update(item.input_artifact_id for item in self.adapter_inputs)

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
