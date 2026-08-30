from enum import StrEnum

from pydantic import Field, model_validator

from gmoney.contracts.common import ContractModel


class Point(ContractModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Polygon(ContractModel):
    points: tuple[Point, ...]

    @model_validator(mode="after")
    def require_polygon(self) -> "Polygon":
        if len(self.points) < 3:
            raise ValueError("a polygon needs at least three points")
        return self


class TransformChain(ContractModel):
    page_number: int = Field(ge=1)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    derived_width: int = Field(gt=0)
    derived_height: int = Field(gt=0)
    forward_matrix: tuple[tuple[float, float, float], ...]
    inverse_matrix: tuple[tuple[float, float, float], ...]
    operations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_three_by_three_matrices(self) -> "TransformChain":
        for matrix in (self.forward_matrix, self.inverse_matrix):
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("transform matrices must be 3x3")
        return self


class PageQualityFlag(StrEnum):
    BLUR = "blur"
    LOW_CONTRAST = "low_contrast"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    LOW_EDGE_DENSITY = "low_edge_density"
    SKEWED = "skewed"
    ORIENTATION_UNVERIFIED = "orientation_unverified"


class PageQuality(ContractModel):
    page_number: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    mean_luminance: float = Field(ge=0, le=255)
    contrast_stddev: float = Field(ge=0)
    laplacian_variance: float = Field(ge=0)
    edge_density: float = Field(ge=0, le=1)
    estimated_skew_degrees: float
    orientation_degrees: int = 0
    flags: tuple[PageQualityFlag, ...] = ()


class PreprocessingVariant(StrEnum):
    RAW = "raw"
    GEOMETRY_300 = "geometry_300"
    CAMERA_400 = "camera_400"


class PreprocessingCandidate(ContractModel):
    variant: PreprocessingVariant
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_relative_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    transform: TransformChain
    quality: PageQuality
    route_reasons: tuple[str, ...] = ()
    selected: bool = False
    selection_reason: str | None = None
    ocr_token_count: int | None = Field(default=None, ge=0)
    ocr_median_confidence: float | None = Field(default=None, ge=0, le=1)
    financial_token_count: int | None = Field(default=None, ge=0)
    table_proposal_count: int | None = Field(default=None, ge=0)
    reconstruction_score: tuple[int, ...] | None = None
    ocr_latency_ms: int | None = Field(default=None, ge=0)
    layout_latency_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_consistent_geometry(self) -> "PreprocessingCandidate":
        if self.transform.page_number != self.quality.page_number:
            raise ValueError("preprocessing transform and quality page numbers differ")
        if (self.width, self.height) != (
            self.transform.derived_width,
            self.transform.derived_height,
        ):
            raise ValueError("preprocessing candidate dimensions differ from its transform")
        if (self.width, self.height, self.dpi) != (
            self.quality.width,
            self.quality.height,
            self.quality.dpi,
        ):
            raise ValueError("preprocessing candidate dimensions differ from its quality record")
        if self.artifact_sha256 != self.quality.artifact_sha256:
            raise ValueError("preprocessing candidate and quality hashes differ")
        if self.reconstruction_score is not None and len(self.reconstruction_score) != 7:
            raise ValueError("preprocessing reconstruction score must contain seven metrics")
        return self


class PagePreprocessingRecord(ContractModel):
    policy_version: str = "camera_preprocessing_v1"
    page_number: int = Field(ge=1)
    raw_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_artifact_relative_path: str
    raw_quality: PageQuality
    orientation_degrees: int = 0
    orientation_confidence: float = Field(default=0, ge=0, le=1)
    candidates: tuple[PreprocessingCandidate, ...]
    selected_variant: PreprocessingVariant

    @model_validator(mode="after")
    def require_one_selected_candidate(self) -> "PagePreprocessingRecord":
        if self.raw_quality.page_number != self.page_number:
            raise ValueError("preprocessing record and raw quality page numbers differ")
        if self.raw_quality.artifact_sha256 != self.raw_artifact_sha256:
            raise ValueError("preprocessing record and raw quality hashes differ")
        if not self.candidates:
            raise ValueError("preprocessing record requires at least the raw candidate")
        if self.orientation_degrees not in {0, 90, 180, 270}:
            raise ValueError("preprocessing orientation must be a right angle")
        if any(
            candidate.transform.page_number != self.page_number for candidate in self.candidates
        ):
            raise ValueError("preprocessing candidate belongs to another page")
        if any(
            (
                candidate.transform.source_width,
                candidate.transform.source_height,
            )
            != (self.raw_quality.width, self.raw_quality.height)
            for candidate in self.candidates
        ):
            raise ValueError("preprocessing candidate source dimensions differ from raw page")
        raw = tuple(
            candidate
            for candidate in self.candidates
            if candidate.variant is PreprocessingVariant.RAW
        )
        if len(raw) != 1:
            raise ValueError("preprocessing record requires exactly one raw candidate")
        raw_candidate = raw[0]
        identity_matrix = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        if (
            raw_candidate.artifact_sha256 != self.raw_artifact_sha256
            or raw_candidate.artifact_relative_path != self.raw_artifact_relative_path
            or raw_candidate.quality != self.raw_quality
            or raw_candidate.transform.forward_matrix != identity_matrix
            or raw_candidate.transform.inverse_matrix != identity_matrix
            or raw_candidate.transform.operations
        ):
            raise ValueError("raw preprocessing candidate is not the canonical raw page")
        selected = tuple(candidate for candidate in self.candidates if candidate.selected)
        if len(selected) != 1 or selected[0].variant is not self.selected_variant:
            raise ValueError("preprocessing record requires exactly one selected candidate")
        if len({candidate.variant for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("preprocessing candidate variants must be unique")
        return self


class PageAsset(ContractModel):
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(ge=1)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    relative_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    renderer: str
    renderer_version: str


class TableRegionSource(StrEnum):
    LAYOUT_MODEL = "layout_model"
    OCR_GEOMETRY = "ocr_geometry"
    HEAVY_MODEL = "heavy_model"
    REVIEWER = "reviewer"


class TableRegion(ContractModel):
    region_version: str = "table_region_v1"
    page_number: int = Field(ge=1)
    source: TableRegionSource
    polygon: Polygon
    confidence: float = Field(ge=0, le=1)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_token_ids: tuple[str, ...] = ()


class OcrToken(ContractModel):
    token_id: str
    page_number: int = Field(ge=1)
    text: str
    confidence: float = Field(ge=0, le=1)
    polygon: Polygon
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_name: str
    model_version: str
