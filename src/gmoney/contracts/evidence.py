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
