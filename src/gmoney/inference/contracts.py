from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from gmoney.contracts.common import ContractModel


class ModelKind(StrEnum):
    ORIENTATION = "orientation"
    OCR = "ocr"
    LAYOUT = "layout"
    TABLE = "table"
    DOCUMENT_VLM = "document_vlm"
    ADJUDICATION = "adjudication"


class ModelSpec(ContractModel):
    kind: ModelKind
    provider: str
    model_name: str
    model_version: str
    backend: str
    device: str


class InferenceRequest(ContractModel):
    request_id: str
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    image_path: str
    page_number: int = Field(ge=1)
    options: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(ContractModel):
    request_id: str
    input_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    spec: ModelSpec
    output: dict[str, Any]
    latency_ms: int = Field(ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    memory_scope: Literal["process", "remote_unmeasured"]
