import pytest
from pydantic import ValidationError

from gmoney.inference.contracts import InferenceRequest, ModelKind, ModelSpec


def test_model_spec_is_explicit_and_versioned() -> None:
    spec = ModelSpec(
        kind=ModelKind.OCR,
        provider="paddleocr",
        model_name="PP-OCRv6-medium",
        model_version="PP-OCRv6",
        backend="paddle",
        device="cpu",
    )
    assert spec.kind is ModelKind.OCR


def test_inference_request_requires_artifact_hash() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(
            request_id="request-1",
            artifact_sha256="not-a-hash",
            image_path="page.png",
            page_number=1,
        )
