from pathlib import Path

import httpx
import pytest

from gmoney.inference.contracts import InferenceRequest
from gmoney.inference.paddle import PaddleOcrVlAdapter, _require_output


def test_empty_model_output_is_never_a_success() -> None:
    with pytest.raises(RuntimeError, match="returned no results"):
        _require_output([], "model")


def test_vl_adapter_rejects_empty_provider_output(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"not-decoded-by-mock")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    adapter = PaddleOcrVlAdapter()
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    with pytest.raises(RuntimeError, match="no choices"):
        adapter.predict(
            InferenceRequest(
                request_id="request-1",
                artifact_sha256="a" * 64,
                image_path=str(image),
                page_number=1,
            )
        )
