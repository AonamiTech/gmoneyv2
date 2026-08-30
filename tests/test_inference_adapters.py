from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from gmoney.inference.contracts import InferenceRequest
from gmoney.inference.paddle import (
    PaddleDocOrientationAdapter,
    PaddleOcrV6Adapter,
    PaddleOcrVlAdapter,
    _require_output,
)


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


def test_adapter_device_is_forwarded_and_recorded(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOcr:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOcr),
    )
    adapter = PaddleOcrV6Adapter(device="gpu:0")
    assert captured["device"] == "gpu:0"
    assert adapter.spec.device == "gpu:0"
    assert PaddleOcrVlAdapter(device="cuda:0").spec.device == "cuda:0"


def test_orientation_adapter_is_explicitly_kept_on_cpu(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOrientation:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "paddleocr",
        SimpleNamespace(DocImgOrientationClassification=FakeOrientation),
    )

    adapter = PaddleDocOrientationAdapter(device="cpu")

    assert captured == {"model_name": "PP-LCNet_x1_0_doc_ori", "device": "cpu"}
    assert adapter.spec.device == "cpu"
