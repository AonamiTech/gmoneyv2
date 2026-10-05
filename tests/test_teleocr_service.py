from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from functools import cache
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gmoney.inference.contracts import InferenceRequest
from gmoney.inference.paddle import TELEOCR_TABLE_PROMPT, TeleOcrAdapter

ROOT = Path(__file__).resolve().parents[1]
RECORDED = (ROOT / "tests" / "fixtures" / "teleocr" / "bill6_page2_procedure.otsl").read_text()


@cache
def _service_module():
    spec = importlib.util.spec_from_file_location(
        "teleocr_service", ROOT / "services" / "teleocr" / "teleocr_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _png(width: int = 40, height: int = 20) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    dtype_name = "float16"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], str, int]] = []

    def generate(self, image, prompt, max_new_tokens):
        self.calls.append((image.size, prompt, max_new_tokens))
        return _service_module().Generation(RECORDED, 900, 120)


def test_service_serves_openai_compatible_chat_completions() -> None:
    service = _service_module()
    engine = FakeEngine()
    with TestClient(service.create_app(lambda: engine)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["dtype"] == "float16"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "StarDoc-AI/TeleOCR",
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TELEOCR_TABLE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(_png()).decode()
                                },
                            },
                        ],
                    }
                ],
            },
        )
    assert response.status_code == 200
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    assert content.endswith("<fcel>890.00<nl>")
    assert "<|im_end|>" not in content
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {
        "prompt_tokens": 900,
        "completion_tokens": 120,
        "total_tokens": 1020,
    }
    assert engine.calls == [((40, 20), TELEOCR_TABLE_PROMPT, 8192)]


def test_service_rejects_requests_without_one_image_and_reports_unloaded_model() -> None:
    service = _service_module()
    with TestClient(service.create_app(lambda: FakeEngine())) as client:
        missing = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        )
        assert missing.status_code == 422

    def broken():
        raise RuntimeError("weights not mounted")

    with TestClient(service.create_app(broken)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert "weights not mounted" in health.json()["error"]


@pytest.mark.parametrize(
    ("requested", "cuda", "bf16", "expected"),
    [
        ("auto", True, False, "float16"),
        ("auto", True, True, "bfloat16"),
        ("auto", False, False, "float32"),
        ("bfloat16", True, False, "bfloat16"),
    ],
)
def test_service_dtype_policy(requested: str, cuda: bool, bf16: bool, expected: str) -> None:
    assert (
        _service_module().resolve_dtype_name(requested, cuda=cuda, bf16_supported=bf16) == expected
    )


def test_teleocr_client_sends_table_prompt_and_strips_end_marker(tmp_path: Path) -> None:
    image = tmp_path / "p2-t1.png"
    image.write_bytes(_png())
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": RECORDED + "\n"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 120},
            },
        )

    adapter = TeleOcrAdapter(base_url="http://teleocr.test")
    adapter._client = httpx.Client(
        base_url="http://teleocr.test", transport=httpx.MockTransport(handler)
    )
    response = adapter.predict(
        InferenceRequest(
            request_id="r1",
            artifact_sha256="a" * 64,
            image_path=str(image),
            page_number=2,
        )
    )
    assert seen["path"] == "/v1/chat/completions"
    texts = [
        part["text"] for part in seen["body"]["messages"][0]["content"] if part["type"] == "text"
    ]
    assert texts == [TELEOCR_TABLE_PROMPT]
    assert response.output["content"].endswith("<nl>")
    assert response.spec.model_name == "StarDoc-AI/TeleOCR"
    assert response.output["truncated"] is False
