from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from gmoney.inference.contracts import (
    InferenceRequest,
    InferenceResponse,
    ModelKind,
    ModelSpec,
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "json"):
        payload = value.json
        return payload() if callable(payload) else payload
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _require_output(output: list[Any], model_name: str) -> list[Any]:
    if not output:
        raise RuntimeError(f"{model_name} returned no results")
    return output


class PaddleOcrV6Adapter:
    spec = ModelSpec(
        kind=ModelKind.OCR,
        provider="paddleocr",
        model_name="PP-OCRv6-medium",
        model_version="PP-OCRv6",
        backend="paddle",
        device="cpu",
    )

    def __init__(self, device: str = "cpu") -> None:
        from paddleocr import PaddleOCR

        self.spec = type(self).spec.model_copy(update={"device": device})
        self._model = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="PP-OCRv6_medium_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        output = _require_output(
            list(self._model.predict(str(Path(request.image_path)))),
            self.spec.model_name,
        )
        elapsed = round((time.perf_counter() - started) * 1000)
        return InferenceResponse(
            request_id=request.request_id,
            spec=self.spec,
            output={"pages": _jsonable(output)},
            latency_ms=elapsed,
            peak_rss_bytes=psutil.Process().memory_info().rss,
            memory_scope="process",
        )


class PaddleDocLayoutV3Adapter:
    spec = ModelSpec(
        kind=ModelKind.LAYOUT,
        provider="paddleocr",
        model_name="PP-DocLayoutV3",
        model_version="PP-DocLayoutV3",
        backend="paddle",
        device="cpu",
    )

    def __init__(self, device: str = "cpu") -> None:
        from paddleocr import LayoutDetection

        self.spec = type(self).spec.model_copy(update={"device": device})
        self._model = LayoutDetection(model_name="PP-DocLayoutV3", device=device)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        output = _require_output(
            list(self._model.predict(str(Path(request.image_path)))),
            self.spec.model_name,
        )
        elapsed = round((time.perf_counter() - started) * 1000)
        return InferenceResponse(
            request_id=request.request_id,
            spec=self.spec,
            output={"pages": _jsonable(output)},
            latency_ms=elapsed,
            peak_rss_bytes=psutil.Process().memory_info().rss,
            memory_scope="process",
        )


class PaddleWirelessTableAdapter:
    spec = ModelSpec(
        kind=ModelKind.TABLE,
        provider="paddleocr",
        model_name="SLANeXt_wireless",
        model_version="TableStructureRecognition-v2",
        backend="paddle",
        device="cpu",
    )

    def __init__(self, device: str = "cpu") -> None:
        from paddleocr import TableStructureRecognition

        self.spec = type(self).spec.model_copy(update={"device": device})
        self._model = TableStructureRecognition(
            model_name="SLANeXt_wireless",
            device=device,
        )

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        output = _require_output(
            list(self._model.predict(str(Path(request.image_path)))),
            self.spec.model_name,
        )
        elapsed = round((time.perf_counter() - started) * 1000)
        return InferenceResponse(
            request_id=request.request_id,
            spec=self.spec,
            output={"pages": _jsonable(output)},
            latency_ms=elapsed,
            peak_rss_bytes=psutil.Process().memory_info().rss,
            memory_scope="process",
        )


class PaddleOcrVlAdapter:
    spec = ModelSpec(
        kind=ModelKind.DOCUMENT_VLM,
        provider="paddleocr",
        model_name="PaddleOCR-VL-1.6-GGUF",
        model_version="1.6",
        backend="llama.cpp",
        device="cpu",
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8111",
        timeout_seconds: float = 900,
        device: str = "cpu",
    ) -> None:
        self.spec = type(self).spec.model_copy(update={"device": device})
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        image_path = Path(request.image_path)
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = str(
            request.options.get(
                "prompt",
                "Table Recognition:",
            )
        )
        max_tokens = int(request.options.get("max_tokens", 16384))
        started = time.perf_counter()
        response = self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.spec.model_name,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                            },
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("PaddleOCR-VL returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("PaddleOCR-VL returned empty content")
        finish_reason = choices[0].get("finish_reason")
        usage = payload.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        repeated_tail = bool(
            len(content) >= 512
            and content[-256:].strip()
            and content[-256:].strip() in content[:-256]
        )
        truncated = bool(
            finish_reason in {"length", "max_tokens"}
            or (completion_tokens and completion_tokens >= max_tokens)
            or repeated_tail
        )
        elapsed = round((time.perf_counter() - started) * 1000)
        return InferenceResponse(
            request_id=request.request_id,
            spec=self.spec,
            output={
                "content": content,
                "usage": usage,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "repeated_tail": repeated_tail,
            },
            latency_ms=elapsed,
            peak_rss_bytes=None,
            memory_scope="remote_unmeasured",
        )
