"""OpenAI-compatible TeleOCR table-reading service for one GPU.

Serves the Apache-2.0 ``StarDoc-AI/TeleOCR`` weights through Hugging Face transformers
(``transformers==4.57.1``; 5.x fails in ``ROPE_INIT_FUNCTIONS``).  The service exposes:

* ``POST /v1/chat/completions``: one user message with an image (base64 data URL) and a
  text prompt; greedy decoding; the reply text stops at ``<|im_end|>``.
* ``GET /health``: 200 once the model is loaded, 503 otherwise.

Requests are serialized per GPU (``TELEOCR_MAX_CONCURRENCY``, default 1).  The model is
loaded from a read-only mount (``TELEOCR_MODEL_PATH``); nothing is downloaded at runtime.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

MODEL_NAME = "StarDoc-AI/TeleOCR"
END_MARKER = "<|im_end|>"
DEFAULT_PROMPT = "This is the image of a table. Please output the table in OTSL format."


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Engine(Protocol):
    dtype_name: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> Generation: ...


def resolve_dtype_name(requested: str, *, cuda: bool, bf16_supported: bool) -> str:
    """float16 on GPUs without bf16 (T4 = sm_75), bfloat16 on newer GPUs, float32 on CPU."""
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        if requested not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported TELEOCR_DTYPE: {requested}")
        return requested
    if not cuda:
        return "float32"
    return "bfloat16" if bf16_supported else "float16"


class TransformersEngine:
    def __init__(self, model_path: str, *, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.dtype_name = resolve_dtype_name(
            dtype,
            cuda=cuda,
            bf16_supported=bool(cuda and torch.cuda.is_bf16_supported()),
        )
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.dtype_name),
            )
            .to(device if cuda else "cpu")
            .eval()
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> Generation:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=self.model.device, dtype=self.model.dtype)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[:, prompt_tokens:]
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return Generation(text, prompt_tokens, int(new_tokens.shape[1]))


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None


def decode_image(url: str, max_pixels: int) -> Image.Image:
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("image_url must be a base64 data URL")
    try:
        payload = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (binascii.Error, OSError) as error:
        raise ValueError("image_url is not a readable image") from error
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return image


def split_request(request: ChatRequest) -> tuple[str, str]:
    user = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if user is None or isinstance(user.content, str):
        raise ValueError("a user message with an image part is required")
    images = [part.image_url.url for part in user.content if part.image_url is not None]
    texts = [part.text for part in user.content if part.type == "text" and part.text]
    if len(images) != 1:
        raise ValueError("exactly one image is required")
    return images[0], "\n".join(text.strip() for text in texts) or DEFAULT_PROMPT


def clean_reply(text: str) -> str:
    return text.split(END_MARKER, 1)[0].strip()


def create_app(engine_factory: Any | None = None) -> FastAPI:
    max_concurrency = int(os.environ.get("TELEOCR_MAX_CONCURRENCY", "1"))
    default_max_tokens = int(os.environ.get("TELEOCR_MAX_NEW_TOKENS", "8192"))
    max_pixels = int(os.environ.get("TELEOCR_MAX_PIXELS", str(8000 * 8000)))
    state: dict[str, Any] = {"engine": None, "error": None}
    slots = threading.BoundedSemaphore(max(1, max_concurrency))

    def default_factory() -> Engine:
        return TransformersEngine(
            os.environ.get("TELEOCR_MODEL_PATH", "/models/teleocr"),
            device=os.environ.get("TELEOCR_DEVICE", "cuda:0"),
            dtype=os.environ.get("TELEOCR_DTYPE", "auto"),
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["engine"] = (engine_factory or default_factory)()
        except Exception as error:  # reported by /health instead of crashing the probe
            state["error"] = f"{type(error).__name__}: {error}"
        yield

    app = FastAPI(title="TeleOCR service", lifespan=lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        engine = state["engine"]
        if engine is None:
            return JSONResponse({"status": "unavailable", "error": state["error"]}, status_code=503)
        return JSONResponse(
            {
                "status": "ok",
                "model": MODEL_NAME,
                "dtype": engine.dtype_name,
                "max_concurrency": max_concurrency,
            }
        )

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        engine = state["engine"]
        if engine is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        try:
            image_url, prompt = split_request(request)
            image = decode_image(image_url, max_pixels)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        max_new_tokens = min(request.max_tokens or default_max_tokens, default_max_tokens)
        started = time.perf_counter()
        with slots:
            generation = engine.generate(image, prompt, max_new_tokens)
        content = clean_reply(generation.text)
        stopped = END_MARKER in generation.text
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": (
                        "stop"
                        if stopped or generation.completion_tokens < max_new_tokens
                        else "length"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.prompt_tokens + generation.completion_tokens,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }

    return app


# The model loads in the lifespan hook, so importing this module never touches weights.
app = create_app()
