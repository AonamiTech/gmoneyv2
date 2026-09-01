from __future__ import annotations

import hashlib
import json
import mimetypes
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gmoney.contracts.phase3 import (
    AdjudicationRequest,
    AdjudicationResponse,
    AdjudicationRow,
    GeminiPromotionDecision,
)


class AdjudicationAdapter(Protocol):
    model: str
    config_identity: str

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse: ...


class _PayloadField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    raw_value: str
    token_ids: list[str] = Field(min_length=1)
    polygon: list[dict[str, float]] = Field(min_length=3)
    confidence: float | None = Field(default=None, ge=0, le=1)


class _PayloadRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_order: int = Field(ge=0)
    role: str
    fields: list[_PayloadField]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[_PayloadRow]


def _usage(response: Any, name: str) -> int:
    metadata = getattr(response, "usage_metadata", None)
    return int(getattr(metadata, name, 0) or 0)


def _prompt(request: AdjudicationRequest) -> str:
    tokens = [
        {
            "token_id": token.token_id,
            "text": token.text,
            "polygon": [point.model_dump() for point in token.polygon.points],
        }
        for token in request.tokens
    ]
    context = {
        "page_type": request.page_type,
        "table_type": request.table_type,
        "unresolved_reasons": request.unresolved_reasons,
        "ocr_tokens": tokens,
        "candidate_rows": request.candidate_rows,
    }
    return (
        "Recover only visible hospital-bill ledger rows from this redacted table crop. "
        "Return no field that is not supported by listed OCR token IDs and an in-crop polygon. "
        "Do not infer or repair financial values from arithmetic. Preserve negative rows and "
        "legitimate duplicates. Use only canonical field names description, amount, quantity, "
        "rate, discount, service_date, request_no, service_code, hsn_code, and section.\n\n"
        + json.dumps(context, default=str, separators=(",", ":"))
    )


class GeminiAdjudicationAdapter:
    provider = "google_genai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash",
        timeout_seconds: float = 120,
        input_cost_usd_per_million: float = 0,
        output_cost_usd_per_million: float = 0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Gemini API key is required")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.input_cost = Decimal(str(input_cost_usd_per_million))
        self.output_cost = Decimal(str(output_cost_usd_per_million))
        self.config_identity = hashlib.sha256(
            json.dumps(
                {
                    "adapter": "google_genai_v1",
                    "model": model,
                    "temperature": 0,
                    "timeout_seconds": timeout_seconds,
                    "response_schema": _Payload.model_json_schema(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key, http_options={"timeout": timeout_seconds * 1000})
        self._client = client

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse:
        from google.genai import types

        image_path = Path(request.masked_crop_path)
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                _prompt(request),
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=_Payload.model_json_schema(),
            ),
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        text = str(getattr(response, "text", "") or "")
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        try:
            payload = _Payload.model_validate_json(text)
            rows = tuple(
                AdjudicationRow.model_validate(row.model_dump()) for row in payload.rows
            )
            rejected: tuple[str, ...] = ()
        except (ValidationError, ValueError) as error:
            rows = ()
            rejected = (f"invalid_response_schema:{type(error).__name__}",)
        input_tokens = _usage(response, "prompt_token_count")
        output_tokens = _usage(response, "candidates_token_count")
        cost = (
            self.input_cost * Decimal(input_tokens)
            + self.output_cost * Decimal(output_tokens)
        ) / Decimal(1_000_000)
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "") or None
        return AdjudicationResponse(
            request_id=request.request_id,
            canonical_crop_sha256=request.canonical_crop_sha256,
            provider=self.provider,
            model=self.model,
            rows=rows,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            measured_cost_usd=cost,
            finish_reason=finish_reason,
            rejected_reasons=rejected,
            raw_payload_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def validate_promotion(
    path: Path,
    *,
    model: str,
    prompt_version: str,
    redaction_version: str,
    schema_version: str = "phase3_adjudication_schema_v1",
) -> GeminiPromotionDecision:
    decision = GeminiPromotionDecision.model_validate_json(path.read_text())
    if not decision.approved:
        raise ValueError("Gemini promotion decision is not approved")
    expected = (model, prompt_version, schema_version, redaction_version)
    actual = (
        decision.model,
        decision.prompt_version,
        decision.schema_version,
        decision.redaction_version,
    )
    if actual != expected:
        raise ValueError("Gemini promotion decision does not match active configuration")
    return decision


def adjudication_cache_identity(
    request: AdjudicationRequest, model: str, config_identity: str = ""
) -> str:
    payload = {
        "masked_crop_sha256": request.masked_crop_sha256,
        "canonical_crop_sha256": request.canonical_crop_sha256,
        "model": model,
        "prompt_version": request.prompt_version,
        "schema_version": request.schema_version,
        "redaction_version": request.redaction_version,
        "adapter_config_identity": config_identity,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_adjudication(
    cache_path: Path,
    request: AdjudicationRequest,
    adapter: AdjudicationAdapter,
) -> tuple[AdjudicationResponse, bool]:
    identity = adjudication_cache_identity(
        request,
        adapter.model,
        getattr(adapter, "config_identity", ""),
    )
    if cache_path.exists():
        envelope = json.loads(cache_path.read_text())
        if envelope.get("cache_identity") == identity:
            try:
                response = AdjudicationResponse.model_validate(envelope["response"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if response.canonical_crop_sha256 == request.canonical_crop_sha256:
                    return response, True
    response = adapter.adjudicate(request)
    if response.canonical_crop_sha256 != request.canonical_crop_sha256:
        raise ValueError("adjudication response canonical crop hash differs from request")
    envelope = {
        "cache_identity": identity,
        "request_id": request.request_id,
        "response": response.model_dump(mode="json"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(cache_path)
    return response, False
