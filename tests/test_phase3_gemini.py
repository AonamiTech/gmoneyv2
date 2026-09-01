from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType
from gmoney.contracts.phase3 import (
    AdjudicationField,
    AdjudicationRequest,
    AdjudicationResponse,
    AdjudicationRow,
    GeminiPromotionDecision,
    RecoveryReason,
)
from gmoney.extraction.recovery import ground_adjudication
from gmoney.inference.gemini import cached_adjudication, validate_promotion


def polygon(box: tuple[int, int, int, int]) -> Polygon:
    left, top, right, bottom = box
    return Polygon(
        points=(
            Point(x=left, y=top),
            Point(x=right, y=top),
            Point(x=right, y=bottom),
            Point(x=left, y=bottom),
        )
    )


def token(token_id: str, text: str, box: tuple[int, int, int, int]) -> OcrToken:
    return OcrToken(
        token_id=token_id,
        page_number=1,
        text=text,
        confidence=0.99,
        polygon=polygon(box),
        artifact_sha256="a" * 64,
        model_name="test",
        model_version="1",
    )


def request(tmp_path: Path, prompt: str = "p1") -> AdjudicationRequest:
    crop = tmp_path / "masked.png"
    crop.write_bytes(b"not-used-by-fake")
    return AdjudicationRequest(
        request_id="request-1",
        document_id="document-1",
        page_number=1,
        table_id="p1-t1",
        masked_crop_path=str(crop),
        masked_crop_sha256="b" * 64,
        canonical_crop_sha256="c" * 64,
        table_type=TableType.ITEM_LEDGER,
        tokens=(
            token("description", "Blood Test", (10, 10, 80, 25)),
            token("amount", "120.00", (120, 10, 170, 25)),
        ),
        unresolved_reasons=(RecoveryReason.ZERO_YIELD,),
        prompt_version=prompt,
        redaction_version="r1",
    )


def response() -> AdjudicationResponse:
    return AdjudicationResponse(
        request_id="request-1",
        canonical_crop_sha256="c" * 64,
        provider="fake",
        model="fake-model",
        rows=(
            AdjudicationRow(
                row_order=0,
                role=RowRole.DETAIL,
                fields=(
                    AdjudicationField(
                        name="description",
                        raw_value="Blood Test",
                        token_ids=("description",),
                        polygon=polygon((10, 10, 80, 25)),
                    ),
                    AdjudicationField(
                        name="amount",
                        raw_value="120.00",
                        token_ids=("amount",),
                        polygon=polygon((120, 10, 170, 25)),
                    ),
                ),
            ),
        ),
        latency_ms=12,
        measured_cost_usd=Decimal("0.001"),
    )


class FakeAdapter:
    model = "fake-model"
    config_identity = "fake-config-v1"

    def __init__(self) -> None:
        self.calls = 0

    def adjudicate(self, value: AdjudicationRequest) -> AdjudicationResponse:
        self.calls += 1
        return response().model_copy(update={"request_id": value.request_id})


def test_adjudication_cache_is_content_and_config_bound(tmp_path) -> None:
    adapter = FakeAdapter()
    cache = tmp_path / "cache.json"
    first, hit = cached_adjudication(cache, request(tmp_path), adapter)
    second, second_hit = cached_adjudication(cache, request(tmp_path), adapter)
    assert not hit and second_hit
    assert first == second
    assert adapter.calls == 1

    cached_adjudication(cache, request(tmp_path, prompt="p2"), adapter)
    assert adapter.calls == 2
    cached = cache.read_text()
    assert '"prompt_version"' not in cached
    assert '"ocr_tokens"' not in cached
    assert '"candidate_rows"' not in cached
    assert "p2" not in cached


def test_enabled_mode_promotion_is_frozen_and_configuration_bound(tmp_path) -> None:
    decision = GeminiPromotionDecision(
        approved=True,
        model="gemini-3.5-flash",
        prompt_version="p1",
        redaction_version="r1",
        frozen_manifest_sha256="f" * 64,
        local_f1=0.91,
        challenger_f1=0.92,
        unseen_precision=0.88,
        unseen_recall=0.87,
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    path = tmp_path / "promotion.json"
    path.write_text(decision.model_dump_json())
    assert validate_promotion(
        path,
        model="gemini-3.5-flash",
        prompt_version="p1",
        redaction_version="r1",
    ) == decision
    with pytest.raises(ValueError, match="does not match"):
        validate_promotion(
            path,
            model="gemini-3.5-flash",
            prompt_version="changed",
            redaction_version="r1",
        )


def test_grounding_accepts_exact_printed_evidence() -> None:
    local = (
        token("description", "Blood Test", (10, 10, 80, 25)),
        token("amount", "120.00", (120, 10, 170, 25)),
    )
    page = (
        token("description", "Blood Test", (210, 310, 280, 325)),
        token("amount", "120.00", (320, 310, 370, 325)),
    )
    result = ground_adjudication(
        response(),
        validation_tokens=local,
        evidence_tokens=page,
        crop_width=200,
        crop_height=100,
        table_type=TableType.ITEM_LEDGER,
    )
    assert len(result.rows) == 1
    assert result.rows[0].candidate.amount == Decimal("120.00")
    assert result.rows[0].evidence_box == (210, 310, 370, 325)


def test_grounding_rejects_hallucinated_numeric_value() -> None:
    original = response()
    bad = original.model_copy(
        update={
            "rows": (
                original.rows[0].model_copy(
                    update={
                        "fields": (
                            original.rows[0].fields[0],
                            original.rows[0].fields[1].model_copy(
                                update={"raw_value": "999.00"}
                            ),
                        )
                    }
                ),
            )
        }
    )
    tokens = (
        token("description", "Blood Test", (10, 10, 80, 25)),
        token("amount", "120.00", (120, 10, 170, 25)),
    )
    result = ground_adjudication(
        bad,
        validation_tokens=tokens,
        evidence_tokens=tokens,
        crop_width=200,
        crop_height=100,
        table_type=TableType.ITEM_LEDGER,
    )
    assert not result.rows
    assert "row_0:unsupported_value:amount" in result.rejected_reasons


def test_grounding_rejects_numeric_token_outside_declared_column() -> None:
    tokens = (
        token("description", "Blood Test", (10, 10, 80, 25)),
        token("amount", "120.00", (120, 10, 170, 25)),
    )
    result = ground_adjudication(
        response(),
        validation_tokens=tokens,
        evidence_tokens=tokens,
        crop_width=200,
        crop_height=100,
        table_type=TableType.ITEM_LEDGER,
        column_centers={"amount": 0.2},
    )
    assert not result.rows
    assert "row_0:column_mismatch:amount" in result.rejected_reasons
