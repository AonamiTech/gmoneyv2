from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.phase3 import GeminiMode, RecoveryReason, RecoveryStage
from gmoney.extraction.ocr_rows import ReconstructionResult
from gmoney.extraction.recovery import (
    decide_recovery,
    is_implausibly_low_yield,
    map_crop_tokens_to_page,
)


def test_recovery_ladder_is_local_first_and_gemini_last() -> None:
    reconstruction = ReconstructionResult(
        rows=(),
        schema=None,
        diagnostics={"table_type": "unknown"},
    )
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.CHALLENGER)
    assert decision.reasons == (
        RecoveryReason.ZERO_YIELD,
        RecoveryReason.NO_SCHEMA,
        RecoveryReason.UNKNOWN_TABLE,
    )
    assert decision.planned_stages == (
        RecoveryStage.HIGH_RESOLUTION,
        RecoveryStage.PHOTOMETRIC,
        RecoveryStage.CROP_OCR,
        RecoveryStage.LOCAL_VLM,
        RecoveryStage.GEMINI,
        RecoveryStage.REVIEW,
    )


def test_crop_tokens_map_back_to_original_page() -> None:
    source = OcrToken(
        token_id="1",
        page_number=1,
        text="Amount",
        confidence=1,
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=100, y=0),
                Point(x=100, y=50),
                Point(x=0, y=50),
            )
        ),
        artifact_sha256="b" * 64,
        model_name="test",
        model_version="1",
    )
    mapped = map_crop_tokens_to_page(
        (source,),
        (200, 300, 400, 400),
        crop_width=200,
        crop_height=100,
        page_artifact_sha256="a" * 64,
    )[0]
    assert mapped.token_id == "recovery:1"
    assert mapped.polygon.points[0] == Point(x=200, y=300)
    assert mapped.polygon.points[2] == Point(x=300, y=350)


def test_implausibly_low_yield_is_escalated() -> None:
    reconstruction = ReconstructionResult(
        rows=(object(),),
        schema=None,
        diagnostics={"ocr_line_count": 20, "ocr_row_count": 1},
    )
    assert is_implausibly_low_yield(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)
    assert RecoveryReason.LOW_YIELD in decision.reasons
