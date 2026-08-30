from pathlib import Path

import pytest

from gmoney.contracts.evidence import (
    OcrToken,
    PageAsset,
    PagePreprocessingRecord,
    PageQuality,
    Point,
    Polygon,
    PreprocessingCandidate,
    PreprocessingVariant,
    TransformChain,
)
from gmoney.extraction.offline import (
    PageInferenceBundle,
    _orientation_correction,
    _select_page_inference,
)
from gmoney.geometry.preprocess import PreparedPageCandidate
from gmoney.geometry.transform import identity
from gmoney.inference.contracts import (
    InferenceResponse,
    ModelKind,
    ModelSpec,
)


def _quality(artifact_sha256: str) -> PageQuality:
    return PageQuality(
        page_number=1,
        artifact_sha256=artifact_sha256,
        width=100,
        height=200,
        dpi=300,
        mean_luminance=200,
        contrast_stddev=40,
        laplacian_variance=100,
        edge_density=0.05,
        estimated_skew_degrees=0,
    )


def _transform() -> TransformChain:
    matrix = identity()
    return TransformChain(
        page_number=1,
        source_width=100,
        source_height=200,
        derived_width=100,
        derived_height=200,
        forward_matrix=matrix,
        inverse_matrix=matrix,
    )


def _token(index: int, text: str, confidence: float, artifact_sha256: str) -> OcrToken:
    return OcrToken(
        token_id=f"token-{artifact_sha256[0]}-{index}",
        page_number=1,
        text=text,
        confidence=confidence,
        polygon=Polygon(
            points=(
                Point(x=1, y=1 + index * 4),
                Point(x=20, y=1 + index * 4),
                Point(x=20, y=4 + index * 4),
                Point(x=1, y=4 + index * 4),
            )
        ),
        artifact_sha256=artifact_sha256,
        model_name="test",
        model_version="1",
    )


def _bundle(
    variant: PreprocessingVariant,
    artifact_sha256: str,
    texts: tuple[str, ...],
    confidence: float,
    table_count: int,
    reconstruction_score: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
) -> PageInferenceBundle:
    contract = PreprocessingCandidate(
        variant=variant,
        artifact_sha256=artifact_sha256,
        artifact_relative_path=f"preprocessing/{variant.value}.png",
        width=100,
        height=200,
        dpi=300,
        transform=_transform(),
        quality=_quality(artifact_sha256),
    )
    response = InferenceResponse(
        request_id=f"request-{variant.value}",
        spec=ModelSpec(
            kind=ModelKind.OCR,
            provider="test",
            model_name="test",
            model_version="1",
            backend="test",
            device="cpu",
        ),
        output={},
        latency_ms=1,
        peak_rss_bytes=1,
        memory_scope="process",
    )
    tokens = tuple(_token(index, text, confidence, "a" * 64) for index, text in enumerate(texts))
    boxes = tuple((0, index * 20, 100, index * 20 + 15) for index in range(table_count))
    return PageInferenceBundle(
        candidate=PreparedPageCandidate(Path(contract.artifact_relative_path), contract),
        ocr_response=response,
        ocr_cache_hit=False,
        layout_response=response,
        layout_cache_hit=False,
        tokens=tokens,
        source_tokens=tokens,
        layout_boxes=boxes,
        geometry_boxes=(),
        reconstruction_score=reconstruction_score,
    )


def test_orientation_output_is_converted_to_reversible_image_coordinates() -> None:
    degrees, confidence = _orientation_correction(
        {"pages": [{"res": {"label_names": ["90"], "scores": [0.97]}}]}
    )
    assert degrees == 270
    assert confidence == 0.97


def test_candidate_selection_rejects_more_text_when_financial_evidence_is_lost() -> None:
    raw = _bundle(
        PreprocessingVariant.RAW,
        "a" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.80,
        1,
    )
    candidate = _bundle(
        PreprocessingVariant.GEOMETRY_300,
        "b" * 64,
        ("DESCRIPTION", "TOTAL", "extra", "extra2", "extra3"),
        0.95,
        1,
    )

    selected, annotated = _select_page_inference((raw, candidate))

    assert selected is raw
    assert next(item for item in annotated if item.selected).variant is PreprocessingVariant.RAW


def test_candidate_selection_accepts_preserved_financial_text_and_new_table() -> None:
    raw = _bundle(
        PreprocessingVariant.RAW,
        "a" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.80,
        0,
    )
    candidate = _bundle(
        PreprocessingVariant.GEOMETRY_300,
        "b" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.82,
        1,
    )

    selected, annotated = _select_page_inference((raw, candidate))

    assert selected is candidate
    selected_contract = next(item for item in annotated if item.selected)
    assert selected_contract.variant is PreprocessingVariant.GEOMETRY_300
    assert selected_contract.selection_reason == "selected_safe_structural_improvement"


def test_candidate_selection_rejects_worse_reconstructed_structure() -> None:
    raw = _bundle(
        PreprocessingVariant.RAW,
        "a" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.80,
        0,
        reconstruction_score=(0, 2, 0, 8, 10, 2, 2),
    )
    candidate = _bundle(
        PreprocessingVariant.GEOMETRY_300,
        "b" * 64,
        ("DESCRIPTION", "100.00", "TOTAL", "extra", "extra2", "extra3"),
        0.95,
        1,
        reconstruction_score=(0, 1, 0, 4, 5, 1, 1),
    )

    selected, _annotated = _select_page_inference((raw, candidate))

    assert selected is raw


def test_candidate_selection_accepts_better_reconstructed_structure() -> None:
    raw = _bundle(
        PreprocessingVariant.RAW,
        "a" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.80,
        0,
        reconstruction_score=(0, 1, 0, 4, 5, 1, 1),
    )
    candidate = _bundle(
        PreprocessingVariant.GEOMETRY_300,
        "b" * 64,
        ("DESCRIPTION", "100.00", "TOTAL"),
        0.82,
        0,
        reconstruction_score=(0, 2, 0, 8, 10, 2, 2),
    )

    selected, annotated = _select_page_inference((raw, candidate))

    assert selected is candidate
    assert next(item for item in annotated if item.selected).reconstruction_score == (
        0,
        2,
        0,
        8,
        10,
        2,
        2,
    )


def test_revision_four_preprocessing_record_requires_exactly_one_selection() -> None:
    raw_hash = "a" * 64
    raw = PreprocessingCandidate(
        variant=PreprocessingVariant.RAW,
        artifact_sha256=raw_hash,
        artifact_relative_path="pages/raw.png",
        width=100,
        height=200,
        dpi=300,
        transform=_transform(),
        quality=_quality(raw_hash),
        selected=True,
        selection_reason="selected_raw_no_safe_improvement",
    )
    record = PagePreprocessingRecord(
        page_number=1,
        raw_artifact_sha256=raw_hash,
        raw_artifact_relative_path="pages/raw.png",
        raw_quality=_quality(raw_hash),
        orientation_degrees=0,
        orientation_confidence=0.99,
        candidates=(raw,),
        selected_variant=PreprocessingVariant.RAW,
    )
    asset = PageAsset(
        document_sha256="b" * 64,
        page_number=1,
        artifact_sha256=raw_hash,
        relative_path="pages/raw.png",
        width=100,
        height=200,
        dpi=300,
        renderer="test",
        renderer_version="1",
    )

    assert record.raw_artifact_sha256 == asset.artifact_sha256


def test_preprocessing_record_rejects_a_derivative_without_raw_candidate() -> None:
    derivative_hash = "c" * 64
    derivative = PreprocessingCandidate(
        variant=PreprocessingVariant.GEOMETRY_300,
        artifact_sha256=derivative_hash,
        artifact_relative_path="preprocessing/geometry.png",
        width=100,
        height=200,
        dpi=300,
        transform=_transform(),
        quality=_quality(derivative_hash),
        selected=True,
        selection_reason="invalid_test_candidate",
    )

    with pytest.raises(ValueError, match="exactly one raw candidate"):
        PagePreprocessingRecord(
            page_number=1,
            raw_artifact_sha256="a" * 64,
            raw_artifact_relative_path="pages/raw.png",
            raw_quality=_quality("a" * 64),
            candidates=(derivative,),
            selected_variant=PreprocessingVariant.GEOMETRY_300,
        )
