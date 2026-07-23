from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType
from gmoney.contracts.phase3 import (
    GeminiMode,
    ProfileMatch,
    RecoveryReason,
    RecoveryStage,
)
from gmoney.extraction import offline as offline_module
from gmoney.extraction.ocr_rows import (
    ReconstructionResult,
    TableSchemaState,
)
from gmoney.extraction.offline import OfflineExtractor, TableWork
from gmoney.extraction.recovery import (
    decide_recovery,
    is_implausibly_low_yield,
    map_crop_tokens_to_page,
    needs_field_quality_recovery,
    reconstruction_quality,
    safely_improves_reconstruction,
)
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.extraction.spatial import AlignedLedgerRow


def _schema(table_type: TableType = TableType.ITEM_LEDGER) -> TableSchemaState:
    return TableSchemaState(
        source_page=1,
        source_table="table-1",
        table_type=table_type,
        column_centers={
            "description": 0.2,
            "quantity": 0.5,
            "rate": 0.7,
            "amount": 0.9,
        },
        confidence=1,
        header_token_ids=("header",),
    )


def _aligned_row(
    source_row: int,
    *,
    description: str = "Anaesthetist",
    quantity: Decimal | None = None,
    rate: Decimal | None = Decimal("100"),
    flags: tuple[str, ...] = (),
    mapped_fields: tuple[str, ...] = ("description", "rate", "amount"),
) -> AlignedLedgerRow:
    field_token_ids = {
        field: (f"token-{source_row}-{field}",) for field in mapped_fields
    }
    return AlignedLedgerRow(
        candidate=CandidateLedgerRow(
            source_row=source_row,
            role=RowRole.DETAIL,
            cells=(description, str(quantity or ""), str(rate or ""), "100"),
            description=description,
            quantity=quantity,
            rate=rate,
            amount=Decimal("100"),
            table_type=TableType.ITEM_LEDGER,
            source_route="ocr_spatial_graph",
            validation_flags=flags,
        ),
        field_token_ids=field_token_ids,
        evidence_token_ids=tuple(
            token_id for token_ids in field_token_ids.values() for token_id in token_ids
        ),
        evidence_box=(0, 0, 100, 20),
        grounding_ratio=1,
        source_routes=("ocr_spatial_graph",),
    )


def _reconstruction(
    rows: tuple[AlignedLedgerRow, ...],
    *,
    table_type: TableType = TableType.ITEM_LEDGER,
) -> ReconstructionResult:
    return ReconstructionResult(
        rows=rows,
        schema=_schema(table_type),
        diagnostics={
            "table_type": table_type.value,
            "ocr_line_count": len(rows) + 1,
            "ocr_row_count": len(rows),
        },
    )


def _source_table(*raw_values: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        rows=(
            SimpleNamespace(
                cells=tuple(
                    SimpleNamespace(raw_value=raw_value) for raw_value in raw_values
                )
            ),
        )
    )


def _ocr_token(text: str, artifact_sha256: str, token_id: str | None = None) -> OcrToken:
    return OcrToken(
        token_id=token_id or f"local-{text}",
        page_number=1,
        text=text,
        confidence=1,
        polygon=Polygon(
            points=(
                Point(x=0, y=0),
                Point(x=100, y=0),
                Point(x=100, y=50),
                Point(x=0, y=50),
            )
        ),
        artifact_sha256=artifact_sha256,
        model_name="test",
        model_version="1",
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


def test_empty_metadata_region_is_terminal_without_recovery() -> None:
    reconstruction = ReconstructionResult(
        rows=(),
        schema=None,
        diagnostics={"table_type": TableType.METADATA.value},
    )
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)
    assert decision.reasons == ()
    assert decision.planned_stages == ()
    assert decision.route == "local_ocr"


@pytest.mark.parametrize(
    "flag",
    (
        "missing_labeled_quantity",
        "missing_labeled_unit_price",
        "line_arithmetic_mismatch",
    ),
)
def test_grounded_field_defects_route_to_local_validation_recovery(flag: str) -> None:
    reconstruction = _reconstruction((_aligned_row(0, flags=(flag,)),))

    assert needs_field_quality_recovery(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)

    assert decision.reasons == (RecoveryReason.VALIDATION_FAILURE,)
    assert decision.route == "local_recovery"
    assert RecoveryStage.CROP_OCR in decision.planned_stages


def test_selected_profile_with_field_defects_still_routes_to_local_recovery() -> None:
    reconstruction = _reconstruction(
        (_aligned_row(0, flags=("missing_labeled_quantity",)),)
    )
    selected_profile = ProfileMatch(
        profile_key="hospital",
        profile_version=1,
        score=0.99,
        margin=0.5,
        selected=True,
    )

    decision = decide_recovery(
        reconstruction,
        gemini_mode=GeminiMode.OFF,
        profile_match=selected_profile,
    )

    assert decision.route == "local_recovery"
    assert decision.reasons == (RecoveryReason.VALIDATION_FAILURE,)


@pytest.mark.parametrize("table_type", (TableType.METADATA, TableType.PAYMENT))
def test_terminal_regions_ignore_field_quality_defects(table_type: TableType) -> None:
    reconstruction = _reconstruction(
        (_aligned_row(0, flags=("missing_labeled_quantity",)),),
        table_type=table_type,
    )

    assert not needs_field_quality_recovery(reconstruction)
    decision = decide_recovery(reconstruction, gemini_mode=GeminiMode.OFF)

    assert decision.reasons == ()
    assert decision.planned_stages == ()


def test_haemorrhoidectomy_anaesthetist_same_row_recovery_is_a_safe_improvement() -> None:
    unchanged = tuple(
        _aligned_row(index, description=f"Service {index}") for index in range(12)
    )
    baseline = _reconstruction(
        (
            *unchanged,
            _aligned_row(
                12,
                flags=("missing_labeled_quantity",),
            ),
        )
    )
    recovered = _reconstruction(
        (
            *unchanged,
            _aligned_row(
                12,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert len(baseline.rows) == len(recovered.rows) == 13
    assert recovered.rows[-1].candidate.quantity == Decimal("1")
    assert "missing_labeled_quantity" not in recovered.rows[-1].candidate.validation_flags
    assert reconstruction_quality(recovered) > reconstruction_quality(baseline)
    assert safely_improves_reconstruction(baseline, recovered)


def test_candidate_with_fewer_canonical_rows_cannot_replace_baseline() -> None:
    baseline = _reconstruction((_aligned_row(0), _aligned_row(1)))
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert not safely_improves_reconstruction(baseline, candidate)


@pytest.mark.parametrize(
    "candidate",
    (
        _reconstruction(
            (
                _aligned_row(
                    0,
                    flags=(
                        "missing_labeled_quantity",
                        "missing_labeled_unit_price",
                    ),
                ),
            )
        ),
        _reconstruction(
            (
                _aligned_row(
                    0,
                    quantity=Decimal("1"),
                    mapped_fields=("description", "amount"),
                ),
            )
        ),
    ),
)
def test_validation_or_mapped_field_regression_cannot_replace_baseline(
    candidate: ReconstructionResult,
) -> None:
    baseline = _reconstruction(
        (_aligned_row(0, flags=("missing_labeled_quantity",)),)
    )

    assert not safely_improves_reconstruction(baseline, candidate)


def test_populated_source_cells_break_an_otherwise_equal_quality_tie() -> None:
    rows = (_aligned_row(0),)
    baseline = ReconstructionResult(
        rows=rows,
        schema=_schema(),
        diagnostics={"table_type": TableType.ITEM_LEDGER.value},
        source_tables=(_source_table(None, None),),
    )
    candidate = ReconstructionResult(
        rows=rows,
        schema=baseline.schema,
        diagnostics=baseline.diagnostics,
        source_tables=(_source_table("Anaesthetist", "1"),),
    )

    assert reconstruction_quality(candidate) > reconstruction_quality(baseline)
    assert safely_improves_reconstruction(baseline, candidate)


def test_source_cell_regression_does_not_veto_better_grounded_field_quality() -> None:
    baseline_row = _aligned_row(0, flags=("missing_labeled_quantity",))
    baseline = ReconstructionResult(
        rows=(baseline_row,),
        schema=_schema(),
        diagnostics={"table_type": TableType.ITEM_LEDGER.value},
        source_tables=(_source_table("Anaesthetist", "100"),),
    )
    candidate = _reconstruction(
        (
            _aligned_row(
                0,
                quantity=Decimal("1"),
                mapped_fields=("description", "quantity", "rate", "amount"),
            ),
        )
    )

    assert safely_improves_reconstruction(baseline, candidate)


def test_crop_recovery_reconstructs_all_variants_and_selects_best_grounded_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_artifact_sha256 = "a" * 64
    high_artifact_sha256 = "b" * 64
    clahe_artifact_sha256 = "c" * 64
    baseline = _reconstruction(
        (
            _aligned_row(
                0,
                flags=(
                    "missing_labeled_quantity",
                    "missing_labeled_unit_price",
                    "line_arithmetic_mismatch",
                ),
            ),
        )
    )
    observed_token_batches: list[tuple[OcrToken, ...]] = []

    monkeypatch.setattr(
        offline_module,
        "render_pdf_region",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "high.png",
            artifact_sha256=high_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "clahe_variant",
        lambda *args, **kwargs: SimpleNamespace(
            output_path=tmp_path / "clahe.png",
            artifact_sha256=clahe_artifact_sha256,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "_cached_prediction",
        lambda path, request, adapter: (
            SimpleNamespace(
                output={"variant": request.options["input_variant"]},
                latency_ms=1,
            ),
            False,
        ),
    )
    monkeypatch.setattr(
        offline_module,
        "paddle_ocr_tokens",
        lambda output, page_number, artifact_sha256: tuple(
            _ocr_token(
                text,
                artifact_sha256,
                f"local-{output['variant']}-{field}",
            )
            for field, text in (
                (
                    ("description", "Anaesthetist"),
                    ("quantity", "1"),
                    ("amount", "100"),
                )
                if output["variant"] == "high_resolution"
                else (
                    ("description", "Anaesthetist"),
                    ("quantity", "1"),
                    ("rate", "100"),
                    ("amount", "100"),
                )
            )
        ),
    )
    monkeypatch.setattr(
        offline_module.cv2,
        "imread",
        lambda path, mode: SimpleNamespace(shape=(100, 200, 3)),
    )

    def fake_reconstruct(tokens, **kwargs):
        observed_token_batches.append(tokens)
        field_tokens = {
            token.token_id.rsplit("-", maxsplit=1)[-1]: (token.token_id,)
            for token in tokens
        }
        quantity = Decimal(
            next(token.text for token in tokens if token.token_id.endswith("-quantity"))
        )
        if len(tokens) == 3:
            row = _aligned_row(
                0,
                quantity=quantity,
                flags=(
                    "missing_labeled_unit_price",
                    "line_arithmetic_mismatch",
                ),
                mapped_fields=(),
            )
        else:
            rate = Decimal(
                next(token.text for token in tokens if token.token_id.endswith("-rate"))
            )
            row = _aligned_row(
                0,
                quantity=quantity,
                rate=rate,
                mapped_fields=(),
            )
        return _reconstruction(
            (
                replace(
                    row,
                    field_token_ids=field_tokens,
                    evidence_token_ids=tuple(
                        token_id
                        for token_ids in field_tokens.values()
                        for token_id in token_ids
                    ),
                ),
            )
        )

    monkeypatch.setattr(offline_module, "reconstruct_ocr_rows", fake_reconstruct)
    extractor = object.__new__(OfflineExtractor)
    extractor.ocr = object()
    work = TableWork(
        table_id="table-1",
        page_number=1,
        page_artifact_sha256=page_artifact_sha256,
        crop_path=tmp_path / "primary.png",
        crop_sha256="d" * 64,
        box=(200, 300, 400, 400),
    )

    recovered, attempts = extractor._recover_crop_ocr(
        source=tmp_path / "bill.pdf",
        artifact_root=tmp_path,
        work=work,
        prior_schemas=(),
        page_artifact_sha256=page_artifact_sha256,
        baseline=baseline,
    )

    assert recovered is not None
    assert recovered.rows[0].candidate.validation_flags == ()
    assert recovered.rows[0].candidate.quantity == Decimal("1")
    assert [len(tokens) for tokens in observed_token_batches] == [3, 4]
    observed_tokens = tuple(
        token for tokens in observed_token_batches for token in tokens
    )
    assert all(token.artifact_sha256 == page_artifact_sha256 for token in observed_tokens)
    assert all(token.token_id.startswith("recovery:") for token in observed_tokens)
    assert recovered.rows[0].field_token_ids["quantity"] == (
        "recovery:local-photometric-quantity",
    )
    crop_attempts = [
        attempt for attempt in attempts if attempt.stage is RecoveryStage.CROP_OCR
    ]
    assert [attempt.status for attempt in crop_attempts] == [
        "no_improvement",
        "recovered",
    ]
    assert [attempt.accepted_rows for attempt in crop_attempts] == [0, 1]
