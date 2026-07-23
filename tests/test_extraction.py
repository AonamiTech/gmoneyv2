from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.extraction import RowRole
from gmoney.extraction.offline import VlAsset, _cached_prediction, _safe_box, _vertical_vl_tiles
from gmoney.extraction.otsl import parse_otsl, split_otsl_tables
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.extraction.typed_values import parse_decimal, parse_service_date
from gmoney.inference.contracts import (
    InferenceRequest,
    InferenceResponse,
    ModelKind,
    ModelSpec,
)


class _FakeAdapter:
    spec = ModelSpec(
        kind=ModelKind.TABLE,
        provider="test",
        model_name="fixture",
        model_version="1",
        backend="test",
        device="cpu",
    )

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        return InferenceResponse(
            request_id=request.request_id,
            spec=self.spec,
            output={"content": "fixture"},
            latency_ms=1,
            memory_scope="remote_unmeasured",
        )


def test_otsl_parser_preserves_rows_cells_and_empty_cells() -> None:
    table = parse_otsl(
        "<fcel>Description<fcel>Qty<fcel>Amount<nl>"
        "<fcel>Blood Test<fcel>1<fcel>500.00<nl>"
        "<fcel>Nursing<fcel>2<fcel>1,000.00<nl>"
    )
    assert table.column_count == 3
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["Description", "Qty", "Amount"],
        ["Blood Test", "1", "500.00"],
        ["Nursing", "2", "1,000.00"],
    ]


def test_plain_prefix_is_treated_as_spanning_first_cell() -> None:
    table = parse_otsl("Ward Room Name<lcel><lcel><nl><fcel>CCU<fcel>414<ecel><nl>")
    assert table.rows[0][0].text == "Ward Room Name"
    assert table.rows[0][1].is_span_marker


def test_combined_vl_output_splits_on_later_header() -> None:
    table = parse_otsl(
        "<fcel>Ward<fcel>Bed<fcel>Status<nl><fcel>CCU<fcel>414<fcel>Admitted<nl>"
        "<fcel>Description<fcel>Qty<fcel>Amount<nl>"
        "<fcel>Blood Test<fcel>1<fcel>500<nl>"
    )
    tables = split_otsl_tables(table)
    assert len(tables) == 2
    assert tables[1].rows[0][0].text == "Description"


def test_candidate_rows_type_values_and_separate_totals() -> None:
    table = parse_otsl(
        "<fcel>Service Name<fcel>Qty<fcel>Rate<fcel>Discount<fcel>Total Amount<nl>"
        "<fcel>Blood Test<fcel>2<fcel>250.00<fcel>0<fcel>500.00<nl>"
        "<fcel>Total<ecel><ecel><ecel><fcel>500.00<nl>"
        "<fcel>Advance Received<ecel><ecel><ecel><fcel>100.00<nl>"
    )
    rows = extract_candidate_rows(table)
    assert rows[0].role is RowRole.DETAIL
    assert rows[0].description == "Blood Test"
    assert rows[0].quantity == Decimal("2")
    assert rows[0].rate == Decimal("250.00")
    assert rows[0].amount == Decimal("500.00")
    assert rows[1].role is RowRole.SECTION_TOTAL
    assert rows[2].role is RowRole.PAYMENT


def test_amount_rs_before_total_is_rate_and_unit_days_is_quantity() -> None:
    table = parse_otsl(
        "<fcel>Sr.No.<fcel>Particular<fcel>Amount Rs.<fcel>Unit/Days<fcel>Total<nl>"
        "<fcel>2.<fcel>Room Charges<fcel>4,000.00<fcel>7<fcel>28,000<nl>"
    )
    rows = extract_candidate_rows(table)
    assert rows[0].description == "Room Charges"
    assert rows[0].quantity == Decimal("7")
    assert rows[0].rate == Decimal("4000.00")
    assert rows[0].amount == Decimal("28000")


def test_continuation_subtotals_and_zero_placeholders_are_separate_roles() -> None:
    table = parse_otsl(
        "<fcel>Service Name<fcel>Qty/Days<fcel>Amount<fcel>Total Amount<nl>"
        "<fcel>Registration<ecel><ecel><ecel><nl>"
        "<ecel><ecel><fcel>1.00<fcel>100.00<fcel>100.00<nl>"
        "<fcel>Sub Total: Registration<ecel><ecel><fcel>100.00<nl>"
        "<fcel>Empty Ward<fcel>0.00<fcel>0.00<fcel>0.00<nl>"
    )
    rows = extract_candidate_rows(table)
    assert rows[0].description == "Registration"
    assert rows[0].quantity == Decimal("1.00")
    assert rows[0].rate == Decimal("100.00")
    assert rows[0].amount == Decimal("100.00")
    assert rows[1].role is RowRole.SECTION_TOTAL
    # A printed zero-valued line remains an auditable charge row; zero is not
    # evidence that the row is metadata.
    assert rows[2].role is RowRole.DETAIL


def test_collapsed_pharmacy_header_maps_columns_and_derives_missing_amount() -> None:
    table = parse_otsl(
        "# Particulars Batch Expiry Rate Qty Amount<lcel><lcel><lcel><lcel><lcel><nl>"
        "<fcel>2<fcel>Ondet 2ML<fcel>A26<fcel>Dec-2027<fcel>12.70<fcel>2<nl>"
    )
    rows = extract_candidate_rows(table)
    assert rows[0].description == "Ondet 2ML"
    assert rows[0].rate == Decimal("12.70")
    assert rows[0].quantity == Decimal("2")
    assert rows[0].amount == Decimal("25.40")
    assert rows[0].amount_derived is True


def test_numeric_parser_is_strict_and_supports_indian_financial_forms() -> None:
    assert parse_decimal("₹ 1,23,456.75") == Decimal("123456.75")
    assert parse_decimal("(1,000.00)") == Decimal("-1000.00")
    assert parse_decimal("500 CR") == Decimal("-500")
    assert parse_decimal("1O0.5O") == Decimal("100.50")
    assert parse_decimal("26/06/2026") is None
    assert parse_decimal("IP/12345") is None
    assert parse_decimal("999999999999999999") is None


def test_service_date_parser_supports_numeric_and_alphabetic_indian_dates() -> None:
    assert parse_service_date("12-Feb-2026") == "2026-02-12"
    assert parse_service_date("12/02/2026") == "2026-02-12"
    assert parse_service_date("12 February 2026") == "2026-02-12"
    assert parse_service_date("2026-02-12") is None
    assert parse_service_date("20/01/2026 10:24:02") == "2026-01-20"
    assert parse_service_date("20/01/202610:24:02") == "2026-01-20"
    assert parse_service_date("20/01/2026 99:99:99") is None
    assert parse_service_date("20/01/2026 - 21/01/2026") is None


def test_inference_cache_is_bound_to_artifact_options_and_model(tmp_path: Path) -> None:
    adapter = _FakeAdapter()
    request = InferenceRequest(
        request_id="first",
        artifact_sha256="a" * 64,
        image_path="fixture.png",
        page_number=1,
        options={"prompt": "Table Recognition:"},
    )
    first, first_hit = _cached_prediction(tmp_path / "stage.json", request, adapter)
    second, second_hit = _cached_prediction(tmp_path / "stage.json", request, adapter)
    changed = request.model_copy(update={"request_id": "changed", "artifact_sha256": "b" * 64})
    third, third_hit = _cached_prediction(tmp_path / "stage.json", changed, adapter)
    changed_options = changed.model_copy(
        update={"request_id": "changed-options", "options": {"prompt": "new prompt"}}
    )
    fourth, fourth_hit = _cached_prediction(
        tmp_path / "stage.json",
        changed_options,
        adapter,
    )

    assert first.output == second.output == third.output == fourth.output
    assert (first_hit, second_hit, third_hit, fourth_hit) == (False, True, False, False)
    assert adapter.calls == 3


def test_table_box_has_extra_vertical_tolerance_for_trailing_rows() -> None:
    assert _safe_box((100, 200, 900, 1000), 1200, 1400) == (80, 100, 920, 1100)


def test_dense_vlm_asset_is_split_into_overlapping_vertical_tiles(tmp_path: Path) -> None:
    source = tmp_path / "table.png"
    assert cv2.imwrite(str(source), np.zeros((3300, 100, 3), dtype=np.uint8))
    tiles = _vertical_vl_tiles(
        VlAsset(source, "a" * 64, "fixture"),
        tmp_path,
        "p1-t1",
    )
    assert len(tiles) == 3
    assert all(tile.path.exists() and len(tile.artifact_sha256) == 64 for tile in tiles)
