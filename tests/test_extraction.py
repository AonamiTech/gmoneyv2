from decimal import Decimal
from pathlib import Path

from gmoney.contracts.extraction import RowRole
from gmoney.extraction.offline import _cached_prediction
from gmoney.extraction.otsl import parse_otsl, split_otsl_tables
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.extraction.typed_values import parse_decimal
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
    assert rows[2].role is RowRole.METADATA


def test_numeric_parser_is_strict_and_supports_indian_financial_forms() -> None:
    assert parse_decimal("₹ 1,23,456.75") == Decimal("123456.75")
    assert parse_decimal("(1,000.00)") == Decimal("-1000.00")
    assert parse_decimal("500 CR") == Decimal("-500")
    assert parse_decimal("1O0.5O") == Decimal("100.50")
    assert parse_decimal("26/06/2026") is None
    assert parse_decimal("IP/12345") is None
    assert parse_decimal("999999999999999999") is None


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
    changed = request.model_copy(
        update={"request_id": "changed", "artifact_sha256": "b" * 64}
    )
    third, third_hit = _cached_prediction(tmp_path / "stage.json", changed, adapter)

    assert first.output == second.output == third.output
    assert (first_hit, second_hit, third_hit) == (False, True, False)
    assert adapter.calls == 2
