from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from gmoney.contracts.extraction import RowRole
from gmoney.extraction.reader_consensus import (
    PageBudget,
    gemini_candidates,
    needs_reviewer,
    reconcile_readers,
    second_read_table,
)
from gmoney.extraction.rows import CandidateLedgerRow
from gmoney.inference.gemini import GeminiTableReader, TableReaderRow, TableReadResponse


def _row(index: int, description: str, amount: str, role: RowRole = RowRole.DETAIL):
    return CandidateLedgerRow(
        source_row=index,
        role=role,
        cells=(description, amount),
        description=description,
        amount=Decimal(amount),
    )


class FakeReader:
    model = "gemini-test"

    def __init__(self, rows: list[TableReaderRow], cost: str = "0.001") -> None:
        self.rows = rows
        self.cost = Decimal(cost)
        self.sent: list[bytes] = []

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        return self.cost

    def read_table(self, crop_bytes: bytes, mime_type: str = "image/png") -> TableReadResponse:
        self.sent.append(crop_bytes)
        return TableReadResponse(
            model=self.model,
            rows=tuple(self.rows),
            measured_cost_usd=self.cost,
            input_sha256="0" * 64,
        )


def test_only_redacted_table_crop_bytes_are_sent(tmp_path: Path) -> None:
    crop = tmp_path / "p2-t1-redacted.png"
    crop.write_bytes(b"redacted table crop")
    page = tmp_path / "page-2.png"
    page.write_bytes(b"full page with patient header")
    reader = FakeReader([TableReaderRow(description="Bed Charges", amount="11100.00")])
    budget = PageBudget.from_inr(0.30, 88)

    response, reason = second_read_table(
        reader,
        redacted_crop=crop,
        crop_box=(0, 400, 1000, 900),
        page_width=1000,
        page_height=1400,
        budget=budget,
    )
    assert reason is None and response is not None
    assert reader.sent == [b"redacted table crop"]

    response, reason = second_read_table(
        reader,
        redacted_crop=page,
        crop_box=(0, 0, 1000, 1400),
        page_width=1000,
        page_height=1400,
        budget=budget,
    )
    assert response is None and reason == "full_page_crop_not_sent"
    assert b"full page with patient header" not in reader.sent


def test_page_cost_cap_stops_calls(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"crop")
    # 0.30 INR at 88 INR/USD = 0.003409 USD; each call costs 0.0015 USD -> two calls.
    reader = FakeReader([], cost="0.0015")
    budget = PageBudget.from_inr(0.30, 88)
    reasons = [
        second_read_table(
            reader,
            redacted_crop=crop,
            crop_box=(0, 0, 100, 100),
            page_width=1000,
            page_height=1000,
            budget=budget,
        )[1]
        for _ in range(4)
    ]
    assert reasons == [None, None, "page_cost_cap_reached", "page_cost_cap_reached"]
    assert budget.calls == 2
    assert budget.spent_usd == Decimal("0.0030")


def test_gemini_reader_uses_structured_json_and_reports_cost() -> None:
    sent: dict[str, object] = {}

    class Models:
        def generate_content(self, *, model, contents, config):
            sent["contents"] = contents
            return SimpleNamespace(
                text='{"rows":[{"description":"Bifilac","amount":"126.56","is_return":true}]}',
                usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=200),
            )

    reader = GeminiTableReader(
        api_key="test",
        client=SimpleNamespace(models=Models()),
        input_cost_usd_per_million=0.5,
        output_cost_usd_per_million=3,
    )
    response = reader.read_table(b"crop-bytes")
    assert response.measured_cost_usd == Decimal("0.0011")
    candidates = gemini_candidates(response)
    assert candidates[0].role is RowRole.REFUND
    assert candidates[0].amount == Decimal("126.56")
    image_part = sent["contents"][1]
    assert image_part.inline_data.data == b"crop-bytes"


def test_agreeing_rows_are_accepted() -> None:
    tele = (_row(0, "Bed Charges", "11100.00"), _row(1, "Doctor Visit", "650.00"))
    gem = (_row(0, "BED CHARGES.", "11100.00"), _row(1, "Doctor  visit", "650.00"))
    result = reconcile_readers(tele, gem)
    assert [row.validation_flags for row in result.rows] == [("reader_agree",), ("reader_agree",)]
    assert result.counts["rows_agreed"] == 2
    assert not any(needs_reviewer(row.validation_flags) for row in result.rows)


def test_amount_disagreement_is_resolved_by_printed_total() -> None:
    tele = (
        _row(0, "Chemotherapy Pushing Charges (Long CT Per Day)", "700.00"),
        _row(1, "Chemotherapy Pushing Charges (Ultra Short)", "160.00"),
        _row(2, "Total Amount", "890.00", RowRole.SECTION_TOTAL),
    )
    gem = (
        _row(0, "Chemotherapy Pushing Charges (Long CT Per Day)", "700.00"),
        _row(1, "Chemotherapy Pushing Charges (Ultra Short)", "190.00"),
    )
    result = reconcile_readers(tele, gem)
    details = [row for row in result.rows if row.role is RowRole.DETAIL]
    assert details[1].amount == Decimal("190.00")
    assert "reader_amount_resolved_by_total" in details[1].validation_flags
    assert result.disagreements[0]["resolution"] == "gemini_reconciles"
    assert not needs_reviewer(details[1].validation_flags)


def test_unresolvable_amount_disagreement_flags_cell_with_both_values() -> None:
    tele = (_row(0, "Room Rent", "1000.00"),)
    gem = (_row(0, "Room Rent", "1100.00"),)
    result = reconcile_readers(tele, gem)
    assert result.rows[0].amount == Decimal("1000.00")
    assert "reader_amount_disagreement" in result.rows[0].validation_flags
    assert result.disagreements == [
        {
            "field": "amount",
            "source_row": 0,
            "description": "Room Rent",
            "teleocr": "1000.00",
            "gemini": "1100.00",
            "resolution": "unresolved",
        }
    ]
    assert needs_reviewer(result.rows[0].validation_flags)


def test_description_disagreement_keeps_teleocr_and_stores_alternative() -> None:
    tele = (_row(0, "Neukine 300 Mcg", "3000.00"),)
    gem = (_row(0, "Neukine 300 Mg", "3000.00"),)
    result = reconcile_readers(tele, gem)
    assert result.rows[0].description == "Neukine 300 Mcg"
    assert "reader_description_disagreement" in result.rows[0].validation_flags
    assert result.disagreements[0]["gemini"] == "Neukine 300 Mg"


def test_rows_from_only_one_reader_are_kept_and_flagged() -> None:
    tele = (_row(0, "Bed Charges", "11100.00"), _row(1, "Nursing Charges", "800.00"))
    gem = (_row(0, "Bed Charges", "11100.00"), _row(1, "Doctor Visit", "650.00"))
    result = reconcile_readers(tele, gem)
    flags = {row.description: row.validation_flags for row in result.rows}
    assert flags["Nursing Charges"] == ("reader_only_teleocr",)
    assert flags["Doctor Visit"] == ("reader_only_gemini",)
    assert result.counts["only_teleocr"] == 1
    assert result.counts["only_gemini"] == 1
