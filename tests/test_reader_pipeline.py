from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import ReviewDisposition, RowRole, TableType
from gmoney.extraction.reader_consensus import PageBudget
from gmoney.extraction.reader_pipeline import finish_reader_table
from gmoney.extraction.reconciliation import reconcile
from gmoney.extraction.teleocr_reader import read_otsl_rows
from gmoney.inference.gemini import TableReaderRow, TableReadResponse

FIXTURES = Path(__file__).parent / "fixtures" / "teleocr"


def _token(token_id: str, text: str, x: float, y: float, width: float = 60) -> OcrToken:
    return OcrToken(
        token_id=token_id,
        page_number=2,
        text=text,
        polygon=Polygon(
            points=(
                Point(x=x, y=y),
                Point(x=x + width, y=y),
                Point(x=x + width, y=y + 14),
                Point(x=x, y=y + 14),
            )
        ),
        artifact_sha256="a" * 64,
        confidence=0.99,
        model_name="PP-OCRv6",
        model_version="test",
    )


def _procedure_tokens() -> tuple[OcrToken, ...]:
    return (
        _token("h1", "Procedure Name", 10, 40, 200),
        _token("d1", "Chemotherapy Pushing Charges (Long CT Per Day)", 10, 100, 300),
        _token("r1", "700.00", 420, 100),
        _token("q1", "1", 500, 100, 10),
        _token("a1", "700.00", 560, 100),
        _token("d2", "Chemotherapy Pushing Charges (Ultra Short)", 10, 130, 300),
        _token("r2", "190.00", 420, 130),
        _token("q2", "1", 500, 130, 10),
        _token("a2", "190.00", 560, 130),
        _token("t1", "Total Amount", 10, 160, 120),
        _token("t2", "890.00", 560, 160),
    )


def _finish(candidates, tokens, **options):
    return finish_reader_table(
        document_id="d" * 64,
        page_number=2,
        table_id="p2-t1",
        page_artifact_sha256="a" * 64,
        candidates=candidates,
        tokens=tokens,
        starting_order=0,
        table_type=TableType.ITEM_LEDGER,
        **options,
    )


def test_recorded_teleocr_rows_are_grounded_and_reconcile() -> None:
    candidates = read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text())
    outcome = _finish(candidates, _procedure_tokens())

    details = [row for row in outcome.rows if row.role is RowRole.DETAIL]
    assert [(row.description, row.net_amount) for row in details] == [
        ("Chemotherapy Pushing Charges (Long CT Per Day)", Decimal("700.00")),
        ("Chemotherapy Pushing Charges (Ultra Short)", Decimal("190.00")),
    ]
    assert all(set(row.field_evidence) >= {"description", "amount"} for row in details)
    assert outcome.ungrounded == []
    report = reconcile(
        {
            "document_total": {"amount": "890.00", "label": "Total Amount"},
            "rows": [row.model_dump(mode="json") for row in outcome.rows],
        }
    )
    assert report["status"] == "verified"


def test_ungrounded_reader_rows_are_reported_not_silently_dropped() -> None:
    candidates = read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text())
    tokens = tuple(token for token in _procedure_tokens() if token.token_id not in {"d2", "a2"})
    outcome = _finish(candidates, tokens)
    assert len([row for row in outcome.rows if row.role is RowRole.DETAIL]) == 1
    assert outcome.ungrounded == [
        {
            "description": "Chemotherapy Pushing Charges (Ultra Short)",
            "amount": "190.00",
            "section": "Bed Procedure",
            "role": "detail",
            "source_route": "teleocr_otsl",
            "grounded_fields": [],
        }
    ]
    assert outcome.diagnostic["reader_ungrounded_rows"] == outcome.ungrounded


def test_full_page_guard_read_keeps_only_tables_not_already_read() -> None:
    candidates = read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text())
    outcome = _finish(
        candidates,
        _procedure_tokens(),
        is_guard=True,
        existing_page_rows=(("Chemotherapy Pushing Charges (Long CT Per Day)", "700.00"),),
    )
    assert [row.description for row in outcome.rows if row.role is RowRole.DETAIL] == [
        "Chemotherapy Pushing Charges (Ultra Short)"
    ]
    assert outcome.diagnostic["reader_full_page_guard"] == {"candidates": 3, "already_read": 1}


class FakeGemini:
    model = "gemini-test"

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        return Decimal("0.0005")

    def read_table(self, crop_bytes: bytes, mime_type: str = "image/png") -> TableReadResponse:
        self.sent.append(crop_bytes)
        return TableReadResponse(
            model=self.model,
            rows=(
                TableReaderRow(
                    description="Chemotherapy Pushing Charges (Long CT Per Day)", amount="700.00"
                ),
                TableReaderRow(
                    description="Chemotherapy Pushing Charges (Ultra Short)", amount="190.00"
                ),
                TableReaderRow(description="Total Amount", amount="890.00", is_total=True),
            ),
            measured_cost_usd=Decimal("0.0005"),
            input_sha256="0" * 64,
        )


def test_gemini_second_reader_sends_only_the_redacted_crop(tmp_path: Path) -> None:
    crop = tmp_path / "p2-t1.png"
    cv2.imwrite(str(crop), np.full((200, 640, 3), 255, dtype=np.uint8))
    page = tmp_path / "page-2.png"
    cv2.imwrite(str(page), np.full((1400, 1000, 3), 128, dtype=np.uint8))
    gemini = FakeGemini()
    candidates = read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text())
    tokens = _procedure_tokens()
    outcome = _finish(
        candidates,
        tokens,
        gemini_reader=gemini,
        gemini_allowed=True,
        budget=PageBudget.from_inr(0.30, 88),
        crop_path=crop,
        crop_tokens=tokens,
        crop_size=(640, 200),
        crop_box=(0, 300, 640, 500),
        page_size=(1000, 1400),
        redaction_output=tmp_path / "p2-t1-redacted.png",
    )
    assert gemini.sent == [(tmp_path / "p2-t1-redacted.png").read_bytes()]
    assert page.read_bytes() not in gemini.sent
    assert outcome.gemini_calls == 1
    assert outcome.gemini_cost_usd == Decimal("0.0005")
    assert outcome.diagnostic["reader_consensus"]["counts"]["rows_agreed"] == 2
    assert all(
        row.review_disposition is ReviewDisposition.ACCEPTED
        and "reader_agree" in row.validation_flags
        for row in outcome.rows
        if row.role is RowRole.DETAIL
    )


def test_gemini_is_not_called_during_recovery_or_for_guard_bands(tmp_path: Path) -> None:
    gemini = FakeGemini()
    candidates = read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text())
    for options in ({"gemini_allowed": False}, {"gemini_allowed": True, "is_guard": True}):
        outcome = _finish(
            candidates,
            _procedure_tokens(),
            gemini_reader=gemini,
            budget=PageBudget.from_inr(0.30, 88),
            **options,
        )
        assert outcome.gemini_calls == 0
        assert "gemini_reader_block_reason" in outcome.diagnostic
    assert gemini.sent == []


def test_disagreeing_rows_are_published_pending_review(tmp_path: Path) -> None:
    class Disagreeing(FakeGemini):
        def read_table(self, crop_bytes: bytes, mime_type: str = "image/png"):
            response = super().read_table(crop_bytes, mime_type)
            rows = list(response.rows)
            rows[0] = TableReaderRow(description="Chemo Pushing (Long)", amount="710.00")
            return response.model_copy(update={"rows": tuple(rows[:2])})

    crop = tmp_path / "crop.png"
    cv2.imwrite(str(crop), np.full((200, 640, 3), 255, dtype=np.uint8))
    tokens = _procedure_tokens()
    outcome = _finish(
        read_otsl_rows((FIXTURES / "bill6_page2_procedure.otsl").read_text()),
        tokens,
        gemini_reader=Disagreeing(),
        gemini_allowed=True,
        budget=PageBudget.from_inr(0.30, 88),
        crop_path=crop,
        crop_tokens=tokens,
        crop_size=(640, 200),
        crop_box=(0, 300, 640, 500),
        page_size=(1000, 1400),
        redaction_output=tmp_path / "redacted.png",
    )
    dispositions = {
        row.description: row.review_disposition
        for row in outcome.rows
        if row.role is RowRole.DETAIL
    }
    assert ReviewDisposition.PENDING in dispositions.values()
    assert outcome.diagnostic["reader_consensus"]["disagreements"]
