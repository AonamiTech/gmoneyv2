"""End-to-end reader flow without models: PP-OCR printed table -> TeleOCR rows -> linking
-> reconciliation.  Only the PP-OCR tokens and the TeleOCR reply are synthetic/recorded."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.contracts.extraction import RowRole, TableType, TokenManifestEntry
from gmoney.extraction.ocr_rows import reconstruct_ocr_rows
from gmoney.extraction.offline import _link_source_tables
from gmoney.extraction.reader_consensus import PageBudget
from gmoney.extraction.reader_pipeline import finish_reader_table, reader_table_decision
from gmoney.extraction.reconciliation import reconcile
from gmoney.extraction.teleocr_reader import read_otsl_rows

FIXTURES = Path(__file__).parent / "fixtures" / "teleocr"
RECORDED = (FIXTURES / "bill6_page2_procedure.otsl").read_text()


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


def _page_tokens(*extra: OcrToken) -> tuple[OcrToken, ...]:
    return (
        _token("s0", "Bed Procedure", 10, 10, 150),
        _token("h1", "Procedure Name", 10, 40, 200),
        _token("h2", "Date", 330, 40, 50),
        _token("h3", "Rate", 420, 40, 50),
        _token("h4", "Quantity", 490, 40, 60),
        _token("h5", "Amount ( Rs )", 560, 40, 90),
        _token("d1", "Chemotherapy Pushing Charges (Long CT Per Day)", 10, 100, 300),
        _token("dt1", "10/02/2026", 330, 100, 80),
        _token("r1", "700.00", 420, 100),
        _token("q1", "1", 500, 100, 10),
        _token("a1", "700.00", 560, 100),
        _token("d2", "Chemotherapy Pushing Charges (Ultra Short)", 10, 130, 300),
        _token("dt2", "11/02/2026", 330, 130, 80),
        _token("r2", "190.00", 420, 130),
        _token("q2", "1", 500, 130, 10),
        _token("a2", "190.00", 560, 130),
        _token("t1", "Total Amount", 10, 160, 120),
        _token("t2", "890.00", 560, 160),
        *extra,
    )


def _run(otsl: str, tokens: tuple[OcrToken, ...], printed_total: str = "890.00"):
    reconstruction = reconstruct_ocr_rows(
        tokens, page_number=2, table_id="p2-t1", box=(0, 0, 700, 260)
    )
    outcome = finish_reader_table(
        document_id="d" * 64,
        page_number=2,
        table_id="p2-t1",
        page_artifact_sha256="a" * 64,
        candidates=read_otsl_rows(otsl),
        tokens=tokens,
        starting_order=0,
        table_type=TableType.ITEM_LEDGER,
    )
    lookup = {
        token.token_id: TokenManifestEntry(
            token_id=token.token_id,
            page_number=2,
            text=token.text,
            polygon=token.polygon,
            artifact_sha256=token.artifact_sha256,
            artifact_relative_path="pages/page-2.png",
            confidence=token.confidence,
        )
        for token in tokens
    }
    tables = _link_source_tables(reconstruction.source_tables, outcome.rows, token_lookup=lookup)
    result = {
        "document_total": {"amount": printed_total, "label": "Total Amount", "page_number": 2},
        "rows": [row.model_dump(mode="json") for row in outcome.rows],
        "source_tables": [table.model_dump(mode="json") for table in tables],
    }
    return outcome, tables, reconcile(result)


def test_teleocr_rows_link_to_ppocr_printed_rows_and_verify() -> None:
    outcome, tables, report = _run(RECORDED, _page_tokens())
    linked = [row.canonical_row_id for table in tables for row in table.rows]
    assert linked[:2] == [str(row.id) for row in outcome.rows if row.role is RowRole.DETAIL]
    assert linked[2] is None  # the printed "Total Amount" row stays a printed total
    assert report["status"] == "verified"
    outcomes = {check["id"].split(":")[0]: check["outcome"] for check in report["checks"]}
    assert outcomes["C1"] == "pass"
    assert outcomes["C2"] == "pass"


def test_misread_amount_is_never_published_silently() -> None:
    misread = RECORDED.replace("<fcel>190.00<fcel>1<fcel>190.00", "<fcel>160.00<fcel>1<fcel>160.00")
    outcome, _tables, report = _run(misread, _page_tokens())
    # 160.00 is not printed anywhere on the page, so the row cannot be grounded...
    assert [row["amount"] for row in outcome.ungrounded] == ["160.00"]
    # ...and the bill does not verify: 700 of 890 is accounted for.
    assert report["status"] == "flagged"
    c1 = next(check for check in report["checks"] if check["id"] == "C1")
    assert (c1["expected"], c1["actual"]) == ("890.00", "700.00")


def test_footer_bill_number_does_not_become_a_charge_through_the_flow() -> None:
    otsl = RECORDED.replace("<|im_end|>", "") + "<fcel>Bill No.<ecel><ecel><ecel><fcel>4172<nl>"
    tokens = _page_tokens(_token("f1", "Bill No.", 10, 230, 60), _token("f2", "4172", 560, 230))
    outcome, _tables, report = _run(otsl, tokens)
    assert all(row.net_amount != Decimal("4172") for row in outcome.rows)
    assert report["status"] == "verified"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Normal read replaces PP-OCR rows.
        (dict(has_candidates=True, ocr_charge_rows=2, reader_charge_rows=2), (True, None)),
        # Provider error: flagged even when PP-OCR also found nothing.
        (
            dict(provider_error="provider_error:ConnectError", ocr_charge_rows=0),
            (False, "reader_provider_failed"),
        ),
        (dict(truncated=True, ocr_charge_rows=0), (False, "reader_truncated")),
        # Reader read the table but found no charges where PP-OCR did: keep PP-OCR, flag.
        (
            dict(has_candidates=True, ocr_charge_rows=3, reader_charge_rows=0),
            (False, "reader_no_rows"),
        ),
        (dict(ocr_charge_rows=3), (False, "reader_no_rows")),
        # Non-financial table: nothing to flag.
        (dict(has_candidates=True, ocr_charge_rows=0, reader_charge_rows=0), (True, None)),
        (dict(), (False, None)),
    ],
)
def test_reader_table_decision_never_passes_a_failed_read_silently(kwargs, expected) -> None:
    options = {
        "provider_error": None,
        "truncated": False,
        "has_candidates": False,
        "ocr_charge_rows": 0,
        "reader_charge_rows": 0,
        **kwargs,
    }
    assert reader_table_decision(**options) == expected


def test_header_tables_are_read_locally_but_never_sent_to_gemini(tmp_path: Path) -> None:
    class Recorder:
        model = "gemini-test"
        sent: list[bytes] = []

        def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
            return Decimal(0)

        def read_table(self, crop_bytes: bytes, mime_type: str = "image/png"):
            self.sent.append(crop_bytes)
            raise AssertionError("header crop sent to Gemini")

    crop = tmp_path / "p1-t1.png"
    crop.write_bytes(b"not read")
    reader = Recorder()
    outcome = finish_reader_table(
        document_id="d" * 64,
        page_number=2,
        table_id="p2-t1",
        page_artifact_sha256="a" * 64,
        candidates=read_otsl_rows(RECORDED),
        tokens=_page_tokens(),
        starting_order=0,
        header_table=True,
        gemini_reader=reader,
        gemini_allowed=True,
        budget=PageBudget.from_inr(0.30, 88),
        crop_path=crop,
        crop_tokens=_page_tokens(),
        crop_size=(700, 260),
        crop_box=(0, 0, 700, 260),
        page_size=(1000, 1400),
        redaction_output=tmp_path / "redacted.png",
    )
    assert reader.sent == []
    assert outcome.diagnostic["gemini_reader_block_reason"] == "header_table_not_sent"
    assert outcome.gemini_calls == 0
