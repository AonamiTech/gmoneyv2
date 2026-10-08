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
        # Partial read: rows are used, but the table is still flagged.
        (
            dict(truncated=True, has_candidates=True, ocr_charge_rows=2, reader_charge_rows=1),
            (True, "reader_truncated"),
        ),
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


def _otsl(*rows: list[str]) -> str:
    return "".join(
        "".join(f"<fcel>{cell}" if cell else "<ecel>" for cell in row) + "<nl>" for row in rows
    )


def _pharmacy_tokens() -> tuple[OcrToken, ...]:
    return (
        _token("h1", "Item Name", 10, 40, 120),
        _token("h2", "Qty", 420, 40, 40),
        _token("h3", "Amount", 560, 40, 70),
        _token("s1", "Item Issues", 10, 70, 120),
        _token("i1", "Taxol 100mg", 10, 100, 150),
        _token("iq", "1", 420, 100, 10),
        _token("ia", "5000.00", 560, 100, 70),
        _token("it", "Item Issues Total", 10, 130, 180),
        _token("ita", "5000.00", 560, 130, 70),
        _token("s2", "Return Item", 10, 160, 120),
        _token("r1", "Bifilac", 10, 190, 80),
        _token("rq", "1", 420, 190, 10),
        _token("ra", "126.56", 560, 190, 70),
        _token("rt", "Item Returns Total", 10, 220, 180),
        _token("rta", "126.56", 560, 220, 70),
    )


PHARMACY_OTSL = _otsl(
    ["Item Name", "Qty", "Amount"],
    ["Item Issues", "", ""],
    ["Taxol 100mg", "1", "5000.00"],
    ["Item Issues Total", "", "5000.00"],
    ["Return Item", "", ""],
    ["Bifilac", "1", "126.56"],
    ["Item Returns Total", "", "126.56"],
)


def test_refund_rows_ground_to_their_printed_amount_and_reconcile() -> None:
    outcome, _tables, report = _run(PHARMACY_OTSL, _pharmacy_tokens(), printed_total="4873.44")
    assert outcome.ungrounded == []
    refunds = [row for row in outcome.rows if row.role is RowRole.REFUND]
    assert [(row.description, row.net_amount) for row in refunds] == [
        ("Bifilac", Decimal("126.56"))
    ]
    assert report["status"] == "verified", report["reasons"]
    ids = {check["id"].split(":")[0] for check in report["checks"]}
    assert {"C1", "C1R", "C2"} <= ids


def test_dropped_return_section_cannot_verify_against_the_gross_total() -> None:
    # TeleOCR omits the returns, so rows equal the printed gross 5000.00, but the
    # printed "Return Item" rows in the PP-OCR table expose the missing refund.
    without_returns = PHARMACY_OTSL.split("<fcel>Return Item")[0]
    _outcome, _tables, report = _run(without_returns, _pharmacy_tokens(), printed_total="5000.00")
    assert report["status"] == "flagged"
    returns = next(check for check in report["checks"] if check["id"] == "C1R")
    assert returns["outcome"] == "fail"
    assert any("prints returns" in reason for reason in report["reasons"])


def test_headerless_tile_reuses_the_first_tile_header() -> None:
    from gmoney.extraction.teleocr_reader import otsl_header

    first = _otsl(["Description", "Qty", "Amount"], ["Room Rent", "1", "1500.00"])
    second = _otsl(["Nursing Charges", "1", "800.00"], ["Doctor Visit", "1", "650.00"])
    assert read_otsl_rows(second) == ()
    rows = read_otsl_rows(second, inherited_header=otsl_header(first))
    assert [(row.description, row.amount) for row in rows] == [
        ("Nursing Charges", Decimal("800.00")),
        ("Doctor Visit", Decimal("650.00")),
    ]
    assert all("reader_inherited_header" in row.validation_flags for row in rows)


def test_tile_overlap_dedupe_keeps_same_charge_on_another_day() -> None:
    from gmoney.extraction.teleocr_reader import drop_rows_already_read

    rows = read_otsl_rows(
        _otsl(
            ["Description", "Date", "Amount"],
            ["Room Rent", "11/02/2026", "1500.00"],
            ["Room Rent", "12/02/2026", "1500.00"],
        )
    )
    kept = drop_rows_already_read(rows, [("Room Rent", Decimal("1500.00"), "11/02/2026")])
    assert [row.service_date for row in kept] == ["12/02/2026"]


def test_charge_names_containing_total_or_id_words_stay_charges() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Service Name", "Code", "Amount"],
            ["Total Knee Replacement Implant", "", "125000.00"],
            ["IPD Registration", "", "500"],
            ["Patient ID Band", "", "50"],
            ["Implant inclusion", "1001", ""],
            ["Gross Amount", "", "125550.00"],
            ["Net Amount", "", "125550.00"],
        )
    )
    charges = [(row.description, row.amount) for row in rows if row.role is RowRole.DETAIL]
    assert charges == [
        ("Total Knee Replacement Implant", Decimal("125000.00")),
        ("IPD Registration", Decimal("500")),
        ("Patient ID Band", Decimal("50")),
    ]
    totals = [row.description for row in rows if row.role is RowRole.SECTION_TOTAL]
    assert totals == ["Gross Amount", "Net Amount"]


def test_gemini_skips_tables_with_identifier_rows(tmp_path: Path) -> None:
    class Recorder:
        model = "gemini-test"

        def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
            return Decimal(0)

        def read_table(self, crop_bytes: bytes, mime_type: str = "image/png"):
            raise AssertionError("crop with identifier rows sent to Gemini")

    otsl = RECORDED.replace("<|im_end|>", "") + "<fcel>UHID<ecel><ecel><ecel><fcel>88231<nl>"
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"x")
    outcome = finish_reader_table(
        document_id="d" * 64,
        page_number=2,
        table_id="p2-t1",
        page_artifact_sha256="a" * 64,
        candidates=read_otsl_rows(otsl),
        tokens=_page_tokens(),
        starting_order=0,
        gemini_reader=Recorder(),
        gemini_allowed=True,
        budget=PageBudget.from_inr(0.30, 88),
        crop_path=crop,
        crop_tokens=_page_tokens(),
        crop_size=(700, 260),
        crop_box=(0, 0, 700, 260),
        page_size=(1000, 1400),
        redaction_output=tmp_path / "redacted.png",
    )
    assert outcome.diagnostic["gemini_reader_block_reason"] == "identifier_rows_in_crop"


def test_teleocr_gemini_without_api_key_fails_instead_of_degrading() -> None:
    from gmoney.extraction.offline import OfflineExtractor
    from gmoney.settings import Settings

    with pytest.raises(ValueError, match="teleocr_gemini_requires_gemini_api_key"):
        OfflineExtractor(
            "http://vl.invalid",
            table_reader="teleocr_gemini",
            settings=Settings(_env_file=None, gemini_api_key=""),
        )


def test_reader_failure_reason_is_named_in_the_review_issue() -> None:
    from gmoney.demo.review import structural_issues

    diagnostic = {
        "page_number": 2,
        "table_id": "p2-t1",
        "phase3_route": {"reasons": ["low_yield"]},
        "recovery_attempts": [
            {"stage": "review", "status": "pending", "reason": "low_yield"},
            {"stage": "review", "status": "pending", "reason": "reader_provider_failed"},
        ],
    }
    issues = structural_issues({"diagnostics": [diagnostic]}, {"issue_overrides": {}})
    assert issues[0]["reason_codes"] == ["low_yield", "reader_provider_failed"]


def test_positive_printed_refunds_reduce_review_totals_like_reconciliation() -> None:
    from gmoney.demo.review import totals_summary
    from gmoney.extraction.reconciliation import signed_field_value

    outcome, _tables, report = _run(PHARMACY_OTSL, _pharmacy_tokens(), printed_total="4873.44")
    rows = [row.model_dump(mode="json") for row in outcome.rows]
    result = {
        "document_total": {"amount": "4873.44", "label": "Total Bill Amount"},
        "rows": rows,
    }
    totals = totals_summary(result, {"row_overrides": {}, "added_rows": {}}, rows)
    assert (totals["items_total"], totals["comparison"]) == ("4873.44", "match")
    assert report["status"] == "verified"
    assert signed_field_value({"role": "refund", "net_amount": "126.56"}) == Decimal("-126.56")
    # Heuristic refunds are printed negative and are unchanged.
    assert signed_field_value({"role": "refund", "net_amount": "-126.56"}) == Decimal("-126.56")
    assert signed_field_value({"role": "detail", "net_amount": "126.56"}) == Decimal("126.56")


@pytest.mark.parametrize(
    "footer",
    ["Medicines once sold cannot be returned", "No return without bill"],
)
def test_return_policy_text_does_not_demand_refund_rows(footer: str) -> None:
    tokens = _page_tokens(_token("f1", footer, 10, 230, 300))
    _outcome, _tables, report = _run(RECORDED, tokens)
    assert not any(check["id"] == "C1R" for check in report["checks"])
    assert report["status"] == "verified"


def test_lab_tests_named_total_stay_charges() -> None:
    rows = read_otsl_rows(
        _otsl(
            ["Test Name", "Amount"],
            ["Bilirubin Total", "250.00"],
            ["Protein Total", "150.00"],
            ["Lab Total", "400.00"],
            ["Item Issues Total", "400.00"],
        )
    )
    assert [row.description for row in rows if row.role is RowRole.DETAIL] == [
        "Bilirubin Total",
        "Protein Total",
    ]
    assert [row.description for row in rows if row.role is RowRole.SECTION_TOTAL] == [
        "Lab Total",
        "Item Issues Total",
    ]
