from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from gmoney.evaluation import compare as compare_module
from gmoney.evaluation.compare import compare_bills


def _result(mode: str) -> dict[str, Any]:
    rows = [
        {
            "id": "r1",
            "role": "detail",
            "review_disposition": "accepted",
            "description": "Advamab 100mg Inj",
            "net_amount": "700.00",
        },
        {
            "id": "r2",
            "role": "detail",
            "review_disposition": "pending" if mode == "teleocr_gemini" else "accepted",
            "description": "Chemotherapy Pushing Charges (Ultra Short)",
            "net_amount": "190.00" if mode != "heuristic" else "4172",
        },
    ]
    adapter = {"adapter_name": "StarDoc-AI/TeleOCR", "cache_hit": False}
    return {
        "pages": 2,
        "hospital": {"name": "Secret Hospital"},
        "document_total": {"amount": "890.00", "label": "Total Bill Amount", "page_number": 2},
        "rows": rows,
        "diagnostics": [
            {
                "adapter_inputs": [adapter] if mode != "heuristic" else [],
                "reader_consensus": (
                    {"counts": {"rows_compared": 2, "rows_agreed": 1, "amount_disagreements": 1}}
                    if mode == "teleocr_gemini"
                    else {}
                ),
            }
        ],
        "provider_usage": {
            "aggregate": {
                "gemini_calls": 1 if mode == "teleocr_gemini" else 0,
                "gemini_measured_cost_usd": "0.0022" if mode == "teleocr_gemini" else "0",
            }
        },
    }


class FakeExtractor:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def extract(self, source: Path, artifact_root: Path) -> dict[str, Any]:
        if source.name.startswith("Broken"):
            raise RuntimeError(f"cannot open {source}")
        return _result(self.mode)


def test_compare_reports_each_mode_without_patient_or_file_names(tmp_path: Path) -> None:
    source_dir = tmp_path / "Sample Bills"
    source_dir.mkdir()
    (source_dir / "Saroj Gupta final bill.pdf").write_bytes(b"%PDF-1.4 bill one")
    (source_dir / "Broken Vijaya.pdf").write_bytes(b"%PDF-1.4 bill two")
    output = tmp_path / "out"

    report = compare_bills(
        sorted(source_dir.iterdir()),
        ["heuristic", "teleocr", "teleocr_gemini"],
        output,
        extractor_factory=lambda mode, _options: FakeExtractor(mode),
        validator=lambda *_args: "passed",
        inr_per_usd=88,
        gpu_cost_inr_per_hour=36,
    )

    broken, saroj = report["bills"]
    assert broken["runs"]["teleocr"] == {
        "mode": "teleocr",
        "status": "error",
        "error": "RuntimeError",
        "seconds": broken["runs"]["teleocr"]["seconds"],
    }
    heuristic = saroj["runs"]["heuristic"]
    assert heuristic["reconciliation_status"] == "flagged"
    assert heuristic["failed_checks"][0]["difference"] == "3982.00"
    assert heuristic["teleocr"] == {"calls": 0, "cache_hits": 0}
    teleocr = saroj["runs"]["teleocr"]
    assert teleocr["reconciliation_status"] == "verified"
    assert teleocr["rows_by_role"] == {"detail": 2}
    assert teleocr["teleocr"]["calls"] == 1
    gemini = saroj["runs"]["teleocr_gemini"]
    assert gemini["gemini_calls"] == 1
    assert gemini["estimated_inr_per_page"]["gemini"] == "0.0968"
    assert gemini["reader_disagreements"]["amount_disagreements"] == 1
    assert gemini["rows_pending_review"] == 1

    for name in ("report.md", "report.json"):
        text = (output / name).read_text()
        for secret in ("Saroj", "Gupta", "Vijaya", "Secret Hospital", "Advamab", "Chemotherapy"):
            assert secret not in text
    markdown = (output / "report.md").read_text()
    assert "| bill-02 | 2 | flagged |" in markdown
    assert "| teleocr | 1 | 0 | 0 | 1 |" in markdown
    aliases = json.loads((output / "aliases.local.json").read_text())
    assert aliases == {"bill-01": "Broken Vijaya.pdf", "bill-02": "Saroj Gupta final bill.pdf"}


def test_compare_cli_rejects_unknown_modes(tmp_path: Path) -> None:
    (tmp_path / "bills").mkdir()
    result = CliRunner().invoke(
        compare_module.app,
        ["--source-dir", str(tmp_path / "bills"), "--output", str(tmp_path / "o"), "--modes", "x"],
    )
    assert result.exit_code != 0
