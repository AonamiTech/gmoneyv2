from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gmoney.inference.ocr_table_fallback import propose_tables_from_ocr

app = typer.Typer(no_args_is_help=True)


@app.command("evaluate")
def evaluate(
    benchmark: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    payload = json.loads(benchmark.read_text())
    results: list[dict[str, object]] = []
    for result in payload.get("results") or []:
        page = (result.get("output", {}).get("pages") or [{}])[0].get("res") or {}
        proposals = propose_tables_from_ocr(
            page.get("rec_boxes") or [],
            page.get("rec_texts") or [],
        )
        results.append(
            {
                "image": result.get("image"),
                "ocr_tokens": len(page.get("rec_texts") or []),
                "recovered": bool(proposals),
                "proposals": [
                    {
                        "box": proposal.box,
                        "token_indexes": proposal.token_indexes,
                        "row_count": proposal.row_count,
                        "confidence": proposal.confidence,
                    }
                    for proposal in proposals
                ],
            }
        )
    report = {
        "report_version": "ocr_geometry_table_recovery_v1",
        "input_benchmark": str(benchmark),
        "pages": len(results),
        "recovered_pages": sum(bool(result["recovered"]) for result in results),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    app()
