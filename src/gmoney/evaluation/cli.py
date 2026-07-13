from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gmoney.evaluation.metrics import canonical_metric_v2, legacy_metric_v1, row_view

app = typer.Typer(no_args_is_help=True)


@app.command("rows")
def rows(
    gold: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    actual: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    gold_payload = json.loads(gold.read_text())
    actual_payload = json.loads(actual.read_text())
    gold_rows = [row_view(row, index) for index, row in enumerate(gold_payload.get("rows", []))]
    actual_rows = [
        row_view(row, index)
        for index, row in enumerate(actual_payload.get("rows", actual_payload))
    ]
    result = {
        "legacy_metric_v1": legacy_metric_v1(gold_rows, actual_rows).to_dict(),
        "canonical_metric_v2": canonical_metric_v2(gold_rows, actual_rows).to_dict(),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        typer.echo(rendered, nl=False)
