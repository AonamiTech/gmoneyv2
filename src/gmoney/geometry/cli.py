from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gmoney.geometry.crop import crop_region
from gmoney.geometry.normalize import normalize_page
from gmoney.geometry.render import render_pdf

app = typer.Typer(no_args_is_help=True)


@app.command("crop")
def crop(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    box: Annotated[str, typer.Option(help="left,top,right,bottom")],
    page_number: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    coordinates = tuple(int(value.strip()) for value in box.split(","))
    if len(coordinates) != 4:
        raise typer.BadParameter("box must contain left,top,right,bottom")
    result = crop_region(source, output, page_number, coordinates)
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "artifact_sha256": result.artifact_sha256,
                "transform": result.transform.model_dump(mode="json"),
            },
            indent=2,
        )
    )


@app.command("render")
def render(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    artifacts: Annotated[Path, typer.Option(file_okay=False)],
    dpi: Annotated[int, typer.Option(min=72, max=600)] = 300,
) -> None:
    manifest = render_pdf(source, artifacts, dpi)
    typer.echo(json.dumps(manifest.to_dict(), indent=2))


@app.command("normalize")
def normalize(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    page_number: Annotated[int, typer.Option(min=1)] = 1,
    dpi: Annotated[int, typer.Option(min=72, max=600)] = 300,
    orientation_degrees: int = 0,
    rotation_degrees: float = 0.0,
) -> None:
    result = normalize_page(
        source,
        output,
        page_number=page_number,
        dpi=dpi,
        orientation_correction_degrees=orientation_degrees,
        rotation_correction_degrees=rotation_degrees,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "artifact_sha256": result.artifact_sha256,
                "transform": result.transform.model_dump(mode="json"),
            },
            indent=2,
        )
    )
