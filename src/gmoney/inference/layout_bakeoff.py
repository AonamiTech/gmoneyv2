from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import cv2
import fitz
import typer

from gmoney.evaluation.corpus import infer_hospital_id, sha256_file
from gmoney.inference.contracts import InferenceRequest
from gmoney.inference.paddle import PaddleDocLayoutV3Adapter

app = typer.Typer(no_args_is_help=True)


def table_boxes(output: dict[str, Any], minimum_score: float = 0.3) -> list[dict[str, Any]]:
    pages = output.get("pages") or []
    if not pages:
        return []
    result = pages[0].get("res") or {}
    return [
        box
        for box in result.get("boxes") or []
        if box.get("label") == "table" and float(box.get("score") or 0) >= minimum_score
    ]


def _render_pdf_page(source: Path, page_number: int, output: Path, dpi: int) -> None:
    scale = dpi / 72
    with fitz.open(source) as document:
        if page_number > len(document):
            raise ValueError(f"{source} has no page {page_number}")
        pixmap = document[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        pixmap.save(temporary)
        temporary.replace(output)


def _materialize_page(source_paths: list[Path], page_number: int, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    if len(source_paths) == 1 and source_paths[0].suffix.casefold() == ".pdf":
        _render_pdf_page(source_paths[0], page_number, output, dpi)
        return
    if page_number > len(source_paths):
        raise ValueError(f"image sequence has no page {page_number}")
    image = cv2.imread(str(source_paths[page_number - 1]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {source_paths[page_number - 1]}")
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"cannot write {output}")


@app.command("run")
def run(
    gold_directory: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    artifacts: Annotated[Path, typer.Option(file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    dpi: Annotated[int, typer.Option(min=72, max=600)] = 300,
    minimum_score: Annotated[float, typer.Option(min=0, max=1)] = 0.3,
    one_page_per_bill: bool = False,
) -> None:
    work: list[dict[str, Any]] = []
    for gold_path in sorted(gold_directory.glob("*.gold.json")):
        gold = json.loads(gold_path.read_text())
        source_paths = [Path(path) for path in gold.get("source_paths") or []]
        if not source_paths or any(not path.exists() for path in source_paths):
            raise FileNotFoundError(f"missing sources for {gold_path.name}")
        pages = sorted({int(row["page_number"]) for row in gold.get("rows") or []})
        if one_page_per_bill:
            pages = pages[:1]
        hospital_id = infer_hospital_id(gold_path.name)
        for page_number in pages:
            page_path = artifacts / hospital_id / gold_path.stem / f"page-{page_number:04d}.png"
            _materialize_page(source_paths, page_number, page_path, dpi)
            work.append(
                {
                    "hospital_id": hospital_id,
                    "gold_file": gold_path.name,
                    "page_number": page_number,
                    "image": page_path,
                }
            )

    adapter = PaddleDocLayoutV3Adapter()
    results: list[dict[str, Any]] = []
    for index, item in enumerate(work, 1):
        image = item["image"]
        response = adapter.predict(
            InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=sha256_file(image),
                image_path=str(image.resolve()),
                page_number=item["page_number"],
            )
        )
        boxes = table_boxes(response.output, minimum_score)
        result = {
            **{key: value for key, value in item.items() if key != "image"},
            "image": str(image),
            "found": bool(boxes),
            "table_boxes": boxes,
            "latency_ms": response.latency_ms,
            "peak_rss_bytes": response.peak_rss_bytes,
        }
        results.append(result)
        typer.echo(
            f"[{index}/{len(work)}] {item['gold_file']} "
            f"p{item['page_number']}: {len(boxes)}"
        )

    by_hospital: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_hospital[result["hospital_id"]].append(result["found"])
    found = sum(result["found"] for result in results)
    report = {
        "benchmark_version": "row_bearing_page_table_recall_v1",
        "model": adapter.spec.model_dump(mode="json"),
        "dpi": dpi,
        "minimum_score": minimum_score,
        "gold_pages": len(results),
        "pages_with_table": found,
        "page_table_recall": found / len(results) if results else 0,
        "per_hospital": {
            hospital: {
                "gold_pages": len(values),
                "pages_with_table": sum(values),
                "recall": sum(values) / len(values),
            }
            for hospital, values in sorted(by_hospital.items())
        },
        "latency_ms": {
            "total": sum(result["latency_ms"] for result in results),
            "maximum": max((result["latency_ms"] for result in results), default=0),
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"results", "per_hospital"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
