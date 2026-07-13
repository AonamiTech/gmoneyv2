from __future__ import annotations

import json
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from gmoney.evaluation.corpus import sha256_file
from gmoney.inference.contracts import InferenceRequest
from gmoney.inference.paddle import (
    PaddleDocLayoutV3Adapter,
    PaddleOcrV6Adapter,
    PaddleOcrVlAdapter,
    PaddleWirelessTableAdapter,
)

app = typer.Typer(no_args_is_help=True)
ADAPTERS = {
    "pp-ocrv6-medium": PaddleOcrV6Adapter,
    "pp-doclayoutv3": PaddleDocLayoutV3Adapter,
    "slanext-wireless": PaddleWirelessTableAdapter,
    "paddleocr-vl-1.6": lambda: PaddleOcrVlAdapter(
        base_url=os.getenv("PADDLEOCR_VL_URL", "http://127.0.0.1:8111")
    ),
}


@app.command("run")
def run(
    model: Annotated[str, typer.Option(help="Model adapter name")],
    images: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    concurrency: Annotated[int, typer.Option(min=1, max=3)] = 1,
) -> None:
    if model not in ADAPTERS:
        raise typer.BadParameter(f"unknown model {model}; choose from {sorted(ADAPTERS)}")
    adapter = ADAPTERS[model]()

    def predict(item: tuple[int, Path]) -> dict[str, object]:
        page_number, image = item
        response = adapter.predict(
            InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=sha256_file(image),
                image_path=str(image.resolve()),
                page_number=page_number,
            )
        )
        result = response.model_dump(mode="json")
        result["image"] = str(image)
        return result

    if concurrency > 1 and model != "paddleocr-vl-1.6":
        raise typer.BadParameter(
            "concurrent in-process Paddle models are unsafe; use isolated worker processes"
        )
    started = time.perf_counter()
    indexed_images = list(enumerate(images, 1))
    if concurrency == 1:
        results = [predict(item) for item in indexed_images]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(predict, indexed_images))
    wall_elapsed_ms = round((time.perf_counter() - started) * 1000)
    latencies = [int(result["latency_ms"]) for result in results]
    report = {
        "benchmark_version": "component_benchmark_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "model": model,
        "count": len(results),
        "concurrency": concurrency,
        "wall_elapsed_ms": wall_elapsed_ms,
        "throughput_per_minute": len(results) / wall_elapsed_ms * 60_000,
        "latency_ms": {
            "minimum": min(latencies),
            "median": statistics.median(latencies),
            "maximum": max(latencies),
        },
        "peak_rss_bytes": max(
            (int(result["peak_rss_bytes"]) for result in results if result["peak_rss_bytes"]),
            default=None,
        ),
        "memory_scope": results[0]["memory_scope"],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "results"}
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
