from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from gmoney.contracts.extraction import CanonicalRow
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_tokens import paddle_ocr_tokens
from gmoney.extraction.otsl import parse_otsl, split_otsl_tables
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.extraction.spatial import align_candidate_rows
from gmoney.geometry.crop import crop_region
from gmoney.geometry.render import render_pdf
from gmoney.inference.contracts import InferenceRequest, InferenceResponse
from gmoney.inference.ocr_table_fallback import propose_tables_from_ocr
from gmoney.inference.paddle import (
    PaddleDocLayoutV3Adapter,
    PaddleOcrV6Adapter,
    PaddleOcrVlAdapter,
)

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class TableWork:
    table_id: str
    page_number: int
    page_artifact_sha256: str
    crop_path: Path
    crop_sha256: str


def _cached_prediction(
    cache_path: Path,
    request: InferenceRequest,
    adapter: Any,
) -> tuple[InferenceResponse, bool]:
    """Reuse only an inference result tied to the same immutable input and model."""
    if cache_path.exists():
        envelope = json.loads(cache_path.read_text())
        if (
            envelope.get("artifact_sha256") == request.artifact_sha256
            and envelope.get("options") == request.options
            and envelope.get("model_spec") == adapter.spec.model_dump(mode="json")
        ):
            return InferenceResponse.model_validate(envelope["response"]), True

    response = adapter.predict(request)
    envelope = {
        "artifact_sha256": request.artifact_sha256,
        "options": request.options,
        "model_spec": adapter.spec.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(cache_path)
    return response, False


def _layout_boxes(output: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    pages = output.get("pages") or []
    if not pages:
        return []
    boxes = pages[0].get("res", {}).get("boxes") or []
    return [
        tuple(round(value) for value in box["coordinate"])
        for box in boxes
        if box.get("label") == "table" and float(box.get("score") or 0) >= 0.3
    ]


def _safe_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int = 20,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _deduplicate(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    selected: dict[tuple[object, ...], CanonicalRow] = {}
    for row in rows:
        description = re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip()
        evidence_ids = tuple(
            sorted(token_id for evidence in row.evidence for token_id in evidence.token_ids)
        )
        key = (row.page_number, description, row.net_amount, evidence_ids)
        current = selected.get(key)
        if current is None or len(row.evidence) > len(current.evidence):
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row.page_number, row.row_order))


def _remove_cross_page_rollups(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    output: list[CanonicalRow] = []
    for row in rows:
        later = [
            candidate
            for candidate in rows
            if candidate.page_number > row.page_number and candidate.net_amount is not None
        ]
        is_rollup = bool(
            row.net_amount is not None
            and len(later) >= 2
            and sum((candidate.net_amount for candidate in later), start=0) == row.net_amount
        )
        if not is_rollup:
            output.append(row)
    return output


class OfflineExtractor:
    def __init__(self, vl_url: str) -> None:
        self.ocr = PaddleOcrV6Adapter()
        self.layout = PaddleDocLayoutV3Adapter()
        self.vl = PaddleOcrVlAdapter(base_url=vl_url)

    def extract(
        self,
        source: Path,
        artifact_root: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        manifest = render_pdf(source, artifact_root / "pages", dpi=300)
        if progress:
            progress(0, len(manifest.pages))
        document_id = manifest.document_sha256
        all_rows: list[CanonicalRow] = []
        diagnostics: list[dict[str, Any]] = []
        for page_asset in manifest.pages:
            page_path = artifact_root / "pages" / page_asset.relative_path
            ocr_request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=page_asset.artifact_sha256,
                image_path=str(page_path.resolve()),
                page_number=page_asset.page_number,
            )
            ocr_response, ocr_cache_hit = _cached_prediction(
                artifact_root / "inference" / f"page-{page_asset.page_number}.ocr.json",
                ocr_request,
                self.ocr,
            )
            tokens = paddle_ocr_tokens(
                ocr_response.output,
                page_asset.page_number,
                page_asset.artifact_sha256,
            )
            layout_request = InferenceRequest(
                request_id=str(uuid4()),
                artifact_sha256=page_asset.artifact_sha256,
                image_path=str(page_path.resolve()),
                page_number=page_asset.page_number,
            )
            layout_response, layout_cache_hit = _cached_prediction(
                artifact_root / "inference" / f"page-{page_asset.page_number}.layout.json",
                layout_request,
                self.layout,
            )
            boxes = _layout_boxes(layout_response.output)
            route = "layout"
            if not boxes:
                result = (ocr_response.output.get("pages") or [{}])[0].get("res") or {}
                proposals = propose_tables_from_ocr(
                    result.get("rec_boxes") or [],
                    result.get("rec_texts") or [],
                )
                boxes = [tuple(round(value) for value in proposal.box) for proposal in proposals]
                route = "ocr_geometry"

            table_work: list[TableWork] = []
            for table_index, box in enumerate(boxes):
                table_id = f"p{page_asset.page_number}-t{table_index + 1}"
                safe_box = _safe_box(box, page_asset.width, page_asset.height)
                crop = crop_region(
                    page_path,
                    artifact_root / "crops" / f"{table_id}.png",
                    page_asset.page_number,
                    safe_box,
                )
                table_work.append(
                    TableWork(
                        table_id=table_id,
                        page_number=page_asset.page_number,
                        page_artifact_sha256=page_asset.artifact_sha256,
                        crop_path=crop.output_path,
                        crop_sha256=crop.artifact_sha256,
                    )
                )

            def parse_table(
                work: TableWork,
                page_tokens=tokens,
            ) -> tuple[list[CanonicalRow], dict[str, Any]]:
                vl_request = InferenceRequest(
                    request_id=str(uuid4()),
                    artifact_sha256=work.crop_sha256,
                    image_path=str(work.crop_path.resolve()),
                    page_number=work.page_number,
                    options={"prompt": "Table Recognition:"},
                )
                vl_response, vl_cache_hit = _cached_prediction(
                    artifact_root / "inference" / f"{work.table_id}.vl.json",
                    vl_request,
                    self.vl,
                )
                content = str(vl_response.output.get("content") or "")
                parsed_rows: list[CanonicalRow] = []
                candidate_count = 0
                for table in split_otsl_tables(parse_otsl(content)):
                    candidates = extract_candidate_rows(table)
                    candidate_count += len(candidates)
                    aligned = align_candidate_rows(candidates, page_tokens)
                    parsed_rows.extend(
                        canonicalize_rows(
                            document_id,
                            work.page_number,
                            work.table_id,
                            work.page_artifact_sha256,
                            aligned,
                            starting_order=len(parsed_rows),
                        )
                    )
                return parsed_rows, {
                    "table_id": work.table_id,
                    "crop_sha256": work.crop_sha256,
                    "vl_latency_ms": vl_response.latency_ms,
                    "vl_cache_hit": vl_cache_hit,
                    "candidate_count": candidate_count,
                    "canonical_count": len(parsed_rows),
                    "content": content,
                }

            with ThreadPoolExecutor(max_workers=3) as executor:
                table_results = list(executor.map(parse_table, table_work))
            for rows, table_diagnostic in table_results:
                all_rows.extend(rows)
                diagnostics.append(
                    {
                        "page_number": page_asset.page_number,
                        "route": route,
                        "ocr_latency_ms": ocr_response.latency_ms,
                        "ocr_cache_hit": ocr_cache_hit,
                        "layout_latency_ms": layout_response.latency_ms,
                        "layout_cache_hit": layout_cache_hit,
                        **table_diagnostic,
                    }
                )
            if progress:
                progress(page_asset.page_number, len(manifest.pages))

        rows = _deduplicate(_remove_cross_page_rollups(all_rows))
        return {
            "output_version": "offline_accuracy_spine_v1",
            "document_id": document_id,
            "source_sha256": sha256_file(source),
            "source_name": source.name,
            "pages": len(manifest.pages),
            "page_assets": [
                {
                    "page_number": page.page_number,
                    "artifact_sha256": page.artifact_sha256,
                    "width": page.width,
                    "height": page.height,
                    "relative_path": str(Path("pages") / page.relative_path),
                }
                for page in manifest.pages
            ],
            "rows": [row.model_dump(mode="json") for row in rows],
            "diagnostics": diagnostics,
        }


@app.command("run")
def run(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option(file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    vl_url: str = "http://127.0.0.1:8111",
) -> None:
    result = OfflineExtractor(vl_url).extract(source, artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "document_id": result["document_id"],
                "pages": result["pages"],
                "rows": len(result["rows"]),
            }
        )
    )


if __name__ == "__main__":
    app()
