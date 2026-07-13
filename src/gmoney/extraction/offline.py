from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import cv2
import typer

from gmoney.contracts.extraction import CanonicalRow, RowRole, TableType
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_rows import (
    TableSchemaState,
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
    row_category,
    tokens_in_box,
)
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
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VlAsset:
    path: Path
    artifact_sha256: str
    identity: str


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
    horizontal_padding: int = 20,
    vertical_padding: int = 100,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, left - horizontal_padding),
        max(0, top - vertical_padding),
        min(width, right + horizontal_padding),
        min(height, bottom + vertical_padding),
    )


def _merge_table_boxes(
    layout_boxes: list[tuple[int, int, int, int]],
    geometry_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Fuse layout proposals with OCR row geometry, expanding partial tables."""
    merged = list(layout_boxes)
    for proposal in geometry_boxes:
        p_left, p_top, p_right, p_bottom = proposal
        p_area = (p_right - p_left) * (p_bottom - p_top)
        match_index: int | None = None
        for index, candidate in enumerate(merged):
            left, top, right, bottom = candidate
            intersection = max(0, min(right, p_right) - max(left, p_left)) * max(
                0, min(bottom, p_bottom) - max(top, p_top)
            )
            candidate_area = (right - left) * (bottom - top)
            if intersection / max(1, min(candidate_area, p_area)) >= 0.5:
                match_index = index
                break
        if match_index is None:
            merged.append(proposal)
            continue
        left, top, right, bottom = merged[match_index]
        merged[match_index] = (
            min(left, p_left),
            min(top, p_top),
            max(right, p_right),
            max(bottom, p_bottom),
        )
    return sorted(merged, key=lambda box: (box[1], box[0]))


def _write_vl_image(image, output: Path) -> VlAsset:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write VLM image: {temporary}")
    temporary.replace(output)
    return VlAsset(output, sha256_file(output), output.stem)


def _vl_asset(work: TableWork, artifact_root: Path, orientation: str) -> VlAsset:
    if orientation == "upright":
        return VlAsset(work.crop_path, work.crop_sha256, work.table_id)
    image = cv2.imread(str(work.crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read table crop: {work.crop_path}")
    rotation = (
        cv2.ROTATE_90_CLOCKWISE if orientation == "clockwise_90" else cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    rotated = cv2.rotate(image, rotation)
    return _write_vl_image(
        rotated,
        artifact_root / "crops" / f"{work.table_id}-oriented.png",
    )


def _vertical_vl_tiles(
    asset: VlAsset,
    artifact_root: Path,
    table_id: str,
    *,
    tile_height: int = 1600,
    overlap: int = 160,
) -> tuple[VlAsset, ...]:
    image = cv2.imread(str(asset.path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read VLM image: {asset.path}")
    height = image.shape[0]
    if height <= tile_height:
        return ()
    output: list[VlAsset] = []
    top = 0
    index = 1
    while top < height:
        bottom = min(height, top + tile_height)
        output.append(
            _write_vl_image(
                image[top:bottom],
                artifact_root / "crops" / f"{table_id}-tile-{index}.png",
            )
        )
        if bottom == height:
            break
        top = bottom - overlap
        index += 1
    return tuple(output)


def _deduplicate(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    selected: dict[tuple[object, ...], CanonicalRow] = {}
    for row in rows:
        description = re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip()
        amount_evidence_ids = tuple(
            sorted(
                token_id
                for evidence in row.field_evidence.get("amount", ())
                for token_id in evidence.token_ids
            )
        )
        key = (row.page_number, row.net_amount, amount_evidence_ids or description)
        current = selected.get(key)
        score = (
            "ocr_spatial_graph" in row.source_routes,
            len(row.field_evidence),
            len(description),
        )
        current_score = (
            bool(current and "ocr_spatial_graph" in current.source_routes),
            len(current.field_evidence) if current else -1,
            len(current.description or "") if current else -1,
        )
        if current is None or score > current_score:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row.page_number, row.row_order))


def _apply_document_role_policy(rows: list[CanonicalRow]) -> list[CanonicalRow]:
    """Apply the authoritative finest-available charge policy.

    A category charge is retained only when no item-level row covers that same
    category. Unlike the old exact-sum heuristic, this cannot delete an
    unrelated legitimate row merely because later values happen to add to it.
    """
    detailed_categories = {
        row.section for row in rows if row.role in {RowRole.DETAIL, RowRole.REFUND} and row.section
    }
    detailed_keys = {
        (
            re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip(),
            row.net_amount,
        )
        for row in rows
        if row.role in {RowRole.DETAIL, RowRole.REFUND}
    }
    selected = [
        row
        for row in rows
        if not (
            row.role is RowRole.CATEGORY_ROLLUP
            and (
                row.net_amount == 0
                or (row.section is not None and row.section in detailed_categories)
                or (
                    re.sub(r"[^a-z0-9]+", " ", (row.description or "").casefold()).strip(),
                    row.net_amount,
                )
                in detailed_keys
            )
        )
    ]
    return [row.model_copy(update={"row_order": order}) for order, row in enumerate(selected)]


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
        schema_states: list[TableSchemaState] = []
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
            layout_boxes = _layout_boxes(layout_response.output)
            result = (ocr_response.output.get("pages") or [{}])[0].get("res") or {}
            proposals = propose_tables_from_ocr(
                result.get("rec_boxes") or [],
                result.get("rec_texts") or [],
            )
            geometry_boxes = [
                tuple(round(value) for value in proposal.box) for proposal in proposals
            ]
            boxes = _merge_table_boxes(layout_boxes, geometry_boxes)
            route = (
                "layout+ocr_geometry"
                if layout_boxes and geometry_boxes
                else ("layout" if layout_boxes else "ocr_geometry")
            )

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
                        box=safe_box,
                    )
                )

            if not table_work:
                diagnostics.append(
                    {
                        "page_number": page_asset.page_number,
                        "route": route,
                        "ocr_latency_ms": ocr_response.latency_ms,
                        "ocr_cache_hit": ocr_cache_hit,
                        "layout_latency_ms": layout_response.latency_ms,
                        "layout_cache_hit": layout_cache_hit,
                        "table_count": 0,
                        "status": "no_table_detected",
                    }
                )
            for work in table_work:
                reconstruction = reconstruct_ocr_rows(
                    tokens,
                    page_number=work.page_number,
                    table_id=work.table_id,
                    box=work.box,
                    prior_schemas=tuple(schema_states),
                )
                if reconstruction.schema is not None:
                    schema_states.append(reconstruction.schema)
                parsed_rows = list(
                    canonicalize_rows(
                        document_id,
                        work.page_number,
                        work.table_id,
                        work.page_artifact_sha256,
                        reconstruction.rows,
                    )
                )
                content = ""
                candidate_count = 0
                provider_candidates = []
                vl_latency_ms = 0
                vl_cache_hit = False
                vl_finish_reason: str | None = None
                vl_truncated = False
                vl_tile_count = 0
                vl_retry_reason: str | None = None
                # OCR is the deterministic primary route. The heavy parser is
                # invoked only when OCR could not reconstruct a trustworthy
                # table, avoiding context pressure on dense, already-readable
                # ledgers.
                advisor_eligible = bool(
                    reconstruction.schema is None
                    or reconstruction.schema.table_type
                    not in {
                        TableType.CATEGORY_SUMMARY,
                        TableType.PACKAGE_SUMMARY,
                        TableType.PAYMENT,
                        TableType.METADATA,
                    }
                )
                use_vl = bool(
                    advisor_eligible
                    and (
                        not parsed_rows
                        or reconstruction.diagnostics.get("orientation") != "upright"
                        or (
                            not reconstruction.diagnostics.get("header_found")
                            and not reconstruction.diagnostics.get("schema_inherited")
                        )
                    )
                )
                if use_vl:
                    scoped_tokens = tokens_in_box(tokens, work.box)
                    vl_cache_hits: list[bool] = []
                    vl_contents: list[str] = []
                    orientation = str(reconstruction.diagnostics.get("orientation") or "upright")
                    primary_asset = _vl_asset(work, artifact_root, orientation)
                    vl_jobs: list[tuple[VlAsset, str, int | None]] = [
                        (primary_asset, f"{work.table_id}.vl.json", None)
                    ]
                    job_index = 0
                    while job_index < len(vl_jobs):
                        asset, cache_name, tile_index = vl_jobs[job_index]
                        vl_request = InferenceRequest(
                            request_id=str(uuid4()),
                            artifact_sha256=asset.artifact_sha256,
                            image_path=str(asset.path.resolve()),
                            page_number=work.page_number,
                            options={
                                "prompt": "Table Recognition:",
                                "prompt_version": "phase2-grounded-v2",
                                "max_tokens": 8192 if tile_index is not None else 16384,
                                "context_total": 24576,
                                "parallel_slots": 1,
                                "tiling_policy": "orientation-aware-overlap-v2",
                                "asset_identity": asset.identity,
                                "tile_index": tile_index,
                            },
                        )
                        vl_response, cache_hit = _cached_prediction(
                            artifact_root / "inference" / cache_name,
                            vl_request,
                            self.vl,
                        )
                        vl_cache_hits.append(cache_hit)
                        vl_latency_ms += vl_response.latency_ms
                        response_content = str(vl_response.output.get("content") or "")
                        vl_contents.append(response_content)
                        response_candidate_count = 0
                        for table in split_otsl_tables(parse_otsl(response_content)):
                            candidates = extract_candidate_rows(table)
                            if reconstruction.schema is not None:
                                candidates = tuple(
                                    replace(
                                        candidate,
                                        role=(
                                            RowRole.CATEGORY_ROLLUP
                                            if reconstruction.schema.table_type
                                            in {
                                                TableType.CATEGORY_SUMMARY,
                                                TableType.PACKAGE_SUMMARY,
                                            }
                                            and candidate.role is RowRole.DETAIL
                                            else candidate.role
                                        ),
                                        table_type=reconstruction.schema.table_type,
                                        category=row_category(
                                            candidate.description or "",
                                            reconstruction.schema.table_type,
                                        ),
                                    )
                                    for candidate in candidates
                                )
                            provider_candidates.extend(candidates)
                            response_candidate_count += len(candidates)
                            aligned = align_candidate_rows(candidates, scoped_tokens)
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
                        candidate_count += response_candidate_count
                        response_truncated = bool(vl_response.output.get("truncated"))
                        vl_truncated = vl_truncated or response_truncated
                        if job_index == 0:
                            vl_finish_reason = vl_response.output.get("finish_reason")
                            if response_truncated or response_candidate_count == 0:
                                vl_retry_reason = (
                                    "truncated" if response_truncated else "no_candidates"
                                )
                                tiles = _vertical_vl_tiles(
                                    primary_asset,
                                    artifact_root,
                                    work.table_id,
                                )
                                vl_tile_count = len(tiles)
                                vl_jobs.extend(
                                    (
                                        tile_asset,
                                        f"{work.table_id}.tile-{index}.vl.json",
                                        index,
                                    )
                                    for index, tile_asset in enumerate(tiles, start=1)
                                )
                        job_index += 1

                    vl_cache_hit = bool(vl_cache_hits) and all(vl_cache_hits)
                    content = "\n".join(vl_contents)
                fused_rows = fuse_provider_descriptions(
                    reconstruction.rows,
                    tuple(provider_candidates),
                )
                if fused_rows != reconstruction.rows:
                    parsed_rows.extend(
                        canonicalize_rows(
                            document_id,
                            work.page_number,
                            work.table_id,
                            work.page_artifact_sha256,
                            fused_rows,
                            starting_order=len(parsed_rows),
                        )
                    )
                all_rows.extend(parsed_rows)
                diagnostics.append(
                    {
                        "page_number": page_asset.page_number,
                        "route": route,
                        "ocr_latency_ms": ocr_response.latency_ms,
                        "ocr_cache_hit": ocr_cache_hit,
                        "layout_latency_ms": layout_response.latency_ms,
                        "layout_cache_hit": layout_cache_hit,
                        "table_id": work.table_id,
                        "crop_sha256": work.crop_sha256,
                        "box": work.box,
                        "vl_invoked": use_vl,
                        "vl_latency_ms": vl_latency_ms,
                        "vl_cache_hit": vl_cache_hit,
                        "vl_finish_reason": vl_finish_reason,
                        "vl_truncated": vl_truncated,
                        "vl_tile_count": vl_tile_count,
                        "vl_retry_reason": vl_retry_reason,
                        "candidate_count": candidate_count,
                        "canonical_count": len(parsed_rows),
                        "content": content,
                        **reconstruction.diagnostics,
                    }
                )
            if progress:
                progress(page_asset.page_number, len(manifest.pages))

        rows = _apply_document_role_policy(_deduplicate(all_rows))
        return {
            "output_version": "offline_accuracy_spine_v2",
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
