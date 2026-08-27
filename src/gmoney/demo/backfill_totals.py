from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.demo.store import JobStore
from gmoney.extraction.document_total import (
    DOCUMENT_TOTAL_VERSION,
    DOCUMENT_TOTALS_VERSION,
    DocumentTotalCandidate,
    extract_document_total_candidates,
    select_document_total,
    select_document_totals,
)
from gmoney.extraction.ocr_tokens import paddle_ocr_tokens

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cached_tokens(job_dir: Path, result: dict[str, Any]) -> tuple[DocumentTotalCandidate, ...]:
    candidates: list[DocumentTotalCandidate] = []
    for asset in result.get("page_assets", []):
        page_number = int(asset["page_number"])
        artifact_sha256 = str(asset["artifact_sha256"])
        cache_path = job_dir / "artifacts" / "inference" / f"page-{page_number}.ocr.json"
        if not cache_path.is_file():
            raise ValueError(f"missing OCR cache for page {page_number}")
        envelope = json.loads(cache_path.read_text())
        if envelope.get("artifact_sha256") != artifact_sha256:
            raise ValueError(f"OCR cache hash differs for page {page_number}")
        response = envelope.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("output"), dict):
            raise ValueError(f"OCR cache response is invalid for page {page_number}")
        tokens = paddle_ocr_tokens(response["output"], page_number, artifact_sha256)
        candidates.extend(extract_document_total_candidates(tokens))
    return tuple(candidates)


def backfill_totals(*, root: Path, apply: bool = False) -> dict[str, Any]:
    store = JobStore(root)
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "updated": 0,
        "would_update": 0,
        "already_current": 0,
        "not_found": 0,
        "failed": 0,
        "failures": [],
    }
    for state in store.states():
        if state.get("status") not in {"complete", "needs_review"}:
            continue
        job_id = str(state["id"])
        result_path = store.job_dir(job_id) / "result.json"
        try:
            result = json.loads(result_path.read_text())
            if (
                result.get("document_total_version") == DOCUMENT_TOTAL_VERSION
                and result.get("document_totals_version") == DOCUMENT_TOTALS_VERSION
            ):
                summary["already_current"] += 1
                continue
            candidates = _cached_tokens(store.job_dir(job_id), result)
            total = select_document_total(candidates)
            totals = select_document_totals(candidates)
            if total is None:
                summary["not_found"] += 1
            if not apply:
                summary["would_update"] += 1
                continue
            updated = {
                **result,
                "document_total_version": DOCUMENT_TOTAL_VERSION,
                "document_totals_version": DOCUMENT_TOTALS_VERSION,
                "document_total": total.model_dump(mode="json") if total is not None else None,
                "document_totals": [item.model_dump(mode="json") for item in totals],
            }
            _atomic_json(result_path, updated)
            summary["updated"] += 1
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            summary["failed"] += 1
            summary["failures"].append({"job_id": job_id, "error": str(error)})
    return summary


@app.callback()
def run(
    root: Annotated[Path, typer.Option(file_okay=False, resolve_path=True)],
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Atomically update completed result files."),
    ] = False,
) -> None:
    summary = backfill_totals(root=root, apply=apply)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
