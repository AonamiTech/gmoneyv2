from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.contracts.gold import GoldAnnotation
from gmoney.evaluation.field_quality import DocumentFields, evaluate_with_layout_slices

app = typer.Typer(no_args_is_help=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_documents(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    split: str,
    actual_key: str,
) -> tuple[list[DocumentFields], list[str]]:
    documents: list[DocumentFields] = []
    failures: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for entry in payload.get("documents", []):
        if entry.get("split") != split:
            continue
        document_id = str(entry["document_id"])
        if document_id in seen_ids:
            failures.append(f"duplicate_document_id:{document_id}")
            continue
        seen_ids.add(document_id)
        gold_path = _resolve(manifest_path.parent, str(entry["gold"]))
        actual_path = _resolve(manifest_path.parent, str(entry[actual_key]))
        gold = GoldAnnotation.model_validate_json(gold_path.read_text())
        digest = gold.document_sha256
        if not digest:
            failures.append(f"missing_document_sha256:{document_id}")
        elif digest in seen_hashes:
            failures.append(f"duplicate_document_sha256:{digest}")
        else:
            seen_hashes.add(digest)
        if not gold.image_review or gold.image_review.passes < 2:
            failures.append(f"two_pass_image_review_required:{document_id}")
        layout = str(entry.get("layout_family_id") or gold.layout_family_id or "")
        if not layout:
            failures.append(f"layout_family_required:{document_id}")
            layout = "unknown"
        documents.append(
            DocumentFields(
                document_id=document_id,
                layout_family_id=layout,
                gold=gold,
                actual=json.loads(actual_path.read_text()),
            )
        )
    if not documents:
        failures.append(f"no_documents_for_split:{split}")
    return documents, failures


@app.command("run")
def run(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    split: str = "holdout",
    threshold: float = 0.95,
) -> None:
    if not 0 < threshold < 1:
        raise typer.BadParameter("threshold must be between zero and one")
    payload = json.loads(manifest.read_text())
    candidate, review_failures = _load_documents(
        manifest, payload, split=split, actual_key="candidate"
    )
    candidate_report = evaluate_with_layout_slices(candidate, threshold=threshold)
    baseline_report = None
    selected_entries = [
        entry for entry in payload.get("documents", []) if entry.get("split") == split
    ]
    if selected_entries and all(entry.get("baseline") for entry in selected_entries):
        baseline, baseline_failures = _load_documents(
            manifest, payload, split=split, actual_key="baseline"
        )
        review_failures.extend(baseline_failures)
        baseline_report = evaluate_with_layout_slices(baseline, threshold=threshold)
    blocking = [*review_failures, *candidate_report["blocking_reasons"]]
    result = {
        "report_version": "field_quality_report_v1",
        "evaluated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": _sha256(manifest),
        "split": split,
        "candidate": candidate_report,
        "baseline": baseline_report,
        "blocking_reasons": blocking,
        "passed": not blocking,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps({"passed": result["passed"], "blocking": blocking}))
    if blocking:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
