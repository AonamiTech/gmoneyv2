from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.contracts.gold import GoldAnnotation

app = typer.Typer(no_args_is_help=True)
HOSPITAL_ID = re.compile(r"(H\d{5,}[A-Z0-9]*)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_hospital_id(name: str) -> str:
    match = HOSPITAL_ID.search(name)
    if match:
        return match.group(1).upper()
    return "UNKNOWN-" + hashlib.sha256(name.encode()).hexdigest()[:12]


def grouped_splits(groups: list[str], seed: str) -> dict[str, str]:
    ordered = sorted(
        set(groups),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    count = len(ordered)
    if count >= 3:
        validation_count = max(1, round(count * 0.2))
        holdout_count = max(1, round(count * 0.2))
        train_count = count - validation_count - holdout_count
        if train_count < 1:
            train_count, validation_count, holdout_count = 1, 1, count - 2
    else:
        train_count, validation_count, holdout_count = count, 0, 0
    validation_end = train_count + validation_count
    assignments: dict[str, str] = {}
    for index, group in enumerate(ordered):
        if index < train_count:
            assignments[group] = "train"
        elif index < validation_end:
            assignments[group] = "validation"
        else:
            assignments[group] = "holdout"
    return assignments


def build_catalog(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    root = Path(os.getenv(config["legacy_root_env"], config["default_legacy_root"]))
    if not root.is_dir():
        raise FileNotFoundError(f"legacy corpus root does not exist: {root}")

    entries: list[dict[str, Any]] = []
    for source in config["sources"]:
        source_path = root / source["path"]
        if source["kind"] == "gold_directory":
            for path in sorted(source_path.glob(source.get("pattern", "*.json"))):
                raw = json.loads(path.read_text())
                hospital_id = infer_hospital_id(path.name)
                annotation = GoldAnnotation.model_validate({**raw, "hospital_id": hospital_id})
                entries.append(
                    {
                        "kind": "gold_annotation",
                        "source": source["name"],
                        "relative_path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                        "hospital_id": hospital_id,
                        "bill_file": annotation.bill_file,
                        "row_count": len(annotation.rows),
                    }
                )
        elif source["kind"] == "pdf_directory":
            for path in sorted(source_path.glob(source.get("pattern", "*.pdf"))):
                entries.append(
                    {
                        "kind": "pdf",
                        "source": source["name"],
                        "relative_path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                        "hospital_id": infer_hospital_id(path.name),
                        "size_bytes": path.stat().st_size,
                    }
                )
        elif source["kind"] == "document_manifest":
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            entries.append(
                {
                    "kind": "document_manifest",
                    "source": source["name"],
                    "relative_path": str(source_path.relative_to(root)),
                    "sha256": sha256_file(source_path),
                }
            )
        else:
            raise ValueError(f"unsupported source kind: {source['kind']}")

    gold_entries = [entry for entry in entries if entry["kind"] == "gold_annotation"]
    assignments = grouped_splits(
        [entry["hospital_id"] for entry in gold_entries],
        config["split"]["seed"],
    )
    for entry in gold_entries:
        entry["split"] = assignments[entry["hospital_id"]]

    split_counts: dict[str, int] = defaultdict(int)
    split_rows: dict[str, int] = defaultdict(int)
    for entry in gold_entries:
        split_counts[entry["split"]] += 1
        split_rows[entry["split"]] += entry["row_count"]

    return {
        "catalog_version": "corpus_catalog_v1",
        "registry_version": config["version"],
        "split_seed": config["split"]["seed"],
        "forbidden_imports": config["forbidden_imports"],
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "gold_count": len(gold_entries),
            "gold_rows": sum(entry["row_count"] for entry in gold_entries),
            "split_documents": dict(sorted(split_counts.items())),
            "split_rows": dict(sorted(split_rows.items())),
        },
    }


def freeze_phase3_candidates(
    catalog: dict[str, Any],
    *,
    exposed_sha256: str,
    required_hospitals: int = 10,
) -> dict[str, Any]:
    samples = [
        entry
        for entry in catalog.get("entries", [])
        if entry.get("kind") == "pdf" and entry.get("source") == "sample_bills"
    ]
    if not samples:
        raise ValueError("catalog contains no sample bills")
    if sum(entry.get("sha256") == exposed_sha256 for entry in samples) != 1:
        raise ValueError("exposed sample hash must identify exactly one bill")
    candidates = []
    for entry in sorted(samples, key=lambda item: item["relative_path"]):
        exposed = entry["sha256"] == exposed_sha256
        candidates.append(
            {
                "relative_path": entry["relative_path"],
                "sha256": entry["sha256"],
                "status": "exposed_regression" if exposed else "candidate_unseen",
                "hospital_id": None,
                "identity_status": "not_applicable" if exposed else "pending_review",
                "annotation_status": "not_required" if exposed else "required",
                "eligible_for_frozen_gate": False,
            }
        )
    candidate_count = sum(item["status"] == "candidate_unseen" for item in candidates)
    return {
        "manifest_version": "phase3_sealed_candidates_v1",
        "source_catalog_version": catalog.get("catalog_version"),
        "required_distinct_hospitals": required_hospitals,
        "entries": candidates,
        "summary": {
            "candidate_documents": candidate_count,
            "confirmed_distinct_hospitals": 0,
            "frozen_gold_documents": 0,
            "gate_ready": False,
            "blocking_reasons": [
                "hospital_identity_review_required",
                "frozen_gold_annotation_required",
            ],
        },
    }


@app.command("build")
def build(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    catalog = build_catalog(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(catalog["summary"], sort_keys=True))


@app.command("validate")
def validate(
    catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    payload = json.loads(catalog.read_text())
    gold = [entry for entry in payload["entries"] if entry["kind"] == "gold_annotation"]
    by_hospital: dict[str, set[str]] = defaultdict(set)
    for entry in gold:
        by_hospital[entry["hospital_id"]].add(entry["split"])
    leaked = {key: value for key, value in by_hospital.items() if len(value) != 1}
    if leaked:
        raise typer.Exit(f"hospital split leakage: {leaked}")
    typer.echo(f"valid catalog: {len(gold)} gold annotations, no hospital leakage")


@app.command("freeze-phase3-candidates")
def freeze_phase3(
    catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    exposed_sha256: Annotated[str, typer.Option()],
    required_hospitals: int = 10,
) -> None:
    manifest = freeze_phase3_candidates(
        json.loads(catalog.read_text()),
        exposed_sha256=exposed_sha256,
        required_hospitals=required_hospitals,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(manifest["summary"], sort_keys=True))
