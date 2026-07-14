from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import NAMESPACE_URL, uuid5

import typer

from gmoney.demo.store import JobStore, utc_now

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hardlink(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise RuntimeError(
                "History results, sources, artifacts, and runtime must share one filesystem"
            ) from error
        raise
    return destination


def _link_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=_hardlink, symlinks=False)


def _safe_relative(root: Path, value: str) -> Path:
    target = (root / value).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Artifact path escapes its root: {value}")
    return target


def _validate_inputs(
    result: dict[str, Any],
    source: Path,
    artifact_root: Path,
) -> None:
    source_sha = str(result.get("source_sha256") or "")
    if source_sha and _digest(source) != source_sha:
        raise ValueError(f"Source hash differs for {source.name}")
    assets = result.get("page_assets", [])
    if len(assets) != int(result.get("pages") or 0):
        raise ValueError(f"Page inventory differs for {source.name}")
    for asset in assets:
        page_path = _safe_relative(artifact_root, str(asset["relative_path"]))
        if not page_path.is_file():
            raise ValueError(f"Page artifact is missing: {page_path}")
        if _digest(page_path) != str(asset["artifact_sha256"]):
            raise ValueError(f"Page artifact hash differs: {page_path}")


def _source_index(root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for path in root.rglob("*.pdf"):
        key = path.name.casefold()
        if key in sources and sources[key].resolve() != path.resolve():
            raise ValueError(f"Duplicate source filename: {path.name}")
        sources[key] = path
    return sources


def import_history(
    *,
    root: Path,
    results: Path,
    artifacts: Path,
    sources: Path,
    batch_id: str,
) -> dict[str, Any]:
    store = JobStore(root)
    safe_batch = re.sub(r"[^A-Za-z0-9._-]+", "-", batch_id).strip("-.")
    if not safe_batch:
        raise ValueError("batch_id must contain a letter or number")
    manifest_path = root / "imports" / f"{safe_batch}.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {
            "batch_id": batch_id,
            "created_at": utc_now(),
            "documents": {},
        }
    )
    source_by_name = _source_index(sources)
    imported = 0
    skipped = 0
    for result_path in sorted(results.glob("*.json")):
        result = json.loads(result_path.read_text())
        document_id = str(result.get("document_id") or "")
        if not document_id:
            raise ValueError(f"Result has no document_id: {result_path}")
        if document_id in manifest["documents"]:
            skipped += 1
            continue
        original_name = Path(str(result.get("source_name") or result_path.stem)).name
        source = source_by_name.get(original_name.casefold())
        if source is None:
            raise ValueError(f"Source PDF is missing for {original_name}")
        artifact_root = artifacts / result_path.stem
        if not artifact_root.is_dir():
            raise ValueError(f"Artifact directory is missing: {artifact_root}")
        _validate_inputs(result, source, artifact_root)

        job_id = str(uuid5(NAMESPACE_URL, f"gmoney-demo-history:{document_id}"))
        target = store.job_dir(job_id)
        hospital = result.get("hospital") or {}
        created_at = next(
            (
                str(row["created_at"])
                for row in result.get("rows", [])
                if row.get("created_at")
            ),
            utc_now(),
        )
        if target.exists():
            existing_path = target / "result.json"
            existing = json.loads(existing_path.read_text()) if existing_path.is_file() else {}
            if str(existing.get("document_id") or "") != document_id:
                raise ValueError(f"History job ID collision: {job_id}")
        else:
            temporary = store.jobs_root / f".{job_id}.import"
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(mode=0o700)
            try:
                _hardlink(str(source), str(temporary / "source.pdf"))
                _hardlink(str(result_path), str(temporary / "result.json"))
                _link_tree(artifact_root, temporary / "artifacts")
                temporary.replace(target)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        if not (target / "state.json").is_file():
            store.write(
                job_id,
                {
                    "id": job_id,
                    "status": "complete",
                    "original_name": original_name[:200],
                    "created_at": created_at,
                    "page": int(result.get("pages") or 0),
                    "pages": int(result.get("pages") or 0),
                    "row_count": len(result.get("rows", [])),
                    "hospital_name": hospital.get("name"),
                    "hospital_confidence": hospital.get("confidence"),
                    "error": None,
                    "import_batch": batch_id,
                    "imported_at": utc_now(),
                },
            )
        manifest["documents"][document_id] = {
            "job_id": job_id,
            "source_name": original_name,
            "imported_at": utc_now(),
        }
        _atomic_json(manifest_path, manifest)
        imported += 1
    return {
        "batch_id": batch_id,
        "imported": imported,
        "skipped": skipped,
        "documents": len(manifest["documents"]),
        "manifest": str(manifest_path),
    }


@app.command("run")
def run(
    root: Annotated[Path, typer.Option(file_okay=False)],
    results: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    artifacts: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    sources: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    batch_id: Annotated[str, typer.Option()],
) -> None:
    typer.echo(
        json.dumps(
            import_history(
                root=root,
                results=results,
                artifacts=artifacts,
                sources=sources,
                batch_id=batch_id,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
