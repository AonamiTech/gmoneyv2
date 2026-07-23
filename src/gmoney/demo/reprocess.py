from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from gmoney.contracts.extraction import SourceTable
from gmoney.demo.review import structural_issues
from gmoney.demo.store import JobStore, utc_now
from gmoney.evaluation.corpus import sha256_file
from gmoney.extraction.offline import OfflineExtractor
from gmoney.extraction.typed_values import parse_decimal

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass(frozen=True)
class PreparedJob:
    job_id: str
    stage_dir: Path
    old_result: dict[str, Any]
    new_result: dict[str, Any]
    migrated_review: dict[str, Any]
    review_marker: tuple[bool, str | None]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _file_digest(path: Path) -> str:
    return sha256_file(path)


def _review_marker(job_dir: Path) -> tuple[bool, str | None]:
    path = job_dir / "review.json"
    return (path.is_file(), _file_digest(path) if path.is_file() else None)


def _marker_payload(marker: tuple[bool, str | None]) -> dict[str, Any]:
    return {"exists": marker[0], "sha256": marker[1]}


def _marker_from_payload(payload: dict[str, Any]) -> tuple[bool, str | None]:
    return bool(payload.get("exists")), (
        str(payload["sha256"]) if payload.get("sha256") is not None else None
    )


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _field_token_ids(row: dict[str, Any], field: str | None = None) -> set[str]:
    evidence_by_field = row.get("field_evidence") or {}
    evidence = evidence_by_field.get(field, []) if field else [
        item for values in evidence_by_field.values() for item in values
    ]
    return {
        str(token_id)
        for item in evidence
        for token_id in (item.get("token_ids") or [])
    }


def _map_reviewed_row(
    old_row: dict[str, Any], new_rows: list[dict[str, Any]]
) -> str | None:
    old_description_ids = _field_token_ids(old_row, "description")
    old_all_ids = _field_token_ids(old_row)
    old_description = _normalized(old_row.get("description"))
    scored: list[tuple[tuple[int, int, float], str]] = []
    for candidate in new_rows:
        if int(candidate.get("page_number") or 0) != int(old_row.get("page_number") or 0):
            continue
        description_overlap = len(
            old_description_ids & _field_token_ids(candidate, "description")
        )
        all_overlap = len(old_all_ids & _field_token_ids(candidate))
        similarity = SequenceMatcher(
            None,
            old_description,
            _normalized(candidate.get("description")),
        ).ratio()
        if description_overlap or all_overlap:
            scored.append(
                (
                    (description_overlap, all_overlap, similarity),
                    str(candidate["id"]),
                )
            )
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _migrate_review(
    job_id: str,
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    old_rows = {str(row["id"]): row for row in old_result.get("rows", [])}
    new_rows = list(new_result.get("rows", []))
    new_ids = {str(row["id"]) for row in new_rows}
    migrated_overrides: dict[str, Any] = {}
    migrated_added = json.loads(json.dumps(review.get("added_rows", {})))
    preserved_unmapped: list[str] = []
    for old_id, override in review.get("row_overrides", {}).items():
        target_id = old_id if old_id in new_ids else None
        if target_id is None:
            old_row = old_rows.get(old_id)
            if old_row is None:
                raise ValueError(f"reviewed row is absent from the old result: {old_id}")
            target_id = _map_reviewed_row(old_row, new_rows)
        if target_id is None:
            preserved = json.loads(json.dumps(old_rows[old_id]))
            preserved.update(override.get("changes", {}))
            preserved_id = f"reprocessed-{old_id}"
            suffix = 1
            while preserved_id in migrated_added or preserved_id in new_ids:
                suffix += 1
                preserved_id = f"reprocessed-{old_id}-{suffix}"
            preserved["id"] = preserved_id
            preserved["contract_version"] = "canonical_row_reviewer_v1"
            preserved["source_routes"] = sorted(
                {*preserved.get("source_routes", []), "reviewer", "reprocess_preserved"}
            )
            preserved["validation_flags"] = sorted(
                {*preserved.get("validation_flags", []), "reviewer_preserved"}
            )
            preserved["review_reason"] = override.get("reason") or (
                "Reviewer correction preserved because the upgraded machine row "
                "could not be mapped uniquely"
            )
            migrated_added[preserved_id] = preserved
            preserved_unmapped.append(old_id)
            continue
        if target_id in migrated_overrides:
            raise ValueError(f"multiple reviewed rows map to {target_id}")
        migrated_overrides[target_id] = override

    issue_probe = {**review, "issue_overrides": {}}
    new_issue_ids = {
        str(issue["id"]) for issue in structural_issues(new_result, issue_probe)
    }
    old_issue_overrides = review.get("issue_overrides", {})
    missing_issues = set(old_issue_overrides) - new_issue_ids
    active_issue_overrides = {
        issue_id: override
        for issue_id, override in old_issue_overrides.items()
        if issue_id in new_issue_ids
    }

    revision = int(review.get("revision") or 0) + 1
    migrated = json.loads(json.dumps(review))
    migrated["revision"] = revision
    migrated["updated_at"] = utc_now()
    migrated["row_overrides"] = migrated_overrides
    migrated["added_rows"] = migrated_added
    migrated["issue_overrides"] = active_issue_overrides
    archived = migrated.setdefault("archived_issue_overrides", {})
    archived.update(
        {issue_id: old_issue_overrides[issue_id] for issue_id in sorted(missing_issues)}
    )
    migrated["approval"] = None
    migrated.setdefault("events", []).append(
        {
            "revision": revision,
            "action": "document_reprocessed",
            "target_id": job_id,
            "reviewer": "system-reprocessor",
            "reason": "Machine extraction upgraded with preserved review history",
            "changes": {
                "old_rows": len(old_result.get("rows", [])),
                "new_rows": len(new_result.get("rows", [])),
                "mapped_row_overrides": len(migrated_overrides),
                "preserved_unmapped_row_overrides": preserved_unmapped,
                "archived_issue_overrides": sorted(missing_issues),
            },
            "created_at": utc_now(),
        }
    )
    return migrated


def _validate_result(
    source: Path,
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    artifact_root: Path,
) -> None:
    if new_result.get("document_id") != old_result.get("document_id"):
        raise ValueError("document identity changed")
    if new_result.get("source_sha256") != _file_digest(source):
        raise ValueError("source hash changed")
    if int(new_result.get("pages") or 0) != int(old_result.get("pages") or 0):
        raise ValueError("page count changed")
    assets = new_result.get("page_assets", [])
    if len(assets) != int(new_result.get("pages") or 0):
        raise ValueError("page inventory is incomplete")
    for asset in assets:
        path = (artifact_root / str(asset["relative_path"])).resolve()
        if artifact_root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"page artifact is missing: {asset['relative_path']}")
        if _file_digest(path) != str(asset["artifact_sha256"]):
            raise ValueError(f"page artifact hash changed: {asset['relative_path']}")
    for row in new_result.get("rows", []):
        fields = row.get("field_evidence") or {}
        if not row.get("description") or "description" not in fields:
            raise ValueError(f"row lacks grounded description: {row.get('id')}")
        if row.get("role") == "informational":
            if row.get("net_amount") is not None:
                raise ValueError(f"informational row has an amount: {row.get('id')}")
            if not {"service_date", "request_no", "service_code", "hsn_code"}.intersection(
                fields
            ):
                raise ValueError(f"informational row lacks typed evidence: {row.get('id')}")
        elif row.get("role") in {"detail", "refund", "category_rollup"}:
            if row.get("net_amount") is None or "amount" not in fields:
                raise ValueError(f"billable row lacks grounded amount: {row.get('id')}")

    source_payload = new_result.get("source_tables")
    if new_result.get("rows") and not source_payload:
        raise ValueError("canonical rows require validated source tables")
    try:
        source_tables = [
            SourceTable.model_validate(table) for table in source_payload or []
        ]
    except ValidationError as error:
        raise ValueError(f"source tables failed grounding validation: {error}") from error

    canonical_rows = {
        str(row["id"]): row for row in new_result.get("rows", [])
    }
    linked_ids: set[str] = set()
    evidence_fields = {
        "description": "description",
        "service_date_raw": "service_date",
        "request_no": "request_no",
        "service_code": "service_code",
        "hsn_code": "hsn_code",
        "quantity": "quantity",
        "unit_price": "rate",
        "gross_amount": "gross_amount",
        "discount": "discount",
        "net_amount": "amount",
    }
    numeric_fields = {
        "quantity",
        "unit_price",
        "gross_amount",
        "discount",
        "net_amount",
    }
    for table in source_tables:
        for source_row in table.rows:
            if source_row.canonical_row_id is None:
                continue
            canonical = canonical_rows.get(source_row.canonical_row_id)
            if canonical is None:
                raise ValueError(
                    f"source row links unknown canonical row: {source_row.canonical_row_id}"
                )
            if source_row.canonical_row_id in linked_ids:
                raise ValueError(
                    f"canonical row has multiple source links: {source_row.canonical_row_id}"
                )
            linked_ids.add(source_row.canonical_row_id)
            cells = {cell.column_id: cell for cell in source_row.cells}
            for column in table.columns:
                field = column.canonical_field
                if field is None:
                    continue
                cell = cells[column.id]
                canonical_value = canonical.get(field)
                printed_present = bool(cell.raw_value and cell.raw_value.strip())
                canonical_present = canonical_value is not None and (
                    not isinstance(canonical_value, str) or bool(canonical_value.strip())
                )
                if printed_present and not canonical_present:
                    raise ValueError(
                        f"{field} has a printed value but is missing canonical value "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if canonical_present and not printed_present:
                    raise ValueError(
                        f"{field} has a canonical value but is missing printed value "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if not printed_present:
                    continue
                evidence_field = evidence_fields[field]
                field_token_ids = {
                    str(token_id)
                    for item in (canonical.get("field_evidence") or {}).get(
                        evidence_field, []
                    )
                    for token_id in item.get("token_ids") or []
                }
                cell_token_ids = {
                    token_id
                    for item in cell.evidence
                    for token_id in item.token_ids
                }
                evidence_matches = (
                    bool(field_token_ids & cell_token_ids)
                    if field == "description"
                    else field_token_ids.issubset(cell_token_ids)
                )
                if not field_token_ids or not evidence_matches:
                    raise ValueError(
                        f"{field} evidence is not in its mapped source cell "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
                if field in numeric_fields:
                    printed_value = parse_decimal(cell.raw_value or "")
                    canonical_value = parse_decimal(str(canonical[field]))
                    if printed_value is None or printed_value != canonical_value:
                        raise ValueError(
                            f"{field} value does not match its mapped source cell "
                            f"for canonical row {source_row.canonical_row_id}"
                        )
                elif field != "description" and _normalized(
                    cell.raw_value
                ) != _normalized(canonical_value):
                    raise ValueError(
                        f"{field} value does not match its mapped source cell "
                        f"for canonical row {source_row.canonical_row_id}"
                    )
    for row_id, row_payload in canonical_rows.items():
        if (
            "ocr_spatial_graph" in (row_payload.get("source_routes") or [])
            and row_id not in linked_ids
        ):
            raise ValueError(f"canonical OCR row lacks a source-table link: {row_id}")


def _prepare_job(
    *,
    store: JobStore,
    job_id: str,
    stage_root: Path,
    extractor: Any,
) -> PreparedJob:
    job_dir = store.job_dir(job_id)
    source = job_dir / "source.pdf"
    result_path = job_dir / "result.json"
    artifact_root = job_dir / "artifacts"
    stage_dir = stage_root / job_id
    if stage_dir.exists():
        raise ValueError(f"staging directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=True, mode=0o700)
    with store.job_lock(job_id, exclusive=False):
        store._require_stable_workspace(job_id)
        state = store.read(job_id)
        if state.get("status") != "complete":
            raise ValueError("only complete jobs can be reprocessed")
        if not source.is_file() or not result_path.is_file() or not artifact_root.is_dir():
            raise ValueError("source, result, or artifacts are missing")
        old_result = json.loads(result_path.read_text())
        review = store._read_review_unlocked(job_id)
        marker = _review_marker(job_dir)
        shutil.copytree(artifact_root, stage_dir / "artifacts")
    new_result = extractor.extract(source, stage_dir / "artifacts")
    _validate_result(source, old_result, new_result, stage_dir / "artifacts")
    migrated_review = _migrate_review(job_id, old_result, new_result, review)
    _atomic_json(stage_dir / "result.json", new_result)
    _atomic_json(stage_dir / "review.json", migrated_review)
    return PreparedJob(
        job_id=job_id,
        stage_dir=stage_dir,
        old_result=old_result,
        new_result=new_result,
        migrated_review=migrated_review,
        review_marker=marker,
    )


def _cutover_job(
    *,
    store: JobStore,
    prepared: PreparedJob,
    backup_root: Path,
    commit_marker: Path,
) -> Path:
    job_dir = store.job_dir(prepared.job_id)
    backup_dir = backup_root / prepared.job_id
    journal_path = job_dir / ".cutover.json"

    def restore_path(name: str) -> None:
        live = job_dir / name
        backup = backup_dir / name
        staged = prepared.stage_dir / name
        if not backup.exists():
            return
        if live.exists():
            if not staged.exists():
                live.replace(staged)
            elif live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
        backup.replace(live)

    if backup_dir.exists():
        raise ValueError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, mode=0o700)
    _atomic_json(
        journal_path,
        {
            "version": "job_cutover_v1",
            "job_id": prepared.job_id,
            "stage_dir": str(prepared.stage_dir),
            "backup_dir": str(backup_dir),
            "commit_marker": str(commit_marker),
            "started_at": utc_now(),
        },
    )
    try:
        shutil.copy2(job_dir / "state.json", backup_dir / "state.json")
        if (job_dir / "review.json").is_file():
            shutil.copy2(job_dir / "review.json", backup_dir / "review.json")

        (job_dir / "artifacts").replace(backup_dir / "artifacts")
        (prepared.stage_dir / "artifacts").replace(job_dir / "artifacts")
        (job_dir / "result.json").replace(backup_dir / "result.json")
        (prepared.stage_dir / "result.json").replace(job_dir / "result.json")
        _atomic_json(job_dir / "review.json", prepared.migrated_review)
        hospital = prepared.new_result.get("hospital") or {}
        store.update(
            prepared.job_id,
            status="complete",
            page=int(prepared.new_result.get("pages") or 0),
            pages=int(prepared.new_result.get("pages") or 0),
            row_count=len(prepared.new_result.get("rows", [])),
            hospital_name=hospital.get("name"),
            hospital_confidence=hospital.get("confidence"),
            error=None,
            reprocessed_at=utc_now(),
        )
    except BaseException:
        try:
            restore_path("result.json")
            restore_path("artifacts")
            _atomic_json(
                job_dir / "state.json",
                json.loads((backup_dir / "state.json").read_text()),
            )
            if (backup_dir / "review.json").is_file():
                _atomic_json(
                    job_dir / "review.json",
                    json.loads((backup_dir / "review.json").read_text()),
                )
            elif (job_dir / "review.json").is_file():
                (job_dir / "review.json").unlink()
            journal_path.unlink(missing_ok=True)
            shutil.rmtree(backup_dir, ignore_errors=True)
        except BaseException as recovery_error:
            raise RuntimeError(
                f"cutover recovery required for job {prepared.job_id}"
            ) from recovery_error
        raise
    return backup_dir


def _rollback_cutover_job(
    *,
    store: JobStore,
    prepared: PreparedJob,
    backup_dir: Path,
) -> None:
    job_dir = store.job_dir(prepared.job_id)

    def restore_path(name: str) -> None:
        live = job_dir / name
        backup = backup_dir / name
        staged = prepared.stage_dir / name
        if live.exists():
            if not staged.exists():
                live.replace(staged)
            elif live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
        backup.replace(live)

    restore_path("result.json")
    restore_path("artifacts")
    _atomic_json(
        job_dir / "state.json",
        json.loads((backup_dir / "state.json").read_text()),
    )
    if (backup_dir / "review.json").is_file():
        _atomic_json(
            job_dir / "review.json",
            json.loads((backup_dir / "review.json").read_text()),
        )
    elif (job_dir / "review.json").is_file():
        (job_dir / "review.json").unlink()
    (job_dir / ".cutover.json").unlink(missing_ok=True)
    shutil.rmtree(backup_dir, ignore_errors=True)


def reprocess_jobs(
    *,
    root: Path,
    job_ids: list[str] | None = None,
    apply: bool = False,
    stage_root: Path | None = None,
    backup_root: Path | None = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    extractor: Any | None = None,
) -> dict[str, Any]:
    store = JobStore(root)
    selected = job_ids or [
        str(state["id"]) for state in store.states() if state.get("status") == "complete"
    ]
    selected = sorted(set(selected))
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "selected": len(selected),
        "would_reprocess": 0,
        "reprocessed": 0,
        "documents": [],
    }
    for job_id in selected:
        state = store.read(job_id)
        if state.get("status") != "complete":
            raise ValueError(f"job is not complete: {job_id}")
        summary["documents"].append(
            {
                "job_id": job_id,
                "source_name": state.get("original_name"),
                "rows_before": state.get("row_count"),
            }
        )
    if not apply:
        summary["would_reprocess"] = len(selected)
        return summary

    staged = stage_reprocess_jobs(
        root=root,
        job_ids=selected,
        stage_root=stage_root,
        vl_url=vl_url,
        paddle_device=paddle_device,
        vl_device=vl_device,
        extractor=extractor,
    )
    applied = apply_staged_jobs(
        root=root,
        stage_batch=Path(staged["staging_root"]),
        backup_root=backup_root,
    )
    return {**summary, **applied, "mode": "apply", "would_reprocess": len(selected)}


def stage_reprocess_jobs(
    *,
    root: Path,
    job_ids: list[str] | None = None,
    stage_root: Path | None = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    extractor: Any | None = None,
) -> dict[str, Any]:
    """Build and validate replacement results without mutating live jobs."""
    store = JobStore(root)
    selected = sorted(
        set(
            job_ids
            or [
                str(state["id"])
                for state in store.states()
                if state.get("status") == "complete"
            ]
        )
    )
    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    staging = (stage_root or store.jobs_root / ".reprocess-staging") / timestamp
    staging.mkdir(parents=True, mode=0o700)
    active_extractor = extractor or OfflineExtractor(
        vl_url,
        paddle_device=paddle_device,
        vl_device=vl_device,
    )
    prepared = [
        _prepare_job(
            store=store,
            job_id=job_id,
            stage_root=staging,
            extractor=active_extractor,
        )
        for job_id in selected
    ]
    if any(
        _review_marker(store.job_dir(item.job_id)) != item.review_marker
        for item in prepared
    ):
        raise ValueError("a review changed while staging; no results were applied")
    documents = [
        {
            "job_id": item.job_id,
            "source_name": item.old_result.get("source_name"),
            "rows_before": len(item.old_result.get("rows", [])),
            "rows_after": len(item.new_result.get("rows", [])),
            "review_marker": _marker_payload(item.review_marker),
        }
        for item in prepared
    ]
    _atomic_json(
        staging / "manifest.json",
        {
            "version": "reprocess_stage_v1",
            "created_at": utc_now(),
            "documents": documents,
        },
    )
    return {
        "mode": "stage",
        "selected": len(selected),
        "staged": len(prepared),
        "staging_root": str(staging),
        "documents": documents,
    }


def apply_staged_jobs(
    *,
    root: Path,
    stage_batch: Path,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically cut over a validated batch after rechecking review markers."""
    store = JobStore(root)
    manifest_path = stage_batch / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "reprocess_stage_v1":
        raise ValueError("unsupported or incomplete reprocess stage")
    prepared: list[PreparedJob] = []
    for document in manifest.get("documents", []):
        job_id = str(document["job_id"])
        job_dir = store.job_dir(job_id)
        stage_dir = stage_batch / job_id
        marker = _marker_from_payload(document["review_marker"])
        if _review_marker(job_dir) != marker:
            raise ValueError("review changed after staging; no results were applied")
        old_result = json.loads((job_dir / "result.json").read_text())
        new_result = json.loads((stage_dir / "result.json").read_text())
        migrated_review = json.loads((stage_dir / "review.json").read_text())
        _validate_result(
            job_dir / "source.pdf",
            old_result,
            new_result,
            stage_dir / "artifacts",
        )
        prepared.append(
            PreparedJob(
                job_id=job_id,
                stage_dir=stage_dir,
                old_result=old_result,
                new_result=new_result,
                migrated_review=migrated_review,
                review_marker=marker,
            )
        )
    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    backups = (backup_root or store.jobs_root / ".reprocess-backups") / timestamp
    backups.mkdir(parents=True, mode=0o700)
    commit_marker = backups / ".committed.json"
    documents: list[dict[str, Any]] = []
    completed: list[tuple[PreparedJob, Path]] = []
    with ExitStack() as locks:
        for item in sorted(prepared, key=lambda candidate: candidate.job_id):
            locks.enter_context(store.job_lock(item.job_id, exclusive=True))
        for item in prepared:
            if _review_marker(store.job_dir(item.job_id)) != item.review_marker:
                raise ValueError("review changed after staging; no results were applied")
        try:
            for item in prepared:
                backup = _cutover_job(
                    store=store,
                    prepared=item,
                    backup_root=backups,
                    commit_marker=commit_marker,
                )
                completed.append((item, backup))
                documents.append(
                    {
                        "job_id": item.job_id,
                        "source_name": item.old_result.get("source_name"),
                        "rows_before": len(item.old_result.get("rows", [])),
                        "rows_after": len(item.new_result.get("rows", [])),
                        "backup": str(backup),
                    }
                )
            _atomic_json(
                commit_marker,
                {
                    "version": "reprocess_batch_commit_v1",
                    "committed_at": utc_now(),
                    "job_ids": [item.job_id for item in prepared],
                },
            )
        except BaseException:
            for item, backup in reversed(completed):
                _rollback_cutover_job(
                    store=store,
                    prepared=item,
                    backup_dir=backup,
                )
            raise
        for item in prepared:
            (store.job_dir(item.job_id) / ".cutover.json").unlink(missing_ok=True)
    return {
        "reprocessed": len(prepared),
        "documents": documents,
        "staging_root": str(stage_batch),
        "backup_root": str(backups),
    }


def rollback_jobs(*, root: Path, backup_batch: Path) -> dict[str, Any]:
    store = JobStore(root)
    backup_jobs = sorted(path for path in backup_batch.iterdir() if path.is_dir())
    for backup in backup_jobs:
        if not (backup / "result.json").is_file() or not (backup / "artifacts").is_dir():
            raise ValueError(f"backup is incomplete: {backup}")
        store.read(backup.name)
    timestamp = re.sub(r"[^0-9]", "", utc_now())[:14]
    displaced_root = root / "reprocess-rollback-current" / timestamp
    displaced_root.mkdir(parents=True, mode=0o700)
    restored: list[dict[str, str]] = []
    for backup in backup_jobs:
        job_id = backup.name
        job_dir = store.job_dir(job_id)
        displaced = displaced_root / job_id
        displaced.mkdir(mode=0o700)
        (job_dir / "artifacts").replace(displaced / "artifacts")
        (job_dir / "result.json").replace(displaced / "result.json")
        (backup / "artifacts").replace(job_dir / "artifacts")
        (backup / "result.json").replace(job_dir / "result.json")
        _atomic_json(job_dir / "state.json", json.loads((backup / "state.json").read_text()))
        if (backup / "review.json").is_file():
            _atomic_json(
                job_dir / "review.json",
                json.loads((backup / "review.json").read_text()),
            )
        elif (job_dir / "review.json").is_file():
            (job_dir / "review.json").unlink()
        restored.append(
            {
                "job_id": job_id,
                "displaced_result": str(displaced),
            }
        )
    return {
        "restored": len(restored),
        "documents": restored,
        "displaced_root": str(displaced_root),
    }


@app.callback()
def run(
    root: Annotated[Path, typer.Option(file_okay=False, resolve_path=True)],
    job_id: Annotated[list[str] | None, typer.Option("--job-id")] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Stage, validate, and replace completed results."),
    ] = False,
    stage_only: Annotated[
        bool,
        typer.Option("--stage-only", help="Stage and validate results without cutover."),
    ] = False,
    apply_staged: Annotated[
        Path | None,
        typer.Option(
            "--apply-staged",
            exists=True,
            file_okay=False,
            help="Cut over a previously validated staging batch.",
        ),
    ] = None,
    stage_root: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    backup_root: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    rollback_from: Annotated[
        Path | None,
        typer.Option(
            "--rollback-from",
            exists=True,
            file_okay=False,
            help="Restore every job from a previous backup batch.",
        ),
    ] = None,
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
) -> None:
    if rollback_from is not None:
        typer.echo(
            json.dumps(
                rollback_jobs(root=root, backup_batch=rollback_from),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if apply_staged is not None:
        if apply or stage_only or job_id:
            raise typer.BadParameter(
                "--apply-staged cannot be combined with job selection or staging"
            )
        typer.echo(
            json.dumps(
                apply_staged_jobs(
                    root=root,
                    stage_batch=apply_staged,
                    backup_root=backup_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if stage_only:
        if apply:
            raise typer.BadParameter("--stage-only and --apply are mutually exclusive")
        typer.echo(
            json.dumps(
                stage_reprocess_jobs(
                    root=root,
                    job_ids=job_id,
                    stage_root=stage_root,
                    vl_url=vl_url,
                    paddle_device=paddle_device,
                    vl_device=vl_device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = reprocess_jobs(
        root=root,
        job_ids=job_id,
        apply=apply,
        stage_root=stage_root,
        backup_root=backup_root,
        vl_url=vl_url,
        paddle_device=paddle_device,
        vl_device=vl_device,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
