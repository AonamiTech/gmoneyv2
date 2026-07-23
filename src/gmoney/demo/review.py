from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from gmoney.demo.store import JobStore, JobTransactionError, utc_now

EDITABLE_TEXT_FIELDS = {
    "description",
    "section",
    "service_date_raw",
    "service_date_iso",
    "request_no",
    "service_code",
    "hsn_code",
}
EDITABLE_DECIMAL_FIELDS = {
    "quantity",
    "unit_price",
    "gross_amount",
    "discount",
    "net_amount",
}
EDITABLE_ENUM_FIELDS = {"role", "review_disposition"}
EDITABLE_FIELDS = EDITABLE_TEXT_FIELDS | EDITABLE_DECIMAL_FIELDS | EDITABLE_ENUM_FIELDS
ITEM_TOTAL_ROLES = {"detail", "refund", "category_rollup"}
RAW_FIELD = {
    "quantity": "quantity_raw",
    "unit_price": "unit_price_raw",
    "gross_amount": "gross_amount_raw",
    "discount": "discount_raw",
    "net_amount": "net_amount_raw",
}
ALLOWED_ROLES = {"detail", "informational", "category_rollup", "refund"}
ALLOWED_DISPOSITIONS = {"accepted", "pending", "rejected", "unreadable"}
EXPORT_FIELDS = (
    "hospital_name",
    "id",
    "page_number",
    "role",
    "review_disposition",
    "section",
    "description",
    "service_date_iso",
    "request_no",
    "service_code",
    "hsn_code",
    "quantity",
    "unit_price",
    "gross_amount",
    "discount",
    "net_amount",
)


class ReviewValidationError(ValueError):
    pass


def load_result(store: JobStore, job_id: str) -> dict[str, Any]:
    try:
        return store.read_result(job_id)
    except JobTransactionError as error:
        raise ReviewValidationError("Extraction result is unavailable") from error


def public_page_assets(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: asset[key]
            for key in ("page_number", "artifact_sha256", "width", "height")
        }
        for asset in result.get("page_assets", [])
    ]


def normalize_changes(changes: dict[str, Any]) -> dict[str, Any]:
    unsupported = set(changes) - EDITABLE_FIELDS
    if unsupported:
        raise ReviewValidationError(f"Unsupported fields: {', '.join(sorted(unsupported))}")
    normalized: dict[str, Any] = {}
    for field, value in changes.items():
        if field in EDITABLE_TEXT_FIELDS:
            if value is None or not str(value).strip():
                normalized[field] = None
            else:
                limit = 500 if field == "description" else 200
                normalized[field] = str(value).strip()[:limit]
        elif field in EDITABLE_DECIMAL_FIELDS:
            if value is None or str(value).strip() == "":
                normalized[field] = None
                normalized[RAW_FIELD[field]] = None
                continue
            candidate = str(value).strip()
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", candidate):
                raise ReviewValidationError(f"{field} is not a valid decimal")
            try:
                parsed = Decimal(candidate)
            except InvalidOperation as error:
                raise ReviewValidationError(f"{field} is not a valid decimal") from error
            if not parsed.is_finite() or abs(parsed) >= Decimal("1000000000000000"):
                raise ReviewValidationError(f"{field} is outside the supported range")
            rendered = format(parsed, "f")
            normalized[field] = rendered
            normalized[RAW_FIELD[field]] = rendered
        elif field == "role":
            if value not in ALLOWED_ROLES:
                raise ReviewValidationError("role is not editable to that value")
            normalized[field] = value
        elif field == "review_disposition":
            if value not in ALLOWED_DISPOSITIONS:
                raise ReviewValidationError("review_disposition is invalid")
            normalized[field] = value
    if not normalized:
        raise ReviewValidationError("At least one editable field is required")
    return normalized


def _review_metadata(
    source: str,
    *,
    modified: bool,
    reason: str | None,
    machine_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "modified": modified,
        "reason": reason,
        "machine_values": machine_values or {},
    }


def project_rows(result: dict[str, Any], review: dict[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, machine in enumerate(result.get("rows", [])):
        row = json.loads(json.dumps(machine))
        row.setdefault("page_number", 1)
        row.setdefault("row_order", index)
        row.setdefault("review_disposition", "pending")
        override = review.get("row_overrides", {}).get(str(row["id"]))
        if override:
            changes = override.get("changes", {})
            row.update(changes)
            flags = list(row.get("validation_flags", []))
            if "reviewer_corrected" not in flags:
                flags.append("reviewer_corrected")
            row["validation_flags"] = flags
            routes = list(row.get("source_routes", []))
            if "reviewer" not in routes:
                routes.append("reviewer")
            row["source_routes"] = routes
            row["review"] = _review_metadata(
                "machine",
                modified=True,
                reason=override.get("reason"),
                machine_values={key: machine.get(key) for key in changes},
            )
        else:
            row["review"] = _review_metadata("machine", modified=False, reason=None)
        projected.append(row)
    for added in review.get("added_rows", {}).values():
        row = json.loads(json.dumps(added))
        row["review"] = _review_metadata(
            "reviewer", modified=True, reason=row.pop("review_reason", None)
        )
        projected.append(row)
    projected.sort(key=lambda row: (int(row["page_number"]), int(row["row_order"]), row["id"]))
    return projected


def project_hospital(result: dict[str, Any], review: dict[str, Any]) -> dict[str, Any] | None:
    machine = result.get("hospital")
    override = review.get("document_overrides", {}).get("hospital")
    if override:
        evidence = override["evidence"]
        return {
            "name": override["name"],
            "source": "reviewer",
            "machine_name": machine.get("name") if machine else None,
            "confidence": machine.get("confidence") if machine else None,
            "page_number": evidence["page_number"],
            "evidence": evidence,
            "reason": override.get("reason"),
        }
    if not machine:
        return None
    return {
        **machine,
        "source": "machine",
        "machine_name": machine.get("name"),
        "reason": None,
    }


def totals_summary(
    result: dict[str, Any],
    review: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    projected = rows if rows is not None else project_rows(result, review)
    included = [
        row
        for row in projected
        if row.get("review_disposition") != "rejected"
        and row.get("role") in ITEM_TOTAL_ROLES
    ]
    item_total = Decimal("0")
    missing_amounts = 0
    for row in included:
        value = row.get("net_amount")
        try:
            parsed = Decimal(str(value)) if value is not None else None
        except InvalidOperation:
            parsed = None
        if parsed is None or not parsed.is_finite():
            missing_amounts += 1
        else:
            item_total += parsed

    machine_total = result.get("document_total")
    machine_totals = result.get("document_totals")
    if not isinstance(machine_totals, list):
        machine_totals = [machine_total] if isinstance(machine_total, dict) else []
    bill_total = None
    bill_amount: Decimal | None = None
    if isinstance(machine_total, dict):
        try:
            parsed = Decimal(str(machine_total.get("amount")))
        except InvalidOperation:
            parsed = Decimal("NaN")
        if parsed.is_finite():
            bill_amount = parsed
            bill_total = {
                key: machine_total.get(key)
                for key in (
                    "amount",
                    "label",
                    "kind",
                    "scope",
                    "page_number",
                    "evidence",
                )
            }

    printed_totals: list[dict[str, Any]] = []
    comparable_amounts: set[Decimal] = set()
    for machine_item in machine_totals:
        if not isinstance(machine_item, dict):
            continue
        try:
            amount = Decimal(str(machine_item.get("amount")))
        except (InvalidOperation, TypeError):
            continue
        if not amount.is_finite():
            continue
        kind = str(machine_item.get("kind") or "bill_total")
        scope = str(machine_item.get("scope") or "document")
        printed_totals.append(
            {
                **{
                    key: machine_item.get(key)
                    for key in ("amount", "label", "kind", "scope", "page_number", "evidence")
                },
                "kind": kind,
                "scope": scope,
                "is_primary": bool(
                    isinstance(machine_total, dict)
                    and machine_item.get("evidence") == machine_total.get("evidence")
                    and machine_item.get("amount") == machine_total.get("amount")
                    and machine_item.get("label") == machine_total.get("label")
                ),
            }
        )
        if scope == "document" and kind == "bill_total":
            comparable_amounts.add(amount)

    difference: Decimal | None = None
    if missing_amounts:
        comparison = "items_partial"
    elif bill_amount is None:
        comparison = "bill_total_missing"
    elif len(comparable_amounts) > 1:
        comparison = "multiple_printed_totals"
    else:
        difference = item_total - bill_amount
        comparison = "match" if abs(difference) <= Decimal("0.01") else "mismatch"
    return {
        "items_total": format(item_total, "f"),
        "bill_total": bill_total,
        "printed_totals": printed_totals,
        "difference": format(difference, "f") if difference is not None else None,
        "comparison": comparison,
        "missing_item_amounts": missing_amounts,
    }


def structural_issues(result: dict[str, Any], review: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    overrides = review.get("issue_overrides", {})
    for diagnostic in result.get("diagnostics", []):
        pending = [
            attempt
            for attempt in diagnostic.get("recovery_attempts", [])
            if attempt.get("stage") == "review" and attempt.get("status") == "pending"
        ]
        if not pending:
            continue
        route = diagnostic.get("phase3_route", {})
        reasons = route.get("reasons") or [attempt.get("reason") for attempt in pending]
        reasons = [str(reason) for reason in reasons if reason]
        page_number = int(diagnostic.get("page_number") or 1)
        table_id = str(diagnostic.get("table_id") or f"page-{page_number}")
        issue_key = f"{page_number}:{table_id}:{','.join(reasons)}"
        issue_id = hashlib.sha256(issue_key.encode()).hexdigest()[:20]
        override = overrides.get(issue_id, {})
        issues.append(
            {
                "id": issue_id,
                "page_number": page_number,
                "table_id": table_id,
                "table_type": diagnostic.get("table_type") or "unknown",
                "reason_codes": reasons or ["manual_review"],
                "status": override.get("status", "open"),
                "resolution_reason": override.get("reason"),
                "updated_at": override.get("updated_at"),
            }
        )
    field_flags = {
        "missing_labeled_quantity",
        "missing_labeled_unit_price",
        "line_arithmetic_mismatch",
    }
    flagged_tables: dict[tuple[int, str], set[str]] = {}
    for row in result.get("rows", []):
        reasons = field_flags & set(row.get("validation_flags", []))
        if reasons:
            key = (int(row.get("page_number") or 1), str(row.get("table_id") or "unknown"))
            flagged_tables.setdefault(key, set()).update(reasons)
    for (page_number, table_id), reason_set in sorted(flagged_tables.items()):
        reasons = sorted(reason_set)
        issue_key = f"{page_number}:{table_id}:{','.join(reasons)}"
        issue_id = hashlib.sha256(issue_key.encode()).hexdigest()[:20]
        if any(issue["id"] == issue_id for issue in issues):
            continue
        override = overrides.get(issue_id, {})
        issues.append(
            {
                "id": issue_id,
                "page_number": page_number,
                "table_id": table_id,
                "table_type": "field_validation",
                "reason_codes": reasons,
                "status": override.get("status", "open"),
                "resolution_reason": override.get("reason"),
                "updated_at": override.get("updated_at"),
            }
        )
    return issues


def review_summary(result: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    rows = project_rows(result, review)
    issues = structural_issues(result, review)
    active = [row for row in rows if row.get("review_disposition") != "rejected"]
    return {
        "revision": review["revision"],
        "rows_total": len(rows),
        "rows_active": len(active),
        "rows_modified": sum(bool(row["review"]["modified"]) for row in rows),
        "rows_pending": sum(row.get("review_disposition") != "accepted" for row in active),
        "issues_open": sum(issue["status"] == "open" for issue in issues),
        "issues": issues,
        "hospital": project_hospital(result, review),
        "approval": review.get("approval"),
    }


def _polygon_area(points: list[dict[str, float]]) -> float:
    return abs(
        sum(
            first["x"] * second["y"] - second["x"] * first["y"]
            for first, second in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2


def evidence_for_page(
    result: dict[str, Any], page_number: int, points: list[dict[str, float]]
) -> dict[str, Any]:
    asset = next(
        (item for item in result.get("page_assets", []) if item["page_number"] == page_number),
        None,
    )
    if asset is None:
        raise ReviewValidationError("Evidence page does not exist")
    if len(points) < 3 or not all(
        math.isfinite(float(point[axis]))
        and 0 <= float(point[axis]) <= float(asset["width" if axis == "x" else "height"])
        for point in points
        for axis in ("x", "y")
    ):
        raise ReviewValidationError("Evidence polygon is outside the page")
    canonical = [{"x": float(point["x"]), "y": float(point["y"])} for point in points]
    if _polygon_area(canonical) < 4:
        raise ReviewValidationError("Evidence polygon is too small")
    return {
        "page_number": page_number,
        "table_id": None,
        "polygon": {"points": canonical},
        "artifact_sha256": asset["artifact_sha256"],
        "token_ids": [],
    }


def reviewer_row(
    result: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    values: dict[str, Any],
    page_number: int,
    points: list[dict[str, float]],
    reason: str,
) -> dict[str, Any]:
    normalized = normalize_changes(values)
    if not normalized.get("description") or normalized.get("net_amount") is None:
        raise ReviewValidationError("Added rows require description and net_amount")
    evidence = evidence_for_page(result, page_number, points)
    row_order = max((int(row["row_order"]) for row in existing_rows), default=-1) + 1
    row: dict[str, Any] = {
        "id": str(uuid4()),
        "contract_version": "canonical_row_reviewer_v1",
        "created_at": utc_now(),
        "document_id": result["document_id"],
        "page_number": page_number,
        "table_id": None,
        "page_type": None,
        "table_type": None,
        "row_order": row_order,
        "role": "detail",
        "review_disposition": "accepted",
        "section": None,
        "description": None,
        "service_date_raw": None,
        "service_date_iso": None,
        "request_no": None,
        "service_code": None,
        "hsn_code": None,
        "quantity_raw": None,
        "quantity": None,
        "unit_price_raw": None,
        "unit_price": None,
        "gross_amount_raw": None,
        "gross_amount": None,
        "discount_raw": None,
        "discount": None,
        "net_amount_raw": None,
        "net_amount": None,
        "evidence": [evidence],
        "field_evidence": {"description": [evidence], "amount": [evidence]},
        "candidate_ids": [],
        "source_routes": ["reviewer"],
        "validation_flags": ["reviewer_added"],
        "review_reason": reason,
    }
    row.update(normalized)
    return row


def approval_blockers(
    store: JobStore, job_id: str, result: dict[str, Any], review: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    rows = project_rows(result, review)
    active = [row for row in rows if row.get("review_disposition") != "rejected"]
    billable = [row for row in active if row.get("role") in ITEM_TOTAL_ROLES]
    informational = [row for row in active if row.get("role") == "informational"]
    if not billable:
        blockers.append("no_active_rows")
    if any(row.get("review_disposition") != "accepted" for row in active):
        blockers.append("pending_rows")
    if any(not row.get("description") or row.get("net_amount") is None for row in billable):
        blockers.append("missing_required_values")
    if any(
        not {"description", "amount"}.issubset(row.get("field_evidence", {}))
        for row in billable
    ):
        blockers.append("missing_field_evidence")
    if any(
        not row.get("description")
        or "description" not in row.get("field_evidence", {})
        or not {
            "service_date",
            "request_no",
            "service_code",
            "hsn_code",
        }.intersection(row.get("field_evidence", {}))
        for row in informational
    ):
        blockers.append("missing_informational_evidence")
    if any(issue["status"] == "open" for issue in structural_issues(result, review)):
        blockers.append("open_structural_issues")
    assets = result.get("page_assets", [])
    if len(assets) != int(result.get("pages") or 0):
        blockers.append("incomplete_page_inventory")
    artifact_root = (store.job_dir(job_id) / "artifacts").resolve()
    if any(
        not (artifact_root / asset.get("relative_path", "")).resolve().is_file()
        for asset in assets
    ):
        blockers.append("missing_page_artifacts")
    return sorted(set(blockers))


def export_payload(result: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    return {
        "export_version": "demo_review_export_v1",
        "document_id": result["document_id"],
        "source_name": result.get("source_name"),
        "hospital": project_hospital(result, review),
        "pages": result["pages"],
        "review_revision": review["revision"],
        "approval": review["approval"],
        "rows": [
            row
            for row in project_rows(result, review)
            if row.get("review_disposition") == "accepted"
        ],
    }


def _safe_spreadsheet_text(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def export_csv(result: dict[str, Any], review: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    hospital = project_hospital(result, review)
    for row in export_payload(result, review)["rows"]:
        writer.writerow(
            {
                field: _safe_spreadsheet_text(
                    hospital.get("name")
                    if hospital and field == "hospital_name"
                    else row.get(field)
                )
                if field not in EDITABLE_DECIMAL_FIELDS
                else row.get(field)
                for field in EXPORT_FIELDS
            }
        )
    return output.getvalue()


def create_evidence_bundle(
    store: JobStore,
    job_id: str,
    result: dict[str, Any],
    review: dict[str, Any],
) -> Path:
    exports = store.job_dir(job_id) / "exports"
    exports.mkdir(exist_ok=True)
    target = exports / f"evidence-r{review['revision']}.zip"
    if target.is_file():
        return target
    reviewed = (
        json.dumps(export_payload(result, review), indent=2, sort_keys=True) + "\n"
    ).encode()
    machine = (
        json.dumps(
            {
                "output_version": result.get("output_version"),
                "document_id": result.get("document_id"),
                "source_sha256": result.get("source_sha256"),
                "source_name": result.get("source_name"),
                "hospital": result.get("hospital"),
                "pages": result.get("pages"),
                "page_assets": public_page_assets(result),
                "rows": result.get("rows", []),
                "provider_usage": result.get("provider_usage", {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    files: list[tuple[str, bytes]] = [
        ("reviewed-output.json", reviewed),
        ("machine-output.json", machine),
    ]
    artifact_root = (store.job_dir(job_id) / "artifacts").resolve()
    for asset in result.get("page_assets", []):
        page_path = (artifact_root / asset["relative_path"]).resolve()
        if artifact_root not in page_path.parents or not page_path.is_file():
            raise ReviewValidationError("A page artifact is unavailable")
        files.append(
            (f"pages/page-{asset['page_number']}.png", page_path.read_bytes())
        )
    manifest_lines: list[str] = []
    for name, content in files:
        digest = hashlib.sha256(content).hexdigest()
        manifest_lines.append(f"{digest}  {name}")
    files.append(("manifest.sha256", ("\n".join(manifest_lines) + "\n").encode()))
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, content in files:
                archive.writestr(name, content)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
