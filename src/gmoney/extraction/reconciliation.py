"""Prove a bill against its own printed arithmetic, or flag exactly why it cannot be proved.

The gate is a pure function of an extraction result (optionally with reviewer-projected
rows).  It never repairs values; it only compares extracted rows with totals the bill
itself prints:

* C1 grand total: detail + refund rows (refunds negative) equal the printed gross bill total.
* C2 section sub-totals: every printed sub-total/total row in the source tables equals a
  contiguous run of canonical rows ending immediately above it.
* C3 summary: category roll-ups, when printed alongside granular rows, sum to the C1 total.
* C4 row arithmetic: quantity x rate - discount = amount.  Reported only; never blocks.

A difference strictly below one rupee passes as ``rounded`` when the printed amount is a
whole rupee.  ``unprovable`` (no printed bill total) is treated like ``flagged``.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from gmoney.extraction.total_labels import (
    TOTAL_PREFIXES,
    is_total_label,
    normalized_label,
)
from gmoney.extraction.typed_values import parse_decimal

RECONCILIATION_VERSION = "reconciliation_v1"
GRANULAR_ROLES = {"detail", "refund"}
LEDGER_ROLES = GRANULAR_ROLES | {"category_rollup"}
DOCUMENT_TOTAL_KINDS = {"bill_total", "gross_total"}
ONE_RUPEE = Decimal("1.00")
# Printed totals that are not sums of preceding charges.  They are reported for the
# reviewer but never required to reconcile.
SETTLEMENT_WORDS = (
    "advance",
    "balance",
    "cgst",
    "claim",
    "company",
    "concession",
    "copay",
    "deposit",
    "discount",
    "due",
    "gst",
    "paid",
    "payable",
    "payer",
    "receiv",
    "recelv",
    "round",
    "settlement",
    "sgst",
    "tax",
    "tpa",
)
STRUCTURED_FIELDS = {"service_date_raw", "request_no", "service_code", "hsn_code"}


def _decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _text(value: Decimal | None) -> str | None:
    return None if value is None else format(value.quantize(Decimal("0.01")), "f")


def match_outcome(expected: Decimal, actual: Decimal) -> str:
    """Exact to the paisa, or ``rounded`` for a sub-rupee gap against a whole-rupee total."""
    difference = abs(actual - expected)
    if difference < Decimal("0.005"):
        return "pass"
    if difference < ONE_RUPEE and expected == expected.to_integral_value():
        return "rounded"
    return "fail"


def signed_amount(row: dict[str, Any]) -> Decimal | None:
    amount = _decimal(row.get("net_amount"))
    if amount is None:
        return None
    return -abs(amount) if row.get("role") == "refund" else amount


def _active(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("review_disposition") != "rejected"]


def _check(
    check_id: str,
    kind: str,
    *,
    expected: Decimal | None,
    actual: Decimal | None,
    outcome: str,
    label: str | None = None,
    page: int | None = None,
    section: str | None = None,
    blocking: bool = True,
    evidence: Any = None,
) -> dict[str, Any]:
    difference = actual - expected if actual is not None and expected is not None else None
    return {
        "id": check_id,
        "kind": kind,
        "page": page,
        "section": section,
        "label": label,
        "expected": _text(expected),
        "actual": _text(actual),
        "difference": _text(difference),
        "outcome": outcome,
        "blocking": blocking,
        "evidence": evidence,
    }


def _printed_bill_totals(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Document-scope gross/bill totals, the selected primary total first."""
    primary = result.get("document_total")
    others = result.get("document_totals")
    candidates = [primary, *(others if isinstance(others, list) else [])]
    printed: list[dict[str, Any]] = []
    seen: set[Decimal] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        amount = _decimal(item.get("amount"))
        kind = str(item.get("kind") or "bill_total")
        scope = str(item.get("scope") or "document")
        if amount is None or kind not in DOCUMENT_TOTAL_KINDS or scope != "document":
            continue
        if amount in seen:
            continue
        seen.add(amount)
        printed.append(
            {
                "amount": amount,
                "label": item.get("label"),
                "kind": kind,
                "page_number": item.get("page_number"),
                "evidence": item.get("evidence"),
            }
        )
    return printed


def _reported_totals(result: dict[str, Any]) -> list[dict[str, Any]]:
    others = result.get("document_totals")
    reported = []
    for item in others if isinstance(others, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "bill_total")
        scope = str(item.get("scope") or "document")
        if kind in DOCUMENT_TOTAL_KINDS and scope == "document":
            continue
        amount = _decimal(item.get("amount"))
        reported.append(
            {
                "label": item.get("label"),
                "kind": kind,
                "scope": scope,
                "amount": _text(amount),
                "page": item.get("page_number"),
            }
        )
    return reported


def _grand_total_check(
    rows: list[dict[str, Any]], printed: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, Decimal | None]:
    granular = [row for row in rows if row.get("role") in GRANULAR_ROLES]
    rollups = [row for row in rows if row.get("role") == "category_rollup"]
    basis = granular or rollups
    missing = [str(row.get("id")) for row in basis if signed_amount(row) is None]
    total = sum((value for row in basis if (value := signed_amount(row)) is not None), Decimal(0))
    if not printed:
        return None, total
    primary = printed[0]
    if missing:
        return (
            _check(
                "C1",
                "grand_total",
                expected=primary["amount"],
                actual=total,
                outcome="fail",
                label=primary["label"],
                page=primary["page_number"],
                evidence={"missing_amount_row_ids": missing},
            ),
            total,
        )
    if not granular:
        # Package bills can print the package roll-up both in a summary and again
        # above its unpriced inclusions; one roll-up equal to the bill total proves it.
        for candidate in printed:
            if any(_decimal(row.get("net_amount")) == candidate["amount"] for row in rollups):
                return (
                    _check(
                        "C1",
                        "grand_total",
                        expected=candidate["amount"],
                        actual=candidate["amount"],
                        outcome="pass",
                        label=candidate["label"],
                        page=candidate["page_number"],
                        evidence={"basis": "package_rollup"},
                    ),
                    candidate["amount"],
                )
    for candidate in printed:
        outcome = match_outcome(candidate["amount"], total)
        if outcome != "fail":
            return (
                _check(
                    "C1",
                    "grand_total",
                    expected=candidate["amount"],
                    actual=total,
                    outcome=outcome,
                    label=candidate["label"],
                    page=candidate["page_number"],
                    evidence={"basis": "granular_rows" if granular else "category_rollups"},
                ),
                total,
            )
    return (
        _check(
            "C1",
            "grand_total",
            expected=primary["amount"],
            actual=total,
            outcome="fail",
            label=primary["label"],
            page=primary["page_number"],
            evidence={
                "basis": "granular_rows" if granular else "category_rollups",
                "other_printed_totals": [_text(item["amount"]) for item in printed[1:]],
            },
        ),
        total,
    )


def _summary_check(rows: list[dict[str, Any]], printed: list[dict[str, Any]]) -> dict | None:
    granular = [row for row in rows if row.get("role") in GRANULAR_ROLES]
    rollups = [row for row in rows if row.get("role") == "category_rollup"]
    if not granular or not rollups or not printed:
        return None
    # A summary printed twice (front page and again above components) is one summary.
    distinct: dict[tuple[str, Decimal | None], dict[str, Any]] = {}
    for row in rollups:
        distinct.setdefault(
            (normalized_label(row.get("description")), _decimal(row.get("net_amount"))), row
        )
    missing = [str(row.get("id")) for row in distinct.values() if signed_amount(row) is None]
    total = sum(
        (value for row in distinct.values() if (value := signed_amount(row)) is not None),
        Decimal(0),
    )
    expected = printed[0]["amount"]
    outcome = "fail" if missing else match_outcome(expected, total)
    if outcome == "fail" and not missing:
        for candidate in printed[1:]:
            if match_outcome(candidate["amount"], total) != "fail":
                expected = candidate["amount"]
                outcome = match_outcome(expected, total)
                break
    return _check(
        "C3",
        "summary",
        expected=expected,
        actual=total,
        outcome=outcome,
        label="category roll-ups",
        evidence={"rollup_row_ids": [str(row.get("id")) for row in distinct.values()]}
        | ({"missing_amount_row_ids": missing} if missing else {}),
    )


def _is_settlement_label(label: str) -> bool:
    return any(word in label for word in SETTLEMENT_WORDS)


def _is_checkable_total_label(label: str) -> bool:
    words = set(label.split())
    return bool(words & {"total", "totals", "subtotal"}) or is_total_label(label)


def _row_label_and_targets(
    table: dict[str, Any], source_row: dict[str, Any]
) -> tuple[str, list[Decimal], list[Decimal]]:
    fields = {
        str(column.get("id")): column.get("canonical_field") for column in table.get("columns", [])
    }
    labels: list[str] = []
    net_values: list[Decimal] = []
    all_values: list[Decimal] = []
    for cell in source_row.get("cells", []):
        raw = str(cell.get("raw_value") or "").strip()
        if not raw:
            continue
        field = fields.get(str(cell.get("column_id")))
        value = parse_decimal(raw)
        if value is None:
            if field not in STRUCTURED_FIELDS or is_total_label(normalized_label(raw)):
                labels.append(raw)
            continue
        if field == "quantity":
            continue
        all_values.append(value)
        if field == "net_amount":
            net_values.append(value)
    return normalized_label(" ".join(labels)), net_values, all_values


def _section_checks(
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    printed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row.get("id")): row for row in rows}
    document_amounts = {item["amount"] for item in printed}
    tables = [table for table in result.get("source_tables") or [] if isinstance(table, dict)]
    ordered = sorted(
        enumerate(tables), key=lambda item: (int(item[1].get("page_number") or 0), item[0])
    )
    stream: list[dict[str, Any]] = []
    boundary_index = 0
    checks: list[dict[str, Any]] = []
    reported: list[dict[str, Any]] = []
    for _, table in ordered:
        page = int(table.get("page_number") or 0) or None
        source_rows = sorted(table.get("rows") or [], key=lambda row: int(row.get("order") or 0))
        for source_row in source_rows:
            canonical_id = source_row.get("canonical_row_id")
            if canonical_id is not None:
                canonical = by_id.get(str(canonical_id))
                if canonical is not None and canonical.get("role") in LEDGER_ROLES:
                    stream.append(canonical)
                continue
            label, net_values, all_values = _row_label_and_targets(table, source_row)
            if not all_values or not label:
                continue
            if _is_settlement_label(label):
                reported.append(
                    {
                        "label": label,
                        "amount": _text(net_values[0] if net_values else all_values[-1]),
                        "page": page,
                        "kind": "printed_settlement_row",
                    }
                )
                continue
            if not _is_checkable_total_label(label):
                continue
            targets = net_values or all_values
            check_id = f"C2:p{page}:{table.get('table_id')}:{source_row.get('id')}"
            outcome, expected, actual, run = _match_contiguous_run(stream, targets)
            section_rows = stream[boundary_index:]
            section = next(
                (str(row["section"]) for row in reversed(section_rows) if row.get("section")),
                None,
            )
            if outcome == "fail" and (
                label.startswith(TOTAL_PREFIXES)
                or any(value in document_amounts for value in targets)
            ):
                # The printed bill total is proved (or flagged) by C1.
                boundary_index = len(stream)
                continue
            if outcome == "fail":
                expected = targets[-1]
                actual = sum(
                    (value for row in section_rows if (value := signed_amount(row)) is not None),
                    Decimal(0),
                )
            checks.append(
                _check(
                    check_id,
                    "section_total",
                    expected=expected,
                    actual=actual,
                    outcome=outcome,
                    label=label,
                    page=page,
                    section=section,
                    evidence={
                        "source_row_id": source_row.get("id"),
                        "table_id": table.get("table_id"),
                        "printed_values": [_text(value) for value in all_values],
                        "run_row_ids": [str(row.get("id")) for row in run],
                    },
                )
            )
            boundary_index = len(stream)
    return checks, reported


def _match_contiguous_run(
    stream: list[dict[str, Any]], targets: list[Decimal]
) -> tuple[str, Decimal | None, Decimal | None, list[dict[str, Any]]]:
    """Search back from the row above a printed total for a run whose sum it equals."""
    best: tuple[str, Decimal, Decimal, list[dict[str, Any]]] | None = None
    total = Decimal(0)
    all_refunds = True
    for start in range(len(stream) - 1, -1, -1):
        row = stream[start]
        value = signed_amount(row)
        if value is None:
            continue
        total += value
        all_refunds = all_refunds and row.get("role") == "refund"
        for target in targets:
            for candidate in (total, -total) if all_refunds else (total,):
                outcome = match_outcome(target, candidate)
                if outcome == "pass":
                    return outcome, target, candidate, stream[start:]
                if outcome == "rounded" and best is None:
                    best = (outcome, target, candidate, stream[start:])
    if best is not None:
        return best
    return "fail", None, None, []


def _row_arithmetic_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for row in rows:
        if row.get("role") not in GRANULAR_ROLES:
            continue
        quantity = _decimal(row.get("quantity"))
        rate = _decimal(row.get("unit_price"))
        amount = _decimal(row.get("net_amount"))
        if quantity is None or rate is None or amount is None:
            continue
        discount = _decimal(row.get("discount")) or Decimal(0)
        expected = quantity * rate - abs(discount)
        gross = _decimal(row.get("gross_amount"))
        outcome = match_outcome(expected, abs(amount))
        if outcome == "fail" and gross is not None:
            outcome = match_outcome(quantity * rate, abs(gross))
        checks.append(
            _check(
                f"C4:{row.get('id')}",
                "row_arithmetic",
                expected=expected,
                actual=abs(amount),
                outcome=outcome,
                label=row.get("description"),
                page=row.get("page_number"),
                section=row.get("section"),
                blocking=False,
            )
        )
    return checks


def _reason(check: dict[str, Any]) -> str:
    where = " ".join(
        part
        for part in (
            f"page {check['page']}" if check.get("page") else "",
            f"section '{check['section']}'" if check.get("section") else "",
        )
        if part
    )
    evidence = check.get("evidence") or {}
    if isinstance(evidence, dict) and evidence.get("missing_amount_row_ids"):
        detail = f"{len(evidence['missing_amount_row_ids'])} row(s) have no amount"
    else:
        detail = (
            f"rows sum to {check['actual']} but the bill prints {check['expected']} "
            f"(difference {check['difference']})"
        )
    label = f" '{check['label']}'" if check.get("label") else ""
    return f"{check['id']} {check['kind']}{label}{' ' + where if where else ''}: {detail}"


def reconcile(result: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a verified/flagged/unprovable report for an extraction result."""
    active = _active(rows if rows is not None else result.get("rows") or [])
    printed = _printed_bill_totals(result)
    grand_total, rows_total = _grand_total_check(active, printed)
    section_checks, reported = _section_checks(result, active, printed)
    checks = [
        *([grand_total] if grand_total else []),
        *section_checks,
        *([summary] if (summary := _summary_check(active, printed)) else []),
        *_row_arithmetic_checks(active),
    ]
    failed = [check for check in checks if check["blocking"] and check["outcome"] == "fail"]
    reasons = [_reason(check) for check in failed]
    if failed:
        status = "flagged"
    elif grand_total is None:
        status = "unprovable"
        reasons.append("C1 grand_total: no printed gross bill total was found")
    else:
        status = "verified"
    return {
        "reconciliation_version": RECONCILIATION_VERSION,
        "status": status,
        "rows_total": _text(rows_total),
        "checks": checks,
        "reasons": reasons,
        "reported": [*_reported_totals(result), *reported],
    }


def gate_enforced(table_reader: str | None, gate: str | None = None) -> bool:
    """Whether an unverified reconciliation blocks completion and approval."""
    if gate is None:
        from gmoney.settings import get_settings

        gate = get_settings().reconciliation_gate
    if gate == "enforce":
        return True
    if gate == "report":
        return False
    return (table_reader or "heuristic") != "heuristic"


def recorded_or_computed(result: dict[str, Any]) -> dict[str, Any]:
    """The reconciliation stored at extraction time, or one computed on read."""
    recorded = result.get("reconciliation")
    if isinstance(recorded, dict) and recorded.get("status"):
        return recorded
    return {
        **reconcile(result),
        "enforced": gate_enforced(result.get("table_reader")),
        "computed_on_read": True,
    }


def is_enforced(result: dict[str, Any]) -> bool:
    recorded = result.get("reconciliation")
    if isinstance(recorded, dict) and isinstance(recorded.get("enforced"), bool):
        return recorded["enforced"]
    return gate_enforced(result.get("table_reader"))
