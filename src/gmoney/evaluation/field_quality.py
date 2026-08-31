from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from gmoney.contracts.gold import GoldAnnotation

FIELD_METRIC_VERSION = "field_quality_v1"
CANONICAL_FIELDS = (
    "section",
    "description",
    "service_date",
    "request_no",
    "service_code",
    "hsn_code",
    "quantity",
    "unit_price",
    "gross_amount",
    "discount",
    "net_amount",
)
TEXT_FIELDS = {"section", "description"}
CODE_FIELDS = {"request_no", "service_code", "hsn_code"}
MONEY_FIELDS = {"unit_price", "gross_amount", "discount", "net_amount"}


@dataclass(frozen=True)
class FieldMetric:
    gold_values: int
    predicted_values: int
    compared_cells: int
    correct_values: int
    precision: float
    recall: float
    accuracy: float
    observed: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentFields:
    document_id: str
    layout_family_id: str
    gold: GoldAnnotation
    actual: dict[str, Any]


def normalize_header(value: object) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return normalized or None


def normalize_value(field: str, value: object) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field in MONEY_FIELDS or field == "quantity":
        try:
            decimal = Decimal(str(value).replace(",", "").strip())
        except (InvalidOperation, ValueError):
            return normalize_header(value)
        if field in MONEY_FIELDS:
            return format(decimal.quantize(Decimal("0.01")), "f")
        return format(decimal.normalize(), "f")
    if field == "service_date":
        text = str(value).strip()
        for pattern in (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%d %b %Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(text, pattern).date().isoformat()
            except ValueError:
                continue
        return normalize_header(text)
    if field in CODE_FIELDS:
        normalized = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
        return normalized or None
    if field in TEXT_FIELDS or field.startswith("source:") or field.startswith("header:"):
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value).casefold())
        return re.sub(r"\s+", " ", normalized).strip() or None
    return normalize_header(value)


def _metric(
    gold: dict[tuple[object, ...], str | None],
    predicted: dict[tuple[object, ...], str | None],
    *,
    threshold: float,
) -> FieldMetric:
    keys = set(gold) | set(predicted)
    gold_values = sum(gold.get(key) is not None for key in keys)
    predicted_values = sum(predicted.get(key) is not None for key in keys)
    compared = sum(gold.get(key) is not None or predicted.get(key) is not None for key in keys)
    correct = sum(
        gold.get(key) is not None and gold.get(key) == predicted.get(key) for key in keys
    )
    precision = (
        correct / predicted_values if predicted_values else (1.0 if not gold_values else 0.0)
    )
    recall = correct / gold_values if gold_values else 1.0
    accuracy = correct / compared if compared else 1.0
    observed = compared > 0
    return FieldMetric(
        gold_values=gold_values,
        predicted_values=predicted_values,
        compared_cells=compared,
        correct_values=correct,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        observed=observed,
        passed=not observed or min(precision, recall, accuracy) > threshold,
    )


def _canonical_raw(row: dict[str, Any], field: str, *, gold: bool) -> object:
    if field == "service_date":
        return (
            row.get("service_date")
            if gold
            else row.get("service_date_iso") or row.get("service_date_raw")
        )
    if field == "unit_price":
        return row.get("rate") if gold else row.get("unit_price")
    if field == "net_amount":
        return row.get("amount") if gold else row.get("net_amount")
    return row.get(field)


def _canonical_maps(
    documents: Iterable[DocumentFields],
) -> tuple[
    dict[str, dict[tuple[object, ...], str | None]],
    dict[str, dict[tuple[object, ...], str | None]],
]:
    gold_maps = {field: {} for field in CANONICAL_FIELDS}
    actual_maps = {field: {} for field in CANONICAL_FIELDS}
    for document in documents:
        gold_rows = [row.model_dump(mode="json") for row in document.gold.rows]
        actual_rows = [row for row in document.actual.get("rows", []) if isinstance(row, dict)]
        for is_gold, rows, output in (
            (True, gold_rows, gold_maps),
            (False, actual_rows, actual_maps),
        ):
            occurrences: dict[tuple[int, str, int], int] = {}
            for index, row in enumerate(rows):
                page = int(row.get("page_number") or 0)
                table = str(row.get("table_id") or "")
                order = int(row.get("row_order") if row.get("row_order") is not None else index)
                identity = (page, table, order)
                duplicate = occurrences.get(identity, 0)
                occurrences[identity] = duplicate + 1
                key = (document.document_id, page, table, order, duplicate)
                for field in CANONICAL_FIELDS:
                    output[field][key] = normalize_value(
                        field,
                        _canonical_raw(row, field, gold=is_gold),
                    )
    return gold_maps, actual_maps


def _table_positions(tables: Iterable[object]) -> list[dict[str, Any]]:
    selected = [table for table in tables if isinstance(table, dict)]
    return sorted(
        selected,
        key=lambda table: (
            int(table.get("page_number") or 0),
            str(table.get("table_id") or table.get("id") or ""),
        ),
    )


def _source_maps(
    documents: Iterable[DocumentFields],
) -> tuple[
    dict[str, dict[str, dict[tuple[object, ...], str | None]]],
    dict[str, int],
]:
    maps: dict[str, dict[str, dict[tuple[object, ...], str | None]]] = {
        "gold_headers": {},
        "actual_headers": {},
        "gold_cells": {},
        "actual_cells": {},
    }
    excluded: dict[str, int] = {}
    for document in documents:
        gold_tables = _table_positions(
            table.model_dump(mode="json") for table in document.gold.source_tables
        )
        actual_tables = _table_positions(document.actual.get("source_tables", []))
        table_count = max(len(gold_tables), len(actual_tables))
        for table_position in range(table_count):
            gold_table = gold_tables[table_position] if table_position < len(gold_tables) else {}
            actual_table = (
                actual_tables[table_position] if table_position < len(actual_tables) else {}
            )
            page = int(gold_table.get("page_number") or actual_table.get("page_number") or 0)
            gold_columns = sorted(
                gold_table.get("columns", []),
                key=lambda column: int(column.get("order") or 0),
            )
            actual_columns = sorted(
                actual_table.get("columns", []),
                key=lambda column: int(column.get("order") or 0),
            )
            gold_rows = {int(row.get("order") or 0): row for row in gold_table.get("rows", [])}
            actual_rows = {int(row.get("order") or 0): row for row in actual_table.get("rows", [])}
            column_count = max(len(gold_columns), len(actual_columns))
            for column_order in range(column_count):
                gold_column = gold_columns[column_order] if column_order < len(gold_columns) else {}
                actual_column = (
                    actual_columns[column_order]
                    if column_order < len(actual_columns)
                    else {}
                )
                family = normalize_header(
                    gold_column.get("canonical_field")
                    or gold_column.get("label")
                    or actual_column.get("canonical_field")
                    or actual_column.get("label")
                    or f"column_{column_order + 1}"
                )
                assert family is not None
                for name in maps:
                    maps[name].setdefault(family, {})
                header_key = (document.document_id, page, table_position, column_order)
                maps["gold_headers"][family][header_key] = normalize_value(
                    f"header:{family}", gold_column.get("label")
                )
                maps["actual_headers"][family][header_key] = normalize_value(
                    f"header:{family}", actual_column.get("label")
                )
                gold_column_id = gold_column.get("id")
                actual_column_id = actual_column.get("id")
                row_orders = set(gold_rows) | set(actual_rows)
                for row_order in row_orders:
                    gold_cell = next(
                        (
                            cell
                            for cell in gold_rows.get(row_order, {}).get("cells", [])
                            if cell.get("column_id") == gold_column_id
                        ),
                        {},
                    )
                    if gold_cell and not gold_cell.get("readable", True):
                        excluded[family] = excluded.get(family, 0) + 1
                        continue
                    actual_cell = next(
                        (
                            cell
                            for cell in actual_rows.get(row_order, {}).get("cells", [])
                            if cell.get("column_id") == actual_column_id
                        ),
                        {},
                    )
                    cell_key = (
                        document.document_id,
                        page,
                        table_position,
                        row_order,
                        column_order,
                    )
                    maps["gold_cells"][family][cell_key] = normalize_value(
                        f"source:{family}", gold_cell.get("raw_value")
                    )
                    maps["actual_cells"][family][cell_key] = normalize_value(
                        f"source:{family}", actual_cell.get("raw_value")
                    )
    return maps, excluded


def evaluate_field_quality(
    documents: Iterable[DocumentFields],
    *,
    threshold: float = 0.95,
) -> dict[str, Any]:
    selected = tuple(documents)
    gold, actual = _canonical_maps(selected)
    canonical = {
        field: _metric(gold[field], actual[field], threshold=threshold).to_dict()
        for field in CANONICAL_FIELDS
    }
    source_maps, excluded = _source_maps(selected)
    families = sorted(set(source_maps["gold_headers"]) | set(source_maps["actual_headers"]))
    source = {
        family: {
            "header": _metric(
                source_maps["gold_headers"].get(family, {}),
                source_maps["actual_headers"].get(family, {}),
                threshold=threshold,
            ).to_dict(),
            "cells": _metric(
                source_maps["gold_cells"].get(family, {}),
                source_maps["actual_cells"].get(family, {}),
                threshold=threshold,
            ).to_dict(),
            "excluded_unreadable_cells": excluded.get(family, 0),
        }
        for family in families
    }
    failures = [
        f"canonical:{field}"
        for field, metric in canonical.items()
        if metric["observed"] and not metric["passed"]
    ]
    for family, metrics in source.items():
        for kind in ("header", "cells"):
            metric = metrics[kind]
            if metric["observed"] and not metric["passed"]:
                failures.append(f"source:{family}:{kind}")
    return {
        "metric_version": FIELD_METRIC_VERSION,
        "threshold_exclusive": threshold,
        "documents": len(selected),
        "canonical_fields": canonical,
        "source_column_families": source,
        "blocking_reasons": failures,
        "passed": not failures,
    }


def evaluate_with_layout_slices(
    documents: Iterable[DocumentFields],
    *,
    threshold: float = 0.95,
) -> dict[str, Any]:
    selected = tuple(documents)
    aggregate = evaluate_field_quality(selected, threshold=threshold)
    layouts = {
        layout: evaluate_field_quality(
            (document for document in selected if document.layout_family_id == layout),
            threshold=threshold,
        )
        for layout in sorted({document.layout_family_id for document in selected})
    }
    blocking = list(aggregate["blocking_reasons"])
    blocking.extend(
        f"layout:{layout}:{reason}"
        for layout, report in layouts.items()
        for reason in report["blocking_reasons"]
    )
    return {
        "metric_version": FIELD_METRIC_VERSION,
        "threshold_exclusive": threshold,
        "aggregate": aggregate,
        "layout_families": layouts,
        "blocking_reasons": blocking,
        "passed": not blocking,
    }
