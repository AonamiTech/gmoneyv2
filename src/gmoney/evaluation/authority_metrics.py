"""Deterministic structural metrics for the authoritative table corpus.

The evaluator deliberately has no dependency on a particular producer.  The authority
contract is a Pydantic contract, but accepting its ``model_dump`` representation as well
is useful for sealed JSON manifests and makes it possible to score an old extraction
envelope without adapting the envelope into a second, lossy schema.

Matching is structural and one-to-one:

* tables are matched on page and source-space geometry;
* columns are matched on canonical role, geometry, and order;
* rows are matched on table, geometry, order, role, and continuation; and
* cells are compared only after their row and column have been aligned.

Values never participate in the primary structural match.  This is important: a value
which is in a neighbouring column is a column error, not a correctly matched value with
a lucky amount match.  All tie breaks use source order and then a canonical digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Any

EVALUATOR_VERSION = "authority_metrics_v3"

# These are the frozen Table Magic accuracy floors.  Keep the names stable because a
# promotion manifest refers to these keys rather than to display labels.
TABLE_MAGIC_FLOORS: dict[str, float] = {
    "critical_numeric_cell_precision": 0.98,
    "critical_numeric_cell_recall": 0.95,
    "line_item_row_recall": 0.95,
    "correct_column_assignment": 0.98,
    "header_schema_accuracy": 0.97,
    "grand_total_exact_accuracy": 0.99,
    "auto_accepted_document_correctness": 0.99,
}
TABLE_MAGIC_THRESHOLDS = TABLE_MAGIC_FLOORS

# A ratio with no observations is not evidence of correctness.  Keep the
# denominator mapping separate from the display metrics because the latter retain
# their historical empty-ratio spelling for diagnostic compatibility; floor
# evaluation itself is fail-closed when one of these denominators is zero.
_FLOOR_DENOMINATORS: dict[str, str] = {
    "critical_numeric_cell_precision": "critical_actual",
    "critical_numeric_cell_recall": "critical_gold",
    "line_item_row_recall": "line_item_gold_rows",
    "correct_column_assignment": "gold_cells",
    "header_schema_accuracy": "gold_columns",
    "grand_total_exact_accuracy": "grand_total_gold",
    "auto_accepted_document_correctness": "auto_accepted_documents",
}


class _EmptyAuthorityInputError(ValueError, TypeError):
    """Fail closed for callers that probe an optional evaluator API."""

_MONEY_ROLES = {
    "amount",
    "net_amount",
    "gross_amount",
    "total",
    "document_total",
    "bill_total",
    "gross_total",
    "payable_total",
    "settlement_total",
    "rate",
    "unit_price",
    "mrp",
    "discount",
    "tax",
    "cgst",
    "sgst",
    "igst",
    "subtotal",
    "price",
}
_NUMERIC_ROLES = _MONEY_ROLES | {
    "quantity",
    "qty",
    "units",
    "count",
    "percentage",
    "percent",
}
_TOTAL_WORDS = ("total", "subtotal", "payable", "settlement", "balance")
_NON_LINE_ROLES = {
    "header",
    "column_header",
    "section_header",
    "category_rollup",
    "section_total",
    "document_total",
    "total",
    "payment",
    "deposit",
    "refund",
    "metadata",
    "footer_noise",
}


def _json_ready(value: Any) -> Any:
    """Convert contract objects into stable JSON-compatible primitives."""

    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Decimal):
        # Decimal's spelling is intentionally retained for identity hashes.  Numeric
        # comparison uses ``_normalise_value`` below.
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_ready(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_ready(model_dump(mode="python"))
        except TypeError:
            return _json_ready(model_dump())
    if isinstance(value, (str, bytes, bytearray)):
        return value.decode() if isinstance(value, (bytes, bytearray)) else value
    if isinstance(value, Sequence):
        return [_json_ready(item) for item in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_ready(vars(value))
    return value


def identity_sha256(value: Any) -> str:
    """Return the canonical SHA-256 used in evaluator reports and baselines."""

    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _report_identity(value: Any, *, gold: bool) -> str:
    """Prefer a contract-supplied sealed digest, falling back to canonical content."""

    names = (
        ("gold_sha256", "gold_manifest_sha256")
        if gold
        else ("result_sha256", "actual_sha256", "prediction_sha256")
    )
    for name in names:
        candidate = _get(value, name, default=None)
        if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
            return candidate
    return identity_sha256(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            result = model_dump(mode="python")
        except TypeError:
            result = model_dump()
        if isinstance(result, Mapping):
            return result
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = dataclasses.asdict(value)
        if isinstance(result, Mapping):
            return result
    result = getattr(value, "__dict__", None)
    return result if isinstance(result, Mapping) else {}


_MISSING = object()


def _get(value: Any, *names: str, default: Any = None) -> Any:
    """Read aliases from mappings or contract objects without truthiness surprises."""

    source = _mapping(value)
    for name in names:
        if name in source:
            return source[name]
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            candidate = _MISSING
        if candidate is not _MISSING:
            return candidate
    return default


def _scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _text(value: Any, default: str = "") -> str:
    value = _scalar(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalise_key(value: Any) -> str:
    value = _text(value).strip().casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def _items(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    return []


def _number(value: Any, default: int | float | None = None) -> int | float | None:
    value = _scalar(value)
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number.is_integer():
        return int(number)
    return number


def _index(value: Any, default: int) -> int:
    number = _number(value)
    return int(number) if number is not None else default


def _page(value: Any, default: int | None = None) -> int | None:
    raw = _get(value, "page_number", "page", "page_no", "source_page_number", default=default)
    return _index(raw, default if default is not None else 0) or default


def _point_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        x = _number(value.get("x", value.get("left")))
        y = _number(value.get("y", value.get("top")))
        if x is not None and y is not None:
            return float(x), float(y)
        return None
    values = _items(value)
    if len(values) >= 2:
        x = _number(values[0])
        y = _number(values[1])
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def _polygon(value: Any) -> tuple[tuple[float, float], ...] | None:
    """Accept Polygon, point lists, rectangle mappings, and bbox tuples."""

    if value is None:
        return None
    # Pydantic Polygon and the planned authority Polygon both expose points.
    points = _get(value, "points", "vertices", "coordinates", default=None)
    if points is not None:
        parsed = tuple(point for item in _items(points) if (point := _point_pair(item)) is not None)
        if len(parsed) >= 3:
            return parsed
    if isinstance(value, Mapping):
        # x/y/width/height and left/top/right/bottom are common serialized forms.
        left = _number(value.get("left", value.get("x", value.get("x1"))))
        top = _number(value.get("top", value.get("y", value.get("y1"))))
        right = _number(value.get("right", value.get("x2")))
        bottom = _number(value.get("bottom", value.get("y2")))
        width = _number(value.get("width", value.get("w")))
        height = _number(value.get("height", value.get("h")))
        if right is None and left is not None and width is not None:
            right = left + width
        if bottom is None and top is not None and height is not None:
            bottom = top + height
        if None not in (left, top, right, bottom):
            return (
                (float(left), float(top)),
                (float(right), float(top)),
                (float(right), float(bottom)),
                (float(left), float(bottom)),
            )
        for key in ("bbox", "box", "rect", "bounds"):
            if key in value:
                return _polygon(value[key])
    values = _items(value)
    if len(values) == 4 and all(_number(item) is not None for item in values):
        left, top, right, bottom = (float(_number(item)) for item in values)
        return ((left, top), (right, top), (right, bottom), (left, bottom))
    if len(values) >= 3:
        parsed = tuple(point for item in values if (point := _point_pair(item)) is not None)
        if len(parsed) >= 3:
            return parsed
    return None


def _bbox(
    polygon: tuple[tuple[float, float], ...] | None,
) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(
    left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None
) -> float:
    if left is None or right is None:
        return 0.0
    left_width = max(0.0, left[2] - left[0])
    left_height = max(0.0, left[3] - left[1])
    right_width = max(0.0, right[2] - right[0])
    right_height = max(0.0, right[3] - right[1])
    left_area = left_width * left_height
    right_area = right_width * right_height
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _centre_distance(
    left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None
) -> float:
    if left is None or right is None:
        return 1.0
    left_centre = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_centre = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    scale = max(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1], 1.0)
    return min(
        1.0, math.hypot(left_centre[0] - right_centre[0], left_centre[1] - right_centre[1]) / scale
    )


def _geometry(value: Any) -> tuple[tuple[float, float], ...] | None:
    raw = _get(
        value,
        "source_polygon",
        "source_page_polygon",
        "page_polygon",
        "polygon",
        "table_polygon",
        "cell_polygon",
        "bounds",
        "bbox",
        "box",
        default=None,
    )
    return _polygon(raw)


def _role(value: Any) -> str:
    return _normalise_key(
        _get(
            value,
            "canonical_role",
            "canonical_field",
            "field_role",
            "row_kind",
            "role",
            "column_role",
            "field",
            "name",
            "label",
            default="",
        )
    )


def _value_type(value: Any, role: str = "") -> str:
    type_name = _normalise_key(_get(value, "value_type", "data_type", "type", "kind", default=""))
    if type_name:
        return type_name
    if role in _MONEY_ROLES or any(word in role for word in ("amount", "price", "total", "rate")):
        return "money"
    if role in _NUMERIC_ROLES:
        return "number"
    if "date" in role:
        return "date"
    return "text"


def _readable(value: Any, default: bool = True) -> bool:
    raw = _get(value, "readable", "is_readable", "legible", default=_MISSING)
    if raw is not _MISSING:
        if isinstance(raw, str):
            return _normalise_key(raw) not in {"false", "no", "unreadable", "illegible", "0"}
        return bool(raw)
    status = _normalise_key(_get(value, "readability", "readability_status", "status", default=""))
    if status:
        return status not in {"unreadable", "illegible", "not_readable", "false"}
    return default


def _unreadable_marker(value: Any) -> bool:
    role = _role(value)
    flags = _get(value, "validation_flags", "flags", "issues", default=())
    flag_names = {_normalise_key(item) for item in _items(flags)}
    return (
        not _readable(value)
        or role in {"unreadable", "illegible"}
        or bool(flag_names & {"unreadable", "illegible", "not_readable"})
    )


def _raw_value(value: Any, *, prefer_normalized: bool = True) -> Any:
    """Read a value while keeping producer normalization out of machine scoring.

    GoldDocumentV2 may carry a reviewer-provided normalized value, so the default
    preserves that contract's historical behavior.  Machine actuals call this
    function with ``prefer_normalized=False`` and are normalized from visible raw
    output by this evaluator instead.
    """

    names = (
        (
            "normalized_value",
            "normalised_value",
            "typed_value",
            "value",
            "raw_value",
            "text",
            "content",
            "amount",
            "amount_raw",
            "number",
        )
        if prefer_normalized
        else (
            "raw_value",
            "value",
            "typed_value",
            "text",
            "content",
            "amount",
            "amount_raw",
            "number",
        )
    )
    for name in names:
        candidate = _get(value, name, default=_MISSING)
        if candidate is not _MISSING and candidate not in (None, ""):
            return candidate
    return None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    text = _text(value).strip()
    if not text:
        return None
    # Currency symbols and thousands separators are presentation, not value.
    text = text.replace("₹", "").replace("INR", "").replace("inr", "")
    text = text.replace(",", "").replace(" ", "")
    text = re.sub(r"[()]", "", text)
    negative = text.startswith("-") or ("(" in _text(value) and ")" in _text(value))
    text = text.lstrip("+-")
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return -parsed if negative else parsed


def _normalise_value(value: Any, value_type: str, role: str = "") -> Any:
    """Normalise money to INR paise precision and typed values deterministically."""

    if value is None:
        return None
    type_name = _normalise_key(value_type)
    role = _normalise_key(role)
    is_money = type_name in {"money", "currency", "inr", "amount"} or role in _MONEY_ROLES
    is_number = is_money or type_name in {"number", "numeric", "decimal", "integer", "float"}
    if is_number:
        parsed = _decimal(value)
        if parsed is not None:
            if is_money:
                return str(parsed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            return str(parsed.normalize())
    if type_name in {"date", "datetime"} or "date" in role:
        text = _text(value).strip()
        try:
            if type_name == "datetime":
                return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
            return date.fromisoformat(text).isoformat()
        except ValueError:
            for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, pattern).date().isoformat()
                except ValueError:
                    continue
            return re.sub(r"\s+", " ", text).casefold()
    text = _text(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold() if text else None


def _has_asserted_value(value: Any, *, prefer_normalized: bool = True) -> bool:
    raw = _raw_value(value, prefer_normalized=prefer_normalized)
    return raw is not None and _text(raw).strip() != ""


def _has_actual_value(value: Any) -> bool:
    return _has_asserted_value(value, prefer_normalized=False)


def _grounded(value: Any) -> bool:
    explicit = _get(value, "grounded", "is_grounded", "evidence_grounded", default=_MISSING)
    if explicit is not _MISSING:
        return bool(explicit)
    evidence = _get(
        value,
        "evidence",
        "field_evidence",
        "source_evidence",
        "evidence_refs",
        "grounding",
        "token_ids",
        "ocr_token_ids",
        default=None,
    )
    if isinstance(evidence, Mapping):
        evidence = list(evidence.values())
    if evidence is None:
        return False
    if isinstance(evidence, (str, bytes)):
        return bool(evidence.strip())
    for item in _items(evidence):
        token_ids = _get(item, "token_ids", "ocr_token_ids", default=None)
        if token_ids is not None and _items(token_ids):
            return True
        if _get(item, "token_id", default=None):
            return True
        if isinstance(item, str) and item.strip():
            return True
    return False


def _continuation(value: Any) -> str:
    explicit = _get(
        value,
        "continuation_group",
        "continuation_group_id",
        "continuation_id",
        "continuation_of",
        "continues_row_id",
        "continuation_key",
        default=None,
    )
    if explicit is not None:
        return _normalise_key(explicit)
    if bool(_get(value, "is_continuation", "continues_previous", default=False)):
        return "continuation"
    return "continuation" if _role(value) == "continuation" else ""


def _identifier(value: Any, level: str, index: int) -> str:
    names = {
        "table": ("table_id", "id", "table_key", "logical_table_id", "table_anchor"),
        "row": ("row_id", "id", "canonical_row_id", "row_anchor", "row_key"),
        "column": ("column_id", "id", "column_key", "key"),
        "cell": ("cell_id", "id", "key"),
    }
    raw = _get(value, *names.get(level, ("id",)), default=None)
    return _text(raw) if raw not in (None, "") else f"{level}-{index:06d}"


def _sequence_from_document(document: Any, names: Sequence[str]) -> list[Any]:
    for name in names:
        value = _get(document, name, default=_MISSING)
        if value is not _MISSING and value is not None:
            if isinstance(value, Mapping):
                return [value]
            return _items(value)
    return []


def _source_tables_with_v6_geometry(document: Any, tables: Sequence[Any]) -> list[Any]:
    """Join V6 source tables to their source-page crop polygons.

    ``SourceTableV2`` deliberately contains the reconstructed table rather than
    duplicating artifact geometry.  Its source-space polygon lives on the
    corresponding ``CanonicalTableArtifact``.  Authority matching is geometric,
    so retain that normalized V6 relationship when evaluating the public
    envelope instead of silently falling back to producer list order.

    A missing or ambiguous relationship is left untouched.  Valid V6 envelopes
    guarantee uniqueness; this defensive behavior keeps the generic evaluator
    compatible with partial diagnostic dictionaries without inventing geometry.
    """

    artifacts = _sequence_from_document(document, ("canonical_table_artifacts",))
    if not artifacts:
        return list(tables)

    geometry_by_key: dict[tuple[int, str], list[Any]] = {}
    for artifact in artifacts:
        logical_table_id = _text(_get(artifact, "logical_table_id", default=""))
        page_number = _page(artifact)
        geometry = _get(artifact, "crop_polygon_in_source_raw", default=None)
        if not logical_table_id or page_number is None or _polygon(geometry) is None:
            continue
        geometry_by_key.setdefault((page_number, logical_table_id), []).append(geometry)

    enriched: list[Any] = []
    for table in tables:
        if _geometry(table) is not None:
            enriched.append(table)
            continue
        table_id = _text(_get(table, "table_id", "logical_table_id", default=""))
        page_number = _page(table)
        candidates = geometry_by_key.get((page_number, table_id), [])
        if len(candidates) != 1:
            enriched.append(table)
            continue
        table_payload = dict(_mapping(table))
        table_payload["source_polygon"] = candidates[0]
        enriched.append(table_payload)
    return enriched


def _tables(document: Any) -> list[Any]:
    direct = _sequence_from_document(document, ("tables",))
    if direct:
        return direct
    source_tables = _sequence_from_document(document, ("source_tables",))
    if source_tables:
        return _source_tables_with_v6_geometry(document, source_tables)
    direct = _sequence_from_document(
        document, ("logical_tables", "table_records", "table_regions")
    )
    if direct:
        return direct
    pages = _sequence_from_document(document, ("pages", "page_records"))
    gathered: list[Any] = []
    for page in pages:
        page_tables = _sequence_from_document(page, ("tables", "source_tables", "logical_tables"))
        if page_tables:
            gathered.extend(page_tables)
    if gathered:
        return gathered
    if (
        _get(document, "columns", default=_MISSING) is not _MISSING
        or _get(document, "rows", default=_MISSING) is not _MISSING
    ):
        return [document]
    nested = _get(document, "gold", "actual", "prediction", "result", "extraction", default=None)
    if nested is not None and nested is not document:
        return _tables(nested)
    return []


def _rows(table: Any) -> list[Any]:
    direct = _sequence_from_document(table, ("rows", "source_rows", "row_records", "line_items"))
    if direct:
        return direct
    row = _get(table, "row", default=_MISSING)
    return [row] if row is not _MISSING and row is not None else []


def _columns(table: Any, rows: Sequence[Any]) -> list[Any]:
    direct = _sequence_from_document(table, ("columns", "source_columns", "column_records"))
    if direct:
        return direct
    seen: dict[str, Any] = {}
    for row in rows:
        for index, cell in enumerate(_cells(row)):
            key = _cell_column_key(cell, index)
            if key not in seen:
                seen[key] = {"id": key, "label": key, "canonical_field": key, "order": index}
    return list(seen.values())


def _cells(row: Any) -> list[Any]:
    direct = _sequence_from_document(
        row, ("cells", "source_cells", "cell_records", "values", "fields")
    )
    if direct:
        return direct
    # CanonicalRow-shaped inputs store fields as named attributes rather than a grid.
    known = (
        "section",
        "service_date",
        "service_date_iso",
        "request_no",
        "description",
        "service_code",
        "hsn_code",
        "quantity",
        "rate",
        "unit_price",
        "gross_amount",
        "discount",
        "amount",
        "net_amount",
        "tax",
        "total",
    )
    evidence_map = _get(row, "field_evidence", "evidence_by_field", default={})
    evidence_map = evidence_map if isinstance(evidence_map, Mapping) else {}
    generated: list[dict[str, Any]] = []
    for name in known:
        raw = _get(row, name, default=_MISSING)
        if raw is not _MISSING:
            generated.append(
                {
                    "column_id": name,
                    "canonical_field": name,
                    "value": raw,
                    "evidence": evidence_map.get(name, ()),
                }
            )
    return generated


def _cell_column_key(cell: Any, index: int) -> str:
    raw = _get(
        cell,
        "column_id",
        "column_key",
        "column",
        "canonical_field",
        "field",
        "key",
        "name",
        default=None,
    )
    if isinstance(raw, Mapping):
        raw = _get(raw, "id", "key", "name", "canonical_field", default=None)
    return _text(raw) if raw not in (None, "") else f"column-{index:06d}"


def _cell_role(cell: Any, table: Any, index: int) -> str:
    direct = _role(cell)
    if direct:
        return direct
    key = _cell_column_key(cell, index)
    rows = _rows(table)
    columns = _columns(table, rows)
    for column_index, column in enumerate(columns):
        column_key = _text(
            _get(column, "id", "column_id", "key", default=f"column-{column_index:06d}")
        )
        if (
            column_key == key
            or _index(_get(column, "order", "column_order", default=column_index), column_index)
            == index
        ):
            return _role(column)
    return ""


def _all_document_rows(document: Any, table_list: Sequence[Any]) -> list[Any]:
    # A source table grid is authoritative when present.  Top-level canonical rows are
    # used only when no table has rows, or when tables are merely region proposals.
    table_rows = [row for table in table_list for row in _rows(table)]
    if table_rows:
        return table_rows
    return _sequence_from_document(document, ("rows", "canonical_rows", "line_items"))


def _totals(document: Any, table_list: Sequence[Any]) -> list[Any]:
    direct = _sequence_from_document(document, ("totals", "document_totals", "total_records"))
    if direct:
        return direct
    one = _get(document, "document_total", "total", "grand_total", default=_MISSING)
    if one is not _MISSING and one is not None:
        return [one]
    derived: list[Any] = []
    for table in table_list:
        for row in _rows(table):
            role = _role(row)
            if role in {"document_total", "section_total", "total", "grand_total"}:
                derived.append(row)
    if not derived:
        nested = _get(
            document, "gold", "actual", "prediction", "result", "extraction", default=None
        )
        if nested is not None and nested is not document:
            return _totals(nested, _tables(nested))
    return derived


def _doc_key(document: Any, default: str) -> str:
    value = _get(
        document,
        "document_sha256",
        "source_sha256",
        "document_id",
        "unit_key",
        "bill_file",
        "id",
        default=None,
    )
    if value not in (None, ""):
        return _text(value)
    nested = _get(document, "gold", "actual", "prediction", "result", "extraction", default=None)
    if nested is not None and nested is not document:
        return _doc_key(nested, default)
    return default


def _score_tuple(score: float) -> tuple[float, ...]:
    # Rounding prevents insignificant binary noise from changing a tie break across
    # Python versions while retaining enough precision for geometry.
    return (round(score, 12),)


def _assignment(
    left: Sequence[Any],
    right: Sequence[Any],
    score_fn: Any,
    eligible_fn: Any,
) -> tuple[dict[int, tuple[int, float]], set[int], set[int]]:
    """Deterministic maximum-cardinality matching with weighted tie breaks.

    A small dynamic program gives maximum total score after cardinality for the normal
    table/column cases.  Large row grids use deterministic augmenting paths, preserving
    maximum cardinality without exponential memory.
    """

    candidates: dict[int, list[tuple[int, float]]] = {}
    for li, left_item in enumerate(left):
        options = []
        for ri, right_item in enumerate(right):
            score = score_fn(left_item, right_item, li, ri)
            if score is not None and eligible_fn(left_item, right_item, score):
                options.append((ri, float(score)))
        candidates[li] = sorted(options, key=lambda item: (-round(item[1], 12), item[0]))

    if not left or not right:
        return {}, set(range(len(left))), set(range(len(right)))

    if len(left) <= 12 and len(right) <= 20:
        memo: dict[tuple[int, int], tuple[int, float, tuple[tuple[int, int], ...]]] = {}

        def solve(index: int, used: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
            key = (index, used)
            if key in memo:
                return memo[key]
            if index >= len(left):
                result = (0, 0.0, ())
                memo[key] = result
                return result
            best = solve(index + 1, used)
            for right_index, score in candidates[index]:
                bit = 1 << right_index
                if used & bit:
                    continue
                count, total, pairs = solve(index + 1, used | bit)
                candidate = (count + 1, total + score, ((index, right_index),) + pairs)
                # Prefer cardinality, then score, then lexicographically earliest
                # right-side assignments.  ``pairs`` already follows left order.
                if (candidate[0], round(candidate[1], 12), tuple(-p[1] for p in candidate[2])) > (
                    best[0],
                    round(best[1], 12),
                    tuple(-p[1] for p in best[2]),
                ):
                    best = candidate
            memo[key] = best
            return best

        pairs = solve(0, 0)[2]
        matched = {
            left_index: (right_index, candidates[left_index][0][1])
            for left_index, right_index in pairs
        }
        # Recover exact selected score rather than the first candidate score.
        matched = {
            li: (ri, next(score for candidate_ri, score in candidates[li] if candidate_ri == ri))
            for li, ri in pairs
        }
    else:
        right_match: dict[int, int] = {}
        selected_score: dict[int, float] = {}

        def visit(left_index: int, seen: set[int]) -> bool:
            for right_index, score in candidates[left_index]:
                if right_index in seen:
                    continue
                seen.add(right_index)
                old_left = right_match.get(right_index)
                if old_left is None or visit(old_left, seen):
                    right_match[right_index] = left_index
                    selected_score[left_index] = score
                    return True
            return False

        for left_index in range(len(left)):
            visit(left_index, set())
        matched = {
            left_index: (right_index, selected_score[left_index])
            for right_index, left_index in sorted(right_match.items())
        }

    matched_left = set(matched)
    matched_right = {right_index for right_index, _ in matched.values()}
    return matched, set(range(len(left))) - matched_left, set(range(len(right))) - matched_right


def _table_score(gold: Any, actual: Any, gi: int, ai: int) -> float | None:
    gp, ap = _page(gold), _page(actual)
    if gp is not None and ap is not None and gp != ap:
        return None
    gold_geometry, actual_geometry = _bbox(_geometry(gold)), _bbox(_geometry(actual))
    has_geometry = gold_geometry is not None and actual_geometry is not None
    overlap = _bbox_iou(gold_geometry, actual_geometry)
    gold_id = _text(_get(gold, "table_id", "id", "logical_table_id", default=""))
    actual_id = _text(_get(actual, "table_id", "id", "logical_table_id", default=""))
    if has_geometry and overlap < 0.01:
        return None
    score = 0.0
    score += 0.45 if gp is not None and ap is not None and gp == ap else 0.20
    if has_geometry:
        score += 0.45 * overlap + 0.10 * (1.0 - _centre_distance(gold_geometry, actual_geometry))
    elif gold_id and actual_id and gold_id == actual_id:
        score += 0.65
    else:
        go = _index(_get(gold, "order", "table_order", "ordinal", default=gi), gi)
        ao = _index(_get(actual, "order", "table_order", "ordinal", default=ai), ai)
        score += 0.55 * max(0.0, 1.0 - min(abs(go - ao), 5) / 5)
    if gold_id and actual_id and gold_id == actual_id:
        score += 0.08
    if _normalise_key(
        _get(gold, "table_type", "table_kind", "type", "kind", default="")
    ) == _normalise_key(_get(actual, "table_type", "table_kind", "type", "kind", default="")):
        score += 0.04
    return score


def _column_score(gold: Any, actual: Any, gi: int, ai: int) -> float | None:
    gold_role, actual_role = _role(gold), _role(actual)
    gp, ap = _bbox(_geometry(gold)), _bbox(_geometry(actual))
    has_geometry = gp is not None and ap is not None
    overlap = _bbox_iou(gp, ap)
    go = _index(_get(gold, "order", "column_order", "ordinal", default=gi), gi)
    ao = _index(_get(actual, "order", "column_order", "ordinal", default=ai), ai)
    role_known = bool(gold_role and actual_role)
    score = 0.0
    if role_known and gold_role == actual_role:
        score += 0.65
    elif role_known:
        score += 0.04  # retain the geometric match so wrong_role is observable
    if has_geometry:
        score += 0.27 * overlap + 0.08 * (1.0 - _centre_distance(gp, ap))
    else:
        score += 0.27 * max(0.0, 1.0 - min(abs(go - ao), 5) / 5)
    if go == ao:
        score += 0.08
    return score if score >= 0.18 or gold_role == actual_role else None


def _row_score(gold: Any, actual: Any, gi: int, ai: int) -> float | None:
    gp, ap = _bbox(_geometry(gold)), _bbox(_geometry(actual))
    has_geometry = gp is not None and ap is not None
    overlap = _bbox_iou(gp, ap)
    gold_role, actual_role = _role(gold), _role(actual)
    gold_cont, actual_cont = _continuation(gold), _continuation(actual)
    go = _index(_get(gold, "order", "row_order", "ordinal", default=gi), gi)
    ao = _index(_get(actual, "order", "row_order", "ordinal", default=ai), ai)
    gold_id = _text(_get(gold, "row_id", "id", "canonical_row_id", "row_anchor", default=""))
    actual_id = _text(_get(actual, "row_id", "id", "canonical_row_id", "row_anchor", default=""))
    score = 0.0
    if gold_id and actual_id and gold_id == actual_id:
        score += 0.45
    if gold_role and actual_role and gold_role == actual_role:
        score += 0.18
    elif gold_role and actual_role:
        score += 0.02
    if gold_cont and actual_cont and gold_cont == actual_cont:
        score += 0.12
    elif gold_cont or actual_cont:
        score += 0.01
    if has_geometry:
        score += 0.35 * overlap + 0.15 * (1.0 - _centre_distance(gp, ap))
        if overlap < 0.005:
            return None
    else:
        score += 0.35 * max(0.0, 1.0 - min(abs(go - ao), 5) / 5)
    if go == ao:
        score += 0.08
    # Rows without any structural cue cannot be safely matched.
    return score if score >= 0.20 else None


def _row_signature(row: Any, *, prefer_normalized: bool = True) -> tuple[Any, ...]:
    values: list[Any] = []
    for index, cell in enumerate(_cells(row)):
        role = _role(cell) or _normalise_key(_cell_column_key(cell, index))
        values.append(
            (
                role,
                _normalise_value(
                    _raw_value(cell, prefer_normalized=prefer_normalized),
                    _value_type(cell, role),
                    role,
                ),
            )
        )
    if values:
        return tuple(values)
    return (_normalise_key(_get(row, "description", "raw_text", default="")),)


def _same_signature(left: Any, right: Any) -> bool:
    left_values = {
        item for item in _row_signature(left, prefer_normalized=True) if item[1] not in (None, "")
    }
    right_values = {
        item for item in _row_signature(right, prefer_normalized=False) if item[1] not in (None, "")
    }
    return bool(left_values and left_values == right_values)


def _row_has_ungrounded_value(row: Any) -> bool:
    return any(_has_actual_value(cell) and not _grounded(cell) for cell in _cells(row))


def _status_priority(issues: Iterable[str]) -> str:
    order = (
        "wrong_table",
        "duplicate",
        "missed",
        "spurious",
        "unreadable_mishandled",
        "wrong_column",
        "wrong_role",
        "wrong_value",
        "ungrounded",
        "matched",
    )
    issue_set = set(issues)
    for name in order:
        if name in issue_set:
            return name
    return "matched"


@dataclass(frozen=True)
class AlignmentOutcome:
    level: str
    status: str
    gold_index: int | None
    actual_index: int | None
    gold_id: str | None
    actual_id: str | None
    score: float = 0.0
    issues: tuple[str, ...] = ()

    @property
    def outcome(self) -> str:
        return self.status

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CellOutcome(AlignmentOutcome):
    row_gold_index: int | None = None
    row_actual_index: int | None = None
    column_gold_index: int | None = None
    column_actual_index: int | None = None
    gold_value: Any = None
    actual_value: Any = None
    readable_gold: bool = True
    readable_actual: bool = True
    grounded_actual: bool = False
    exact: bool = False
    canonical_role: str = ""
    value_type: str = "text"
    gold_critical: bool = False
    actual_critical: bool = False


@dataclass(frozen=True)
class RowOutcome(AlignmentOutcome):
    table_gold_index: int | None = None
    table_actual_index: int | None = None
    structural_match: bool = False
    complete: bool = False


@dataclass(frozen=True)
class AuthorityDocumentReport:
    document_key: str
    gold_identity_sha256: str
    actual_identity_sha256: str
    pair_identity_sha256: str
    metrics: Mapping[str, Any]
    tables: tuple[AlignmentOutcome, ...] = ()
    columns: tuple[AlignmentOutcome, ...] = ()
    rows: tuple[RowOutcome, ...] = ()
    cells: tuple[CellOutcome, ...] = ()
    totals: tuple[CellOutcome, ...] = ()
    floors: Mapping[str, bool] = field(default_factory=dict)
    passed: bool = True

    @property
    def row_outcomes(self) -> tuple[RowOutcome, ...]:
        return self.rows

    @property
    def document_metrics(self) -> Mapping[str, Any]:
        return self.metrics

    @property
    def table_outcomes(self) -> tuple[AlignmentOutcome, ...]:
        return self.tables

    def to_dict(self) -> dict[str, Any]:
        data = {
            "evaluator_version": EVALUATOR_VERSION,
            "document_key": self.document_key,
            "gold_identity_sha256": self.gold_identity_sha256,
            "actual_identity_sha256": self.actual_identity_sha256,
            "pair_identity_sha256": self.pair_identity_sha256,
            "tables": [item.to_dict() for item in self.tables],
            "table_outcomes": [item.to_dict() for item in self.tables],
            "columns": [item.to_dict() for item in self.columns],
            "column_outcomes": [item.to_dict() for item in self.columns],
            "rows": [item.to_dict() for item in self.rows],
            "row_outcomes": [item.to_dict() for item in self.rows],
            "cells": [item.to_dict() for item in self.cells],
            "cell_outcomes": [item.to_dict() for item in self.cells],
            "totals": [item.to_dict() for item in self.totals],
            "metrics": dict(self.metrics),
            "floors": dict(self.floors),
            "passed": self.passed,
        }
        # Flat aliases are convenient for release gates and preserve compatibility with
        # callers that treat a metric report as a mapping.
        data.update(self.metrics)
        return data

    def __getitem__(self, key: str) -> Any:
        if key in self.metrics:
            return self.metrics[key]
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class AuthorityCohortReport:
    documents: tuple[AuthorityDocumentReport, ...]
    metrics: Mapping[str, Any]
    floors: Mapping[str, bool]
    passed: bool
    cohort_identity_sha256: str

    @property
    def document_reports(self) -> tuple[AuthorityDocumentReport, ...]:
        return self.documents

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "evaluator_version": EVALUATOR_VERSION,
            "cohort_identity_sha256": self.cohort_identity_sha256,
            "documents": [document.to_dict() for document in self.documents],
            "document_reports": [document.to_dict() for document in self.documents],
            "metrics": dict(self.metrics),
            "floors": dict(self.floors),
            "passed": self.passed,
        }
        data.update(self.metrics)
        return data

    def __getitem__(self, key: str) -> Any:
        if key in self.metrics:
            return self.metrics[key]
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _outcome(
    level: str,
    status: str,
    gold_index: int | None,
    actual_index: int | None,
    gold: Any = None,
    actual: Any = None,
    score: float = 0.0,
    issues: Iterable[str] = (),
) -> AlignmentOutcome:
    return AlignmentOutcome(
        level=level,
        status=status,
        gold_index=gold_index,
        actual_index=actual_index,
        gold_id=_identifier(gold, level, gold_index or 0) if gold is not None else None,
        actual_id=_identifier(actual, level, actual_index or 0) if actual is not None else None,
        score=round(float(score), 12),
        issues=tuple(sorted(set(issues))),
    )


def _metrics_from_counts(counts: Mapping[str, int | float]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str, *, empty: float = 1.0) -> float:
        denominator_value = float(counts.get(denominator, 0))
        return (
            round(float(counts.get(numerator, 0)) / denominator_value, 12)
            if denominator_value
            else empty
        )

    metrics: dict[str, Any] = dict(counts)
    metrics["precision"] = ratio("exact_cells", "actual_asserted_cells", empty=1.0)
    metrics["recall"] = ratio("exact_cells", "gold_asserted_cells", empty=1.0)
    metrics["f1"] = (
        round(
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"]),
            12,
        )
        if metrics["precision"] + metrics["recall"]
        else 0.0
    )
    for name, matched, gold_count, actual_count in (
        ("table", "matched_tables", "gold_tables", "actual_tables"),
        ("column", "matched_columns", "gold_columns", "actual_columns"),
        ("row", "matched_rows", "gold_rows", "actual_rows"),
        ("cell_structure", "aligned_cells", "gold_cells", "actual_cells"),
        ("total", "matched_totals", "gold_totals", "actual_totals"),
    ):
        metrics[f"{name}_precision"] = ratio(matched, actual_count, empty=1.0)
        metrics[f"{name}_recall"] = ratio(matched, gold_count, empty=1.0)
        precision, recall = metrics[f"{name}_precision"], metrics[f"{name}_recall"]
        metrics[f"{name}_f1"] = (
            round(2 * precision * recall / (precision + recall), 12) if precision + recall else 0.0
        )
    metrics["cell_value_precision"] = ratio("exact_cells", "actual_asserted_cells", empty=1.0)
    metrics["cell_value_recall"] = ratio("exact_cells", "gold_asserted_cells", empty=1.0)
    metrics["readable_cell_precision"] = ratio(
        "exact_readable_cells", "actual_readable_cells", empty=1.0
    )
    metrics["readable_cell_recall"] = ratio(
        "exact_readable_cells", "gold_readable_cells", empty=1.0
    )
    metrics["grounding_precision"] = ratio(
        "grounded_actual_cells", "actual_asserted_cells", empty=1.0
    )
    metrics["grounding_recall"] = ratio("grounded_exact_cells", "gold_readable_cells", empty=1.0)
    metrics["unreadable_recall"] = ratio(
        "correct_unreadable_cells", "gold_unreadable_cells", empty=1.0
    )
    metrics["unreadable_precision"] = ratio(
        "correct_unreadable_cells", "actual_unreadable_cells", empty=1.0
    )
    metrics["readability_accuracy"] = ratio(
        "readability_correct_cells", "readability_labelled_cells", empty=1.0
    )
    metrics["column_assignment_precision"] = ratio("aligned_cells", "actual_cells", empty=1.0)
    metrics["column_assignment_recall"] = ratio("aligned_cells", "gold_cells", empty=1.0)
    metrics["correct_column_assignment"] = ratio(
        "correctly_assigned_cells", "gold_cells", empty=1.0
    )
    metrics["readability_precision"] = metrics["readable_cell_precision"]
    metrics["readability_recall"] = metrics["readable_cell_recall"]
    metrics["grounding_cell_precision"] = metrics["grounding_precision"]
    metrics["grounding_cell_recall"] = metrics["grounding_recall"]
    metrics["totals_exact"] = int(counts.get("exact_totals", 0))
    return metrics


def _floor_results(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Apply frozen floors without treating an absent denominator as success."""

    floors: dict[str, bool] = {}
    for name, threshold in TABLE_MAGIC_FLOORS.items():
        denominator = _FLOOR_DENOMINATORS.get(name)
        denominator_value = metrics.get(denominator, 0) if denominator else 1
        floors[name] = bool(
            isinstance(denominator_value, (int, float))
            and not isinstance(denominator_value, bool)
            and denominator_value > 0
            and metrics.get(name, 0.0) >= threshold
        )
    return floors


def _critical(role: str, value_type: str) -> bool:
    role = _normalise_key(role)
    value_type = _normalise_key(value_type)
    return (
        value_type in {"money", "currency", "number", "numeric", "decimal", "integer"}
        or role in _NUMERIC_ROLES
    )


def _line_item(row: Any) -> bool:
    role = _role(row)
    return role not in _NON_LINE_ROLES and role not in {"unreadable", "unresolved"}


def _evaluate_document(
    gold: Any, actual: Any, *, document_key: str | None = None
) -> AuthorityDocumentReport:
    gold_tables, actual_tables = _tables(gold), _tables(actual)
    gold_key = document_key or _doc_key(gold, "document")
    actual_key = _doc_key(actual, "actual")

    table_matches, missing_tables, extra_tables = _assignment(
        gold_tables, actual_tables, _table_score, lambda _left, _right, score: score >= 0.30
    )
    table_outcomes: list[AlignmentOutcome] = []
    matched_table_actuals: set[int] = set()
    for gi, (ai, score) in sorted(table_matches.items()):
        matched_table_actuals.add(ai)
        table_outcomes.append(
            _outcome("table", "matched", gi, ai, gold_tables[gi], actual_tables[ai], score)
        )
    for gi in sorted(missing_tables):
        table_outcomes.append(_outcome("table", "missed", gi, None, gold_tables[gi]))
    for ai in sorted(extra_tables):
        status = (
            "duplicate"
            if any(
                _table_score(gold_table, actual_tables[ai], gold_index, ai) is not None
                for gold_index, gold_table in enumerate(gold_tables)
            )
            else "spurious"
        )
        table_outcomes.append(_outcome("table", status, None, ai, actual=actual_tables[ai]))

    column_outcomes: list[AlignmentOutcome] = []
    row_outcomes: list[RowOutcome] = []
    cell_outcomes: list[CellOutcome] = []
    counts: dict[str, int | float] = {
        "gold_tables": len(gold_tables),
        "actual_tables": len(actual_tables),
        "matched_tables": len(table_matches),
        "gold_columns": 0,
        "actual_columns": 0,
        "matched_columns": 0,
        "gold_rows": 0,
        "actual_rows": 0,
        "matched_rows": 0,
        "gold_cells": 0,
        "actual_cells": 0,
        "aligned_cells": 0,
        "exact_cells": 0,
        "gold_asserted_cells": 0,
        "actual_asserted_cells": 0,
        "gold_readable_cells": 0,
        "actual_readable_cells": 0,
        "exact_readable_cells": 0,
        "grounded_actual_cells": 0,
        "grounded_exact_cells": 0,
        "gold_unreadable_cells": 0,
        "actual_unreadable_cells": 0,
        "correct_unreadable_cells": 0,
        "readability_correct_cells": 0,
        "readability_labelled_cells": 0,
        "wrong_table": 0,
        "wrong_column": 0,
        "wrong_role": 0,
        "wrong_value": 0,
        "ungrounded": 0,
        "unreadable_mishandled": 0,
        "duplicates": 0,
        "missed_rows": 0,
        "spurious_rows": 0,
        "complete_rows": 0,
        "complete_tables": 0,
    }
    critical_gold = critical_exact = critical_actual = 0
    schema_gold = schema_correct = 0
    line_gold = line_matched = 0

    def append_cell(
        gold_cell: Any | None,
        actual_cell: Any | None,
        *,
        role_hint: str = "",
        gold_role_hint: str = "",
        actual_role_hint: str = "",
        gold_index: int | None,
        actual_index: int | None,
        row_gold_index: int | None,
        row_actual_index: int | None,
        column_gold_index: int | None,
        column_actual_index: int | None,
    ) -> CellOutcome:
        gold_role = (_role(gold_cell) if gold_cell is not None else "") or gold_role_hint
        actual_role = (_role(actual_cell) if actual_cell is not None else "") or actual_role_hint
        role = gold_role or actual_role or role_hint
        value_type = _value_type(gold_cell or actual_cell, role)
        gold_unreadable = gold_cell is not None and _unreadable_marker(gold_cell)
        actual_unreadable = actual_cell is None or _unreadable_marker(actual_cell)
        gold_value = (
            _normalise_value(_raw_value(gold_cell), value_type, role)
            if gold_cell is not None
            else None
        )
        actual_value = (
            _normalise_value(
                _raw_value(actual_cell, prefer_normalized=False),
                _value_type(actual_cell, role),
                role,
            )
            if actual_cell is not None
            else None
        )
        gold_has = gold_cell is not None and _has_asserted_value(gold_cell)
        actual_has = actual_cell is not None and _has_actual_value(actual_cell)
        gold_critical = bool(
            gold_has
            and not gold_unreadable
            and _critical(gold_role, _value_type(gold_cell, gold_role))
        )
        actual_critical = bool(
            actual_has
            and not actual_unreadable
            and _critical(actual_role, _value_type(actual_cell, actual_role))
        )
        exact = (gold_unreadable and actual_unreadable and not actual_has) or (
            not gold_unreadable and gold_has and actual_has and gold_value == actual_value
        )
        grounded = bool(actual_cell is not None and actual_has and _grounded(actual_cell))
        issues: list[str] = []
        if gold_cell is None:
            if actual_has:
                issues.append("spurious")
        elif actual_cell is None:
            if gold_has and not gold_unreadable:
                issues.append("missed")
        else:
            if gold_role and actual_role and gold_role != actual_role:
                issues.append("wrong_role")
            if (gold_unreadable and actual_has) or (
                not gold_unreadable and actual_unreadable and gold_has
            ):
                issues.append("unreadable_mishandled")
            elif gold_has and actual_has and not exact:
                issues.append("wrong_value")
            if actual_has and not grounded:
                issues.append("ungrounded")
        if not issues:
            status = (
                "matched"
                if gold_cell is not None and actual_cell is not None
                else ("spurious" if gold_cell is None else "missed")
            )
        else:
            status = _status_priority(issues)
        return CellOutcome(
            level="cell",
            status=status,
            gold_index=gold_index,
            actual_index=actual_index,
            gold_id=_identifier(gold_cell, "cell", gold_index or 0)
            if gold_cell is not None
            else None,
            actual_id=_identifier(actual_cell, "cell", actual_index or 0)
            if actual_cell is not None
            else None,
            score=1.0 if gold_cell is not None and actual_cell is not None else 0.0,
            issues=tuple(sorted(set(issues))),
            row_gold_index=row_gold_index,
            row_actual_index=row_actual_index,
            column_gold_index=column_gold_index,
            column_actual_index=column_actual_index,
            gold_value=gold_value,
            actual_value=actual_value,
            readable_gold=not gold_unreadable,
            readable_actual=not actual_unreadable,
            grounded_actual=grounded,
            exact=exact,
            canonical_role=role,
            value_type=value_type,
            gold_critical=gold_critical,
            actual_critical=actual_critical,
        )

    for gi, ai_score in sorted(table_matches.items()):
        ai, table_score = ai_score
        gold_table, actual_table = gold_tables[gi], actual_tables[ai]
        gold_rows, actual_rows = _rows(gold_table), _rows(actual_table)
        gold_columns, actual_columns = (
            _columns(gold_table, gold_rows),
            _columns(actual_table, actual_rows),
        )
        counts["gold_columns"] += len(gold_columns)
        counts["actual_columns"] += len(actual_columns)
        column_matches, missing_columns, extra_columns = _assignment(
            gold_columns, actual_columns, _column_score, lambda _left, _right, score: score >= 0.18
        )
        counts["matched_columns"] += len(column_matches)
        table_rows_complete = (
            not missing_columns
            and not extra_columns
            and all(
                not _role(gold_columns[gci])
                or not _role(actual_columns[aci])
                or _role(gold_columns[gci]) == _role(actual_columns[aci])
                for gci, (aci, _column_score_value) in column_matches.items()
            )
        )
        schema_gold += len(gold_columns)
        schema_correct += sum(
            _role(gold_columns[gci]) == _role(actual_columns[aci])
            for gci, (aci, _score) in column_matches.items()
            if _role(gold_columns[gci]) and _role(actual_columns[aci])
        )
        for gci, (aci, score) in sorted(column_matches.items()):
            issues = []
            if (
                _role(gold_columns[gci])
                and _role(actual_columns[aci])
                and _role(gold_columns[gci]) != _role(actual_columns[aci])
            ):
                issues.append("wrong_role")
            if _index(_get(gold_columns[gci], "order", "column_order", default=gci), gci) != _index(
                _get(actual_columns[aci], "order", "column_order", default=aci), aci
            ) and not (
                _role(gold_columns[gci]) and _role(gold_columns[gci]) == _role(actual_columns[aci])
            ):
                issues.append("wrong_column")
            column_outcomes.append(
                _outcome(
                    "column",
                    _status_priority(issues),
                    gci,
                    aci,
                    gold_columns[gci],
                    actual_columns[aci],
                    score,
                    issues,
                )
            )
        for gci in sorted(missing_columns):
            column_outcomes.append(_outcome("column", "missed", gci, None, gold_columns[gci]))
        for aci in sorted(extra_columns):
            column_outcomes.append(
                _outcome("column", "spurious", None, aci, actual=actual_columns[aci])
            )

        counts["gold_rows"] += len(gold_rows)
        counts["actual_rows"] += len(actual_rows)
        row_matches, missing_rows, extra_rows = _assignment(
            gold_rows, actual_rows, _row_score, lambda _left, _right, score: score >= 0.20
        )
        counts["matched_rows"] += len(row_matches)
        for gri, (ari, score) in sorted(row_matches.items()):
            gold_row, actual_row = gold_rows[gri], actual_rows[ari]
            issues: list[str] = []
            if _role(gold_row) and _role(actual_row) and _role(gold_row) != _role(actual_row):
                issues.append("wrong_role")
            if _continuation(gold_row) != _continuation(actual_row) and (
                _continuation(gold_row) or _continuation(actual_row)
            ):
                issues.append("wrong_role")
            row_complete = True
            row_gold_cells = _cells(gold_row)
            row_actual_cells = _cells(actual_row)
            if any(
                _role(gold_columns[gci])
                and _role(actual_columns[aci])
                and _role(gold_columns[gci]) != _role(actual_columns[aci])
                for gci, (aci, _column_score_value) in column_matches.items()
            ):
                issues.append("wrong_role")
            gold_cell_by_key = {
                _cell_column_key(cell, index): (index, cell)
                for index, cell in enumerate(row_gold_cells)
            }
            actual_cell_by_key = {
                _cell_column_key(cell, index): (index, cell)
                for index, cell in enumerate(row_actual_cells)
            }
            # Compare each aligned column, regardless of the producer's spelling of its id.
            row_cell_outcomes: list[CellOutcome] = []
            for gci, (aci, _column_score_value) in sorted(column_matches.items()):
                gold_column = gold_columns[gci]
                actual_column = actual_columns[aci]
                gold_column_key = _text(
                    _get(gold_column, "id", "column_id", "key", default="")
                )
                actual_key = _text(_get(actual_column, "id", "column_id", "key", default=""))
                gcell_pair = gold_cell_by_key.get(gold_column_key)
                acell_pair = actual_cell_by_key.get(actual_key)
                # If an authority producer uses canonical roles instead of source ids,
                # try the canonical role only after exact column ids.
                if gcell_pair is None:
                    gcell_pair = next(
                        (
                            (index, cell)
                            for index, cell in enumerate(row_gold_cells)
                            if _role(cell) == _role(gold_column)
                        ),
                        None,
                    )
                if acell_pair is None:
                    acell_pair = next(
                        (
                            (index, cell)
                            for index, cell in enumerate(row_actual_cells)
                            if _role(cell) == _role(actual_column)
                        ),
                        None,
                    )
                outcome = append_cell(
                    gcell_pair[1] if gcell_pair else None,
                    acell_pair[1] if acell_pair else None,
                    role_hint=_role(gold_column) or _role(actual_column),
                    gold_role_hint=_role(gold_column),
                    actual_role_hint=_role(actual_column),
                    gold_index=gcell_pair[0] if gcell_pair else None,
                    actual_index=acell_pair[0] if acell_pair else None,
                    row_gold_index=gri,
                    row_actual_index=ari,
                    column_gold_index=gci,
                    column_actual_index=aci,
                )
                row_cell_outcomes.append(outcome)
                row_complete &= outcome.exact and not outcome.issues
            # A value appearing under another aligned column is a placement error, not
            # merely a value typo.  Emit wrong_column in addition to wrong_value.
            corrected_cells: list[CellOutcome] = []
            for outcome in row_cell_outcomes:
                misplaced = bool(
                    "wrong_value" in outcome.issues
                    and outcome.gold_value is not None
                    and any(
                        other.actual_value == outcome.gold_value
                        and other.column_gold_index != outcome.column_gold_index
                        for other in row_cell_outcomes
                    )
                )
                if misplaced and "wrong_column" not in outcome.issues:
                    corrected_cells.append(
                        dataclasses.replace(
                            outcome,
                            status="wrong_column",
                            issues=tuple(sorted((*outcome.issues, "wrong_column"))),
                        )
                    )
                else:
                    corrected_cells.append(outcome)
            cell_outcomes.extend(corrected_cells)
            for outcome in corrected_cells:
                issues.extend(
                    issue for issue in outcome.issues if issue not in {"missed", "spurious"}
                )
            # Missing/extra columns are visible at row level as column errors.
            if missing_columns:
                issues.append("wrong_column")
            if extra_columns and _same_signature(gold_row, actual_row):
                issues.append("wrong_column")
            if not issues and row_complete:
                status = "matched"
            else:
                status = _status_priority(issues or ["wrong_value"])
            if not row_complete and not issues:
                issues.append("wrong_value")
            if row_complete:
                counts["complete_rows"] += 1
            table_rows_complete &= row_complete and not issues
            row_outcomes.append(
                RowOutcome(
                    level="row",
                    status=status,
                    gold_index=gri,
                    actual_index=ari,
                    gold_id=_identifier(gold_row, "row", gri),
                    actual_id=_identifier(actual_row, "row", ari),
                    score=round(score, 12),
                    issues=tuple(sorted(set(issues))),
                    table_gold_index=gi,
                    table_actual_index=ai,
                    structural_match=True,
                    complete=row_complete and not issues,
                )
            )
            if _line_item(gold_row):
                line_gold += 1
                if row_complete or not any(issue in {"missed", "wrong_table"} for issue in issues):
                    line_matched += 1
        for gri in sorted(missing_rows):
            counts["missed_rows"] += 1
            table_rows_complete = False
            issues = ("missed",)
            row_outcomes.append(
                RowOutcome(
                    level="row",
                    status="missed",
                    gold_index=gri,
                    actual_index=None,
                    gold_id=_identifier(gold_rows[gri], "row", gri),
                    actual_id=None,
                    issues=issues,
                    table_gold_index=gi,
                    table_actual_index=ai,
                )
            )
            if _line_item(gold_rows[gri]):
                line_gold += 1
        for ari in sorted(extra_rows):
            duplicate = any(
                _same_signature(gold_rows[gri], actual_rows[ari]) for gri in row_matches
            )
            status = "duplicate" if duplicate else "spurious"
            table_rows_complete = False
            counts["duplicates" if duplicate else "spurious_rows"] += 1
            row_issues = [status]
            if _row_has_ungrounded_value(actual_rows[ari]):
                row_issues.append("ungrounded")
            row_outcomes.append(
                RowOutcome(
                    level="row",
                    status=status,
                    gold_index=None,
                    actual_index=ari,
                    gold_id=None,
                    actual_id=_identifier(actual_rows[ari], "row", ari),
                    issues=tuple(sorted(row_issues)),
                    table_gold_index=gi,
                    table_actual_index=ai,
                )
            )
        if table_rows_complete and len(row_matches) == len(gold_rows) == len(actual_rows):
            counts["complete_tables"] += 1

    # Rows in an unmatched table are wrong-table candidates when their source page and
    # structural identity make the intended table obvious; otherwise they are missed or
    # spurious, just like rows in an aligned table.
    wrong_table_gold: set[tuple[int, int]] = set()
    wrong_table_actual: set[tuple[int, int]] = set()
    for gi in sorted(missing_tables):
        gold_table = gold_tables[gi]
        for gri, gold_row in enumerate(_rows(gold_table)):
            for ai in sorted(extra_tables):
                actual_table = actual_tables[ai]
                if _page(gold_table) != _page(actual_table):
                    continue
                for ari, actual_row in enumerate(_rows(actual_table)):
                    row_score = _row_score(gold_row, actual_row, gri, ari)
                    same_signature = _same_signature(gold_row, actual_row)
                    if same_signature or (row_score is not None and row_score >= 0.55):
                        row_issues = ["wrong_table"]
                        if _row_has_ungrounded_value(actual_row):
                            row_issues.append("ungrounded")
                        row_outcomes.append(
                            RowOutcome(
                                level="row",
                                status="wrong_table",
                                gold_index=gri,
                                actual_index=ari,
                                gold_id=_identifier(gold_row, "row", gri),
                                actual_id=_identifier(actual_row, "row", ari),
                                score=round(row_score or 0.0, 12),
                                issues=tuple(sorted(row_issues)),
                                table_gold_index=gi,
                                table_actual_index=ai,
                                structural_match=False,
                            )
                        )
                        counts["wrong_table"] += 1
                        wrong_table_gold.add((gi, gri))
                        wrong_table_actual.add((ai, ari))
                        break

    for gi in sorted(missing_tables):
        for gri, gold_row in enumerate(_rows(gold_tables[gi])):
            if (gi, gri) not in wrong_table_gold:
                counts["missed_rows"] += 1
                row_outcomes.append(
                    RowOutcome(
                        level="row",
                        status="missed",
                        gold_index=gri,
                        actual_index=None,
                        gold_id=_identifier(gold_row, "row", gri),
                        actual_id=None,
                        issues=("missed",),
                        table_gold_index=gi,
                        table_actual_index=None,
                    )
                )
    for ai in sorted(extra_tables):
        for ari, actual_row in enumerate(_rows(actual_tables[ai])):
            if (ai, ari) not in wrong_table_actual:
                counts["spurious_rows"] += 1
                row_issues = ["spurious"]
                if _row_has_ungrounded_value(actual_row):
                    row_issues.append("ungrounded")
                row_outcomes.append(
                    RowOutcome(
                        level="row",
                        status="spurious",
                        gold_index=None,
                        actual_index=ari,
                        gold_id=None,
                        actual_id=_identifier(actual_row, "row", ari),
                        issues=tuple(sorted(row_issues)),
                        table_gold_index=None,
                        table_actual_index=ai,
                    )
                )

    # A missed/spurious table still contributes to document row/column denominators.
    # The matched-table loop above has already counted its own structures.
    for gi in sorted(missing_tables):
        missed_table_rows = _rows(gold_tables[gi])
        missed_table_columns = _columns(gold_tables[gi], missed_table_rows)
        counts["gold_rows"] += len(missed_table_rows)
        counts["gold_columns"] += len(missed_table_columns)
        schema_gold += len(missed_table_columns)
        line_gold += sum(1 for row in missed_table_rows if _line_item(row))
    for ai in sorted(extra_tables):
        extra_table_rows = _rows(actual_tables[ai])
        extra_table_columns = _columns(actual_tables[ai], extra_table_rows)
        counts["actual_rows"] += len(extra_table_rows)
        counts["actual_columns"] += len(extra_table_columns)

    # Evaluate totals separately from line-item rows.  Totals are value-exact and use
    # label/kind/scope plus order; a total in a different row/table must not disappear.
    gold_totals, actual_totals = _totals(gold, gold_tables), _totals(actual, actual_tables)
    total_matches, missing_totals, extra_totals = _assignment(
        gold_totals,
        actual_totals,
        lambda left, right, li, ri: _total_score(left, right, li, ri),
        lambda _left, _right, score: score >= 0.35,
    )
    counts["gold_totals"] = len(gold_totals)
    counts["actual_totals"] = len(actual_totals)
    counts["matched_totals"] = len(total_matches)
    total_outcomes: list[CellOutcome] = []
    for gti, (ati, score) in sorted(total_matches.items()):
        total = _total_cell_outcome(gold_totals[gti], actual_totals[ati], gti, ati, score)
        total_outcomes.append(total)
        if total.exact:
            counts["exact_totals"] = counts.get("exact_totals", 0) + 1
    for gti in sorted(missing_totals):
        total_outcomes.append(_total_cell_outcome(gold_totals[gti], None, gti, None, 0.0))
    for ati in sorted(extra_totals):
        total_outcomes.append(_total_cell_outcome(None, actual_totals[ati], None, ati, 0.0))

    # Cell counters are computed from outcomes; extra cells not on a matched row are
    # represented by the row/table spurious counts and do not inflate value recall.
    all_gold_cells = [
        (cell, table, cell_index)
        for table in gold_tables
        for row in _rows(table)
        for cell_index, cell in enumerate(_cells(row))
    ]
    all_actual_cells = [
        (cell, table, cell_index)
        for table in actual_tables
        for row in _rows(table)
        for cell_index, cell in enumerate(_cells(row))
    ]
    gold_cell_values = [item[0] for item in all_gold_cells]
    actual_cell_values = [item[0] for item in all_actual_cells]
    counts["gold_cells"] = len(gold_cell_values)
    counts["actual_cells"] = len(actual_cell_values)
    counts["gold_readable_cells"] = sum(not _unreadable_marker(cell) for cell in gold_cell_values)
    counts["actual_readable_cells"] = sum(
        not _unreadable_marker(cell) for cell in actual_cell_values
    )
    counts["gold_unreadable_cells"] = len(gold_cell_values) - counts["gold_readable_cells"]
    counts["actual_unreadable_cells"] = len(actual_cell_values) - counts["actual_readable_cells"]
    counts["gold_asserted_cells"] = sum(
        _has_asserted_value(cell) and not _unreadable_marker(cell) for cell in gold_cell_values
    )
    counts["actual_asserted_cells"] = sum(
        _has_actual_value(cell) and not _unreadable_marker(cell) for cell in actual_cell_values
    )
    counts["grounded_actual_cells"] = sum(
        _has_actual_value(cell) and _grounded(cell) for cell in actual_cell_values
    )
    critical_gold = sum(
        _has_asserted_value(cell)
        and not _unreadable_marker(cell)
        and _critical(
            _cell_role(cell, table, index),
            _value_type(cell, _cell_role(cell, table, index)),
        )
        for cell, table, index in all_gold_cells
    )
    critical_actual = sum(
        _has_actual_value(cell)
        and not _unreadable_marker(cell)
        and _critical(
            _cell_role(cell, table, index),
            _value_type(cell, _cell_role(cell, table, index)),
        )
        for cell, table, index in all_actual_cells
    )
    critical_exact = 0
    for outcome in cell_outcomes:
        if outcome.gold_index is not None and outcome.actual_index is not None:
            counts["aligned_cells"] += 1
        if outcome.exact:
            counts["exact_cells"] += 1
            if outcome.readable_gold and outcome.readable_actual:
                counts["exact_readable_cells"] += 1
        if outcome.actual_value is not None and outcome.grounded_actual and outcome.exact:
            counts["grounded_exact_cells"] += 1
        if outcome.gold_index is not None and outcome.actual_index is not None:
            counts["readability_labelled_cells"] += 1
            if outcome.readable_gold == outcome.readable_actual:
                counts["readability_correct_cells"] += 1
            if not outcome.readable_gold and not outcome.readable_actual:
                counts["correct_unreadable_cells"] += 1
        for issue in outcome.issues:
            if issue in counts:
                counts[issue] += 1

        if (
            outcome.gold_index is not None
            and outcome.actual_index is not None
            and outcome.exact
            and outcome.gold_critical
            and outcome.actual_critical
        ):
            critical_exact += 1

    counts["correctly_assigned_cells"] = max(
        0,
        int(counts["aligned_cells"])
        - sum("wrong_column" in outcome.issues for outcome in cell_outcomes),
    )
    counts["critical_gold"] = critical_gold
    counts["critical_actual"] = critical_actual
    counts["critical_exact"] = critical_exact
    counts["line_item_gold_rows"] = line_gold
    counts["line_item_matched_rows"] = line_matched
    metrics = _metrics_from_counts(counts)
    metrics["critical_numeric_cell_precision"] = (
        round(critical_exact / critical_actual, 12) if critical_actual else 1.0
    )
    metrics["critical_numeric_cell_recall"] = (
        round(critical_exact / critical_gold, 12) if critical_gold else 1.0
    )
    metrics["line_item_row_recall"] = round(line_matched / line_gold, 12) if line_gold else 1.0
    metrics["header_schema_accuracy"] = (
        round(schema_correct / schema_gold, 12) if schema_gold else 1.0
    )
    metrics["grand_total_exact_accuracy"] = _grand_total_accuracy(gold_totals, total_outcomes)
    grand_indexes = {index for index, total in enumerate(gold_totals) if _is_grand_total(total)}
    metrics["grand_total_gold"] = len(grand_indexes)
    metrics["grand_total_exact"] = sum(
        outcome.exact for outcome in total_outcomes if outcome.gold_index in grand_indexes
    )
    metrics["complete_row_exact"] = (
        round(counts["complete_rows"] / counts["gold_rows"], 12) if counts["gold_rows"] else 1.0
    )
    metrics["complete_table_exact"] = (
        round(counts["complete_tables"] / counts["gold_tables"], 12)
        if counts["gold_tables"]
        else 1.0
    )
    metrics["complete_row_recall"] = metrics["complete_row_exact"]
    metrics["complete_table_recall"] = metrics["complete_table_exact"]
    all_actual_accepted = _accepted(actual)
    document_complete = bool(
        len(table_matches) == len(gold_tables)
        and not missing_tables
        and not extra_tables
        and all(row.complete for row in row_outcomes if row.gold_index is not None)
        and not any(
            row.status in {"spurious", "duplicate", "missed", "wrong_table"} for row in row_outcomes
        )
        and metrics["grand_total_exact_accuracy"] >= 1.0
    )
    metrics["complete_document_exact"] = 1.0 if document_complete else 0.0
    metrics["document_complete"] = document_complete
    metrics["complete_document"] = document_complete
    metrics["complete_document_exact_match"] = metrics["complete_document_exact"]
    metrics["auto_accepted"] = all_actual_accepted
    metrics["auto_accepted_correct"] = bool(all_actual_accepted and document_complete)
    metrics["auto_accepted_documents"] = int(all_actual_accepted)
    metrics["auto_accepted_correct_documents"] = int(
        all_actual_accepted and document_complete
    )
    # Correctness is precision over accepted documents.  With zero accepted
    # documents it is undefined, and the authority floor must fail rather than
    # reward a producer for never accepting anything.  Coverage is reported
    # separately so the zero-acceptance case is explicit to downstream gates.
    metrics["auto_accepted_document_coverage"] = 1.0 if all_actual_accepted else 0.0
    metrics["auto_accepted_document_correctness"] = (
        1.0 if all_actual_accepted and document_complete else 0.0
    )
    # A document report exposes floors too; cohort reports apply the same thresholds to
    # pooled counts.  A document is not promoted solely from a per-document floor.
    floors = _floor_results(metrics)
    passed = all(floors.values())
    gold_identity = _report_identity(gold, gold=True)
    actual_identity = _report_identity(actual, gold=False)
    pair_identity = identity_sha256(
        {"gold": gold_identity, "actual": actual_identity, "evaluator": EVALUATOR_VERSION}
    )
    return AuthorityDocumentReport(
        document_key=gold_key,
        gold_identity_sha256=gold_identity,
        actual_identity_sha256=actual_identity,
        pair_identity_sha256=pair_identity,
        metrics=metrics,
        tables=tuple(table_outcomes),
        columns=tuple(column_outcomes),
        rows=tuple(sorted(row_outcomes, key=_outcome_sort_key)),
        cells=tuple(cell_outcomes),
        totals=tuple(total_outcomes),
        floors=floors,
        passed=passed,
    )


def _outcome_sort_key(value: AlignmentOutcome) -> tuple[int, int, int, str]:
    return (
        value.table_gold_index
        if isinstance(value, RowOutcome) and value.table_gold_index is not None
        else 0,
        value.gold_index if value.gold_index is not None else 10**9,
        value.actual_index if value.actual_index is not None else 10**9,
        value.status,
    )


def _total_score(gold: Any, actual: Any, gi: int, ai: int) -> float | None:
    gold_kind = _normalise_key(_get(gold, "kind", "total_kind", "role", default=""))
    actual_kind = _normalise_key(_get(actual, "kind", "total_kind", "role", default=""))
    gold_scope = _normalise_key(_get(gold, "scope", "total_scope", default=""))
    actual_scope = _normalise_key(_get(actual, "scope", "total_scope", default=""))
    gold_label = _normalise_key(_get(gold, "label", "name", "description", default=""))
    actual_label = _normalise_key(_get(actual, "label", "name", "description", default=""))
    score = 0.0
    if gold_kind and actual_kind and gold_kind == actual_kind:
        score += 0.42
    if gold_scope and actual_scope and gold_scope == actual_scope:
        score += 0.18
    if gold_label and actual_label and gold_label == actual_label:
        score += 0.20
    score += 0.20 * max(0.0, 1.0 - min(abs(gi - ai), 5) / 5)
    return score if score >= 0.35 else None


def _total_value(value: Any, *, prefer_normalized: bool = True) -> Any:
    role = _normalise_key(_get(value, "kind", "role", "label", default="total"))
    raw = _raw_value(value, prefer_normalized=prefer_normalized)
    return _normalise_value(raw, "money", role)


def _total_cell_outcome(
    gold: Any | None, actual: Any | None, gi: int | None, ai: int | None, score: float
) -> CellOutcome:
    gold_value, actual_value = (
        _total_value(gold) if gold is not None else None,
        _total_value(actual, prefer_normalized=False) if actual is not None else None,
    )
    gold_unreadable = gold is not None and _unreadable_marker(gold)
    actual_unreadable = actual is None or _unreadable_marker(actual)
    actual_has = actual is not None and _has_actual_value(actual)
    exact = (
        gold is not None
        and actual is not None
        and (
            (gold_unreadable and actual_unreadable and not actual_has)
            or (not gold_unreadable and not actual_unreadable and gold_value == actual_value)
        )
    )
    issues: list[str] = []
    if gold is None:
        issues.append("spurious")
    elif actual is None:
        issues.append("missed")
    elif (gold_unreadable and actual_has) or (not gold_unreadable and actual_unreadable):
        issues.append("unreadable_mishandled")
    elif not exact:
        issues.append("wrong_value")
    if actual_has and not _grounded(actual):
        issues.append("ungrounded")
    return CellOutcome(
        level="total",
        status=_status_priority(issues),
        gold_index=gi,
        actual_index=ai,
        gold_id=_identifier(gold, "cell", gi or 0) if gold is not None else None,
        actual_id=_identifier(actual, "cell", ai or 0) if actual is not None else None,
        score=score,
        issues=tuple(sorted(set(issues))),
        gold_value=gold_value,
        actual_value=actual_value,
        readable_gold=not gold_unreadable,
        readable_actual=not actual_unreadable,
        grounded_actual=_grounded(actual) if actual is not None else False,
        exact=exact,
        canonical_role=_normalise_key(
            _get(gold or actual, "kind", "total_kind", "role", "label", default="total")
        ),
        value_type="money",
    )


def _grand_total_accuracy(gold_totals: Sequence[Any], outcomes: Sequence[CellOutcome]) -> float:
    indexes = [index for index, total in enumerate(gold_totals) if _is_grand_total(total)]
    if not indexes:
        return 1.0
    exact_by_gold = {
        outcome.gold_index: outcome.exact for outcome in outcomes if outcome.gold_index is not None
    }
    return sum(bool(exact_by_gold.get(index)) for index in indexes) / len(indexes)


def _is_grand_total(total: Any) -> bool:
    kind = _normalise_key(_get(total, "kind", "total_kind", "role", default=""))
    label = _normalise_key(_get(total, "label", "name", "description", default=""))
    return kind in {
        "bill_total",
        "payable_total",
        "settlement_total",
        "document_total",
        "grand_total",
    } or any(
        word in label for word in ("grand_total", "bill_total", "document_total", "payable_total")
    )


def _accepted(document: Any) -> bool:
    value = _get(document, "auto_accepted", "accepted", "straight_through", default=_MISSING)
    if value is not _MISSING:
        return bool(value)
    status = _normalise_key(_get(document, "status", "review_disposition", default=""))
    return status in {"accepted", "auto_accepted", "straight_through"}


def _empty_input(value: Any) -> bool:
    """Identify missing document/cohort containers without rejecting blank pages."""

    if value is None:
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return not value.strip()
    if isinstance(value, Mapping):
        return not bool(value)
    if isinstance(value, Sequence):
        return not bool(value)
    return not bool(_mapping(value))


def _cohort_identity(value: Any, *, label: str, index: int) -> tuple[str, str]:
    """Return a validated primary identity for one cohort document."""

    source = _get(value, "source_sha256", "document_sha256", default=None)
    document_id = _get(value, "document_id", default=None)
    if source not in (None, ""):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{label} document {index} has an empty source identity")
        return "source_sha256", source.strip()
    if document_id not in (None, ""):
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"{label} document {index} has an empty document identity")
        return "document_id", document_id.strip()
    raise ValueError(
        f"{label} document {index} requires source_sha256 or document_id for cohort pairing"
    )


def _cohort_documents(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, Mapping):
        nested = _sequence_from_document(value, ("documents", "gold_documents", "records"))
        if nested:
            return nested
        if any(
            key in value
            for key in ("source_sha256", "document_sha256", "document_id", "tables", "pages")
        ):
            return [value]
        return list(value.values())
    direct = _items(value)
    if direct:
        return direct
    nested = _sequence_from_document(value, ("documents", "gold_documents", "records"))
    if nested:
        return nested
    raise ValueError(f"{label} cohort is empty or has no document sequence")


def _validate_cohort_pair_order(pairs: Sequence[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    if not pairs:
        raise ValueError("authority cohorts may not be empty")
    gold_keys: list[tuple[str, str]] = []
    actual_keys: list[tuple[str, str]] = []
    for index, (left, right) in enumerate(pairs):
        gold_keys.append(_cohort_identity(left, label="gold", index=index))
        actual_keys.append(_cohort_identity(right, label="actual", index=index))
    if len(set(gold_keys)) != len(gold_keys):
        raise ValueError("gold cohort contains duplicate source identities")
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("actual cohort contains duplicate source identities")
    if set(gold_keys) != set(actual_keys):
        raise ValueError("gold and actual cohort source identity sets must match")
    if gold_keys != actual_keys:
        raise ValueError("gold and actual cohort source identity order must match")
    return list(pairs)


def _cohort_pairs(gold: Any, actual: Any | None) -> list[tuple[Any, Any]]:
    if _empty_input(gold) or (actual is not None and _empty_input(actual)):
        raise ValueError("authority cohorts may not be empty")
    if actual is not None:
        gold_items = _cohort_documents(gold, label="gold")
        actual_items = _cohort_documents(actual, label="actual")
        if len(gold_items) != len(actual_items):
            raise ValueError("gold and actual cohort lengths must match")
        if isinstance(gold, Mapping) and isinstance(actual, Mapping):
            # Mappings are already keyed collections; pair by validated document
            # identity rather than dictionary insertion order.
            gold_by_identity = {
                _cohort_identity(item, label="gold", index=index): item
                for index, item in enumerate(gold_items)
            }
            actual_by_identity = {
                _cohort_identity(item, label="actual", index=index): item
                for index, item in enumerate(actual_items)
            }
            if len(gold_by_identity) != len(gold_items):
                raise ValueError("gold cohort contains duplicate source identities")
            if len(actual_by_identity) != len(actual_items):
                raise ValueError("actual cohort contains duplicate source identities")
            if set(gold_by_identity) != set(actual_by_identity):
                raise ValueError("gold and actual cohort source identity sets must match")
            return [
                (gold_by_identity[key], actual_by_identity[key])
                for key in sorted(gold_by_identity, key=str)
            ]
        pairs = list(zip(gold_items, actual_items, strict=True))
        return _validate_cohort_pair_order(pairs)
    if isinstance(gold, Mapping):
        # A keyed mapping may contain {gold, actual} pair records or keyed gold/actual
        # documents.  Never pair by iteration order when explicit identities exist.
        if "gold" in gold and "actual" in gold:
            return _validate_cohort_pair_order([(gold["gold"], gold["actual"])])
        pairs: list[tuple[Any, Any]] = []
        for key in sorted(gold, key=str):
            value = gold[key]
            if isinstance(value, Mapping) and "gold" in value and "actual" in value:
                pairs.append((value["gold"], value["actual"]))
        if pairs:
            return _validate_cohort_pair_order(pairs)
    records = _items(gold) or _sequence_from_document(gold, ("documents", "pairs", "records"))
    pairs = []
    for record in records:
        if isinstance(record, Mapping):
            left, right = record.get("gold"), record.get("actual", record.get("prediction"))
        else:
            left = _get(record, "gold", default=None)
            right = _get(record, "actual", "prediction", default=None)
        if left is None or right is None:
            raise ValueError("cohort records require gold and actual documents")
        pairs.append((left, right))
    return _validate_cohort_pair_order(pairs)


def _aggregate_documents(documents: Sequence[AuthorityDocumentReport]) -> dict[str, Any]:
    summed: dict[str, int | float] = {}
    for document in documents:
        for key, value in document.metrics.items():
            if isinstance(value, bool):
                continue
            if (
                isinstance(value, (int, float))
                and key.endswith(("tables", "columns", "rows", "cells", "totals"))
                or key
                in {
                    "matched_tables",
                    "matched_columns",
                    "matched_rows",
                    "gold_tables",
                    "actual_tables",
                    "gold_columns",
                    "actual_columns",
                    "gold_rows",
                    "actual_rows",
                    "gold_cells",
                    "actual_cells",
                    "aligned_cells",
                    "correctly_assigned_cells",
                    "exact_cells",
                    "gold_asserted_cells",
                    "actual_asserted_cells",
                    "gold_readable_cells",
                    "actual_readable_cells",
                    "exact_readable_cells",
                    "grounded_actual_cells",
                    "grounded_exact_cells",
                    "gold_unreadable_cells",
                    "actual_unreadable_cells",
                    "correct_unreadable_cells",
                    "gold_totals",
                    "actual_totals",
                    "matched_totals",
                    "exact_totals",
                    "grand_total_gold",
                    "grand_total_exact",
                    "critical_gold",
                    "critical_actual",
                    "critical_exact",
                    "line_item_gold_rows",
                    "line_item_matched_rows",
                }
            ):
                summed[key] = summed.get(key, 0) + value
    metrics = _metrics_from_counts(summed)
    for key, numerator, denominator in (
        ("critical_numeric_cell_precision", "critical_exact", "critical_actual"),
        ("critical_numeric_cell_recall", "critical_exact", "critical_gold"),
        ("line_item_row_recall", "line_item_matched_rows", "line_item_gold_rows"),
        ("header_schema_accuracy", "matched_columns", "gold_columns"),
        ("grand_total_exact_accuracy", "grand_total_exact", "grand_total_gold"),
    ):
        metrics[key] = (
            round(summed.get(numerator, 0) / summed.get(denominator, 1), 12)
            if summed.get(denominator, 0)
            else 1.0
        )
    # Header/schema accuracy should count role-correct columns, not merely aligned ones.
    if documents:
        metrics["header_schema_accuracy"] = (
            round(
                sum(
                    document.metrics.get("header_schema_accuracy", 1.0)
                    * document.metrics.get("gold_columns", 0)
                    for document in documents
                )
                / sum(document.metrics.get("gold_columns", 0) for document in documents),
                12,
            )
            if sum(document.metrics.get("gold_columns", 0) for document in documents)
            else 1.0
        )
    accepted = sum(1 for document in documents if document.metrics.get("auto_accepted"))
    correct = sum(1 for document in documents if document.metrics.get("auto_accepted_correct"))
    metrics["auto_accepted_documents"] = accepted
    metrics["auto_accepted_correct_documents"] = correct
    metrics["auto_accepted_document_coverage"] = (
        round(accepted / len(documents), 12) if documents else 0.0
    )
    metrics["auto_accepted_document_correctness"] = (
        round(correct / accepted, 12) if accepted else 0.0
    )
    metrics["complete_document_exact"] = (
        round(
            sum(bool(document.metrics.get("document_complete")) for document in documents)
            / len(documents),
            12,
        )
        if documents
        else 0.0
    )
    metrics["document_precision"] = (
        round(
            sum(document.metrics.get("complete_document_exact", 0.0) for document in documents)
            / len(documents),
            12,
        )
        if documents
        else 0.0
    )
    metrics["document_recall"] = metrics["document_precision"]
    metrics["document_f1"] = metrics["document_precision"]
    metrics["document_count"] = len(documents)
    return metrics


def evaluate_document(
    gold: Any,
    actual: Any | None = None,
    *,
    prediction: Any | None = None,
    predicted: Any | None = None,
    result: Any | None = None,
    document_key: str | None = None,
) -> AuthorityDocumentReport:
    """Evaluate one gold/prediction document pair."""

    if _empty_input(gold):
        raise _EmptyAuthorityInputError("gold document may not be empty")
    if actual is None:
        actual = prediction if prediction is not None else predicted
    if actual is None:
        actual = result
    if actual is None:
        raise TypeError("evaluate_document requires a prediction/actual document")
    if _empty_input(actual):
        raise _EmptyAuthorityInputError("actual document may not be empty")
    return _evaluate_document(gold, actual, document_key=document_key)


def evaluate_authority(
    gold: Any,
    actual: Any | None = None,
    *,
    prediction: Any | None = None,
    predicted: Any | None = None,
    result: Any | None = None,
    document_key: str | None = None,
) -> AuthorityDocumentReport:
    """Primary single-document entry point (alias retained in public API docs)."""

    return evaluate_document(
        gold,
        actual,
        prediction=prediction,
        predicted=predicted,
        result=result,
        document_key=document_key,
    )


def evaluate_cohort(
    gold: Any,
    actual: Any | None = None,
    *,
    predictions: Any | None = None,
) -> AuthorityCohortReport:
    """Evaluate a frozen cohort and apply all Table Magic floors to pooled metrics."""

    if actual is None and predictions is not None:
        actual = predictions
    pairs = _cohort_pairs(gold, actual)
    reports = tuple(
        evaluate_document(left, right, document_key=_doc_key(left, f"document-{index:06d}") or None)
        for index, (left, right) in enumerate(pairs)
    )
    metrics = _aggregate_documents(reports)
    floors = _floor_results(metrics)
    cohort_identity = identity_sha256(
        {
            "documents": [report.pair_identity_sha256 for report in reports],
            "evaluator": EVALUATOR_VERSION,
        }
    )
    return AuthorityCohortReport(reports, metrics, floors, all(floors.values()), cohort_identity)


# Short aliases make call sites readable while retaining one implementation and one
# evaluator version for release-manifest identity.
evaluate = evaluate_authority
evaluate_cohort_metrics = evaluate_cohort
evaluate_authoritative = evaluate_authority
evaluate_authority_document = evaluate_document
evaluate_authoritative_document = evaluate_document
score_document = evaluate_document
score_cohort = evaluate_cohort
AuthorityReport = AuthorityDocumentReport
DocumentEvaluation = AuthorityDocumentReport
CohortEvaluation = AuthorityCohortReport


__all__ = [
    "AlignmentOutcome",
    "AuthorityCohortReport",
    "AuthorityDocumentReport",
    "AuthorityReport",
    "CellOutcome",
    "CohortEvaluation",
    "DocumentEvaluation",
    "EVALUATOR_VERSION",
    "RowOutcome",
    "TABLE_MAGIC_FLOORS",
    "TABLE_MAGIC_THRESHOLDS",
    "evaluate",
    "evaluate_authority",
    "evaluate_cohort",
    "evaluate_cohort_metrics",
    "evaluate_document",
    "evaluate_authoritative",
    "evaluate_authoritative_document",
    "evaluate_authority_document",
    "identity_sha256",
    "score_cohort",
    "score_document",
]
