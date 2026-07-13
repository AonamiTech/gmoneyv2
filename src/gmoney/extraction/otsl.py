from __future__ import annotations

import re
from dataclasses import dataclass

TAGS = ("<fcel>", "<ecel>", "<nl>", "<lcel>", "<ucel>", "<xcel>")
TAG_PATTERN = re.compile(r"(<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)", re.IGNORECASE)
HEADER_TERMS = {
    "description",
    "particular",
    "service",
    "item",
    "amount",
    "rate",
    "quantity",
    "qty",
    "discount",
    "date",
}


@dataclass(frozen=True)
class OtslCell:
    text: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    is_span_marker: bool = False


@dataclass(frozen=True)
class OtslTable:
    rows: tuple[tuple[OtslCell, ...], ...]

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_otsl(content: str) -> OtslTable:
    content = content.strip()
    first_tag = TAG_PATTERN.search(content)
    if not first_tag:
        return OtslTable(rows=())
    if first_tag.start() > 0 and _clean_text(content[: first_tag.start()]):
        content = "<fcel>" + content
    parts = TAG_PATTERN.split(content)
    rows: list[list[OtslCell]] = [[]]
    row_index = 0
    column_index = 0
    index = 1 if parts and parts[0] == "" else 0
    while index < len(parts):
        tag = parts[index].casefold()
        following = parts[index + 1] if index + 1 < len(parts) else ""
        text = _clean_text(following)
        if tag == "<nl>":
            if rows[-1]:
                rows.append([])
                row_index += 1
            column_index = 0
        elif tag in {"<fcel>", "<ecel>"}:
            rows[-1].append(
                OtslCell(
                    text=text if tag == "<fcel>" else "",
                    row=row_index,
                    column=column_index,
                )
            )
            column_index += 1
        elif tag in {"<lcel>", "<ucel>", "<xcel>"}:
            rows[-1].append(
                OtslCell(
                    text="",
                    row=row_index,
                    column=column_index,
                    is_span_marker=True,
                )
            )
            column_index += 1
        index += 2
    if rows and not rows[-1]:
        rows.pop()
    width = max((len(row) for row in rows), default=0)
    normalized: list[tuple[OtslCell, ...]] = []
    for row_index, row in enumerate(rows):
        padded = list(row)
        while len(padded) < width:
            padded.append(OtslCell(text="", row=row_index, column=len(padded)))
        normalized.append(tuple(padded))
    return OtslTable(rows=tuple(normalized))


def _header_score(row: tuple[OtslCell, ...]) -> int:
    words = set(re.findall(r"[a-z]+", " ".join(cell.text for cell in row).casefold()))
    return len(words & HEADER_TERMS)


def split_otsl_tables(table: OtslTable) -> tuple[OtslTable, ...]:
    if not table.rows:
        return ()
    starts = [index for index, row in enumerate(table.rows) if _header_score(row) >= 2]
    if not starts:
        return (table,)
    if starts[0] != 0:
        starts.insert(0, 0)
    tables: list[OtslTable] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(table.rows)
        rows = table.rows[start:end]
        if rows:
            tables.append(OtslTable(rows=rows))
    return tuple(tables)

