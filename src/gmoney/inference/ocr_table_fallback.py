from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

NUMERIC = re.compile(r"(?:₹|rs\.?\s*)?[+-]?(?:\d[\d,]*)(?:\.\d{1,2})?", re.IGNORECASE)


@dataclass(frozen=True)
class TokenBox:
    index: int
    text: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True)
class GeometryTableProposal:
    box: tuple[float, float, float, float]
    token_indexes: tuple[int, ...]
    row_count: int
    confidence: float


def _is_numeric(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    return bool(NUMERIC.fullmatch(compact))


def _lines(tokens: list[TokenBox]) -> list[list[TokenBox]]:
    if not tokens:
        return []
    tolerance = max(4.0, median(token.height for token in tokens) * 0.75)
    lines: list[list[TokenBox]] = []
    for token in sorted(tokens, key=lambda item: (item.center_y, item.left)):
        if not lines:
            lines.append([token])
            continue
        current_center = median(item.center_y for item in lines[-1])
        if abs(token.center_y - current_center) <= tolerance:
            lines[-1].append(token)
        else:
            lines.append([token])
    return [sorted(line, key=lambda item: item.left) for line in lines]


def propose_tables_from_ocr(
    rec_boxes: list[list[float]],
    rec_texts: list[str],
    minimum_rows: int = 2,
) -> list[GeometryTableProposal]:
    if len(rec_boxes) != len(rec_texts):
        raise ValueError("OCR boxes and texts must have the same length")
    tokens = [
        TokenBox(index, text, float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for index, (box, text) in enumerate(zip(rec_boxes, rec_texts, strict=True))
        if len(box) == 4 and text.strip()
    ]
    lines = _lines(tokens)
    candidate_lines = [
        line
        for line in lines
        if len(line) >= 3 and sum(_is_numeric(token.text) for token in line) >= 1
    ]
    if len(candidate_lines) < minimum_rows:
        return []

    typical_height = median(token.height for line in candidate_lines for token in line)
    groups: list[list[list[TokenBox]]] = []
    for line in candidate_lines:
        if not groups:
            groups.append([line])
            continue
        prior_bottom = max(token.bottom for token in groups[-1][-1])
        current_top = min(token.top for token in line)
        if current_top - prior_bottom <= typical_height * 3:
            groups[-1].append(line)
        else:
            groups.append([line])

    proposals: list[GeometryTableProposal] = []
    for group in groups:
        if len(group) < minimum_rows:
            continue
        group_tokens = [token for line in group for token in line]
        numeric_count = sum(_is_numeric(token.text) for token in group_tokens)
        numeric_fraction = numeric_count / len(group_tokens)
        confidence = min(0.95, 0.5 + len(group) * 0.05 + numeric_fraction * 0.25)
        proposals.append(
            GeometryTableProposal(
                box=(
                    min(token.left for token in group_tokens),
                    min(token.top for token in group_tokens),
                    max(token.right for token in group_tokens),
                    max(token.bottom for token in group_tokens),
                ),
                token_indexes=tuple(sorted(token.index for token in group_tokens)),
                row_count=len(group),
                confidence=confidence,
            )
        )
    return proposals
