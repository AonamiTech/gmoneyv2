"""Deterministic source-space matching and ranking for Table Magic M5.

This module deliberately has no inference dependencies.  It consumes proposals whose
geometry has already been mapped to SOURCE_RAW and returns a complete, reproducible
decision graph.  Runtime code may persist these decisions in shadow mode without
changing the authoritative V6 extraction payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

MATCH_POLICY_VERSION = "table_magic_match_v1"
LOGICAL_TABLE_ID_VERSION = "logical_table_id_v1"
MATCH_ACCEPT_THRESHOLD = 0.60
MATCH_AMBIGUITY_MARGIN = 0.10
DERIVATIVE_AGREEMENT_IOU = 0.65


class TableCandidateVariant(StrEnum):
    ORIENTED_RAW = "oriented_raw"
    PROJECTIVE = "projective"
    PROJECTIVE_ENHANCED = "projective_enhanced"
    UVDOC = "uvdoc"
    UVDOC_ENHANCED = "uvdoc_enhanced"


VARIANT_PREFERENCE = tuple(TableCandidateVariant)


@dataclass(frozen=True)
class TableProposal:
    proposal_id: str
    source_sha256: str
    page_number: int
    page_width: int
    page_height: int
    variant: TableCandidateVariant
    source_box: tuple[float, float, float, float]
    reading_order: int
    header_tokens: tuple[str, ...] = ()
    table_type: str | None = None
    page_artifact_sha256: str | None = None
    candidate_box: tuple[int, int, int, int] | None = None
    grounded_rescue: bool = False
    transform_valid: bool = True
    distortion: float = 0.0
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left, top, right, bottom = self.source_box
        if not self.proposal_id or self.page_number < 1:
            raise ValueError("table proposal identity is invalid")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("table proposal page dimensions are invalid")
        if not all(math.isfinite(value) for value in self.source_box):
            raise ValueError("table proposal geometry is non-finite")
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("table proposal box is invalid")
        if right > self.page_width or bottom > self.page_height:
            raise ValueError("table proposal box exceeds SOURCE_RAW")
        if self.reading_order < 0 or not math.isfinite(self.distortion) or self.distortion < 0:
            raise ValueError("table proposal ranking metadata is invalid")


@dataclass(frozen=True)
class MatchFeatures:
    polygon_iou: float
    center_proximity: float
    reading_order: float
    table_type: float
    header_similarity: float
    weighted_score: float


@dataclass(frozen=True)
class MatchEdge:
    anchor_proposal_id: str
    candidate_proposal_id: str
    candidate_variant: TableCandidateVariant
    features: MatchFeatures
    accepted: bool
    reason: str


@dataclass(frozen=True)
class LogicalTable:
    logical_table_id: str
    source_sha256: str
    page_number: int
    anchor_proposal_id: str | None
    proposal_ids: tuple[str, ...]
    derivative_only: bool = False
    grounded_rescue: bool = False


@dataclass(frozen=True)
class MatchingResult:
    logical_tables: tuple[LogicalTable, ...]
    edges: tuple[MatchEdge, ...]


def normalize_header_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    """Return the frozen M5 header signature normalization."""

    output: list[str] = []
    for token in tokens:
        normalized = unicodedata.normalize("NFKC", str(token)).casefold()
        normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        output.extend(part for part in normalized.split() if part)
    return tuple(output)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_table_id(proposal: TableProposal) -> str:
    """Hash stable source identity, quantized geometry, order, and header signature."""

    left, top, right, bottom = proposal.source_box
    quantized = tuple(
        max(0, min(1000, round(value * 1000 / dimension)))
        for value, dimension in zip(
            (left, top, right, bottom),
            (proposal.page_width, proposal.page_height) * 2,
            strict=True,
        )
    )
    return _canonical_digest(
        {
            "version": LOGICAL_TABLE_ID_VERSION,
            "source_sha256": proposal.source_sha256,
            "page_number": proposal.page_number,
            "source_box_q1000": quantized,
            "reading_order": proposal.reading_order,
            "header_signature": normalize_header_tokens(proposal.header_tokens),
        }
    )


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _center_proximity(left: TableProposal, right: TableProposal) -> float:
    left_center = (
        (left.source_box[0] + left.source_box[2]) / 2,
        (left.source_box[1] + left.source_box[3]) / 2,
    )
    right_center = (
        (right.source_box[0] + right.source_box[2]) / 2,
        (right.source_box[1] + right.source_box[3]) / 2,
    )
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    diagonal = math.hypot(left.page_width, left.page_height)
    return max(0.0, 1.0 - distance / diagonal)


def _header_similarity(left: TableProposal, right: TableProposal) -> float:
    left_tokens = set(normalize_header_tokens(left.header_tokens))
    right_tokens = set(normalize_header_tokens(right.header_tokens))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _table_type_agreement(left: TableProposal, right: TableProposal) -> float:
    if not left.table_type or not right.table_type or "unknown" in {
        left.table_type.casefold(),
        right.table_type.casefold(),
    }:
        return 0.5
    return float(left.table_type.casefold() == right.table_type.casefold())


def match_features(anchor: TableProposal, candidate: TableProposal) -> MatchFeatures:
    if (
        anchor.source_sha256 != candidate.source_sha256
        or anchor.page_number != candidate.page_number
        or anchor.page_width != candidate.page_width
        or anchor.page_height != candidate.page_height
    ):
        raise ValueError("cannot match proposals from different source pages")
    iou = _intersection_over_union(anchor.source_box, candidate.source_box)
    center = _center_proximity(anchor, candidate)
    order = 1.0 / (1.0 + abs(anchor.reading_order - candidate.reading_order))
    table_type = _table_type_agreement(anchor, candidate)
    header = _header_similarity(anchor, candidate)
    weighted = 0.45 * iou + 0.15 * center + 0.10 * order + 0.10 * table_type + 0.20 * header
    return MatchFeatures(
        polygon_iou=round(iou, 6),
        center_proximity=round(center, 6),
        reading_order=round(order, 6),
        table_type=round(table_type, 6),
        header_similarity=round(header, 6),
        weighted_score=round(weighted, 6),
    )


def _maximum_weight_assignment(
    anchors: Sequence[TableProposal],
    candidates: Sequence[TableProposal],
    weights: Mapping[tuple[str, str], float],
) -> set[tuple[str, str]]:
    """Exact deterministic Hungarian assignment with zero-weight dummy edges."""

    size = max(len(anchors), len(candidates))
    if size == 0:
        return set()
    ordered_anchors = sorted(anchors, key=lambda item: item.proposal_id)
    ordered_candidates = sorted(candidates, key=lambda item: item.proposal_id)
    tie_scale = size * size + 1
    encoded: list[list[int]] = [[0] * size for _ in range(size)]
    pair_rank = 0
    for row, anchor in enumerate(ordered_anchors):
        for column, candidate in enumerate(ordered_candidates):
            score = weights.get((anchor.proposal_id, candidate.proposal_id), 0.0)
            if score > 0:
                tie_bonus = size * size - pair_rank
                encoded[row][column] = round(score * 1_000_000) * tie_scale + tie_bonus
            pair_rank += 1
    maximum = max(max(row) for row in encoded)
    costs = [[maximum - value for value in row] for row in encoded]

    # 1-indexed implementation of the rectangular Hungarian algorithm.  The
    # matrix is square because dummy rows/columns represent unmatched proposals.
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    infinity = 10**30
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = infinity
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    selected: set[tuple[str, str]] = set()
    for column in range(1, size + 1):
        row = p[column]
        if row <= len(ordered_anchors) and column <= len(ordered_candidates):
            pair = (
                ordered_anchors[row - 1].proposal_id,
                ordered_candidates[column - 1].proposal_id,
            )
            if weights.get(pair, 0.0) > 0:
                selected.add(pair)
    return selected


def _candidate_edges(
    anchors: Sequence[TableProposal], candidates: Sequence[TableProposal]
) -> tuple[dict[tuple[str, str], MatchFeatures], set[tuple[str, str]], set[tuple[str, str]]]:
    features = {
        (anchor.proposal_id, candidate.proposal_id): match_features(anchor, candidate)
        for anchor in anchors
        for candidate in candidates
    }
    above_threshold = {
        pair
        for pair, item in features.items()
        if item.weighted_score >= MATCH_ACCEPT_THRESHOLD
    }
    ambiguous: set[tuple[str, str]] = set()
    for pair in above_threshold:
        anchor_id, candidate_id = pair
        competing = [
            item.weighted_score
            for other, item in features.items()
            if other != pair
            and other in above_threshold
            and (other[0] == anchor_id or other[1] == candidate_id)
        ]
        runner_up = max(competing, default=0.0)
        if features[pair].weighted_score - runner_up < MATCH_AMBIGUITY_MARGIN:
            ambiguous.add(pair)
    eligible = above_threshold - ambiguous
    return features, eligible, ambiguous


def match_logical_tables(
    proposals: Sequence[TableProposal],
    *,
    anchor_variant: TableCandidateVariant = TableCandidateVariant.ORIENTED_RAW,
) -> MatchingResult:
    """Match proposals per branch and retain a complete auditable edge inventory."""

    if len({item.proposal_id for item in proposals}) != len(proposals):
        raise ValueError("table proposal IDs must be unique")
    pages = {(item.source_sha256, item.page_number) for item in proposals}
    if len(pages) > 1:
        raise ValueError("logical matching operates on one source page at a time")
    anchors = sorted(
        (item for item in proposals if item.variant is anchor_variant),
        key=lambda item: item.proposal_id,
    )
    groups: dict[str, list[str]] = {item.proposal_id: [item.proposal_id] for item in anchors}
    edges: list[MatchEdge] = []
    unmatched: list[TableProposal] = []
    for variant in VARIANT_PREFERENCE:
        if variant is anchor_variant:
            continue
        candidates = sorted(
            (item for item in proposals if item.variant is variant),
            key=lambda item: item.proposal_id,
        )
        features, eligible, ambiguous = _candidate_edges(anchors, candidates)
        assignment = _maximum_weight_assignment(
            anchors,
            candidates,
            {pair: features[pair].weighted_score for pair in eligible},
        )
        matched_candidate_ids = {candidate_id for _anchor_id, candidate_id in assignment}
        for pair, item in sorted(features.items()):
            accepted = pair in assignment
            reason = (
                "accepted_maximum_weight"
                if accepted
                else (
                    "abstained_ambiguous"
                    if pair in ambiguous
                    else (
                        "rejected_below_threshold"
                        if item.weighted_score < MATCH_ACCEPT_THRESHOLD
                        else "rejected_competing_assignment"
                    )
                )
            )
            edges.append(
                MatchEdge(
                    anchor_proposal_id=pair[0],
                    candidate_proposal_id=pair[1],
                    candidate_variant=variant,
                    features=item,
                    accepted=accepted,
                    reason=reason,
                )
            )
            if accepted:
                groups[pair[0]].append(pair[1])
        unmatched.extend(
            item for item in candidates if item.proposal_id not in matched_candidate_ids
        )

    proposal_by_id = {item.proposal_id: item for item in proposals}
    logical_tables = [
        LogicalTable(
            logical_table_id=logical_table_id(anchor),
            source_sha256=anchor.source_sha256,
            page_number=anchor.page_number,
            anchor_proposal_id=anchor.proposal_id,
            proposal_ids=tuple(groups[anchor.proposal_id]),
        )
        for anchor in anchors
    ]

    # Derivative-only proposals form connected components only across independent
    # variants.  A single branch can never invent a logical table on its own.
    remaining = {item.proposal_id for item in unmatched}
    adjacency: dict[str, set[str]] = {item: set() for item in remaining}
    ordered_remaining = sorted(remaining)
    for index, left_id in enumerate(ordered_remaining):
        left = proposal_by_id[left_id]
        for right_id in ordered_remaining[index + 1 :]:
            right = proposal_by_id[right_id]
            if left.variant is right.variant:
                continue
            compatible_headers = (
                not normalize_header_tokens(left.header_tokens)
                or not normalize_header_tokens(right.header_tokens)
                or _header_similarity(left, right) >= 0.5
            )
            if (
                _intersection_over_union(left.source_box, right.source_box)
                >= DERIVATIVE_AGREEMENT_IOU
                and compatible_headers
            ):
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        remaining -= component
        members = sorted((proposal_by_id[item] for item in component), key=lambda item: (
            VARIANT_PREFERENCE.index(item.variant), item.proposal_id
        ))
        variants = {item.variant for item in members}
        grounded = any(item.grounded_rescue for item in members)
        if len(variants) < 2 and not grounded:
            continue
        anchor = members[0]
        logical_tables.append(
            LogicalTable(
                logical_table_id=logical_table_id(anchor),
                source_sha256=anchor.source_sha256,
                page_number=anchor.page_number,
                anchor_proposal_id=None,
                proposal_ids=tuple(item.proposal_id for item in members),
                derivative_only=True,
                grounded_rescue=grounded,
            )
        )
    return MatchingResult(
        logical_tables=tuple(sorted(logical_tables, key=lambda item: item.logical_table_id)),
        edges=tuple(
            sorted(
                edges,
                key=lambda item: (
                    item.anchor_proposal_id,
                    item.candidate_variant.value,
                    item.candidate_proposal_id,
                ),
            )
        ),
    )


def stage_one_key(proposal: TableProposal) -> tuple[Any, ...]:
    """Frozen deterministic screen used to choose at most two reconstructions."""

    metrics = proposal.metrics
    valid = bool(proposal.transform_valid and metrics.get("lineage_valid", True))
    reconstruction_value = metrics.get("reconstruction_score", ())
    reconstruction_score = (
        tuple(int(value) for value in reconstruction_value)
        if isinstance(reconstruction_value, (list, tuple))
        else (int(reconstruction_value),)
    )
    return (
        int(valid),
        reconstruction_score,
        int(metrics.get("financial_token_count", 0)),
        int(metrics.get("header_token_count", 0)),
        int(metrics.get("table_confidence_ppm", 0)),
        int(metrics.get("ocr_coverage_ppm", 0)),
        int(metrics.get("ocr_confidence_ppm", 0)),
        int(metrics.get("column_stability_ppm", 0)),
        int(metrics.get("straightness_ppm", 0)),
        -int(metrics.get("duplicate_token_count", 0)),
        -int(metrics.get("invalid_character_count", 0)),
        -round(proposal.distortion * 1_000_000),
        -VARIANT_PREFERENCE.index(proposal.variant),
        proposal.proposal_id,
    )


def select_stage_one(proposals: Sequence[TableProposal]) -> tuple[TableProposal, ...]:
    valid = [item for item in proposals if item.transform_valid]
    return tuple(sorted(valid, key=stage_one_key, reverse=True)[:2])


def stage_two_key(
    proposal: TableProposal, reconstruction_metrics: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Frozen whole-table ranking; exact ties retain the least transformed branch."""

    return (
        -int(reconstruction_metrics.get("critical_error_count", 0)),
        int(reconstruction_metrics.get("required_column_count", 0)),
        int(reconstruction_metrics.get("grounded_complete_row_count", 0)),
        int(reconstruction_metrics.get("evidence_linkage_ppm", 0)),
        int(reconstruction_metrics.get("arithmetic_consistency_count", 0)),
        int(reconstruction_metrics.get("cross_channel_agreement_ppm", 0)),
        -int(reconstruction_metrics.get("conflict_count", 0)),
        -int(reconstruction_metrics.get("duplicate_row_count", 0)),
        -int(reconstruction_metrics.get("missing_row_count", 0)),
        -round(proposal.distortion * 1_000_000),
        -VARIANT_PREFERENCE.index(proposal.variant),
        proposal.proposal_id,
    )


def select_stage_two(
    candidates: Sequence[tuple[TableProposal, Mapping[str, Any]]],
) -> tuple[TableProposal, Mapping[str, Any]]:
    if not candidates:
        raise ValueError("stage two requires at least one reconstructed candidate")
    return max(candidates, key=lambda item: stage_two_key(item[0], item[1]))


__all__ = [
    "DERIVATIVE_AGREEMENT_IOU",
    "LOGICAL_TABLE_ID_VERSION",
    "MATCH_ACCEPT_THRESHOLD",
    "MATCH_AMBIGUITY_MARGIN",
    "MATCH_POLICY_VERSION",
    "LogicalTable",
    "MatchEdge",
    "MatchFeatures",
    "MatchingResult",
    "TableCandidateVariant",
    "TableProposal",
    "logical_table_id",
    "match_features",
    "match_logical_tables",
    "normalize_header_tokens",
    "select_stage_one",
    "select_stage_two",
    "stage_one_key",
    "stage_two_key",
]
