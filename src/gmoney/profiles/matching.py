from __future__ import annotations

import re
from statistics import mean

from gmoney.contracts.phase3 import (
    LayoutObservation,
    LayoutProfile,
    ProfileLifecycle,
    ProfileMatch,
)
from gmoney.extraction.ocr_rows import TableSchemaState


def _normalize_tokens(values: tuple[str, ...]) -> set[str]:
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1
    }


def _header_score(profile: LayoutProfile, observation: LayoutObservation) -> float | None:
    left = _normalize_tokens((*profile.header_tokens, *profile.stable_anchors))
    right = _normalize_tokens(observation.header_tokens)
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def _column_score(profile: LayoutProfile, observation: LayoutObservation) -> float | None:
    common = profile.column_centers.keys() & observation.column_centers.keys()
    if not common:
        return None
    scores = []
    for role in common:
        tolerance = max(0.02, profile.column_tolerances.get(role, 0.06))
        distance = abs(profile.column_centers[role] - observation.column_centers[role])
        scores.append(max(0.0, 1 - distance / tolerance))
    return mean(scores)


def _geometry_score(profile: LayoutProfile, observation: LayoutObservation) -> float:
    aspect_delta = abs(profile.page_aspect_ratio - observation.page_aspect_ratio) / max(
        profile.page_aspect_ratio, observation.page_aspect_ratio
    )
    box_delta = mean(
        abs(left - right)
        for left, right in zip(profile.table_box, observation.table_box, strict=True)
    )
    return max(0.0, 1 - aspect_delta * 2 - box_delta * 2)


def score_profile(
    profile: LayoutProfile, observation: LayoutObservation
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    if profile.hospital_id and observation.hospital_id != profile.hospital_id:
        return 0.0, ("hospital_mismatch",)
    identity = (
        1.0
        if profile.hospital_id == observation.hospital_id and profile.hospital_id
        else 0.5
    )
    if profile.global_family:
        identity = max(identity, 0.5)
    type_score = 0.0
    if profile.page_type == observation.page_type:
        type_score += 0.4
    else:
        reasons.append("page_type_mismatch")
    if profile.table_type == observation.table_type:
        type_score += 0.6
    else:
        reasons.append("table_type_mismatch")
    header = _header_score(profile, observation)
    columns = _column_score(profile, observation)
    geometry = _geometry_score(profile, observation)
    components = [(identity, 0.2), (type_score, 0.15), (geometry, 0.1)]
    if header is not None:
        components.append((header, 0.3))
    if columns is not None:
        components.append((columns, 0.25))
    score = sum(value * weight for value, weight in components) / sum(
        weight for _, weight in components
    )
    if header is not None and header < 0.5:
        reasons.append("weak_header_match")
    if columns is not None and columns < 0.5:
        reasons.append("weak_column_match")
    return max(0.0, min(1.0, score)), tuple(reasons)


def match_profile(
    profiles: tuple[LayoutProfile, ...],
    observation: LayoutObservation,
    *,
    include_shadow: bool = False,
) -> ProfileMatch:
    eligible_states = {ProfileLifecycle.ACTIVE}
    if include_shadow:
        eligible_states.add(ProfileLifecycle.SHADOW)
    candidates = [item for item in profiles if item.lifecycle in eligible_states]
    scored = sorted(
        ((score_profile(item, observation)[0], item) for item in candidates),
        key=lambda item: (item[0], item[1].profile_version),
        reverse=True,
    )
    if not scored:
        return ProfileMatch(score=0, margin=0, selected=False, reasons=("no_profile_candidates",))
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    selected = best_score >= best.retrieval_threshold and margin >= best.retrieval_margin
    reasons = score_profile(best, observation)[1]
    if best_score < best.retrieval_threshold:
        reasons = (*reasons, "below_absolute_threshold")
    if margin < best.retrieval_margin:
        reasons = (*reasons, "below_candidate_margin")
    return ProfileMatch(
        profile_key=best.profile_key,
        profile_version=best.profile_version,
        score=best_score,
        margin=margin,
        selected=selected,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def profile_to_schema(profile: LayoutProfile, page_number: int, table_id: str) -> TableSchemaState:
    return TableSchemaState(
        source_page=page_number,
        source_table=table_id,
        table_type=profile.table_type,
        column_centers=dict(profile.column_centers),
        confidence=profile.metrics.precision if profile.metrics else profile.retrieval_threshold,
        header_token_ids=(),
        orientation="upright",
    )
