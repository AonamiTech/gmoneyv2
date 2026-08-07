from __future__ import annotations

import re
from collections import Counter
from statistics import median

from gmoney.contracts.phase3 import LayoutObservation, LayoutProfile


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _common_headers(
    observations: tuple[LayoutObservation, ...], threshold: float
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for observation in observations:
        counts.update(
            {
                _normalized(value)
                for value in observation.header_tokens
                if _normalized(value)
            }
        )
    minimum = max(1, round(len(observations) * threshold + 0.499999))
    return tuple(sorted(value for value, count in counts.items() if count >= minimum))


def _stable_anchors(observations: tuple[LayoutObservation, ...]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for observation in observations:
        words = {
            word
            for header in observation.header_tokens
            for word in _normalized(header).split()
            if len(word) > 2
        }
        counts.update(words)
    minimum = max(1, round(len(observations) * 0.8 + 0.499999))
    return tuple(sorted(word for word, count in counts.items() if count >= minimum))


def build_profile(
    observations: tuple[LayoutObservation, ...],
    *,
    profile_key: str,
    profile_version: int,
    construction_dataset_ids: tuple[str, ...],
    hospital_id: str | None = None,
    hospital_name: str | None = None,
    global_family: str | None = None,
    retrieval_threshold: float = 0.8,
    retrieval_margin: float = 0.1,
) -> LayoutProfile:
    """Construct declarative median geometry from construction-only observations."""
    if not observations:
        raise ValueError("profile construction needs observations")
    if not construction_dataset_ids:
        raise ValueError("profile construction needs dataset provenance")
    if len({item.document_id for item in observations}) != len(observations):
        raise ValueError("profile construction observations must use independent documents")
    if not hospital_id and not global_family:
        raise ValueError("profile construction needs a hospital or global family")
    if hospital_id and any(item.hospital_id != hospital_id for item in observations):
        raise ValueError("construction observation hospital identity mismatch")
    first = observations[0]
    if any(item.page_type != first.page_type for item in observations):
        raise ValueError("profile construction cannot mix page types")
    if any(item.table_type != first.table_type for item in observations):
        raise ValueError("profile construction cannot mix table types")

    roles = set.intersection(*(set(item.column_centers) for item in observations))
    centers = {
        role: median(item.column_centers[role] for item in observations) for role in sorted(roles)
    }
    tolerances: dict[str, float] = {}
    for role, center in centers.items():
        deviations = [abs(item.column_centers[role] - center) for item in observations]
        default = 0.04 if role == "amount" else 0.06
        tolerances[role] = min(0.2, max(0.02, default, max(deviations) * 2 + 0.01))
    boxes = tuple(zip(*(item.table_box for item in observations), strict=True))
    table_box = tuple(median(coordinate) for coordinate in boxes)
    quality_flags = tuple(sorted({flag for item in observations for flag in item.quality_flags}))
    return LayoutProfile(
        contract_version="layout_profile_v1",
        profile_key=profile_key,
        profile_version=profile_version,
        hospital_id=hospital_id,
        hospital_name=hospital_name.strip() if hospital_name else None,
        global_family=global_family,
        page_type=first.page_type,
        table_type=first.table_type,
        page_aspect_ratio=median(item.page_aspect_ratio for item in observations),
        table_box=table_box,
        header_tokens=_common_headers(observations, 0.5),
        stable_anchors=_stable_anchors(observations),
        column_centers=centers,
        column_tolerances=tolerances,
        supported_fields=tuple(sorted({"description", *centers})),
        quality_flags=quality_flags,
        retrieval_threshold=retrieval_threshold,
        retrieval_margin=retrieval_margin,
        construction_dataset_ids=construction_dataset_ids,
    )
