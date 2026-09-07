from __future__ import annotations

from dataclasses import replace

import pytest

from gmoney.extraction.table_selection import (
    MATCH_ACCEPT_THRESHOLD,
    TableCandidateVariant,
    TableProposal,
    logical_table_id,
    match_logical_tables,
    normalize_header_tokens,
    select_stage_one,
    select_stage_two,
)

SOURCE = "a" * 64


def proposal(
    proposal_id: str,
    variant: TableCandidateVariant,
    box: tuple[float, float, float, float],
    *,
    order: int = 0,
    headers: tuple[str, ...] = ("Description", "Amount"),
    metrics: dict[str, int | bool] | None = None,
    distortion: float = 0.0,
) -> TableProposal:
    return TableProposal(
        proposal_id=proposal_id,
        source_sha256=SOURCE,
        page_number=1,
        page_width=1000,
        page_height=2000,
        variant=variant,
        source_box=box,
        reading_order=order,
        header_tokens=headers,
        table_type="ledger",
        page_artifact_sha256=(proposal_id[0] * 64),
        transform_valid=True,
        distortion=distortion,
        metrics=metrics or {},
    )


def test_header_normalization_and_logical_identity_are_stable() -> None:
    first = proposal("anchor", TableCandidateVariant.ORIENTED_RAW, (100, 200, 900, 1000))
    equivalent = replace(
        first,
        proposal_id="different-runtime-id",
        source_box=(100.2, 199.6, 899.7, 1000.4),
        header_tokens=(" DESCRIPTION ", "amount!!"),
    )

    assert normalize_header_tokens((" A/B ", "ＡＭＯＵＮＴ")) == ("a", "b", "amount")
    assert logical_table_id(first) == logical_table_id(equivalent)


def test_matches_each_derivative_branch_with_maximum_total_weight() -> None:
    anchors = (
        proposal("a1", TableCandidateVariant.ORIENTED_RAW, (0, 0, 400, 400), order=0),
        proposal("a2", TableCandidateVariant.ORIENTED_RAW, (600, 0, 1000, 400), order=1),
    )
    candidates = (
        proposal("p1", TableCandidateVariant.PROJECTIVE, (5, 5, 405, 405), order=0),
        proposal("p2", TableCandidateVariant.PROJECTIVE, (595, 5, 995, 405), order=1),
    )

    result = match_logical_tables((*anchors, *candidates))
    reordered = match_logical_tables(tuple(reversed((*anchors, *candidates))))

    accepted = {
        (edge.anchor_proposal_id, edge.candidate_proposal_id)
        for edge in result.edges
        if edge.accepted
    }
    assert accepted == {("a1", "p1"), ("a2", "p2")}
    assert reordered == result
    assert len(result.logical_tables) == 2
    assert all(len(table.proposal_ids) == 2 for table in result.logical_tables)


def test_ambiguous_competing_edges_abstain() -> None:
    anchors = (
        proposal("a1", TableCandidateVariant.ORIENTED_RAW, (0, 0, 500, 500)),
        proposal("a2", TableCandidateVariant.ORIENTED_RAW, (20, 0, 520, 500)),
    )
    candidate = proposal("p1", TableCandidateVariant.PROJECTIVE, (10, 0, 510, 500))

    result = match_logical_tables((*anchors, candidate))

    relevant = [
        edge
        for edge in result.edges
        if edge.features.weighted_score >= MATCH_ACCEPT_THRESHOLD
    ]
    assert relevant
    assert all(not edge.accepted and edge.reason == "abstained_ambiguous" for edge in relevant)


def test_derivative_only_table_requires_independent_branch_agreement() -> None:
    projective = proposal(
        "p1", TableCandidateVariant.PROJECTIVE, (100, 200, 900, 1000)
    )
    alone = match_logical_tables((projective,))
    assert alone.logical_tables == ()

    uvdoc = replace(projective, proposal_id="u1", variant=TableCandidateVariant.UVDOC)
    agreed = match_logical_tables((projective, uvdoc))
    assert len(agreed.logical_tables) == 1
    assert agreed.logical_tables[0].derivative_only is True


def test_stage_one_rejects_invalid_and_returns_only_top_two() -> None:
    raw = proposal(
        "raw",
        TableCandidateVariant.ORIENTED_RAW,
        (100, 200, 900, 1000),
        metrics={"financial_token_count": 4, "lineage_valid": True},
    )
    projective = replace(
        raw,
        proposal_id="projective",
        variant=TableCandidateVariant.PROJECTIVE,
        metrics={"financial_token_count": 8, "lineage_valid": True},
    )
    uvdoc = replace(
        raw,
        proposal_id="uvdoc",
        variant=TableCandidateVariant.UVDOC,
        metrics={"financial_token_count": 20, "lineage_valid": False},
    )

    selected = select_stage_one((raw, projective, uvdoc))

    assert tuple(item.proposal_id for item in selected) == ("projective", "raw")


def test_stage_two_prefers_quality_then_least_transformed_exact_tie() -> None:
    raw = proposal("raw", TableCandidateVariant.ORIENTED_RAW, (100, 200, 900, 1000))
    uvdoc = replace(raw, proposal_id="uvdoc", variant=TableCandidateVariant.UVDOC)
    tied = {
        "critical_error_count": 0,
        "required_column_count": 5,
        "grounded_complete_row_count": 10,
        "evidence_linkage_ppm": 1_000_000,
    }
    winner, _metrics = select_stage_two(((uvdoc, tied), (raw, tied)))
    assert winner is raw

    improved = {**tied, "grounded_complete_row_count": 11}
    winner, _metrics = select_stage_two(((raw, tied), (uvdoc, improved)))
    assert winner is uvdoc


def test_proposal_rejects_out_of_bounds_geometry() -> None:
    with pytest.raises(ValueError, match="exceeds SOURCE_RAW"):
        proposal("bad", TableCandidateVariant.ORIENTED_RAW, (0, 0, 1001, 100))
