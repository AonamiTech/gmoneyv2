from __future__ import annotations

import pytest
from pydantic import ValidationError

from gmoney.contracts.v6 import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    ExtractionResultV6,
    IdentityMapping,
    LogicalTableSelection,
    PageArtifact,
    TableCandidateScore,
    TableCandidateVariant,
    TableMatchEdge,
    TableSelectionRun,
)
from gmoney.settings import Settings


def _artifact(kind: ArtifactKind, image: str, *, parent: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=kind,
        image_sha256=image * 64,
        artifact_relative_path=f"artifacts/{image}.png",
        width=100,
        height=100,
        parent_artifact_id=parent,
        producer="test",
        producer_version="1",
        configuration_sha256="c" * 64,
        child_to_parent_mapping=IdentityMapping(),
    )


def test_table_selection_mode_defaults_off_and_has_three_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings(_env_file=None).table_selection_mode == "off"
    for mode in ("off", "shadow", "enabled"):
        monkeypatch.setenv("GMONEY_TABLE_SELECTION_MODE", mode)
        assert Settings(_env_file=None).table_selection_mode == mode


def test_v6_m5_fields_are_default_empty_for_old_payloads() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    envelope = ExtractionResultV6(
        document_id="doc",
        source_sha256="e" * 64,
        source_name="bill.pdf",
        pages=1,
        artifact_manifest=ArtifactManifest(artifacts=(source,)),
        page_artifacts=(PageArtifact(artifact=source, page_number=1, dpi=300),),
    )
    assert envelope.table_match_edges == ()
    assert envelope.table_candidate_scores == ()
    assert envelope.table_selection_runs == ()


def test_m5_selection_is_whole_table_and_matches_artifact_variant() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    oriented = _artifact(ArtifactKind.ORIENTED_RAW, "b", parent=source.artifact_id)
    manifest = ArtifactManifest(artifacts=(source, oriented))
    score = TableCandidateScore(
        logical_table_id="p1-t1",
        page_number=1,
        candidate_id="oriented-proposal",
        candidate_artifact_id=oriented.artifact_id,
        candidate_variant="oriented_raw",
        score=0.9,
        selected=True,
    )
    table = LogicalTableSelection(
        logical_table_id="p1-t1",
        page_number=1,
        candidate_evaluations=(score,),
        decision="selected",
        selected_candidate_id=score.candidate_id,
        selected_candidate_variant=TableCandidateVariant.ORIENTED_RAW,
        selected_artifact_id=oriented.artifact_id,
    )
    run = TableSelectionRun(
        document_id="doc",
        source_sha256="e" * 64,
        mode="shadow",
        logical_tables=(table,),
    )
    envelope = ExtractionResultV6(
        document_id="doc",
        source_sha256="e" * 64,
        source_name="bill.pdf",
        pages=1,
        artifact_manifest=manifest,
        page_artifacts=(
            PageArtifact(artifact=source, page_number=1, dpi=300),
            PageArtifact(artifact=oriented, page_number=1, dpi=300, role="ORIENTED_RAW"),
        ),
        table_selection_runs=(run,),
    )
    assert envelope.table_selection_runs[0].tables[0].whole_table is True


def test_shadow_run_can_introduce_stable_logical_id_without_rewriting_legacy_table() -> None:
    source = _artifact(ArtifactKind.SOURCE_RAW, "a")
    oriented = _artifact(ArtifactKind.ORIENTED_RAW, "b", parent=source.artifact_id)
    score = TableCandidateScore(
        logical_table_id="f" * 64,
        page_number=1,
        candidate_id="oriented-proposal",
        candidate_artifact_id=oriented.artifact_id,
        candidate_variant="ORIENTED_RAW",
        score=1,
        selected=True,
        evaluation_status="selected",
    )
    run = TableSelectionRun(
        document_id="doc",
        source_sha256="e" * 64,
        mode="shadow",
        logical_tables=(
            LogicalTableSelection(
                logical_table_id="f" * 64,
                page_number=1,
                candidate_evaluations=(score,),
                decision="selected",
                selected_candidate_id=score.candidate_id,
                selected_candidate_variant=score.candidate_variant,
                selected_artifact_id=oriented.artifact_id,
            ),
        ),
    )

    envelope = ExtractionResultV6(
        document_id="doc",
        source_sha256="e" * 64,
        source_name="bill.pdf",
        pages=1,
        artifact_manifest=ArtifactManifest(artifacts=(source, oriented)),
        page_artifacts=(
            PageArtifact(artifact=source, page_number=1, dpi=300),
            PageArtifact(artifact=oriented, page_number=1, dpi=300, role="ORIENTED_RAW"),
        ),
        table_selection_runs=(run,),
    )

    assert envelope.table_selection_runs[0].tables[0].logical_table_id == "f" * 64


def test_m5_rejects_ambiguous_winner_and_uvdoc_winner() -> None:
    with pytest.raises(ValidationError, match="ambiguity reason"):
        TableMatchEdge(
            logical_table_id="p1-t1",
            anchor_proposal_id="a",
            candidate_proposal_id="b",
            anchor_artifact_id="a" * 64,
            candidate_artifact_id="b" * 64,
            candidate_variant="PROJECTIVE",
            match_score=0.5,
            decision="ambiguous",
            ambiguous=True,
        )

    table = LogicalTableSelection(
            logical_table_id="p1-t1",
            page_number=1,
            candidate_evaluations=(
                TableCandidateScore(
                    logical_table_id="p1-t1",
                    candidate_id="uvdoc",
                    candidate_artifact_id="a" * 64,
                    candidate_variant="UVDOC",
                    score=1,
                    selected=True,
                ),
            ),
            decision="selected",
            selected_candidate_id="uvdoc",
            selected_candidate_variant="UVDOC",
            selected_artifact_id="a" * 64,
        )
    with pytest.raises(ValidationError, match="UVDoc shadow"):
        TableSelectionRun(
            document_id="doc",
            mode="enabled",
            logical_tables=(table,),
        )
