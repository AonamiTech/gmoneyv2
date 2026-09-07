from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gmoney.contracts.v6 import M5ShadowProjectionV1
from gmoney.evaluation.authority_metrics import AuthorityDocumentReport
from gmoney.evaluation.m5_shadow import (
    project_m5_shadow_runs,
    summarize_authority_reports,
    verify_m5_shadow_projection,
    write_m5_shadow_projection,
)

SOURCE = "a" * 64


def _proposal(proposal_id: str, artifact: str) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "source_sha256": SOURCE,
        "page_number": 1,
        "variant": "oriented_raw" if proposal_id == "baseline" else "projective",
        "page_artifact_sha256": artifact,
        "source_box": [0, 0, 100, 100],
        "candidate_box": [0, 0, 100, 100],
        "reading_order": 0,
    }


def _reconstruction(proposal_id: str, artifact: str, value: str) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "source_sha256": SOURCE,
        "page_number": 1,
        "variant": "oriented_raw" if proposal_id == "baseline" else "projective",
        "artifact_sha256": artifact,
        "source_box": [0, 0, 100, 100],
        "source_tables": [{"id": proposal_id, "rows": [{"cells": [{"raw_value": value}]}]}],
        "diagnostics": {"value": value},
    }


def _run(
    *,
    tables: list[dict[str, object]],
    proposals: list[dict[str, object]] | None = None,
    page_number: int = 1,
) -> dict[str, object]:
    return {
        "policy_version": "table_magic_match_v1",
        "mode": "shadow",
        "status": "complete",
        "source_sha256": SOURCE,
        "page_number": page_number,
        "proposals": proposals
        or [_proposal("baseline", "b" * 64), _proposal("candidate", "c" * 64)],
        "logical_tables": tables,
        "edges": [],
    }


def _table(*, ambiguous: bool = False) -> dict[str, object]:
    return {
        "logical_table_id": "logical",
        "anchor_proposal_id": "baseline",
        "proposal_ids": ["baseline", "candidate"],
        "selected_proposal_id": "candidate",
        "baseline_proposal_id": "baseline",
        "ambiguous": ambiguous,
        "ambiguity_reason": "abstained_ambiguous" if ambiguous else None,
        "baseline_reconstruction": _reconstruction("baseline", "b" * 64, "baseline"),
        "selected_reconstruction": _reconstruction("candidate", "c" * 64, "candidate"),
    }


def test_projection_preserves_winning_whole_table_reconstruction() -> None:
    projection = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table()])],
    )

    table = projection.tables[0]
    assert table.decision == "selected"
    assert table.selected_proposal_id == "candidate"
    assert table.selected_reconstruction is not None
    assert table.selected_reconstruction.source_tables[0]["id"] == "candidate"
    assert table.selected_artifact_sha256 == "c" * 64
    assert projection.canonical_publication is False
    assert projection.authority == "non_canonical"


def test_ambiguity_falls_back_to_baseline_content() -> None:
    projection = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table(ambiguous=True)])],
    )

    table = projection.tables[0]
    assert table.decision == "baseline_fallback"
    assert table.selected_proposal_id == "baseline"
    assert table.selected_reconstruction is not None
    assert table.selected_reconstruction.source_tables[0]["id"] == "baseline"


def test_lost_and_duplicate_content_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate proposal"):
        project_m5_shadow_runs(
            document_id=SOURCE,
            source_sha256=SOURCE,
            source_name="bill.pdf",
            runs=[
                _run(
                    tables=[_table()],
                    proposals=[_proposal("baseline", "b" * 64), _proposal("baseline", "b" * 64)],
                )
            ],
        )

    table = _table()
    table["selected_reconstruction"] = None
    with pytest.raises(ValueError, match="lost winning"):
        project_m5_shadow_runs(
            document_id=SOURCE,
            source_sha256=SOURCE,
            source_name="bill.pdf",
            runs=[_run(tables=[table])],
        )


def test_digest_tampering_is_rejected(tmp_path: Path) -> None:
    projection = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table()])],
    )
    path = write_m5_shadow_projection(tmp_path, projection)
    payload = json.loads(path.read_text())
    payload["pages"][0]["tables"][0]["selected_reconstruction"]["diagnostics"]["value"] = "tampered"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="digest"):
        verify_m5_shadow_projection(path)


def test_projection_order_is_deterministic_and_summary_uses_na() -> None:
    first = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table()])],
    )
    second = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table()])],
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    report = AuthorityDocumentReport(
        document_key="bill",
        gold_identity_sha256=SOURCE,
        actual_identity_sha256=SOURCE,
        pair_identity_sha256=SOURCE,
        metrics={
            "critical_exact": 3,
            "critical_actual": 4,
            "critical_gold": 5,
        },
    )
    summary = summarize_authority_reports([report])
    assert summary["metrics"]["critical_numeric_cell_precision"] == 0.75
    assert summary["metrics"]["critical_numeric_cell_recall"] == 0.6

    empty_summary = summarize_authority_reports(
        [
            report.__class__(
                document_key="empty",
                gold_identity_sha256=SOURCE,
                actual_identity_sha256=SOURCE,
                pair_identity_sha256=SOURCE,
                metrics={},
            )
        ]
    )
    assert empty_summary["metrics"]["critical_numeric_cell_precision"] == "N/A"
    assert "critical_numeric_cell_precision" in empty_summary["zero_denominator_metrics"]


def test_projection_json_round_trip_revalidates() -> None:
    projection = project_m5_shadow_runs(
        document_id=SOURCE,
        source_sha256=SOURCE,
        source_name="bill.pdf",
        runs=[_run(tables=[_table()])],
    )
    restored = M5ShadowProjectionV1.model_validate_json(
        json.dumps(projection.model_dump(mode="json"))
    )
    assert restored == projection


def test_expected_page_count_rejects_silent_page_omission() -> None:
    with pytest.raises(ValidationError, match="incomplete"):
        project_m5_shadow_runs(
            document_id=SOURCE,
            source_sha256=SOURCE,
            source_name="bill.pdf",
            runs=[_run(tables=[_table()])],
            expected_page_count=2,
        )
