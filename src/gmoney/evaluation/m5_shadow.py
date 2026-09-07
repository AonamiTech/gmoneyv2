"""Evaluation-only projection and verification for M5 shadow decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.contracts.v6 import (
    M5ShadowPageProjectionV1,
    M5ShadowProjectionV1,
    M5ShadowProposalV1,
    M5ShadowReconstructionV1,
    M5ShadowTableProjectionV1,
)
from gmoney.evaluation.authority_metrics import AuthorityDocumentReport, evaluate_document
from gmoney.extraction.table_selection import MATCH_POLICY_VERSION

M5_SHADOW_PROJECTION_RELATIVE_PATH = Path("evaluation/m5-shadow-projection-v1.json")
_RATIO_PAIRS = {
    "critical_numeric_cell_precision": ("critical_exact", "critical_actual"),
    "critical_numeric_cell_recall": ("critical_exact", "critical_gold"),
    "line_item_row_recall": ("line_item_matched_rows", "line_item_gold_rows"),
    "correct_column_assignment": ("correctly_assigned_cells", "gold_cells"),
    "header_schema_accuracy": ("matched_columns", "gold_columns"),
    "grand_total_exact_accuracy": ("grand_total_exact", "grand_total_gold"),
    "cell_exact_match_accuracy": ("exact_cells", "gold_cells"),
    "table_recall": ("matched_tables", "gold_tables"),
    "row_recall": ("matched_rows", "gold_rows"),
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python")
    raise TypeError(f"expected mapping or model, got {type(value).__name__}")


def _reconstruction(value: Any) -> M5ShadowReconstructionV1 | None:
    if value is None:
        return None
    return M5ShadowReconstructionV1.model_validate(value)


def _proposal(
    value: Mapping[str, Any],
    *,
    source_sha256: str,
    page_number: int,
) -> M5ShadowProposalV1:
    data = dict(value)
    data.setdefault("source_sha256", source_sha256)
    data.setdefault("page_number", page_number)
    if "artifact_sha256" not in data:
        data["artifact_sha256"] = data.get("page_artifact_sha256")
    data.pop("page_artifact_sha256", None)
    for key in (
        "header_tokens",
        "table_type",
        "transform_valid",
        "distortion",
        "stage_one_metrics",
        "stage_two_metrics",
    ):
        data.pop(key, None)
    return M5ShadowProposalV1.model_validate(data)


def _page_projection(
    raw_run: Mapping[str, Any],
    *,
    source_sha256: str,
) -> M5ShadowPageProjectionV1:
    page_number = int(raw_run["page_number"])
    status = str(raw_run.get("status", "failed"))
    if status != "complete":
        return M5ShadowPageProjectionV1(
            page_number=page_number,
            status="failed",
            reason=str(raw_run.get("reason") or "m5_shadow_run_failed"),
        )

    raw_proposals = tuple(raw_run.get("proposals") or ())
    proposals = tuple(
        sorted(
            (
                _proposal(
                    _as_mapping(item),
                    source_sha256=source_sha256,
                    page_number=page_number,
                )
                for item in raw_proposals
            ),
            key=lambda item: item.proposal_id,
        )
    )
    proposal_ids = {item.proposal_id for item in proposals}
    if len(proposal_ids) != len(proposals):
        raise ValueError("M5 shadow projection rejects duplicate proposal IDs")
    proposal_by_id = {item.proposal_id: item for item in proposals}

    edges = tuple(raw_run.get("edges") or ())
    tables: list[M5ShadowTableProjectionV1] = []
    for raw_table in sorted(
        (_as_mapping(item) for item in raw_run.get("logical_tables") or ()),
        key=lambda item: str(item.get("logical_table_id", "")),
    ):
        logical_id = str(raw_table["logical_table_id"])
        table_proposal_ids = tuple(str(item) for item in raw_table.get("proposal_ids") or ())
        if len(set(table_proposal_ids)) != len(table_proposal_ids):
            raise ValueError("M5 shadow projection rejects duplicate table proposals")
        if set(table_proposal_ids) - proposal_ids:
            raise ValueError("M5 shadow projection rejects a lost proposal lineage")
        lineage = tuple(proposal_by_id[item] for item in sorted(table_proposal_ids))
        anchor_id = raw_table.get("anchor_proposal_id")
        baseline_id = raw_table.get("baseline_proposal_id") or anchor_id
        selected_id = raw_table.get("selected_proposal_id")
        edge_ambiguity = any(
            str(_as_mapping(edge).get("anchor_proposal_id")) == str(anchor_id)
            and str(_as_mapping(edge).get("reason")) == "abstained_ambiguous"
            for edge in edges
        )
        ambiguous = bool(raw_table.get("ambiguous")) or edge_ambiguity
        ambiguity_reason = raw_table.get("ambiguity_reason")
        if ambiguous and not ambiguity_reason:
            ambiguity_reason = "abstained_ambiguous"

        baseline_reconstruction = _reconstruction(raw_table.get("baseline_reconstruction"))
        selected_reconstruction = _reconstruction(raw_table.get("selected_reconstruction"))
        if ambiguous:
            if baseline_id is None:
                raise ValueError("M5 ambiguity cannot fall back without a baseline proposal")
            if baseline_reconstruction is None:
                raise ValueError("M5 shadow projection rejects a lost baseline reconstruction")
            selected_id = baseline_id
            selected_reconstruction = baseline_reconstruction
            decision = "baseline_fallback"
        elif selected_id is None:
            selected_reconstruction = None
            decision = "abstained"
        else:
            if selected_reconstruction is None:
                raise ValueError("M5 shadow projection rejects a lost winning reconstruction")
            decision = "selected"
        if selected_id is not None and selected_id not in proposal_by_id:
            raise ValueError("M5 selected proposal is missing from lineage")

        baseline_proposal = proposal_by_id.get(str(baseline_id)) if baseline_id else None
        selected_proposal = proposal_by_id.get(str(selected_id)) if selected_id else None
        tables.append(
            M5ShadowTableProjectionV1(
                logical_table_id=logical_id,
                source_sha256=source_sha256,
                page_number=page_number,
                anchor_proposal_id=str(anchor_id) if anchor_id is not None else None,
                proposal_ids=tuple(sorted(table_proposal_ids)),
                proposals=lineage,
                baseline_proposal_id=str(baseline_id) if baseline_id is not None else None,
                selected_proposal_id=str(selected_id) if selected_id is not None else None,
                baseline_artifact_sha256=(
                    baseline_proposal.artifact_sha256 if baseline_proposal else None
                ),
                selected_artifact_sha256=(
                    selected_proposal.artifact_sha256 if selected_proposal else None
                ),
                baseline_variant=baseline_proposal.variant if baseline_proposal else None,
                selected_variant=selected_proposal.variant if selected_proposal else None,
                decision=decision,
                ambiguous=ambiguous,
                ambiguity_reason=(str(ambiguity_reason) if ambiguity_reason else None),
                selected_metrics=dict(raw_table.get("selected_metrics") or {}),
                candidate_ranking=tuple(raw_table.get("candidate_ranking") or ()),
                baseline_reconstruction=baseline_reconstruction,
                selected_reconstruction=selected_reconstruction,
            )
        )
    return M5ShadowPageProjectionV1(
        page_number=page_number,
        status="complete",
        tables=tuple(tables),
    )


def project_m5_shadow_runs(
    *,
    document_id: str,
    source_sha256: str,
    source_name: str,
    runs: Sequence[Mapping[str, Any]],
    expected_page_count: int | None = None,
) -> M5ShadowProjectionV1:
    """Project raw M5 page runs in deterministic order into the sidecar contract."""

    ordered_runs = sorted(
        (_as_mapping(run) for run in runs),
        key=lambda run: int(run.get("page_number", 0)),
    )
    pages = tuple(
        _page_projection(run, source_sha256=source_sha256) for run in ordered_runs
    )
    if len({page.page_number for page in pages}) != len(pages):
        raise ValueError("M5 shadow projection rejects duplicate page runs")
    if any(
        str(run.get("source_sha256", source_sha256)) != source_sha256
        for run in ordered_runs
    ):
        raise ValueError("M5 shadow projection source hash mismatch")
    policy_version = next(
        (
            str(run["policy_version"])
            for run in ordered_runs
            if run.get("policy_version")
        ),
        MATCH_POLICY_VERSION,
    )
    return M5ShadowProjectionV1(
        document_id=document_id,
        source_sha256=source_sha256,
        source_name=source_name,
        policy_version=policy_version,
        expected_page_count=expected_page_count,
        pages=pages,
    )


def projection_path(artifact_root: Path) -> Path:
    """Return the sidecar path without adding it to the canonical result."""

    return artifact_root / M5_SHADOW_PROJECTION_RELATIVE_PATH


def write_m5_shadow_projection(
    artifact_root: Path,
    projection: M5ShadowProjectionV1,
) -> Path:
    """Atomically write a canonical-JSON sidecar and return its path."""

    destination = projection_path(artifact_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(projection.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def verify_m5_shadow_projection(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> M5ShadowProjectionV1:
    """Read and validate a sidecar, including its internal content digest."""

    projection = M5ShadowProjectionV1.model_validate_json(path.read_text(encoding="utf-8"))
    if expected_sha256 is not None and projection.projection_sha256 != expected_sha256:
        raise ValueError("M5 shadow projection digest differs from expected digest")
    return projection


def evaluate_m5_shadow_projection(
    gold: Any,
    projection: M5ShadowProjectionV1,
) -> AuthorityDocumentReport:
    """Score a verified sidecar with the existing authority evaluator."""

    return evaluate_document(
        gold,
        projection.authority_document(),
        document_key=projection.document_id,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | str:
    return round(float(numerator) / float(denominator), 12) if denominator else "N/A"


def summarize_authority_reports(
    reports: Sequence[AuthorityDocumentReport],
) -> dict[str, Any]:
    """Pool authority reports with explicit N/A zero-denominator metrics."""

    ordered = tuple(sorted(reports, key=lambda report: report.document_key))
    count_keys = {
        key
        for pair in _RATIO_PAIRS.values()
        for key in pair
    }
    counts: dict[str, int | float] = {}
    for report in ordered:
        for key, value in report.metrics.items():
            if (
                key not in count_keys
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            counts[key] = counts.get(key, 0) + value
    metrics: dict[str, float | str] = {
        name: _ratio(counts.get(numerator, 0), counts.get(denominator, 0))
        for name, (numerator, denominator) in _RATIO_PAIRS.items()
    }
    return {
        "document_count": len(ordered),
        "document_keys": [report.document_key for report in ordered],
        "counts": dict(sorted(counts.items())),
        "metrics": metrics,
        "zero_denominator_metrics": sorted(
            name for name, value in metrics.items() if value == "N/A"
        ),
    }


app = typer.Typer(no_args_is_help=True)


@app.command("verify")
def verify_command(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    expected_sha256: Annotated[str | None, typer.Option()] = None,
) -> None:
    projection = verify_m5_shadow_projection(path, expected_sha256=expected_sha256)
    typer.echo(
        json.dumps(
            {
                "projection_sha256": projection.projection_sha256,
                "document_id": projection.document_id,
                "pages": len(projection.pages),
                "tables": len(projection.tables),
                "authority": projection.authority,
                "canonical_publication": projection.canonical_publication,
            },
            sort_keys=True,
        )
    )


@app.command("summary")
def summary_command(
    paths: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
) -> None:
    projections = tuple(verify_m5_shadow_projection(path) for path in paths)
    table_count = sum(len(projection.tables) for projection in projections)
    fallback_count = sum(
        table.decision == "baseline_fallback"
        for projection in projections
        for table in projection.tables
    )
    selected_count = sum(
        table.decision == "selected"
        for projection in projections
        for table in projection.tables
    )
    denominator = table_count
    payload = {
        "document_count": len(projections),
        "table_count": table_count,
        "selected_tables": selected_count,
        "baseline_fallback_tables": fallback_count,
        "selected_table_ratio": _ratio(selected_count, denominator),
        "fallback_table_ratio": _ratio(fallback_count, denominator),
        "projection_sha256s": sorted(item.projection_sha256 for item in projections),
    }
    typer.echo(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    app()


__all__ = [
    "M5_SHADOW_PROJECTION_RELATIVE_PATH",
    "evaluate_m5_shadow_projection",
    "project_m5_shadow",
    "project_m5_shadow_runs",
    "projection_path",
    "summarize_authority_reports",
    "verify_projection",
    "verify_m5_shadow_projection",
    "write_m5_shadow_projection",
]

project_m5_shadow = project_m5_shadow_runs
verify_projection = verify_m5_shadow_projection
