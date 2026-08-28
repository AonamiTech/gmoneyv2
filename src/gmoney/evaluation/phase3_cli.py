from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from gmoney.contracts.phase3 import GeminiPromotionDecision
from gmoney.evaluation.corpus import sha256_file
from gmoney.evaluation.metrics import row_view
from gmoney.evaluation.phase3 import Phase3Document, evaluate_phase3_quality

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    bootstrap_samples: int = 1_000,
) -> None:
    payload = json.loads(manifest.read_text())
    documents: list[Phase3Document] = []
    for entry in payload.get("documents", []):
        gold_payload = json.loads(Path(entry["gold"]).read_text()).get("rows", [])
        actual_document = json.loads(Path(entry["actual"]).read_text())
        actual_payload = actual_document.get("rows", actual_document)
        gold = tuple(row_view(row, index) for index, row in enumerate(gold_payload))
        actual = tuple(row_view(row, index) for index, row in enumerate(actual_payload))
        ungrounded = sum(
            row.get("review_disposition") == "accepted"
            and not {"description", "amount"}.issubset(row.get("field_evidence", {}))
            for row in actual_payload
        )
        provider = actual_document.get("provider_usage", {})
        aggregate_provider = provider.get("aggregate", provider)
        documents.append(
            Phase3Document(
                document_id=str(entry["document_id"]),
                hospital_id=str(entry["hospital_id"]) if entry.get("hospital_id") else None,
                cohort=str(entry["cohort"]),
                route=str(entry.get("route") or "unknown"),
                gold=gold,
                actual=actual,
                accepted_ungrounded_rows=ungrounded,
                privacy_failures=int(entry.get("privacy_failures") or 0),
                gemini_calls=int(aggregate_provider.get("gemini_calls") or 0),
                ordinary_active=bool(entry.get("ordinary_active")),
            )
        )
    result = evaluate_phase3_quality(tuple(documents), bootstrap_samples=bootstrap_samples)
    result_payload = result.to_dict()
    result_payload["evaluation_manifest_sha256"] = sha256_file(manifest)
    rendered = json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    typer.echo(json.dumps({"passed": result.passed, "blocking": result.blocking_reasons}))


@app.command("promote-gemini")
def promote_gemini(
    local_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    challenger_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    frozen_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    model: str = "gemini-3.5-flash",
    prompt_version: str = "phase3-grounded-v1",
    redaction_version: str = "phase3-redaction-v1",
) -> None:
    local = json.loads(local_report.read_text())
    challenger = json.loads(challenger_report.read_text())
    manifest_sha256 = sha256_file(frozen_manifest)
    if {
        local.get("evaluation_manifest_sha256"),
        challenger.get("evaluation_manifest_sha256"),
    } != {manifest_sha256}:
        raise typer.BadParameter("reports do not use the supplied frozen manifest")
    unseen = next(
        (item for item in challenger.get("cohorts", []) if item.get("cohort") == "unseen"),
        None,
    )
    if unseen is None:
        raise typer.BadParameter("challenger report has no unseen cohort")
    local_f1 = float(local["aggregate"]["canonical_f1"])
    challenger_f1 = float(challenger["aggregate"]["canonical_f1"])
    unseen_quality = unseen["quality"]
    unseen_precision = min(
        float(unseen_quality["legacy_precision"]),
        float(unseen_quality["canonical_precision"]),
    )
    unseen_recall = min(
        float(unseen_quality["legacy_recall"]),
        float(unseen_quality["canonical_recall"]),
    )
    approved = bool(
        challenger.get("passed")
        and challenger_f1 > local_f1
        and challenger_f1 >= 0.9
        and min(unseen_precision, unseen_recall) >= 0.85
        and not challenger["aggregate"]["accepted_ungrounded_rows"]
        and not challenger.get("privacy_failures")
    )
    decision = GeminiPromotionDecision(
        approved=approved,
        model=model,
        prompt_version=prompt_version,
        redaction_version=redaction_version,
        frozen_manifest_sha256=manifest_sha256,
        local_f1=local_f1,
        challenger_f1=challenger_f1,
        unseen_precision=unseen_precision,
        unseen_recall=unseen_recall,
        accepted_ungrounded_rows=int(
            challenger["aggregate"]["accepted_ungrounded_rows"]
        ),
        privacy_failures=int(challenger.get("privacy_failures") or 0),
        decided_at=datetime.now(UTC),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps({"approved": decision.approved}))


if __name__ == "__main__":
    app()
