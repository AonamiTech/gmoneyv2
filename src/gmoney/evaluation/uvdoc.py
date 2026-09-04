"""M4 UVDoc shadow gate bound to preregistered corpus and evaluator evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import Field, model_validator

from gmoney.contracts.common import ContractModel
from gmoney.demo.store import JobStore
from gmoney.evaluation.corpus import sha256_file
from gmoney.inference.uvdoc import UvdocPreregistration, load_preregistration

app = typer.Typer(no_args_is_help=True)
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class UvdocAccuracyObservation(ContractModel):
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    cohort: Literal["curved", "flat"]
    branch: Literal["UVDOC", "UVDOC_ENHANCED"]
    improved_failure_ids: tuple[str, ...] = ()
    new_critical_error_ids: tuple[str, ...] = ()
    lost_correct_ids: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)


class UvdocAccuracyReport(ContractModel):
    report_version: Literal["gmoney_uvdoc_accuracy_v1"] = "gmoney_uvdoc_accuracy_v1"
    release_corpus_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    audited_gold_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_name: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    evaluator_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=SHA256_PATTERN)
    observations: tuple[UvdocAccuracyObservation, ...]

    @model_validator(mode="after")
    def require_unique_observations(self) -> UvdocAccuracyReport:
        keys = {(item.source_sha256, item.page_number, item.branch) for item in self.observations}
        if len(keys) != len(self.observations):
            raise ValueError("UVDoc accuracy observations must be unique")
        return self


def _accuracy_identity_matches(
    preregistration: UvdocPreregistration, report: UvdocAccuracyReport
) -> bool:
    return (
        preregistration.release_corpus_manifest_sha256 == report.release_corpus_manifest_sha256
        and preregistration.audited_gold_sha256 == report.audited_gold_sha256
        and preregistration.evaluator_name == report.evaluator_name
        and preregistration.evaluator_version == report.evaluator_version
        and preregistration.evaluator_sha256 == report.evaluator_sha256
        and preregistration.model_sha256 == report.model_sha256
        and preregistration.model_config_sha256 == report.model_config_sha256
        and preregistration.adapter_config_sha256 == report.adapter_config_sha256
    )


def _shadow_inventory(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    store = JobStore(data_root)
    inventory: dict[str, list[dict[str, Any]]] = {}
    for state in store.states():
        job_id = str(state["id"])
        try:
            result = store.read_result(job_id)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        source_sha256 = result.get("source_sha256")
        if not isinstance(source_sha256, str):
            continue
        inventory.setdefault(source_sha256, []).append(result)
    return inventory


def evaluate_uvdoc_gate(
    preregistration: UvdocPreregistration,
    data_root: Path,
    accuracy: UvdocAccuracyReport | None,
) -> dict[str, Any]:
    inventory = _shadow_inventory(data_root)
    shadow_failures: list[dict[str, Any]] = []
    for target in preregistration.targets:
        candidates = inventory.get(target.source_sha256, [])
        if len(candidates) != 1:
            shadow_failures.append(
                {
                    "source_sha256": target.source_sha256,
                    "page_number": target.page_number,
                    "reason": (
                        "candidate_workspace_missing"
                        if not candidates
                        else "multiple_candidate_workspaces"
                    ),
                }
            )
            continue
        runs = [
            item
            for item in candidates[0].get("uvdoc_shadow_runs") or []
            if int(item.get("page_number") or 0) == target.page_number
        ]
        if len(runs) != 1 or runs[0].get("status") != "valid":
            shadow_failures.append(
                {
                    "source_sha256": target.source_sha256,
                    "page_number": target.page_number,
                    "reason": (
                        "uvdoc_shadow_run_missing"
                        if not runs
                        else str(runs[0].get("reason_code") or "uvdoc_shadow_invalid")
                    ),
                }
            )

    accuracy_failures: list[str] = []
    curved_improvements: set[str] = set()
    if accuracy is None:
        accuracy_failures.append("authoritative_accuracy_report_missing")
    elif not _accuracy_identity_matches(preregistration, accuracy):
        accuracy_failures.append("accuracy_identity_mismatch")
    else:
        expected = {
            (target.source_sha256, target.page_number) for target in preregistration.targets
        }
        primary = {
            (item.source_sha256, item.page_number): item
            for item in accuracy.observations
            if item.branch == "UVDOC"
        }
        if set(primary) != expected:
            accuracy_failures.append("primary_observation_set_mismatch")
        for target in preregistration.targets:
            observation = primary.get((target.source_sha256, target.page_number))
            if observation is None or observation.cohort != target.cohort:
                continue
            if observation.new_critical_error_ids:
                accuracy_failures.append("new_critical_error")
            if target.cohort == "flat" and observation.lost_correct_ids:
                accuracy_failures.append("flat_previously_correct_regression")
            if target.cohort == "curved":
                curved_improvements.update(
                    set(observation.improved_failure_ids) & set(target.failure_ids)
                )
        if not curved_improvements:
            accuracy_failures.append("no_preregistered_curved_improvement")

    passed = not shadow_failures and not accuracy_failures
    return {
        "report_version": "gmoney_uvdoc_gate_v1",
        "status": "shadow" if passed else "hold",
        "passed": passed,
        "shadow_failures": shadow_failures,
        "accuracy_failures": sorted(set(accuracy_failures)),
        "improved_curved_failure_ids": sorted(curved_improvements),
    }


@app.command("evaluate")
def evaluate(
    preregistration_path: Annotated[Path, typer.Option("--preregistration")],
    release_corpus_manifest: Annotated[Path, typer.Option("--release-corpus-manifest")],
    audited_gold: Annotated[Path, typer.Option("--audited-gold")],
    evaluator: Annotated[Path, typer.Option("--evaluator")],
    data_root: Annotated[Path, typer.Option("--data-root")],
    output: Annotated[Path, typer.Option("--output")],
    accuracy_report: Annotated[Path | None, typer.Option("--accuracy-report")] = None,
) -> None:
    preregistration, preregistration_sha256 = load_preregistration(preregistration_path)
    actual = {
        "release_corpus_manifest_sha256": sha256_file(release_corpus_manifest),
        "audited_gold_sha256": sha256_file(audited_gold),
        "evaluator_sha256": sha256_file(evaluator),
    }
    expected = {
        "release_corpus_manifest_sha256": preregistration.release_corpus_manifest_sha256,
        "audited_gold_sha256": preregistration.audited_gold_sha256,
        "evaluator_sha256": preregistration.evaluator_sha256,
    }
    if actual != expected:
        raise typer.BadParameter("preregistered corpus, gold, or evaluator digest mismatch")
    accuracy = (
        UvdocAccuracyReport.model_validate_json(accuracy_report.read_bytes())
        if accuracy_report is not None
        else None
    )
    report = evaluate_uvdoc_gate(preregistration, data_root, accuracy)
    report["preregistration_sha256"] = preregistration_sha256
    report["accuracy_report_sha256"] = (
        hashlib.sha256(accuracy_report.read_bytes()).hexdigest()
        if accuracy_report is not None
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise typer.Exit(1)


__all__ = [
    "UvdocAccuracyObservation",
    "UvdocAccuracyReport",
    "evaluate_uvdoc_gate",
]
