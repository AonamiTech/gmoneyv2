from __future__ import annotations

import pytest

from gmoney.contracts.authority import (
    BaselineDocumentV1,
    BaselineManifestV1,
    EvaluatorManifestV1,
    GateCohortResultV1,
    GateDecision,
    GoldDocumentV2,
    MetricObservationV1,
    SourceClass,
    TableKind,
    TotalKind,
    source_document_id_for,
)
from gmoney.evaluation import authority_metrics, uvdoc_authority
from gmoney.evaluation.uvdoc_authority import (
    PROMOTE_MEANS_SHADOW_ONLY,
    TABLE_MAGIC_FLOORS,
    DownstreamAblationProofV2,
    FloorResultV2,
    UvdocAccuracyReportV2,
    UvdocPreregistrationV2,
    evaluate_uvdoc_accuracy_v2,
    evaluate_uvdoc_gate_v2,
)


def _h(value: str) -> str:
    return value * 64


def _authority_cohorts(*, passed: bool = True) -> tuple[GateCohortResultV1, ...]:
    return tuple(
        GateCohortResultV1(
            cohort=cohort,
            required_count=count,
            observed_count=count,
            passed=passed,
            metrics=tuple(
                MetricObservationV1(
                    metric_name=metric_name,
                    value=1.0,
                    numerator=1,
                    denominator=1,
                    cohort=cohort,
                )
                for metric_name in TABLE_MAGIC_FLOORS
            ),
            blocking_reasons=() if passed else ("fixture_hold",),
        )
        for cohort, count in (
            ("production14", 14),
            ("passing36", 36),
            ("staging159", 159),
        )
    )


def _polygon(left: int, top: int, right: int, bottom: int) -> dict[str, object]:
    return {
        "points": [
            {"x": left, "y": top},
            {"x": right, "y": top},
            {"x": right, "y": bottom},
            {"x": left, "y": bottom},
        ]
    }


def _gold(source: str) -> GoldDocumentV2:
    payload: dict[str, object] = {
        "source_sha256": source,
        "source_manifest_sha256": _h("a"),
        "page_count": 1,
        "pages": [
            {
                "source_sha256": source,
                "page_number": 1,
                "artifact_sha256": _h("b"),
                "width": 100,
                "height": 100,
                "dpi": 300,
                "source_class": SourceClass.FLAT_SCAN,
                "tables": [
                    {
                        "table_kind": TableKind.ITEM_LEDGER,
                        "polygon": _polygon(0, 0, 100, 100),
                        "columns": [
                            {
                                "canonical_field": "description",
                                "polygon": _polygon(0, 0, 50, 100),
                            },
                            {
                                "canonical_field": "amount",
                                "polygon": _polygon(50, 0, 100, 100),
                            },
                        ],
                        "rows": [
                            {
                                "polygon": _polygon(0, 0, 100, 50),
                                "cells": [
                                    {
                                        "canonical_field": "description",
                                        "polygon": _polygon(0, 0, 50, 50),
                                        "raw_value": "Dressing",
                                    },
                                    {
                                        "canonical_field": "amount",
                                        "polygon": _polygon(50, 0, 100, 50),
                                        "raw_value": "₹10.00",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        "annotation_group_id": "uvdoc-authority-test",
        "split": "holdout",
    }
    seed = GoldDocumentV2.model_validate(payload)
    table_id = seed.pages[0].tables[0].table_id
    payload["totals"] = [
        {
            "source_sha256": source,
            "total_order": 0,
            "scope_id": table_id,
            "total_kind": TotalKind.SUBTOTAL,
            "polygon": _polygon(0, 50, 100, 75),
            "raw_value": "₹10.00",
        },
        {
            "source_sha256": source,
            "total_order": 1,
            "scope_id": seed.document_id,
            "total_kind": TotalKind.DOCUMENT_TOTAL,
            "polygon": _polygon(0, 75, 100, 100),
            "raw_value": "₹10.00",
        },
    ]
    return GoldDocumentV2.model_validate(payload)


def _actual(gold: GoldDocumentV2, *, subtotal: str = "10.00") -> dict[str, object]:
    source = gold.source_sha256
    table = gold.pages[0].tables[0]
    return {
        "source_sha256": source,
        "auto_accepted": True,
        "tables": [
            {
                "table_kind": "item_ledger",
                "polygon": [0, 0, 100, 100],
                "columns": [
                    {"canonical_field": "description", "polygon": [0, 0, 50, 100]},
                    {"canonical_field": "amount", "polygon": [50, 0, 100, 100]},
                ],
                "rows": [
                    {
                        "polygon": [0, 0, 100, 50],
                        "cells": [
                            {
                                "canonical_field": "description",
                                "raw_value": "Dressing",
                                "evidence": [{"token_ids": ["description-token"]}],
                            },
                            {
                                "canonical_field": "amount",
                                "raw_value": "10.00",
                                "evidence": [{"token_ids": ["amount-token"]}],
                            },
                        ],
                    }
                ],
            }
        ],
        "totals": [
            {
                "total_id": gold.totals[0].total_id,
                "kind": "subtotal",
                "scope": table.table_id,
                "raw_value": subtotal,
                "polygon": [0, 50, 100, 75],
                "evidence": [{"token_ids": ["subtotal-token"]}],
            },
            {
                "total_id": gold.totals[1].total_id,
                "kind": "document_total",
                "scope": gold.document_id,
                "raw_value": "10.00",
                "polygon": [0, 75, 100, 100],
                "evidence": [{"token_ids": ["total-token"]}],
            },
        ],
    }


def _materials() -> dict[str, object]:
    curved_source, flat_source = _h("c"), _h("d")
    gold_documents = {
        curved_source: _gold(curved_source),
        flat_source: _gold(flat_source),
    }
    candidate_documents = {
        source: _actual(gold) for source, gold in gold_documents.items()
    }
    baseline_documents = {
        curved_source: _actual(gold_documents[curved_source], subtotal="9.00"),
        flat_source: _actual(gold_documents[flat_source]),
    }
    evaluator = EvaluatorManifestV1(
        evaluator_id="authority-metrics",
        evaluator_version="authority_metrics_v1",
        code_sha256=_h("5"),
        configuration_sha256=_h("6"),
        source_manifest_sha256=_h("a"),
        gold_manifest_sha256=_h("b"),
        metric_names=tuple(TABLE_MAGIC_FLOORS),
        normalization_policy="money-inr-paise-v1",
        matching_policy="geometry-order-role-v1",
        unreadable_policy="exclude-readable-denominator-report-separately-v1",
    )
    curved_failure_id = gold_documents[curved_source].totals[0].total_id
    targets = (
        {
            "source_sha256": curved_source,
            "page_number": 1,
            "cohort": "curved",
            "failure_ids": (curved_failure_id,),
        },
        {"source_sha256": flat_source, "page_number": 1, "cohort": "flat"},
    )
    baseline_manifest_documents = tuple(
        BaselineDocumentV1(
            document_id=source_document_id_for(item["source_sha256"]),
            source_sha256=item["source_sha256"],
            result_sha256=authority_metrics.identity_sha256(
                baseline_documents[item["source_sha256"]]
            ),
            status="complete",
        )
        for item in targets
    )
    baseline = BaselineManifestV1(
        source_manifest_sha256=_h("a"),
        gold_manifest_sha256=_h("b"),
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        documents=baseline_manifest_documents,
        release_revision="release-2026-09-07",
        model_identity="gmoney-baseline-v1",
        configuration_sha256=_h("9"),
    )
    preregistration = UvdocPreregistrationV2(
        source_manifest_sha256=_h("a"),
        gold_manifest_sha256=_h("b"),
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        baseline_manifest_sha256=baseline.manifest_sha256,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        candidate_revision="candidate-20260907",
        candidate_configuration_sha256=_h("e"),
        model_sha256=_h("f"),
        model_config_sha256=_h("1"),
        adapter_config_sha256=_h("2"),
        targets=targets,
    )

    proof_inputs = ("1", "2")
    proofs = []
    for target, input_id in zip(targets, proof_inputs, strict=True):
        gold = gold_documents[target["source_sha256"]]
        candidate = candidate_documents[target["source_sha256"]]
        authority_report = authority_metrics.evaluate_document(
            gold, candidate, document_key=target["source_sha256"]
        )
        proofs.append(
            DownstreamAblationProofV2(
                source_sha256=target["source_sha256"],
                page_number=target["page_number"],
                branch="UVDOC",
                input_artifact_id=_h(input_id),
                input_artifact_sha256=_h(str(int(input_id) + 2)),
                artifact_manifest_sha256=_h(str(int(input_id) + 4)),
                downstream_pipeline_id="gmoney-shadow-downstream",
                downstream_pipeline_version="v2",
                downstream_configuration_sha256=_h("3"),
                output_result_sha256=authority_metrics.identity_sha256(candidate),
                authority_report_sha256=uvdoc_authority._digest(authority_report.to_dict()),
                lineage_artifact_ids=(_h(input_id),),
                canonical_result_sha256_before=_h("4"),
                canonical_result_sha256_after=_h("4"),
            )
        )
    preregistration = UvdocPreregistrationV2(
        source_manifest_sha256=_h("a"),
        gold_manifest_sha256=_h("b"),
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        baseline_manifest_sha256=baseline.manifest_sha256,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        candidate_revision="candidate-20260907",
        candidate_configuration_sha256=_h("e"),
        model_sha256=_h("f"),
        model_config_sha256=_h("1"),
        adapter_config_sha256=_h("2"),
        targets=targets,
    )
    authority_cohorts = _authority_cohorts()
    report = evaluate_uvdoc_accuracy_v2(
        preregistration=preregistration,
        evaluator_manifest=evaluator,
        baseline_manifest=baseline,
        gold_documents=gold_documents,
        baseline_documents=baseline_documents,
        candidate_documents=candidate_documents,
        authority_cohorts=authority_cohorts,
        proofs=tuple(proofs),
    )
    return {
        "preregistration": preregistration,
        "evaluator": evaluator,
        "baseline": baseline,
        "report": report,
        "gold_documents": gold_documents,
        "baseline_documents": baseline_documents,
        "candidate_documents": candidate_documents,
        "authority_cohorts": authority_cohorts,
        "proofs": tuple(proofs),
    }


def _gate(materials, report=None):
    return evaluate_uvdoc_gate_v2(
        materials["preregistration"],
        report if report is not None else materials["report"],
        evaluator_manifest=materials["evaluator"],
        baseline_manifest=materials["baseline"],
        gold_documents=materials["gold_documents"],
        baseline_documents=materials["baseline_documents"],
        candidate_documents=materials["candidate_documents"],
        authority_cohorts=materials["authority_cohorts"],
        proofs=materials["proofs"],
    )


def test_v2_gate_promotes_shadow_only_with_canonical_authority_cohorts() -> None:
    gate = _gate(_materials())

    assert gate.decision is GateDecision.PROMOTE
    assert PROMOTE_MEANS_SHADOW_ONLY is True
    assert tuple(item.cohort for item in gate.cohorts) == (
        "production14",
        "passing36",
        "staging159",
    )
    assert all(item.required_count == item.observed_count for item in gate.cohorts)
    assert gate.metrics
    assert {item.metric_name for item in gate.metrics} == set(TABLE_MAGIC_FLOORS)


def test_model_only_report_cannot_promote_without_supplied_results() -> None:
    materials = _materials()

    gate = evaluate_uvdoc_gate_v2(
        materials["preregistration"],
        materials["report"],
        evaluator_manifest=materials["evaluator"],
        baseline_manifest=materials["baseline"],
        gold_documents=materials["gold_documents"],
        authority_cohorts=materials["authority_cohorts"],
        proofs=materials["proofs"],
    )

    assert gate.decision is GateDecision.HOLD
    assert "recomputation_invalid" in gate.blocking_reasons


def test_report_rejects_missing_preregistered_failure_comparison() -> None:
    report = _materials()["report"]
    payload = report.model_dump(mode="python")
    payload["comparisons"] = ()
    payload.pop("report_sha256", None)

    with pytest.raises(ValueError, match="exactly cover preregistered failures"):
        UvdocAccuracyReportV2.model_validate(payload)


def test_legacy_v1_report_cannot_enter_v2_api() -> None:
    materials = _materials()

    with pytest.raises(ValueError, match="v1"):
        evaluate_uvdoc_gate_v2(
            materials["preregistration"],
            {
                "report_version": "gmoney_uvdoc_accuracy_v1",
            },
            evaluator_manifest=materials["evaluator"],
            baseline_manifest=materials["baseline"],
            gold_documents=materials["gold_documents"],
            baseline_documents=materials["baseline_documents"],
            candidate_documents=materials["candidate_documents"],
            authority_cohorts=materials["authority_cohorts"],
            proofs=materials["proofs"],
        )


def test_floor_contract_rejects_zero_denominator_and_report_missing_floor() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        FloorResultV2(
            branch="UVDOC",
            cohort="flat",
            metric_name="critical_numeric_cell_precision",
            numerator=0,
            denominator=0,
            observed_value=0.0,
            threshold=0.98,
            passed=False,
        )

    report = _materials()["report"]
    payload = report.model_dump(mode="python")
    payload["floors"] = tuple(
        item
        for item in payload["floors"]
        if not (
            item["branch"] == "UVDOC"
            and item["cohort"] == "flat"
            and item["metric_name"] == "grand_total_exact_accuracy"
        )
    )
    with pytest.raises(ValueError, match="missing required"):
        UvdocAccuracyReportV2.model_validate(payload)


def test_flat_floor_regression_holds_even_with_curved_improvement() -> None:
    materials = _materials()
    candidate_documents = dict(materials["candidate_documents"])
    flat_source = _h("d")
    candidate_documents[flat_source] = _actual(
        materials["gold_documents"][flat_source], subtotal="9.00"
    )
    gate_materials = {**materials, "candidate_documents": candidate_documents}
    gate = _gate(gate_materials)

    assert gate.decision is GateDecision.HOLD
    assert "recomputation_invalid" in gate.blocking_reasons


def test_flat_exact_regression_holds_even_if_floor_fixture_is_unchanged() -> None:
    materials = _materials()
    report = materials["report"]
    payload = report.model_dump(mode="python")
    payload["comparisons"] = (
        {
            **payload["comparisons"][0],
            "cohort": "flat",
            "baseline_exact": True,
            "candidate_exact": False,
            "baseline_grounded": True,
            "candidate_grounded": True,
            "comparison_sha256": "",
        },
    )
    payload.pop("report_sha256", None)
    changed = UvdocAccuracyReportV2.model_validate(payload)
    gate = _gate(materials, changed)

    assert gate.decision is GateDecision.HOLD
    assert "recomputed_report_mismatch" in gate.blocking_reasons


def test_new_grounding_or_unreadable_error_holds() -> None:
    materials = _materials()
    report = materials["report"]
    payload = report.model_dump(mode="python")
    observations = list(payload["observations"])
    for index, observation in enumerate(observations):
        if observation["branch"] == "UVDOC" and observation["cohort"] == "flat":
            updated = dict(observation)
            updated["grounding_error_ids"] = ("flat-cell",)
            updated.pop("observation_sha256", None)
            observations[index] = updated
    payload["observations"] = tuple(observations)
    payload.pop("report_sha256", None)
    changed = UvdocAccuracyReportV2.model_validate(payload)
    gate = _gate(materials, changed)

    assert gate.decision is GateDecision.HOLD
    assert "recomputed_report_mismatch" in gate.blocking_reasons


def test_missing_curved_resolution_holds() -> None:
    materials = _materials()
    report = materials["report"]
    payload = report.model_dump(mode="python")
    payload["comparisons"] = (
        {
            **payload["comparisons"][0],
            "candidate_exact": False,
            "candidate_status": "wrong_value",
            "comparison_sha256": "",
        },
    )
    payload.pop("report_sha256", None)
    changed = UvdocAccuracyReportV2.model_validate(payload)
    gate = _gate(materials, changed)

    assert gate.decision is GateDecision.HOLD
    assert "recomputed_report_mismatch" in gate.blocking_reasons


def test_downstream_proof_requires_shadow_only_and_unchanged_canonical_hash() -> None:
    with pytest.raises(ValueError, match="changed the canonical"):
        DownstreamAblationProofV2(
            source_sha256=_h("a"),
            page_number=1,
            branch="UVDOC",
            input_artifact_id=_h("b"),
            input_artifact_sha256=_h("c"),
            artifact_manifest_sha256=_h("d"),
            downstream_pipeline_id="pipeline",
            downstream_pipeline_version="v1",
            downstream_configuration_sha256=_h("e"),
            output_result_sha256=_h("f"),
            authority_report_sha256=_h("1"),
            lineage_artifact_ids=(_h("b"),),
            canonical_result_sha256_before=_h("2"),
            canonical_result_sha256_after=_h("3"),
        )


def test_downstream_proof_must_bind_to_authority_observation() -> None:
    report = _materials()["report"]
    payload = report.model_dump(mode="python")
    proof = dict(payload["proofs"][0])
    proof["authority_report_sha256"] = _h("f")
    proof.pop("proof_sha256", None)
    changed_proof = DownstreamAblationProofV2.model_validate(proof)
    payload["proofs"] = (changed_proof, payload["proofs"][1])
    observations = list(payload["observations"])
    for index, observation in enumerate(observations):
        if observation["branch"] == "UVDOC" and observation["cohort"] == "curved":
            updated = dict(observation)
            updated["downstream_proof_sha256"] = changed_proof.proof_sha256
            updated.pop("observation_sha256", None)
            observations[index] = updated
    payload["observations"] = tuple(observations)
    payload.pop("report_sha256", None)
    with pytest.raises(ValueError, match="authority report differs"):
        UvdocAccuracyReportV2.model_validate(payload)


def test_mutated_downstream_proof_cannot_promote_recomputed_results() -> None:
    materials = _materials()
    proof_payload = materials["proofs"][0].model_dump(mode="python")
    proof_payload["output_result_sha256"] = _h("0")
    proof_payload.pop("proof_sha256", None)
    mutated_proof = DownstreamAblationProofV2.model_validate(proof_payload)
    mutated_materials = {**materials, "proofs": (mutated_proof, *materials["proofs"][1:])}

    gate = _gate(mutated_materials)

    assert gate.decision is GateDecision.HOLD
    assert "recomputation_invalid" in gate.blocking_reasons


def test_direct_gold_document_v2_is_accepted_by_authority_evaluator() -> None:
    source = _h("a")

    def polygon(left: int, top: int, right: int, bottom: int) -> dict[str, object]:
        return {
            "points": [
                {"x": left, "y": top},
                {"x": right, "y": top},
                {"x": right, "y": bottom},
                {"x": left, "y": bottom},
            ]
        }

    gold = GoldDocumentV2(
        source_sha256=source,
        source_manifest_sha256=_h("b"),
        page_count=1,
        pages=[
            {
                "source_sha256": source,
                "page_number": 1,
                "artifact_sha256": _h("c"),
                "width": 100,
                "height": 100,
                "dpi": 300,
                "source_class": SourceClass.FLAT_SCAN,
                "tables": [
                    {
                        "table_kind": TableKind.ITEM_LEDGER,
                        "polygon": polygon(0, 0, 100, 100),
                        "columns": [
                            {"canonical_field": "description", "polygon": polygon(0, 0, 50, 100)},
                            {"canonical_field": "amount", "polygon": polygon(50, 0, 100, 100)},
                        ],
                        "rows": [
                            {
                                "polygon": polygon(0, 0, 100, 50),
                                "cells": [
                                    {
                                        "polygon": polygon(0, 0, 50, 50),
                                        "canonical_field": "description",
                                        "raw_value": "Dressing",
                                    },
                                    {
                                        "polygon": polygon(50, 0, 100, 50),
                                        "canonical_field": "amount",
                                        "raw_value": "₹10.00",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        annotation_group_id="uvdoc-authority-test",
        split="holdout",
    )
    actual = {
        "source_sha256": source,
        "tables": [
            {
                "table_kind": "item_ledger",
                "polygon": [0, 0, 100, 100],
                "columns": [
                    {"canonical_field": "description", "polygon": [0, 0, 50, 100]},
                    {"canonical_field": "amount", "polygon": [50, 0, 100, 100]},
                ],
                "rows": [
                    {
                        "polygon": [0, 0, 100, 50],
                        "cells": [
                            {
                                "canonical_field": "description",
                                "raw_value": "Dressing",
                                "evidence": [{"token_ids": ["token-1"]}],
                            },
                            {
                                "canonical_field": "amount",
                                "raw_value": "10.00",
                                "evidence": [{"token_ids": ["token-2"]}],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    report = authority_metrics.evaluate_document(gold, actual)

    assert report.gold_identity_sha256 == gold.gold_sha256
    assert report.metrics["critical_numeric_cell_recall"] == 1.0
    assert report.metrics["critical_numeric_cell_precision"] == 1.0


def test_report_hash_is_deterministic() -> None:
    first = _materials()["report"]
    second = _materials()["report"]

    assert first.report_sha256 == second.report_sha256
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
