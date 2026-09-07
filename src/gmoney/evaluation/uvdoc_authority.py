"""Authoritative, fail-closed M4 UVDoc accuracy evidence.

The legacy UVDoc gate records that a shadow adapter ran.  This module is the
small, sealed bridge between that telemetry and the structural authority
evaluator.  It intentionally does not read a workspace or infer a result from
arbitrary dictionaries: callers provide content-addressed authority reports,
baseline manifests, and shadow-only downstream proofs.

``PROMOTE`` in the report returned by :func:`evaluate_uvdoc_gate_v2` means
that the M4 *shadow* milestone passed.  It does not enable UVDoc for canonical
publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from gmoney.contracts.authority import (
    AUTHORITY_COHORT_COUNTS,
    AUTHORITY_COHORTS,
    AuthorityModel,
    BaselineManifestV1,
    EvaluatorManifestV1,
    GateCohortResultV1,
    GoldDocumentV2,
    M0M4GateReportV1,
    MetricObservationV1,
)
from gmoney.contracts.v6 import canonical_sha256
from gmoney.evaluation.authority_metrics import (
    TABLE_MAGIC_FLOORS,
    AuthorityDocumentReport,
    evaluate_cohort,
    evaluate_document,
    identity_sha256,
)

SHA256_PATTERN = r"^[a-f0-9]{64}$"
REPORT_VERSION = "gmoney_uvdoc_accuracy_v2"
PREREGISTRATION_VERSION = "gmoney_uvdoc_preregistration_v2"
PROOF_VERSION = "uvdoc_downstream_ablation_v2"
FLOOR_POLICY_VERSION = "table_magic_floors_v1"
PROMOTE_MEANS_SHADOW_ONLY = True

BRANCHES = ("BASELINE", "UVDOC", "UVDOC_ENHANCED")
REQUIRED_BRANCHES = ("BASELINE", "UVDOC")
REQUIRED_COHORTS = ("curved", "flat")

_CRITICAL_ISSUES = frozenset(
    {
        "wrong_table",
        "duplicate",
        "missed",
        "spurious",
        "wrong_column",
        "wrong_role",
        "wrong_value",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"expected a mapping or contract model, got {type(value).__name__}")


def _json_ready(value: Any) -> Any:
    """Convert nested contract/dataclass values to deterministic JSON values."""

    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    return value


def _hash_payload(kind: str, value: Any) -> str:
    return canonical_sha256(
        {
            "uvdoc_authority_version": "uvdoc_authority_v2",
            "kind": kind,
            "payload": _json_ready(value),
        }
    )


def _digest(value: Any) -> str:
    return canonical_sha256(_json_ready(value))


def floor_policy_sha256() -> str:
    """Digest the frozen names and thresholds used by this gate."""

    return _hash_payload(
        "floor_policy",
        {"version": FLOOR_POLICY_VERSION, "floors": TABLE_MAGIC_FLOORS},
    )


FLOOR_POLICY_SHA256 = floor_policy_sha256()


class UvdocTargetV2(AuthorityModel):
    """One preregistered source page and its frozen failure inventory."""

    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    cohort: Literal["curved", "flat"]
    failure_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_failures(self) -> UvdocTargetV2:
        if len(set(self.failure_ids)) != len(self.failure_ids):
            raise ValueError("UVDoc target failure IDs must be unique")
        if self.cohort == "curved" and not self.failure_ids:
            raise ValueError("curved UVDoc targets require preregistered failure IDs")
        return self


class UvdocPreregistrationV2(AuthorityModel):
    """Versioned identities required before an accuracy report can pass."""

    manifest_version: Literal[PREREGISTRATION_VERSION] = PREREGISTRATION_VERSION
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    gold_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_policy_sha256: str = Field(default=FLOOR_POLICY_SHA256, pattern=SHA256_PATTERN)
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    candidate_revision: str = Field(min_length=1)
    candidate_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=SHA256_PATTERN)
    targets: tuple[UvdocTargetV2, ...]
    preregistration_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        data.setdefault("manifest_version", PREREGISTRATION_VERSION)
        if not data.get("floor_policy_sha256"):
            data["floor_policy_sha256"] = FLOOR_POLICY_SHA256
        targets = tuple(UvdocTargetV2.model_validate(item) for item in data.get("targets", ()))
        data["targets"] = targets
        if not data.get("preregistration_sha256"):
            payload = {key: item for key, item in data.items() if key != "preregistration_sha256"}
            data["preregistration_sha256"] = _hash_payload("preregistration", payload)
        return data

    @model_validator(mode="after")
    def validate_targets_and_hash(self) -> UvdocPreregistrationV2:
        if self.floor_policy_sha256 != FLOOR_POLICY_SHA256:
            raise ValueError("UVDoc preregistration uses an unknown floor policy")
        keys = [(item.source_sha256, item.page_number) for item in self.targets]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("UVDoc v2 targets must be non-empty and unique")
        if {item.cohort for item in self.targets} != set(REQUIRED_COHORTS):
            raise ValueError("UVDoc v2 preregistration requires curved and flat targets")
        payload = self.model_dump(mode="python", exclude={"preregistration_sha256"})
        expected = _hash_payload("preregistration", payload)
        if self.preregistration_sha256 != expected:
            raise ValueError("UVDoc preregistration hash does not match canonical content")
        return self


class FloorResultV2(AuthorityModel):
    """One exact numerator/denominator comparison against a frozen floor."""

    branch: Literal["BASELINE", "UVDOC", "UVDOC_ENHANCED"]
    cohort: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    numerator: int = Field(ge=0)
    denominator: int = Field(gt=0)
    observed_value: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    floor_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_result_hash_and_pass(cls, value: Any) -> Any:
        data = _as_dict(value)
        if "observed_value" not in data and "value" in data:
            data["observed_value"] = data.pop("value")
        if "observed_value" not in data:
            denominator = int(data.get("denominator", 0))
            if denominator > 0:
                data["observed_value"] = round(
                    int(data.get("numerator", 0)) / denominator, 12
                )
        if "passed" not in data:
            data["passed"] = float(data.get("observed_value", 0.0)) >= float(
                data.get("threshold", 0.0)
            )
        if not data.get("floor_sha256"):
            payload = {key: item for key, item in data.items() if key != "floor_sha256"}
            data["floor_sha256"] = _hash_payload("floor_result", payload)
        return data

    @model_validator(mode="after")
    def validate_ratio_and_hash(self) -> FloorResultV2:
        expected_value = round(self.numerator / self.denominator, 12)
        if self.observed_value != expected_value:
            raise ValueError("floor observed value must equal its exact ratio")
        expected_passed = self.observed_value >= self.threshold
        if self.passed != expected_passed:
            raise ValueError("floor pass flag does not match observed value and threshold")
        payload = self.model_dump(mode="python", exclude={"floor_sha256"})
        expected_hash = _hash_payload("floor_result", payload)
        if self.floor_sha256 != expected_hash:
            raise ValueError("floor result hash does not match canonical content")
        return self

    @property
    def value(self) -> float:
        """Compatibility spelling for callers that call the ratio ``value``."""

        return self.observed_value


class DownstreamAblationProofV2(AuthorityModel):
    """Content-addressed proof that a shadow downstream branch consumed UVDoc."""

    proof_version: Literal[PROOF_VERSION] = PROOF_VERSION
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    branch: Literal["BASELINE", "UVDOC", "UVDOC_ENHANCED"]
    input_artifact_id: str = Field(pattern=SHA256_PATTERN)
    input_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    downstream_pipeline_id: str = Field(min_length=1)
    downstream_pipeline_version: str = Field(min_length=1)
    downstream_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    output_result_sha256: str = Field(pattern=SHA256_PATTERN)
    authority_report_sha256: str = Field(pattern=SHA256_PATTERN)
    lineage_artifact_ids: tuple[str, ...] = ()
    fallback_used: Literal[False] = False
    publication_scope: Literal["shadow_only"] = "shadow_only"
    canonical_result_sha256_before: str = Field(pattern=SHA256_PATTERN)
    canonical_result_sha256_after: str = Field(pattern=SHA256_PATTERN)
    proof_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        data.setdefault("proof_version", PROOF_VERSION)
        data.setdefault("lineage_artifact_ids", ())
        data.setdefault("fallback_used", False)
        data.setdefault("publication_scope", "shadow_only")
        if not data.get("proof_sha256"):
            payload = {key: item for key, item in data.items() if key != "proof_sha256"}
            data["proof_sha256"] = _hash_payload("downstream_proof", payload)
        return data

    @model_validator(mode="after")
    def validate_proof(self) -> DownstreamAblationProofV2:
        if self.canonical_result_sha256_before != self.canonical_result_sha256_after:
            raise ValueError("shadow ablation changed the canonical result")
        if self.input_artifact_id not in self.lineage_artifact_ids:
            raise ValueError("downstream proof must include its selected input artifact")
        if len(set(self.lineage_artifact_ids)) != len(self.lineage_artifact_ids):
            raise ValueError("downstream proof lineage artifact IDs must be unique")
        payload = self.model_dump(mode="python", exclude={"proof_sha256"})
        expected_hash = _hash_payload("downstream_proof", payload)
        if self.proof_sha256 != expected_hash:
            raise ValueError("downstream proof hash does not match canonical content")
        return self


class FailureComparisonV2(AuthorityModel):
    """Machine-derived baseline/candidate outcome for one preregistered failure."""

    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    cohort: Literal["curved", "flat"]
    candidate_branch: Literal["UVDOC", "UVDOC_ENHANCED"] = "UVDOC"
    failure_id: str = Field(min_length=1)
    baseline_status: str = Field(min_length=1)
    candidate_status: str = Field(min_length=1)
    baseline_exact: bool
    candidate_exact: bool
    baseline_grounded: bool
    candidate_grounded: bool
    baseline_critical: bool
    candidate_critical: bool
    baseline_grounding_error: bool = False
    candidate_grounding_error: bool = False
    baseline_unreadable_mishandled: bool = False
    candidate_unreadable_mishandled: bool = False
    comparison_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        data.setdefault("candidate_branch", "UVDOC")
        data.setdefault("baseline_grounding_error", False)
        data.setdefault("candidate_grounding_error", False)
        data.setdefault("baseline_unreadable_mishandled", False)
        data.setdefault("candidate_unreadable_mishandled", False)
        if not data.get("comparison_sha256"):
            payload = {key: item for key, item in data.items() if key != "comparison_sha256"}
            data["comparison_sha256"] = _hash_payload("failure_comparison", payload)
        return data

    @model_validator(mode="after")
    def validate_hash(self) -> FailureComparisonV2:
        payload = self.model_dump(mode="python", exclude={"comparison_sha256"})
        if self.comparison_sha256 != _hash_payload("failure_comparison", payload):
            raise ValueError("failure comparison hash does not match canonical content")
        return self

    @property
    def resolved(self) -> bool:
        return (
            not self.baseline_exact
            and self.candidate_exact
            and self.candidate_grounded
            and self.candidate_status in {"matched", "exact", "correct"}
        )


class UvdocAccuracyObservationV2(AuthorityModel):
    """One sealed authority document evaluation for one branch."""

    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    cohort: Literal["curved", "flat"]
    branch: Literal["BASELINE", "UVDOC", "UVDOC_ENHANCED"]
    gold_sha256: str = Field(pattern=SHA256_PATTERN)
    actual_result_sha256: str = Field(pattern=SHA256_PATTERN)
    authority_report_sha256: str = Field(pattern=SHA256_PATTERN)
    authority_pair_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    metrics: tuple[MetricObservationV1, ...] = ()
    critical_error_ids: tuple[str, ...] = ()
    grounding_error_ids: tuple[str, ...] = ()
    unreadable_error_ids: tuple[str, ...] = ()
    baseline_failure_ids: tuple[str, ...] = ()
    resolved_failure_ids: tuple[str, ...] = ()
    downstream_proof_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    observation_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        data.setdefault("metrics", ())
        data.setdefault("critical_error_ids", ())
        data.setdefault("grounding_error_ids", ())
        data.setdefault("unreadable_error_ids", ())
        data.setdefault("baseline_failure_ids", ())
        data.setdefault("resolved_failure_ids", ())
        data.setdefault("downstream_proof_sha256", None)
        for field_name in (
            "critical_error_ids",
            "grounding_error_ids",
            "unreadable_error_ids",
            "baseline_failure_ids",
            "resolved_failure_ids",
        ):
            data[field_name] = tuple(data.get(field_name, ()))
        if not data.get("observation_sha256"):
            payload = {key: item for key, item in data.items() if key != "observation_sha256"}
            data["observation_sha256"] = _hash_payload("accuracy_observation", payload)
        return data

    @model_validator(mode="after")
    def validate_observation(self) -> UvdocAccuracyObservationV2:
        for field_name in (
            "critical_error_ids",
            "grounding_error_ids",
            "unreadable_error_ids",
            "baseline_failure_ids",
            "resolved_failure_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
        if self.branch in {"UVDOC", "UVDOC_ENHANCED"} and self.downstream_proof_sha256 is None:
            raise ValueError(f"{self.branch} observations require downstream proof")
        if self.branch == "BASELINE" and self.resolved_failure_ids:
            raise ValueError("baseline observations cannot resolve candidate failures")
        payload = self.model_dump(mode="python", exclude={"observation_sha256"})
        if self.observation_sha256 != _hash_payload("accuracy_observation", payload):
            raise ValueError("accuracy observation hash does not match canonical content")
        return self


class UvdocAccuracyReportV2(AuthorityModel):
    """Frozen baseline/candidate authority evidence for the M4 shadow gate."""

    report_version: Literal[REPORT_VERSION] = REPORT_VERSION
    preregistration_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    gold_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_id: str = Field(pattern=SHA256_PATTERN)
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    candidate_revision: str = Field(min_length=1)
    candidate_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    targets: tuple[UvdocTargetV2, ...]
    observations: tuple[UvdocAccuracyObservationV2, ...]
    floors: tuple[FloorResultV2, ...]
    proofs: tuple[DownstreamAblationProofV2, ...]
    # The M0--M4 gate contract has three canonical release cohorts.  Curved/flat
    # UVDoc target metrics remain in ``floors``; this tuple carries only the
    # already-sealed authority cohort decisions and their required counts.
    authority_cohorts: tuple[GateCohortResultV1, ...]
    comparisons: tuple[FailureComparisonV2, ...] = ()
    required_cohorts: tuple[str, ...] = REQUIRED_COHORTS
    report_sha256: str = Field(default="", pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def fill_hash(cls, value: Any) -> Any:
        data = _as_dict(value)
        data["targets"] = tuple(
            UvdocTargetV2.model_validate(item) for item in data.get("targets", ())
        )
        data["observations"] = tuple(
            UvdocAccuracyObservationV2.model_validate(item)
            for item in data.get("observations", ())
        )
        data["floors"] = tuple(
            FloorResultV2.model_validate(item) for item in data.get("floors", ())
        )
        data["proofs"] = tuple(
            DownstreamAblationProofV2.model_validate(item)
            for item in data.get("proofs", ())
        )
        data["authority_cohorts"] = tuple(
            GateCohortResultV1.model_validate(item)
            for item in data.get("authority_cohorts", ())
        )
        data.setdefault("comparisons", ())
        data["comparisons"] = tuple(
            FailureComparisonV2.model_validate(item) for item in data.get("comparisons", ())
        )
        data.setdefault("report_version", REPORT_VERSION)
        data.setdefault("required_cohorts", REQUIRED_COHORTS)
        if not data.get("report_sha256"):
            payload = {key: item for key, item in data.items() if key != "report_sha256"}
            data["report_sha256"] = _hash_payload("accuracy_report", payload)
        return data

    @model_validator(mode="after")
    def validate_report(self) -> UvdocAccuracyReportV2:
        if self.floor_policy_sha256 != FLOOR_POLICY_SHA256:
            raise ValueError("accuracy report uses an unknown floor policy")
        if tuple(self.required_cohorts) != REQUIRED_COHORTS:
            raise ValueError("M4 requires canonical curved and flat cohorts")
        if tuple(item.cohort for item in self.authority_cohorts) != AUTHORITY_COHORTS:
            raise ValueError("accuracy report requires canonical authority cohort order")
        if any(
            item.required_count != AUTHORITY_COHORT_COUNTS[item.cohort]
            for item in self.authority_cohorts
        ):
            raise ValueError("accuracy report authority cohort counts are not frozen")
        for authority_cohort in self.authority_cohorts:
            metric_names = {item.metric_name for item in authority_cohort.metrics}
            if not set(TABLE_MAGIC_FLOORS).issubset(metric_names):
                raise ValueError(
                    f"authority cohort {authority_cohort.cohort} is missing Table Magic floors"
                )
            if any(
                item.metric_name in TABLE_MAGIC_FLOORS
                and (item.denominator is None or item.denominator <= 0)
                for item in authority_cohort.metrics
            ):
                raise ValueError(
                    f"authority cohort {authority_cohort.cohort} has a zero floor denominator"
                )
        for floor in self.floors:
            expected_threshold = TABLE_MAGIC_FLOORS.get(floor.metric_name)
            if expected_threshold is not None and floor.threshold != expected_threshold:
                raise ValueError(
                    f"{floor.metric_name} floor threshold differs from the frozen policy"
                )
        target_keys = {(item.source_sha256, item.page_number) for item in self.targets}
        if not target_keys or len(target_keys) != len(self.targets):
            raise ValueError("accuracy report targets must be unique and non-empty")
        target_cohorts = {
            (item.source_sha256, item.page_number): item.cohort for item in self.targets
        }
        observation_keys = {
            (item.source_sha256, item.page_number, item.branch) for item in self.observations
        }
        if len(observation_keys) != len(self.observations):
            raise ValueError("accuracy observations must be unique by target and branch")
        for branch in REQUIRED_BRANCHES:
            expected = {(source, page, branch) for source, page in target_keys}
            actual = {key for key in observation_keys if key[2] == branch}
            if actual != expected:
                raise ValueError(f"{branch} observations do not exactly cover targets")
        enhanced = {key for key in observation_keys if key[2] == "UVDOC_ENHANCED"}
        if enhanced and enhanced != {
            (source, page, "UVDOC_ENHANCED") for source, page in target_keys
        }:
            raise ValueError("enhanced observations must cover every target or none")
        for item in self.observations:
            if target_cohorts.get((item.source_sha256, item.page_number)) != item.cohort:
                raise ValueError("observation cohort differs from its preregistered target")
        floor_keys = {(item.branch, item.cohort, item.metric_name) for item in self.floors}
        if len(floor_keys) != len(self.floors):
            raise ValueError("floor results must be unique by branch, cohort, and metric")
        expected_floor_keys = {
            (branch, cohort, metric)
            for branch in REQUIRED_BRANCHES
            for cohort in self.required_cohorts
            for metric in TABLE_MAGIC_FLOORS
        }
        if not expected_floor_keys.issubset(floor_keys):
            raise ValueError("accuracy report is missing required Table Magic floors")
        proof_by_hash = {proof.proof_sha256: proof for proof in self.proofs}
        proof_keys = {
            (proof.source_sha256, proof.page_number, proof.branch) for proof in self.proofs
        }
        if len(proof_keys) != len(self.proofs):
            raise ValueError("downstream proofs must be unique by target and branch")
        candidate_obs = [
            item for item in self.observations if item.branch in {"UVDOC", "UVDOC_ENHANCED"}
        ]
        expected_proof_keys = {
            (item.source_sha256, item.page_number, item.branch) for item in candidate_obs
        }
        if proof_keys != expected_proof_keys:
            raise ValueError("downstream proofs must exactly cover candidate targets")
        for item in candidate_obs:
            if item.downstream_proof_sha256 not in proof_by_hash:
                raise ValueError(f"{item.branch} observation references a missing downstream proof")
            proof = proof_by_hash[item.downstream_proof_sha256]
            if (proof.source_sha256, proof.page_number, proof.branch) != (
                item.source_sha256,
                item.page_number,
                item.branch,
            ):
                raise ValueError(f"{item.branch} observation proof identity differs")
            if proof.output_result_sha256 != item.actual_result_sha256:
                raise ValueError("UVDOC proof output differs from evaluated result")
            if proof.authority_report_sha256 != item.authority_report_sha256:
                raise ValueError("downstream proof authority report differs from observation")
        comparison_keys = {
            (item.source_sha256, item.page_number, item.candidate_branch, item.failure_id)
            for item in self.comparisons
        }
        if len(comparison_keys) != len(self.comparisons):
            raise ValueError("failure comparisons must be unique")
        expected_comparison_keys = {
            (target.source_sha256, target.page_number, "UVDOC", failure_id)
            for target in self.targets
            for failure_id in target.failure_ids
        }
        if comparison_keys != expected_comparison_keys:
            raise ValueError("failure comparisons do not exactly cover preregistered failures")
        payload = self.model_dump(mode="python", exclude={"report_sha256"})
        if self.report_sha256 != _hash_payload("accuracy_report", payload):
            raise ValueError("accuracy report hash does not match canonical content")
        return self


def _coerce_preregistration(value: Any) -> UvdocPreregistrationV2:
    raw = _as_dict(value)
    version = raw.get("manifest_version")
    if version != PREREGISTRATION_VERSION:
        raise ValueError("legacy or missing UVDoc preregistration v1 cannot enter v2 gate")
    return UvdocPreregistrationV2.model_validate(value)


def _coerce_report(value: Any) -> UvdocAccuracyReportV2:
    raw = _as_dict(value)
    version = raw.get("report_version")
    if version != REPORT_VERSION:
        raise ValueError("legacy or missing UVDoc accuracy report v1 cannot enter v2 gate")
    return UvdocAccuracyReportV2.model_validate(value)


def _coerce_evaluator(value: Any) -> EvaluatorManifestV1:
    return EvaluatorManifestV1.model_validate(value)


def _coerce_baseline(value: Any) -> BaselineManifestV1:
    return BaselineManifestV1.model_validate(value)


def _resolve_result_mapping(
    primary: Mapping[str, Any] | None,
    alias: Mapping[str, Any] | None,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Require one unambiguous result mapping at the authority boundary."""

    if primary is None and alias is None:
        raise ValueError(f"{label} results are required")
    if primary is not None and alias is not None and primary != alias:
        raise ValueError(f"{label} document/result mappings disagree")
    selected = primary if primary is not None else alias
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError(f"{label} results must be a non-empty mapping")
    return selected


def _canonical_authority_cohorts(
    values: Sequence[GateCohortResultV1 | Mapping[str, Any]],
) -> tuple[GateCohortResultV1, ...]:
    """Validate the sealed 14/36/159 authority cohort evidence."""

    models = tuple(GateCohortResultV1.model_validate(item) for item in values)
    if len(models) != len(AUTHORITY_COHORTS):
        raise ValueError(
            "authority cohorts must contain exactly production14, passing36, staging159"
        )
    if tuple(item.cohort for item in models) != AUTHORITY_COHORTS:
        raise ValueError("authority cohorts must use canonical order")
    for item, cohort in zip(models, AUTHORITY_COHORTS, strict=True):
        expected_count = AUTHORITY_COHORT_COUNTS[cohort]
        if (
            item.required_count != expected_count
            or item.observed_count != expected_count
            or not item.passed
            or item.blocking_reasons
        ):
            raise ValueError(f"authority cohort evidence is not passed/sealed: {cohort}")
        metrics = {metric.metric_name: metric for metric in item.metrics}
        if set(metrics) != set(TABLE_MAGIC_FLOORS) or len(metrics) != len(item.metrics):
            raise ValueError(f"authority cohort floors are incomplete or duplicated: {cohort}")
        if any(
            metric.cohort != cohort
            or metric.denominator is None
            or metric.denominator <= 0
            or metric.value < TABLE_MAGIC_FLOORS[metric.metric_name]
            for metric in metrics.values()
        ):
            raise ValueError(f"authority cohort floors are invalid: {cohort}")
    return models


def _floor_map(report: UvdocAccuracyReportV2, branch: str, cohort: str) -> dict[str, FloorResultV2]:
    return {
        item.metric_name: item
        for item in report.floors
        if item.branch == branch and item.cohort == cohort
    }


def _metric_observations(
    floor_results: Sequence[FloorResultV2], *, cohort: str
) -> tuple[MetricObservationV1, ...]:
    return tuple(
        MetricObservationV1(
            metric_name=item.metric_name,
            value=item.observed_value,
            numerator=item.numerator,
            denominator=item.denominator,
            cohort=cohort,
        )
        for item in sorted(floor_results, key=lambda result: result.metric_name)
    )


def _same_or_better(candidate: FloorResultV2, baseline: FloorResultV2) -> bool:
    return candidate.numerator * baseline.denominator >= baseline.numerator * candidate.denominator


def _failure_outcome(
    report: AuthorityDocumentReport,
    failure_id: str,
    *,
    gold: GoldDocumentV2 | None = None,
) -> Any | None:
    # Failure inventories may disambiguate evaluator outcome levels with a
    # ``level:identifier`` prefix.  Totals use the generic authority evaluator's
    # cell outcome IDs, so the prefix is necessary whenever a cell and total
    # share the same positional identifier.
    level, separator, wanted = failure_id.partition(":")
    if separator and level == "cell":
        candidates = report.cells
    elif separator and level == "row":
        candidates = report.rows
    elif separator and level == "table":
        candidates = report.tables
    elif separator and level == "total":
        candidates = report.totals
    else:
        candidates = (*report.cells, *report.rows, *report.totals, *report.tables)
        wanted = failure_id
    for outcome in candidates:
        if outcome.gold_id in {failure_id, wanted} or outcome.actual_id in {failure_id, wanted}:
            return outcome
    # GoldTotalV2 has a stable total_id, while the generic evaluator represents
    # totals as CellOutcome instances and therefore gives them positional cell
    # IDs.  Resolve the contract identity back to the matching gold-indexed
    # outcome instead of making callers invent evaluator-specific IDs.
    if gold is not None and not separator:
        for total_index, total in enumerate(gold.totals):
            if total.total_id != failure_id:
                continue
            return next(
                (
                    outcome
                    for outcome in report.totals
                    if outcome.gold_index == total_index
                ),
                None,
            )
    return None


def _outcome_flags(outcome: Any) -> tuple[str, bool, bool, bool, bool, bool, bool]:
    issues = set(getattr(outcome, "issues", ()))
    status = str(getattr(outcome, "status", "unknown"))
    exact = bool(getattr(outcome, "exact", getattr(outcome, "complete", False)))
    grounded = bool(getattr(outcome, "grounded_actual", False)) and "ungrounded" not in issues
    readable = bool(getattr(outcome, "readable_actual", True))
    critical = bool(issues & _CRITICAL_ISSUES)
    return (
        status,
        exact,
        grounded,
        critical,
        "ungrounded" in issues,
        "unreadable_mishandled" in issues,
        readable,
    )


def _report_error_ids(
    report: AuthorityDocumentReport,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    critical: set[str] = set()
    grounding: set[str] = set()
    unreadable: set[str] = set()
    outcomes = (*report.cells, *report.rows, *report.totals, *report.tables)
    for outcome in outcomes:
        identifier = outcome.actual_id or outcome.gold_id or "unknown"
        issues = set(outcome.issues)
        for issue in issues & _CRITICAL_ISSUES:
            critical.add(f"{identifier}:{issue}")
        if "ungrounded" in issues:
            grounding.add(identifier)
        if "unreadable_mishandled" in issues:
            unreadable.add(identifier)
    return tuple(sorted(critical)), tuple(sorted(grounding)), tuple(sorted(unreadable))


def _actual_identity(value: Any) -> str:
    # A producer-declared result hash is an input field, not proof of the
    # result object's identity.  Derive the identity from supplied content so
    # a downstream proof cannot pass by merely echoing a self-asserted digest.
    return identity_sha256(value)


def _gold_identity(value: Any) -> str:
    raw = _as_dict(value)
    candidate = raw.get("gold_sha256")
    if isinstance(candidate, str) and len(candidate) == 64:
        return candidate
    return identity_sha256(value)


def _metric_counts(metrics: Mapping[str, Any], name: str) -> tuple[int, int]:
    mapping = {
        "critical_numeric_cell_precision": ("critical_exact", "critical_actual"),
        "critical_numeric_cell_recall": ("critical_exact", "critical_gold"),
        "line_item_row_recall": ("line_item_matched_rows", "line_item_gold_rows"),
        "correct_column_assignment": ("correctly_assigned_cells", "gold_cells"),
        "header_schema_accuracy": ("matched_columns", "gold_columns"),
        "grand_total_exact_accuracy": ("grand_total_exact", "grand_total_gold"),
        "auto_accepted_document_correctness": (
            "auto_accepted_correct_documents",
            "auto_accepted_documents",
        ),
    }
    numerator, denominator = mapping[name]
    return int(metrics.get(numerator, 0)), int(metrics.get(denominator, 0))


def _document_metric_observations(
    report: AuthorityDocumentReport, *, cohort: str
) -> tuple[MetricObservationV1, ...]:
    output: list[MetricObservationV1] = []
    for metric_name in TABLE_MAGIC_FLOORS:
        numerator, denominator = _metric_counts(report.metrics, metric_name)
        output.append(
            MetricObservationV1(
                metric_name=metric_name,
                value=float(report.metrics.get(metric_name, 0.0)),
                numerator=numerator,
                denominator=denominator,
                cohort=cohort,
            )
        )
    return tuple(output)


def evaluate_uvdoc_accuracy_v2(
    *,
    preregistration: UvdocPreregistrationV2 | Mapping[str, Any],
    evaluator_manifest: EvaluatorManifestV1 | Mapping[str, Any],
    baseline_manifest: BaselineManifestV1 | Mapping[str, Any],
    gold_documents: Mapping[str, Any],
    baseline_documents: Mapping[str, Any] | None = None,
    candidate_documents: Mapping[str, Any] | None = None,
    baseline_results: Mapping[str, Any] | None = None,
    candidate_results: Mapping[str, Any] | None = None,
    authority_cohorts: Sequence[GateCohortResultV1 | Mapping[str, Any]],
    proofs: Sequence[DownstreamAblationProofV2 | Mapping[str, Any]],
) -> UvdocAccuracyReportV2:
    """Score sealed baseline/UVDOC documents with the structural authority evaluator.

    ``baseline_results`` and ``candidate_results`` are accepted as aliases for
    the document mappings because result repositories commonly use that term.
    The function deliberately accepts the authority evaluator's generic input
    shape; gold values are nevertheless parsed as ``GoldDocumentV2``.
    """

    prereg = _coerce_preregistration(preregistration)
    evaluator = _coerce_evaluator(evaluator_manifest)
    baseline = _coerce_baseline(baseline_manifest)
    if evaluator.manifest_sha256 != prereg.evaluator_manifest_sha256:
        raise ValueError("evaluator manifest identity differs from preregistration")
    if evaluator.evaluator_id != prereg.evaluator_id:
        raise ValueError("evaluator identity differs from preregistration")
    if evaluator.evaluator_version != prereg.evaluator_version:
        raise ValueError("evaluator version differs from preregistration")
    if evaluator.source_manifest_sha256 != prereg.source_manifest_sha256:
        raise ValueError("evaluator source identity differs from preregistration")
    if evaluator.gold_manifest_sha256 != prereg.gold_manifest_sha256:
        raise ValueError("evaluator gold identity differs from preregistration")
    if baseline.manifest_sha256 != prereg.baseline_manifest_sha256:
        raise ValueError("baseline manifest identity differs from preregistration")
    if baseline.source_manifest_sha256 != prereg.source_manifest_sha256:
        raise ValueError("baseline source identity differs from preregistration")
    if baseline.gold_manifest_sha256 != prereg.gold_manifest_sha256:
        raise ValueError("baseline gold identity differs from preregistration")
    if baseline.evaluator_manifest_sha256 != prereg.evaluator_manifest_sha256:
        raise ValueError("baseline evaluator identity differs from preregistration")
    if not isinstance(gold_documents, Mapping) or not gold_documents:
        raise ValueError("gold documents are required")
    gold_by_source = {
        str(key): GoldDocumentV2.model_validate(value) for key, value in gold_documents.items()
    }
    baseline_by_source = _resolve_result_mapping(
        baseline_documents, baseline_results, label="baseline"
    )
    candidate_by_source = _resolve_result_mapping(
        candidate_documents, candidate_results, label="candidate"
    )
    authority_cohort_models = _canonical_authority_cohorts(authority_cohorts)
    proof_models = tuple(_coerce_proof(item) for item in proofs)
    target_sources = {item.source_sha256 for item in prereg.targets}
    if set(gold_by_source) != target_sources:
        raise ValueError("gold documents do not exactly cover preregistered sources")
    if set(baseline_by_source) != target_sources or set(candidate_by_source) != target_sources:
        raise ValueError("baseline and candidate results do not exactly cover sources")
    if set(item.source_sha256 for item in baseline.documents) != target_sources:
        raise ValueError("baseline manifest does not exactly cover preregistered sources")
    for source_sha256, document in gold_by_source.items():
        if document.source_sha256 != source_sha256:
            raise ValueError("gold document source identity differs from its mapping key")
    for source_sha256, document in (*baseline_by_source.items(), *candidate_by_source.items()):
        declared_source = _as_dict(document).get("source_sha256")
        if declared_source not in (None, "", source_sha256):
            raise ValueError("result source identity differs from its mapping key")
    if any(
        item.source_manifest_sha256 != prereg.source_manifest_sha256
        for item in gold_by_source.values()
    ):
        raise ValueError("gold document source identity differs from preregistration")
    for target in prereg.targets:
        if target.page_number not in {
            page.page_number for page in gold_by_source[target.source_sha256].pages
        }:
            raise ValueError("gold documents do not exactly cover preregistered pages")
    baseline_result_ids = {item.source_sha256: item.result_sha256 for item in baseline.documents}
    for source_sha256 in target_sources:
        if (
            _actual_identity(baseline_by_source[source_sha256])
            != baseline_result_ids[source_sha256]
        ):
            raise ValueError("baseline result identity differs from baseline manifest")
    expected_proof_keys = {
        (item.source_sha256, item.page_number, "UVDOC") for item in prereg.targets
    }
    proof_by_key = {
        (item.source_sha256, item.page_number, item.branch): item for item in proof_models
    }
    if len(proof_by_key) != len(proof_models) or set(proof_by_key) != expected_proof_keys:
        raise ValueError("downstream proofs must exactly cover preregistered targets")
    observations: list[UvdocAccuracyObservationV2] = []
    document_reports: dict[tuple[str, str], AuthorityDocumentReport] = {}
    for target in prereg.targets:
        gold = gold_by_source[target.source_sha256]
        baseline_report = evaluate_document(
            gold, baseline_by_source[target.source_sha256], document_key=target.source_sha256
        )
        candidate_report = evaluate_document(
            gold, candidate_by_source[target.source_sha256], document_key=target.source_sha256
        )
        document_reports[(target.source_sha256, "BASELINE")] = baseline_report
        document_reports[(target.source_sha256, "UVDOC")] = candidate_report
        proof = proof_by_key[(target.source_sha256, target.page_number, "UVDOC")]
        if proof.output_result_sha256 != _actual_identity(
            candidate_by_source[target.source_sha256]
        ):
            raise ValueError("downstream proof output identity differs from candidate result")
        if proof.authority_report_sha256 != _digest(candidate_report.to_dict()):
            raise ValueError("downstream proof authority report differs from recomputed result")
        for branch, actual, authority_report in (
            ("BASELINE", baseline_by_source[target.source_sha256], baseline_report),
            ("UVDOC", candidate_by_source[target.source_sha256], candidate_report),
        ):
            critical, grounding, unreadable = _report_error_ids(authority_report)
            observations.append(
                UvdocAccuracyObservationV2(
                    source_sha256=target.source_sha256,
                    page_number=target.page_number,
                    cohort=target.cohort,
                    branch=branch,
                    gold_sha256=_gold_identity(gold),
                    actual_result_sha256=_actual_identity(actual),
                    authority_report_sha256=_digest(authority_report.to_dict()),
                    authority_pair_identity_sha256=authority_report.pair_identity_sha256,
                    metrics=_document_metric_observations(authority_report, cohort=target.cohort),
                    critical_error_ids=critical,
                    grounding_error_ids=grounding,
                    unreadable_error_ids=unreadable,
                    baseline_failure_ids=target.failure_ids if branch == "BASELINE" else (),
                    downstream_proof_sha256=(
                        proof.proof_sha256
                        if branch == "UVDOC"
                        else None
                    ),
                )
            )
    floor_results: list[FloorResultV2] = []
    for cohort in REQUIRED_COHORTS:
        cohort_targets = [item for item in prereg.targets if item.cohort == cohort]
        source_keys = sorted({item.source_sha256 for item in cohort_targets})
        gold_cohort = {key: gold_by_source[key] for key in source_keys}
        for branch, actual_map in (
            ("BASELINE", {key: baseline_by_source[key] for key in source_keys}),
            ("UVDOC", {key: candidate_by_source[key] for key in source_keys}),
        ):
            cohort_report = evaluate_cohort(gold_cohort, actual_map)
            for metric_name, threshold in TABLE_MAGIC_FLOORS.items():
                numerator, denominator = _metric_counts(cohort_report.metrics, metric_name)
                floor_results.append(
                    FloorResultV2(
                        branch=branch,
                        cohort=cohort,
                        metric_name=metric_name,
                        numerator=numerator,
                        denominator=denominator,
                        observed_value=round(numerator / denominator, 12)
                        if denominator
                        else 0.0,
                        threshold=threshold,
                        passed=bool(denominator and numerator / denominator >= threshold),
                    )
                )
    comparisons: list[FailureComparisonV2] = []
    for target in prereg.targets:
        baseline_report = document_reports[(target.source_sha256, "BASELINE")]
        candidate_report = document_reports[(target.source_sha256, "UVDOC")]
        gold = gold_by_source[target.source_sha256]
        for failure_id in target.failure_ids:
            baseline_outcome = _failure_outcome(baseline_report, failure_id, gold=gold)
            candidate_outcome = _failure_outcome(candidate_report, failure_id, gold=gold)
            if baseline_outcome is None or candidate_outcome is None:
                raise ValueError(
                    f"preregistered failure {failure_id} is absent from authority reports"
                )
            bs, be, bg, bc, bge, bur, _ = _outcome_flags(baseline_outcome)
            cs, ce, cg, cc, cge, cur, _ = _outcome_flags(candidate_outcome)
            comparisons.append(
                FailureComparisonV2(
                    source_sha256=target.source_sha256,
                    page_number=target.page_number,
                    cohort=target.cohort,
                    failure_id=failure_id,
                    baseline_status=bs,
                    candidate_status=cs,
                    baseline_exact=be,
                    candidate_exact=ce,
                    baseline_grounded=bg,
                    candidate_grounded=cg,
                    baseline_critical=bc,
                    candidate_critical=cc,
                    baseline_grounding_error=bge,
                    candidate_grounding_error=cge,
                    baseline_unreadable_mishandled=bur,
                    candidate_unreadable_mishandled=cur,
                )
            )
    comparison_keys = {
        (item.source_sha256, item.page_number, item.candidate_branch, item.failure_id)
        for item in comparisons
    }
    expected_comparison_keys = {
        (target.source_sha256, target.page_number, "UVDOC", failure_id)
        for target in prereg.targets
        for failure_id in target.failure_ids
    }
    if comparison_keys != expected_comparison_keys:
        raise ValueError("failure comparisons do not exactly cover preregistered failures")
    candidate_observations = {
        (item.source_sha256, item.page_number): item
        for item in observations
        if item.branch == "UVDOC"
    }
    resolved_by_target: dict[tuple[str, int], tuple[str, ...]] = {}
    for key, observation in candidate_observations.items():
        resolved_by_target[key] = tuple(
            comparison.failure_id
            for comparison in comparisons
            if (comparison.source_sha256, comparison.page_number) == key and comparison.resolved
        )
        if resolved_by_target[key]:
            object.__setattr__(
                observation,
                "resolved_failure_ids",
                resolved_by_target[key],
            )
            object.__setattr__(
                observation,
                "observation_sha256",
                _hash_payload(
                    "accuracy_observation",
                    observation.model_dump(mode="python", exclude={"observation_sha256"}),
                ),
            )
    report = UvdocAccuracyReportV2(
        preregistration_sha256=prereg.preregistration_sha256,
        source_manifest_sha256=prereg.source_manifest_sha256,
        gold_manifest_sha256=prereg.gold_manifest_sha256,
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        baseline_manifest_sha256=baseline.manifest_sha256,
        baseline_id=baseline.baseline_id,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        candidate_revision=prereg.candidate_revision,
        candidate_configuration_sha256=prereg.candidate_configuration_sha256,
        model_sha256=prereg.model_sha256,
        model_config_sha256=prereg.model_config_sha256,
        adapter_config_sha256=prereg.adapter_config_sha256,
        floor_policy_sha256=prereg.floor_policy_sha256,
        targets=prereg.targets,
        observations=tuple(observations),
        floors=tuple(floor_results),
        proofs=proof_models,
        authority_cohorts=authority_cohort_models,
        comparisons=tuple(comparisons),
    )
    return report


def _coerce_proof(value: Any) -> DownstreamAblationProofV2:
    return DownstreamAblationProofV2.model_validate(value)


def _hold_report(
    report: UvdocAccuracyReportV2,
    reasons: Sequence[str],
) -> M0M4GateReportV1:
    reason_set = tuple(sorted(set(reasons))) or ("authoritative_accuracy_hold",)
    cohorts = tuple(
        GateCohortResultV1(
            cohort=item.cohort,
            required_count=item.required_count,
            observed_count=item.observed_count,
            passed=False,
            metrics=item.metrics,
            blocking_reasons=tuple(sorted(set(item.blocking_reasons) | set(reason_set))),
        )
        for item in report.authority_cohorts
    )
    return M0M4GateReportV1(
        milestone="M4",
        decision="hold",
        source_manifest_sha256=report.source_manifest_sha256,
        gold_manifest_sha256=report.gold_manifest_sha256,
        evaluator_manifest_sha256=report.evaluator_manifest_sha256,
        baseline_manifest_sha256=report.baseline_manifest_sha256,
        candidate_revision=report.candidate_revision,
        candidate_configuration_sha256=report.candidate_configuration_sha256,
        cohorts=cohorts,
        metrics=(),
        evidence_sha256s=(report.report_sha256,),
        blocking_reasons=reason_set,
    )


def evaluate_uvdoc_gate_v2(
    preregistration: UvdocPreregistrationV2 | Mapping[str, Any],
    report: UvdocAccuracyReportV2 | Mapping[str, Any],
    *,
    evaluator_manifest: EvaluatorManifestV1 | Mapping[str, Any],
    baseline_manifest: BaselineManifestV1 | Mapping[str, Any],
    gold_documents: Mapping[str, Any],
    authority_cohorts: Sequence[GateCohortResultV1 | Mapping[str, Any]],
    proofs: Sequence[DownstreamAblationProofV2 | Mapping[str, Any]],
    baseline_documents: Mapping[str, Any] | None = None,
    candidate_documents: Mapping[str, Any] | None = None,
    baseline_results: Mapping[str, Any] | None = None,
    candidate_results: Mapping[str, Any] | None = None,
) -> M0M4GateReportV1:
    """Recompute v2 evidence and return a typed M4 shadow gate report.

    The submitted report is only a claim.  Promotion is possible only when the
    complete report recomputed from the supplied gold/results/proof objects is
    byte-for-byte equal to that claim.
    """

    preregistration_v2 = _coerce_preregistration(preregistration)
    report_v2 = _coerce_report(report)
    try:
        recomputed = evaluate_uvdoc_accuracy_v2(
            preregistration=preregistration_v2,
            evaluator_manifest=evaluator_manifest,
            baseline_manifest=baseline_manifest,
            gold_documents=gold_documents,
            baseline_documents=baseline_documents,
            candidate_documents=candidate_documents,
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            authority_cohorts=authority_cohorts,
            proofs=proofs,
        )
    except (TypeError, ValueError) as error:
        return _hold_report(report_v2, ("recomputation_invalid", str(error)))
    if (
        report_v2.report_sha256 != recomputed.report_sha256
        or report_v2.model_dump(mode="json") != recomputed.model_dump(mode="json")
    ):
        return _hold_report(report_v2, ("recomputed_report_mismatch",))
    reasons: list[str] = []
    if report_v2.preregistration_sha256 != preregistration_v2.preregistration_sha256:
        reasons.append("preregistration_identity_mismatch")
    if report_v2.targets != preregistration_v2.targets:
        reasons.append("target_set_mismatch")
    for field_name in (
        "source_manifest_sha256",
        "gold_manifest_sha256",
        "evaluator_manifest_sha256",
        "baseline_manifest_sha256",
        "floor_policy_sha256",
        "candidate_revision",
        "candidate_configuration_sha256",
        "model_sha256",
        "model_config_sha256",
        "adapter_config_sha256",
    ):
        if getattr(report_v2, field_name) != getattr(preregistration_v2, field_name):
            reasons.append(f"{field_name}_mismatch")
    try:
        evaluator = (
            _coerce_evaluator(evaluator_manifest) if evaluator_manifest is not None else None
        )
        baseline = _coerce_baseline(baseline_manifest) if baseline_manifest is not None else None
    except (TypeError, ValueError) as error:
        return _hold_report(report_v2, ("manifest_invalid", str(error)))
    if evaluator is None:
        reasons.append("evaluator_manifest_missing")
    else:
        if evaluator.manifest_sha256 != report_v2.evaluator_manifest_sha256:
            reasons.append("evaluator_manifest_identity_mismatch")
        if evaluator.evaluator_id != report_v2.evaluator_id:
            reasons.append("evaluator_id_mismatch")
        if evaluator.evaluator_version != report_v2.evaluator_version:
            reasons.append("evaluator_version_mismatch")
        if evaluator.source_manifest_sha256 != report_v2.source_manifest_sha256:
            reasons.append("evaluator_source_manifest_mismatch")
        if evaluator.gold_manifest_sha256 != report_v2.gold_manifest_sha256:
            reasons.append("evaluator_gold_manifest_mismatch")
        if not set(TABLE_MAGIC_FLOORS).issubset(set(evaluator.metric_names)):
            reasons.append("evaluator_metric_set_incomplete")
    if baseline is None:
        reasons.append("baseline_manifest_missing")
    else:
        if baseline.manifest_sha256 != report_v2.baseline_manifest_sha256:
            reasons.append("baseline_manifest_identity_mismatch")
        if baseline.baseline_id != report_v2.baseline_id:
            reasons.append("baseline_id_mismatch")
        if baseline.source_manifest_sha256 != report_v2.source_manifest_sha256:
            reasons.append("baseline_source_manifest_mismatch")
        if baseline.gold_manifest_sha256 != report_v2.gold_manifest_sha256:
            reasons.append("baseline_gold_manifest_mismatch")
        if baseline.evaluator_manifest_sha256 != report_v2.evaluator_manifest_sha256:
            reasons.append("baseline_evaluator_manifest_mismatch")
        expected_sources = {item.source_sha256 for item in preregistration_v2.targets}
        baseline_sources = {item.source_sha256 for item in baseline.documents}
        if baseline_sources != expected_sources:
            reasons.append("baseline_source_set_mismatch")
        if any(item.status != "complete" for item in baseline.documents):
            reasons.append("baseline_document_incomplete")
        baseline_results = {
            (item.source_sha256, item.result_sha256)
            for item in baseline.documents
        }
        for observation in report_v2.observations:
            if observation.branch == "BASELINE" and (
                observation.source_sha256,
                observation.actual_result_sha256,
            ) not in baseline_results:
                reasons.append("baseline_result_identity_mismatch")
    for authority_cohort in report_v2.authority_cohorts:
        if not authority_cohort.passed:
            reasons.append(f"authority_cohort_not_passed:{authority_cohort.cohort}")
        if authority_cohort.blocking_reasons:
            reasons.append(f"authority_cohort_blocked:{authority_cohort.cohort}")
        authority_metrics = {
            item.metric_name: item for item in authority_cohort.metrics
        }
        for metric_name, threshold in TABLE_MAGIC_FLOORS.items():
            metric = authority_metrics.get(metric_name)
            if metric is None or metric.denominator is None or metric.denominator <= 0:
                reasons.append(f"authority_floor_denominator_invalid:{authority_cohort.cohort}:{metric_name}")
            elif metric.value < threshold:
                reasons.append(f"authority_floor_failed:{authority_cohort.cohort}:{metric_name}")
    if gold_documents is not None:
        try:
            gold_by_source = {
                str(key): GoldDocumentV2.model_validate(value)
                for key, value in gold_documents.items()
            }
            expected_sources = {item.source_sha256 for item in preregistration_v2.targets}
            if set(gold_by_source) != expected_sources:
                reasons.append("gold_source_set_mismatch")
            for observation in report_v2.observations:
                gold = gold_by_source.get(observation.source_sha256)
                if gold is None or gold.gold_sha256 != observation.gold_sha256:
                    reasons.append("gold_document_identity_mismatch")
        except (TypeError, ValueError):
            reasons.append("gold_document_invalid")
    # Recheck canonical hashes after model construction; model_copy/update can
    # otherwise leave a stale sealed hash in memory.
    if report_v2.report_sha256 != _hash_payload(
        "accuracy_report", report_v2.model_dump(mode="python", exclude={"report_sha256"})
    ):
        reasons.append("accuracy_report_hash_mismatch")
    if preregistration_v2.floor_policy_sha256 != FLOOR_POLICY_SHA256:
        reasons.append("floor_policy_mismatch")
    if not reasons:
        for cohort in report_v2.required_cohorts:
            baseline_floors = _floor_map(report_v2, "BASELINE", cohort)
            candidate_floors = _floor_map(report_v2, "UVDOC", cohort)
            for metric_name in TABLE_MAGIC_FLOORS:
                baseline_floor = baseline_floors.get(metric_name)
                candidate_floor = candidate_floors.get(metric_name)
                if baseline_floor is None or candidate_floor is None:
                    reasons.append(f"missing_floor:{cohort}:{metric_name}")
                    continue
                if not baseline_floor.passed:
                    reasons.append(f"baseline_floor_failed:{cohort}:{metric_name}")
                if not candidate_floor.passed:
                    reasons.append(f"candidate_floor_failed:{cohort}:{metric_name}")
                if cohort == "flat" and not _same_or_better(candidate_floor, baseline_floor):
                    reasons.append(f"flat_regression:{metric_name}")
        baseline_by_key = {
            (item.source_sha256, item.page_number): item
            for item in report_v2.observations
            if item.branch == "BASELINE"
        }
        candidate_by_key = {
            (item.source_sha256, item.page_number): item
            for item in report_v2.observations
            if item.branch == "UVDOC"
        }
        for key, candidate in candidate_by_key.items():
            baseline = baseline_by_key[key]
            for field_name, reason in (
                ("critical_error_ids", "new_critical_error"),
                ("grounding_error_ids", "grounding_regression"),
                ("unreadable_error_ids", "unreadable_regression"),
            ):
                added = set(getattr(candidate, field_name)) - set(getattr(baseline, field_name))
                if added:
                    reasons.append(f"{reason}:{candidate.source_sha256}:{candidate.page_number}")
        for comparison in report_v2.comparisons:
            if comparison.candidate_critical and not comparison.baseline_critical:
                reasons.append(f"new_critical_failure:{comparison.failure_id}")
            if comparison.candidate_grounding_error and not comparison.baseline_grounding_error:
                reasons.append(f"grounding_regression:{comparison.failure_id}")
            if (
                comparison.candidate_unreadable_mishandled
                and not comparison.baseline_unreadable_mishandled
            ):
                reasons.append(f"unreadable_regression:{comparison.failure_id}")
            if (
                comparison.cohort == "flat"
                and comparison.baseline_exact
                and not comparison.candidate_exact
            ):
                reasons.append(f"flat_regression:{comparison.failure_id}")
            if (
                comparison.cohort == "flat"
                and comparison.baseline_grounded
                and not comparison.candidate_grounded
            ):
                reasons.append(f"flat_grounding_regression:{comparison.failure_id}")
        curved_resolutions = {
            comparison.failure_id
            for comparison in report_v2.comparisons
            if comparison.cohort == "curved"
            and comparison.candidate_branch == "UVDOC"
            and comparison.resolved
            and any(
                target.source_sha256 == comparison.source_sha256
                and target.page_number == comparison.page_number
                and comparison.failure_id in target.failure_ids
                for target in preregistration_v2.targets
            )
        }
        if not curved_resolutions:
            reasons.append("no_preregistered_curved_improvement")
    if reasons:
        return _hold_report(report_v2, reasons)
    m4_metrics = tuple(
        MetricObservationV1(
            metric_name=item.metric_name,
            value=item.observed_value,
            numerator=item.numerator,
            denominator=item.denominator,
            cohort=item.cohort,
        )
        for item in report_v2.floors
        if item.branch == "UVDOC"
    )
    if not m4_metrics:
        return _hold_report(report_v2, ("recomputed_candidate_floors_missing",))
    evidence = tuple(
        sorted(
            {
                report_v2.report_sha256,
                *(proof.proof_sha256 for proof in report_v2.proofs),
                *(observation.observation_sha256 for observation in report_v2.observations),
            }
        )
    )
    return M0M4GateReportV1(
        milestone="M4",
        decision="promote",
        source_manifest_sha256=report_v2.source_manifest_sha256,
        gold_manifest_sha256=report_v2.gold_manifest_sha256,
        evaluator_manifest_sha256=report_v2.evaluator_manifest_sha256,
        baseline_manifest_sha256=report_v2.baseline_manifest_sha256,
        candidate_revision=report_v2.candidate_revision,
        candidate_configuration_sha256=report_v2.candidate_configuration_sha256,
        cohorts=report_v2.authority_cohorts,
        metrics=m4_metrics,
        evidence_sha256s=evidence,
    )


__all__ = [
    "BRANCHES",
    "FLOOR_POLICY_SHA256",
    "FLOOR_POLICY_VERSION",
    "PROMOTE_MEANS_SHADOW_ONLY",
    "PREREGISTRATION_VERSION",
    "PROOF_VERSION",
    "REPORT_VERSION",
    "TABLE_MAGIC_FLOORS",
    "FailureComparisonV2",
    "FloorResultV2",
    "DownstreamAblationProofV2",
    "UvdocAccuracyObservationV2",
    "UvdocAccuracyReportV2",
    "UvdocPreregistrationV2",
    "UvdocTargetV2",
    "evaluate_uvdoc_accuracy_v2",
    "evaluate_uvdoc_gate_v2",
    "floor_policy_sha256",
]
