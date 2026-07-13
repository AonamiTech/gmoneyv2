from gmoney.evaluation.corpus import (
    freeze_phase3_candidates,
    grouped_splits,
    infer_hospital_id,
)


def test_hospital_groups_never_cross_splits() -> None:
    groups = ["H1", "H1", "H2", "H3", "H4", "H5"]
    assignments = grouped_splits(groups, "frozen-seed")
    assert set(assignments) == set(groups)
    assert {assignments[group] for group in set(groups)} >= {"train", "validation", "holdout"}


def test_hospital_id_is_extracted_or_stably_anonymized() -> None:
    assert infer_hospital_id("Hospital_H560041N0015_bill.gold.json") == "H560041N0015"
    assert infer_hospital_id("unknown.pdf") == infer_hospital_id("unknown.pdf")


def test_phase3_samples_remain_blocked_until_identity_and_gold_are_frozen() -> None:
    catalog = {
        "catalog_version": "v1",
        "entries": [
            {
                "kind": "pdf",
                "source": "sample_bills",
                "relative_path": "Sample Bills/Bill 12.pdf",
                "sha256": "a" * 64,
            },
            {
                "kind": "pdf",
                "source": "sample_bills",
                "relative_path": "Sample Bills/Bill 1.pdf",
                "sha256": "b" * 64,
            },
        ],
    }
    manifest = freeze_phase3_candidates(catalog, exposed_sha256="a" * 64)
    assert manifest["entries"][1]["status"] == "exposed_regression"
    candidate = manifest["entries"][0]
    assert candidate["status"] == "candidate_unseen"
    assert not candidate["eligible_for_frozen_gate"]
    assert not manifest["summary"]["gate_ready"]
