from gmoney.evaluation.corpus import grouped_splits, infer_hospital_id


def test_hospital_groups_never_cross_splits() -> None:
    groups = ["H1", "H1", "H2", "H3", "H4", "H5"]
    assignments = grouped_splits(groups, "frozen-seed")
    assert set(assignments) == set(groups)
    assert {assignments[group] for group in set(groups)} >= {"train", "validation", "holdout"}


def test_hospital_id_is_extracted_or_stably_anonymized() -> None:
    assert infer_hospital_id("Hospital_H560041N0015_bill.gold.json") == "H560041N0015"
    assert infer_hospital_id("unknown.pdf") == infer_hospital_id("unknown.pdf")

