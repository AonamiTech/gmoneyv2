from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import fitz
import pytest
from typer.testing import CliRunner

from gmoney.contracts.authority import (
    GoldDocumentV2,
    ReviewDecision,
    ReviewKind,
    ReviewRecordV1,
    ReviewState,
    SourceClass,
)
from gmoney.evaluation import authority_cli

RUNNER = CliRunner()


def _pdf(path: Path, text: str = "bill") -> None:
    document = fitz.open()
    page = document.new_page(width=100, height=100)
    page.insert_text((10, 20), text)
    document.save(path)
    document.close()


def _run(vault: Path, *args: str):
    return RUNNER.invoke(authority_cli.app, ["--vault", str(vault), *args])


def _identity() -> dict[str, object]:
    return {
        "release_revision": "1" * 40,
        "images": {"api": "sha256:" + "2" * 64},
        "models": {"worker": "3" * 64},
        "config_sha256": "4" * 64,
        "corpus_sha256": "6" * 64,
        "gold_sha256": "7" * 64,
        "evaluator_sha256": "5" * 64,
    }


def _authority_envelope(
    source_sha256: str, page: dict[str, object], manifest_sha256: str
) -> dict[str, object]:
    """Build one minimal, contract-valid V2 gold plus four frozen reviews."""

    gold = GoldDocumentV2(
        source_sha256=source_sha256,
        source_manifest_sha256="a" * 64,
        page_count=1,
        pages=[
            {
                "source_sha256": source_sha256,
                "page_number": 1,
                "artifact_sha256": page["sha256"],
                "width": page["width"],
                "height": page["height"],
                "dpi": 300,
                "source_class": SourceClass.NATIVE_TEXT_PDF,
            }
        ],
        annotation_group_id="authority-cli-test",
        split="staging",
    )
    common = {
        "document_id": gold.document_id,
        "source_sha256": source_sha256,
        "reviewer_identity": "reviewer",
        "model_identity": "human-review",
        "tool_identity": "authority-cli-test",
        "reasoning_identity": "reasoning-v1",
        "prompt_identity": "prompt-v1",
        "prompt_sha256": "b" * 64,
        "image_identities": ("page-1",),
        "image_sha256s": (page["sha256"],),
        "image_manifest_sha256": manifest_sha256,
        "model_config_sha256": "c" * 64,
        "observations_sha256": "d" * 64,
        "decision": ReviewDecision.ACCEPT,
        "state": ReviewState.FROZEN,
        "frozen_at": datetime(2026, 9, 7, tzinfo=UTC),
        "frozen_by": "authority-cli-test",
    }

    def review(kind: ReviewKind, reviewer: str, **overrides: object) -> ReviewRecordV1:
        values = {**common, "review_kind": kind, "reviewer_identity": reviewer, **overrides}
        return ReviewRecordV1(**values)

    independent_a = review(ReviewKind.INDEPENDENT_A, "reviewer-a")
    independent_b = review(ReviewKind.INDEPENDENT_B, "reviewer-b")
    adjudicator = review(
        ReviewKind.ADJUDICATOR,
        "adjudicator",
        blind=False,
        independent_first=True,
        independent_review_ids=(independent_a.review_id, independent_b.review_id),
    )
    red_team = review(ReviewKind.RED_TEAM, "red-team")
    return {
        "gold_document": gold.model_dump(mode="json"),
        "review_records": [
            record.model_dump(mode="json")
            for record in (independent_a, independent_b, adjudicator, red_team)
        ],
    }


def test_vault_must_be_outside_repository(tmp_path: Path) -> None:
    with pytest.raises(authority_cli.AuthorityError, match="outside the repository"):
        authority_cli._vault_path(Path.cwd() / "authority-test-vault")


def test_inventory_render_review_and_seal_are_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authority_cli,
        "COHORT_COUNTS",
        {"production14": 1, "passing36": 1, "staging159": 1},
    )
    vault = tmp_path / "vault"
    roots = {}
    master_source = tmp_path / "master.pdf"
    _pdf(master_source, "same source bytes")
    for name in authority_cli.COHORT_COUNTS:
        root = tmp_path / name
        root.mkdir()
        shutil.copyfile(master_source, root / "bill.pdf")
        roots[name] = root

    inventory = _run(
        vault,
        "inventory",
        "--cohort",
        f"production14={roots['production14']}",
        "--cohort",
        f"passing36={roots['passing36']}",
        "--cohort",
        f"staging159={roots['staging159']}",
    )
    assert inventory.exit_code == 0, inventory.stdout
    rendered = _run(vault, "render")
    assert rendered.exit_code == 0, rendered.stdout
    rerendered = _run(vault, "render")
    assert rerendered.exit_code == 0, rerendered.stdout
    render_report = json.loads((vault / "control" / "render-manifest.json").read_text())
    assert render_report["dpi"] == 300
    assert render_report["colorspace"] == "sRGB"
    page = render_report["documents"][0]["pages"][0]
    assert len(page["sha256"]) == 64
    assert ImageMode(vault, page["relative_path"]) == "RGB"

    inventory_payload = json.loads((vault / "control" / "inventory.json").read_text())
    assert len(inventory_payload["documents"]) == 1
    assert inventory_payload["documents"][0]["cohorts"] == list(authority_cli.COHORT_COUNTS)

    gold_root = vault / "gold"
    gold_root.mkdir()
    for item in inventory_payload["documents"]:
        source_sha = item["source_sha256"]
        rendered_document = next(
            doc
            for doc in render_report["documents"]
            if doc["document_sha256"] == source_sha
        )
        payload = _authority_envelope(
            source_sha, rendered_document["pages"][0], rendered_document["manifest_sha256"]
        )
        (gold_root / f"{source_sha}.json").write_text(json.dumps(payload, sort_keys=True))
    reviewed = _run(vault, "validate-review")
    assert reviewed.exit_code == 0, reviewed.stdout

    identity_file = tmp_path / "identity.json"
    identity_file.write_text(json.dumps(_identity()))
    sealed = _run(vault, "seal", "--identity", str(identity_file))
    assert sealed.exit_code == 0, sealed.stdout
    sealed_payload = json.loads((vault / "control" / "authority-seal.json").read_text())
    assert sealed_payload["cohort_counts"] == authority_cli.COHORT_COUNTS
    assert len(sealed_payload["documents"]) == 1
    assert sealed_payload["documents"][0]["cohorts"] == list(authority_cli.COHORT_COUNTS)

    review_report = json.loads((vault / "control" / "review-report.json").read_text())
    summary = review_report["annotations"][0]
    review_object = vault / summary["review_record_relpaths"][0]
    review_object.write_text("{}")
    with pytest.raises(authority_cli.AuthorityError, match="review object hash mismatch"):
        authority_cli._validated_review_objects(
            vault,
            summary,
            source_sha256=summary["source_sha256"],
            rendered_document=render_report["documents"][0],
        )


def ImageMode(vault: Path, relative_path: str) -> str:
    from PIL import Image

    with Image.open(vault / relative_path) as image:
        return image.mode


def test_inventory_and_render_allow_unassigned_partial_sources_but_legacy_gold_is_rejected(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "bill.pdf"
    _pdf(source)
    inventory = _run(vault, "inventory", "--source-root", str(source_root))
    assert inventory.exit_code == 0, inventory.stdout
    rendered = _run(vault, "render")
    assert rendered.exit_code == 0, rendered.stdout
    gated = _run(vault, "gate-m0-m4")
    assert gated.exit_code == 1
    gate_report = json.loads((vault / "control" / "gate-m0-m4.json").read_text())
    assert gate_report["m0_m4_passed"] is False
    assert "m0" in gate_report["blocking_milestones"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    review_root = vault / "gold"
    review_root.mkdir()
    (review_root / "bad.json").write_text(
        json.dumps(
            {
                "review_status": "pending",
                "bill_file": "bill.pdf",
                "document_sha256": digest,
                "rows": [],
                "image_review": {
                    "reviewer": "reviewer",
                    "method": "human",
                    "passes": 1,
                    "reviewed_at": "2026-09-07T00:00:00Z",
                    "page_asset_sha256": ["0" * 64],
                },
            }
        )
    )
    result = _run(vault, "validate-review")
    assert result.exit_code != 0
    error = str(result.exception).lower()
    assert "golddocumentv2" in error or "reviewrecordv1" in error


def test_inventory_deduplicates_content_within_one_root(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    first = source_root / "first.pdf"
    second = source_root / "second.pdf"
    _pdf(first)
    shutil.copyfile(first, second)
    result = _run(vault, "inventory", "--source-root", str(source_root))
    assert result.exit_code == 0, result.stdout
    payload = json.loads((vault / "control" / "inventory.json").read_text())
    assert len(payload["documents"]) == 1
    assert payload["unassigned_count"] == 1


def test_inventory_accepts_repeatable_partial_source_roots(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "first.pdf"
    duplicate = second_root / "same.pdf"
    other = second_root / "other.pdf"
    _pdf(first, "same bytes")
    shutil.copyfile(first, duplicate)
    _pdf(other, "different bytes")
    result = _run(
        vault,
        "inventory",
        "--source-root",
        str(first_root),
        "--source-root",
        str(second_root),
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((vault / "control" / "inventory.json").read_text())
    assert len(payload["documents"]) == 2
    assert payload["unassigned_count"] == 2


def test_inventory_merges_repeated_cohort_roots(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    first_root = tmp_path / "production-a"
    second_root = tmp_path / "production-b"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "same-a.pdf"
    duplicate = second_root / "same-b.pdf"
    other = second_root / "other.pdf"
    _pdf(first, "same bytes")
    shutil.copyfile(first, duplicate)
    _pdf(other, "other bytes")
    result = _run(
        vault,
        "inventory",
        "--cohort",
        f"production14={first_root}",
        "--cohort",
        f"production14={second_root}",
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((vault / "control" / "inventory.json").read_text())
    assert len(payload["documents"]) == 2
    assert payload["cohort_counts"]["production14"] == 2


def test_baseline_requires_identity_and_identical_replays(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    payload = {"identity": _identity(), "metrics": {"passed": True}}
    first.write_text(json.dumps(payload, sort_keys=True))
    second.write_text(json.dumps(payload, sort_keys=True))
    result = _run(
        vault,
        "baseline",
        "--replay",
        str(first),
        "--replay",
        str(second),
        "--identity",
        str(identity),
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads((vault / "control" / "baseline.json").read_text())
    assert report["replay_identical"] is True
    assert len(report["replay_digests"]) == 2

    second.write_text(json.dumps({**payload, "changed": True}, sort_keys=True))
    failed = _run(
        vault,
        "baseline",
        "--replay",
        str(first),
        "--replay",
        str(second),
        "--identity",
        str(identity),
        "--output",
        str(tmp_path / "different-baseline.json"),
    )
    assert failed.exit_code != 0
    assert "byte-identical" in str(failed.exception)


def test_machine_output_cannot_supply_authority_gold(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    payload = {"identity": _identity(), "gold": {}, "actual": {}}
    first.write_text(json.dumps(payload, sort_keys=True))
    second.write_text(json.dumps(payload, sort_keys=True))

    result = _run(
        vault,
        "baseline",
        "--replay",
        str(first),
        "--replay",
        str(second),
        "--identity",
        str(identity),
    )

    assert result.exit_code != 0
    assert "may not supply authority-controlled gold" in str(result.exception)


def _eligibility_manifest(paths: list[Path], *, synthetic: bool = False) -> dict[str, object]:
    return {
        "manifest_version": "working152_eligibility_v1",
        "documents": [
            {
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "eligible": True,
                "synthetic": synthetic,
                "duplicate": False,
            }
            for path in paths
        ],
    }


def test_working_intake_requires_explicit_safe_decisions_and_stays_non_authoritative(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "working"
    source_root.mkdir()
    sources = [source_root / "a.pdf", source_root / "b.pdf"]
    _pdf(sources[0], "first")
    _pdf(sources[1], "second")
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text(json.dumps(_eligibility_manifest(sources)))
    vault = tmp_path / "vault"

    result = _run(
        vault,
        "intake",
        "--source-root",
        str(source_root),
        "--eligibility",
        str(eligibility),
        "--expected-count",
        "2",
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((vault / "control" / "working152-inventory.json").read_text())
    assert payload["inventory_version"] == authority_cli.WORKING_INVENTORY_VERSION
    assert payload["authority_status"] == "non_authoritative_working152"
    assert payload["authority_claim"] == "never_authoritative"
    assert payload["intake"]["eligible_count"] == 2
    assert payload["intake"]["synthetic_count"] == 0
    assert payload["intake"]["duplicate_count"] == 0

    blocked = _run(
        vault,
        "seal",
        "--inventory",
        str(vault / "control" / "working152-inventory.json"),
    )
    assert blocked.exit_code != 0
    assert "non-authoritative" in str(blocked.exception)


def test_working_intake_rejects_synthetic_and_duplicate_bytes(tmp_path: Path) -> None:
    source_root = tmp_path / "working"
    source_root.mkdir()
    source = source_root / "source.pdf"
    _pdf(source)
    eligibility = tmp_path / "synthetic.json"
    eligibility.write_text(json.dumps(_eligibility_manifest([source], synthetic=True)))
    synthetic = _run(
        tmp_path / "synthetic-vault",
        "intake",
        "--source-root",
        str(source_root),
        "--eligibility",
        str(eligibility),
        "--expected-count",
        "1",
    )
    assert synthetic.exit_code != 0
    assert "synthetic" in str(synthetic.exception).lower()

    duplicate_root = tmp_path / "duplicates"
    duplicate_root.mkdir()
    shutil.copyfile(source, duplicate_root / "same.pdf")
    duplicate_eligibility = tmp_path / "duplicate.json"
    duplicate_eligibility.write_text(
        json.dumps(_eligibility_manifest([source, duplicate_root / "same.pdf"]))
    )
    duplicate = _run(
        tmp_path / "duplicate-vault",
        "intake",
        "--source-root",
        str(source_root),
        "--source-root",
        str(duplicate_root),
        "--eligibility",
        str(duplicate_eligibility),
        "--expected-count",
        "2",
    )
    assert duplicate.exit_code != 0
    assert "duplicate" in str(duplicate.exception).lower()


def test_nested_assignment_is_deterministic_and_rejects_non_nested_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authority_cli,
        "COHORT_COUNTS",
        {"production14": 1, "passing36": 2, "staging159": 3},
    )
    vault = tmp_path / "vault"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    paths = [source_root / f"{name}.pdf" for name in ("c", "a", "b")]
    for path in paths:
        _pdf(path, path.stem)
    inventory = _run(vault, "inventory", "--source-root", str(source_root))
    assert inventory.exit_code == 0, inventory.stdout
    digests = sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    assigned = _run(
        vault,
        "assign",
        "--production14",
        digests[0],
        "--passing36",
        ",".join(digests[:2]),
        "--staging159",
        ",".join(digests),
    )
    assert assigned.exit_code == 0, assigned.stdout
    payload = json.loads((vault / "control" / "assigned-inventory.json").read_text())
    assert payload["membership_complete"] is True
    assert payload["assignment"]["strategy"] == "explicit_source_sha256"
    assert [item["source_sha256"] for item in payload["documents"]] == digests
    for item in payload["documents"]:
        if item["source_sha256"] == digests[0]:
            assert item["cohorts"] == ["production14", "passing36", "staging159"]
        elif item["source_sha256"] == digests[1]:
            assert item["cohorts"] == ["passing36", "staging159"]
        else:
            assert item["cohorts"] == ["staging159"]

    non_nested = _run(
        vault,
        "assign",
        "--production14",
        digests[0],
        "--passing36",
        digests[1],
        "--staging159",
        ",".join(digests),
        "--output",
        str(tmp_path / "bad-assignment.json"),
    )
    assert non_nested.exit_code != 0
    assert "subset" in str(non_nested.exception)


def test_review_queue_and_readiness_are_explicitly_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authority_cli,
        "COHORT_COUNTS",
        {"production14": 1, "passing36": 1, "staging159": 1},
    )
    vault = tmp_path / "vault"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "bill.pdf"
    _pdf(source)
    inventory = _run(
        vault,
        "inventory",
        "--cohort",
        f"production14={source_root}",
        "--cohort",
        f"passing36={source_root}",
        "--cohort",
        f"staging159={source_root}",
    )
    assert inventory.exit_code == 0, inventory.stdout
    rendered = _run(vault, "render")
    assert rendered.exit_code == 0, rendered.stdout
    queued = _run(vault, "review-queue")
    assert queued.exit_code == 0, queued.stdout
    queue = json.loads((vault / "control" / "review-queues.json").read_text())
    assert queue["authority_status"] == "non_authoritative_review_queue"
    assert queue["review_kinds"] == list(authority_cli.REVIEW_KINDS)
    assert len(queue["passes"]) == 4
    assert all(item["machine_output_allowed"] is False for item in queue["passes"])
    assert _run(vault, "validate-review-queue").exit_code == 0

    readiness = _run(vault, "seal-readiness", "--output", str(tmp_path / "readiness.json"))
    assert readiness.exit_code == 1
    report = json.loads((tmp_path / "readiness.json").read_text())
    assert report["seal_ready"] is False
    assert report["seal_claim_allowed"] is False
    assert report["authority_claim"] == "no_authority_seal_claimed"
    assert report["review_coverage"]["full_four_pass_complete"] is False
