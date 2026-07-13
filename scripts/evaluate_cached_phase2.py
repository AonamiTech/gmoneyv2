from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from gmoney.evaluation.metrics import row_view
from gmoney.evaluation.quality import evaluate_phase2_quality
from gmoney.extraction.canonicalize import canonicalize_rows
from gmoney.extraction.ocr_rows import (
    TableSchemaState,
    fuse_provider_descriptions,
    reconstruct_ocr_rows,
)
from gmoney.extraction.ocr_tokens import paddle_ocr_tokens
from gmoney.extraction.offline import (
    _apply_document_role_policy,
    _deduplicate,
    _layout_boxes,
    _merge_table_boxes,
    _safe_box,
)
from gmoney.extraction.otsl import parse_otsl, split_otsl_tables
from gmoney.extraction.rows import extract_candidate_rows
from gmoney.inference.ocr_table_fallback import propose_tables_from_ocr

PAGE_NUMBER = re.compile(r"page-(\d+)")


def _page_number(path: Path) -> int:
    match = PAGE_NUMBER.search(path.name)
    if not match:
        raise ValueError(f"cannot infer page number from {path}")
    return int(match.group(1))


def _envelope_output(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    return envelope["response"]["output"], envelope["artifact_sha256"]


def _provider_rows(path: Path):
    if not path.exists():
        return ()
    output, _ = _envelope_output(path)
    content = str(output.get("content") or "")
    return tuple(
        candidate
        for table in split_otsl_tables(parse_otsl(content))
        for candidate in extract_candidate_rows(table)
    )


def reconstruct_document(artifact_root: Path, document_id: str):
    inference = artifact_root / "inference"
    schemas: list[TableSchemaState] = []
    rows = []
    for ocr_path in sorted(inference.glob("page-*.ocr.json"), key=_page_number):
        page_number = _page_number(ocr_path)
        ocr_output, artifact_sha256 = _envelope_output(ocr_path)
        tokens = paddle_ocr_tokens(ocr_output, page_number, artifact_sha256)
        layout_output, _ = _envelope_output(inference / f"page-{page_number}.layout.json")
        result = (ocr_output.get("pages") or [{}])[0].get("res") or {}
        geometry_boxes = [
            tuple(round(value) for value in proposal.box)
            for proposal in propose_tables_from_ocr(
                result.get("rec_boxes") or [],
                result.get("rec_texts") or [],
            )
        ]
        boxes = _merge_table_boxes(_layout_boxes(layout_output), geometry_boxes)
        for table_index, box in enumerate(boxes, 1):
            table_id = f"p{page_number}-t{table_index}"
            reconstruction = reconstruct_ocr_rows(
                tokens,
                page_number=page_number,
                table_id=table_id,
                box=_safe_box(box, 1_000_000, 1_000_000),
                prior_schemas=tuple(schemas),
            )
            if reconstruction.schema:
                schemas.append(reconstruction.schema)
            aligned = fuse_provider_descriptions(
                reconstruction.rows,
                _provider_rows(inference / f"{table_id}.vl.json"),
            )
            rows.extend(
                canonicalize_rows(
                    document_id,
                    page_number,
                    table_id,
                    artifact_sha256,
                    aligned,
                    starting_order=len(rows),
                )
            )
    return _apply_document_role_policy(_deduplicate(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    corpus_gold = []
    corpus_actual = []
    for document_index, artifact_root in enumerate(sorted(args.cache_root.glob("*-artifacts"))):
        document_id = artifact_root.name.removesuffix("-artifacts")
        gold_paths = list(args.gold_root.glob(f"*{document_id}*.gold.json"))
        if len(gold_paths) != 1:
            raise ValueError(f"expected one gold file for {document_id}, found {gold_paths}")
        gold_payload = json.loads(gold_paths[0].read_text())["rows"]
        actual_rows = reconstruct_document(artifact_root, document_id)
        gold = tuple(row_view(payload, index) for index, payload in enumerate(gold_payload))
        actual = tuple(
            row_view(row.model_dump(mode="json"), index) for index, row in enumerate(actual_rows)
        )
        accepted_ungrounded = sum(
            row.review_disposition.value == "accepted"
            and not {"description", "amount"}.issubset(row.field_evidence)
            for row in actual_rows
        )
        quality = evaluate_phase2_quality(
            gold,
            actual,
            accepted_ungrounded_rows=accepted_ungrounded,
        )
        results.append(
            {
                "document_id": document_id,
                "gold_rows": len(gold),
                "actual_rows": len(actual),
                **quality.to_dict(),
            }
        )
        page_offset = (document_index + 1) * 1000
        corpus_gold.extend(replace(row, page_number=row.page_number + page_offset) for row in gold)
        corpus_actual.extend(
            replace(row, page_number=row.page_number + page_offset) for row in actual
        )

    corpus_quality = evaluate_phase2_quality(
        tuple(corpus_gold),
        tuple(corpus_actual),
        accepted_ungrounded_rows=sum(int(result["accepted_ungrounded_rows"]) for result in results),
    )
    payload = {
        "benchmark_version": "phase2_cached_reconstruction_v2",
        "documents": results,
        "corpus": corpus_quality.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["corpus"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
