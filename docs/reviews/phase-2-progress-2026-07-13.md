# Phase 2 Pause / Resume Record — 2026-07-13

This is a work-in-progress save point, not the Phase 2 checkpoint. Do not create the
`checkpoint-phase-2` tag until the sealed holdout gate passes under both frozen evaluators.

## Repository state

- Resume commit before this record: `0b6344b` (`feat: build phase 2 extraction spine`)
- Phase 0 checkpoint: `5af6545`, tag `checkpoint-phase-0`
- Phase 1 checkpoint: `bfd2b1a`, tag `checkpoint-phase-1`
- Working directory was clean before this record was added.
- Full verification at the save point: 39 tests passed; Ruff passed.
- Repository secret scan found no Gemini key or populated `GEMINI_API_KEY`.

## Implemented Phase 2 slice

- OTSL table parsing and repeated-header table splitting
- Generalized header/column inference
- Strict typed financial parsing with Indian-number support
- Detail, subtotal, document-total, payment, deposit, refund, metadata, and zero-placeholder roles
- Adjacent description/numeric continuation reconstruction
- PP-OCRv6 token normalization with page-coordinate evidence
- Monotonic duplicate-safe spatial alignment
- Evidence-gated canonical rows and deterministic deduplication
- Offline PDF-to-machine-output orchestration with up to three concurrent VLM table crops
- Content-bound OCR, layout, and VLM stage caches keyed by immutable artifact hash, model spec,
  and inference options

## Accuracy observed so far

| Split | Bill | Gold rows | Actual rows | Precision | Recall | Evaluators |
|---|---:|---:|---:|---:|---:|---|
| Train | `7030777110.pdf` | 9 | 9 | 100% | 100% | legacy v1 and canonical v2 |
| Validation | `9739459460.pdf` | 7 | 7 | 100% | 100% | legacy v1 and canonical v2 |

The first validation pass had 100% recall and 58.33% precision. Error review found only
deterministic role/continuation defects: prefixed subtotals, a zero-value placeholder, and a
description/value pair split across adjacent rows. General rules fixed those defects; no
hospital-specific text or profile was added.

The second validation run demonstrated a fully cached repeat in about seven seconds, with OCR,
layout, and VLM cache hits all true.

## Interrupted work

The larger validation bill `9986117545.pdf` was intentionally interrupted at the user's request.
No machine-output file was completed. The following verified cache entries remain and will be
reused:

- `data/phase2/9986117545/inference/page-1.ocr.json`
- `data/phase2/9986117545/inference/page-1.layout.json`

Resume from `/home/azureuser/gmoneyv2` with:

```bash
.venv/bin/python -m gmoney.extraction.offline \
  --source '/home/azureuser/07-07-2026 BILLS/ ONCOVILLE PRIVATE LIMITED - H560072N006/9986117545.pdf' \
  --artifact-root data/phase2/9986117545 \
  --output artifacts/phase2/9986117545.machine.json \
  --vl-url http://127.0.0.1:8111
```

Then evaluate it against:

```text
/home/azureuser/gmoney/backend/benchmark_outputs/best_quality_guarded_20260710T0454Z/gold_drafts/ONCOVILLE_PRIVATE_LIMITED_-_H560072N006_9986117545.gold.json
```

## Runtime state at pause

- The API, PostgreSQL, MinIO, Temporal, and Temporal UI containers remain running.
- PaddleOCR-VL remains running on host port 8111 and its host health endpoint returns
  `{"status":"ok"}`.
- Docker marks the VLM container unhealthy because its internal health command cannot connect to
  `localhost:8080`; this health-check mismatch should be corrected before the Phase 2 checkpoint.
- The interrupted extractor process exited with code 130 and is no longer running.

## Next actions

1. Resume and score `9986117545.pdf` using its partial verified cache.
2. Analyze validation failures by the frozen error taxonomy and make only general corrections.
3. Rerun all tests, lint, and secret scanning.
4. Run the sealed Phase 2 holdout exactly once after validation is frozen. Phase 2 extraction has
   not inspected or scored holdout row values; Phase 1 used holdout pages only for table-region
   coverage measurement.
5. If both evaluators report at least 85% holdout row precision and recall, write
   `docs/reviews/phase-2.md`, commit, and create `checkpoint-phase-2`. Otherwise stop with the
   required categorized failure report and do not weaken the gate.
