# Phase 3 — Targeted Recovery and Known-Layout Profiles

## Summary

Extend the Phase 2 OCR-first pipeline with crop-scoped recovery, an optional
de-identified Gemini challenger, and declarative known-layout profiles. PaddleOCR-VL
remains the first heavy fallback. Gemini is promoted only when a frozen comparison
shows grounded quality improvement. Phase 4 persistence, APIs, UI, and workflow work
remain out of scope.

The target population is 5,000 hospitals with approximately three to four variants
each. Aggregate row precision, recall, and F1 must reach 90%; unseen hospitals must
stay above 85%; any active profile must reach 95%. Every accepted description and
amount remains linked to printed evidence.

## Implementation

- Add versioned recovery, adjudication, route, profile, match, metric, and drift
  contracts.
- Run a targeted recovery ladder only for unresolved crops: higher-resolution render,
  alternate photometric input, crop OCR, PaddleOCR-VL, optional Gemini, then review.
- Send Gemini only a de-identified unresolved crop with OCR token IDs and polygons.
  Reject unknown evidence, out-of-crop geometry, unsupported numerics, invalid schema,
  timeouts, rate limits, and budget exhaustion.
- Cache adjudication by masked-crop, model, prompt, schema, redaction, and settings
  hashes. Keep the API key only in process environment.
- Store profiles as versioned declarative data. Retrieve per page/table using trusted
  hospital identity, page/table type, headers, columns, anchors, and normalized
  geometry. Ambiguous matches use the heavy route.
- Enforce candidate, shadow, active, drifted, and archived states; promotion, pause,
  rollback; deterministic 5% heavy-route sampling; and activation gates without an
  override.
- Keep Phase 3 file-backed and offline. Expose repository and adapter protocols that
  Phase 4 can replace with PostgreSQL and durable workflows.

## Evaluation

- Treat the existing 24 annotated bills and Sample Bill 12 as exposed regression
  data. Reserve ten eligible hospital groups from the other eleven Sample Bills for
  sealed unseen evaluation, supplementing them if identities overlap.
- Require a separate repeat-layout cohort with independent construction data and at
  least ten holdout bills and 200 gold rows before activating a profile.
- Compare local-only, local-plus-Gemini challenger, and profile-fast-route outputs.
- Require both row evaluators to pass; amount accuracy at least 95%; negative recall
  100%; zero accepted ungrounded rows; aggregate bootstrap lower bounds above 85%;
  and at least 95% of ordinary active-profile bills with zero Gemini calls.
- Exercise 20,000 profile variants, cache determinism, provider failure, redaction,
  drift, rollback, and bounded D16 memory/concurrency.

## Verification and checkpoint

Run Python tests, Ruff, frontend lint and typecheck, Compose validation, secret scans,
remote accuracy/resource benchmarks, and repeated-cache checks. Use the D16 host in an
isolated worktree, reuse its existing runtime model cache, and do not alter the legacy
stack on port 3000.

Commit the tested implementation and a Phase 3 review report. Create the annotated
`checkpoint-phase-3` tag only after the unseen and active-profile datasets and all
quality, grounding, privacy, routing, and resource gates pass.

## Execution note

On 2026-07-13 all eleven reserved Sample Bills were processed at the user's explicit
request. They are therefore exposed regression documents, not a sealed unseen cohort.
`corpus/phase3-sealed-candidates.json` records that state. A replacement hospital-
disjoint cohort with confirmed identities and frozen gold is required before the
unseen quality gate or the Phase 3 checkpoint can pass.
