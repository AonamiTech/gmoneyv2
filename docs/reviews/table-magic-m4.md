# Table Magic M4 Review

Status: hold
Date: 2026-09-07 UTC
Owner: Codex
Git SHAs: `1e3fced0a8e12a6a93e9f8a7c5682978f065c554`,
`6ae7e4d20531a177895b3be4046aedda0f462057`,
`8dfdd4ae3b11aa03c41e0a3e4a41e2b60853f6b5`,
`0c8650a40e24e1881b10ad839d8645d769c4a1b5` (GPU candidate target)
Image digest: `sha256:d7f1b7c780746ef4c29a975da238591b1ca4caa1892aa441b4673d67c03847c9`
Corpus manifest SHA-256: unavailable — inherited M2 external-data hold
Gold/evaluator versions: unavailable
Component/config versions: `gmoney_uvdoc_shadow_v1`, `gmoney_uvdoc_hf_to_paddle_v1`,
`gmoney_uvdoc_preregistration_v1` (diagnostic only), `gmoney_uvdoc_accuracy_v1` (diagnostic
only), `gmoney_uvdoc_gate_v1` (non-promoting), `gmoney_uvdoc_preregistration_v2`,
`gmoney_uvdoc_accuracy_v2`;
model repository
`PaddlePaddle/UVDoc_safetensors` at revision
`7b8c629d7a15656889d0b21c73df206ac8a732b5`

## Decision

Hold M4 at the accuracy boundary. The adapter, exact-grid capture, independent reproduction,
shadow-only V6 lineage, preregistration contract, and fail-closed gate runner are implemented.
The isolated GPU substrate gate passes. An isolated candidate may and should run in `shadow` mode
for substrate and integration testing, but that does not promote M4. M4 cannot pass its accuracy
boundary until the authoritative sealed 14/36/159 corpus, audited detailed gold, evaluator
identity, and baseline are available to prove curved gain and flat non-regression.

## Change evaluated

- Added a version-pinned, GMoney-owned Paddle dynamic forward path that returns the rectified RGB
  image and exact full-resolution backward grid used by `grid_sample`.
- Added an independent NumPy zero-padded bilinear replay check with a maximum error of one value
  per channel, plus M3 grid storage and diagnostics.
- Added preregistered curved/flat eligibility and additive `uvdoc_shadow_runs` records.
- Added unselected `UVDOC` dense artifacts and identity-mapped `UVDOC_ENHANCED` derivatives.
- Prohibited shadow artifacts from production table, adapter, token, and evidence lineage.
- Added a digest-bound gate runner that returns `hold` when authoritative accuracy evidence is
  missing and considers primary `UVDOC`, not the enhanced exploratory branch, for promotion.
- Made the v1 gate explicitly diagnostic-only and non-promoting. The v2 gate recomputes the full
  report from typed gold, baseline/candidate outputs, evaluator/baseline manifests, frozen cohort
  evidence, and shadow downstream proofs before it can emit an M4 report.
- Added a complete, shape-checked safetensors-to-Paddle checkpoint mapping after GPU testing found
  that PaddleX's generic loader silently left the pinned checkpoint unused.

## Local verification

- All 168 focused UVDoc and worker tests passed; 1,047 other backend tests passed with the three
  local `TestClient` files excluded.
- Ruff passed on all changed Python files.
- The repository-wide backend run was interrupted after the pre-existing basic FastAPI
  `TestClient` health request did not return; a focused traceback showed the request waiting in
  AnyIO/Starlette rather than M4 code. This must be rerun in the isolated candidate environment.
- All 23 frontend tests, frontend lint/type checking, the production build, and every
  CPU/demo/GPU/admin Compose configuration passed.

## GPU verification

Luna deployment and monitoring agents, followed by two Luna agents explicitly configured at
maximum reasoning effort, tested commit `0c8650a40e24e1881b10ad839d8645d769c4a1b5` in disposable
GPU containers with networking disabled and candidate source/model mounts read-only. Evidence is
retained at `/home/ubuntu/gmoneyv2-releases/m4-0c8650a-20260904T1458Z/evidence/`.

- Paddle 3.2.2, PaddleOCR 3.7.0, PaddleX 3.7.2, CUDA, and the exact pinned model loaded on the
  Tesla T4. All 251 Paddle tensors matched checkpoint values exactly; 44 foreign-framework batch
  counters were explicitly skipped; missing, unexpected, collision, shape, and value mismatch
  counts were zero.
- A synthetic de-identified RGB input produced a valid `240 x 320 x 2` backward grid. Independent
  reproduction error was `(1, 1, 1)`, foldover count was zero, and out-of-bounds rate was zero.
  The output, enhanced image, and grid SHA-256 values are recorded in
  `evidence/uvdoc-adapter-result.json`.
- The worker container exited zero and was not OOM-killed. The production stack stayed ready and
  release-consistent on `da20ceec63211559ea1707a16c831f600479c592`; restart counts stayed zero,
  and no live configuration, container, port, or release changed.
- The candidate archive SHA-256 is
  `2dde12d6fa1ebbc4e35abce66afd52677569f8efc3bced72ad7eadb2cef7303e`. A fresh extraction's
  196-file content manifest exactly matched the staged candidate manifest at
  `46b46cc6aa8a3b7ed8959227645a421af84bcda63990f87ea3a8719a5aba7e34`.

The runtime worker image intentionally lacks pytest, so repository tests were not claimed from
that image; the real GPU/model probe above exercised the production dependency set, while the
current 168-test worker/UVDoc suite passed locally.

## Gate decision and rollback condition

Keep `GMONEY_UVDOC_MODE=off` in the live production stack. Set it to `shadow` in an isolated
candidate stack when testing so the actual UVDoc path, artifacts, and telemetry are exercised
without changing canonical publication. The M4 gate requires every accepted image to reproduce
from its stored grid, at least one preregistered curved failure to improve under primary `UVDOC`,
no new critical error or lost previously-correct flat control, and unchanged authoritative
output. Failure of any condition keeps M4 held and prohibits `enabled` mode.
