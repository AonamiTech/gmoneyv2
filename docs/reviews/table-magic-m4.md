# Table Magic M4 Review

Status: hold
Date: 2026-09-08 UTC
Owner: Codex
Git SHAs: `1e3fced0a8e12a6a93e9f8a7c5682978f065c554`,
`6ae7e4d20531a177895b3be4046aedda0f462057`,
`8dfdd4ae3b11aa03c41e0a3e4a41e2b60853f6b5`,
`0c8650a40e24e1881b10ad839d8645d769c4a1b5`,
`64bdf9cb0461c61bb8884ddf9d2ad60e6ac6d65d`,
`40bb6ee2fbb21a73af78ea6b4ca92a2a7c60e376` (current GPU shadow target)
Candidate image digest: `sha256:7599977d08d2876fcf892d566e7a8163ea72b14519aa1a776e4326dd7d8833e4`
Corpus manifest SHA-256: unavailable — inherited M2 external-data hold
Gold/evaluator versions: four-pass audited non-authoritative two-document pilot only;
`authority_metrics_v2` is frozen with the pilot and `authority_metrics_v3` requires a new baseline
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

- All 1,187 backend tests passed, including the v2 authority-gate adversarial cases and the
  previously excluded `TestClient` coverage.
- Ruff and diff validation passed for the complete repository.
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

### 2026-09-07 isolated shadow integration run

A Luna deployment agent and an independent read-only Luna monitoring agent, both configured at
maximum reasoning effort, exercised commit `64bdf9cb0461c61bb8884ddf9d2ad60e6ac6d65d` on the T4.
The candidate used loopback-only ports, a fresh data root, `GMONEY_UVDOC_MODE=shadow`, the pinned
model cache read-only, and the exact candidate source mounted read-only over the previously
certified GPU dependency image. No extraction, inference, worker, or GPU-image source changed
between the reference runtime commit and this target; the new authority evaluator was tested by
the local backend suite.

- Two runs of a de-identified synthetic ledger completed with V6 validation passing and ten rows.
  Each stored an unselected `UVDOC` artifact, an unselected `UVDOC_ENHANCED` artifact, and the
  exact `3509 x 2480 x 2` float32 grid. Independent reproduction error was `(1, 1, 1)`, with zero
  foldovers and zero out-of-bounds samples.
- The selected page artifact remained `ORIENTED_RAW`; the canonical table crop and final OCR
  adapter did not consume either UVDoc artifact. The two canonical projections matched at SHA-256
  `87176923cfc8473ffaf5b05c1c862f29ac4b6c87dbe850e67616c81b43b9f92b` after excluding row
  creation timestamps and adapter latency.
- The first image build stopped before the 10-GiB disk floor, a stale reused-worker attempt was
  rejected because its heartbeat reported `off`, and an initial candidate-config ownership error
  and unsuitable blank PDF were corrected. These attempts are retained as failures, not counted
  as passing tests.
- Before teardown the candidate had zero restarts/OOMs and the T4 returned to 4,175 MiB used,
  10,747 MiB free, 0% utilization. The isolated containers were stopped without deleting volumes.
  Production remained ready and release-consistent on `da20ceec63211559ea1707a16c831f600479c592`
  with zero active jobs and no observed restarts, OOMs, or error markers.

Evidence is retained at
`/home/ubuntu/gmoneyv2-releases/m4-shadow-64bdf9-20260907T093359Z/evidence/` on the GPU host.
`shadow-smoke-summary.json` has SHA-256
`2ced7062f54564d6d2770bbd3b24e4c515cec66b4ee9f696e32989d609122b52`; the 35-file evidence
manifest has SHA-256 `24f790eab05ab06af196cfecfb53b9d9f733fc48c9d26e81ef93ea003683a013`.
This is a shadow integration/substrate result only. The synthetic flat control does not establish
curved improvement or authoritative flat non-regression.

### 2026-09-08 audited curved/flat GPU pilot

Exact revision `40bb6ee2fbb21a73af78ea6b4ca92a2a7c60e376` replaced full-resolution
grid persistence with the model's bounded native control grid and deterministic chunked
`align_corners` expansion. The isolated T4 canary then ran the only two four-pass audited pilot
bills twice each with `GMONEY_UVDOC_MODE=shadow`.

- All four UVDoc runs were valid, reproduced within `(1, 1, 1)`, and had zero foldovers and zero
  out-of-bounds samples. V6 validation passed without issues, and artifact identities repeated
  exactly. The curved control grid was 10,484 bytes rather than the rejected roughly 298 MiB
  full-resolution representation.
- Visual review found the curved bill readable and visibly flatter. The already-flat bill was
  unnecessarily warped and clipped at the page edges.
- Downstream selection retained `ORIENTED_RAW` for both curved tables but chose `UVDOC` for both
  flat tables. Frozen pooled critical precision/recall fell from `100.00%/82.98%` to
  `48.72%/40.43%`; the pilot therefore proves runtime feasibility but fails the curved-gain and
  flat-non-regression gate.
- All jobs were sequential with zero worker restarts or OOMs. Production remained healthy and
  unchanged on `da20ceec63211559ea1707a16c831f600479c592`. The failed gate stopped the wider
  working152 campaign.

Evidence is retained under
`/home/azureuser/gmoney-corpus-vault/working152-audit-20260908/pilot-shadow-40bb6ee-20260908`;
the archive `SHA256SUMS` file has SHA-256
`1ee27be1eab0de1fd4e11f97d8094d658d916d240674906015c5735e6506a4c6`.
This clears the bounded-grid GPU integration blocker, not the M4 accuracy or authority blockers.

## Gate decision and rollback condition

Keep `GMONEY_UVDOC_MODE=off` in the live production stack. Set it to `shadow` in an isolated
candidate stack when testing so the actual UVDoc path, artifacts, and telemetry are exercised
without changing canonical publication. The M4 gate requires every accepted image to reproduce
from its stored grid, at least one preregistered curved failure to improve under primary `UVDOC`,
no new critical error or lost previously-correct flat control, and unchanged authoritative
output. Failure of any condition keeps M4 held and prohibits `enabled` mode.
