# Table Magic M4 Review

Status: hold
Date: 2026-09-04 UTC
Owner: Codex
Git SHAs: `1e3fced0a8e12a6a93e9f8a7c5682978f065c554`,
`6ae7e4d20531a177895b3be4046aedda0f462057` (GPU candidate target)
Image digests: unavailable — isolated GPU candidate did not start
Corpus manifest SHA-256: unavailable — inherited M2 external-data hold
Gold/evaluator versions: unavailable
Component/config versions: `gmoney_uvdoc_shadow_v1`, `gmoney_uvdoc_preregistration_v1`,
`gmoney_uvdoc_accuracy_v1`, `gmoney_uvdoc_gate_v1`; model repository
`PaddlePaddle/UVDoc_safetensors` at revision
`7b8c629d7a15656889d0b21c73df206ac8a732b5`

## Decision

Hold M4 at the accuracy boundary. The adapter, exact-grid capture, independent reproduction,
shadow-only V6 lineage, preregistration contract, and fail-closed gate runner are implemented.
M4 cannot move to `shadow` until the authoritative sealed 14/36/159 corpus, audited detailed
gold, evaluator identity, baseline, and a successful isolated GPU run of the pinned UVDoc model
snapshot are available.

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

## Local verification

- All 163 focused UVDoc and worker tests passed; 1,047 other backend tests passed with the three
  local `TestClient` files excluded.
- Ruff passed on all changed Python files.
- The repository-wide backend run was interrupted after the pre-existing basic FastAPI
  `TestClient` health request did not return; a focused traceback showed the request waiting in
  AnyIO/Starlette rather than M4 code. This must be rerun in the isolated candidate environment.
- All 23 frontend tests, frontend lint/type checking, the production build, and every
  CPU/demo/GPU/admin Compose configuration passed.

## GPU verification

Two Luna agents were assigned separately to isolated candidate deployment and read-only
monitoring for commit `6ae7e4d20531a177895b3be4046aedda0f462057`. At 2026-09-04 13:47:59 UTC,
SSH to `34.180.11.221:22` timed out; the public readiness endpoint on port `3100` also timed
out. An independent primary-agent SSH probe reproduced the port-22 timeout.

No remote mutation occurred. Candidate source transfer, model download/load, image build, and
GPU adapter execution never started, so this review does not claim GPU validation. Live
revision, readiness, GPU/VRAM, disk, restart/OOM, and port-isolation evidence could not be
collected during the outage.

## Gate decision and rollback condition

Keep `GMONEY_UVDOC_MODE=off`. The M4 gate requires every accepted image to reproduce from its
stored grid, at least one preregistered curved failure to improve under primary `UVDOC`, no new
critical error or lost previously-correct flat control, and unchanged authoritative output.
Failure of any condition keeps M4 held and prohibits `enabled` mode.
