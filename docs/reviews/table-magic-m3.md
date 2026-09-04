# Table Magic M3 Review

Status: promote
Date: 2026-09-04 UTC
Owner: Codex
Git SHAs: `057b6cf870c2531f5a73baf436082a354585c5a0`,
`5da09b6228b40adb8a5e0b0a774bf39608ddb9da`,
`76de77dfe10b5b9d71b8b5978264432ea1a73915` (GPU candidate source)
Image digests: API `sha256:9ffb6f77f42e429ac1729e62e2988d2b173d87a33889ba0ef3c3c2301c5f0a42`;
frontend `sha256:8c2fa9b1e0718c30ee5251f443d05c1d5f3a0d22bcda8e000e78ce8f013db9dd`;
GPU worker `sha256:363acc89684e9f7dcdb343898eb86bfc5fbe83e90442e5668cb4ceb3949fcfb0`
Corpus manifest SHA-256: unavailable — inherited M2 external-data hold
Gold/evaluator versions: not applicable to the synthetic geometry gate
Component/config versions: `artifact_manifest_v1`, `offline_accuracy_spine_v6`,
`extraction_validation_v6_r2`, `job_certification_v3`

## Decision

Promote the M3 dense-transform foundation. Deterministic storage, mixed-chain geometry,
adaptive polygon projection, diagnostics, V6 publication, certification inventory, evidence
export, and negative fixtures pass their synthetic gate. No extraction route emits a dense grid,
so this promotion does not activate UVDoc or change live extraction selection.

M2 remains `in_progress` because its authoritative frozen corpus is unavailable. M4 and any
production deployment remain blocked by that hold.

## Change evaluated

- Added deterministic single-member NPZ storage and strict resolver-based loading.
- Enabled dense backward grids in the V6 artifact graph with dimensions and metadata bound into
  canonical mapping/artifact hashes.
- Added bilinear point mapping, mixed identity/homography/dense traversal, one-pixel adaptive
  polygon densification, and typed displacement/scale/anisotropy/Jacobian diagnostics.
- Bound grid files into `job_certification_v3`, restart checks, and evidence bundles.
- Added deterministic mesh/polygon overlay rendering and fail-closed negative fixtures.

## Commands and environment

```text
.venv/bin/ruff check .
.venv/bin/pytest
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend run build
make compose-config
git diff --check
```

Results: 1,130 backend tests and 23 frontend tests passed. Ruff, frontend lint/type checking,
the production frontend build, diff checking, and all CPU/demo/GPU/admin Compose configurations
passed locally. The same 1,130 backend tests, 23 frontend tests, lint, type checking, production
build, Ruff, and Compose checks also passed from exact source on the GPU host. A disposable runner
based on the candidate API image passed 33 dense-artifact and V6 contract tests against the
installed candidate package.

## Isolated GPU-host certification

Revision `76de77dfe10b5b9d71b8b5978264432ea1a73915` was built and tested on the Tesla T4 host in the
isolated `gmoney-v2-canary` project on ports 3110/18100. The candidate reused the live,
digest-pinned PaddleOCR-VL service and did not start a second VLM container. Release attestation,
profile access, and profile validation passed for the exact image labels and manifests.

The de-identified synthetic ledger completed in 25 seconds with V6 revision 6, certification
`passed`, validation `passed` with zero issues, and 10 expected rows. The same result,
certification, and validation remained valid after restarting the candidate application services.
All candidate containers reported zero OOM events and zero restart-policy restarts; the worker
reported `gpu:0`. The live stack remained ready and idle on revision
`da20ceec63211559ea1707a16c831f600479c592` throughout. The candidate was stopped without a live
cutover, and ports 3110/18100 were closed afterward.

The first worker-image export was canceled before candidate startup when free disk briefly fell
to about 6.5 GiB, below the 10 GiB gate. To restore headroom, nine explicit image tags from the
completed M1 candidate and two superseded, failed M2 candidates were removed; their release source
and evidence remain retained, as do the live, rollback, and final M2 candidate images. The worker
then built from its completed cache with 21 GiB free. During the successful running-candidate
phase, free disk remained at least 11,768,188,928 bytes, and GPU memory peaked at 4,595 MiB.

Evidence is retained at
`/home/ubuntu/gmoneyv2-releases/76de77d-m3-gpu-cert-20260904`. Key hashes include:

| Evidence | SHA-256 |
| --- | --- |
| Source attestation | `eece9f0548c4214acda871e564758290846a4565897b22b51e02ee1ba682716e` |
| Release attestation | `d07d6b0b2e4e3351016a3f54fb6364105f89ab1930810bdf39a583be83eaea21` |
| Backend test log | `ea45185dcbc23ce5dfbb5e9e9e8671a41956dee9acdb6b508336e773d49d5f78` |
| Candidate-image dense tests | `31428bd0b592e0201779dae096303d70547edd39fe87edcb4d2023210c1a24b4` |
| Post-restart status | `4d3e47f12a09e9242cfeeca9548ddc007b1a0189b86811e91dc25f22949664a6` |
| Post-restart validation | `612e91ceef6c7d9b639c263243fb640b31d1287bfedafebd13a962dea2119fe6` |

## Geometry and integrity results

- A smooth five-pixel sinusoidal warp had `0.119395678 px` maximum analytic point error.
- Identity, affine-equivalent, sinusoidal, and mixed dense/homography chains remained within the
  two-pixel contract.
- Adaptive polygons preserve ordering, add vertices only when the one-pixel chord criterion
  requires them, and fail at depth 12 or 4,096 points rather than falling back to corner-only
  projection.
- A dense V6 workspace passed publication with `extraction_validation_v6_r2`, produced
  `job_certification_v3`, survived a store reread, and exported its grid in the evidence bundle.
- Changing the grid invalidated certification and evidence export.

Deterministic fixture hashes:

| Fixture | SHA-256 |
| --- | --- |
| 11×11 identity NPZ | `ab01d376704e20cfec5284d6f61ae0137b5dd57e4d31ffa495e4f54fe736e0a7` |
| 11×11 sinusoidal NPZ | `e1c15f92f25b6da8436a16544916b87aa0cfcb3e66b680a9acd5796e93dbf113` |
| Identity diagnostic overlay PNG | `bc7d6920c1a8b6b90d53ee066458161350637ef08fa8dcff198648962f9bc906` |

Stable fail-closed coverage includes:

- `v6_contract_invalid` for malformed mapping metadata.
- `v6_dense_grid_path_invalid`, `v6_dense_grid_missing`,
  `v6_dense_grid_digest_mismatch`, and `v6_dense_grid_archive_invalid` for storage failures.
- `v6_dense_grid_metadata_mismatch` and `v6_dense_grid_non_finite` for invalid arrays.
- `v6_dense_mapping_out_of_bounds` and `v6_dense_grid_jacobian_invalid` for invalid warps.
- `v6_dense_polygon_orientation_invalid`, `v6_dense_polygon_self_intersection`, and
  `v6_dense_polygon_budget_exhausted` for unsafe materialized geometry.

## Resources

Single-process local measurements include Python/OpenCV baseline memory and are indicative, not
deployment capacity claims:

| Grid | Archive | Write | Load | Analyze | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64×64×2 | 9,519 B | 23.49 ms | 1.02 ms | 26.06 ms | 109.17 MiB |
| 256×256×2 | 137,622 B | 79.82 ms | 5.54 ms | 90.13 ms | 128.75 MiB |
| 1024×1024×2 | 2,980,886 B | 758.61 ms | 40.53 ms | 765.58 ms | 454.69 MiB |

The loader rejects an archive above 128 MiB uncompressed before materializing its array. GPU-host
certification validates the release images, container runtime, isolation, and existing linear V6
inference path. Dense transforms themselves remain CPU geometry code, and no extraction route
emits a dense grid in M3.

## Regressions and unresolved risks

- No real UVDoc-produced grid has been evaluated; that belongs to M4.
- Corpus accuracy and flat/known-layout non-regression remain unknown because M2 is held.
- Very large valid grids have material analysis cost; M4 must benchmark the actual UVDoc grid
  resolution before selecting it in shadow mode.

## Gate decision and rollback condition

M3 is promoted at the repository/contract level. Keep dense generation inactive. If any existing
V5 or linear V6 payload changes, any malformed grid is accepted, a certified grid can change
without invalidation, or a synthetic mapping exceeds two pixels, revert `5da09b6` and `057b6cf`
and retain the matrix-only V6 path. Do not start M4 or perform a production cutover until the M2
corpus hold is resolved.
