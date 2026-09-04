# Table Magic M3 Review

Status: promote
Date: 2026-09-04 UTC
Owner: Codex
Git SHA: `057b6cf870c2531f5a73baf436082a354585c5a0`
Image digests: not applicable — repository-only dormant substrate; no deployment
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
deployment remain blocked by that hold.

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
passed.

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

The loader rejects an archive above 128 MiB uncompressed before materializing its array. No GPU
or live candidate was used for M3 because the feature is a dormant geometry substrate.

## Regressions and unresolved risks

- No real UVDoc-produced grid has been evaluated; that belongs to M4.
- Corpus accuracy and flat/known-layout non-regression remain unknown because M2 is held.
- Very large valid grids have material analysis cost; M4 must benchmark the actual UVDoc grid
  resolution before selecting it in shadow mode.

## Gate decision and rollback condition

M3 is promoted at the repository/contract level. Keep dense generation inactive. If any existing
V5 or linear V6 payload changes, any malformed grid is accepted, a certified grid can change
without invalidation, or a synthetic mapping exceeds two pixels, revert `057b6cf` and retain the
matrix-only V6 path. Do not start M4 or deploy this revision until the M2 corpus hold is resolved.
