# Table Magic M5 Review

Date: 2026-09-08 UTC

Status: `blocked` / exact GPU pilot failed the accuracy gate

## Implemented repository evidence

- `f6966a5` adds fail-closed `working152` intake, deterministic nested cohort assignment,
  four-pass Luna review queues, review validation, and seal-readiness reporting. It cannot seal a
  partial or working inventory.
- `32e4ad6` adds backward-compatible V6 M5 contracts and
  `GMONEY_TABLE_SELECTION_MODE=off|shadow|enabled`, default `off`.
- `c879192` adds source-space proposal matching, stable logical-table identities, exact
  maximum-weight assignment, ambiguity abstention, derivative-only consensus, top-two
  reconstruction, deterministic whole-table ranking, dense UVDoc mapping, and V6 shadow output.
- `6542841`, `6b39c66`, `4c2cad1`, and `8ccd1e3` bind M5 projections and all adapter,
  fragment, raster, and source-space lineage needed by the evaluator.
- `1a9926f` makes `authority_metrics_v2` match V6 source tables through artifact geometry;
  `cf40033` carries selected reconstruction geometry into the shadow authority projection.
- `40bb6ee` persists the bounded native UVDoc control grid and expands it in chunks for replay,
  replacing the invalid full-resolution grid that exceeded the 128 MiB dense-grid safety cap.
- `259b4cc` introduces `authority_metrics_v3`: a critical true positive must be critical in both
  gold and prediction, so neither critical denominator can be exceeded.
- Shadow winners, including UVDoc, are observations only. Existing validators still reject UVDoc
  artifacts as owners of canonical crops, tokens, adapter inputs, or evidence. The runtime rejects
  `enabled` because no M5 promotion is sealed.

## Local verification

- Full Python suite after `259b4cc`: `1230` tests collected and passed.
- Ruff: passed.
- Frontend lint, typecheck, and production build: passed.
- CPU/demo/GPU/admin Compose configuration: passed.
- Targeted M5 projection and dense-grid boundary tests also passed after the full-suite run.

## Authority state

The protected `working152` intake contains the 142 local documents plus ten GPU-only documents
approved after page review and near-duplicate comparison against all 142 local documents/967
rendered pages. Three known synthetic/certification fixtures were excluded. Repeating intake
produced the identical inventory digest. The audit summary is outside Git at
`/home/azureuser/gmoney-corpus-vault/working152-audit-20260907/metadata/working152-run-summary.json`
with SHA-256 `b052e3e66d07fd26825ce2aab477c2a7578e299bfde373c642026f1a7f2f6e65`;
the inventory SHA-256 is
`fc2cefc1bd82d0ed35ba1402a8cb046505a1709a7761bdb982c9dc7b163e2742`.

The fresh immutable readiness report is
`/home/azureuser/gmoney-corpus-vault/authoritative-v1/control/seal-readiness-working152-20260907-m5.json`,
SHA-256 `06954189fb9f17325c7427a33b472db5ef1ad69bb5df130e0da3cefbae6e48db`.
It correctly reports no authority claim: exactly seven genuine PDFs are still missing, cohort
membership is unassigned, the working inventory cannot be sealed, the render identity is
incomplete, and all 608 required four-pass review records are absent. Arbitrary or synthetic
padding is prohibited.

## Exact GPU shadow pilot

The exact revision `40bb6ee2fbb21a73af78ea6b4ca92a2a7c60e376` was built and deployed to
the isolated GPU canary with `GMONEY_UVDOC_MODE=shadow` and
`GMONEY_TABLE_SELECTION_MODE=shadow`. The worker image is
`sha256:7599977d08d2876fcf892d566e7a8163ea72b14519aa1a776e4326dd7d8833e4`.
Production stayed on `da20ceec63211559ea1707a16c831f600479c592`, healthy and unchanged.

The preregistered pilot used the only two four-pass audited gold documents: one curved bill and
one flat bill, each run twice and sequentially. This is a non-authoritative pilot, not the absent
14/36/159 corpus. All four jobs completed with zero V6 validation issues, restarts, or OOMs.
Every UVDoc run was `valid`, reproduced within `[1, 1, 1]` per RGB channel, and reported zero
foldovers and zero out-of-bounds samples. The formerly oversized curved grid is now a bounded
10,484-byte artifact.

Repeatability passed at the M5 boundary. The curved projection digest was
`141a2368e81470cfc204094cb57a3257dee41b1763ac199e826ed61bc23250f7` in both runs; the
flat digest was `21db5040b5a6ad60677a93c6261dd499859b28fc9407a1e10c34b86cbeb94fd4`
in both runs. Logical table IDs, decisions, ordering, and artifact IDs also repeated exactly. No
match was ambiguous. The selector retained `oriented_raw` for both curved tables but selected
`uvdoc` for both flat tables.

The frozen `authority_metrics_v2` comparison failed decisively:

| Metric | OFF/OFF baseline | M5 shadow projection |
| --- | ---: | ---: |
| Critical numeric precision | 100.00% | 48.72% |
| Critical numeric recall | 82.98% | 40.43% |
| Line-item row recall | 92.86% | 92.86% |
| Correct column assignment | 65.75% | 64.64% |
| Header/schema accuracy | 20.00% | 20.00% |
| Grand-total exact accuracy | 0.00% | 0.00% |
| Cell-value precision | 81.25% | 43.30% |
| Cell-value recall | 56.52% | 30.43% |

The v2 evaluator also exposed a fail-closed defect: its flat per-document critical precision is
reported as `14/12`, or 116.67%, because `critical_exact` uses the aligned gold role while
`critical_actual` uses the producer role. The pooled candidate result above is therefore
upper-biased, not evidence in the candidate's favour; even that result is far below the release
floor. The frozen report is retained unchanged, but evaluator correction and a newly sealed
baseline are required before another promotion claim.

The post-pilot v3 diagnostic does not replace the preregistered v2 result. It reports bounded
baseline critical counts `36/39/47` (92.31% precision, 76.60% recall) and candidate counts
`16/39/47` (41.03% precision, 34.04% recall). Both fail the 95% recall target, and the candidate
still regresses substantially. A new authority-bound baseline must seal v3 before these corrected
figures can be used for promotion.

Visual inspection agrees with the metrics. The curved rectification is readable and visibly
flatter, but the already-flat bill is unnecessarily warped and clipped at the page edges. The
current selector therefore cannot distinguish a useful curved transform from a harmful flat one.

The complete candidate workspaces and monitoring evidence are retained outside Git under
`/home/azureuser/gmoney-corpus-vault/working152-audit-20260908/pilot-shadow-40bb6ee-20260908`;
its `SHA256SUMS` file has SHA-256
`1ee27be1eab0de1fd4e11f97d8094d658d916d240674906015c5735e6506a4c6`.
The frozen candidate evaluator bundle is
`/home/azureuser/gmoney-corpus-vault/working152-audit-20260908/gpu-pilot-evaluator-v2-40bb6ee`;
`candidate-evaluation.json` has SHA-256
`207e13a86d4f98a0a1f1d707273136a4501199f7a624f74d8c8940e0d44dc6cb` and its
`SHA256SUMS` file has SHA-256
`7c5fa306de758e98dc685e43002acd0ddfe14a93d49fe168b1cdab934320ff63`.
The deployment archive's `SHA256SUMS` file has SHA-256
`6eb732e54423596158dad1a618d7e45e10bb55da7c2320afa1bf377211d26d61`.
The failed gate stopped the campaign before any other working152 bill was submitted. M4's GPU
integration/replay blocker is resolved, but its curved-accuracy promotion gate is not; this M5
candidate is rejected and production promotion remains prohibited.

The user-authorized cleanup removed the stopped Surya POC containers, its replaceable 2.8 GiB
model volume, and unique 1.92 GB API image while preserving its 36 KiB SQLite data volume and
source tree. Its llama image layers were retained because production actively uses the same image.
The unused `m5-cf40033` API and frontend images were also removed after container-reference
checks. The two cleanup steps recovered 5,567,782,912 bytes in total; current canary and
production images and rollback state were retained.

## Remaining promotion gate

1. Reach exactly 159 eligible real PDFs and freeze nested 14/36/159 memberships.
2. Complete and freeze all four isolated Luna reviews per document, audited GoldDocumentV2,
   evaluator identity, and two identical passing36 baseline replays.
3. Seal `authority_metrics_v3` and a fresh baseline, then promote M0-M4 against those exact
   identities.
4. Change the selection policy so flat pages abstain or retain baseline, attest a new exact
   candidate, and run two
   deterministic replays across production14, passing36, and staging159.
5. Require no lost/duplicate tables, ambiguity abstention, no flat or critical regression, all
   release floors, and a preregistered curved-table improvement before M5 promotion.
