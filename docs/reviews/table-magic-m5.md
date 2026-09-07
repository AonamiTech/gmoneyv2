# Table Magic M5 Review

Date: 2026-09-07 UTC

Status: `in_progress` / authority and GPU accuracy hold

## Implemented repository evidence

- `f6966a5` adds fail-closed `working152` intake, deterministic nested cohort assignment,
  four-pass Luna review queues, review validation, and seal-readiness reporting. It cannot seal a
  partial or working inventory.
- `32e4ad6` adds backward-compatible V6 M5 contracts and
  `GMONEY_TABLE_SELECTION_MODE=off|shadow|enabled`, default `off`.
- `c879192` adds source-space proposal matching, stable logical-table identities, exact
  maximum-weight assignment, ambiguity abstention, derivative-only consensus, top-two
  reconstruction, deterministic whole-table ranking, dense UVDoc mapping, and V6 shadow output.
- Shadow winners, including UVDoc, are observations only. Existing validators still reject UVDoc
  artifacts as owners of canonical crops, tokens, adapter inputs, or evidence. The runtime rejects
  `enabled` because no M5 promotion is sealed.

## Local verification

- Full Python suite: `1205 passed` (`1205` tests collected).
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

## GPU preparation and hold

A max-reasoning Luna operator removed ten stopped candidate containers and aged journals without
removing volumes, images, tags, client jobs, source PDFs, model caches, or rollback archives. The
host recovered 3,818,528,768 bytes and ended with 15,239,282,688 bytes free. Live revision
`da20ceec63211559ea1707a16c831f600479c592` remained ready and release-consistent with zero active
jobs, restarts, or OOMs. All 14 production hashes remain present and complete (113 pages).

The post-cleanup evidence is
`/home/ubuntu/gmoneyv2-ops/m5-gpu-postcleanup-20260907T114916Z.json`, SHA-256
`d17a0776a3ab3ab1b2fdfec653e2aab00949c93120f17d17963a3da4fda729f2`.

M5 was not deployed or run over the client PDFs. This is intentional: the selected execution plan
requires the authoritative corpus and M0-M4 promotion first, and the host also remains below the
20 GiB campaign-start target. Starting an isolated M5 candidate now would produce integration
telemetry but could not satisfy the curved/flat accuracy gate.

## Remaining promotion gate

1. Reach exactly 159 eligible real PDFs and freeze nested 14/36/159 memberships.
2. Complete and freeze all four isolated Luna reviews per document, audited GoldDocumentV2,
   evaluator identity, and two identical passing36 baseline replays.
3. Promote M0-M4 against those exact identities.
4. Free at least 20 GiB safely, attest an isolated `c879192`-descended candidate, and run two
   deterministic replays across production14, passing36, and staging159.
5. Require no lost/duplicate tables, ambiguity abstention, no flat or critical regression, all
   release floors, and a preregistered curved-table improvement before M5 promotion.
