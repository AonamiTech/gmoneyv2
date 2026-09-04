# Table Magic M2 Review

Status: hold
Date: 2026-09-04 UTC
Owner: Codex
Git SHA: `ec17b0756d9f1143c0ba91b0579181f9a385e401`
Candidate release: `ec17b07-m2-certfinal-20260904` (stopped; no cutover)
Evidence root: `/home/ubuntu/gmoneyv2-releases/ec17b07-m2-certfinal-20260904`
Corpus manifest SHA-256: unavailable — the authoritative frozen 14/36/159 inputs are not present
Gold/evaluator versions: unavailable — the matching sealed gold and baseline are not present
Component/model/config versions: bound by the release attestation and candidate image labels

## Decision

Hold M2 at `in_progress`. The implementation, local verification, isolated GPU candidate,
deterministic replay, V5 migration/rollback, and tamper gates passed. The required frozen
flat/known-layout accuracy gate could not be run because the exact sealed corpus, audited gold,
and baseline are absent locally and on the GPU host. The synthetic canary is an integrity and
operability check; it is not a substitute accuracy corpus.

M3 is planned in `tableMagic.md` but remains `not_started`. Its implementation must not begin
until this M2 hold is resolved and M2 is promoted.

## Change evaluated

- `077d8e6`: closed V6 page-role and adapter-lineage gaps and recorded the M3 plan.
- `b947332`: bound projected V6 tokens to the exact source artifact selected from the manifest.
- `ec17b07`: preserved stable machine-row timestamps during cached reprocessing.
- V5 reading, V5-to-V6 staged projection, V6 publication validation, certification, and
  rollback were exercised without modifying the live runtime.

## Commands and environment

Local verification at `ec17b07`:

```text
git diff --check
.venv/bin/ruff check .
.venv/bin/pytest
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend run build
make compose-config
```

Results: 1,113 backend tests passed; 23 frontend tests passed; lint, type checking, production
build, Ruff, diff checking, and all CPU/demo/GPU/admin Compose configurations passed.

The candidate used one T4 GPU lane, candidate ports 3110/18100, the live model cache, and the
existing live PaddleOCR-VL service. It did not start a second VLM. Candidate API, frontend, and
worker labels, manifests, readiness, and `/build.json` all matched `ec17b07`; profile access and
validation passed with no recovery, migration, or pending journals.

Image IDs:

| Component | Image ID |
| --- | --- |
| API | `sha256:700d3d91fac4c2650c6a854b69a7826c98dbb403d17cc65ac8b6cbaa83256e63` |
| Frontend | `sha256:76a827f310ee2e560e041856f964c39fe3684d706ea84354b2227d3d458048f6` |
| GPU worker | `sha256:c6e877ee4fb82f48cf1980acb46f45bf17b5840c9c27ab98788e0d14321d1b01` |
| Nginx | `sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10` |

## Accuracy results

Frozen overall, critical-field, curved-photo, flat/known-layout, and unreadable/reupload metrics:
not run. The authoritative sealed inputs and audited gold required to calculate them were not
available. No accuracy delta or generalization claim is made.

The de-identified one-page synthetic ledger completed with V6 revision 6, ten expected rows,
the exact expected descriptions, quantities, unit prices, and document total `458.25`. It passed
validation with zero issues and produced `job_certification_v3`.

## Determinism and compatibility

- Two cached replays had identical artifact/cache files and byte-identical raw `result.json`.
- Certified result, validation, and certification payloads were identical across both replays.
- Raw result SHA-256: `04213587e66e30c696ac5458df58279f6c045ac3a3e23a58098a3150cd710051`.
- Derived validation SHA-256: `220871b3ff3bda68379c18380f105ef1d05bd85011b558ca353cce8e718c89e9`.
- Derived certification SHA-256: `be05d0a100fa301f666e09569b6092a4882029e94eb985e0863f7615c01e4350`.
- Review and stage-batch audit timestamps differ by design and are outside the stable machine
  payload comparison.

An isolated V5 workspace staged 10→10 rows into V6, applied with passed validation and
`job_certification_v3`, and rolled back to V5. The before/after source, result, and artifact
inventories have the same SHA-256
`f7b7ad08c0d566ac3030cb48c6d29105e9d5180785bf4b615549c558b4e05e99`; the byte-diff file is
empty (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

## Evidence integrity

Four mutations were made only in isolated copies of the synthetic V6 workspace:

| Mutation | Result |
| --- | --- |
| Artifact image bytes | Fatal `v6_artifact_hash_mismatch` |
| Homography matrix | Fatal `v6_contract_invalid` |
| Manifest digest | Fatal `v6_contract_invalid` |
| Adapter logical-table lineage | Fatal `v6_contract_invalid` |

The tamper matrix passed and records `published_jobs_modified: false`; its SHA-256 is
`62f209499039741eee6ebadf1c158699b29e38e9597b60af7d00968a900bb1d9`.
Linear mapping and lineage assertions, including the two-pixel contract, passed in the full
local test suite. Dense mappings continue to fail closed as reserved for M3.

## Resources and deployment safety

- Fresh canary peak: 10,729 MiB VRAM and 73% GPU utilization; idle returned to 4,175 MiB.
- Cached migration staging peak: 4,595 MiB VRAM.
- Peak process RSS was not independently sampled; retained Docker snapshots show no OOM or
  restart, but are not reported as a peak measurement.
- Initialization and latency percentiles are not meaningful for this one-document integrity
  canary and were not used as a promotion signal.
- Free disk remained about 17 GiB, above the 10 GiB floor.
- Live and candidate stayed ready and idle throughout testing. Every observed container had
  restart count zero and `OOMKilled=false`; no fatal log marker was observed.
- The candidate was stopped after certification. No volume, image, job, or release evidence was
  deleted. Live remained on `da20ceec63211559ea1707a16c831f600479c592`, ready and
  release-consistent with 47 jobs and zero active work.

## Regressions and unresolved risks

- The exact frozen 14/36/159 source manifests, audited gold, evaluator identity, and baseline
  report must be recovered or rebuilt under a new sealed protocol.
- Without that material, critical precision/recall non-regression and flat/known-layout
  non-regression remain unknown.
- A single synthetic document proves contract behavior and deployment isolation, not corpus
  accuracy, hospital generalization, or production readiness.

## Gate decision and rollback condition

Decision: **hold**. Keep M2 `in_progress`, keep M3 `not_started`, and do not cut over the
candidate. Resume the M2 promotion review only after running the exact sealed corpus against its
audited gold and unchanged evaluator and confirming every Table Magic non-regression floor.

If a later candidate changes any stable replay payload, accepts any tamper case, exceeds the
resource floor, loses revision consistency, or regresses a frozen metric, stop that candidate
and retain the current live release. The verified pre-candidate runtime backup is retained at
`/home/ubuntu/gmoneyv2-releases/pre-077d8e6-m2-cert-20260904T0626Z`.
