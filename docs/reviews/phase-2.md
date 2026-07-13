# Phase 2 Checkpoint Review

**Decision:** FAIL — hard accuracy gate not met  
**Review date:** 2026-07-13 UTC  
**Frozen extraction SHA:** `8c04f2f49bd97f199473fa94166d6f229b3ecbe0`  
**Checkpoint tag:** not created  
**Deployment:** stopped; V2 was not exposed on port 3100

## Executive finding

The minimum accuracy spine is not accurate enough to deploy. On the sealed 409-row,
unseen-layout holdout it emitted 93 rows and matched only 44 under the legacy evaluator and 41
under the canonical evaluator.

| Evaluator | Gold | Emitted | Matched | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `legacy_metric_v1` | 409 | 93 | 44 | 47.31% | 10.76% | 17.53% |
| `canonical_metric_v2` | 409 | 93 | 41 | 44.09% | 10.02% | 16.33% |

The required gate was at least 85% precision and 85% recall under both evaluators. The gate
failed by a wide margin. No threshold, evaluator, gold row, or acceptance criterion was changed.

## Protocol and pre-holdout freeze

The extraction implementation was frozen before holdout scoring. Before the freeze, all three
development bills passed both evaluators:

| Split | Bill | Gold/emitted/matched | Precision | Recall |
|---|---|---:|---:|---:|
| Train | `7030777110` | 9/9/9 | 100% | 100% |
| Validation | `9739459460` | 7/7/7 | 100% | 100% |
| Validation | `9986117545` | 33/33/33 | 100% | 100% |

The four holdout PDFs were extracted on the D16 host without transferring their gold files to
that host. Machine outputs were closed for all four bills before the gold rows were inspected or
the parser was changed. The frozen evaluators were then run locally against untouched machine
output.

The holdout is no longer sealed after this review. These four bills must now become named
regressions; a future Phase 2 gate requires a new hospital-grouped unseen holdout.

## Holdout results

| Bill | Gold | Emitted | Legacy P/R | Canonical P/R |
|---|---:|---:|---:|---:|
| `1782547800688` | 97 | 9 | 0.00% / 0.00% | 0.00% / 0.00% |
| `1782552821660` | 199 | 29 | 100.00% / 14.57% | 100.00% / 14.57% |
| `9021114986` | 66 | 48 | 31.25% / 22.73% | 25.00% / 18.18% |
| `9890913867` | 47 | 7 | 0.00% / 0.00% | 0.00% / 0.00% |

Before matching, 93 emitted rows limited possible aggregate recall to 22.74%. Under the legacy
metric there were 365 false negatives and 49 false positives. Under the canonical metric there
were 368 false negatives and 52 false positives.

## Stage diagnosis

### Table localization was not the main failure

Every one of the 23 gold-row-bearing pages received at least one table crop. Across the four
documents, layout or OCR geometry produced 31 crops and the VLM returned a cached response for
every crop. The failure happened after table localization.

### VLM serialization was inconsistent

- Only 13 of 31 responses contained any parseable OTSL rows.
- Eighteen responses returned prose or flattened line text with no OTSL structure.
- Only seven crops produced any candidate rows, and only six produced canonical detail rows.
- The current engine has no OCR-native row reconstruction fallback. OCR tokens are used mainly
  to ground an already-parsed VLM row, so a non-OTSL VLM response becomes zero rows even when the
  OCR tokens and crop are readable.

This explains the page pattern directly. Bill `1782552821660` matched 29 of 30 page-1 rows, then
emitted nothing for pages 2–7. Bill `1782547800688` emitted only page-1 summary rows while its
97 gold rows were on pages 2–5. Bill `9890913867` followed the same summary-only failure mode.

### Dense tables exceeded the per-slot context

The llama.cpp service divided a 24,576-token context across three slots, leaving 8,192 tokens
per request. Three dense crops reached exactly 8,192 total tokens and were truncated:

- `1782552821660`, page 4
- `9890913867`, page 4
- `9890913867`, page 5

Their tails show incomplete or repetitive generation, not a clean table terminator. Three slots
fit in memory, but they are not accuracy-safe for dense hospital tables. This defect did not
cause the gate decision by itself: the two already-completed, non-truncated bills limited maximum
aggregate recall to 74.08%, and the final aggregate limit was 22.74%.

### Header and typed-amount inference failed

- `1782547800688` page 2 produced 22 candidates from valid OTSL. The inferred `Amount` column was
  the zero-discount column, while the unlabeled rightmost column contained the line total. All 22
  rows were reclassified as zero-value metadata and discarded.
- Pathology tables headed `Test Name` and `Rate` were readable and structured, but `Test Name`
  was not recognized as a description header. Headerless continuation pages also had no
  document-level schema state to inherit.
- On the pharmacy pages that did emit rows, VLM cell shifts assigned several amounts to the
  preceding description. Spatial grounding failed to find that amount on the same OCR row, but
  the ungrounded VLM amount was still retained instead of being rejected or repaired from OCR.
  For `9021114986`, pages 4–6 emitted 48 rows but only 15 legacy/12 canonical matches.

### Page and row-role classification failed

The nine rows emitted for `1782547800688` and seven rows emitted for `9890913867` came from
page-1 summary/category tables that the gold policy intentionally excludes. They were classified
as detail rows. The engine needs an explicit page/table role model and document-level hierarchy;
row text rules alone cannot reliably distinguish charge summaries from item ledgers.

### OCR errors were present but secondary

Examples include `CEFTUM 500` versus `CEFTRIA 500`, `VENFLON` versus `SURGIVENT`, and `VELFIX`
versus `VEEFIX BD`. These errors reduced matches on otherwise extracted pharmacy rows, but they
do not explain the much larger zero-row pages. The primary failures were provider structure,
schema/header inference, amount alignment, and role filtering.

## Frozen error taxonomy

| Category | Severity | Evidence |
|---|---|---|
| OCR recognition | Medium | Product-name substitutions on emitted pharmacy rows |
| Table localization | Low | Crops existed for 23/23 gold-bearing pages |
| Provider/table serialization | Critical | 18/31 crops had no parseable OTSL; three context truncations |
| Row grouping/schema continuity | Critical | Headerless and `Test Name` continuation tables produced zero candidates |
| Typed field mapping | Critical | Rightmost net amount lost; zero-discount column selected as amount |
| Evidence grounding/fusion | Critical | Ungrounded shifted VLM amounts survived instead of being repaired/rejected |
| Role/page classification | Critical | Summary tables emitted as details; valid zero-discount rows discarded |
| Missing/extra rows | Critical | 365/49 legacy and 368/52 canonical FN/FP |

## D16 runtime review

- Host: Azure `Standard_D16ds_v6`, 16 vCPU, approximately 64 GiB RAM, no swap.
- Three documents remained active without OOM, corruption, or worker failure.
- Observed available memory remained approximately 51–52 GiB during the run.
- Thirty-one VLM crops had a 64.2 s minimum, 185.4 s median, 678.5 s p95, and 772.6 s maximum
  latency under sustained three-slot contention.
- Three-slot inference is memory-safe but neither latency-safe nor context-safe for dense crops.

The evaluation-only model container was stopped after machine outputs were copied. The existing
legacy stack was not changed and continued returning HTTP 200 on port 3000.

## Verification

- Python: 47 tests passed.
- Ruff: passed.
- Frontend ESLint: passed.
- Frontend TypeScript: passed.
- Local API and frontend production images built successfully before the gate run.
- Repository secret scan found no populated Gemini key.

These checks establish software integrity; they do not offset the failed accuracy result.

## Required rework before a new Phase 2 gate

1. Make OCR-native spatial row reconstruction the always-on baseline. VLM output may propose
   structure, but it must not be the only path from a crop to rows.
2. Carry a confidence-scored schema state across page and table boundaries, including headerless
   continuations and synonyms such as `Test Name`.
3. Infer financial columns with geometry and arithmetic, prefer the rightmost line-total/net
   column, and validate `quantity × rate - discount` against the selected amount.
4. Reject ungrounded provider values. Repair descriptions and amounts from same-row OCR polygons;
   otherwise send the row to review instead of silently accepting it.
5. Add page/table roles for item detail, department summary, subtotal, document total, payment,
   and metadata before canonical fusion.
6. Use one full-context VLM slot on this CPU host. Vertically tile dense tables with overlap,
   detect context exhaustion or repetition, and retry only the affected tile. Three documents may
   remain active while heavy calls are admission-controlled.
7. Turn these four exposed bills into regression tests and construct a fresh sealed holdout from
   unseen hospitals/variants. Rerun the unchanged 85% precision and recall gate under both frozen
   evaluators.

## Stop decision

Do not create `checkpoint-phase-2`, start Phase 3, deploy the V2 demo, open port 3100, or claim
85% accuracy. Resume implementation only with the reworked Phase 2 design above and a new sealed
holdout. This is the failure behavior required by `IMPLEMENTATION_PLAN.md` and
`docs/DEMO_DEPLOYMENT_PLAN.md`.
