# Table Magic: Curved-Table Recovery and Accuracy Roadmap

## Document control

| Field | Value |
| --- | --- |
| Status | Active technical roadmap |
| Current milestone | M2 promotion evidence and M3 implementation planning |
| Last updated | 2026-09-04 |
| Primary problem | Curved or wavy photographed bills that defeat one global projective transform |
| Scope | Extraction accuracy, evidence geometry, recovery, evaluation, and promotion gates |
| Out of scope | Replacing GMoney's deterministic extractor, hospital-specific repair rules, and unsupported value invention |

This document is the decision record and milestone tracker for improving GMoney's table
extraction. It complements [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and does not
weaken its evidence, accuracy, or release gates. Every milestone below is promoted from
reproducible results, not visual inspection alone.

## Executive decision

GMoney will use a hybrid, evidence-grounded ensemble. The current deterministic GMoney
reconstruction remains the primary extractor. New preprocessing and table-understanding
components are introduced as candidates or advisors, measured in shadow mode, and promoted
only when they improve a frozen holdout without weakening evidence integrity.

The implementation order is deliberately conservative:

1. Fix the current canonical-image inconsistency.
2. Make immutable artifacts and canonical coordinates first-class.
3. Add nonlinear transform support.
4. Evaluate UVDoc in shadow mode.
5. Select preprocessing per logical table.
6. Evaluate TableRecognitionPipelineV2 and Paddle's VLM client as independent challengers.
7. Add targeted disagreement recovery.
8. Build general cell fusion or learned scoring only when the collected evidence justifies
   the additional complexity.

This combines the strongest parts of both engineering proposals while correcting three
unsafe assumptions:

- Paddle's client is not assumed to be more accurate than GMoney's direct llama.cpp adapter.
- The current 14-job, 113-page working set is not enough to train or calibrate a reliable
  learned ranker.
- UVDoc cannot affect published evidence until GMoney can persist and validate its nonlinear
  canonical-to-source mapping.

## Current verified state

The following facts are true of the implementation as of 2026-09-04:

- [`OfflineExtractor`](src/gmoney/extraction/offline.py) generates raw, geometry-normalized,
  and enhanced page candidates, then chooses one candidate for the whole page.
- Canonical table crops now come from the selected page artifact, and table adapters retain the
  canonical crop hash. M2 closure additionally binds adapter records to their page, logical table,
  and canonical artifact lineage.
- V6 persists explicit `SOURCE_RAW`, `ORIENTED_RAW`, page-candidate, table-crop, and recovery
  artifact lineage while keeping V5 jobs readable.
- [`TransformChain`](src/gmoney/contracts/evidence.py) and the active V6 traversal support only
  identity and 3x3 matrix mappings. The V6 schema reserves `DENSE_BACKWARD_GRID`, but publication
  deliberately rejects it until M3 implements byte and geometry validation.
- Recovery is already evidence-aware, scoped, fail-closed, and prohibited from changing
  grounded financial values without proof. Those protections must be retained.
- `needs_review` is already a public terminal status used by the API, frontend, filters,
  approval rules, history, and tests. Adding a new status would be a product-wide migration.
- PaddleOCR 3.7.0, PaddleX 3.7.2, and PaddlePaddle 3.2.2 are installed. Table Recognition V2,
  UVDoc, and a llama.cpp-backed PaddleOCR-VL client are available but not integrated into
  GMoney's published extraction path.

## Problem statement

A projective transform applies one homography to a flat page. It can correct rotation, skew,
and a trapezoidal camera view, but it cannot flatten a physically curved, folded, or wavy
sheet whose distortion changes across the page.

On a curved bill, different rows may bend in different directions. Column boundaries drift,
characters stretch, and local scale varies. OCR and row reconstruction may then:

- assign an MRP, quantity, rate, or amount to a neighbouring column;
- merge adjacent rows or split one cell into several tokens;
- miss digits compressed by the curve;
- lose a row whose baseline no longer aligns with neighbouring rows; or
- produce high-confidence text in the wrong structural position.

The solution must be generic. It may use page quality, geometry, evidence, and learned model
outputs, but it must not encode repair rules for a specific hospital or bill.

## Goals and non-goals

### Goals

- Improve row and critical numeric-cell recall on readable curved photographs.
- Preserve or improve precision on flat scans and existing hospital layouts.
- Ensure all reconciled table-level extractors process the same immutable canonical crop.
- Store evidence first in canonical artifact coordinates and derive source-page presentation
  polygons through a validated mapping.
- Distinguish unreadable source content from extraction failure.
- Measure improvements by cell, column, row, table, document, cohort, latency, and resources.
- Support safe abstention and reviewer escalation when the evidence is insufficient.

### Non-goals

- Replacing GMoney reconstruction with a generic Paddle pipeline.
- Enabling UVDoc unconditionally.
- Treating arithmetic reconciliation as proof of correctness.
- Accepting a VLM-only value without independent evidence.
- Training a model from the current small working set.
- Claiming 95% from public document-parsing benchmarks or from an exposed regression corpus.
- Changing the public job status enum during this roadmap unless a separate product decision
  requires it.

## Safety and evidence invariants

These invariants apply to every milestone:

1. The uploaded PDF is immutable. Each rendered page has an immutable `SOURCE_RAW` root
   artifact.
2. `SOURCE_RAW` and `ORIENTED_RAW` are distinct even when their pixels happen to match.
3. Screeners may inspect multiple page artifacts, but all table-level outputs considered for
   reconciliation must declare the same canonical table artifact ID and image hash.
4. Outputs produced from different crop hashes are observations about different artifacts and
   cannot be automatically fused.
5. One logical table has one selected canonical artifact. Switching to a runner-up switches the
   whole table and reruns its extractors; it never silently mixes coordinate systems.
6. Extraction evidence is recorded in canonical coordinates first. Source-page polygons are
   derived for validation and presentation.
7. Nonlinear mappings never map only the four corners of a rectangle. Polygon edges are sampled
   adaptively before projection.
8. Every accepted value is observed in grounded source evidence or is a permitted derived field
   with explicit operand provenance.
9. Arithmetic can rank observed candidates, request re-recognition, reduce confidence, or send
   work to review. It cannot create or modify a number to make totals balance.
10. Hospital profiles are priors. They may adjust a schema score but cannot force a token into a
    field unsupported by current evidence.
11. Validation remains fail-closed. A contract, mapping, hash, or validator defect prevents
    automatic publication.
12. Shadow components never modify the production result, review overlay, approval, or export.

## Target architecture

```text
Immutable PDF
    |
    v
SOURCE_RAW rendered page ------------------------------------+
    |                                                        |
    +--> native PDF words/glyphs (when present)              |
    |                                                        |
    v                                                        |
Source/readability diagnostics                               |
    |                                                        |
    v                                                        |
ORIENTED_RAW                                                 |
    |                                                        |
    +--> PROJECTIVE                                          |
    +--> PROJECTIVE_ENHANCED                                 |
    +--> UVDOC (shadow until promoted)                       |
    +--> UVDOC_ENHANCED (shadow until promoted)              |
    |                                                        |
    v                                                        |
Cheap OCR + layout screening per candidate                   |
    |                                                        |
    v                                                        |
Proposals mapped into SOURCE_RAW space <---------------------+
    |
    v
Cross-candidate logical-table matching
    |
    v
Deterministic top-two candidate evaluation
    |
    v
ONE CANONICAL TABLE ARTIFACT
    |
    +----------------+-------------------+-------------------+
    |                |                   |                   |
    v                v                   v                   v
GMoney OCR      Table V2 shadow     VLM challenger     Native PDF
reconstruction or targeted advisor  or rescue          alignment
    |                |                   |                   |
    +----------------+-------------------+-------------------+
                             |
                             v
                 Disagreement observations
                             |
                             v
                 Evidence-aware validation
                             |
             +---------------+----------------+
             |                                |
             v                                v
      Complete/high confidence       Targeted cell/row recovery
                                               |
                                     +---------+----------+
                                     |                    |
                                     v                    v
                                  Complete       needs_review with
                                                 structured reason
```

Page-level PaddleOCR-VL rescue is a separate path used only when screening cannot find a
credible table. Its detected table first becomes a canonical GMoney table artifact; only then
may its values enter normal grounding and validation.

## Artifact and coordinate design

### Artifact lineage

Artifacts form an immutable parent-child graph. Each child stores the mapping from its own
coordinates to its parent's coordinates. Following parent links maps any table or cell artifact
back to `SOURCE_RAW`.

```text
ArtifactRef
  artifact_id
  artifact_kind
  image_sha256
  artifact_relative_path
  width
  height
  parent_artifact_id
  producer
  producer_version
  configuration_sha256
  child_to_parent_mapping
```

`artifact_id` is the SHA-256 of canonical JSON containing artifact kind, image hash, parent ID,
producer identity, configuration hash, and mapping hash. `image_sha256` remains separate so two
semantically different derivations with identical pixels do not lose their lineage.

Artifact kinds are:

```text
SOURCE_RAW
ORIENTED_RAW
PROJECTIVE
PROJECTIVE_ENHANCED
UVDOC
UVDOC_ENHANCED
TABLE_CROP
CELL_CROP
```

### Mapping contract

```text
ArtifactMapping
  mapping_type: IDENTITY | HOMOGRAPHY | DENSE_BACKWARD_GRID
  mapping_sha256

HomographyMapping
  child_to_parent_matrix: 3x3

DenseBackwardGridMapping
  grid_relative_path
  grid_sha256
  grid_dtype: float32
  grid_shape
  child_width
  child_height
  parent_width
  parent_height
  coordinate_domain: normalized_minus_one_to_one
  interpolation: bilinear
  align_corners: true
  padding_mode
```

For UVDoc, store the smallest predicted control grid that can deterministically reproduce the
dense backward map, plus the exact interpolation metadata. Store it as a compressed NumPy
artifact rather than expanding it to a full-resolution grid for every page.

The minimum required direction is canonical child to source parent. A source-to-canonical
inverse is optional because nonlinear inversion may be ambiguous. Candidate proposals and
evidence are projected from candidate space into `SOURCE_RAW`, so a trusted inverse is not
required for matching or presentation.

### Polygon mapping

To map a canonical polygon through a dense grid:

1. Begin with its vertices and edges in canonical coordinates.
2. Recursively subdivide each edge until the projected midpoint differs from the projected
   straight segment by at most one source pixel.
3. Map every retained point through the child-to-parent grid and then the remaining parent chain.
4. Reject non-finite, out-of-bounds, self-intersecting, or orientation-reversing results.
5. Simplify only after mapping, with a maximum one-pixel source-space error.

Synthetic and projective mappings must round-trip within two pixels where a valid inverse is
available. Dense mappings are validated in the required canonical-to-source direction and by
reproducing the rectified artifact.

### Page and table artifacts

```text
PageArtifact
  artifact: ArtifactRef
  page_number
  dpi
  quality_metrics
  route_reasons
  transform_metrics

CanonicalTableArtifact
  artifact: ArtifactRef
  logical_table_id
  page_artifact_id
  crop_polygon_in_page_artifact
  crop_polygon_in_source_raw
  context_margin
  table_type_hint
```

The table crop includes a bounded context margin so headers and bordering tokens are not lost.
The margin is part of the configuration hash and must be identical for all adapters in an A/B
comparison.

### Evidence contract

New extraction publishes an explicit V6 envelope; V5 remains readable for historical jobs and
is labelled as legacy evidence until reprocessed.

```text
EvidenceRefV2
  artifact_id
  artifact_sha256
  canonical_polygon
  source_page_polygon
  source_page_number
  ocr_token_ids
  extractor
  model_name
  model_version
  recognition_variant
```

`source_page_polygon` is a materialized, validated projection for clients. The canonical polygon
and artifact lineage remain authoritative if the projection code changes later.

Every table-level adapter response also contains:

```text
input_artifact_id
input_artifact_sha256
adapter_name
adapter_version
configuration_sha256
latency_ms
cache_hit
```

Publication rejects an adapter response whose declared input does not match the canonical table
artifact selected for that logical table.

## Routing and preprocessing candidates

### Source and readability diagnostics

The router records one source-class diagnostic:

```text
NATIVE_TEXT_PDF
FLAT_SCAN
CAMERA_PHOTO
CAMERA_PHOTO_WITH_CURVATURE
UNREADABLE_OR_INCOMPLETE
```

The diagnostic is a routing hint, not hard truth. Raw and projective candidates remain available
even when curvature is suspected. Curvature signals may include baseline bending, row-dependent
column drift, inconsistent local skew, and safe deformation estimates.

Unreadable content remains a `needs_review` result with a blocking issue such as:

```json
{
  "code": "source_unreadable",
  "reason": "glare_occludes_critical_cells",
  "recommended_action": "reupload"
}
```

This avoids an immediate status migration while separating source quality from model failure in
metrics. A distinct `needs_reupload` status requires a separate API/product decision.

### Candidate factory

The immutable candidate tree is:

```text
SOURCE_RAW
    |
    +-- ORIENTED_RAW
            |
            +-- PROJECTIVE
            |       |
            |       +-- PROJECTIVE_ENHANCED
            |
            +-- UVDOC
                    |
                    +-- UVDOC_ENHANCED
```

Rules:

- Keep `SOURCE_RAW` even when orientation confidence is high.
- Apply UVDoc to `ORIENTED_RAW`, not to projective output by default.
- Apply enhancement after geometry correction.
- Permit `PROJECTIVE -> UVDOC` only as a separately named benchmark experiment.
- UVDoc is generated in shadow mode for the photographed/curved cohort until promoted.
- A UVDoc candidate is invalid if its mapping contains non-finite values, any fold-over inside an
  eligible table, or more than 0.5% out-of-bounds samples in that table.
- Log mean/max displacement, local scale percentiles, anisotropy, Jacobian determinant percentiles,
  fold-over count, out-of-bounds rate, and baseline-straightness change.

### Logical-table matching

Each candidate produces table proposals in its own coordinate system. Proposals are mapped into
`SOURCE_RAW` and matched deterministically:

1. Use `SOURCE_RAW` proposals as anchors when available.
2. Match derivative proposals to anchors using polygon overlap, centre/reading order, table type,
   and normalized header-token similarity.
3. Resolve competing matches with deterministic maximum-weight bipartite matching.
4. A derivative-only proposal becomes a logical table only when at least two independent
   candidates agree or a page-level rescue is grounded and validated.
5. Ambiguous matches abstain; they do not merge tables automatically.
6. Persist matching features and the reason for every accepted or rejected edge.

Logical table IDs are stable hashes of the document, page number, quantized source polygon,
reading order, and normalized header signature.

### Deterministic candidate ranking

Stage 1 screens every page candidate using:

- OCR coverage and confidence distribution;
- financial-token and header coverage;
- table proposal count and confidence;
- baseline straightness and column-position stability;
- duplicate-token and invalid-character rates;
- deformation magnitude and transform validity; and
- the current reconstruction-quality tuple.

The best two candidates per logical table proceed to full GMoney reconstruction. Stage 2 scores:

- required-column and header/schema coverage;
- aligned publishable rows and populated grounded fields;
- evidence linkage rate;
- arithmetic and document-total diagnostics;
- duplicate, conflicting, or missing-row indicators;
- cross-channel agreement when shadow observations exist; and
- a distortion penalty.

Selection is deterministic. Ties prefer the least transformed valid candidate in this order:
`ORIENTED_RAW`, `PROJECTIVE`, `PROJECTIVE_ENHANCED`, `UVDOC`, `UVDOC_ENHANCED`.

## Extraction channels

### A — GMoney reconstruction

GMoney remains authoritative. Screening OCR is not automatically final OCR. Once a canonical
table is selected, the primary table OCR and reconstruction operate on that exact table crop,
record its hash, and retain the existing profile, alias, cross-page schema, typing, grounding,
and validation behaviour.

### B — TableRecognitionPipelineV2

Table V2 begins as an offline/sequential shadow runner over frozen canonical crops. Its internal
document preprocessing and layout detection are disabled because GMoney owns the artifact:

```text
use_doc_orientation_classify = false
use_doc_unwarping = false
use_layout_detection = false
use_ocr_results_with_table_cells = true
```

Its initial outputs are observations only:

- wired/wireless classification;
- cell polygons and row/column structure;
- alternative splits for OCR regions crossing cell boundaries; and
- cell text disagreements.

It is not enabled for every live table. After shadow evaluation it may be promoted only as a
targeted advisor for table types or validation failures where it has demonstrated unique gains.

### C — PaddleOCR-VL

The existing direct llama.cpp adapter remains the default until an A/B test says otherwise.

The challenger uses Paddle's client against the same llama.cpp server with:

```text
use_doc_orientation_classify = false
use_doc_unwarping = false
use_layout_detection = false
prompt_label = table
```

Both lanes receive identical canonical crop bytes, prompts, temperature, token limits, and
grounding checks. Client orchestration is treated as a testable implementation choice, not an
automatic accuracy improvement.

A full PaddleOCR-VL page workflow is a separate missed-table rescue lane. It may propose table
regions but cannot bypass GMoney's canonical artifact, evidence, or validation contracts.

### D — Native PDF text

Native PDF extraction is a separate track because it cannot help curved photographs. When a PDF
contains a usable text layer, retain words/glyphs, coordinates, line/font metadata, rotation, and
embedded-image information. Map PDF coordinates to `SOURCE_RAW` and treat the text as another
evidence source, never as trusted truth without rendered-page alignment.

## Targeted disagreement recovery

Before building general cell fusion, record disagreements and recover only failed cells or rows:

1. Re-recognize the canonical cell crop at higher scale.
2. Try conservative grayscale and contrast variants of that crop.
3. Use validated Table V2 geometry to split merged OCR regions.
4. Reconstruct the affected row with neighbouring-row context.
5. Ask the VLM to read the canonical table, row, or cell crop.
6. Run the complete table on the runner-up preprocessing artifact.
7. Switch the whole table only when the runner-up is safely better.
8. Apply local mesh/baseline correction only to remaining curved regions.
9. Route unresolved evidence to review.

Each observation uses this shape:

```text
CellObservation
  logical_table_id
  row_anchor
  canonical_field
  raw_text
  normalized_value
  extractor
  artifact_id
  evidence_polygon
  supporting_token_ids
  model_confidence
  recognition_variant
```

The validator may choose only among grounded observations. A VLM-only suggestion must be
re-read by OCR or supported by another extractor before automatic acceptance.

## Conditional cell fusion and learning

General cell-lattice fusion is intentionally conditional. Build it only if shadow results show:

- at least two secondary channels each provide independently correct recoveries;
- each channel uniquely fixes at least ten reviewed critical cells missed by the primary path;
- oracle fusion improves critical numeric-cell recall by at least two absolute percentage points;
  and
- all winning candidates share the canonical artifact and can satisfy evidence validation.

If these conditions are not met, retain the simpler targeted-recovery design.

Do not train a candidate ranker until the corpus contains at least:

- 1,000 labelled logical tables;
- 50 hospital/template groups;
- 100 reviewed cases where a non-default candidate is correct; and
- an untouched hospital-, template-, and time-separated holdout.

Confidence calibration additionally requires enough reviewed incorrect outcomes to estimate
field-specific false-accept rates. Until then, log all proposed features and use deterministic
thresholds and abstention.

## Metrics and corpus policy

### Corpus

The current 14 jobs and 113 pages are useful for regression, failure reproduction, shadow
feasibility, and resource benchmarking. They are not sufficient to train a ranker or prove a
90–95% population result.

The frozen evaluation manifest records:

- source and gold hashes;
- hospital, template, and capture-time groups;
- native PDF, flat scan, ordinary photo, curved photo, fold, glare, blur, low contrast, and
  incomplete/unreadable labels;
- wired, wireless, multi-table, and multi-page-table cohorts;
- exact cell values, canonical field roles, row/column identity, table polygons, cell polygons,
  totals, and cross-page continuation; and
- readable versus unreadable at cell and table level.

Splits are by hospital, template, and time—not random pages from the same layout.

### Required reporting

| Metric | Improvement target on frozen readable holdout |
| --- | ---: |
| Critical numeric cell precision | >= 98% |
| Critical numeric cell recall | >= 95% |
| Line-item row recall | >= 95% |
| Correct column assignment | >= 98% |
| Header/schema accuracy | >= 97% |
| Grand-total exact accuracy | >= 99% |
| Auto-accepted document correctness | >= 99% |

Also report description token F1, normalized cell exact match, structure score, complete-row,
complete-table and complete-document exact match, straight-through processing coverage, review
rate, recommended-reupload rate, and every existing GMoney release metric.

Report each metric overall and for curved photographs, other photographs, flat scans, native
PDFs, known layouts, unseen layouts, wired tables, wireless tables, and critical financial
columns. Unreadable cells are excluded from readable-cell accuracy but reported separately; they
must never disappear from completeness reporting.

### Ablations

Run each addition independently:

```text
Baseline GMoney
Baseline + canonical crop consistency
Baseline + artifact/coordinate V6
Baseline + UVDoc shadow candidate
Baseline + per-table selection
Baseline + Table V2 observations
Baseline + Paddle VLM client challenger
Baseline + targeted disagreement recovery
Baseline + conditional cell fusion
```

No aggregate improvement may hide a regression in critical numeric precision, flat scans,
existing hospitals, or evidence-grounding integrity.

## Milestone tracker

Allowed states are `not_started`, `in_progress`, `blocked`, `shadow`, `promoted`, and `rejected`.

| ID | Milestone | State | Required gate | Evidence |
| --- | --- | --- | --- | --- |
| M0 | Freeze baseline and labels | in_progress | Deterministic replay and per-column/cohort baseline | Existing frozen release corpus; Table Magic cohort report pending |
| M1 | Canonical-image consistency | in_progress | All table adapters share one crop hash; no flat regression | `afe8caa`; frozen non-regression report pending |
| M2 | Artifact and evidence V6 | in_progress | V5 read compatibility; tamper-safe lineage; <=2 px linear mapping error | `f4e3b8f`, `d0dd91c`, `556e4fd`; promotion review pending |
| M3 | Dense transform foundation | not_started | Synthetic dense mappings pass and invalid grids fail closed | Implementation plan below |
| M4 | UVDoc feasibility/shadow | not_started | Reproducible grid, valid geometry, curved gain, no critical regression | TBD |
| M5 | Per-table matching/selection | not_started | Stable identity, no lost/duplicate tables, holdout improvement | TBD |
| M6 | Table V2 shadow benchmark | not_started | Unique recoveries and acceptable GPU/latency cost | TBD |
| M7 | PaddleOCR-VL client A/B | not_started | Frozen metric winner selected; direct adapter retained on tie | TBD |
| M8 | Targeted disagreement recovery | not_started | Recall gain without precision or grounding regression | TBD |
| M9 | Conditional cell fusion | not_started | Complementarity trigger and >=2 pp oracle recall gain | TBD |
| M10 | Native PDF route | not_started | Digital-PDF gain without image-evidence regression | TBD |
| M11 | Learned ranker/confidence | blocked | Corpus sufficiency thresholds met | TBD |

### M0 — Freeze the baseline

Deliverables:

- Frozen corpus and label manifest with source hashes and split policy.
- Current release SHA, image/model versions, configuration, and resource envelope.
- Per-field, per-column, per-table, per-document, and per-cohort report.
- Explicit inventory of curved-table failures and unreadable cells.

Gate: two cached replays produce identical machine output and the baseline report contains every
required metric. Until then, all later accuracy claims are provisional.

### M1 — Fix canonical-image consistency

Deliverables:

- Retain table proposals in candidate coordinates as well as source coordinates.
- Crop the selected page artifact rather than the raw page.
- Run final table OCR/reconstruction and recovery advisors on that crop.
- Persist and validate the canonical crop hash in every adapter response and cache key.

Gate: no accepted adapter output has a different crop hash, source mappings remain valid, and
the frozen flat/known-layout corpus has no critical precision or recall regression.

### M2 — Introduce artifact and evidence V6

Deliverables:

- `ArtifactRef`, `CanonicalTableArtifact`, mapping union, and `EvidenceRefV2`.
- Explicit `SOURCE_RAW` and `ORIENTED_RAW` lineage.
- V6 writer, V5 reader, legacy-job labelling, reprocessing path, and certification updates.
- Hash and lineage validation at publication.

Gate: V5 jobs remain readable, V6 jobs are deterministic, matrix mappings remain within two
pixels, and missing/tampered artifacts fail closed.

### M3 — Add dense transform support

Deliverables:

- Dense-grid serialization and content hashing.
- Canonical-to-parent point and adaptive-polygon mapping.
- Jacobian, fold-over, local scale, anisotropy, and bounds diagnostics.
- Synthetic curved-grid fixtures and overlays.

Gate: known synthetic warps map polygons within two source pixels; any non-finite grid,
self-intersection, fold-over, corrupt digest, or incompatible metadata is rejected.

#### M2 closure before M3 implementation

The V6 implementation is present, but M2 remains `in_progress` until its promotion evidence is
recorded. The code closure consists of:

- Bind every `TableAdapterInputV2` to a page and logical table, and reject an adapter input that
  is not the canonical table artifact or one of its descendants.
- Emit that page/table identity during V5-to-V6 projection and keep deterministic adapter
  ordering.
- Validate exactly one `SOURCE_RAW`, one `ORIENTED_RAW`, and one selected page artifact for every
  declared page, with explicit fatal issue codes at the publication boundary.
- Test cross-table adapter substitution, recovery-artifact descendants, missing page roles, and
  multiple selected candidates.

Local closure verification on 2026-09-04 passed 1,111 backend tests, frontend tests, frontend
lint, frontend type checking, the production frontend build, Ruff, diff checking, and every
CPU/demo/GPU/admin Compose configuration. This verifies the repository state but is not a
substitute for the cached-replay, frozen-corpus, or deployed-release gates below.

Before M2 is promoted, complete the non-code gate:

1. Run the complete backend, frontend, and Compose verification from `connections.md`.
2. Run two cached replays and confirm byte-identical V6 result, manifest, validation, and
   certification payloads.
3. Run the frozen flat/known-layout corpus and record critical precision/recall non-regression.
4. Exercise a staged V5-to-V6 reprocess and prove that the V5 source remains readable and
   rollback-safe.
5. Tamper with one image, mapping, manifest entry, and adapter table reference in an isolated
   workspace and record the expected fatal failures.
6. Complete an M2 milestone review using the template below. Deployment is a separate certified
   operation; repository completion must not be presented as GPU deployment completion.

#### M3 scope boundary

M3 implements and certifies the general dense-transform substrate. It does not invoke UVDoc,
select a nonlinear candidate, change extraction results, or introduce a feature flag. Those are
M4 concerns. V5 `TransformChain` remains matrix-only for historical compatibility; dense mapping
is added only to the V6 artifact graph.

#### M3 frozen technical decisions

- Store one C-contiguous `float32` control grid in a compressed NumPy container with a single
  fixed member name. Loading always uses `allow_pickle=false` and rejects extra members.
- Hash the exact stored grid bytes. Bind that digest, path, shape, image dimensions, coordinate
  domain, interpolation, alignment, and padding metadata into `mapping_sha256`.
- Interpret each grid sample as a canonical-child to parent coordinate in normalized `[-1, 1]`
  space with bilinear interpolation and `align_corners=true`.
- Keep grid paths relative to the job artifact root. Absolute paths, traversal, symlinks, missing
  files, digest conflicts, and oversized or malformed arrays fail closed.
- Load grids through an explicit resolver supplied to geometry traversal. Do not use a process
  global artifact root or an implicit current directory.
- Map canonical polygons through the complete child-to-`SOURCE_RAW` chain. Densify each edge
  recursively until its projected midpoint differs from the projected chord by at most one
  source pixel.
- Bound adaptive mapping by a fixed recursion depth and point budget. Failure to converge is a
  validation error, never permission to fall back to corner-only mapping.
- Reject non-finite coordinates, any source-bounds violation, non-positive Jacobians, fold-over,
  orientation reversal, and self-intersection. Retain unsimplified polygons unless a later
  source-space simplification can prove at most one-pixel error.

#### M3 implementation slices

**M3.1 — Contract and deterministic grid storage**

- Harden `DenseBackwardGridMapping` so `grid_shape` is exactly `(height, width, 2)`, both control
  dimensions are at least two, and child/parent dimensions support `align_corners` semantics.
- Add a geometry-owned serializer/loader that writes atomically, returns the content digest, and
  validates path containment, member name, dtype, rank, shape, byte order, and finite values.
- Keep model validation filesystem-independent; perform byte/path validation at the publication
  boundary.

Commit gate: deterministic serialization round trips, identical inputs produce identical bytes,
and malformed containers are rejected.

**M3.2 — Point, chain, polygon, and diagnostic geometry**

- Implement bilinear sampling from child pixel coordinates into parent pixels.
- Extend artifact-chain traversal to accept a dense-grid resolver and support mixed identity,
  homography, and dense mappings.
- Add adaptive closed-polygon mapping, source bounds checks, signed-area/orientation checks, and
  non-adjacent segment intersection detection.
- Produce typed diagnostics for mean/maximum displacement, local-scale percentiles, anisotropy,
  Jacobian determinant percentiles, fold-over count, and out-of-bounds rate.

Commit gate: identity, affine-equivalent, smooth sinusoidal, and mixed dense/homography fixtures
match their analytic source coordinates within two pixels.

**M3.3 — V6 publication and certification integration**

- Permit a valid dense mapping in `ExtractionResultV6` and remove the M2 reservation failure.
- Verify the grid once per digest during publication, then use the verified resolver for table,
  token, and evidence projections.
- Compare materialized source polygons to adaptively projected canonical polygons within the
  two-pixel contract, including vertex-count and ordering checks.
- Add mapping files to the V6 artifact inventory so `job_certification_v3`, evidence bundles,
  reprocessing, recertification, and release snapshots bind the grid bytes.
- Preserve V5 reading, review projection, export shape, and legacy certification behaviour.

Commit gate: a valid dense V6 workspace certifies and survives restart/recertification; changing
or removing its grid invalidates approval and export.

**M3.4 — Negative fixtures, overlays, and milestone evidence**

- Add fixtures for bad digest, missing grid, path traversal, symlink, unexpected archive member,
  wrong dtype/shape/domain/dimensions, NaN/Inf, out-of-bounds coordinates, zero/negative Jacobian,
  fold-over, orientation reversal, self-intersection, and adaptive-mapping budget exhaustion.
- Generate deterministic overlays showing control-mesh lines, original canonical polygons, and
  mapped source polygons. Overlays are diagnostic artifacts and never substitute for numeric
  assertions.
- Run the full local verification matrix and record elapsed time and peak memory for representative
  page and table grids.
- Complete an M3 review with exact commands, fixture hashes, mapping errors, failure codes, and
  rollback conditions.

Commit gate: every invalid fixture fails closed with a stable issue code, all synthetic valid
warps stay within two source pixels, and V5/V6 linear mapping tests remain unchanged.

#### M3 expected code and test ownership

| Area | Primary files |
| --- | --- |
| Contract | `src/gmoney/contracts/v6.py` |
| Serialization and sampling | `src/gmoney/geometry/dense.py` |
| Artifact traversal and polygons | `src/gmoney/geometry/artifacts.py` |
| Publication validation | `src/gmoney/extraction/validation.py` |
| Certification inventory | `src/gmoney/demo/store.py` |
| Release snapshot binding | `src/gmoney/evaluation/release_gate.py` |
| Contract/geometry tests | `tests/test_v6_contracts.py`, `tests/test_dense_artifacts.py` |
| Publication/certification tests | `tests/test_extraction_validation.py`, `tests/test_demo_worker.py` |
| Contract documentation | `docs/EXTRACTION_RESULT_V6.md` |

M4 may begin only after the M3 review is promoted. Its UVDoc adapter must consume the public M3
serializer and mapping APIs; it must not introduce a second grid format or bypass M3 validation.

### M4 — Add UVDoc in shadow mode

Deliverables:

- A GMoney-owned, version-pinned UVDoc adapter that emits the rectified image and the control grid
  used to produce it.
- A compatibility probe for PaddleOCR 3.7.0/PaddleX 3.7.2, model/config digests, and expected
  tensor semantics.
- `UVDOC` and `UVDOC_ENHANCED` shadow artifacts plus transform telemetry.
- Curved and flat ablation report.

The adapter may use a controlled local wrapper/fork of the UVDoc forward path, but it must not
monkey-patch unspecified private state at runtime. If a stored grid cannot reproduce the output
within one pixel per channel, UVDoc remains experimental and cannot be selected.

Gate: at least one pre-registered curved failure improves without a new critical error on the
frozen flat cohort. Passing this gate permits continued shadow evaluation, not a 95% claim.

### M5 — Match logical tables and select per table

Deliverables:

- Deterministic source-space table matching and stable logical table IDs.
- Logged match scores, ambiguity decisions, and derivative-only proposals.
- Top-two deterministic ranking and full reconstruction per table.
- Whole-table canonical switching with stable evidence lineage.

Gate: repeated runs produce identical matches; no source table is silently lost or duplicated;
ambiguous matches abstain; curved-table metrics improve without violating release floors.

### M6 — Benchmark Table Recognition V2

Deliverables:

- Offline/sequential shadow runner using canonical table artifacts.
- Wired/wireless structure, cell polygon, split-OCR, and disagreement records.
- Peak GPU VRAM, host RSS, model initialization, p50/p95 table latency, and throughput report.

Gate: no OOM, at least 20% peak VRAM headroom on the target GPU, no more than 20% throughput loss
for the proposed targeted route, and reviewed unique recoveries. Otherwise reject or keep it
offline-only.

### M7 — A/B test PaddleOCR-VL orchestration

Deliverables:

- Direct and Paddle-client adapters with identical canonical input and decoding configuration.
- Exact-value, row/table, grounding, hallucination, latency, and resource comparison.
- Separate full-page missed-table rescue experiment.

Gate: choose the frozen-metric winner. Retain the direct adapter on a metric tie or when any gain
does not justify the operational cost. The page rescue lane must first canonicalize and validate
its detected tables.

### M8 — Add targeted disagreement recovery

Deliverables:

- Persisted `CellObservation` records from eligible secondary channels.
- Cell/row re-OCR, Table V2 split geometry, VLM reread, and runner-up table switch ladder.
- Validation rules that select only grounded observed candidates.

Gate: critical-cell recall improves on the frozen holdout with no critical precision loss, no
new unsupported values, and no increase in incorrectly auto-accepted documents.

### M9 — Decide on universal cell fusion

Build a canonical cell lattice only when the complementarity trigger in this document passes.
If it does not pass, mark this milestone `rejected` and retain targeted recovery. Rejection is a
successful outcome when it avoids complexity without losing demonstrated accuracy.

### M10 — Add native PDF extraction

Deliverables:

- Native word/glyph adapter and PDF-to-rendered-page mapping.
- Alignment and disagreement diagnostics against image OCR.
- Digital-PDF cohort report.

Gate: critical exact-match or description recall improves on digital PDFs without allowing an
unaligned text layer to bypass image evidence.

### M11 — Train ranking and confidence models

This milestone remains `blocked` until the stated corpus sufficiency thresholds are met. When
unblocked, pre-register features, targets, splits, and promotion criteria before training. Compare
against the deterministic scorer and retain the deterministic scorer unless the learned model
wins the untouched holdout and calibration checks.

## Rollout and promotion policy

Each optional component uses an independent mode:

```text
off       component is not invoked
shadow    output and telemetry are retained but cannot change publication
enabled   output may affect the result under its frozen promotion decision
```

Suggested configuration names are:

```text
GMONEY_UVDOC_MODE
GMONEY_TABLE_V2_MODE
GMONEY_PADDLE_VL_CLIENT_MODE
GMONEY_NATIVE_PDF_MODE
```

Defaults remain `off`. An `enabled` component requires a frozen promotion manifest containing:

- release Git SHA and container image digests;
- corpus manifest and gold digests;
- component, model, prompt, preprocessing, and contract versions;
- configuration and promotion-report hashes;
- aggregate and cohort metric deltas;
- grounding/privacy gates;
- latency, VRAM, RSS, and throughput results; and
- explicit rollback conditions.

Roll back to `off` when evidence validation regresses, a model/config digest changes, resource
headroom is violated, or a monitored cohort crosses its frozen floor. Re-enable through a new
promotion decision rather than editing the old decision.

## Test matrix

Tests must cover:

- correct and incorrect orientation classification while retaining `SOURCE_RAW`;
- identity, scale, crop, homography, and nonlinear mapping chains;
- adaptive mapping of curved cell polygons;
- corrupt, missing, out-of-bounds, non-finite, and folded dense grids;
- canonical crop/hash mismatch between OCR, Table V2, and VLM;
- raw, projective, UVDoc, and enhanced-candidate ties and regressions;
- cross-candidate table matching, nested/adjacent tables, ambiguous matches, and missed tables;
- wired and wireless tables, merged OCR boxes, wrapped descriptions, and multi-page schemas;
- critical numeric disagreements, negative/refund rows, totals, and arithmetic abstention;
- VLM hallucination, truncation, timeout, cache mismatch, and page-rescue grounding;
- unreadable, clipped, glare-covered, blurred, and incomplete sources;
- cancellation and worker restart during artifact generation;
- deterministic cached replay and targeted reprocessing;
- V5 historical results and V6 certification; and
- feature modes proving shadow output cannot alter production results.

## Milestone evidence template

Copy this section for each milestone review:

```text
# Table Magic Mx Review

Status: promote | hold | reject
Date:
Owner:
Git SHA:
Image digests:
Corpus manifest SHA-256:
Gold/evaluator versions:
Component/model/config versions:

## Change evaluated

## Commands and environment

## Accuracy results
- Overall before/after/delta
- Critical fields before/after/delta
- Curved-photo cohort before/after/delta
- Flat/known-layout non-regression
- Unreadable/reupload accounting

## Evidence integrity
- Artifact/hash validation
- Mapping error
- Unsupported accepted values

## Resources
- Peak VRAM and RSS
- Initialization time
- p50/p95 latency
- Throughput

## Regressions and unresolved risks

## Gate decision and rollback condition
```

## Risks and explicit mitigations

| Risk | Mitigation |
| --- | --- |
| UVDoc looks visually better but damages digits | Select on downstream frozen metrics; retain raw/projective candidates |
| Grid capture depends on Paddle internals | Version-pin a controlled adapter, verify output reproduction, fail closed on incompatibility |
| Dense maps consume excessive storage | Persist compressed control grids and interpolation metadata rather than full-resolution maps |
| Per-table matching merges adjacent tables | Deterministic source-space matching, ambiguity abstention, stable inventory validation |
| Table V2 exhausts GPU memory | Offline sequential shadow benchmark, telemetry, targeted invocation, resource gate |
| Paddle client changes VLM output unexpectedly | Same-input A/B test; keep current direct adapter on tie/regression |
| Ensemble adds complexity without unique gains | Targeted observations first; explicit complementarity gate before fusion |
| Arithmetic hides OCR errors | Restrict it to observed candidates and escalation |
| Small corpus overstates improvement | Frozen grouped splits, readable/unreadable accounting, no ML or 95% claim from current set |
| New status breaks clients/history | Keep `needs_review` and add structured unreadable/reupload reason |
| Reviewer corrections leak into evaluation | Score untouched machine output and keep review overlays separate |

## References

- [PaddleOCR-VL usage tutorial](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [General Table Recognition V2 pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/table_recognition_v2.html)
- [Paddle text image rectification/UVDoc module](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_image_unwarping.html)
- [UVDoc: Neural Grid-based Document Unwarping](https://arxiv.org/html/2302.02887v2)
- [GMoney Phase 3 implementation review](docs/reviews/phase-3.md)
- [GMoney consolidated architecture](sol-architecture-v2.md)

The external benchmark figures in the Paddle documentation are context only. GMoney promotion
decisions use hospital-bill cell, column, row, table, and document measurements from the frozen
GMoney corpus.
