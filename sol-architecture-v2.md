# Hospital Bill Extraction V2 — Consolidated Solution Architecture

**Status:** Approved architecture plan; no application implementation in this document

**Supersedes for implementation planning:** `fable-architecture.md`, `sol-architecture.md`, and `sol-plan.md`

**Primary objective:** Sustain at least 90% aggregate row precision and recall, never fall below an 85% cohort/confidence-bound floor, and support approximately 5,000 hospitals with 3–4 bill variants each.
**Delivery decision:** Build a new end-to-end service in a separate repository, database, API, and UI. The existing service remains read-only for historical access.

---

## 1. Executive decision

Build an evidence-first, page- and table-routed extraction platform with a confidence-driven inference cascade:

1. Preserve the uploaded PDF immutably.
2. Correct page and table geometry through reversible transforms.
3. Run inexpensive self-hosted OCR on every relevant page.
4. Route each page/table independently using page type, hospital identity, and layout-profile evidence.
5. Use a profile-guided spatial reconstruction path for active known layouts.
6. Use a self-hosted document VLM for unknown, ambiguous, drifted, or difficult layouts.
7. Use Vertex Gemini only to adjudicate unresolved table crops with strict evidence grounding.
8. Fuse provider candidates into one auditable canonical ledger.
9. Use financial relationships as validation and escalation signals, never as permission to invent, remove, or alter rows.
10. Route irrecoverable or calibrated uncertainty to targeted human review.

The new architecture does not extend the current fixed-band and hospital-rule-pack parser. It retains the current product's useful concepts—gold data, evidence review, explicit row dispositions, guarded approval/export, hospital identity, reconciliation diagnostics, and auditability—but gives them a new extraction foundation.

### 1.1 Accuracy policy

- Engineering release target: at least 90% aggregate row precision and 90% aggregate row recall before human correction.
- Hard floor: at least 85% macro/cohort precision and recall, including confidence-interval lower bounds.
- Known-layout automatic route target: at least 95% precision and 95% recall.
- No aggregate pass may conceal a failing high-volume layout, capture type, page type, or unseen-layout cohort.

### 1.2 Cost policy

- Known active layouts normally use only self-hosted OCR and the profile-guided reconstruction engine.
- At least 95% of ordinary active-layout bills must make zero Gemini calls.
- Expensive processing is performed at page/table/crop granularity, never automatically for the entire bill.
- Total cost includes infrastructure, managed-provider charges, storage, and reviewer time.

### 1.3 Safety policy

- No automatically accepted field without source evidence.
- No incomplete document may be approved or exported as complete.
- No malformed OCR number may cause database failure or silently become a financial value.
- No layout may become active from two seed bills or reconciliation alone.
- No model output may be treated as correct because it is valid JSON or because totals happen to match.

---

## 2. Source-plan assessment

### 2.1 Primary base: `sol-plan.md`

`sol-plan.md` is the strongest implementation base because it specifies:

- concrete service and infrastructure boundaries;
- page/table routing;
- self-hosted and managed inference routes;
- persistence entities;
- public APIs;
- review workflows;
- model/profile lifecycle;
- test cases;
- delivery phases and go/no-go gates.

V2 retains that structure while correcting premature or conflicting choices.

### 2.2 Safety and evaluation additions from `sol-architecture.md`

V2 incorporates:

- a provider-neutral immutable evidence contract;
- original-document immutability and reversible coordinate transforms;
- global extraction before layout optimization;
- calibrated uncertainty and out-of-distribution routing;
- separate workflow, completeness, review, layout, approval, and export state axes;
- reconciliation as a soft validator rather than an extractor;
- a larger staged corpus and confidence-bound gates;
- explicit prohibition against hospital-specific executable repair branches;
- provider bake-offs measured by end-to-end hospital-bill rows.

### 2.3 Ideas retained from `fable-architecture.md`

V2 retains:

- the concrete failure-taxonomy-to-mitigation map;
- strict typed cell parsing;
- zero-yield escalation instead of confident empty output;
- token-ID and polygon grounding for generative output;
- an early accuracy-spine milestone that can falsify the architecture before the full platform is built;
- arithmetic residuals as diagnostics for locating questionable evidence.

### 2.4 Fable decisions explicitly rejected

The following do not carry into V2:

- treating reconciliation as a near-perfect correctness oracle;
- automatically repairing rows to force a printed total;
- activating a layout after approximately three reconciled bills;
- assuming that agreement between two machine paths is equivalent to independent gold validation;
- requiring human review only when reconciliation fails;
- accepting projected latency or cost figures before measurement.

A row set can reconcile while containing category totals instead of detail rows, compensating errors, missing descriptions, or wrong quantities/rates. Row-level gold remains authoritative.

---

## 3. Problem definition and constraints

The system must process image-only PDFs and images produced from mobile captures, scanners, or Xerox copies. The source population includes:

- borderless, partially bordered, and bordered tables;
- multiple tables on one page;
- different page schemas in one bill;
- summary, hospital-detail, pharmacy, laboratory, receipt/payment, narrative, cover, blank, and handwritten pages;
- full-page and table-local rotation;
- fine skew and perspective/keystone distortion;
- curved pages, shadows, glare, low contrast, compression, blur, and small text;
- descriptions containing dates, batch numbers, expiry values, phone-like numbers, registration IDs, and product codes;
- wrapped descriptions and continuation rows;
- negative pharmacy returns and refunds;
- legitimate repeated rows with identical text and amount;
- section/category totals, document totals, payments, deposits, metadata, and footer noise mixed with detail rows.

At 5,000 hospitals and 3–4 variants per hospital, the expected population is approximately 15,000–20,000 variants before drift. One executable parser per variant is not viable.

### 3.1 Non-goals for the first production release

- Perfect automatic extraction of handwriting.
- One trained model per hospital or layout.
- Automatic correction of unsupported values using arithmetic.
- Importing current schemas, rule packs, layout fingerprints, or activation state.
- Preserving the legacy API or database shape.
- Rebuilding the new system inside the current repository.

---

## 4. Accuracy and release contract

Accuracy is measured on untouched machine output before human correction.

### 4.1 Row metrics

| Metric | Production minimum |
|---|---:|
| Aggregate micro row precision | 90% |
| Aggregate micro row recall | 90% |
| Aggregate micro row F1 | 90% |
| Macro document/layout precision | 85% |
| Macro document/layout recall | 85% |
| 95% bootstrap CI lower bound for aggregate precision | 85% |
| 95% bootstrap CI lower bound for aggregate recall | 85% |
| Unseen-hospital/layout precision and recall | 85% each |
| Known-layout automatic-route precision and recall | 95% each |
| Negative/return-row recall | 100% on release set |
| Incomplete-page exports | 0 |

Known, unseen-layout, digital/ordinary scan, mobile, Xerox, skew/perspective, rotated, long-document, and multi-table cohorts must be reported independently.

### 4.2 Field metrics

| Populated field | Minimum accuracy |
|---|---:|
| Amount/net amount | 95% within INR 0.01 |
| Description | 90% after approved normalization |
| Quantity | 90% |
| Unit price/rate | 90% |
| Gross amount | 90% |
| Discount | 90% |
| Service date | 85% |
| Request number | 85% |
| Service code | 85% |
| HSN code | 85% |

A field can be excluded only when the profile's reviewed field contract declares it unsupported. Absence in a small sample is not proof of unsupported status.

### 4.3 Evaluators

- Preserve the current matcher as `legacy_metric_v1` for direct comparison with 47.53% precision and 62.52% recall.
- Add `canonical_metric_v2` using gold page, table, row order, polygons, and deterministic one-to-one matching independent of later field scoring.
- Report field metrics after canonical row matching.
- Freeze evaluator versions before model/prompt results are inspected.
- Both row evaluators must pass the production row gates.
- Accepted and review-pending detail candidates form the scored visible ledger. Uncertainty cannot be hidden by deleting rows.

---

## 5. Target architecture

```text
New Next.js UI / API clients
             |
             v
FastAPI document and review API --------------------------+
             |                                            |
             v                                            v
Immutable object storage                           PostgreSQL domain state
             |                                            |
             +------------------+-------------------------+
                                v
                       Temporal document workflow
                                |
                    page inventory and fan-out
                                |
                                v
              quality assessment + reversible normalization
                                |
                                v
                   OCR tokens + page/table proposals
                                |
                                v
                 page classification + profile retrieval
                        /                       \
                       /                         \
              active known layout          unknown / drift / difficult
                       |                         |
             PP-OCRv6 Medium +           PP-OCRv6 Medium +
             profile-guided graph        PaddleOCR-VL-1.6
                       |                         |
                       +------------+------------+
                                    |
                            unresolved crop only
                                    |
                           Vertex Gemini adjudication
                                    |
                                    v
                         grounded candidate fusion
                                    |
                                    v
                  role/type/structure/financial validation
                                    |
                        +-----------+-----------+
                        |                       |
                 accepted output          targeted review
                        |                       |
                        +-----------+-----------+
                                    v
                       versioned canonical ledger
                                    |
                     approval / export / learning / drift
```

### 5.1 Deployable boundaries

1. **API/domain service** — document, evidence, review, profile, approval, export, authorization, and audit contracts.
2. **Temporal workers** — document/page/table/crop orchestration, deterministic retry policy, timeouts, cancellation, and reuse of completed artifacts.
3. **Preprocessing service** — rendering, page-quality analysis, geometry normalization, image variants, and transform metadata.
4. **Fast OCR service** — persistent PP-OCRv6 Medium detection/recognition with GPU batching.
5. **Layout service** — persistent PP-DocLayoutV3 page/table region detection and page reading-order proposals.
6. **Heavy local parser** — dedicated PaddleOCR-VL-1.6 inference deployment.
7. **Extraction/fusion service** — spatial graph, profile constraints, provider candidate alignment, canonical rows, and provenance.
8. **Validation/risk service** — typed parsing, role classification, arithmetic/structure validation, confidence calibration, and routing.
9. **Review web application** — new Next.js UI against the V2 APIs.
10. **Evaluation tooling** — dataset registry, gold annotation, benchmark execution, calibration, reporting, and model/profile promotion.

Logical services may initially share deployment units, but the model processes and provider adapters remain resource-isolated.

### 5.2 Platform choices

- Python and FastAPI.
- Temporal for workflow orchestration.
- PostgreSQL for domain and audit state.
- S3-compatible encrypted object storage.
- Persistent GPU services using supported Paddle/FastDeploy or vLLM serving paths.
- Next.js for the new UI.
- Docker Compose for local development.
- Kubernetes for production.
- OpenTelemetry-compatible logs, metrics, and traces without bill content in ordinary telemetry.

---

## 6. State and lifecycle model

Do not encode all business state in one document status.

### 6.1 Workflow state

```text
uploaded → preprocessing → extracting → validating → completed
                                            └──────→ failed
any nonterminal state → cancelled
```

### 6.2 Completeness state

- `complete`
- `incomplete`
- `unreadable`

Any failed/missing nonblank page makes the document incomplete. Blank pages are complete when blank classification passes its calibrated threshold.

### 6.3 Review state

- `not_required`
- `pending`
- `in_review`
- `resolved`

### 6.4 Layout state

- `unknown`
- `candidate`
- `shadow`
- `active`
- `drifted`
- `archived`

### 6.5 Approval state

- `unapproved`
- `approved`
- `rejected`

### 6.6 Export state

- `blocked`
- `ready`
- `exported`

The API may expose a derived display status, but no component may use it instead of the authoritative axes.

---

## 7. Immutable ingestion and artifacts

### 7.1 Upload

- Accept PDF with client idempotency key and optional trusted external hospital ID.
- Validate MIME/content, PDF structure, encryption, page count, size, and malware/content policy.
- Compute SHA-256 before storage.
- Store the original immutably and never rotate, rewrite, compress, or annotate it in place.
- Deduplicate artifacts by tenant, content hash, stage, and configuration hash without cross-tenant data disclosure.
- Create the database record and Temporal workflow only after durable object/database persistence.

### 7.2 Artifact identity

Each artifact key includes:

- tenant;
- document content hash;
- page/table/crop identity;
- stage name;
- preprocessing/model/prompt/profile version;
- configuration hash.

Retries reuse successful matching artifacts. A downstream logic change does not force re-OCR; an OCR/preprocessing change invalidates only affected descendants.

### 7.3 Historical boundary

Import only:

- adjudicated gold annotations;
- benchmark manifests and named failure cases;
- annotation provenance and dataset splits;
- approved hospital identities/aliases needed for evaluation or future uploads.

Do not import:

- schemas or table bands;
- rule packs;
- layout fingerprints or profile activation;
- extracted rows as V2 production truth;
- legacy model confidence;
- current reconciliation-derived trust.

The old service remains read-only for historical users until its retention period ends.

---

## 8. Page quality and reversible geometry

### 8.1 Quality record

Record for every page:

- native-text presence and visual agreement;
- effective DPI and estimated character height;
- blur/motion blur;
- contrast and background uniformity;
- glare/shadow;
- clipping/cropping risk;
- ink fraction and blank probability;
- coarse orientation and confidence;
- continuous skew angle;
- perspective/keystone score;
- curvature/dewarp score;
- handwriting probability;
- applied image variants and quality changes.

These features participate in routing and calibration.

### 8.2 Normalization sequence

1. Render at 300 DPI.
2. Detect coarse orientation before full OCR.
3. Correct 0/90/180/270 orientation in a derivative.
4. Estimate fine skew from detected text-baseline angles and correct it.
5. Detect the physical page quadrilateral and apply homography only above a calibrated confidence threshold.
6. Apply UVDoc dewarping only for classified curved pages and only when downstream quality improves.
7. Generate restrained color, grayscale/CLAHE, and binarized variants only when quality routing requests them.
8. Rerender or crop at 400 DPI when character-size/OCR coverage is inadequate.
9. Detect tables on the normalized page.
10. Estimate and rectify table-local rotation independently.

### 8.3 Coordinate contract

- Preserve polygons, not only axis-aligned boxes.
- Store a reversible 3×3 transform chain for original PDF → normalized page → normalized table crop.
- Store geometry in original and active coordinate spaces.
- Require automated round trips to return overlays within two rendered pixels.
- Show both original and normalized views in review.

Geometry-changing preprocessing is rejected for a page if the transform is noninvertible, overlay tests fail, or measured OCR/structure quality materially degrades.

---

## 9. Page and table understanding

### 9.1 Page classification

Classify every page independently as one or more of:

- hospital itemized charges;
- pharmacy/medicine;
- laboratory;
- receipt/payment;
- category/summary;
- narrative/discharge;
- handwritten clinical/ICU/OT;
- metadata/cover;
- blank;
- unreadable;
- mixed.

Narrative, cover, and handwritten pages do not enter automatic financial extraction unless a validated table region is present.

### 9.2 Table detection

- Detect zero, one, or many tables per page.
- Support borderless, partially bordered, bordered, merged-header, and continuation tables.
- Retain table polygons, reading order, header candidates, row/column/cell proposals, and confidence.
- Run table detection for every relevant page; do not limit it by hospital confidence or page number.
- Use PP-DocLayoutV3 as the initial detector.
- Benchmark PP-StructureV3 as the challenger.
- Fine-tune or replace the detector only if frozen hospital-bill table-region recall is below 98%.

### 9.3 Page/table layout retrieval

Combine:

- trusted external hospital ID or confirmed alias;
- page visual embedding;
- OCR header tokens;
- stable visual anchors and relative positions;
- page dimensions/aspect ratio;
- table count and region signatures;
- column semantic candidates;
- neighboring page-type sequence.

Select profiles per page/table, not per document. Require both an absolute match threshold and a top-candidate margin. Ambiguous or new layouts take the heavy route.

---

## 10. Inference routes

### 10.1 Fast OCR

Use [PP-OCRv6 Medium](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) as the initial self-hosted OCR engine.

Retain for every token:

- raw text;
- polygon and baseline angle;
- detection and recognition confidence;
- OCR model/version;
- preprocessing variant;
- coordinate space;
- stable run-scoped token ID;
- alternative recognition where supported.

Native PDF text is an additional candidate source. It becomes authoritative only when coverage and visual agreement pass validation.

### 10.2 Known active-layout route

1. Normalize page/table geometry.
2. Run PP-OCRv6 Medium.
3. Align current anchors to the active profile.
4. Apply profile constraints to the global spatial graph.
5. Construct rows and semantic fields from current evidence.
6. Run all role, numeric, completeness, grounding, and calibrated-confidence validators.
7. Escalate only failed pages/tables/cells.

Profile regions and column positions are constraints with tolerances, never the only row/cell assignment method.

### 10.3 Heavy local route

Use [PaddleOCR-VL-1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html) as the initial self-hosted document parser for:

- unknown layouts;
- ambiguous profile matches;
- drift;
- poor-quality pages;
- complex multi-table pages;
- failures of the known route;
- unresolved table structure or reading order.

PaddleOCR-VL proposes layout, table, row, cell, and reading-order structure. Its text/values are aligned to the OCR token lattice or retained as a separate grounded candidate; they are not accepted solely from generated markup.

Deploy the layout and VLM components in compatible isolated environments and behind persistent bounded inference services.

### 10.4 Vertex Gemini adjudication

Use Vertex Gemini only after local routes remain unresolved.

- Use the approved region and GA models only.
- Use the current GA Flash model first; use GA Pro only if Flash fails the contract or confidence gate.
- Pin model IDs and prompt/schema versions.
- Send normalized table crops and nearby headers rather than complete PDFs.
- Include the canonical field contract and OCR token IDs/polygons.
- Require schema-constrained JSON.
- Require token IDs and a reviewable polygon for every returned cell.
- Reject unsupported or out-of-region values from automatic output.
- Use deterministic provider settings where supported.
- Disable Search grounding and unrelated tools.
- Cache by tenant, crop hash, model ID, prompt version, and response-schema hash.
- Apply per-tenant budgets and circuit breakers; exhausted budgets route to local review, never truncated extraction.

Follow [Vertex zero-retention guidance](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/vertex-ai-zero-data-retention): obtain applicable abuse-monitoring exceptions, avoid retained grounding/session features, and disable project caching when policy requires it.

---

## 11. Evidence graph and row reconstruction

### 11.1 Graph representation

Nodes represent tokens, lines, table elements, cells, rows, and headers. Edges encode:

- baseline and vertical overlap;
- horizontal/vertical distance;
- polygon overlap;
- reading order;
- table membership;
- header-to-column affinity;
- repeated column patterns;
- provider agreement;
- profile anchor/column constraints;
- cross-page continuation evidence.

The first release uses deterministic graph construction plus provider structure proposals and calibrated scoring. Do not build a custom GNN in the initial release. A learned graph ranker is considered only after the production corpus demonstrates a repeatable residual gap.

### 11.2 Row construction

- Form row candidates only inside a table or explicitly reviewed unmatched region.
- Join wrapped descriptions using geometry, indentation, vertical gaps, and neighboring numeric occupancy.
- Preserve batch/expiry lines as descriptions or dedicated auxiliary cells; never delete their host row.
- Preserve negative returns and credits.
- Preserve identical legitimate rows at distinct evidence positions.
- Identify first/last rows near table boundaries.
- Emit sections, continuations, and totals explicitly.
- Retain alternative hypotheses for ambiguous cell/row assignment.
- A known-route zero-yield or implausibly-low-yield result automatically escalates.

### 11.3 Canonical row roles

Every row receives exactly one role:

- `detail`
- `continuation`
- `section_header`
- `category_rollup`
- `section_total`
- `document_total`
- `payment`
- `deposit`
- `refund`
- `metadata`
- `footer_noise`
- `unresolved`

Use a global versioned multi-class classifier trained on annotated rows, plus hard safety constraints for evidence and totals. No hospital-specific classifier code is permitted.

### 11.4 Cross-page continuity

- Detect repeated headers positionally and semantically.
- Carry column semantics only when layout and table-continuation confidence pass.
- Merge a row across pages only when the prior page ends with incomplete row evidence and the next page begins with compatible continuation evidence.
- Preserve original page/table/row position for review and duplicate decisions.

---

## 12. Candidate contract and fusion

### 12.1 Provider candidate

Every provider candidate contains:

- candidate ID;
- page/table and proposed row order;
- proposed role;
- canonical/raw fields;
- supporting token IDs and polygons;
- provider/model/prompt/profile version;
- raw and normalized values;
- provider confidence;
- structure/type/arithmetic validation features;
- grounding status.

### 12.2 Fusion

- Align candidates using token IDs, polygon overlap, normalized text, and row order.
- Use maximum-weight bipartite/assignment matching rather than greedy amount-only matching.
- Score fields independently using OCR confidence, provider agreement, profile fit, geometry, type validity, arithmetic consistency, and page/table context.
- Preserve every disagreement and rejected alternative in audit evidence.
- Calibrate route-specific confidence using isotonic calibration on the held-out calibration set.
- Do not delete low-confidence evidence-backed detail candidates; mark them review-pending.

### 12.3 Targeted recovery ladder

For an unresolved row/cell/table:

1. Rerender the local region at higher resolution.
2. Try the alternate approved photometric variant.
3. Rerun fast OCR on the crop.
4. Run PaddleOCR-VL on the crop/table.
5. Invoke Gemini on the crop if local routes remain ambiguous.
6. Route to review or recapture.

Do not reprocess the entire bill for a local failure.

---

## 13. Typed fields and financial validation

### 13.1 Parsing order

1. Establish page and table context.
2. Establish row membership.
3. Assign column semantics.
4. Parse typed values.
5. Apply financial and document validation.

This prevents dates, phones, registration IDs, batch values, expiry dates, and page numbers from competing with amounts before semantic context exists.

### 13.2 Typed grammar

- `money`: locale-aware currency, Indian grouping, parentheses, explicit negative/CR/refund notation; reject identifier/date/phone shapes and impossible magnitudes.
- `quantity`: bounded numeric/unit form; cannot become money without explicit source-column evidence.
- `date`: canonical date with original text retained.
- `code`: alphanumeric identifier without numeric coercion.
- `batch` and `expiry`: first-class auxiliary types.
- `text`: preserved verbatim with approved whitespace normalization.

Store model/OCR values as strings until validated. Persist raw and normalized representations. Quarantine invalid/oversized values instead of retrying or coercing to zero.

### 13.3 Validators

Where supported by source fields, compute:

- quantity × unit price;
- gross, discount, tax, and net relationships;
- section subtotal against detail rows;
- printed gross/discount/net against accepted detail totals;
- return/refund sign semantics;
- duplicate-evidence versus repeated-row evidence;
- invalid identifier patterns in financial columns;
- magnitude and precision bounds.

Reconciliation and residuals may lower confidence, target a crop, or explain review. They may not invent, alter, delete, or auto-reclassify a row solely to force a total.

---

## 14. Layout profiles and cost control

### 14.1 Profile contents

- hospital and global layout-family association;
- page/table type and signatures;
- stable visual/OCR anchors;
- anchor-relative regions and tolerances;
- header variants and canonical field mapping;
- expected column order and relationships;
- continuation and section constraints;
- supported-field contract;
- validated capture-quality range;
- calibration artifact;
- construction/validation dataset IDs;
- measured metrics;
- lifecycle, drift, and rollback history.

Profiles contain data and declarative constraints only. No executable hospital-specific repairs are allowed.

### 14.2 Lifecycle

```text
discovered → candidate → shadow → active
active → drifted → shadow | archived
```

### 14.3 Activation

Require all of:

- at least 10 independent holdout bills;
- at least 200 populated gold detail rows;
- different documents for profile construction and validation;
- more than one capture condition when the layout receives mixed inputs;
- at least 95% row precision, recall, and F1 on profile holdout;
- 95% confidence-interval lower bounds above 85%;
- amount accuracy at least 95%;
- supported populated-field gates passed;
- no incomplete pages, ungrounded accepted values, identity ambiguity, negative-row miss, or unsafe numeric failure.

Low-volume layouts remain on the heavy route. They are usable without activation.

### 14.4 Drift

Monitor:

- embedding/profile distance;
- header/anchor changes;
- table/column-count changes;
- route escalation;
- validator/reconciliation failures;
- local heavy-route disagreement on samples;
- correction rate;
- calibration and capture-quality shift.

Sample 5% of active traffic through the heavy local route for drift measurement. Do not call Gemini solely for drift sampling. Pause only the affected page/table profile, not all layouts for the hospital.

---

## 15. Persistence model

### 15.1 Core entities

#### Document

Tenant, immutable source, hash, hospital identity, independent state axes, totals, approval/export state, and current output version.

#### PageAsset

Original render, derivatives, quality metrics, page type, artifact hashes, and workflow activity state.

#### TransformChain

Ordered original/page/table transforms, inverse matrices, coordinate spaces, quality verdict, and version.

#### OcrRun and OcrToken

Engine/config/preprocessing identity, tokens, alternatives, polygons, confidence, and evidence IDs.

#### TableRegion

Page polygon, crop artifact, orientation, table type, structure proposals, layout signature, and profile match.

#### ExtractionCandidate

Provider-specific proposed row/cells, grounding, validation features, and versions.

#### CanonicalRow

Page/table/order, role, section, raw and normalized fields, confidence, validation, review disposition, and evidence.

#### LayoutProfile

Declarative constraints, lifecycle, metrics, calibration, datasets, drift, activation, and rollback.

#### GoldAnnotation

Immutable document/page/table/row/cell truth, capture labels, annotators, adjudication, and split.

#### ReviewEvent

Original value/structure, correction, reviewer, reason, evidence, and model/profile/output versions.

#### ExtractionRun

Temporal workflow ID, configuration, activities, usage/cost, errors, artifacts, and output version.

#### ExportRecord

Format, scope, authorization, source output version, blockers, object reference, and audit identity.

### 15.2 Canonical fields

Preserve:

- section;
- description;
- service code;
- request number;
- HSN code;
- service date;
- batch and expiry where available;
- quantity and unit;
- unit price/rate;
- gross amount;
- discount;
- tax;
- net amount;
- patient/payer splits where present;
- raw auxiliary columns.

Unknown columns remain auxiliary evidence rather than being discarded.

---

## 16. Temporal workflow and reliability

### 16.1 Workflow shape

- One workflow per document.
- Child activities per page preprocessing/OCR/layout.
- Child activities per table/crop heavy parsing and adjudication.
- Deterministic assembly/validation activity.
- Review wait states are durable signals, not blocked worker processes.

### 16.2 Idempotency

Activity identity includes tenant, document hash, page/table/crop, stage, and configuration hash. Retrying the same activity returns/reuses the existing successful artifact.

### 16.3 Error policy

- Transient provider/network/GPU errors use bounded exponential retry.
- Deterministic malformed input, unsafe numeric, unsupported PDF, and validation failures do not retry as transient errors.
- A timeout produces a durable actionable activity/document state.
- Cancellation terminates pending work and preserves completed artifacts.
- Worker restart resumes from Temporal history.
- Approval/export checks PostgreSQL domain state derived from completed workflow events.

### 16.4 Model-service isolation

- One model family per bounded service/deployment.
- Persistent loaded models with batching.
- GPU memory ceilings, request timeouts, concurrency limits, circuit breakers, health checks, and worker recycling.
- Backpressure based on queue age and GPU capacity.
- No heavyweight model inside the API process or generic workflow worker.

---

## 17. Public APIs

The new UI consumes clean `/v2` contracts; there is no legacy compatibility adapter.

### 17.1 Documents

- `POST /v2/documents` — idempotent upload with optional external hospital ID.
- `GET /v2/documents` — paginated/filterable list.
- `GET /v2/documents/{id}` — state axes, completeness, route, totals, layout/profile, workflow/output version, and blockers.
- `POST /v2/documents/{id}/retry` — retry invalid/failed activities only.
- `POST /v2/documents/{id}/cancel` — cancel workflow.
- `POST /v2/documents/{id}/approve` — guarded approval.
- `POST /v2/documents/{id}/export` — guarded CSV/XLSX/JSON export.

### 17.2 Pages, evidence, and rows

- `GET /v2/documents/{id}/pages`
- `GET /v2/documents/{id}/pages/{page}/evidence`
- `GET /v2/documents/{id}/items`
- `POST /v2/documents/{id}/items` — add evidence-linked missing row.
- `PATCH /v2/items/{id}` — field, role, section, or evidence correction.
- `POST /v2/items/{id}/accept`
- `POST /v2/items/{id}/reject`
- `POST /v2/items/{id}/mark-unreadable`
- Structural review operations for split, merge, and token/cell relink.

### 17.3 Profiles

- List/inspect profiles and versions.
- Inspect discovery candidates and nearest matches.
- View construction, validation, shadow, and drift metrics.
- Promote, pause, archive, and rollback.
- Promotion is rejected server-side unless the complete activation gate passes.

### 17.4 Runs and diagnostics

- Retrieve workflow/activity state, model/profile/config versions, provider calls, measured cost, timing, failures, validation, and output lineage.
- APIs never expose secrets or unrestricted PHI in logs/diagnostic summaries.

---

## 18. Review application

The V2 Next.js application provides:

- upload and processing progress;
- original PDF and normalized page/table views;
- polygon overlays mapped through inverse transforms;
- page/table/row/cell selection;
- accepted, pending, rejected, summary/payment, and unreadable views;
- add, edit, delete, split, merge, role change, section change, and evidence relink;
- raw versus normalized numeric values;
- provider disagreement and grounding status;
- problem grouping: missing row, extra row, structure, layout, numeric, reconciliation, page failure, unreadable;
- recapture request;
- bulk table/page acceptance when eligibility passes;
- profile review/activation administration;
- guarded approval/export;
- keyboard-first high-volume operation.

Every correction retains the original candidate, evidence, reviewer, reason, and versions. Review does not retroactively improve the reported machine metric.

---

## 19. Dataset and evaluation program

### 19.1 Development corpus

Before full platform investment:

- at least 300 fully adjudicated bills;
- at least 20,000 detail rows;
- at least 50 layout families;
- representative ordinary scans, mobile, Xerox, skew/perspective, rotation, long bills, returns, multi-table pages, and mixed documents.

This corpus supports the early accuracy-spine go/no-go.

### 19.2 Production release corpus

Before broad production readiness:

- at least 1,000 bills;
- at least 100,000 detail rows;
- at least 250 hospitals;
- at least 500 layout variants;
- natural capture-quality diversity and rare financial cases.

Double-annotate at least 15% and adjudicate disagreements. Report inter-annotator agreement.

### 19.3 Splits

- construction/training;
- confidence calibration;
- frozen release test;
- hospital-disjoint unseen-layout test;
- future-time known-layout/drift test.

No patient, bill, near duplicate, page, or synthetic transform of a page crosses related splits. Frozen test data never trains a model or creates a profile.

### 19.4 Gold content

- hospital/layout identifiers;
- capture and quality labels;
- page type;
- table type and polygon;
- row polygon, order, role, and section;
- cell polygons and canonical/raw values;
- printed totals;
- negative/repeated/edge labels;
- unreadable and unsupported labels;
- annotator/adjudication/split provenance.

---

## 20. Test requirements

### 20.1 Geometry

- 0/90/180/270 orientation.
- Positive/negative fine skew.
- Perspective/keystone distortion.
- Curved pages/dewarp.
- Independent table rotation.
- Blur, motion blur, faint Xerox, glare, shadow, clipping, compression, and low resolution.
- 300→400 DPI escalation.
- Transform round trips within two pixels.
- Original source immutability.

### 20.2 Tables and rows

- Multiple borderless tables on one page.
- Different schemas across consecutive pages.
- Repeated headers and cross-page continuation.
- Wrapped descriptions and continuation rows.
- First/last rows near boundaries.
- Negative returns/refunds.
- Identical legitimate repeated rows.
- Category, section, document-total, payment, deposit, refund, metadata, and footer roles.
- Narrative and handwritten pages.
- Zero-yield and implausibly-low-yield escalation.

### 20.3 Numeric safety

- Batch and expiry strings.
- Phone and registration numbers.
- Dates/timestamps.
- HSN/service/request codes.
- Parenthesized and CR negatives.
- Indian grouping and OCR substitutions.
- Extreme magnitudes and precision boundaries.
- Invalid staging values never reaching financial decimals.

### 20.4 Provider and workflow failure

- OCR/layout/VLM timeout or crash.
- GPU exhaustion/backpressure.
- Temporal activity retry and worker restart.
- Vertex timeout, rate limit, and budget exhaustion.
- Invalid/partial schema-constrained response.
- Valid JSON with ungrounded or out-of-crop value.
- Retry without duplicate rows/runs.
- Reuse of successful page/table artifacts.
- Model/profile/config invalidation.
- Cancellation and terminal-state consistency.
- Approval/export blocked for incomplete evidence.

### 20.5 Identity and profiles

- Known hospital/known layout.
- Known hospital/new layout.
- Unknown hospital.
- Headerless page with one candidate.
- Headerless page with multiple candidates.
- Alias conflict and concurrent identity creation.
- Same hospital with different page/table variants.
- Drift, automatic pause, shadow return, and rollback.
- Failed extraction leaving no untrusted inferred identity.

### 20.6 Security and operations

- Authentication and RBAC.
- Tenant isolation across database, object, cache, workflow, model, and export layers.
- Retention/deletion/legal hold.
- Audit completeness.
- No PHI in ordinary telemetry.
- Long/concurrent bills under bounded GPU memory.
- Queue recovery after deployment restart.
- Cost/latency telemetry completeness.

---

## 21. Failure taxonomy and mitigation

| Observed failure | V2 mitigation | Required proving test |
|---|---|---|
| Batch/ID/phone/date parsed as amount | Semantic column assignment before typed money parsing | Numeric confusion cohort |
| Metadata/footer numbers become rows | Page/table isolation and explicit row roles | Metadata/footer negative tests |
| Summary/payment rows counted as detail | Multi-class row roles; only accepted detail contributes | Summary/payment cohort |
| Handwritten garbage contaminates totals | Handwriting/unreadable routing and validated table-only exception | Handwritten-page tests |
| Repeated page headers become rows | Positional/semantic repeated-header detection | Cross-page header tests |
| Wrong/generic schema wipes out rows | Page/table profile routing, heavy fallback, zero-yield escalation | Known-new-layout tests |
| Lab/pharmacy/receipt bundle pages disappear | Per-page type and per-table extraction | Mixed-document cohort |
| Negative returns disappear | First-class negative/refund semantics and 100% gate | Return cohort |
| Batch/expiry continuation deletes host row | Continuations attach; host row retained | Pharmacy wrap tests |
| Legitimate repeats are deduplicated | Evidence-position identity, not text/amount alone | Repeated-row cohort |
| Amount taken from quantity/rate/other column | Graph column semantics, typed fields, provider fusion | Column-swap tests |
| Summary category treated only as a heading | Explicit category-rollup role and source structure | Summary-page tests |
| Tilt splits/merges physical rows | Reversible orientation/skew/perspective/table rectification | Geometry robustness set |
| Exact header fingerprint misses same layout | Multimodal profile retrieval and calibrated margin | Noisy-header tests |
| Two-seed overfitting | 10-bill/200-row independent activation gate | Profile promotion tests |
| Numeric overflow fails persistence | String staging, magnitude/precision quarantine | DB boundary tests |
| Heavy model OOM | Persistent isolated bounded model services | Concurrency/memory canary |
| Whole-document retry after local failure | Temporal page/table/crop activities and immutable artifacts | Activity retry tests |
| Good totals hide wrong rows | Reconciliation is soft validation only | Compensating-error tests |

---

## 22. Security, PHI, and tenancy

- Authenticate every user and service.
- Enforce tenant authorization at every API and object reference.
- Use tenant-aware database policies and object prefixes/keys.
- Encrypt in transit and at rest.
- Use short-lived service identities and centralized secret rotation.
- Audit document, evidence, review, profile, approval, and export access.
- Exclude raw bill content, OCR text, crops, prompts, and responses from ordinary logs.
- Define retention, deletion, backups, and legal hold for originals and derivatives.
- Separate production, annotation, calibration, and frozen test datasets.
- Require explicit governance approval for production corrections entering training.
- Use Vertex AI in an approved region.
- Request applicable abuse-monitoring exceptions and configure required zero-retention controls.
- Disable Search/Maps grounding, session resumption, and unrelated tools.
- Disable project caching where required by policy.
- Make the Gemini route tenant-configurable; prohibited tenants use local heavy parsing plus review.
- Restrict exports by role, tenant, purpose, output version, and audit event.

External-provider eligibility is a production gate, not an assumption that overrides policy.

---

## 23. Observability and cost

Record by workflow, page, table, and provider:

- queue wait and execution time;
- retry/cancellation/error class;
- CPU/GPU utilization, memory, concurrency, and batching;
- quality scores and transforms;
- OCR tokens, confidence, and unassigned evidence;
- table/row/cell counts and structure alternatives;
- profile candidates, selected profile, distance, and margin;
- OOD and drift score;
- provider disagreement and grounding failures;
- typed parse rejections;
- arithmetic/reconciliation residuals and basis;
- route/escalation reason;
- Vertex calls, model, tokens, latency, and measured cost;
- review reason/actions/minutes;
- export attempts and blockers;
- eventual accuracy when adjudicated truth becomes available.

Define dashboards for:

- page/document completeness;
- p50/p95 latency by page count and route;
- cost by route and tenant;
- known-route zero-Gemini percentage;
- review and recapture rate;
- accuracy/drift by layout and capture cohort;
- workflow age and stuck activities;
- model/profile version comparison.

Use tenant budgets, provider kill switches, champion/challenger deployment, versioned rollback, and profile canaries.

---

## 24. Delivery phases and gates

### Phase 0 — Corpus, metrics, and contracts

Deliver:

- frozen legacy baseline;
- `legacy_metric_v1` and `canonical_metric_v2`;
- adjudicated development corpus;
- evidence/candidate/canonical-row contracts;
- provider/security eligibility;
- named failure-regression set.

Exit: reproducible metrics, leak-free splits, annotation agreement, and no unresolved evaluator ambiguity.

### Phase 1 — Geometry and component bake-offs

Deliver:

- offline immutable page rendering;
- quality and transform chain;
- PP-OCRv6 benchmark;
- PP-DocLayoutV3 versus PP-StructureV3 table benchmark;
- PaddleOCR-VL-1.6 capacity/accuracy benchmark;
- overlay round-trip validation.

Exit: table-region recall target, bounded memory on long-document canary, and accurate evidence mapping.

### Phase 2 — Minimum heavy accuracy spine

Deliver only enough offline pipeline for:

- page/table classification;
- PP-OCRv6 token evidence;
- PaddleOCR-VL structure candidates;
- deterministic spatial graph;
- typed fields and roles;
- candidate fusion;
- row/field evaluation.

**Early go/no-go:** the unseen-layout development holdout must reach at least 85% row precision and recall before the full service, UI, and profile platform are built. If it fails, stop and improve/replace inference or representation rather than operationalizing a substandard engine.

### Phase 3 — Targeted Gemini and known profiles

Deliver:

- grounded Gemini crop adjudication;
- targeted recovery ladder;
- profile retrieval and declarative constraints;
- lifecycle, shadow, activation, drift, and rollback;
- known-layout fast route.

Exit: aggregate 90% target on development/validation, unknown cohort above floor, known-route 95% target, and 95% ordinary active traffic with zero Gemini.

### Phase 4 — End-to-end platform

Deliver:

- Temporal workflows;
- PostgreSQL and object storage;
- clean `/v2` APIs;
- new Next.js review/profile application;
- security/tenancy/audit;
- guarded approval/export;
- observability and provider budgets.

Exit: complete upload-to-export acceptance, failure recovery, and security suites.

### Phase 5 — Production benchmark and historical replay

Deliver:

- 1,000-bill/250-hospital/500-variant release corpus;
- frozen release report;
- historical input replay and current-system comparison;
- load/capacity/cost reports;
- operations, incident, rollback, and retention runbooks.

Exit: every accuracy, field, confidence, completeness, security, workflow, latency/capacity, and cost-observability gate passes.

There is no live traffic, so no live dual-write or compatibility shadow is required. Historical replay is non-authoritative and must not import legacy schemas.

### Phase 6 — Hard cutover

- Deploy the new API and UI as the only system for new uploads.
- Leave the old service read-only for historical access.
- Import only curated gold/evaluation and approved identity reference data.
- Monitor new traffic through V2 metrics and profile canaries.
- Keep model/profile rollback and artifact-based reprocessing.
- Retire the old service only after the historical retention decision is executed.

---

## 25. Go/no-go checklist

Release only when all are true:

- [ ] Both evaluators pass row gates.
- [ ] Aggregate precision, recall, and F1 are at least 90%.
- [ ] Confidence lower bounds and macro cohorts stay above 85%.
- [ ] Known and unseen-layout cohorts pass independently.
- [ ] Mobile, Xerox, skew/perspective, rotated, and ordinary cohorts pass.
- [ ] Supported populated-field gates pass.
- [ ] Negative/return-row recall is 100% on the release set.
- [ ] Every page is accounted for.
- [ ] Incomplete documents cannot approve/export.
- [ ] Every automatically accepted field is grounded.
- [ ] Invalid numerics cannot fail persistence.
- [ ] Active-layout traffic reaches the 95% row target.
- [ ] At least 95% of ordinary active-layout bills make zero Gemini calls.
- [ ] Profile activation, drift pause, and rollback are proven.
- [ ] Temporal retry/resume/cancel is idempotent.
- [ ] Model services remain within bounded memory under load.
- [ ] Authentication, tenant isolation, PHI, retention, audit, and export authorization pass.
- [ ] Vertex regional and retention controls are approved and configured.
- [ ] Historical replay passes without importing legacy schemas.
- [ ] Incident and rollback runbooks are accepted.

Good reconciliation, corrected human output, or selected successful hospitals cannot substitute for a failed machine-output gate.

---

## 26. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Public model scores do not translate to hospital bills | Frozen end-to-end row/field benchmark before platform investment |
| Gold labels are inconsistent | Double annotation, adjudication, agreement reporting, immutable splits |
| Geometry improves OCR but breaks review evidence | Reversible transform chain and two-pixel round-trip gate |
| Heavy local VLM misses dense rows | Crop/window routing, OCR token coverage, targeted Gemini, visible uncertainty |
| Gemini hallucinates plausible values | Token/polygon grounding; unsupported output cannot auto-accept |
| Wrong active profile produces confident rows | Page/table retrieval margin, OOD, validators, drift sampling, fast escalation |
| Arithmetic forces a false ledger | Soft validation only; no auto row invention/removal |
| Review volume grows with hospital count | Global heavy route, targeted review, reusable global layout clusters, active learning |
| GPU/model service OOM | Persistent isolated deployments, bounded concurrency/memory, canary/load tests |
| Provider outage, model retirement, or price change | Provider-neutral contract, pinned versions, local route, kill switch, promotion benchmark |
| Sparse layouts cannot meet activation sample size | Keep them on heavy route; no unsafe shortcut |
| Legacy data contaminates new evaluation | Curated gold-only import; no schema/profile/output trust import |
| Aggregate metrics hide weak groups | Macro/cohort/confidence-bound gates |

---

## 27. Reasons for selecting this architecture

### 27.1 Representation is the primary gap

The current service can recognize correct text while assigning it to the wrong row/column because it discards polygon orientation and relies on y-center grouping and x bands. Reversible geometry, polygons, table graphs, and delayed semantic assignment address the cause rather than only changing OCR.

### 27.2 Page/table routing matches the document population

One PDF is not one schema. Summary, detail, pharmacy, receipt, and narrative pages need independent classification and extraction, and one page may contain multiple table schemas.

### 27.3 A global heavy route makes the first unknown bill useful

The system cannot require an engineering project or two-seed lifecycle before an unseen hospital yields rows. PaddleOCR-VL plus grounded fusion provides a general first-encounter route; profiles optimize repeated layouts later.

### 27.4 Known profiles preserve the requested cost advantage

Repeated layouts should not pay for a VLM or external provider on every bill. Declarative anchor-relative profiles reduce steady-state cost without becoming executable hospital parsers.

### 27.5 A cascade allocates cost according to uncertainty

Fast OCR is suitable for stable layouts, a document VLM for unfamiliar structure, Gemini for unresolved crops, and humans for irrecoverable evidence. Always running everything is wasteful; always using the cheapest route is inaccurate.

### 27.6 Provider-neutral evidence prevents lock-in

Models and provider versions change. Stable evidence/candidate/canonical contracts let components be benchmarked, upgraded, disabled, or replaced without rewriting review and domain logic.

### 27.7 Evidence grounding is essential for financial safety

Dates, phones, codes, batches, and expiry values can look numeric. Every accepted field must point to source tokens/polygons so a plausible generated value cannot silently enter the ledger.

### 27.8 Reconciliation is valuable but insufficient

A wrong set of rows can still match totals. Arithmetic therefore informs confidence and targeted recovery but never defines row truth.

### 27.9 Statistical activation is safer than machine agreement

Two or three bills do not cover capture variation, optional fields, returns, continuations, or drift. Independent holdout bills, populated-row volume, field gates, and confidence bounds are required.

### 27.10 Temporal fits the actual workload

The pipeline is a durable multi-stage page/table workflow with retries, long-running inference, human signals, cancellation, and partial reuse. Temporal replaces custom task/reaper/state reconciliation with explicit workflow history.

### 27.11 Persistent model services address current latency/OOM patterns

Loading heavyweight models per page is expensive, while putting all models in a generic worker risks uncontrolled memory. Dedicated persistent services retain batching and fault containment.

### 27.12 A separate hard-replacement repository protects both systems

V2 requires immutable artifacts, transforms, candidates, polygons, profile statistics, and independent state axes that do not fit the legacy schema cleanly. A separate system enables clean testing and cutover without importing disproven schema assumptions.

---

## 28. Assumptions and locked decisions

- Scale is approximately 5,000 hospitals and 15,000–20,000 variants.
- Accuracy is measured before human review.
- The engineering target is 90% aggregate precision/recall with an 85% hard floor.
- Known-layout automatic routing targets 95% precision/recall.
- Hybrid self-hosted inference plus selective Vertex Gemini is permitted.
- Temporal, PostgreSQL, S3-compatible storage, FastAPI, Next.js, and Kubernetes are selected.
- PP-OCRv6 Medium, PP-DocLayoutV3, and PaddleOCR-VL-1.6 are the initial local models and remain subject to frozen-corpus promotion.
- Vertex uses pinned GA Flash first and GA Pro only for unresolved cases.
- No preview production models.
- The new system uses a separate database and clean `/v2` API.
- There is no legacy compatibility adapter.
- No legacy schemas, profiles, rule packs, or activation state are imported.
- Curated gold/evaluation and approved identity reference data may be imported.
- There is no current live traffic, so historical replay replaces live shadowing.
- The first production-capable release includes upload, extraction, evidence review, profile administration, guarded approval/export, security, audit, and observability.
- The old system remains read-only until historical retention is resolved.

Any change to a locked decision requires an architecture decision record, impact analysis, and an updated frozen-benchmark plan before implementation.
