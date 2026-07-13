# GMoney V2 Phased Implementation Plan

## Summary

Build GMoney V2 as a clean replacement without modifying the legacy application. The initial delivery is a functional, reviewed V2 validated against the available corpus. External 1,000-bill certification and production cutover remain later gates.

Locked decisions:

- FastAPI, PostgreSQL, Temporal, S3-compatible storage, Next.js, and Docker Compose.
- CPU-first inference on `Standard_D16ds_v6`.
- Three documents may process concurrently.
- PP-OCRv6 Medium for OCR.
- PP-DocLayoutV3 and PP-StructureV3/SLANeXt Wireless evaluated for table regions and borderless structures.
- PaddleOCR-VL-1.6 through GGUF/llama.cpp for difficult tables.
- Selective Gemini AI Studio adjudication using the stable `gemini-3.5-flash` model and structured output.
- No authentication in the initial functional delivery; fixed single-tenant development mode.
- No legacy schemas, rules, profiles, or activation state imported.
- Existing curated gold annotations and bill inputs may be imported through a controlled corpus loader.
- Each passing phase receives a commit, immutable Git tag, and review report. Work continues automatically unless a gate fails or the architecture must materially change.

## Architecture and Interfaces

### Repository and deployments

Use a monorepo containing:

- FastAPI domain/API application.
- Next.js clinical-operations review application.
- Temporal workflows and workers.
- Separately deployable preprocessing, OCR, layout, heavy-parser, extraction, and validation services.
- Shared versioned Python/TypeScript contracts.
- Evaluation CLI and benchmark reports.
- Docker Compose definitions for the control-plane host and D16ds inference worker.
- Synthetic fixtures in Git; real bills, rendered pages, model weights, and generated artifacts outside Git.

The control-plane Compose deployment runs API, UI, PostgreSQL, MinIO, Temporal, and ordinary workers. The D16ds deployment runs persistent model services and CPU inference workers.

### Processing flow

1. Persist the immutable PDF and document record.
2. Inventory and render every page.
3. Measure orientation, skew, perspective, blur, resolution, clipping, and contrast.
4. Produce reversible normalized page variants with forward and inverse transforms.
5. Detect table regions and classify each page/table independently.
6. Run PP-OCRv6 and retain token text, confidence, and polygons.
7. Route difficult or unknown tables through PaddleOCR-VL.
8. Construct deterministic spatial graphs and candidate rows.
9. Parse typed fields and classify row roles.
10. Apply structural, arithmetic, completeness, and evidence-grounding validation.
11. Send only unresolved table crops to Gemini.
12. Fuse candidates into a versioned canonical ledger.
13. Route uncertain output to review; allow approval and export only when completeness gates pass.

Three documents remain active concurrently. The heavy-parser service begins with one inference slot. Phase 1 attempts two and three simultaneous heavy calls; three are enabled only if the D16ds completes the canary without OOM, swapping, or materially worse total throughput. Otherwise, three documents still progress concurrently while heavy calls are admission-controlled.

### Core contracts

Persist versioned `Document`, `PageAsset`, `TransformChain`, `PageQuality`, `OcrRun`, `OcrToken`, `TableRegion`, `ExtractionCandidate`, `ProviderCandidate`, `CanonicalRow`, `LayoutProfile`, `ReviewEvent`, `ExtractionRun`, `ExportRecord`, and `GoldAnnotation` entities.

Coordinates use original-page pixel space as the authoritative evidence system. Every normalized derivative retains the complete invertible transform chain. Round trips must remain within two pixels.

Canonical detail fields include description, quantity, unit price/rate, gross amount, discount, net amount, service date, request number, service code, HSN code, section, and row role. Raw strings are retained; invalid numeric strings are quarantined and never coerced to zero.

### Public API and UI

Expose clean `/v2` endpoints for documents, pages, evidence, items, structural review, profiles, workflows, approval, diagnostics, and export. There is no legacy compatibility adapter.

The UI uses a dense clinical-operations design with a bill queue, side-by-side source/evidence review, polygon overlays, keyboard-first editing, add/delete/split/merge/relink operations, raw-versus-normalized values, targeted problem queues, and guarded approval/export.

## Phases and Checkpoints

### Phase 0 — Foundation, Corpus, and Contracts

Implement the repository, lockfiles, Compose infrastructure, CI, dataset registry, evaluators, and shared contracts.

Import only bill inputs, curated gold labels, and approved identity references. Do not import legacy schemas, output confidence, layout bands, rules, or profiles.

Freeze `legacy_metric_v1` for comparison with 47.53% precision and 62.52% recall and `canonical_metric_v2` for deterministic one-to-one row matching. Split data by bill, hospital, and variant.

Checkpoint:

- Tag: `checkpoint-phase-0`
- Report: `docs/reviews/phase-0.md`
- Gate: reproducible baseline, valid gold conversion, leak-free splits, deterministic metrics, and clean Compose startup.

### Phase 1 — Geometry and Model Bake-Off

Implement immutable rendering, page-quality measurement, reversible normalization, overlays, and isolated model adapters.

Benchmark PP-OCRv6 Medium CPU backends, PP-DocLayoutV3 versus PP-StructureV3, SLANeXt Wireless on borderless tables, and PaddleOCR-VL-1.6 GGUF/llama.cpp on D16ds. Exercise one, two, and three concurrent document canaries.

Checkpoint:

- Tag: `checkpoint-phase-1`
- Report: `docs/reviews/phase-1.md`
- Gate: two-pixel transform round trip, immutable sources, at least 95% table-region recall on annotated data, three active documents without failure, and documented safe heavy-parser concurrency.

### Phase 2 — Minimum Accuracy Spine

Implement page/table classification, OCR token evidence, heavy-parser candidates, spatial graph and row reconstruction, continuation handling, typed parsing, roles, fusion, validation, and untouched machine-output evaluation.

Checkpoint:

- Tag: `checkpoint-phase-2`
- Report: `docs/reviews/phase-2.md`
- Hard gate: unseen-layout holdout row precision and recall must each reach at least 85% under both evaluators.

If the gate fails, stop and report errors by OCR, table localization, row grouping, typing, role classification, and missing/extra rows. Do not operationalize a substandard engine.

### Phase 3 — Gemini Recovery and Known Profiles

Implement a provider-neutral adjudication adapter using the Google GenAI SDK and the specific stable model ID `gemini-3.5-flash`. Send unresolved table crops with OCR evidence, require structured output, reject ungrounded values, cache by crop/model/prompt/schema hash, and record latency, tokens, and cost. Inject the rotated API key only through an environment secret.

Implement known-layout profiles, shadow validation, activation, drift pause, and rollback.

Checkpoint:

- Tag: `checkpoint-phase-3`
- Report: `docs/reviews/phase-3.md`
- Gate: at least 90% aggregate row precision and recall, at least 85% on unseen layouts, and 95% precision/recall for any active profile.

### Phase 4 — End-to-End Product

Implement Temporal workflows, persistence, APIs, the review UI, approval/export, observability, and failure recovery. Use independent workflow, completeness, review, layout, approval, and export state axes. Activities must be idempotent and reuse completed artifacts.

The initial deployment uses fixed single-tenant development mode with authentication disabled and clearly identifies that mode as non-production.

Checkpoint:

- Tag: `checkpoint-phase-4`
- Report: `docs/reviews/phase-4.md`
- Gate: upload-to-export acceptance passes; retries, cancellation, worker restart, corrections, evidence overlays, and three concurrent bills work without duplicate rows or corrupt state.

### Phase 5A — Available-Corpus Certification

Replay every usable available bill through untouched machine output. Produce aggregate and cohort metrics, a legacy comparison, resource/latency/queue/Gemini-cost reports, image digests, deployment instructions, and operational runbooks.

Checkpoint:

- Tag: `checkpoint-phase-5a`
- Report: `docs/reviews/phase-5a.md`
- Gate: aggregate row precision and recall at least 90%, supported macro cohorts at least 85%, amount accuracy at least 95%, zero incomplete exports, and successful three-document load testing.

### Phase 5B and Phase 6 — External Certification and Cutover

These are documented but not complete in the initial implementation. Phase 5B requires the external 1,000-bill, 250-hospital, approximately 500-variant corpus and production authentication/tenant requirements. Phase 6 makes V2 the only destination for new uploads, leaves the legacy application read-only, and imports no legacy schemas.

## Testing and Review Policy

Test orientation, skew, perspective, blur, Xerox fading, clipping, shadow, glare, low resolution, multiple borderless tables, changing page schemas, repeated headers, wrapped descriptions, cross-page rows, negative/refund rows, legitimate duplicates, invalid numeric types, inference/provider failures, retries, cancellation, evidence grounding, profiles, drift, and three concurrent documents.

Each checkpoint report records the Git SHA and image digests, commands and environment, test results, accuracy by cohort and confidence intervals, latency, memory, concurrency, provider use and cost, regressions, unresolved risks, and the gate decision.

Passing phases continue automatically. A failed accuracy/resource gate or required architecture deviation stops implementation and produces a review checkpoint instead of weakening the target.
