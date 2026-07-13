# Phase 2 Completion and D16 Demo Deployment Plan

## Goal

Finish the frozen Phase 2 accuracy gate, then deploy a live upload/results/evidence demo on
`20.57.131.189` without changing the existing application on port 3000.

## Accuracy gate

1. Resume the larger validation bill and make only general parser corrections.
2. Regress the already-passing 9-row training and 7-row validation bills after any correction.
3. Freeze the extraction commit before evaluating the sealed 409-row unseen-layout holdout.
4. Require at least 85% precision and recall under both frozen evaluators.
5. On pass, write the Phase 2 checkpoint report and tag `checkpoint-phase-2`. On failure, stop
   public deployment and publish the required categorized failure report.

## Demo slice

- Add PDF upload, job status/progress, canonical rows, rendered page, evidence, and delete APIs.
- Use an atomic filesystem queue and three reusable worker processes with content-addressed stage
  caches. Recover interrupted jobs after restart.
- Add a Next.js client demo with upload, progress, extracted item table, page preview, and evidence
  highlighting. Editing, export, authentication, Gemini recovery, profiles, and full Temporal
  orchestration remain later phases.
- Accept only PDF files up to 25 MiB, cap the queue at 20 jobs, never list documents, never expose
  raw OCR/VLM diagnostics or paths, and delete demo artifacts after six hours.

## Side-by-side deployment

- Deploy an isolated `gmoney-v2-demo` Compose project under `/home/azureuser/gmoneyv2`.
- Expose only nginx on public port 3100. Bind backend diagnostics to localhost:8100 and
  PaddleOCR-VL to localhost:8111.
- Preserve the existing seven-container legacy project, its port 3000, images, volumes, and data.
- Tag images with the Git SHA, record their digests, and keep runtime data under
  `/home/azureuser/gmoneyv2-runtime`.
- The user opens Azure NSG TCP 3100. Public unauthenticated access and its PHI risk were explicitly
  accepted for this time-limited demo.

## Acceptance

- Python tests/Ruff and frontend lint/type/build pass.
- Known regression bills remain at 100% precision and recall.
- The sealed holdout passes the 85%/85% gate under both evaluators.
- Three concurrent documents complete on the D16 without OOM, swapping, corruption, or duplicate
  rows.
- A public upload on port 3100 shows extracted rows and synchronized page evidence.
- Port 3000 continues returning HTTP 200; ports 8100 and 8111 are not public.
