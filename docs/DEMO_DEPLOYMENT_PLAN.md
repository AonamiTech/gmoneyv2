# Editable D16 Client Demo Deployment Plan

## Goal

Deploy the Phase 3 extraction engine as an explicitly non-production, unauthenticated
client demo on `20.57.131.189` without changing the existing application on port 3000.

## Quality boundary

The annotated exposed regression passes the accuracy targets recorded in the Phase 3
review. Newly uploaded bills have no gold annotation, so the UI makes no per-document
precision or recall claim. This deployment does not satisfy or weaken the blocked
unseen-hospital Phase 3 checkpoint gate.

## Demo slice

- Provide PDF upload, job status/progress, canonical rows, rendered pages, evidence,
  revisioned review, approval, export, and delete APIs.
- Use an atomic filesystem queue and two reusable worker processes with content-addressed stage
  caches. Recover interrupted jobs after restart.
- Provide a Next.js review desk with a browser-local document queue, two visible
  inference lanes, row filtering, original-versus-corrected values, evidence relinking,
  reviewer-added rows, structural issue resolution, approval, and CSV/JSON/evidence exports.
- Preserve immutable machine output and store reviewer changes in an atomic revisioned overlay.
  Authentication, a global document list, PostgreSQL, and Temporal remain outside this demo.
- Accept only PDF files up to 25 MiB, cap the queue at 20 jobs, never list documents, never expose
  raw OCR/VLM diagnostics or paths, reject documents over 200 pages, and delete demo artifacts
  after six hours.

## Side-by-side deployment

- Deploy an isolated `gmoney-v2-demo` Compose project under `/home/azureuser/gmoneyv2`.
- Expose only nginx on public port 3100. Bind backend diagnostics to localhost:8100 and
  PaddleOCR-VL to localhost:8111.
- Preserve the existing seven-container legacy project, its port 3000, images, volumes, and data.
- Build images on the control host, tag them with the Git SHA, stream them to D16,
  record their digests, and keep runtime data under
  `/home/azureuser/gmoneyv2-runtime`.
- The user opens Azure NSG TCP 3100 if required. Public unauthenticated HTTP access,
  unencrypted uploads, and the resulting PHI risk were explicitly accepted for this
  time-limited demo.
- Preserve `/home/azureuser/gmoneyv2-phase3-sample11-final`; it is not mounted into
  the public demo and is not subject to demo cleanup.

## Acceptance

- Python tests/Ruff and frontend lint/type/build pass.
- The committed annotated regression remains above its recorded Phase 3 gates.
- Bills 10 and 11 process concurrently on D16 without OOM, swapping, corruption, or
  duplicate rows and reproduce their retained 80-row and 36-row semantic outputs.
- Row correction, evidence relinking, reviewer addition/rejection, issue resolution,
  approval, and all three exports pass against the deployed API.
- A public upload on port 3100 shows extracted rows and synchronized page evidence;
  ports 8100 and 8111 remain loopback-only.
- Port 3000 continues returning HTTP 200; ports 8100 and 8111 are not public.
