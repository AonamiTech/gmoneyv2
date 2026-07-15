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
- Provide a Next.js review desk with a shared server-backed document queue, two visible
  inference lanes, row filtering, original-versus-corrected values, evidence relinking,
  reviewer-added rows, structural issue resolution, approval, and CSV/JSON/evidence exports.
  The detected hospital name is the primary document label and can be corrected with
  revisioned header evidence. The ledger uses an independently scrolling table, a service-date
  column, a resizable evidence split, page/width fit modes, zoom, evidence focus, and fullscreen.
- Preserve immutable machine output and store reviewer changes in an atomic revisioned overlay.
  Authentication, PostgreSQL, and Temporal remain outside this demo.
- Accept only PDF files up to 25 MiB, cap the queue at 20 jobs, expose paginated Active and
  History indexes with hospital/filename search, never expose raw OCR/VLM diagnostics or paths,
  reject documents over 200 pages, and delete demo artifacts 30 days after their latest
  extraction or review activity.
- Keep originals, page renders, OCR/model artifacts, results, review overlays, and exports on a
  dedicated 1 TB managed data disk. Reject uploads below a 20 GiB free-space floor.

## Side-by-side deployment

The deployment subscription must first attach a zonal 1 TiB managed disk; the VM has no managed
data disk or management identity, so this step runs from an authenticated Azure CLI session:

```bash
az disk create \
  --resource-group aonami_group \
  --name gmoney4workers-history \
  --location westus2 \
  --zone 3 \
  --size-gb 1024 \
  --sku Premium_LRS
az vm disk attach \
  --resource-group aonami_group \
  --vm-name gmoney4workers \
  --name gmoney4workers-history \
  --lun 0
```

After Azure reports the attachment, prepare only `/dev/disk/azure/scsi1/lun0`:

```bash
sudo scripts/prepare_demo_data_disk.sh /dev/disk/azure/scsi1/lun0
```

For a temporary root-disk-only client demo, the operator may explicitly override
`GMONEY_MIN_FREE_BYTES`; never set it below 3 GiB. This is not suitable for the projected
5,000-bill corpus and does not replace the managed-disk requirement for durable scale.

- Deploy an isolated `gmoney-v2-demo` Compose project under `/home/azureuser/gmoneyv2`.
- Expose only nginx on public port 3100. Bind backend diagnostics to localhost:8100 and
  PaddleOCR-VL to localhost:8111.
- Preserve the existing seven-container legacy project, its port 3000, images, volumes, and data.
- Build images on the control host, tag them with the Git SHA, stream them to D16,
  record their digests, and keep job runtime data under
  `/mnt/gmoney-data/gmoneyv2/runtime`. The existing root-disk model cache remains under
  `/home/azureuser/gmoneyv2-runtime/model-cache`.
- The user opens Azure NSG TCP 3100 if required. Public unauthenticated HTTP access,
  unencrypted uploads, and the resulting PHI risk were explicitly accepted for this
  time-limited demo.
- Preserve `/home/azureuser/gmoneyv2-phase3-sample11-final`; it is not mounted into
  the public demo and is not subject to demo cleanup.
- Mount the managed disk at `/mnt/gmoney-data`, set `GMONEY_DATA_ROOT` to
  `/mnt/gmoney-data/gmoneyv2/runtime`, and leave the VM's ephemeral NVMe devices unused for
  durable history. Prepare only a newly attached blank disk of at least 900 GB with
  `scripts/prepare_demo_data_disk.sh`.
- Rerun and import all eleven exposed samples through the idempotent history importer. Public
  history hard-links their evidence from a versioned validation run, so public expiry or deletion
  never removes the checksum-protected evaluation archive.

## Acceptance

- Python tests/Ruff and frontend lint/type/build pass.
- The committed annotated regression remains above its recorded Phase 3 gates.
- All eleven exposed Sample Bills process in two D16 lanes without OOM, swapping,
  corruption, or ungrounded accepted rows: 163 pages and 5,460 grounded charge rows in total.
  Bills 10 and 11 produce 76 and 35 charge rows respectively. Bill 10 excludes four item-total
  and payer-credit footer fragments; Bill 11 excludes the contact-phone false positive.
- Bill 10 exposes 75 structured service dates without retaining date ranges in descriptions;
  Bill 10 and Bill 11 are labelled `Vijaya Group of Hospitals` and
  `Dr.Kamakshi Memorial Hospitals Pvt. Ltd.` from page-one evidence.
- Bill 11 exposes the printed quantity and unit price on all 35 rows. Repeated shifted headers and
  OCR-merged `Quantity UnitPrice`/expiry tokens retain their own grounded column semantics. Across
  the eleven bills, all 1,822 rows with both fields reconcile quantity × unit price to the printed
  amount, with no suspicious high-quantity/low-rate inversions in the validation scan.
- A fresh browser can search and open all eleven historical samples after a service restart;
  each entry reports its latest activity and 30-day expiry.
- Row correction, evidence relinking, reviewer addition/rejection, issue resolution,
  approval, and all three exports pass against the deployed API.
- A public upload on port 3100 shows extracted rows and synchronized page evidence;
  ports 8100 and 8111 remain loopback-only.
- Port 3000 continues returning HTTP 200; ports 8100 and 8111 are not public.
