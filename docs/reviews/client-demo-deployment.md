# Editable client demo deployment review

Date: 2026-07-14

> Historical release record: this documents the currently deployed six-hour build. The
> verified 30-day history and corrected quantity/unit-price candidate supersedes it in the
> repository, but deployment is intentionally waiting for the required managed data disk;
> the VM root disk has only 3.8 GiB free and the two 440 GiB NVMe devices are ephemeral.

## Outcome

The unauthenticated editable evidence demo is deployed at
`http://20.57.131.189:3100/` on D16. A fresh browser discovers the shared active and
recent bill queue from the server; it no longer depends on browser-local job IDs.
The legacy application on port 3000 remains unchanged and returns HTTP 200.

This is an explicitly non-production demo. It uses plain HTTP, has no authentication,
and exposes recent bill filenames, statuses, rendered pages, and reviewed results to
anyone who can reach the URL. Demo jobs are automatically removed six hours after
completion or their last review activity.

## Release

- Review/edit implementation: `b78ea1594f53`.
- Shared queue implementation: `828f946d4ebbdfd7442a887b8003d6d142a5291c`.
- API image: `sha256:3fbd97fce76bb66393a041df395ee8d3ec178aa1b4e66f70e38a39362761759d`.
- Worker image: `sha256:30ce74aad4b5f7fa69b8d5cafd8b5bfcc40fbb696669d2127f13f70ff34c5f89`.
- Frontend image: `sha256:eaf052786829dc1183645aa93d62b31d3012e4ad2a589c7ca27d75ff0967deb9`.
- Release directory: `/home/azureuser/gmoneyv2-releases/828f946d4ebb`.

Only nginx is public on port 3100. The API and PaddleOCR-VL diagnostics remain bound
to loopback ports 8100 and 8111. Public connection attempts to both diagnostic ports
failed as expected.

## Verification

- Full Python suite: 100 tests passed.
- Ruff, frontend ESLint, TypeScript, Next.js production build, Compose configuration,
  and Docker image smoke tests passed.
- Every demo container is running with zero restarts and `OOMKilled=false`; the
  post-acceptance log scan found no error markers.
- The host had 58 GiB memory available after acceptance and no swap was used.
- Removing only superseded demo image tags restored 6.7 GiB free on the 29 GiB root
  filesystem. The retained Phase 3 artifacts were not pruned.

## Concurrent extraction acceptance

Bills 10 and 11 were submitted together and processed in both inference lanes. Both
completed in approximately 5 minutes 34 seconds:

| Bill | Pages | Live rows | Retained rows | Semantic comparison |
|---|---:|---:|---:|---|
| Bill 10 | 4 | 80 | 80 | Exact after excluding `created_at` |
| Bill 11 | 2 | 36 | 36 | Exact after excluding `created_at` |

The live source hashes and page-asset metadata also matched the retained outputs.
There were no OOM events, restarts, duplicate rows, or extraction errors.

## Review and export acceptance

The deployed API passed evidence relinking, revision conflict, reviewer row,
soft-rejection, structural issue, approval, and export checks:

- Bill 11 relinked evidence on a machine row while preserving its original machine
  values. An intentional stale `If-Match` mutation returned HTTP 409.
- A grounded reviewer row was added and then soft-rejected. Approval retained the 36
  active rows and excluded the rejected test row from CSV and JSON exports.
- Bill 10's open structural issue was resolved against its visible page before the
  80-row ledger was approved.
- CSV contained one header plus 36 active rows. JSON contained 36 active rows.
  Evidence ZIP integrity and its internal SHA-256 manifest both passed; the bundle
  contains reviewed output, immutable machine output, and both referenced page images.
- Machine-result hashes were identical before and after all review mutations.

## Preserved row-level evidence

`/home/azureuser/gmoneyv2-phase3-sample11-final` remains intact and separate from
demo cleanup. Its 809 retained artifact checks and 22 final-result checks replayed
successfully. The artifact manifest itself remains
`88550a2143e2acc4594b03b751ccca36832ea86799e8504bb6b8c3df9c6f2ccd`.

The retained eleven-bill set has no frozen gold, so it supports row-level audit and
artifact integrity checks but not a defensible precision, recall, or accuracy claim.
The existing gold regression remains above the 85% gate as recorded in the Phase 3
review. At the observed artifact rate, retaining 5,000 bills requires roughly 565 GB
before replication and headroom; production should use a larger data volume or object
storage instead of this root disk.
