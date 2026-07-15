# Editable client demo deployment review

Date: 2026-07-15

> Current release record: commit `6588d6915f49` deploys the verified 30-day history and
> corrected quantity/unit-price implementation. The explicitly authorized root-disk fallback
> uses a 3 GiB upload floor; a managed disk remains required before scaling toward 5,000 bills.

## Outcome

The unauthenticated editable evidence demo is deployed at
`http://20.57.131.189:3100/` on D16. A fresh browser discovers the shared active and
historical bill indexes from the server; it no longer depends on browser-local job IDs.
The superseded port-3000 project, bill uploads, PostgreSQL, Redis, Celery, containers,
network, and project images were removed. Port 3000 has no listener.

This is an explicitly non-production demo. It uses plain HTTP, has no authentication,
and exposes retained bill filenames, statuses, rendered pages, and reviewed results to
anyone who can reach the URL. Demo jobs are automatically removed 30 days after
completion or their last review activity.

## Release

- History and column-semantics implementation: `8fa3acc44859`.
- Root-disk storage-floor configuration: `6588d6915f49`.
- API image: `sha256:81338fef2df4f4e2391dcb09c3de040560eb49ff1aad1c08333a7a78b37ac426`.
- Worker image: `sha256:cb6797689d67ca0b78a60f400e7505d3f70c082f1af82a465ca73dabfcff332d`.
- Frontend image: `sha256:c0e701cdc58926d68642044b117b7de863e24358f7efa43b1fa998b975d1e4e4`.
- Release directory: `/home/azureuser/gmoneyv2-releases/6588d6915f49`.

Only nginx is public on port 3100. The API and PaddleOCR-VL diagnostics remain bound
to loopback ports 8100 and 8111. Public connection attempts to both diagnostic ports
failed as expected.

## Verification

- Full Python suite: 119 tests passed.
- Ruff, frontend ESLint, TypeScript, Next.js production build, Compose configuration,
  and Docker image smoke tests passed.
- All five current containers are running with zero restarts and `OOMKilled=false`; no fatal,
  traceback, or OOM markers appeared in the post-cutover application logs.
- Removing the port-3000 data and all superseded unused images restored 15 GiB free on the
  29 GiB root filesystem. The checksum-protected Phase 3 artifacts were not pruned.

## Current imported regression acceptance

The validated 11-bill replay was hard-linked into the public history store. A service restart
preserved all 11 entries and a second import skipped all 11 idempotently:

| Bill | Pages | Live rows | Retained rows | Semantic comparison |
|---|---:|---:|---:|---|
| Bill 10 | 4 | 76 | 76 | 75 quantity/unit-price pairs; no date-prefixed descriptions |
| Bill 11 | 2 | 35 | 35 | All 35 quantity/unit-price pairs; page evidence HTTP 200 |

The live source hashes and page-asset metadata also matched the retained outputs.
There were no OOM events, duplicate rows, or extraction errors.

## Historical review and export acceptance

The prior deployment passed evidence relinking, revision conflict, reviewer row,
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
The current 813-file candidate archive and all 11 candidate results also passed their
checksum manifests after deployment; public page evidence is hard-linked to that archive.

The retained eleven-bill set has no frozen gold, so it supports row-level audit and
artifact integrity checks but not a defensible precision, recall, or accuracy claim.
The existing gold regression remains above the 85% gate as recorded in the Phase 3
review. At the observed artifact rate, retaining 5,000 bills requires roughly 565 GB
before replication and headroom; production should use a larger data volume or object
storage instead of this root disk.
