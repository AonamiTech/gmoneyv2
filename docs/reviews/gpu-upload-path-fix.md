# GPU public upload-path fix

Date: 2026-07-15
Release: `409a538740f5-gpu-nolimit-20260715T050917Z`

## Outcome

The public GPU demo at `http://34.180.11.221/` no longer applies Nginx
request-frequency limiting. The exact `/api/v2/documents` location retains
streaming request forwarding, while a 52 MiB proxy envelope leaves multipart
room around the API's configured 50 MiB PDF limit. The API now reports its
configured size in JSON, and the frontend maps non-JSON proxy failures to
status-specific messages.

Browser refresh does not delete bills. Active and History are reconstructed
from the persistent server-side job directories. Completed jobs remain subject
to the existing 720-hour retention period or explicit deletion.

## Validation

- The full local suite passed: 122 Python tests, Ruff, lock validation, frontend
  lint/typecheck/production build, merged Compose validation, shell syntax,
  Nginx syntax, and diff checks.
- The deployed `nginx -T` output contains `client_max_body_size 52m` and no
  `limit_req` directive. Twenty-five rapid public list requests all returned
  HTTP 200 rather than 503.
- A 53 MiB multipart request returned HTTP 413 at Nginx and created no job. The
  deployed frontend bundle contains the specific 413 error mapping.
- The unchanged 36,974,480-byte `Bill 1.pdf` was uploaded through public Nginx,
  accepted with HTTP 202, and completed 30 pages and 1,332 rows. Its uploaded
  source retained SHA-256
  `7dc99f26cc46cecd5e892cb5442d0a511ddcc19da50802d6a5702aa6c0bf3a08`.
  The duplicate validation job was deleted afterward.
- The 12 retained jobs had fingerprint
  `6e197fb4e067b1d2b10d44647cf7fa0529e8f45934d1507487c503121258d5c4`
  before deployment and the same fingerprint after deployment, canary cleanup,
  and an affected-service restart. Both post-restart History fetches returned
  12 complete jobs, 166 pages, and 5,551 rows.
- All five containers remained running with zero restarts and `OOMKilled=false`.
  Public HTTP returned 200, while ports 8100 and 8111 remained externally
  blocked. About 21.2 GB remained free, above the 10 GiB storage floor.

## Operational boundary

The request-frequency limiter is removed, but the API still enforces the
20-active-job queue cap, 50 MiB file cap, 200-page cap, and storage floor. The
previous release remains installed for rollback; persistent runtime data was
never copied, replaced, or remounted.
