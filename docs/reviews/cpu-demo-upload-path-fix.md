# CPU demo upload-path remediation

Date: 2026-07-20

## Outcome

The CPU evidence demo at `http://20.57.131.189:3100/` now runs release
`6f7f96bab8da027066bcc8e73dc7044db2507776`. It uses only
`compose.demo.yaml`: the worker is PaddlePaddle CPU 3.2.2 and the existing
non-CUDA llama.cpp service was not recreated.

Nginx request-frequency limiting was removed. The proxy keeps a 52 MiB
multipart envelope, while the CPU API remains authoritative with a 25 MiB PDF
limit. The runtime configuration is persisted in a mode-600 release `.env`,
including the prior 3 GiB storage floor and existing runtime data root.

## Root cause

The superseded release limited the shared `/api/v2/documents` location to ten
requests per minute with a burst of three. That location served uploads, the
three-second Active poll, and History queries. Its Nginx logs contained 618
rejections: 431 Active polls, 106 legacy list calls, 77 History calls, and four
uploads. Rejected calls returned HTTP 503 before reaching the healthy API.

Refresh did not delete retained jobs. Seven whole-document DELETE requests had
returned HTTP 204: one on July 15 and six on July 20. Per the operator's
decision, those bills were not restored and one-click deletion was unchanged.

## Validation

- Before deployment, 127 Python tests, Ruff, lock validation, frontend
  lint/typecheck/build, CPU/GPU Compose rendering, Nginx syntax, shell syntax,
  credential scanning, and Git diff checks passed.
- Twenty-five rapid public Active-list calls all returned HTTP 200. The live
  Nginx configuration contains no `limit_req` directive or limiting log entry.
- All five retained documents returned HTTP 200 for rows, review, and first-page
  evidence. Their combined inventory is 70 pages and 3,102 rows.
- The five-job source/result fingerprint remained
  `3bc4f4e9db7d8315a7f325858446ad4b3e96a0bfa074d0a32098dd7bb3ed15e0`
  before cutover, after cutover, after canary deletion, and after restart.
- A public `Bill 12.pdf` canary was accepted, completed three pages and 91 rows,
  exposed the new totals response, and retained source SHA-256
  `5d73bbe431954ae6183cb6d34b52c3dd491aae90776956b07c0d5cfab6486c35`.
  Only this duplicate canary was deleted afterward.
- All five containers were running with zero restart counts and
  `OOMKilled=false`; logs contained no traceback, fatal, OOM, or CUDA-error
  marker. Public HTTP returned 200 and ports 8100/8111 remained externally
  blocked.
- The root filesystem retained 8,596,918,272 free bytes, above the configured
  3 GiB floor. The previous release and images remain installed for rollback.

The CPU host accepts `azureuser` with `bhavik-cc_key.pem`; it rejected the
`gsevenlabs.pem` identity used by the GPU host.
