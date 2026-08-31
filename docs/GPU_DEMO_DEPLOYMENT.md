# GPU demo deployment

## Target profile

The initial GPU demo target is `gsevenlabs` in GCE `asia-south1-b` at
`34.180.11.221`. It has 8 vCPUs, 54 GiB RAM, one NVIDIA Tesla T4-class PCI
device, and a 50 GiB persistent-balanced root disk. The deployment is a fresh,
empty runtime: it does not copy jobs, reviews, or validation artifacts from the
CPU demo.

The web app intentionally matches the existing demo boundary: unauthenticated
plain HTTP on public port 3100. API and model diagnostics remain loopback-only
on ports 8100 and 8111. This is not a production PHI deployment.

## GPU runtime

- Install the Google-qualified production NVIDIA driver, Docker Engine, and the
  NVIDIA Container Toolkit. Verify both host `nvidia-smi` and an NVIDIA CUDA
  container before deploying the application.
- Use `compose.demo.yaml` with the `compose.gpu.yaml` overlay. The overlay keeps
  the existing API/frontend behavior, installs PaddlePaddle GPU 3.2.2 for CUDA
  11.8, sends OCR and layout inference to `gpu:0`, and uses the digest-pinned
  CUDA llama.cpp server with GPU layer offload.
- Begin with one Paddle worker and one VLM slot. The T4 shares its 16 GiB VRAM
  between two persistent model stacks; increase concurrency only after a
  two-document canary proves there is no OOM, restart, or throughput regression.
- Keep runtime jobs and model caches under `/home/ubuntu/gmoneyv2-runtime`.
  Set a 10 GiB upload rejection floor because the VM has only a 50 GiB root
  disk. Attach durable storage before retaining a large bill history.

## Release command

### Isolated candidate canary

Before a production cutover, the application services can run against a clean
candidate runtime without exposing another public endpoint or copying candidate
jobs into production. The canary reuses the live digest-pinned VLM container so
the single T4 does not load a second llama.cpp model stack. Start only the listed
application services; the canary must not start its own `paddleocr-vl` service.

```bash
export GMONEY_DATA_ROOT=/home/ubuntu/gmoneyv2-candidate-runtime/$GMONEY_IMAGE_TAG
export GMONEY_PROFILE_ROOT="$GMONEY_DATA_ROOT/profiles"
export GMONEY_MODEL_ROOT=/home/ubuntu/gmoneyv2-runtime/model-cache
export GMONEY_CANARY_HTTP_PORT=3110
export GMONEY_CANARY_API_PORT=18100
install -d -o 10001 -g 10001 \
  "$GMONEY_DATA_ROOT/jobs" "$GMONEY_DATA_ROOT/config" "$GMONEY_PROFILE_ROOT"
docker compose \
  -f compose.demo.yaml -f compose.gpu.yaml -f compose.canary.yaml \
  up -d --build --no-deps api frontend worker nginx
curl --fail http://127.0.0.1:3110/api/v2/health/ready
```

The live stack must be idle before candidate inference starts. Process one
candidate document at a time and pause the candidate worker if a live job
arrives. Stop the canary with the same three Compose files; never use `down -v`
because runtime data is bind-mounted and retained for the evaluation report.

### Mandatory rollback capture

Before building or replacing any application container, capture the exact
currently running release. This is an availability rollback point; it is not
accuracy-certified unless its frozen corpus report passed.

```bash
rollback_dir="/home/ubuntu/gmoneyv2-releases/pre-${GMONEY_IMAGE_TAG}"
install -d -m 700 "$rollback_dir"
docker compose -f compose.demo.yaml -f compose.gpu.yaml ps -q \
  api frontend worker > "$rollback_dir/container-ids.txt"
while read -r container_id; do docker inspect "$container_id"; done \
  < "$rollback_dir/container-ids.txt" > "$rollback_dir/containers.jsonl"
image_ids="$({ while read -r container_id; do docker inspect -f '{{.Image}}' "$container_id"; done; } \
  < "$rollback_dir/container-ids.txt" | sort -u)"
docker image save -o "$rollback_dir/application-images.tar" $image_ids
tar --xattrs --acls -C /home/ubuntu -czf "$rollback_dir/runtime-backup.tar.gz" \
  gmoneyv2-runtime/jobs gmoneyv2-runtime/config gmoneyv2-runtime/profiles
sha256sum "$rollback_dir/application-images.tar" \
  "$rollback_dir/runtime-backup.tar.gz" > "$rollback_dir/SHA256SUMS"
```

Do not prune images, remove application tags, or delete this backup until the
replacement and following release have both been certified. Rebuilding a
deleted historical commit is not an exact image rollback.

From the release directory on the GPU host:

```bash
export GMONEY_IMAGE_TAG=<release-id>
export GMONEY_BUILD_REVISION="$(git rev-parse HEAD)"
test "${#GMONEY_BUILD_REVISION}" -eq 40
export GMONEY_DATA_ROOT=/home/ubuntu/gmoneyv2-runtime
export GMONEY_MODEL_ROOT=/home/ubuntu/gmoneyv2-runtime/model-cache
export GMONEY_PROFILE_ROOT=/home/ubuntu/gmoneyv2-runtime/profiles
export GMONEY_PROFILE_REGISTRY_LOCK=/home/ubuntu/gmoneyv2-runtime/config/profile-registry.lock
export GMONEY_MIN_FREE_BYTES=10737418240
export GMONEY_MAX_UPLOAD_BYTES=0
export GMONEY_GPU_WORKER_CONCURRENCY=1
install -d -o 10001 -g 10001 \
  "$GMONEY_DATA_ROOT/jobs" "$GMONEY_DATA_ROOT/config" "$GMONEY_PROFILE_ROOT"
docker compose -f compose.demo.yaml -f compose.gpu.yaml up -d --build
```

`GMONEY_BUILD_REVISION` is baked into every application image and its
`org.opencontainers.image.revision` label. The same value and the production
enforcement policy are stored read-only in `/etc/gmoney/release.json`. The GPU
overlay refuses to build the API, profile-admin, frontend, or worker with an
unknown or malformed revision. Runtime environment overrides are ignored when
the baked manifest exists.

After the services become healthy, attest the actual running containers and
write the sanitized release record:

```bash
python3 scripts/verify_release_attestation.py \
  --expected-revision "$GMONEY_BUILD_REVISION" \
  --compose-file compose.demo.yaml \
  --compose-file compose.gpu.yaml \
  --output "/home/ubuntu/gmoneyv2-releases/$GMONEY_IMAGE_TAG/release-attestation.json"
```

This fails unless each running image label, each container's immutable release
manifest, API readiness, the worker heartbeat, and frontend `/build.json` all
report the exact expected commit.

The verifier independently parses the worker timestamp and compares its age with
the maximum age and future-skew limits reported by readiness. A response that
merely claims `ready` cannot attest a stale, timezone-naive, malformed, or
future-dated heartbeat.

### Mandatory frozen corpus gate

The authoritative client archive remains outside Git. Seal it by content hash;
the generated manifest contains no client filenames. Detailed expectations are
also keyed only by source SHA-256.

```bash
gmoney-release-corpus seal \
  --cohort "production14=/secure/client-corpus/production14:14" \
  --cohort "passing36=/secure/client-corpus/passing36:36" \
  --cohort "staging159=/secure/client-corpus/staging159:159" \
  --baseline-data-root /secure/client-corpus/baseline-runtime \
  --gold /secure/client-corpus/audited-gold-by-sha256.json \
  --release-revision "$GMONEY_BUILD_REVISION" \
  --api-image-digest "$GMONEY_API_IMAGE_DIGEST" \
  --frontend-image-digest "$GMONEY_FRONTEND_IMAGE_DIGEST" \
  --worker-image-digest "$GMONEY_WORKER_IMAGE_DIGEST" \
  --output "/home/ubuntu/gmoneyv2-releases/$GMONEY_IMAGE_TAG/corpus-manifest.json"
```

Process the sealed sources with the candidate images into a clean candidate
data root, then evaluate it:

```bash
gmoney-release-corpus evaluate \
  --manifest "/home/ubuntu/gmoneyv2-releases/$GMONEY_IMAGE_TAG/corpus-manifest.json" \
  --source-root "production14=/secure/client-corpus/production14" \
  --source-root "passing36=/secure/client-corpus/passing36" \
  --source-root "staging159=/secure/client-corpus/staging159" \
  --data-root /home/ubuntu/gmoneyv2-candidate-runtime \
  --release-revision "$GMONEY_BUILD_REVISION" \
  --api-image-digest "$GMONEY_API_IMAGE_DIGEST" \
  --frontend-image-digest "$GMONEY_FRONTEND_IMAGE_DIGEST" \
  --worker-image-digest "$GMONEY_WORKER_IMAGE_DIGEST" \
  --output "/home/ubuntu/gmoneyv2-releases/$GMONEY_IMAGE_TAG/corpus-report.json"
```

The v2 evaluator rejects unknown manifest fields and duplicate hashes, requires
an audited gold fingerprint for every document and a baseline snapshot for all
36 known-passing documents, and compares exact canonical rows/evidence, Printed
tables/columns/links, dates, receipts, totals, issues, recovery metadata, and
provider usage. It also binds the report to the candidate commit and immutable
image digest. Missing any one of the three exact cohorts blocks deployment.

### Explicit historical v5 recertification

First produce a read-only plan. This does not recover, migrate, or publish any
workspace file:

```bash
gmoney-recertify \
  --data-root /home/ubuntu/gmoneyv2-runtime \
  --output /home/ubuntu/gmoneyv2-releases/recertification-plan.json
```

Review the plan, then apply that exact digest-bound file:

```bash
gmoney-recertify \
  --data-root /home/ubuntu/gmoneyv2-runtime \
  --apply-from /home/ubuntu/gmoneyv2-releases/recertification-plan.json \
  --output /home/ubuntu/gmoneyv2-releases/recertification-report.json
```

Applying publishes a fresh validation report and certification and clears any
old approval in the same publication transaction. Revision-2 jobs that used
recovery are reported as `reprocess_required`; they are never recertified from
insufficient historical recovery audit data.

The base demo always binds host port `3100`. If the cloud firewall only admits
standard HTTP, set `GMONEY_PUBLIC_HTTP_PORT=80` to add a second binding while
retaining port `3100`. This does not change the loopback-only API and VLM ports.
The API and Nginx do not impose a byte-size limit; the page cap, queue cap, and
free-space floor remain the upload safeguards.

The model directory must contain both checksum-verified PaddleOCR-VL 1.6 GGUF
files before startup. The runtime `jobs`, `config`, and model-cache directories
must be writable by container UID/GID `10001:10001`. The `config` directory
retains the versioned hospital column-alias registry shared by API and worker.
Create the profile directory before startup. Its optional `registry.json` is
mounted read-only into the API and worker. Both services coordinate snapshots
through `$GMONEY_DATA_ROOT/config/profile-registry.lock`. Any live
profile mutation must run through the on-demand administrative container, which
uses the same UID, registry paths, and profile-to-alias lock order as the live
services. Host-side `gmoney-profiles` writes are unsupported.

Before a release or profile mutation, validate the combined registries and the
container's real write permissions:

```bash
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile admin \
  run --rm profile-admin check-access
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile admin \
  run --rm profile-admin validate
```

These preflights are inspection-only with respect to registries, reviews, and
pending journals. `validate` reports whether recovery or migration is required;
it does not perform either operation. `check-access` creates and removes only
temporary permission probes and acquires the configured locks.

Run mutating commands through the same service. Registry, alias, jobs, and lock
paths come from the service environment and must not be overridden:

```bash
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile admin \
  run --rm -v /path/to/new-profile.json:/admin-input/profile.json:ro \
  profile-admin add --profile /admin-input/profile.json
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile admin \
  run --rm profile-admin transition PROFILE_KEY PROFILE_VERSION active \
  "holdout gates passed"
```

The admin service is profile-gated and is not started by the normal `up`
command. The API and worker retain read-only profile mounts.

## Acceptance

- Python tests, Ruff, frontend lint/typecheck/build, and merged Compose
  validation pass for the exact release source.
- The `profile-admin check-access` and `profile-admin validate` commands pass
  against the persisted runtime before container cutover.
- Host and disposable CUDA-container `nvidia-smi` checks pass. Paddle reports a
  CUDA build, sees GPU 0, and loads PP-OCRv6 plus PP-DocLayoutV3 on `gpu:0`.
- llama.cpp reports CUDA initialization and GPU layer offload; both model and
  worker processes appear in `nvidia-smi` without exhausting VRAM.
- `/api/v2/health/ready` reports ready with one worker lane, the public UI and
  health path return HTTP 200 on port 3100, and ports 8100/8111 are unreachable
  remotely while remaining healthy through loopback.
- `/api/v2/health/ready` and `/build.json` report the same full release SHA;
  readiness also reports a fresh worker heartbeat, its accepted freshness limits,
  and `release_consistent=true`.
- `docker image inspect` reports that same SHA in the
  `org.opencontainers.image.revision` label for API, frontend, and GPU worker.
- `scripts/verify_release_attestation.py` succeeds and its record is retained
  beside the release without bill contents or other PHI.
- A real PDF upload reaches `complete`, produces grounded rows and page evidence,
  survives a Compose restart, and records `gpu:0`/`cuda:0` in newly written
  inference cache model specifications.
- No container is OOM-killed or restarted, no fatal/error marker appears in
  application logs, and at least 10 GiB remains free after images/models/build
  cache cleanup.
- The frozen 14/36/159 corpus report has `passed=true` for the exact candidate
  image revision.
- After cutover, upload a production canary whose visible filename begins with
  `[CANARY]`. Retain its source, result, validation report, page evidence, and
  logs for the normal 30-day retention period; do not manually delete it.
- Keep both the captured operational images and candidate images. Cleanup is
  permitted only after the next release is certified and its rollback archive
  has been restore-tested.
