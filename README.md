# GMoney V2

Evidence-grounded extraction and review for multi-layout hospital bills.

The implementation is delivered through the gated phases in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). The legacy repository is a read-only corpus source; its extraction schemas and profiles are not imported.

## Phase 0 development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/gmoney-corpus build --config corpus/source-registry.json --output data/catalog.json
.venv/bin/pytest
docker compose config --quiet
```

Copy `.env.example` to `.env` only for local execution. Secrets must not be committed.

## Phase 3 offline tools

Phase 3 remains file-backed and offline. Gemini is off by default; use challenger
mode only with de-identified crops, and use enabled mode only after generating a
passing frozen promotion decision.

```bash
# Reserve candidate unseen bills without treating filename hashes as hospital IDs.
.venv/bin/gmoney-corpus freeze-phase3-candidates \
  --catalog data/catalog.json \
  --output corpus/phase3-sealed-candidates.json \
  --exposed-sha256 5d73bbe431954ae6183cb6d34b52c3dd491aae90776956b07c0d5cfab6486c35

# Inspect profile-registry capacity for 5,000 hospitals / 20,000 variants.
.venv/bin/gmoney-profiles benchmark --variants 20000 --queries 100

# Live hospital-specific profile writes use the coordinated admin container.
docker compose -f compose.demo.yaml --profile admin run --rm \
  -v "$PWD/data/phase3/hospital-observations.json:/admin-input/observations.json:ro" \
  profile-admin construct \
  --observations /admin-input/observations.json \
  --profile-key hospital-items \
  --profile-version 1 \
  --hospital-id hospital-id \
  --hospital-name "Hospital display name"

# Evaluate a frozen local, challenger, or profile manifest.
.venv/bin/gmoney-phase3-evaluate run \
  --manifest data/phase3/evaluation-manifest.json \
  --output artifacts/phase3/quality.json
```

See [phase-3-plan.md](phase-3-plan.md) and
[docs/reviews/phase-3.md](docs/reviews/phase-3.md) for the gates and current status.

## GPU demo

Use `compose.demo.yaml` with `compose.gpu.yaml` to run Paddle OCR/layout and the
PaddleOCR-VL llama.cpp service on one NVIDIA GPU while preserving the existing
API and evidence-review behavior. See
[docs/GPU_DEMO_DEPLOYMENT.md](docs/GPU_DEMO_DEPLOYMENT.md) for host preparation,
storage constraints, release commands, and acceptance gates.

## Editable client demo

The demo Compose project exposes the Next.js evidence desk and FastAPI review API on
port 3100 while keeping model and API diagnostics on loopback. It processes two bills
concurrently, preserves immutable machine rows, and stores reviewer corrections in a
revisioned filesystem overlay. Its shared queue lets a fresh browser discover active and
historical bills without retaining browser-local job IDs. Completed documents are
searchable for 30 days and labelled by
their evidence-grounded hospital identity; reviewers can correct that identity, service dates,
rows, and evidence without changing the machine result. The evidence workspace provides an
independently scrolling ledger plus resize, fit, zoom, focus, and fullscreen page controls.
It also reconciles the active reviewed item sum with an evidence-grounded primary total printed
on the bill and exposes every other explicit printed total with its page evidence. Conflicting
document totals are shown for review instead of producing a misleading difference;
section totals, payments, advances, and balance figures are excluded from the comparison.

```bash
GMONEY_IMAGE_TAG=$(git rev-parse --short=12 HEAD) \
  docker compose -f compose.demo.yaml up -d --build
```

Completed jobs created before document-total extraction can be upgraded from their retained
page OCR caches without rerunning inference. Review the dry-run summary before applying it:

```bash
docker compose -f compose.demo.yaml run --rm --no-deps api \
  gmoney-demo-backfill-totals --root /runtime
docker compose -f compose.demo.yaml run --rm --no-deps api \
  gmoney-demo-backfill-totals --root /runtime --apply
```

Completed jobs can also be re-extracted from their retained source and inference caches while
preserving job IDs and reviewer history. Stage and inspect the entire batch while the service is
live, then stop the API and worker for the short cutover. Cutover rechecks every review digest,
invalidates old approvals, and keeps a durable rollback batch under
`/runtime/jobs/.reprocess-backups`.

```bash
# Selection-only dry run (repeat --job-id for each intended document).
docker compose -f compose.demo.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint gmoney-demo-reprocess worker \
  --root /runtime --job-id <job-id>

# Cached GPU re-extraction into a validated staging batch (repeat --job-id as needed).
docker compose -f compose.demo.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint gmoney-demo-reprocess worker \
  --root /runtime --job-id <job-id> --stage-only \
  --vl-url http://paddleocr-vl:8111 --paddle-device gpu:0 --vl-device cuda:0

# After inspection and during the maintenance window, apply the emitted staging path.
docker compose -f compose.demo.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint gmoney-demo-reprocess worker \
  --root /runtime --apply-staged \
  /runtime/jobs/.reprocess-staging/<batch-timestamp>

# Roll back every document in one emitted backup batch.
docker compose -f compose.demo.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint gmoney-demo-reprocess worker \
  --root /runtime --rollback-from \
  /runtime/jobs/.reprocess-backups/<batch-timestamp>
```

The demo has no authentication and serves plain HTTP. Uploaded PDFs and full evidence/review
artifacts are stored on a dedicated data volume and deleted 30 days after their latest
extraction or review activity; do not treat it as a production PHI system. See
[docs/DEMO_DEPLOYMENT_PLAN.md](docs/DEMO_DEPLOYMENT_PLAN.md).
