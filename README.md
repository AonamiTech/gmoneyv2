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

# Evaluate a frozen local, challenger, or profile manifest.
.venv/bin/gmoney-phase3-evaluate run \
  --manifest data/phase3/evaluation-manifest.json \
  --output artifacts/phase3/quality.json
```

See [phase-3-plan.md](phase-3-plan.md) and
[docs/reviews/phase-3.md](docs/reviews/phase-3.md) for the gates and current status.

## Editable client demo

The demo Compose project exposes the Next.js evidence desk and FastAPI review API on
port 3100 while keeping model and API diagnostics on loopback. It processes two bills
concurrently, preserves immutable machine rows, and stores reviewer corrections in a
revisioned filesystem overlay. Its shared queue lets a fresh browser discover active and
recent bills without retaining browser-local job IDs. Completed documents are labelled by
their evidence-grounded hospital identity; reviewers can correct that identity, service dates,
rows, and evidence without changing the machine result. The evidence workspace provides an
independently scrolling ledger plus resize, fit, zoom, focus, and fullscreen page controls.

```bash
GMONEY_IMAGE_TAG=$(git rev-parse --short=12 HEAD) \
  docker compose -f compose.demo.yaml up -d --build
```

The demo has no authentication and serves plain HTTP. Uploaded PDFs and derived review
artifacts are deleted six hours after completion; do not treat it as a production PHI
system. See [docs/DEMO_DEPLOYMENT_PLAN.md](docs/DEMO_DEPLOYMENT_PLAN.md).
