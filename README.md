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

