#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

.venv/bin/ruff check .
.venv/bin/pytest -W error
.venv/bin/gmoney-corpus build \
  --config corpus/source-registry.json \
  --output data/catalog.json
.venv/bin/gmoney-corpus validate --catalog data/catalog.json
docker compose config --quiet

