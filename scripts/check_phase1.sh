#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

.venv/bin/ruff check .
.venv/bin/pytest -W error
docker compose config --quiet
docker compose -f compose.inference.yaml config --quiet
.venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("benchmarks/phase1-summary.json").read_text())
assert summary["geometry"]["maximum_round_trip_error_pixels"] <= 2
assert summary["layout"]["page_table_recall"] >= 0.95
assert summary["layout"]["effective_page_table_recall"] == 1
assert summary["heavy_parser"]["slots"] == 3
assert summary["heavy_parser"]["concurrent_outputs_identical"] is True
assert summary["heavy_parser"]["observed_server_memory_gib_max"] < 8
print("phase 1 summary gates passed")
PY

