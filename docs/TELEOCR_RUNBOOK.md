# TeleOCR table reader — T4 test runbook

This runbook is for the owner's Tesla T4 (15 GB) host. It starts the TeleOCR service next to
the existing GPU demo stack, runs `gmoney-compare` on the Sample Bills in all three reader
modes, and lists what to send back. Nothing here was run on a GPU before hand-off: the code
is covered by unit tests with recorded/mocked model replies only.

| Mode (`GMONEY_TABLE_READER`) | What reads table rows | Reconciliation gate (`auto`) |
| --- | --- | --- |
| `heuristic` (default) | PP-OCRv6 + DocLayoutV3 row builder, PaddleOCR-VL fallback (unchanged) | reported, not enforced |
| `teleocr` | TeleOCR on every detected table + missed-table full-page guard | enforced |
| `teleocr_gemini` | TeleOCR, plus Gemini on every redacted table crop with cell-level consensus | enforced |

What still runs from Paddle in the TeleOCR modes, and why:

| Component | Used in TeleOCR modes? | Role |
| --- | --- | --- |
| PaddleOCR-VL 1.6 (llama.cpp) | **No** — never called; not started with `compose.teleocr.yaml` | was the VLM fallback |
| PP-DocLayoutV3 + OCR table proposals | Yes | finds the table boxes TeleOCR reads; amounts outside them trigger the full-page guard |
| PP-OCRv6 page/crop OCR | Yes | the token evidence every TeleOCR row must be grounded to (V6 evidence, review UI), document totals, hospital name |
| PP-OCR row builder (`ocr_rows.py`) | Only as printed-table evidence and as a **flagged** fallback | builds the printed source tables used for linking and C2 sub-totals; its rows are used only when TeleOCR failed, truncated, or found no charges where PP-OCR did, and that table is then a blocking review issue (`reader_provider_failed`, `reader_truncated`, `reader_no_rows`) |

"Enforced" means a job is `complete` only when validation passes **and** the bill reconciles
against its own printed totals; otherwise it is `needs_review` and approval is blocked by
`reconciliation_unverified` until reviewer edits reconcile it or a reasoned override is
recorded (`PATCH /api/v2/documents/{id}/reconciliation`). Set
`GMONEY_RECONCILIATION_GATE=enforce` to enforce it for the heuristic reader too, or `report`
to never enforce it.

## 1. Download the TeleOCR weights

Use the official Apache-2.0 weights `StarDoc-AI/TeleOCR` only (not the unofficial
`ldov/TeleOCR` copy). Pin the revision you test so results are reproducible.

```bash
export GMONEY_MODEL_ROOT=/home/ubuntu/gmoneyv2-runtime/model-cache   # same root as PaddleOCR-VL
.venv/bin/pip install 'huggingface-hub>=1,<2'
.venv/bin/python scripts/download_teleocr.py \
  --output "$GMONEY_MODEL_ROOT/teleocr" \
  --revision <commit-sha-from-the-model-page>
sudo chown -R 10001:10001 "$GMONEY_MODEL_ROOT/teleocr"
du -sh "$GMONEY_MODEL_ROOT/teleocr"    # expect roughly 2.5-5 GB
```

The directory must contain `config.json`, the processor/tokenizer files, the safetensors
weights and the remote-code file `modeling_naviocr.py`. The container mounts it read-only at
`/models/teleocr` and runs with `HF_HUB_OFFLINE=1`, so nothing is downloaded at runtime.

## 2. Build and start the service

The service is a separate image (`infra/docker/teleocr.Dockerfile`, PyTorch 2.8 / CUDA 12.6,
`transformers==4.57.1`; transformers 5.x fails with `KeyError: 'default'` in
`ROPE_INIT_FUNCTIONS`). It is behind the `teleocr` Compose profile, so the normal stack is
unchanged unless the profile is enabled.

```bash
export GMONEY_IMAGE_TAG=$(git rev-parse --short=12 HEAD)
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr build teleocr
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr up -d teleocr
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr logs -f teleocr
```

Service settings (Compose passes the `GMONEY_*` names through):

| Variable | Default | Meaning |
| --- | --- | --- |
| `GMONEY_TELEOCR_DTYPE` | `auto` | `auto` = `float16` on GPUs without bf16 (T4 is sm_75), `bfloat16` otherwise; or force `float16`/`bfloat16`/`float32` |
| `GMONEY_TELEOCR_MAX_CONCURRENCY` | `1` | requests generated at once on the GPU; keep 1 on a T4 |
| `GMONEY_TELEOCR_MAX_TOKENS` | `8192` | max new tokens per table read |

## 3. Health check and smoke test

```bash
curl -fsS http://127.0.0.1:8112/health
# {"status":"ok","model":"StarDoc-AI/TeleOCR","dtype":"float16","max_concurrency":1}
```

A 503 with an `error` field means the model did not load (wrong mount, missing remote code,
CUDA error); the message is in the body and the container log.

Smoke-test one table crop (any de-identified table image):

```bash
python3 - <<'EOF'
import base64, json, sys, urllib.request
image = base64.b64encode(open(sys.argv[1] if len(sys.argv) > 1 else "table.png", "rb").read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    {"type": "text", "text": "This is the image of a table. Please output the table in OTSL format."}]}],
    "max_tokens": 8192}
request = urllib.request.Request("http://127.0.0.1:8112/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
reply = json.load(urllib.request.urlopen(request, timeout=900))
print(reply["choices"][0]["message"]["content"][:2000]); print(reply["usage"], reply["latency_ms"], "ms")
EOF
```

Expect `<fcel>…<nl>` OTSL. If the T4 produces empty/garbage text or `!!!!` in `float16`, record
it (fp16 on T4 is untested) and retry with `GMONEY_TELEOCR_DTYPE=float32` to separate a
precision problem from a model problem.

## 4. VRAM budget on the 15 GB T4

| Process | Expected VRAM |
| --- | --- |
| Worker: PP-OCRv6 + PP-DocLayoutV3 (Paddle) | ~2-3 GB |
| PaddleOCR-VL 1.6 llama.cpp (`--ctx-size 24576`, `--fit-target 8192`) | ~4-6 GB |
| TeleOCR ~1.2B, fp16 weights | ~2.5-3 GB, plus vision activations and KV cache: ~4-7 GB peak on large crops |

All three together are near or above 15 GB. In `teleocr` and `teleocr_gemini` modes
PaddleOCR-VL is not called (TeleOCR replaces it for every table), so **stop PaddleOCR-VL for
the TeleOCR runs** and run the heuristic baseline separately with it up:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5   # keep running in a pane
docker compose -f compose.demo.yaml -f compose.gpu.yaml stop paddleocr-vl
```

If TeleOCR runs out of memory on full-page guard reads, lower the image size the service
accepts with `TELEOCR_MAX_PIXELS` (for example `4000000`) in the `teleocr` service
environment and record the change.

## 5. Gemini settings (`teleocr_gemini` only)

Gemini receives only redacted table crops (never a full page, never a guard band), at
temperature 0, with a JSON schema for rows. It needs no Phase-3 promotion file in this mode;
the existing `gemini_mode` rules and the "no Gemini during recovery" guard are unchanged.

```bash
export GEMINI_API_KEY=...                                  # never commit it
export GMONEY_GEMINI_MODEL=gemini-3.5-flash
export GMONEY_GEMINI_INPUT_COST_USD_PER_MILLION=<current list price>
export GMONEY_GEMINI_OUTPUT_COST_USD_PER_MILLION=<current list price>
export GMONEY_GEMINI_READER_MAX_COST_INR_PER_PAGE=0.30      # per-page cap
export GMONEY_INR_PER_USD=88                               # update to the day's rate
```

Set the two price variables from Google's current price list; with the defaults of `0` the
cost cap cannot bind and cost is reported as 0. Calls and measured cost are recorded in
`provider_usage` and in the comparison report.

## 6. Run the comparison on the Sample Bills

`gmoney-compare` runs inside the GPU worker image (it needs Paddle). Rebuild the worker at
this revision first so the command exists.

```bash
docker compose -f compose.demo.yaml -f compose.gpu.yaml build worker
mkdir -p /home/ubuntu/teleocr-compare && sudo chown 10001:10001 /home/ubuntu/teleocr-compare

compare() {  # $1 = modes, $2 = output sub-directory
  docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr run --rm --no-deps \
    -v "$PWD/Sample Bills:/bills:ro" -v /home/ubuntu/teleocr-compare:/out \
    -e GEMINI_API_KEY -e GMONEY_GEMINI_MODEL \
    -e GMONEY_GEMINI_INPUT_COST_USD_PER_MILLION -e GMONEY_GEMINI_OUTPUT_COST_USD_PER_MILLION \
    -e GMONEY_GEMINI_READER_MAX_COST_INR_PER_PAGE -e GMONEY_INR_PER_USD \
    --entrypoint gmoney-compare worker \
    --source-dir /bills --output "/out/$2" --modes "$1" \
    --vl-url http://paddleocr-vl:8111 --paddle-device gpu:0 --vl-device cuda:0 \
    --gpu-cost-inr-per-hour <host price per hour in INR>
}

# Baseline with PaddleOCR-VL running.
docker compose -f compose.demo.yaml -f compose.gpu.yaml up -d paddleocr-vl
compare heuristic heuristic

# TeleOCR modes with PaddleOCR-VL stopped and TeleOCR running.
docker compose -f compose.demo.yaml -f compose.gpu.yaml stop paddleocr-vl
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr up -d teleocr
compare teleocr,teleocr_gemini teleocr
```

If VRAM allows all services at once, a single `compare heuristic,teleocr,teleocr_gemini all`
produces one side-by-side table. Use `--limit 2` for a quick dry run first.

Each output directory contains:

- `report.md` / `report.json`: per bill and mode — pages, rows by role, reconciliation status
  and every failed check (section, expected, actual, difference), reader disagreement counts,
  pending-review and ungrounded reader rows, full-page guard reads, seconds per page, TeleOCR
  and Gemini call counts, and estimated ₹ per page (Gemini measured + GPU time at the given
  hourly price). Bills appear only as `bill-NN` and a source-hash prefix.
- `aliases.local.json`: `bill-NN` → file name. **Keep on the host.**
- `runs/`: full extraction artifacts (page images, crops, caches). **Keep on the host.**

## 7. What to send back

1. `report.md` and `report.json` from each output directory (no `aliases.local.json`, no
   `runs/`).
2. `curl -fsS http://127.0.0.1:8112/health` output (confirms the dtype used).
3. Peak `memory.used` from `nvidia-smi` during each run, and whether PaddleOCR-VL was stopped.
4. `docker compose ... logs --no-color teleocr | tail -n 200` if anything failed.
5. The TeleOCR model revision you downloaded and the Gemini model/prices you set.

What we will look at: how many bills are `verified` per mode; for every `flagged` check
whether the difference is a real extraction miss or a gate false alarm; agreement rates
between TeleOCR and Gemini; seconds per page; ₹ per page against the ≤ ₹2.1/page all-in
target (Gemini ≤ ₹0.30/page).

## 8. Using a mode in the demo stack

`compose.teleocr.yaml` is an overlay for the reader modes: it starts TeleOCR, makes the
worker wait for it instead of PaddleOCR-VL, and leaves PaddleOCR-VL stopped (it is not used
in these modes). Without the overlay, starting the worker also starts PaddleOCR-VL.

```bash
export GMONEY_TABLE_READER=teleocr          # or teleocr_gemini (overlay default: teleocr)
docker compose -f compose.demo.yaml -f compose.gpu.yaml -f compose.teleocr.yaml up -d
docker compose -f compose.demo.yaml -f compose.gpu.yaml stop paddleocr-vl   # if it was running
```

The worker status file reports `table_reader`; each result records `table_reader` and its
`reconciliation` report, and the review payload shows the recomputed gate.

## 9. Rollback

```bash
unset GMONEY_TABLE_READER                   # or export GMONEY_TABLE_READER=heuristic
docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile teleocr stop teleocr
docker compose -f compose.demo.yaml -f compose.gpu.yaml up -d   # no overlay: PaddleOCR-VL back
```

With the reader unset, extraction is exactly the previous heuristic path and the
reconciliation gate (default `auto`) returns to report-only for new jobs. Jobs already
extracted with a TeleOCR mode keep their recorded `enforced` flag; reprocess them with the
heuristic reader if needed.

## 10. Later: vLLM

The official repository ships `TeleOCR-vllm/` (an out-of-tree vLLM plugin, pinned to
vLLM 0.11). It may raise throughput substantially, but it is an optional later step: measure
the transformers service first, then evaluate vLLM as a drop-in behind the same
`/v1/chat/completions` contract (and confirm its output matches on the same crops).
