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

From the release directory on the GPU host:

```bash
export GMONEY_IMAGE_TAG=<release-id>
export GMONEY_DATA_ROOT=/home/ubuntu/gmoneyv2-runtime
export GMONEY_MODEL_ROOT=/home/ubuntu/gmoneyv2-runtime/model-cache
export GMONEY_PROFILE_ROOT=/home/ubuntu/gmoneyv2-runtime/profiles
export GMONEY_MIN_FREE_BYTES=10737418240
export GMONEY_MAX_UPLOAD_BYTES=0
export GMONEY_GPU_WORKER_CONCURRENCY=1
install -d -o 10001 -g 10001 "$GMONEY_DATA_ROOT/jobs" "$GMONEY_DATA_ROOT/config"
docker compose -f compose.demo.yaml -f compose.gpu.yaml up -d --build
```

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
mounted read-only into the API and supplies the trained-hospital directory.

## Acceptance

- Python tests, Ruff, frontend lint/typecheck/build, and merged Compose
  validation pass for the exact release source.
- Host and disposable CUDA-container `nvidia-smi` checks pass. Paddle reports a
  CUDA build, sees GPU 0, and loads PP-OCRv6 plus PP-DocLayoutV3 on `gpu:0`.
- llama.cpp reports CUDA initialization and GPU layer offload; both model and
  worker processes appear in `nvidia-smi` without exhausting VRAM.
- `/api/v2/health/ready` reports ready with one worker lane, the public UI and
  health path return HTTP 200 on port 3100, and ports 8100/8111 are unreachable
  remotely while remaining healthy through loopback.
- A real PDF upload reaches `complete`, produces grounded rows and page evidence,
  survives a Compose restart, and records `gpu:0`/`cuda:0` in newly written
  inference cache model specifications.
- No container is OOM-killed or restarted, no fatal/error marker appears in
  application logs, and at least 10 GiB remains free after images/models/build
  cache cleanup.
