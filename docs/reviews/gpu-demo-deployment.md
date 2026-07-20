# GPU demo deployment review

Date: 2026-07-15

## Outcome

The fresh GPU demo is deployed on `gsevenlabs` at `http://34.180.11.221/`.
Only nginx is public. The API and llama.cpp diagnostics remain bound to loopback
ports 8100 and 8111. The host also binds the planned demo port 3100, but the GCP
VPC firewall filters that port; standard HTTP port 80 was already admitted and is
used for the reachable URL. The VM service account has insufficient OAuth scopes
to inspect or change VPC firewall rules.

This is intentionally an unauthenticated, plain-HTTP demo and is not a production
PHI boundary.

## Target and release

- GCE project/zone: `project-4adfa3d7-395b-4a98-b44`, `asia-south1-b`.
- Compute: 8 vCPUs, 54 GiB RAM, one Tesla T4 with 15,360 MiB VRAM.
- OS: Ubuntu 22.04.5 LTS, kernel `6.8.0-1063-gcp`.
- NVIDIA driver: `610.43.02`; NVIDIA Container Toolkit: `1.19.1`.
- Docker Engine: `29.6.1`; Compose: `5.3.1`.
- Release: `/home/ubuntu/gmoneyv2-releases/409a538740f5-gpu-028401830bf2`.
- API image: `sha256:6e62fbc992335b3ed5e5963cf9de765b0c697318d32abb08ead3ba21e24feddc`.
- Frontend image: `sha256:4fe90bed4ca0fae3eebf3f6e1613a24e41654ecd7f36903a68ee63246f714d93`.
- GPU worker image: `sha256:26b6e05778f064817d52f0818c784d4c2929c470c17f0997adf6a6243183aa59`.
- llama.cpp CUDA image is pinned to digest
  `sha256:b57dce073940d6f347d59230d4d28fc947db8948f2eb162326008380b07bab77`.

The release is a snapshot of base commit `409a538740f5` plus the current GPU
deployment changes. Those changes remain uncommitted in the source worktree and
should be committed before a later promotion needs Git-only reproducibility.

## Verification

- Full Python suite: 120 tests passed. Ruff, frontend ESLint, TypeScript, Next.js
  production build, lock validation, merged Compose validation, and shell syntax
  checks passed.
- The host and a disposable CUDA container both passed `nvidia-smi`. Inside the
  deployed worker image, PaddlePaddle GPU 3.2.2 reported `compiled_cuda=True`,
  selected `gpu:0`, and identified the Tesla T4.
- llama.cpp loaded the PaddleOCR-VL 1.6 model with one 24,576-token slot. During
  the canary, combined GPU use reached about 6.1 GiB and 100% utilization without
  OOM. New inference artifacts recorded `gpu:0` for Paddle and `cuda:0` for VLM.
- A two-page real bill completed with 10 rows and grounded page polygons/token
  evidence. It ran from `03:13:12Z` to `03:14:35Z`, including first-time model
  downloads. The completed result survived a full Compose restart.
- After restart, all five containers were running, API and VLM were healthy,
  restart counts were zero, `OOMKilled` was false, and application logs contained
  no traceback, fatal, OOM, or CUDA-error marker.
- The canary was deleted after verification; the deployed job index is empty as
  required. Paddle model downloads remain in the persistent model cache.
- External UI and readiness requests return HTTP 200. External connections to
  ports 8100 and 8111 time out, while their loopback health checks pass.
- Build cache cleanup left about 24 GiB free on the 49 GiB root filesystem, above
  the configured 10 GiB upload rejection floor.

## Model integrity

- `PaddleOCR-VL-1.6-GGUF.gguf`:
  `f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8`.
- `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`:
  `204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a`.

The initial runtime was deliberately limited to one worker/VLM lane so both
model stacks could share the T4 safely. A subsequent 12-document validation
established two bill workers with one VLM slot as the safe operating point and
raised the explicit upload limit to 50 MiB. See
[`gpu-sample12.md`](gpu-sample12.md) for the measurements and result manifest.
Add authentication/TLS and durable storage before using this deployment for
retained production traffic.

The later public upload-path correction is recorded in
[`gpu-upload-path-fix.md`](gpu-upload-path-fix.md). It removes Nginx request
throttling, aligns the proxy envelope with the GPU API's 50 MiB file limit, and
proves that the original 12 retained jobs survive deployment and refresh.
