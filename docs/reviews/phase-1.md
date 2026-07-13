# Phase 1 Checkpoint Review

**Decision:** PASS for Phase 2 accuracy-spine development  
**Checkpoint tag:** `checkpoint-phase-1`  
**Review date:** 2026-07-13 UTC

## Delivered

- Immutable PyMuPDF rendering with SHA-256 artifact identity.
- Page quality assessment for blur, contrast, exposure, edge density, and skew.
- Reversible orientation, deskew, perspective, and crop transforms.
- Original-page evidence coordinates with inverse mapping within two pixels.
- Isolated adapters for PP-OCRv6 Medium, PP-DocLayoutV3, SLANeXt Wireless, and PaddleOCR-VL-1.6.
- OCR-geometry table fallback for faint borderless pages missed by layout.
- Pinned CPU inference image and pinned llama.cpp heavy-parser service.
- Three-slot heavy-parser configuration with 8,192 context tokens per slot.

## Geometry Gate

All 0/90/180/270 orientation transforms, fine skew, perspective correction, crop transforms, and inverse mapping pass. Synthetic round trips are subpixel, below the two-pixel contract. Real PDF rendering was idempotent and left the source hash unchanged.

## Borderless Table Gate

PP-DocLayoutV3 was evaluated at 300 DPI on every page containing curated gold rows:

| Measure | Result |
|---|---:|
| Gold row-bearing pages | 91 |
| Layout table proposals | 89 |
| Raw page-level table recall | 97.80% |
| Layout misses | 2 |
| Misses recovered by OCR geometry | 2/2 |
| Effective page-level table recall | 100% |
| Hospital overlays visually reviewed | 12/12 accepted |

Both raw misses are readable but faint, borderless tables. They remain named regression cases; the detector threshold was not globally weakened.

This metric proves table-bearing-page coverage, not cell-IoU accuracy. Cell geometry remains grounded in PP-OCR polygons and transform evidence during Phase 2.

## Component Results

| Component | CPU result | Peak memory |
|---|---:|---:|
| PP-OCRv6 Medium | 27.6 s representative 300-DPI page | 1.92 GB process RSS |
| PP-DocLayoutV3 | 500.3 s for 91 pages; 6.39 s max/page | 1.38 GB process RSS |
| SLANeXt Wireless | 7.41 s representative crop | 0.99 GB process RSS |
| PaddleOCR-VL-1.6 | 84.3 s representative crop | 1.79 GiB observed server memory |
| PaddleOCR-VL concurrency 3 | 246.1 s wall time for three simultaneous crops | 1.55 GiB observed after run |

The three heavy requests occupied separate llama.cpp slots and returned identical 314-token structured table results. Concurrency 3 is therefore functionally safe. On this four-vCPU host it improves parallel completion rather than total throughput; the 16-vCPU D16ds v6 must remeasure latency when the image is deployed there.

## Architecture Review

- PaddlePaddle 3.3.1 failed on CPU in oneDNN/PIR. The image pins 3.2.2, the supported working line for PaddleOCR CPU inference.
- The full PP-StructureV3 wrapper was rejected for this service because it loads a redundant document-layout stack. The isolated SLANeXt Wireless table model supplies structure proposals after PP-DocLayoutV3 supplies crops.
- SLANeXt cell polygons are not accepted as source evidence. Its structure tokens are fused with OCR polygons instead.
- PaddleOCR-VL uses the official `Table Recognition:` element prompt and returns explicit row/cell markers.
- Model weights and real bills are mounted artifacts, not image layers.
- The inference image runs as UID 10001 and contains no local data or secrets.

## Images

| Image | Digest/ID |
|---|---|
| CPU Paddle worker | `sha256:a8621a5d3eed7d99fe54ebef190ae79414d1051d827ec45b07ea7367984c9146` |
| llama.cpp server | `sha256:84d3d48839dc645bd85affa2738f447179fbda4e356c5b40942cf2a9463fe779` |

## Verification

```bash
scripts/check_phase1.sh
sudo docker run --rm --entrypoint python gmoney-v2-inference-cpu:phase1 \
  -c "import cv2,paddle,paddleocr; print(cv2.__version__,paddle.__version__,paddleocr.__version__)"
curl --fail http://127.0.0.1:8111/health
```

## Remaining Deployment Measurement

The current host has four vCPUs and 15 GiB RAM. It is a conservative functional/memory canary, not the requested D16 latency result. Run the same frozen component benchmarks on Standard_D16ds_v6 before accepting production capacity numbers. This does not block Phase 2 accuracy development because time is not currently an acceptance constraint and all models fit safely.

## Next Phase Authorization

Phase 1 implementation and accuracy-relevant gates pass. Proceed automatically to Phase 2. Do not promote any production latency claim until the D16 report is attached.
