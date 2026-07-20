# GPU Sample Bills validation

Date: 2026-07-15
Batch: `gpu-sample12-20260715T041552Z`

## Outcome

All 12 PDFs in `Sample Bills/` completed on the Tesla T4 deployment and remain
available at `http://34.180.11.221/`. The run processed 166 pages into 5,551
grounded rows. The 17-minute batch wall time includes the canary, uploads, queue
time, and the upload-limit correction described below.

| Document | Pages | Rows |
| --- | ---: | ---: |
| Bill 1.pdf | 30 | 1,332 |
| Bill 2.pdf | 31 | 1,568 |
| Bill 3.pdf | 8 | 145 |
| bill 4.pdf | 43 | 1,434 |
| Bill 6.pdf | 4 | 12 |
| Bill 7.pdf | 13 | 198 |
| Bill 8.pdf | 9 | 135 |
| Bill 9.pdf | 9 | 217 |
| Bill 10.pdf | 4 | 76 |
| Bill 11.pdf | 2 | 35 |
| Bill 12.pdf | 3 | 91 |
| Bill 13.pdf | 10 | 308 |
| **Total** | **166** | **5,551** |

There is no `Bill 5.pdf` in the supplied directory; the twelfth input is
`Bill 13.pdf`.

## Concurrency decision

The deployed setting is two bill workers and one VLM slot. A concurrent
Bill 11/Bill 12 canary peaked at 9,969 MiB of GPU memory. The complete batch
peaked at 13,223 MiB against the agreed 14,000 MiB safety gate, reached 100%
GPU utilization, and stayed at or below 75 C. No container restarted or was
OOM-killed, and the post-run log scan found no traceback, fatal, OOM, or CUDA
error marker.

Two bill workers is therefore the correct setting for this 15,360 MiB T4. Keep
one VLM slot and the current 300 DPI page, 400 DPI recovery-crop, and token
limits. More per-page memory would not improve the already saturated GPU, while
a second VLM slot would remove too much VRAM headroom. A third bill worker would
mostly wait for the single VLM slot and increase pressure without demonstrated
throughput benefit.

## Integrity and regression checks

The validator checked all 12 jobs, 166 pages, 5,551 rows, and 803 artifacts
totalling 1,257,091,088 bytes. Every source and page hash matched; every row had
evidence plus description/amount evidence; all evidence references were valid.
New artifacts identify `gpu:0` for Paddle and `cuda:0` for the local VLM. The
run made 17 local VLM calls and no Gemini calls.

The 11 inputs that have a retained reconstruction baseline produced exactly
5,460 rows again. Bill 12 produced 91 rows versus 90 in an earlier canary. The
sample set has no frozen gold labels, so these checks establish completion,
routing, grounding, and artifact integrity—not precision or recall.

## Operational notes

The original 25 MiB upload guard rejected Bills 1, 2, and 4 before processing.
Their empty placeholders were removed, the API limit was explicitly raised to
50 MiB, and the unchanged PDFs were accepted and completed. The repository now
passes that setting through Compose.

All results survived a full five-service Compose restart. The API returned to
ready with capacity two and zero active jobs, the public UI and readiness route
returned HTTP 200, and ports 8100 and 8111 remained inaccessible externally.
The source staging directory was deleted after checksum and result validation;
about 23.4 GB remained free, above the 10 GiB storage floor. Completed jobs use
the configured 720-hour retention period.

The machine-readable manifest and exact job IDs are in
[`gpu-sample12-summary.json`](gpu-sample12-summary.json).
