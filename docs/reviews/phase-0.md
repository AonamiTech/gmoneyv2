# Phase 0 Checkpoint Review

**Decision:** PASS  
**Checkpoint tag:** `checkpoint-phase-0`  
**Review date:** 2026-07-13 UTC

## Delivered

- Clean V2 Git repository and pinned Python dependency lock.
- FastAPI health surface and a runnable API image.
- PostgreSQL, MinIO, Temporal, Temporal UI, and API Compose stack.
- Strict versioned gold, evidence, transform, provider-candidate, and canonical-row contracts.
- Frozen `legacy_metric_v1` and independent `canonical_metric_v2` one-to-one row matchers.
- Hash-based corpus catalog with hospital-grouped train/validation/holdout splits.
- Synthetic evaluator and contract regression fixtures.
- Frozen legacy baseline registry.

No legacy schema, profile, layout band, rule pack, activation state, or confidence value is imported.

## Verification Results

| Gate | Result |
|---|---|
| Unit/API/contract/evaluator tests | 10 passed, warnings treated as errors |
| Ruff lint | Passed |
| Compose configuration | Passed |
| Compose runtime | PostgreSQL and MinIO healthy; Temporal `SERVING`; API live and ready |
| Gold annotations validated | 24 |
| Gold rows validated | 1,549 |
| Dataset split | 16 train / 4 validation / 4 holdout bills |
| Hospital leakage | None |
| Frozen baseline arithmetic | 327/688 = 47.53% precision; 327/523 = 62.52% recall |
| Evaluator determinism | Passed |

Split row counts are 1,070 train, 70 validation, and 409 holdout. The holdout remains untouched by extraction tuning after this checkpoint.

## Reproduction

```bash
make install
scripts/check_phase0.sh
sudo docker compose up -d --build
sudo docker exec gmoney-v2-temporal-1 \
  temporal --address temporal:7233 operator cluster health
curl --fail http://127.0.0.1:8000/health/ready
```

## Container Evidence

| Image | Image ID/digest |
|---|---|
| `gmoney-v2-api` | `sha256:501593a6a1142019a064e954fdfaf97f810634e87b9585deb7d600a4cfe3597b` |
| `postgres:16.6-alpine` | `sha256:1d04b9ba1d4996401f2552b51beda8187f175c0645c091e4781134fc9c9a3eef` |
| `minio/minio:RELEASE.2025-04-22T22-12-26Z` | `sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e` |
| `temporalio/auto-setup:1.27.2` | `sha256:b44cbfeb43dbeae42db113b44fb8414c3452f05643b3d6b1592f955277d73526` |
| `temporalio/ui:2.34.0` | `sha256:cb17ea423d76a8a19a269d0bcd81fc12eee1f6365acd2a56b590dafb35696a95` |

## Review Notes

- The original remote 39-bill lifecycle run directory referenced by the legacy review is not present on this host. The reported baseline counts and metrics are therefore frozen and tested from the review evidence, but cannot be replayed row-by-row from that missing run directory.
- The available curated corpus contains 24 gold annotations. Existing gold has page and row content but not complete polygons/table IDs; `canonical_metric_v2` supports those fields and will use them as new annotations are added.
- Real PDFs, generated catalog data, model weights, and artifacts are Git-ignored.
- The current Compose stack is a development checkpoint, not a production deployment.

## Next Phase Authorization

All Phase 0 gates available in this environment pass. Proceed automatically to Phase 1 geometry and component bake-offs.
