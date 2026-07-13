# Phase 3 implementation review

Date: 2026-07-13

## Outcome

The Phase 3 implementation is complete and tested, but the Phase 3 checkpoint is
blocked by the deliberately non-overridable data gate. The eleven unexposed Sample
Bill candidates do not yet have confirmed hospital identities or frozen gold, and
there is no independent repeat-layout holdout with at least ten bills and 200 rows.
No `checkpoint-phase-3` tag is permitted until those datasets pass the gate.

## Delivered

- Crop-scoped 300-to-400 DPI rerender, CLAHE input, crop OCR, PaddleOCR-VL,
  optional Gemini, and review escalation for zero and implausibly low yield.
- Recovery traces and provider failure handling that preserve successful local
  work and route timeouts, invalid responses, redaction failures, and exhausted
  budgets to review.
- De-identified Gemini Developer API challenger with schema-constrained output,
  content/config-bound caching, field/token/polygon/column grounding, exact printed
  numeric validation, call/cost limits, and a document-local circuit breaker.
- Frozen promotion decisions. Enabled Gemini mode rejects missing, unsafe, stale,
  or configuration-mismatched promotion artifacts; challenger mode cannot alter
  canonical output.
- Declarative layout profile construction, retrieval threshold and margin, known
  fast route, isolated shadow evaluation, deterministic 5% heavy-route sampling,
  activation gates, drift pause, atomic version transitions, and rollback history.
- Phase 3 corpus sealing, cohort evaluation, deterministic document bootstrap,
  aggregate/unseen/known-layout gates, privacy and grounding gates, active-route
  zero-Gemini measurement, and Gemini champion/challenger promotion tooling.
- Stable canonical row IDs across reprocessing and explicit removal of document
  totals, balance lines, and advances from the accepted detail ledger.

The provider choice is recorded in `docs/adr/0001-phase3-gemini-developer-api.md`.
The Developer API is approved only for de-identified preproduction challenger
evaluation; it is not approved here for production PHI.

## Accuracy evidence

The valid cached Motherhood regression was replayed after the final row-policy
changes. It contains two exposed bills, 296 gold rows, and is an implementation
regression rather than unseen certification.

| Metric | Result | Gate |
|---|---:|---:|
| Legacy precision / recall / F1 | 98.32% / 98.65% / 98.48% | 85% |
| Canonical precision / recall / F1 | 97.98% / 98.31% / 98.15% | 85% |
| Amount accuracy | 99.32% | 95% |
| Negative-row recall | 100% | 100% |
| Accepted ungrounded rows | 0 | 0 |

The committed Phase 2 review also records a four-bill exposed replay at 95.99%
canonical precision and 97.84% canonical recall. The two old Prapti drafts were not
reused in this final replay because they label the wrong financial column and omit
visible rows, as documented in the Phase 2 review.

The sealed-candidate manifest reserves eleven documents and marks Sample Bill 12 as
exposed. It reports zero confirmed distinct hospitals and zero frozen gold documents,
so it correctly blocks any claim of ten-hospital unseen performance. No profile was
activated without the independent repeat-layout holdout.

## Verification

- Local: 91 Python tests passed; Ruff passed.
- D16 isolated container: the same 91 tests and Ruff passed with 16 CPUs and a
  56 GiB ceiling.
- Frontend ESLint and TypeScript checks passed.
- All three Compose files passed configuration validation.
- Secret-pattern scan found no committed API key or private key material.
- The legacy nginx service on port 3000 remained running and was not modified.

### D16 profile scale

The repeatable `gmoney-profiles benchmark` command exercised 20,000 variants and
100 matches:

- profile construction: 0.092 seconds;
- mean lookup: 13.56 ms;
- maximum process RSS: 72.27 MiB;
- correct selected profile: `profile-19999`.

This covers the expected 5,000 hospitals times three to four variants without
executable per-hospital parsers.

### D16 extraction canary

Sample Bill 12 was processed in an isolated work directory with the cached
PP-OCRv6, PP-DocLayoutV3, and PaddleOCR-VL models:

- three pages and 90 final grounded rows;
- zero accepted rows missing description or amount evidence;
- zero Gemini calls;
- OCR/layout process memory sampled at 1.73 GiB;
- VLM server memory sampled at 1.59 GiB;
- one heavy call took 71.45 seconds and contributed no accepted rows;
- two complete cached replays took 15.14 seconds total;
- cached semantic rows and stable row IDs were identical, excluding only the
  expected per-run `created_at` timestamp.

## Blocking gate

`corpus/phase3-sealed-candidates.json` must first be completed with identity-reviewed,
hospital-disjoint, frozen annotations for at least ten eligible hospitals. A separate
profile construction/holdout cohort must then provide at least ten holdout bills and
200 populated gold rows. Run local-only, Gemini challenger, and profile-fast outputs
through `gmoney-phase3-evaluate`; enable Gemini only through a passing
`promote-gemini` decision. Create the checkpoint tag only when every reported gate is
green.
