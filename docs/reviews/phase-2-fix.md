# Phase 2 recovery implementation

Date: 2026-07-13

## Outcome

Phase 2 no longer depends on OTSL/VLM serialization as its sole row source. The
primary route is now crop-scoped OCR geometry with printed-field evidence. The
VLM is an optional description/schema advisor and cannot overwrite the printed
amount or publish an unsupported field.

## Implemented fixes

- Added `canonical_row_v2` with page/table types, row roles, per-field evidence,
  and source-route provenance.
- Reconstructed rows directly from OCR tokens using row geometry, stable numeric
  lanes, rightmost printed-amount selection, multiline descriptions, and schema
  continuity over two continuation pages.
- Normalized and deskewed 90-degree pharmacy attachments before row grouping,
  while retaining the original page polygons in evidence.
- Fused layout detections with OCR-geometry proposals on every page. OCR geometry
  expands partial layout boxes and recovers row-bearing pages with no layout box.
- Searches the complete provider table for a header, including `Test Name`,
  instead of checking only the first three rows.
- Treats provider values as proposals. Description and amount must each have OCR
  token evidence before a row can be accepted; unsupported optional fields are
  blanked.
- Replaced arithmetic cross-page deletion with the finest-available role policy:
  suppress a category rollup only when item rows cover that category, when the
  rollup exactly repeats a detailed row, or when the category amount is zero.
- Preserved genuine zero-valued detail lines and negative/refund lines.
- Changed the heavy parser to one full-context slot. Cache identity now includes
  prompt/runtime/tiling options; responses record finish reason, repetition, and
  truncation signals.
- Rotates sideways crops into the OCR-selected reading direction before VLM
  inference. A truncated or structurally empty response is retried as overlapping
  vertical tiles with separate cache identities.
- Added explicit Phase 2 quality gates for legacy and canonical precision,
  recall, F1, amount accuracy, negative recall, and accepted ungrounded rows.

## Runtime policy on D16

- OCR/layout: two workers at eight CPU cores each during batch validation. The
  workers use about 2 GiB each; increasing their RAM ceiling does not increase
  throughput because they are CPU-bound.
- VLM: one 24,576-token slot, 14 threads, 16,384 maximum output tokens, and a
  56 GiB memory ceiling. This fixes the failed three-slot configuration that
  reduced each request to 8,192 context tokens.
- The VLM runs only when deterministic reconstruction has no trustworthy rows,
  lacks both a header and inherited schema, or detects a sideways table needing
  description repair.

## Evaluation protocol

The 24 annotated bills and 12 sample bills are preproduction data. Hospital
groups remain isolated. H560077 and H422101 are regression data. H560041 and
H110005 began as the internal held-out gate, but their outputs were inspected
during this recovery and they must now be treated as exposed regression data.
Certification requires 10 hospitals that were not used for development.

Hard gates:

- legacy precision, recall, and F1: at least 85%
- canonical precision, recall, and F1: at least 85%
- amount accuracy on description-matched rows: at least 95%
- negative-row recall: 100%
- accepted rows without description and amount evidence: zero

## Verified Phase 2 results

The deterministic reconstruction was replayed from the raw OCR/layout caches;
the result therefore measures the new extraction logic against the same model
observations that produced the failed run.

| Gate | Legacy P/R/F1 | Canonical P/R/F1 | Amount accuracy | Negative recall | Result |
|---|---:|---:|---:|---:|---:|
| Exposed held-out replay, 4 bills / 416 gold rows | 96.23% / 98.08% / 97.14% | 95.99% / 97.84% / 96.90% | 100% | 100% | pass |
| Motherhood regression `1782547800688` | 97.00% / 100% / 98.48% | 96.00% / 98.97% / 97.46% | 100% | 100% | pass |
| Motherhood regression `1782552821660` | 98.98% / 97.99% / 98.48% | 98.98% / 97.99% / 98.48% | 98.98% | 100% | pass |

Every accepted row in these runs has both description and printed-amount token
evidence. All four held-out bills pass individually as well as together. Because
they were inspected while fixing the engine, this is an implementation regression
gate, not unseen-layout certification. The release gate still requires frozen
annotations from 10 new hospital groups and cannot yet claim production
generalization to all 30–40 variants.

The old Prapti gold drafts cannot be used as an accuracy authority without
re-annotation. They frequently label MRP instead of the rightmost printed
Amount and omit visible pharmacy rows. For example, the printed PANSPED row is
`MRP 237.65 / Amount 158.43`, while the old draft labels `237.65`. The v2 policy
keeps `158.43`; optimizing to the old value would reintroduce the amount-column
bug. These drafts must be corrected and versioned rather than silently changed.

## Scale path for 5,000+ bills and 30–40 variants

The extraction logic is hospital-neutral: schemas are inferred per document and
expire after two pages. New variants therefore enter through evaluation and page
classification rather than hospital-specific rules. Batch scaling should shard
documents across OCR workers, keep deterministic caches content-addressed, and
reserve the single heavy slot for advisor fallbacks. Production release remains
blocked until the 10-new-hospital gate passes with frozen annotations.
