# Demo evidence workspace verification

Date: 2026-07-14

## Outcome

The client demo now identifies each completed document by an OCR-grounded hospital
name, separates service dates from charge descriptions, filters contact metadata from
charge rows, and keeps the ledger and source page usable at desktop and narrow widths.
Hospital corrections are stored in the revisioned review overlay and are included in
JSON, CSV, and evidence exports without modifying the machine result.

The evidence workspace now has an independently scrolling ledger, a keyboard- and
pointer-resizable split, page and width fit modes, 75–300% zoom, evidence focus, and
fullscreen. This removes the table overflow that previously covered structural review.

## Verification

Local verification passed:

- 119 Python tests
- Ruff and whitespace checks
- frontend ESLint, TypeScript, and production Next.js build

The final candidate was replayed in two seven-core lanes on D16 against all eleven
retained Sample Bill inference caches: 163 pages, 5,460 accepted rows, zero accepted
rows without description and amount evidence, and zero Gemini calls. It extracted 672
service dates and left zero date-prefixed descriptions. Bill 10 now has 76 true charge rows
after removing four item-total/payer-credit footer fragments; Bill 11 has 35 after removing
the contact-phone footer. Exact shared-description evidence also removes partial duplicates
from overlapping layout proposals without fuzzy collapsing printed rows.

The quantity/rate regression covers repeated shifted headers, OCR-merged
`Quantity UnitPrice`, `Rate Qty`, expiry-plus-quantity, and concatenated numeric cells.
All 1,822 rows that expose both quantity and unit price reconcile to their printed amount;
the orientation scan found no quantity above 100 paired with a unit price at or below 10.
Bill 10 exposes both fields on 75 of 76 rows, and Bill 11 exposes them on all 35 rows.

The annotated four-bill exposed regression also passed:

| Metric | Result |
|---|---:|
| Legacy precision / recall / F1 | 95.33% / 98.08% / 96.68% |
| Canonical precision / recall / F1 | 94.39% / 97.12% / 95.73% |
| Amount accuracy | 100% |
| Negative recall | 100% |
| Accepted ungrounded rows | 0 |

The original 809 retained Sample Bill artifacts remain unchanged; their checksum
manifest still hashes to
`88550a2143e2acc4594b03b751ccca36832ea86799e8504bb6b8c3df9c6f2ccd`.
The latest reconstruction is preserved on D16 at
`/home/azureuser/gmoneyv2-candidate-history/candidate-artifacts`: it hard-links those
unchanged files and adds four Bill 11 recovery artifacts, for 813 verified files under
manifest SHA-256
`c0f68026f181b4c5286f4c6223b4e58f730d28a27503f64c0a5a3216560420b6`.

An importer smoke test hard-linked that complete archive into a temporary demo store,
published all 11 original filenames, returned the Kamakshi bill by hospital search,
reported a 30-day expiry for every bill, and retained the same page-artifact inode.
A second import skipped all 11 tombstoned documents. The same store is now public on port
3100 and retained all 11 entries across a service restart. The temporary root-disk fallback
has 15 GiB free and rejects new uploads below 3 GiB; the 1 TiB managed disk is still required
before scaling this retention model toward 5,000 bills.

## Quality boundary

The Sample Bills and four annotated documents are exposed regression data. These
results validate this demo change but do not satisfy the blocked hospital-disjoint
unseen checkpoint described in the Phase 3 review.
