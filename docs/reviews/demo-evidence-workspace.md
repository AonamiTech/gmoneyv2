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

- 109 Python tests
- Ruff and whitespace checks
- frontend ESLint, TypeScript, and production Next.js build

The final candidate was replayed in two seven-core lanes on D16 against all eleven
retained Sample Bill inference caches: 163 pages, 5,506 accepted rows, zero accepted
rows without description and amount evidence, zero amounts above ₹10 crore, and zero
Gemini calls. It extracted 681 service dates and left zero date-prefixed descriptions.
Only one row was removed relative to the prior replay: the Bill 11 contact-phone footer.
Bill 10 retained 80 rows and Bill 11 retained 35 rows.

The annotated four-bill exposed regression also passed:

| Metric | Result |
|---|---:|
| Legacy precision / recall / F1 | 96.23% / 98.08% / 97.14% |
| Canonical precision / recall / F1 | 95.99% / 97.84% / 96.90% |
| Amount accuracy | 100% |
| Negative recall | 100% |
| Accepted ungrounded rows | 0 |

The original 809 retained Sample Bill artifacts remain unchanged; their checksum
manifest still hashes to
`88550a2143e2acc4594b03b751ccca36832ea86799e8504bb6b8c3df9c6f2ccd`.

## Quality boundary

The Sample Bills and four annotated documents are exposed regression data. These
results validate this demo change but do not satisfy the blocked hospital-disjoint
unseen checkpoint described in the Phase 3 review.
