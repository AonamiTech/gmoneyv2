# Phase 2 Run Issues

## Run outcome

Phase 2 failed the mandatory accuracy gate on the sealed unseen-layout holdout.

| Evaluator | Gold rows | Emitted rows | Matched rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `legacy_metric_v1` | 409 | 93 | 44 | 47.31% | 10.76% | 17.53% |
| `canonical_metric_v2` | 409 | 93 | 41 | 44.09% | 10.02% | 16.33% |

The gate required at least 85% precision and 85% recall under both evaluators.

## Per-document results

| Bill | Gold | Emitted | Legacy precision/recall | Canonical precision/recall |
|---|---:|---:|---:|---:|
| `1782547800688` | 97 | 9 | 0.00% / 0.00% | 0.00% / 0.00% |
| `1782552821660` | 199 | 29 | 100.00% / 14.57% | 100.00% / 14.57% |
| `9021114986` | 66 | 48 | 31.25% / 22.73% | 25.00% / 18.18% |
| `9890913867` | 47 | 7 | 0.00% / 0.00% | 0.00% / 0.00% |

The run produced 365 false negatives and 49 false positives under the legacy evaluator. The
canonical evaluator produced 368 false negatives and 52 false positives.

## 1. VLM output format was inconsistent

The row parser expects OTSL table markers from PaddleOCR-VL, but the model did not consistently
return that format.

- The run processed 31 table crops.
- Only 13 crops contained any parseable OTSL rows.
- Eighteen crops returned prose or flattened line text without OTSL structure.
- Only seven crops produced candidate rows.
- Only six crops produced canonical detail rows.

When the VLM response did not contain OTSL, the extraction path returned zero rows even when the
table crop and OCR tokens were available and readable.

This caused entire pages to disappear from the output:

- `1782552821660` emitted 29 rows from page 1 and zero rows from pages 2–7.
- `1782547800688` emitted only page-1 rows; all gold rows were on pages 2–5.
- `9890913867` emitted only page-1 rows; all gold rows were on pages 2–6.
- `9021114986` emitted rows only from pharmacy pages 4–6.

## 2. There was no OCR-native row reconstruction fallback

PP-OCR tokens were produced for every processed page, but the implemented pipeline used them
primarily to ground rows already parsed from VLM output. OCR tokens were not independently
converted into rows when VLM structure was absent or malformed.

As a result, a readable page could have:

- a valid OCR response;
- a valid table crop;
- a valid layout response; and
- a non-OTSL VLM response;

but still produce zero candidate rows.

## 3. Dense tables exceeded the VLM context available per slot

The llama.cpp service used a total context of 24,576 tokens with three parallel slots. Each slot
therefore had approximately 8,192 tokens available.

Three dense crops reached exactly 8,192 total tokens and were truncated:

- `1782552821660`, page 4
- `9890913867`, page 4
- `9890913867`, page 5

The truncated responses ended with incomplete or repetitive content rather than a complete table.
Examples included repeated pharmacy lines and long sequences of repeated zero values.

The three-slot configuration was memory-safe, but it was not context-safe for dense tables.

## 4. Three-way VLM concurrency caused severe latency contention

The D16 host remained stable, but dense table requests slowed substantially under sustained
three-slot inference.

| VLM crop latency | Result |
|---|---:|
| Minimum | 64.2 seconds |
| Median | 185.4 seconds |
| p95 | 678.5 seconds |
| Maximum | 772.6 seconds |

Some dense crops generated more than 4,000 tokens and took approximately 10–13 minutes. Memory
was not the bottleneck: approximately 51–52 GiB remained available and no swap or OOM event
occurred.

## 5. Header recognition was too narrow

The parser depended on finding a recognized description header in the first three parsed rows.
Several legitimate schemas were not recognized.

Examples:

- Pathology tables used `Test Name`, which was not recognized as a description column.
- Continuation pages did not always repeat the header.
- Some headers were collapsed into a single VLM cell.
- Some page metadata appeared before the actual header and pushed it outside the inspected rows.
- Some tables had their header content merged into an earlier prose row.

Readable pathology pages containing rows such as `HISTOPATHOLOGY - BIOPSY`, `CBC`, and
`Blood Group` consequently produced zero candidates.

## 6. Schema state was not carried across pages

Each page and table was parsed independently. The engine did not retain a document-level schema
from an earlier page for use on headerless continuation pages.

This particularly affected long bills whose page 1 schema was recognized but whose subsequent
pages contained the same columns without a complete repeated header. The result was high page-1
accuracy followed by zero recall on later pages.

## 7. The wrong financial column was selected

On `1782547800688` page 2, the VLM returned 22 valid-looking item rows with these financial
values:

```text
Rate | Qty | 0.00 | rightmost line total
```

The header labelled the zero-discount column as `Amount`, while the actual line total was in an
unlabelled rightmost column. The parser selected the zero column as the canonical amount.

Consequences:

- All 22 item rows received amount `0.00`.
- Zero-value detail rows were reclassified as metadata.
- All 22 rows were discarded from canonical output.

## 8. Shifted VLM values survived failed OCR grounding

On pharmacy tables, several VLM amounts were associated with the preceding or following item.
The spatial grounding stage could not find those amounts on the same OCR row, but the original
VLM value remained on the canonical candidate.

This created rows with a mostly correct description but the amount from a neighboring item. For
example, `9021114986` pages 4–6 emitted 48 rows, but only 15 matched under the legacy evaluator
and 12 under the canonical evaluator.

The grounding result marked some rows as insufficiently grounded, but that status did not prevent
the rows from appearing in machine output or being counted as extraction results.

## 9. Summary tables were classified as item details

Two documents emitted rows from page-1 charge summaries even though the gold policy treated the
later item-ledger pages as the required output.

- `1782547800688` emitted nine page-1 summary rows and none of its 97 gold item rows.
- `9890913867` emitted seven page-1 numbered summary rows and none of its 47 gold item rows.

The row-role rules classified these summary/category entries as detail rows because they had a
description and an amount. There was no reliable page/table-level distinction between a charge
summary and an item ledger.

## 10. Some summary descriptions were parsed from the serial-number column

The seven rows emitted from `9890913867` page 1 had descriptions such as `2.`, `3.`, and `4.`.
The parser treated the serial-number column as the description column while retaining amounts
from other cells in the row.

These rows were structurally invalid but still entered canonical output as detail rows.

## 11. OTSL row and column alignment was unstable

Some VLM outputs contained recognizable OTSL but had one or more of these structural problems:

- data collapsed into one cell per row;
- empty span cells inserted between fields;
- header and metadata rows represented with different widths;
- a missing cell shifting every later value left or right;
- column names merged into a prose cell;
- trailing line totals placed in an unlabelled column.

The current parser assumes sufficiently consistent column indexes after header inference. Once a
cell was missing or merged, subsequent descriptions and financial values could be assigned to the
wrong fields.

## 12. OCR recognition errors reduced matches on emitted rows

OCR and/or VLM text recognition produced product substitutions such as:

- `CEFTUM 500` instead of `CEFTRIA 500`
- `VENFLON NO 20` instead of `SURGIVENT NO 20`
- `VELFIX` instead of `VEEFIX BD`
- `METROGYL ER 600 TAB` instead of `AZOGYL ER 600 TAB`

These errors reduced precision on the pharmacy pages. They were significant but secondary to the
larger zero-row and financial-alignment failures.

## 13. Validation coverage did not represent the holdout complexity

Before the holdout, three development bills scored 100% under both evaluators:

- 9-row training bill
- 7-row validation bill
- 33-row validation bill

The holdout introduced much longer documents and denser schemas:

- up to 199 gold rows in one bill;
- multi-page continuation tables;
- summary pages followed by detail pages;
- pathology, hospital-charge, pharmacy, and consumable schemas in the same document set;
- crops requiring more output tokens than one parallel VLM slot allowed.

The development set therefore did not expose the main failure modes before the extraction commit
was frozen.

## 14. The exposed holdout can no longer serve as a sealed gate

The four holdout outputs and gold rows were inspected during this failure analysis. They are now
known regression cases and cannot be reused as an unseen-layout certification set without causing
evaluation leakage.

Any later accuracy-gate run will require a new hospital-grouped, unseen holdout.

## Infrastructure observations

The following were not failure causes:

- All 23 gold-row-bearing pages received at least one table crop.
- All 31 crops received a VLM response.
- Three documents remained active concurrently without OOM or swapping.
- Runtime memory remained stable.
- No corrupted or partially written machine-output file was observed.
- The legacy application remained healthy on port 3000.

The Phase 2 failure was therefore primarily an extraction architecture and generalization issue,
not a host-capacity, table-localization, or process-reliability issue.
