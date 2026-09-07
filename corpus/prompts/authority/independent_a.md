# Luna authority audit — independent A

Prompt identity: `authority-independent-a-v1`
Review kind: `independent_a`
The caller replaces every `{{...}}` value with the sealed manifest value before sending this prompt.

## Bound inputs

- `SOURCE_SHA256={{SOURCE_SHA256}}`
- `DOCUMENT_ID={{DOCUMENT_ID}}`
- `SOURCE_MANIFEST_SHA256={{SOURCE_MANIFEST_SHA256}}`
- `IMAGE_MANIFEST_SHA256={{IMAGE_MANIFEST_SHA256}}`
- `IMAGE_SHA256S={{IMAGE_SHA256S}}`
- `PAGE_MANIFEST_JSON={{PAGE_MANIFEST_JSON}}`
- `STRUCTURAL_ID_MAP_JSON={{STRUCTURAL_ID_MAP_JSON}}`

Review only the supplied page images. This is a blind, independent first reading. Do not
inspect, request, infer, summarize, or use any machine output, OCR output, prior
transcription, candidate extraction, baseline, or other reviewer result. Do not guess
blurred, occluded, clipped, or ambiguous text: use `readable: false`, leave the value
null, and put a short visual reason in `note`. Preserve printed spelling and punctuation
when readable. Never calculate, reconcile, or invent a rate, quantity, amount, subtotal,
tax, balance, or total. Record a total only when that total is visibly printed. Do not
create a row merely because arithmetic suggests one should exist.

Use image coordinates for every polygon and preserve page/table/column/row/cell order.
`STRUCTURAL_ID_MAP_JSON` is produced from the pre-gold SourceManifest geometry pass; it
is not machine transcription. Structural IDs are generated from that geometry by the
validator; do not invent or hand-compute IDs. Use only IDs supplied in the map for a
continuation or total scope. Return visible `raw_value` only: always leave
`normalized_value` null because normalization belongs to the evaluator. A missing value
is `null`, not an empty guess. Leave row numeric projections (`rate`, `quantity`,
`gross_amount`, `discount`, and `amount`) and total `normalized_value` null; preserve a
visible printed number only in its raw field.

Return exactly one JSON object and nothing else: no prose, markdown, code fence, comments,
or trailing keys. The object must have exactly these top-level keys and the literal
`gold_version` value shown below:

The caller records only the digest of this final structured JSON as the review output;
do not provide a hidden-reasoning transcript or any extra reasoning field.

```json
{
  "gold_version": "gold_document_v2",
  "source_sha256": "{{SOURCE_SHA256}}",
  "source_manifest_sha256": "{{SOURCE_MANIFEST_SHA256}}",
  "document_id": "{{DOCUMENT_ID}}",
  "page_count": 0,
  "pages": [],
  "continuations": [],
  "totals": [],
  "annotation_group_id": "{{ANNOTATION_GROUP_ID}}",
  "split": "{{SPLIT}}"
}
```

Replace `page_count` and `pages` with the complete visual transcription. Each page must
contain `source_sha256`, `page_number`, `artifact_sha256`, `width`, `height`, `dpi`,
`source_class`, `readability`, and `tables`. Each table must contain `page_number`,
`table_order`, `polygon`, `table_kind`, `readability`, `columns`, and `rows`. Each column
must contain `column_order` and `polygon`; each row must contain `row_order`, `polygon`,
`row_kind`, its printed semantic fields, and `cells`; each cell must contain
`column_order`, `polygon`, `readable`, `raw_value`, `normalized_value`, and any visible
`canonical_field` or `note`. `row_kind` must use the declared ontology (including
`detail`, `informational`, `continuation`, `section_header`, `category_rollup`,
`section_total`, `document_total`, `payment`, `deposit`, `refund`, `metadata`,
`footer_noise`, `unreadable`, or `unresolved`) when applicable. Keep
`normalized_value` null. Use `null` for unknown values. Do not add keys to this shape.
The caller validates all hashes, IDs, ordering, and numeric fields after return.
