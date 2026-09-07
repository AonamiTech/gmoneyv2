# Authority corpus workflow

Raw PDFs, 300-DPI renders, gold annotations, and review records are deliberately kept outside
Git. The default local vault is:

```text
/home/azureuser/gmoney-corpus-vault/authoritative-v1
```

The `gmoney-authority` command builds content-addressed objects and immutable control reports.
The intended sequence is:

```bash
gmoney-authority inventory --source-root /path/to/source/archive
gmoney-authority render
gmoney-authority validate-review --review-root /path/to/four-pass-reviews
gmoney-authority seal --identity /path/to/frozen-identity.json
gmoney-authority baseline --replay /path/to/replay-a --replay /path/to/replay-b \
  --identity /path/to/frozen-identity.json
```

Inventory and rendering may be used on a partial, unassigned source set. Sealing remains
fail-closed until the nested cohorts contain exactly 14, 36, and 159 documents, every document
belongs to `staging159`, every page render matches its hash, and every document has one strict
`GoldDocumentV2` plus frozen independent-A, independent-B, adjudicator, and red-team
`ReviewRecordV1` records. A single Luna visual QA pass is useful for classifying work, but is not
audited gold and cannot satisfy the seal.

The legacy `baseline`, `evaluate`, and `gate-m0-m4` control reports remain fail-closed: replay
diagnostics are explicitly marked non-authoritative, candidate files cannot supply their own
gold, and a gate requires content-addressed typed authority objects. M4 comparisons are built
with `evaluate_uvdoc_accuracy_v2` and validated with `evaluate_uvdoc_gate_v2`; a passing M4
decision means continued shadow testing only.

The prompt templates in [prompts/authority](prompts/authority) define those four review passes.
They prohibit access to machine output, guessing unreadable values, or inventing values through
arithmetic. Structural IDs are derived from immutable source/page geometry rather than from
transcription.

Never commit the vault, client PDFs, rendered pages, annotations, or job archives. Repository
reviews may record only aggregate counts, content digests, commands, and non-sensitive gate
decisions.
