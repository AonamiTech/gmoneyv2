# Authority corpus workflow

Raw PDFs, 300-DPI renders, gold annotations, and review records are deliberately kept outside
Git. The default local vault is:

```text
/home/azureuser/gmoney-corpus-vault/authoritative-v1
```

The `gmoney-authority` command builds content-addressed objects and immutable control reports.
The intended sequence is:

```bash
gmoney-authority intake \
  --source-root /secure/client-corpus/working152 \
  --eligibility /secure/client-corpus/working152-eligibility.json
gmoney-authority assign \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/working152-inventory.json \
  --assignment /secure/client-corpus/nested-assignment.json \
  --allow-incomplete
gmoney-authority render \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/assigned-inventory.json
gmoney-authority review-queue \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/assigned-inventory.json
gmoney-authority validate-review \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/assigned-inventory.json \
  --review-root /path/to/four-pass-reviews
gmoney-authority seal-readiness \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/assigned-inventory.json
gmoney-authority seal \
  --inventory /home/azureuser/gmoney-corpus-vault/authoritative-v1/control/assigned-inventory.json \
  --identity /path/to/frozen-identity.json
gmoney-authority baseline --replay /path/to/replay-a --replay /path/to/replay-b \
  --identity /path/to/frozen-identity.json
```

`intake` is the safe working-corpus boundary. It requires one explicit eligibility decision per
PDF and rejects ineligible, synthetic, duplicate, unreadable, or unclassified sources. Its
`working152` inventory is explicitly non-authoritative and can never be sealed. Sensitive PDFs
and rendered images remain in the external vault; only code, tests, and aggregate control reports
belong in Git.

`assign` accepts source hashes (or a hash manifest), sorts them deterministically, and requires
the nested relationship `production14 ⊆ passing36 ⊆ staging159`. `--allow-incomplete` is for
work-in-progress reporting only; an exact authority assignment still requires precisely 14, 36,
and 159 documents and no unassigned inventory members. The fixed authority contract is never
changed by working-corpus counts or test fixtures.

`review-queue` creates four deterministic, machine-blind Luna work queues: independent A,
independent B, adjudicator, and red team. Queue records contain only content-addressed image
identities and geometry; they are not gold and cannot satisfy the seal. `seal-readiness` reports
exact cohort, render, and four-pass review coverage and exits nonzero while blockers remain. A
single Luna visual QA pass is useful for classifying work, but is not audited gold and cannot
satisfy the seal.

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
