# Task 3 implementation report

## Outcome

Implemented SHA-256-grouped reprocessing and a mandatory source-level visual
audit gate for staged demo jobs.

Byte-identical PDFs are now extracted once. The validated representative
extraction is cloned into every duplicate job's stage, while each job receives
its own source name, result payload, migrated review, review marker, and staged
payload seal. Apply remains an all-job transaction and cannot create backups or
touch live result/artifact paths until the v2 manifest, visual audit, page
inventory, payload seals, source digests, staged results, and review markers
have passed validation.

## Implementation

### Unique-source staging

`stage_reprocess_jobs()` now:

- snapshots every selected completed job before extraction;
- computes the immutable digest of each `source.pdf`;
- groups snapshots by `source_sha256`;
- selects the first sorted job ID as each group's representative;
- invokes the extractor once per source digest;
- validates the representative before cloning its artifacts;
- hard-links representative artifact files into duplicate stages when
  possible, with `copy2` fallback;
- deep-copies the representative result for each job and restores that job's
  source name;
- runs `_validate_result()` and `_migrate_review()` separately for every job;
- rechecks all captured review markers after the batch has been staged.

New manifests use `reprocess_stage_v2`. They contain per-job documents plus a
`sources` inventory with:

- source SHA-256;
- sorted representative job ID;
- every grouped job ID;
- exact page count;
- source names;
- a representative extraction digest.

Each document also records its source SHA and a digest sealing its complete
staged result, review, and artifact tree. The manifest has a UTC `sealed_at`
timestamp.

### Result validation

`_validate_result()` now rejects:

- a page-asset inventory whose page numbers are not exactly `1..N`;
- an unlinked source row whose mapped `net_amount` or `gross_amount` cell
  contains a parseable financial value.

Unlinked footer or non-ledger text with no parsed mapped financial value remains
permitted.

### Mandatory visual audit

`apply_staged_jobs()` accepts only `reprocess_stage_v2` and requires
`visual-audit.json` with version `reprocess_visual_audit_v1`.

For every source group, apply requires exactly one audit entry with:

- the exact 64-character lowercase source SHA;
- a non-blank reviewer;
- an explicit UTC `reviewed_at` at or after the batch seal;
- page entries exactly numbered `1..N`;
- `pass` for every page;
- exactly the six required source-level checks, all set to `pass`.

Malformed SHA types, boolean page numbers, missing/duplicate/extra source
entries, non-UTC timestamps, incomplete page lists, failed pages, and any
failed or missing named check are rejected.

Audit validation and staged-result validation happen before backup-root
creation. Manifest page counts are cross-checked against every staged result,
and page assets are cross-checked against the exact result page inventory.

### Sealed and recoverable cutover

Apply parses the manifest only far enough to derive the complete sorted job
set. It then acquires every job's exclusive lock before claim/resume
validation or cleanup and holds those same locks through validation, claiming,
cutover, rollback, and restoration. Under the locks, it:

1. requires every job workspace to be stable (including no preexisting
   cutover journal);
2. verifies the apply-owned claim namespace is a real contained directory
   hierarchy, never a symlink;
3. checks the manifest, audit, review CAS markers, representative extraction,
   and every payload seal;
4. atomically renames whole per-job stage directories into a
   digest-named apply-owned namespace;
5. validates the claimed immutable paths again;
6. recomputes each migrated review from the locked live review and compares it
   with the staged migration;
7. rechecks control/payload digests immediately before each cutover.

The known pre-claim stage paths no longer name the payload used by cutover.
Root symlinks, descendant symlinks, ambiguous paths, and paths outside the
expected direct-child staging namespaces are rejected.

Claim cleanup is journal-aware:

- validation and CAS failures with no cutover journal restore mixed/preexisting
  claims to the original stage namespace while all batch locks remain held;
- successfully rolled-back failures restore claims;
- a recovery-required failure retains every claimed path while any job journal
  remains, so `JobStore` recovery can still use the journal's recorded
  `stage_dir`;
- successful commits remove journals and restore the residual stage
  directories under the same locks;
- unsafe claim roots or claimed children are rejected without following or
  moving them.

Existing multi-job locks, review CAS checks, cutover journals, atomic rollback,
store recovery, v1/v2 committed backup reading, and manual rollback behavior
remain in place.

`reprocess_jobs(apply=True)` now fails explicitly before extraction and directs
callers to the required two-phase workflow: stage, record the visual audit,
then apply the audited stage.

## Test coverage

`tests/test_demo_reprocess.py` now covers:

- one extractor call for two byte-identical PDFs;
- v2 source-group manifest contents and representative selection;
- staged result/artifact trees for every duplicate;
- independent source names, review revisions, overrides, and migration events;
- duplicate review changes during staging and before apply;
- atomic rollback of a duplicate source group;
- unlinked mapped net/gross totals and allowed non-financial footer text;
- missing, incomplete, failed, mismatched, duplicate, malformed, non-UTC, and
  replayed visual audits;
- every named source-level check failing independently;
- passing audits with existing cutover, rollback, and recovery tests;
- manifest page-count tampering and invalid page-asset numbers;
- result changes after audit and changes injected after lock acquisition;
- swapped duplicate review payloads;
- atomic stage-directory claiming and original-path writer isolation;
- root-symlink/out-of-tree rejection;
- mixed preexisting claim restoration on validation failure;
- preservation of journal-referenced claims on recovery-required failure;
- rejection of preexisting cutover journals without overwriting them;
- symlinked claim-namespace cleanup safety;
- sorted, batch-wide exclusive locking around mixed-claim restoration;
- explicit rejection of the legacy one-step apply workflow.

The focused file contains 51 tests.

## TDD evidence

### Initial Task 3 RED

The first focused command established the three core missing behaviors:

```text
uv run pytest \
  tests/test_demo_reprocess.py::test_stage_extracts_identical_sources_once_and_preserves_per_job_reviews \
  'tests/test_demo_reprocess.py::test_reprocess_validation_rejects_unlinked_printed_financial_total[net_amount]' \
  'tests/test_demo_reprocess.py::test_apply_rejects_invalid_visual_audit_before_creating_backups[missing-visual audit is missing]' \
  -q

3 failed
- extractor was called twice for one source digest
- the unlinked financial total did not raise
- apply succeeded without visual-audit.json
```

The identical command passed after the initial implementation.

### First review RED

The first review found stage/audit sealing, page-inventory, review-integrity,
direct-apply, and strict-type gaps. The focused regression command failed all
six new cases before the hardening:

```text
uv run pytest \
  tests/test_demo_reprocess.py::test_reprocess_validation_requires_exact_page_asset_numbers \
  tests/test_demo_reprocess.py::test_direct_apply_requires_the_two_phase_visual_audit_workflow \
  tests/test_demo_reprocess.py::test_apply_rejects_swapped_duplicate_reviews \
  tests/test_demo_reprocess.py::test_apply_rejects_staged_result_changed_after_visual_audit \
  tests/test_demo_reprocess.py::test_apply_revalidates_staged_payload_after_acquiring_locks \
  tests/test_demo_reprocess.py::test_apply_rejects_visual_audit_replayed_for_a_new_extraction \
  -q

6 failed
```

The identical command passed after sealed digests, replay timing, locked
revalidation, review recomputation, exact page inventories, and explicit
two-phase apply were implemented.

### Second review RED

The second review identified the remaining known-stage-path writer race and
root-symlink redirect:

```text
uv run pytest \
  tests/test_demo_reprocess.py::test_apply_atomically_claims_the_staged_payload_before_cutover \
  tests/test_demo_reprocess.py::test_apply_rejects_a_symlinked_staged_payload_root \
  -q

2 failed
- the original stage path still existed at cutover
- a symlinked stage root was accepted
```

Both passed after atomic whole-directory claiming and containment checks.

### Third review RED

The third review identified journal-aware cleanup and mixed-claim recovery
gaps:

```text
uv run pytest \
  tests/test_demo_reprocess.py::test_apply_preserves_claimed_payload_when_cutover_requires_recovery \
  tests/test_demo_reprocess.py::test_apply_restores_mixed_preexisting_claims_when_validation_aborts \
  -q

2 failed
- the journal-recorded claimed stage path had been moved away
- the preexisting claimed job remained stranded after validation aborted
```

Both passed after claim release became batch-journal-aware and early
prevalidation/CAS cleanup restored unjournaled mixed claims.

### Fourth review RED

The fourth review found that an existing cutover journal could be overwritten,
that unsafe claim-namespace cleanup could follow a symlink toward live jobs,
and that mixed-claim cleanup needed the same batch locks as cutover:

```text
uv run pytest \
  tests/test_demo_reprocess.py::test_apply_cleanup_never_follows_a_symlinked_claim_root \
  tests/test_demo_reprocess.py::test_apply_rejects_an_existing_cutover_journal_without_overwriting_it \
  tests/test_demo_reprocess.py::test_apply_restores_mixed_preexisting_claims_when_validation_aborts \
  -q

3 failed
- cleanup followed an unsafe claim namespace
- a preexisting cutover journal was not rejected before claim
- early mixed-claim cleanup did not hold every exclusive job lock
```

All three pass after moving every claim/resume validation and cleanup operation
inside one sorted, batch-wide exclusive-lock scope, requiring every workspace
to be stable before claim, and rejecting unsafe claim namespaces.

## Review history

Four read-only review cycles were completed before the final review request:

1. The initial review found page-count trust, replay/TOCTOU, staged-review
   swapping, implicit one-step apply, and strict audit-type issues. All received
   focused RED tests and fixes.
2. The second review confirmed those fixes and found the residual known-path
   writer race plus root-symlink containment. Both received focused RED tests
   and fixes.
3. The third review confirmed atomic claiming/containment and found
   recovery-required journal-path cleanup plus mixed-claim early-failure
   cleanup. Both received focused RED tests and fixes.
4. The fourth review confirmed those recovery fixes and found preexisting
   journal overwrite risk, unsafe symlink cleanup, and incomplete lock scope.
   All three received focused RED tests and the apply path was restructured
   around one batch-wide lock lifetime.
5. The final read-only re-review found no critical, important, or minor
   issues. It explicitly confirmed that all prior findings were closed and
   marked the task ready.

## Verification

The project-local `.venv/bin/uv` was used because `uv` is not on the shell
`PATH`.

```text
uv run pytest tests/test_demo_reprocess.py -q
51 passed

uv run pytest tests/test_demo_api.py tests/test_demo_history.py -q
20 passed

uv run ruff check src/gmoney/demo/reprocess.py tests/test_demo_reprocess.py
All checks passed!

git diff --check
exit 0

uv run pytest -q
240 passed
```

## Scope

Changed only demo reprocessing, its focused tests, and this report. No
frontend, hospital extraction, worker concurrency, deployment files, GPU
services, public HTTP APIs, job statuses, or UI behavior were changed.
