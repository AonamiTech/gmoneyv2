# Extraction Result v5

`offline_accuracy_spine_v5` is the only extraction contract that can be newly
certified. Earlier results remain readable as legacy, uncertified history and
must be reprocessed before approval or export.

New publications require `contract_revision: 2`. Revision 2 adds stable
geometry anchors, typed derived-field provenance, complete token-fragment
lineage, and the raw total-candidate contexts required for safe totals-only
maintenance. Revisionless and revision-1 v5 results remain uncertified.

The complete top-level result is decoded by `ExtractionResultV5` before any
semantic validation. Malformed nested values therefore produce fatal,
field-addressed validation issues instead of parser exceptions.

## Grounding

- Every page publishes a complete rendered-page asset.
- Every evidence token exists in the token manifest and its text supports the
  Printed or canonical value that cites it.
- Recovery tokens retain their crop artifact, crop-space polygon, dimensions,
  page-space polygon, and crop-to-page transform.
- Typed fragments retain their parent token and exact character span.
- Page and source-table diagnostic inventories are exact; duplicate and orphan
  diagnostics are fatal.

## Recovery

Targeted recovery consumes typed internal page/table units, never a serialized
public result. A candidate is selected only when it removes a targeted blocker,
introduces no new blocker or fatal issue, preserves unaffected grounded charges,
and leaves every untargeted raw-unit digest unchanged. Otherwise the baseline
unit is retained with `recovery_no_safe_improvement` or
`recovery_target_not_located`.

Local recovery cannot invoke Gemini. Initial, recovery, and aggregate provider
usage remain separately auditable.

## Publication

State and result publication is journaled with base and target hashes. Recovery
accepts only a recorded base or target image. An abort marker does not bypass
digest validation. Historical bases may be uncertified, but recovery always
restores their exact recorded bytes rather than a mixed state.
