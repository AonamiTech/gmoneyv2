# Extraction result v4

`offline_accuracy_spine_v4` is a legacy, uncertified contract. It remains
readable for historical review but cannot receive new machine certification;
use [Extraction Result v5](EXTRACTION_RESULT_V5.md) for current publication.

The result adds three fail-closed grounding inventories:

- `token_manifest` lists every OCR token and typed split fragment referenced by
  evidence. Fragment entries identify their parent token, character range, and
  canonical role.
- `diagnostics` contains exactly one page inventory entry per PDF page and one
  table entry per published `SourceTable.id`. Financial-form classifications and
  crop artifacts carry verifiable evidence or hashes.
- `recovery` records the baseline, candidate, and selected unit digests for the
  single permitted local recovery pass, plus the digest of all untargeted raw
  units.

Provider accounting is split into `provider_usage.initial`,
`provider_usage.recovery`, and `provider_usage.aggregate`. Recovery is required
to report zero Gemini calls; initial Gemini use remains visible in the aggregate.

The semantic validator first validates this envelope, then independently parses
rows, source tables, diagnostics, totals, and evidence. Malformed nested payloads
produce a fatal structured report and the worker surfaces
`extraction_integrity_failed` rather than publishing a partial result.

Publication uses `job_publication_v2`. Its journal records base and target
existence and SHA-256 digests for both result and state. Recovery accepts only a
complete base or complete target image and rejects symlinks or other non-regular
publication objects.
