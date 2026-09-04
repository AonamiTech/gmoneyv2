# Extraction Result V6

V6 is the publication contract for artifact-backed extraction. The V5 result
contract and `job_certification_v2` remain supported unchanged; V6 is selected
only when the complete artifact graph and V2 evidence are published.

## Envelope

`output_version` is `offline_accuracy_spine_v6` and `contract_revision` is `6`.
`artifact_manifest` is authoritative. Every manifest artifact must be present
under the job artifact root, have the declared SHA-256 and dimensions, and use
an immutable, contained path. Graph roots are `SOURCE_RAW`; child mappings are
finite, invertible, orientation-preserving homographies (dense mappings remain
reserved for M3).

Evidence carries both `canonical_polygon` and `source_page_polygon`, the source
page artifact ID, artifact hash, and OCR token IDs. Publication validation checks
ownership, bounds, graph projection, and a two-pixel round-trip tolerance.
Every table adapter record also declares its page and logical table. Its input
artifact must be the selected canonical table crop or a descendant of that crop;
cross-table artifact substitution fails validation. Publication additionally
requires exactly one `SOURCE_RAW`, one `ORIENTED_RAW`, and one selected page
artifact for every declared page.

## Certification and compatibility

V6 workspaces use `job_certification_v3`, binding the result, validation report,
source PDF, complete manifest inventory, and `artifact_graph_sha256`. The review
and export APIs continue to expose the V5 evidence shape by projecting V6 source
page polygons, source hashes, and token IDs. Evidence bundles additionally carry
the V6 manifest and graph digest.

Recertification revalidates the current output version only. A V5 revision 5
result remains V5 and is never silently upgraded. V5-to-V6 conversion requires
the digest-bound staged reprocess flow; source and review data are retained and
approval is cleared only after successful V6 publication.
