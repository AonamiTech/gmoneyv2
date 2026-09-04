# Extraction Result V6

V6 is the publication contract for artifact-backed extraction. The V5 result
contract and `job_certification_v2` remain supported unchanged; V6 is selected
only when the complete artifact graph and V2 evidence are published.

## Envelope

`output_version` is `offline_accuracy_spine_v6` and `contract_revision` is `6`.
`artifact_manifest` is authoritative. Every manifest artifact must be present
under the job artifact root, have the declared SHA-256 and dimensions, and use
an immutable, contained path. Graph roots are `SOURCE_RAW`; child mappings may
be identity, finite orientation-preserving homographies, or verified dense
backward grids.

Dense grids are deterministic compressed NumPy archives containing only
`grid.npy`, a C-contiguous little-endian `float32` array shaped
`(control_height, control_width, 2)`. The two channels contain normalized
child-to-parent coordinates in `[-1, 1]`, sampled bilinearly with
`align_corners=true`. The complete archive digest, contained `.npz` path,
shape, child/parent dimensions, coordinate domain, interpolation, alignment,
and padding metadata are bound into the mapping and artifact IDs. Loading uses
`allow_pickle=false`, a 128 MiB uncompressed limit, explicit artifact-root
resolution, and no symlinks or traversal.

Evidence carries both `canonical_polygon` and `source_page_polygon`, the source
page artifact ID, artifact hash, and OCR token IDs. Publication validation checks
ownership, bounds, graph projection, and a two-pixel round-trip tolerance.
Nonlinear polygons are adaptively densified to at most one pixel of projected
chord error. Publication fails on a missing/corrupt grid, metadata conflict,
non-finite or out-of-bounds coordinate, non-positive Jacobian, fold-over,
orientation reversal, self-intersection, or adaptive mapping budget exhaustion.
Every table adapter record also declares its page and logical table. Its input
artifact must be the selected canonical table crop or a descendant of that crop;
cross-table artifact substitution fails validation. Publication additionally
requires exactly one `SOURCE_RAW`, one `ORIENTED_RAW`, and one selected page
artifact for every declared page.

## UVDoc shadow artifacts

M4 adds optional `uvdoc_shadow_runs` audit records without changing the V6 revision or the V5
reader. A valid run binds the pinned Paddle/model/config identities, exact grid-reproduction
error, deterministic transform diagnostics, and the `ORIENTED_RAW -> UVDOC -> UVDOC_ENHANCED`
artifact IDs. `UVDOC` uses the M3 dense backward-grid mapping; the enhanced derivative is
photometric-only and uses an identity mapping.

Both artifacts must remain unselected `CANDIDATE` page artifacts. The V6 validator rejects a
shadow artifact used as a canonical-table parent, adapter input, token owner, or evidence owner.
Failed and ineligible runs record a stable reason but may not publish candidate artifact IDs.
`GMONEY_UVDOC_MODE` defaults to `off`; M4 supports only `off` and `shadow`, and rejects `enabled`
until a later frozen promotion decision exists.

## Certification and compatibility

V6 workspaces use `job_certification_v3`, binding the result, validation report,
source PDF, complete image-and-grid inventory, and `artifact_graph_sha256`. The review
and export APIs continue to expose the V5 evidence shape by projecting V6 source
page polygons, source hashes, and token IDs. Evidence bundles additionally carry
the V6 manifest and graph digest.

Recertification revalidates the current output version only. A V5 revision 5
result remains V5 and is never silently upgraded. V5-to-V6 conversion requires
the digest-bound staged reprocess flow; source and review data are retained and
approval is cleared only after successful V6 publication.
