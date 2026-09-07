# Table Magic authority-v1 bootstrap review

Date: 2026-09-07 UTC

Status: partial corpus built; authority seal held

## Decision

The repository now has strict contracts, a deterministic structural evaluator, a
content-addressed external-vault workflow, four Luna review prompts, and a fail-closed M4 UVDoc
authority gate. These are the standards and tooling needed to create the missing accuracy
authority; they do not turn incomplete labels into gold.

The local source search recovered 142 unique candidate PDFs from known GMoney-related roots. The
`staging159` master is therefore short by at least 17 unique eligible documents, before the 142
candidates themselves receive eligibility and near-duplicate review. The known 14-job/113-page regression set is
identified as `production14`, but `passing36` and `staging159` membership have not been assigned.
No document currently has a complete `GoldDocumentV2` plus all four frozen review records, so no
authority seal, baseline, or M4 accuracy promotion exists.

## External vault evidence

The non-Git vault is `/home/azureuser/gmoney-corpus-vault/authoritative-v1`.

| Evidence | SHA-256 / result |
| --- | --- |
| Unassigned 142-document inventory | `a35c7c96fa1e8c454babc8000fdb94cb0f37a1e268505b1abe777ca7ea53cdb4` |
| Partial inventory with `production14` membership | `43345ef335a344300ad0d7dde95753c64ff31cb586a1182489ad6078cb857aa2` |
| Unassigned render manifest | `f5f1b952afdbc1105358449e50c79c55e5bad3529e829106b2a35e75592a6509`; 142 documents / 967 pages |
| Render manifest bound to partial `production14` inventory | `14c7a4ea3fb157a69e771f58397a7bcb861572e92020896ea32dc2973c09fb8c`; 142 documents / 967 pages |
| Production membership | 14 documents / 113 pages |
| Passing membership | 0 of 36 assigned |
| Staging master membership | 0 of 159 assigned; at least 17 eligible source PDFs still missing from the discovered union |
| Luna visual QA A, first four production documents | 36 pages; report `ed369e5b2587f1d86779d522112ff6150ddf43380277fdc10816f62498621b89` |
| Luna visual QA A, next seven production documents | 52 pages; report `03a342c3d96c43867d3d053fe7c671c68d7ab4e0792b349c4fc63d796a33e7ee` |
| Luna visual QA A, final three production documents | 25 pages; report `18d93d664c6e2ec077483ab1ed11262d741ce1e949bba7be4763b5958b9deb68` |
| Luna visual QA B, same blind subset | 36 pages; report `7c637f43e216660d2996a6e2a4a381cd616afc41fdd473160a0c53f17f5ce780` |
| Luna visual QA B, remaining ten production documents | 77 pages; report `91e19762225b0ebf1e77ebcf0dea582bbd083c38688116b601e3f8bc0aa82b1b` |

The A and B reports independently cover all 113 production pages. Both reviewers were blind to
machine output, and B did not inspect A's classifications. All 113 page renders passed each
reviewer's image-integrity check. These are image-quality classifications only. They verify the render assets and
identify credible flat controls and curved-photo candidates, but they do not transcribe values,
adjudicate disagreements, or satisfy the four-pass gold contract.

## Fail-closed controls

- Source/page/table/column/row/cell identities are derived from source hashes and geometry, not
  from transcription.
- Inventory permits partial discovery, but sealing requires exact nested 14/36/159 membership
  with every document in `staging159`.
- Authority rendering is fixed at 300 DPI, sRGB, RGB PNG, and binds every page and per-document
  manifest by SHA-256.
- Sealing dereferences and validates each immutable gold object and all four immutable review
  objects; summary booleans and digest-shaped strings are insufficient.
- Independent A/B reviewers must be distinct and blind to machine output. Adjudication must
  reference both reviews; red-team review is separately required.
- Empty documents/cohorts and undefined required denominators fail closed. Machine normalized
  values cannot override visible raw evidence.
- M4 v1 evidence remains diagnostic only. Authoritative M4 evaluation uses v2 identities,
  baseline/UVDOC downstream proofs, exact floor numerators and denominators, curved improvement,
  and flat non-regression.

## Remaining blockers

1. Acquire and eligibility-review at least 17 more unique source PDFs.
2. Assign the nested `passing36` and `staging159` cohorts without changing the already identified
   production set.
3. Complete structural annotation and the four frozen review passes for every sealed document.
4. Freeze the evaluator manifest and two deterministic baseline replays against those exact
   sources and gold.
5. Run the baseline/UVDOC ablation on the isolated GPU candidate. A passing M4 result permits
   continued `shadow` operation only; `enabled` remains prohibited.

Until all five are complete, M0, M1, M2, and M4 retain their existing hold/in-progress states.
M3 remains promoted on its already-recorded synthetic and GPU evidence.
