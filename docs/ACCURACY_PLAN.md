# Accuracy Plan: Verified-or-Flagged Bill Extraction

**Branch:** `feat/acc` · **Status:** planned, to be built and then tested on the GPU host · **Date:** October 2026

## 1. Summary

Today the system cannot tell a correct bill from a wrong one, so wrong bills pass review as `accepted`. This plan changes two things:

1. **A reconciliation gate proves every bill against its own printed arithmetic.** That means section sub-totals, the summary page and the grand total. A bill is either **VERIFIED** or **FLAGGED with the exact section and rupee difference**. Nothing passes silently.
2. **The hand-written row builder is replaced as the primary reader.** In its place come **two independent readers**: TeleOCR, an open-source 1.2B table model run on our own GPU, and Gemini, reading only table crops. Where they disagree, the field is flagged.

Estimated cost is **about ₹35–75k per 1,00,000 pages, all-in**, against a budget of ₹2.1 lakh. These are estimates; the GPU test confirms them.

## 2. The problem today

### How the current pipeline works

1. Pages are rendered at 300 dpi.
2. PP-OCRv6 reads the words and PP-DocLayoutV3 finds the tables.
3. **A hand-written row builder** (`src/gmoney/extraction/ocr_rows.py`, about 5,000 lines) turns words into rows:
   - keyword lists decide which header is "Amount", "Qty" or "Description";
   - fixed x-position tolerances (3–8% of page width) assign each number to a column;
   - more keyword lists decide summary vs detail vs return.
4. PaddleOCR-VL (a vision-language model) runs **only as a fallback**, when a table yields almost no rows.
5. Validation checks that each row points at real pixels ("grounded"). It **does not check that the numbers are right.**

### What we measured

We ran the current pipeline locally on the sample bills and compared the output with each bill's own printed totals.

| Bill | Hospital | Result |
|---|---|---|
| Bill 6 | Saroj Gupta | ❌ Two whole tables on page 2 were never detected: bed charges ₹11,100 and doctor visit ₹650 lost. Procedures were labelled "informational" instead of charges. A pharmacy **return was counted as a charge**. Drug brand names were dropped ("Advamab 100mg", "Adripeg 50mg", "Neukine 300 Mcg"). The grand total was not found. |
| Bill 10 | Vijaya | ❌ The **bill number "4172" was read as a ₹4,172 charge**. 15 rows show codes ("SER0943548") instead of item names ("GRAM STAIN"). |
| Bill 11 | Kamakshi | ✅ Amounts reconcile. Minor label errors (wrong section tags, "4OMG"). |
| Bill 12 | Mediversal | ✅ Correct (package bill). |

**Every wrong row above was marked `accepted`.**

### Root causes

1. **Digit reading is fine.** In every section checked by hand, the individual amounts Paddle read were correct.
2. **Structure is the weak point.** Rigid rules break on real variation:
   - the same hospital's template changes column sets between bills (Mediversal Bills 2 vs 13);
   - descriptions wrap over 3–5 lines;
   - columns are packed tightly;
   - stamps sit on top of numbers;
   - group rows ("Issued Date") are mixed in with items;
   - returns are printed with positive numbers under a "Return" heading.
3. **Table detection misses tables**, and anything outside a detected table is invisible to every later step.
4. **Nothing proves correctness.** The code computes rows-vs-total only for display (`demo/review.py::totals_summary`). A mismatch does not block approval, and section sub-totals and summary pages are never checked.
5. **Per-hospital profiles never activate in the demo.** The hospital ID is never passed (`offline.py`), so no hospital-specific structure is ever used.
6. **The VLM was judged on a weak setup.** It was used as a rare fallback on a T4 with an 8k-token context per slot, which caused the truncation and format failures recorded in `phase-2-run-issues.md`. Run properly (full context, one table per request), the same PaddleOCR-VL-1.6 model recovered everything Bill 6 lost: the summary, the missing tables and the full drug names, with 100% of amounts correct.

## 3. Why the current system is not trustworthy

- **"Grounded" ≠ correct.** Every row has pixel evidence, but evidence only shows *where* a value came from, not that it was read or assigned correctly.
- **No arithmetic gate.** The bill's own checksums (sub-totals, summary, grand total) are available on almost every bill but are never enforced.
- **Single reader.** One rule-based interpretation, with no independent second opinion to catch disagreement.
- **No measured accuracy.** Earlier runs on these sample bills reported "5,551 grounded rows" but had no gold labels. They measured completion, not correctness. The only honest unseen-layout gate (Phase 2) scored 47% precision / 10% recall. The latest audited pilot reported ~77% critical-number recall, ~66% correct column assignment and 0% exact grand totals.

## 4. The plan

### 4.1 Reconciliation gate (always on)

New module `src/gmoney/extraction/reconciliation.py`. Every bill is checked against:

| Check | Rule |
|---|---|
| C1 Grand total | charge rows (returns negative) = printed **gross** bill total |
| C2 Section sub-totals | each printed "Sub Total" / "Total" = the rows directly above it (nested totals supported) |
| C3 Summary page | page-1 category lines add up to the grand total |
| C4 Row arithmetic | qty × rate − discount = amount (reported only; some hospitals print gross in "Rate") |

- **Matching:** exact to the paisa. A difference under ₹1 is accepted only when the printed total is a whole rupee (e.g. 94,140.18 printed as 94,140), and it's labelled "rounded".
- **Outcome:**
  - **VERIFIED**: everything reconciles;
  - **FLAGGED**: page, section, expected, actual and difference for each failure;
  - **UNPROVABLE**: no printed totals found, which is treated as flagged.
- **Approval:** blocked until the reviewer's edits make the bill reconcile, or until the reviewer records an override with a written reason (logged).
- **Job status:** `complete` only when the bill is VERIFIED; otherwise `needs_review`.

### 4.2 Switchable table reader

`GMONEY_TABLE_READER`:

| Mode | Behaviour |
|---|---|
| `heuristic` (default) | today's pipeline, unchanged |
| `teleocr` | TeleOCR reads every table as the primary source of rows |
| `teleocr_gemini` | TeleOCR + Gemini read every table independently; cell-level consensus |

**TeleOCR** (`StarDoc-AI/TeleOCR`, 1.2B, Apache-2.0 weights, released Aug 2026):
- Tops current open table benchmarks: table TEDS 97.05 on OmniDocBench v1.6, and 89.05 vs 85.76 for PaddleOCR-VL-1.6 on real camera photos.
- **Reads curved and folded pages directly, without a flattening step.** This covers the Table Magic goal without UVDoc, which damaged flat pages in the M5 pilot.
- Runs as its own GPU service.

**Missed-table guard.** If a page has amounts outside every detected table, the whole page is also read. This fixes the Bill 6 failure.

**Gemini second reader.** It receives table crops only; the patient header never leaves our servers. Consensus rules:
- amounts that agree → accepted; amounts that disagree → the value that makes the section reconcile wins, otherwise the cell is flagged;
- names that agree → accepted; names that disagree → flagged, with both versions shown to the reviewer;
- a row found by only one reader → flagged.

**Pixel evidence is kept.** PP-OCR word boxes still ground every value, so the review UI keeps highlighting it.

### 4.3 Comparison command and runbook

- `gmoney-compare` runs the sample bills in every mode. Per bill it reports: verified/flagged and each failed check, rows, reader disagreements, seconds per page and cost per page.
- `docs/TELEOCR_RUNBOOK.md` explains how to run it on the T4 host.

### Not in this branch

- Frontend changes.
- Table Magic / UVDoc: parked, since TeleOCR handles curved pages natively.
- Per-hospital learned hints.

## 5. Why VERIFIED bills will be nearly 100% trustworthy

The claim is deliberately narrow: **a bill marked VERIFIED is almost certainly correct, and everything else is flagged with a reason.** No model reads every page perfectly. The trust comes from proof and cross-checking, not from the model.

1. **Amounts are proven, not assumed.** For a wrong amount to pass, it would have to make every section sub-total, the summary page and the grand total still add up to the paisa. A missed row, an extra junk row, a wrong sign or a misread digit each break at least one of these. Every failure we measured (Bills 6 and 10) is caught by C1–C3.
2. **Multiple independent checks on the same numbers.** Section totals, summary-page totals and the grand total are printed separately on most bills. A single error rarely balances all of them by chance.
3. **Two independent readers for what arithmetic can't prove.** Item names and labels have no checksum. TeleOCR and Gemini are different model families trained differently, so they rarely make the same mistake. In our test, PaddleOCR-VL and the current pipeline made *different* text errors on the same page. When both readers agree, the value is very likely right; when they disagree, a human sees exactly that field.
4. **Fail closed.** If anything can't be proven (no printed total found, or an unresolved disagreement), the bill is flagged, never passed.
5. **The gate also measures the system.** Every change can be judged by how many bills it verifies, with no regressions allowed.

### Remaining risks

- **Compensating errors:** two wrong amounts that happen to cancel exactly within the same section. This is very rare, and the readers' agreement check makes it rarer still.
- **Both readers wrong in the same way** on an item name. This is unlikely across different model families, but not impossible.
- **Printed bill is itself wrong** (the hospital's own arithmetic error). The bill is flagged, and the reviewer can override with a reason.
- **Labels that don't affect totals** (e.g. which section a row belongs to) are cross-checked by the two readers but not by arithmetic.

## 6. Cost

**Target:** 1,00,000 pages for ₹2.1 lakh, including OCR hosting. These are estimates to confirm on the GPU host.

| Item | Assumption | Per 1,00,000 pages |
|---|---|---|
| TeleOCR on 1× T4 | ~3–6 s/page, ~85–170 GPU-hours at ~₹60/hr | ₹5–10k |
| Or: T4 host kept on for a month | ~₹60/hr × 730 hr | ~₹45k |
| Gemini second reader, every page | ≤ ₹0.30/page (capped in config) | ≤ ₹30k |
| PP-OCR grounding + retries | runs on the same host | included above |
| **Total** | | **≈ ₹35–75k (≈ ₹0.35–0.75/page)** |

That's roughly a third of the budget, leaving headroom for reruns, a larger GPU if throughput needs it, or Gemini tie-breaks on harder pages.

For comparison, the current pipeline took about 5 minutes per page on CPU in local tests. A full-precision VLM took 8–27 seconds per page even on a laptop, and should take a few seconds on a GPU.

## 7. Risks and open items

- **TeleOCR on T4 is untested:** fp16 (the T4 has no bf16), throughput and memory alongside the existing Paddle services. The GPU run will measure all three.
- **TeleOCR is new** (released Aug 2026) with a small community. Its GitHub code has no license file, so we use only the Apache-2.0 weights through our own service code.
- **Accuracy so far is based on a few sample bills.** A small hand-checked set (10–15 bills) would let us state a measured "verified but wrong" rate.
- **Production use of patient bills** on any rented or external infrastructure, including Gemini with table crops, needs the owner's approval.

## 8. Next steps

1. Build the branch (gate, readers, comparison command, runbook) using mocked model replies only.
2. The owner runs `gmoney-compare` on the sample bills on the T4 host and shares the report.
3. Review the results: verified rate per mode, flagged reasons, speed and cost per page.
4. If `teleocr_gemini` verifies more bills with no regressions and stays within budget, make it the default.
