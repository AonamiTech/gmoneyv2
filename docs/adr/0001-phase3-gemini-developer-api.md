# ADR 0001 — Phase 3 Gemini Developer API challenger

## Status

Accepted for Phase 3 preproduction evaluation only.

## Decision

Use the Google GenAI SDK and stable `gemini-3.5-flash` as a provider-neutral,
de-identified challenger after local recovery and PaddleOCR-VL fail to resolve a
table crop. Gemini is disabled by default and is promoted only by frozen quality
evidence.

Only a redacted crop may leave the local environment. Calls fail closed when
redaction, evidence IDs, geometry, numeric grounding, budget, or provider response
validation fails. Raw bills and unredacted crops are never sent through this route.

## Consequences

- Phase 3 can measure managed-provider quality and cost without making it a baseline
  dependency.
- The Developer API is not approved here for production PHI. Production provider,
  region, retention, and governance eligibility remain a later release gate.
- A local-only result that passes all gates keeps Gemini disabled.
