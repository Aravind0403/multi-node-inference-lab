# Trade-offs & Decisions

Lightweight decision log. One entry per non-obvious choice — add the entry when the decision is made, not reconstructed afterward.

Format:

```
## [YYYY-MM-DD] Phase N — <decision title>

**Decision:** what we chose
**Alternatives considered:** what else was on the table
**Why:** the actual reasoning
**Revisit if:** the condition that would change this decision
```

---

## [example — delete once real entries exist]

## [2026-08-18] Phase 3 — Use vLLM's built-in P/D disaggregation instead of hand-rolling one

**Decision:** Extend and instrument vLLM's merged prefill/decode disaggregation implementation rather than building a parallel prefill-service/decode-service architecture from scratch.
**Alternatives considered:** Custom Go/Python services with a manual KV cache transfer layer (original plan).
**Why:** A hand-rolled version duplicates work the framework already does correctly, costs significantly more build time, and is less credible in an interview than demonstrating fluency inside a real production codebase.
**Revisit if:** vLLM's implementation turns out too opaque to instrument meaningfully within the time available.
