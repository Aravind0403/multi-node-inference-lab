# Phase 3 — Disaggregated Serving

**Question:** Where does prefill/decode separation help, and where does it not?

**Plan:**
- Stand up vLLM's real merged P/D disaggregation implementation
- Instrument: TTFT/TPOT separately for prefill-bound vs. decode-bound workloads, KV cache transfer overhead
- Novel angle: does Clairvoyant's SJF admission control interact differently with a disaggregated backend vs. a monolithic one?

**Write-up (once done):** *Disaggregated Serving on the Lab: Where It Helps, Where It Doesn't*

Status: not started.
