# Phase 2 — SGLang & vLLM Internals

**Question:** How do two production schedulers differ in design, and does it matter empirically?

**Plan:**
- Annotated internals map of SGLang's scheduler (RadixAttention, cache-aware scheduling) vs. vLLM's PagedAttention scheduler
- Clairvoyant SJF proxy benchmarked as a control condition in front of both backends, same trace replay methodology as the original paper
- Target: file at least one real SGLang issue/PR discovered during the internals read

**Write-up (once done):** *vLLM vs. SGLang: A Comparative Study from the Perspective of a Scheduler Builder*

Status: Done. Internals map: [INTERNALS_MAP.md](./INTERNALS_MAP.md). PR filed: [sgl-project/sglang#37067](https://github.com/sgl-project/sglang/pull/37067) (awaiting review). Full Basic/Medium/Hard interview Q&A complete, all staff-level: [INTERVIEW_QUESTIONS.md](./INTERVIEW_QUESTIONS.md). Write-up: [WRITEUP.md](./WRITEUP.md). Descoped: empirical prefix-sharing benchmark (vLLM vs. SGLang on T4) — blocked on Colab multi-stack GPU setup, deferred to Phase 4's paid GPU environment.
