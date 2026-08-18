# Phase 2 — SGLang & vLLM Internals

**Question:** How do two production schedulers differ in design, and does it matter empirically?

**Plan:**
- Annotated internals map of SGLang's scheduler (RadixAttention, cache-aware scheduling) vs. vLLM's PagedAttention scheduler
- Clairvoyant SJF proxy benchmarked as a control condition in front of both backends, same trace replay methodology as the original paper
- Target: file at least one real SGLang issue/PR discovered during the internals read

**Write-up (once done):** *vLLM vs. SGLang: A Comparative Study from the Perspective of a Scheduler Builder*

Status: not started.
