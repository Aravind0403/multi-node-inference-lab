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

## [2026-08-18] Phase 1 — Use Google Colab free tier (T4) instead of renting cloud GPUs

**Decision:** Run all Phase 1 CUDA kernel work on Colab's free T4 GPU rather than renting a cloud instance.
**Alternatives considered:** Lambda Labs / RunPod / GCP paid GPU instance.
**Why:** Phase 1's kernels (vector_add, tiled matmul, attention core loop) don't need multi-GPU or high-end hardware to demonstrate the concepts — a T4 is enough to get real, honest numbers at zero cost. Reserve paid cloud spend for Phase 4 (multi-GPU), where it's actually required.
**Revisit if:** Colab's session limits or lack of persistent `nsys`/`ncu` tooling become a real blocker for the profiling half of this phase.

## [2026-08-27] Phase 1 — Build tiled matmul, descope attention_core.cu and real vLLM kernel profiling

**Decision:** Build `matmul_tiled.cu` as the second (and final) hands-on Phase 1 kernel. Skip building `attention_core.cu` as code (kept as theory/interview-answer only). Descope profiling a real vLLM attention kernel with `nsys`/`ncu` out of Phase 1 entirely.
**Alternatives considered:** Build all three planned kernels + real profiling, matching the original cuda-kernels/README.md plan in full.
**Why:** Within the 20hr total project budget, matmul_tiled.cu is the highest-leverage remaining build — it's the compute-bound counterpart to vector_add's memory-bound story, gives a real before/after tiling benchmark, and closes the loop on shared-memory-tiling theory already covered in the Phase 1 interview Q&A (all tiers answered correctly, including FlashAttention). attention_core.cu would add limited marginal value on top of an already-strong verbal FlashAttention answer, for real build time. Real vLLM kernel profiling needs vLLM installed + a model + working `ncu` hardware counters, which free-tier Colab often restricts (virtualized GPU access) — high risk of burning an hour on infra before getting a single number. Better attempted in Phase 2, where a working vLLM/SGLang environment will exist anyway.
**Revisit if:** Phase 2 environment setup turns out to leave meaningfully more time than expected, or an interviewer specifically probes on real-kernel profiling experience.

## [2026-08-29] Phase 2 — Filed real SGLang PR instead of a synthetic one

**Decision:** File sgl-project/sglang#37067 (optimize `RadixCache.total_size()` from O(N) traversal to O(1) tracked-sum) as the Phase 2 "real issue/PR" deliverable, found by tracing the eviction path while building the internals map.
**Alternatives considered:** Filing a documentation-only issue (lower risk, lower signal); skipping this deliverable if nothing turned up in the read.
**Why:** A real perf PR against production RadixCache code — not just docs — is stronger evidence of internals fluency than a synthetic exercise, and matches the phase plan's original bar ("file at least one real SGLang issue/PR discovered during the internals read").
**Revisit if:** the PR is rejected on grounds that reveal a misunderstanding of `total_size()`'s usage — worth folding that correction into the write-up either way.

## [2026-08-31] Phase 2 — Descope empirical prefix-sharing benchmark

**Decision:** Skip the T4-based prefix-sharing sweep (vLLM vs. SGLang throughput/TTFT at varying prefix-sharing ratios) for now. Internals map + filed PR (#37067) stand as the Phase 2 deliverables.
**Alternatives considered:** Push through on Colab T4 with a smaller model; rent a paid GPU instance now instead of waiting for Phase 4.
**Why:** Running both serving stacks side-by-side needs more GPU/env setup than free-tier Colab reasonably supports (both frameworks installed, compatible CUDA/driver versions, enough VRAM headroom for two servers). Not worth burning hours on infra now when Phase 4 (multi-GPU) already budgets for a real paid cloud instance where this becomes trivial to add.
**Revisit if:** Phase 4 environment is up early and there's slack time — rerun this sweep there instead of building it twice.

## [2026-08-31] Phase 3 — Built Clairvoyant v2 as a standalone repo, validated via simulation before real-GPU run

**Decision:** Build and validate the Two-Phase Predictive SJF scheduler (Clairvoyant v2) as a discrete-event simulation first — both synthetic 4-quadrant workloads and real prompt/response data (Dolly 15K, CNN/DailyMail, LMSYS-Chat-1M, OASST1) — before committing GPU-hours to standing up vLLM's real P/D disaggregation. Kept as a standalone repo (clairvoyant-disagg) rather than building directly inside this lab, so it can stand alone as a citable extension of the original Clairvoyant paper.
**Alternatives considered:** Stand up real vLLM disaggregation first and measure everything on-hardware from the start.
**Why:** The core research question (does Two-Phase SJF's scheduling gain survive disaggregation, and by how much) is answerable analytically/via simulation using measured per-token compute/memory-bandwidth rates, without burning GPU-hours on infra setup first. Simulation also let real-dataset validation happen immediately (no GPU inference needed to replay historical prompt/response length pairs), which surfaced a real and interesting finding on its own: the ~84% synthetic gain shrinks to ~36% on real data, because real quadrant distributions don't have the artificial bimodal extremes of the synthetic workload (see disaggregated-serving/INTERVIEW_QUESTIONS.md Medium-tier Q1 for the full breakdown).
**Revisit if:** the real-GPU run (planned next, alongside Phase 4's paid multi-GPU environment) shows the simulator's calibrated rates were meaningfully wrong — in which case the simulation results need a corrective caveat in the write-up, not a full redo.

