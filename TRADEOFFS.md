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


## [2026-08-31] Phase 4 — First Vast.ai TP=2 run: numbers unverified, treat as representative not exact

**Decision:** Rented a real Vast.ai instance (2x RTX 4070 Ti SUPER, instance 49398763, ~19:03-22:34 IST) and ran the TP=2 vLLM deployment, nsys profiling, and benchmark sweep on it. However, the resulting numbers in `multi-gpu/benchmarks/*.json` and the two Phase 4 report docs were hand-recorded from terminal output rather than saved directly by the scripts (`run_benchmarks.py`'s own `--output` JSON write, and the `nsys` trace/stats files, were not captured/committed) — no `.nsys-rep`/`.sqlite` file exists in the repo, and the JSON values have suspiciously round figures consistent with transcription rather than raw client output.
**Alternatives considered:** Re-run immediately to get clean machine-logged artifacts; discard the numbers entirely and mark Phase 4 empirical work as not-yet-done.
**Why:** Out of Vast.ai credits tonight — can't afford an immediate re-run. Keeping the numbers as directionally representative (real hardware, real ~3.5hr session, plausible scaling shape) is more useful than discarding them, but they should not be presented as exact reproducible measurements until a clean re-run happens.
**Revisit if:** the fresh re-run (planned next session) produces different numbers or a different qualitative shape (e.g. APC crossover point, TP comm/compute split) — replace these values rather than keep both. Until then, do not quote exact figures from this run in interviews without caveating "approximate, from a session where auto-logging failed" — the round numbers won't survive a "walk me through your raw data" follow-up question.

## [2026-08-31] Phase 4 — Prefix-caching benchmark inverted (flagged, rerun pending)

**Finding:** `multi-gpu/benchmarks/vllm_tp2_cached.json` shows the opposite of the expected result — TTFT P50 and throughput are *worse* with prefix caching enabled than the no-caching baseline (`vllm_tp2.json`) at every concurrency level (e.g. TTFT P50 29.5ms → 47.7ms at concurrency=1; throughput 47.0 → 23.3 tok/s), instead of the expected >10x TTFT reduction on cache hits.
**Root cause not yet confirmed. Ranked hypotheses to check on rerun:**
1. `--enable-prefix-caching` is a vLLM *server startup* flag, not per-request — if the same long-running server (started once with APC on) ran both the plain sweep and the "cached" sweep back to back without restart, the "no-caching" baseline wasn't actually a clean comparison, and the cached run may have inherited KV-pool fragmentation/memory pressure from the prior concurrency-64 sweep.
2. Tokenization boundary effects: `f"{shared_prefix} {user_part}"` joins identical shared text with a differing suffix; BPE re-tokenization near the boundary can produce different token IDs for the "same" prefix depending on what follows, breaking the block-level hash match that APC relies on.
3. The committed JSON only has aggregate percentiles (`BenchmarkResult`), not per-request records — `RequestMetric.cached_prefix_hit`/`is_cached_prefix` exists in the code but isn't exported, so there's no way to directly verify cold (request 0) vs. warm (request i>0) TTFT within a single run from the current output.
**Fix for rerun:** restart the vLLM server fresh (clean KV pool) immediately before the caching sweep; export per-request metrics (not just aggregates) so cold-vs-warm can be compared directly within one run; check vLLM's own server-side prefix-cache hit-rate logging/metrics endpoint to confirm hits are actually occurring before trusting the aggregate latency numbers.
**Status:** not logged as a resolved result. Rerun planned for 2026-09-01.


## [2026-09-01] Phase 5 — Synthetic failure-injection numbers drafted, caught before publishing, removed

**Finding:** An early draft of the Phase 5 explainer (`training_vs_inference_infrastructure.md` in the source pipeline repo) included a "failure injection experiments" section — EXP-001 through EXP-005 — with specific throughput/latency numbers (e.g. -81.0% throughput drop, +776% P99 latency, 142 preemptions) presented as empirical results. No `.nsys-rep`, `.sqlite`, or raw benchmark JSON backing these numbers exists anywhere in the repo, and the source file (`Doc_Content/failure_experiments.md`) was never git-tracked.
**Root cause:** the numbers were drafted as illustrative/synthetic targets during early design modeling, then carried into the polished write-up without being labeled as such.
**Fix:** confirmed directly with the source-of-truth repo (not a stale mirror) that no backing artifacts exist; numbers were removed entirely rather than caveated, since there was no real run behind them at all. Write-up now contains only the deterministic memory-formula math (verifiable by running `demo/profile_training_vs_inference.py`) and citations to real repo assets (K8s manifests, DeepSpeed config). Fix commit: `db28b96` (content) and `ebdc6a0` (repo-relative links, local commit not yet pushed as of 2026-09-01).
**Also caught this session:** the pipeline repo referenced by this project's README (`aether-control-llm-infra`) is not the actual working repo — the real one is `Building-and-Optimizing-Production-LLM-Serving-System`. Verifying against the wrong GitHub repo initially made a since-corrected claim ("I fixed the 404s") look false. Confirmed by connecting the local folder directly and checking `git remote -v` + `git log`.
**Revisit if:** the main README's external link to `aether-control-llm-infra` should probably be corrected to point at `Building-and-Optimizing-Production-LLM-Serving-System` — not yet done, flagging here so it isn't lost.
