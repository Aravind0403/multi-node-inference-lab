# vLLM vs. SGLang: A Comparative Study from the Perspective of a Scheduler Builder

*Phase 2 of Multi-Node-Inference-Lab. Time spent: ~5 hours (estimate — adjust if off).*

## The question

Two production inference servers, vLLM and SGLang, both solve the same core problem — serving LLM requests efficiently under a shared, scarce resource (GPU KV cache memory) — with visibly different scheduler and cache designs. I wanted to know whether that difference is cosmetic or load-bearing: does it actually change what workloads each system is good at, or is it two implementations converging on the same idea from different angles?

## The approach

Rather than treat this as a reading exercise, I read both schedulers as source, not documentation: vLLM's `vllm/v1/core/sched/scheduler.py` and `kv_cache_manager.py`, and SGLang's `sglang/srt/managers/scheduler.py` and `mem_cache/radix_cache.py`. The full annotated internals map — architecture comparison table, side-by-side scheduling-loop flowcharts, data structure internals, and code-level walkthroughs of both `schedule()` (vLLM) and `event_loop_normal()` (SGLang) — is in [`INTERNALS_MAP.md`](./INTERNALS_MAP.md).

## What the two designs actually optimize for

Both systems solve out-of-memory the same way in spirit — free something, then retry — but *what* gets freed reveals the real design split. vLLM preempts a whole running request when it runs out of physical blocks: all of that request's allocated memory is released and it's dropped back to WAITING, forced to recompute its entire prefill from scratch on readmission. SGLang instead evicts specific idle radix-tree leaves (`lock_ref == 0`) — no active request is ever interrupted, and shared ancestor nodes (a hot system prompt, say) stay resident even as unrelated cold branches get pruned. One design protects *cache state*, the other protects *whichever request happens to be running*.

That split traces straight back to how each system indexes what's already computed. vLLM hashes fixed 16-token blocks and only matches on closed block boundaries — a 100-token shared prefix caches 96 tokens and silently recomputes the remaining 4 on every request, and worse, in a real multi-turn conversation, every turn boundary lands at a different offset mod 16, so that partial-recompute tax recurs on every single turn, not once. Concretely: a 147-token first turn (100-token shared prefix + 47-token response) leaves 3 tokens stranded outside any complete block, forcing recomputation of those 3 tokens plus the entirety of turn two. SGLang's radix tree has no block-alignment constraint — it stores and matches arbitrary-length token sequences, so that same 147-token history matches exactly, with zero tokens recomputed, and a new turn is just a child node attached to the existing tree.

This isn't a strict "SGLang wins" story, though. Cache-aware scheduling and pure SJF-style admission control can actively conflict: if a scheduler reorders the queue purely by predicted output length, it can force eviction of a hot, heavily-shared prefix to admit a short-output request with a cold prompt — trading a cache hit for an admission-order preference, at real throughput cost. Which one wins depends on the traffic shape, not on which system is "better" in the abstract — see the SJF-vs-SGLang analysis in the interview Q&A for the concrete traffic-pattern breakdown.

## A real find: filing sgl-project/sglang#37067

Tracing SGLang's eviction path surfaced a genuine inefficiency: `RadixCache.total_size()` computed the tree's total cached-token count via a full DFS traversal — O(N) in the number of live tree nodes, which in a production server with deep multi-turn/agentic branching can reach 10^4–10^6 nodes. That traversal isn't rare — it's hit by metrics scrapers and prefix-aware routing control planes, both of which query cache size on a regular interval, meaning an O(N) Python-GIL-blocking traversal was landing in a scheduling-adjacent hot path. The fix: SGLang already tracks `evictable_size_` and `protected_size_` incrementally across every tree mutation (insert, split, lock/unlock, evict) — their sum is a mathematically conserved invariant, so `total_size()` can be O(1) by construction instead of walking the tree. I filed [PR #37067](https://github.com/sgl-project/sglang/pull/37067) with the fix plus unit tests verifying size consistency across all six mutation types; it's awaiting review.

This is the deliverable I'd point to first in an interview: not "I read the SGLang codebase" but "I found and fixed a real algorithmic complexity bug in it."

## What's not done yet

The original plan included an empirical benchmark — a prefix-sharing sweep (0%/50%/90% shared prefix) measuring vLLM vs. SGLang throughput and TTFT on a real Colab T4. That got blocked on environment setup (running both serving stacks side by side needs more GPU/driver/CUDA-version coordination than free-tier Colab reasonably supports) and was deliberately descoped rather than burning hours chasing infra (see TRADEOFFS.md). The Hard-tier interview answer on cache-thrashing-vs-live-lock failure modes under overload is accordingly a reasoned hypothesis from each system's eviction policy, not a measured result — flagged as such in the Q&A log. This benchmark is a natural fit for Phase 4, where a real paid multi-GPU environment is already budgeted, and running it there avoids building GPU infra twice.

## Resume-bullet material

- Authored a code-level comparative internals analysis of vLLM's PagedAttention/block-hash scheduler vs. SGLang's RadixAttention/radix-tree scheduler, covering memory management, prefix caching, and preemption/eviction design tradeoffs.
- Found and fixed an O(N)→O(1) algorithmic complexity issue in SGLang's `RadixCache.total_size()` (filed as [sgl-project/sglang#37067](https://github.com/sgl-project/sglang/pull/37067)), replacing a full tree DFS with an incrementally-tracked invariant, with unit test coverage across all cache mutation paths.
- Reasoned through and proposed a unified cache-aware admission scoring function combining RadixCache prefix-hit ratio with Clairvoyant's SJF output-length prediction, identifying the specific traffic regimes (prefix-sharing ratio, output-length variance) where each scheduling strategy dominates or conflicts with the other.

---
*Internals map: [`INTERNALS_MAP.md`](./INTERNALS_MAP.md). Interview Q&A: [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md). Decision log: [`TRADEOFFS.md`](../TRADEOFFS.md).*
