# What I Learned Writing and Profiling My First CUDA Kernel

*Phase 1 of Multi-Node-Inference-Lab. Hardware: Google Colab free tier (T4 GPU). Time spent: ~4 hours.*

## The question

LLM inference is, underneath everything, a pile of GPU kernels. Before reasoning about SGLang schedulers or disaggregated prefill/decode serving, I wanted a real, hands-on answer to a basic question: when a GPU kernel runs, what actually determines how fast it goes — how much parallel compute you throw at it, or something else entirely?

## The experiment

I wrote `vector_add.cu` — element-wise addition of two ~1M-element float arrays, the simplest possible CUDA kernel — and swept the block size (32, 128, 256, 512, 1024 threads/block), timing kernel execution only (via `cudaEvent`, excluding host-device memory transfer) with a median of 20 runs per configuration to filter out GPU scheduling noise.

| Block size | Time (ms) | Warps/block |
|---|---|---|
| 32   | 0.097 | 1  |
| 128  | 0.054 | 4  |
| 256  | 0.055 | 8  |
| 512  | 0.055 | 16 |
| 1024 | 0.055 | 32 |

## What it showed

Going from 32 → 128 threads/block nearly halves execution time. Going from 128 → 1024 does almost nothing — the timings are flat within noise.

The explanation is occupancy, and it has a hard ceiling. A warp (32 threads, hardware-fixed, not a CUDA setting) is the unit the GPU schedules. With only 1 warp/block, when that warp stalls waiting on a global memory read, the SM has no other warp to switch to — it idles. At 4 warps/block, the scheduler can hide that stall by running a different warp while one waits. But `vectorAdd` does one addition per two memory loads — it's memory-bandwidth-bound, not compute-bound. Once you have enough warps in flight to fully hide memory latency, adding more buys nothing further, because the bottleneck has shifted from "not enough parallel work to hide latency" to "physically can't move bytes faster." More threads can't fix a bandwidth ceiling.

A first, single-sample pass (before switching to median-of-20) produced a spurious 94ms reading at block size 256 — roughly 600x slower than its neighbors, with no architectural explanation. It disappeared entirely under repeated sampling. The lesson mattered more than it first seemed: on shared, free-tier GPU infrastructure, a single timing sample is not a measurement, and any real benchmark needs to say how many runs and what aggregation, or it's not reproducible.

## Why this matters for inference serving

This toy kernel is a stand-in for a real split in how LLM inference behaves. Prefill (processing the input prompt) runs large matmuls across many tokens at once — high arithmetic intensity, compute-bound, like a well-tiled matmul kernel. Decode (generating one token at a time against a growing KV cache) does very little compute per token relative to how much cache it has to read — memory-bound, like `vectorAdd`. This is also, at a more advanced level, the exact problem FlashAttention solves: standard attention is memory-bound because it round-trips a full N×N score matrix through slow HBM; FlashAttention doesn't change the math, it tiles the computation to avoid ever materializing that matrix in slow memory, trading memory traffic (the actual bottleneck) for a bit more on-chip reuse. Understanding "is this memory-bound or compute-bound, and why" is the single mental model that both this toy benchmark and FlashAttention's real optimization sit on top of.


## Round two: the compute-bound counterpart

`vector_add.cu` is memory-bound by construction — there's almost no reuse to exploit. To see the opposite regime, I built `matmul_tiled.cu`: a naive matmul (each thread re-reads full rows/columns from global memory on every step) versus a shared-memory-tiled version (each block cooperatively loads a `TILE×TILE` chunk into fast on-chip memory once, then reuses it).

| N | Naive (ms) | Tiled (ms) | Speedup |
|---|---|---|---|
| 512 | 0.662 | 0.424 | 1.56x |
| 1024 | 5.321 | 2.150 | 2.47x |

Speedup grows with N — tiling's advantage compounds as the problem outgrows what the GPU's L2 cache can passively absorb. Notably, the measured speedup is well below the naive ~16x reduction in global memory traffic that TILE=16 theoretically buys — a reminder that hardware caching already does some of tiling's job for you, and the real benefit has to be measured, not assumed from the tile size alone. Together with vector_add, this gives a concrete before/after on both sides of the memory-bound/compute-bound divide: one workload where optimization can't help (bandwidth-capped, flat regardless of block size) and one where it demonstrably does (tiling, with a scaling trend to prove it).


## A vectorization result that didn't go as expected

I also implemented a `float4`-vectorized variant of vector_add (each thread processes 4 elements via a 128-bit load/store instead of 1). The common assumption is that this is a free win. It wasn't:

| Block size | Scalar (ms) | float4 (ms) |
|---|---|---|
| 32 | 0.097 | 0.055 |
| 128 | 0.054 | 0.057 |
| 256 | 0.055 | 0.057 |
| 512 | 0.055 | 0.058 |
| 1024 | 0.055 | 0.058 |

float4 wins at block=32 (needing fewer total threads to reach sufficient latency-hiding), but is slightly *slower* than the scalar version everywhere else. Once the scalar kernel already has enough warps to saturate memory bandwidth (as shown in the first experiment), reducing the number of memory *transactions* via vectorized loads doesn't help — you're not transaction-count-limited, you're byte-count-limited, and 128-bit loads don't move fewer bytes, just fewer requests. It's a useful correction to a piece of advice ("vectorize your memory access") that's often stated without the caveat of when it actually applies.

## What's not done yet

The full Phase 1 plan also included an annotated attention-core kernel and profiling a real vLLM attention kernel with `nsys`/`ncu`. Both were deliberately descoped (see TRADEOFFS.md) — FlashAttention's mechanism is already covered at a strong verbal level in the interview Q&A, and real-kernel profiling needs a working vLLM environment better set up during Phase 2 anyway. Phase 1 closes here: two working, benchmarked kernels covering both the memory-bound and compute-bound regimes, plus full Basic/Medium/Hard interview prep.

## Resume-bullet material

- Benchmarked CUDA kernel occupancy on GPU hardware, identifying the warp-count threshold (4 warps/block) beyond which throughput plateaus for a memory-bandwidth-bound workload, and diagnosing a spurious 600x outlier as GPU-timing noise, fixed via median-of-N sampling methodology.
- Implemented a shared-memory-tiled CUDA matmul kernel from scratch, measuring a 1.6-2.5x speedup over naive (scaling with problem size) and correctness-verified against the naive baseline.
- Implemented and benchmarked a float4-vectorized (128-bit load/store) variant of a memory-bound kernel, output-verified against the known analytic result, and diagnosed why it did not yield a consistent speedup once the scalar version was already bandwidth-saturated — distinguishing transaction-count-bound from byte-count-bound memory access.

---
*Raw data: [`BENCHMARKS.md`](../BENCHMARKS.md). Reproduction steps: [`RUNBOOK.md`](../RUNBOOK.md). Interview Q&A: [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md).*
