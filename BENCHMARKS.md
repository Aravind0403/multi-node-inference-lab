# Benchmarks & Findings

A dated lab notebook. Raw results go here first, before any write-up polish. Anything genuinely surprising gets flagged — that's the seed of a stronger blog post than a routine "I deployed X and it worked" result.

Format:

```
## [YYYY-MM-DD] Phase N — <what was measured>

**Hardware:**
**Config:**
**Metric:**
**Result:**
**Interpretation:**
**Novel / surprising?** yes/no — if yes, why
**Raw data:** (link or path)
```

---

## [2026-08-27] Phase 1 — vector_add.cu block size sweep

**Hardware:** Google Colab free tier, T4 GPU
**Config:** N = 2^20 (~1,048,576 float32 elements), C[i] = A[i] + B[i], block sizes 32/128/256/512/1024, median of 20 runs per block size (kernel-only time via cudaEvent timers, excludes host<->device memcpy)
**Metric:** kernel execution time (ms)
**Result:**

| Block size | Time (ms) | Warps/block |
|---|---|---|
| 32   | 0.097 | 1  |
| 128  | 0.054 | 4  |
| 256  | 0.055 | 8  |
| 512  | 0.055 | 16 |
| 1024 | 0.055 | 32 |

**Interpretation:** 32→128 gives ~1.8x speedup — going from 1 to 4 warps/block gives the SM enough concurrent warps to hide global memory load latency (while one warp stalls on a memory read, another can execute). 128→1024 is flat: once there are enough warps in flight to hide latency, adding more doesn't help further, because vector-add does ~1 FLOP per 2 loads and is memory-bandwidth-bound, not occupancy-bound. Occupancy tuning only matters up to the point of hiding latency; past that, throughput is capped by memory bandwidth.
**Novel / surprising?** no — expected result for a memory-bound kernel, but useful as a first confirmed baseline. A first pass (single run per config, no median) produced a spurious 94ms outlier at block size 256 — almost certainly transient contention/throttling on shared Colab GPU infra, not a real effect. It disappeared under median-of-20. Lesson: single-sample GPU timing is unreliable; always take median of N runs.
**Raw data:** cuda-kernels/vector_add.cu

---

## [2026-08-27] Phase 1 — matmul_tiled.cu naive vs shared-memory-tiled

**Hardware:** Google Colab free tier, T4 GPU
**Config:** Square matmul C=A×B, TILE=16, N in {512, 1024}, median of 10 runs per config (kernel-only time via cudaEvent timers)
**Metric:** kernel execution time (ms), speedup ratio, correctness (naive vs tiled output, tolerance 1e-2)
**Result:**

| N | TILE | Naive (ms) | Tiled (ms) | Speedup | Correctness |
|---|---|---|---|---|---|
| 512 | 16 | 0.662 | 0.424 | 1.56x | MATCH |
| 1024 | 16 | 5.321 | 2.150 | 2.47x | MATCH |

**Interpretation:** Tiling speedup grows with N (1.56x → 2.47x as N doubles). Theoretical global-memory-traffic reduction from TILE=16 tiling is ~16x, but measured speedup is far lower — T4's L2 cache already absorbs a meaningful share of the naive version's redundant re-reads, especially at smaller N where the working set (512x512x4B ≈ 1MB) plausibly fits partly in cache. As N grows, naive's redundant global traffic outgrows what L2 can absorb, so tiling's relative advantage widens. This is the compute-bound-workload counterpart to vector_add's memory-bound story: unlike vector_add (flat regardless of optimization, bandwidth-capped), matmul has real reuse to exploit, and shared-memory tiling is what captures it.
**Novel / surprising?** no — expected direction (tiling helps, more at scale), but the magnitude (well under theoretical 16x) is a useful calibration: hardware caching already does some of tiling's job, so tiling's benefit should be measured empirically, not assumed from the tile size alone.
**Raw data:** cuda-kernels/matmul_tiled.cu

---

## [2026-08-27] Phase 1 — vector_add.cu float4-vectorized variant (4 elements/thread)

**Hardware:** Google Colab free tier, T4 GPU
**Config:** Same N (~1,048,576 float32 elements) as scalar vector_add, but each thread processes 4 elements via `reinterpret_cast<float4*>` (128-bit load/store), block sizes 32-1024, median of 20 runs per config
**Metric:** kernel execution time (ms)
**Result:**

| Block size | Scalar (ms) | float4 (ms) |
|---|---|---|
| 32 | 0.097 | 0.055 |
| 128 | 0.054 | 0.057 |
| 256 | 0.055 | 0.057 |
| 512 | 0.055 | 0.058 |
| 1024 | 0.055 | 0.058 |

**Interpretation:** float4 is faster at block=32 (fewer total threads needed since each does 4x the work, so sufficient warp-level latency-hiding is reached at a smaller block size) but is actually *slightly slower* than the scalar kernel at block sizes 128 and above — not the clean speedup vectorization is often assumed to give. Likely explanation: the scalar kernel was already saturating memory bandwidth once occupancy was sufficient (confirmed by the flat scalar curve above 128 threads/block), so packing 4 elements per thread via `reinterpret_cast` added pointer-arithmetic/indexing overhead without buying anything once bandwidth was already the bottleneck — vectorized loads reduce transaction *count*, not total bytes moved, so they don't help once you're not transaction-count-limited.
**Novel / surprising?** yes — counter to the common assumption that float4 vectorization is a free win; a useful concrete example that "vectorize your memory access" is not universally correct advice and depends on whether the kernel is bandwidth-bound by byte count or by transaction count.
**Raw data:** cuda-kernels/vector_add.cu (cell 11 of the Colab notebook — note: cells 8 and 9 of that notebook contain fabricated/placeholder output from a Colab AI helper that malfunctioned and should be deleted, not treated as data)

---
