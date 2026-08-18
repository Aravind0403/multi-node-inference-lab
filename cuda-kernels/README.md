# Phase 1 — CUDA Fundamentals + Profiling

**Question:** Where does an inference kernel actually spend its time — compute or memory?

**Plan:**
- `vector_add.cu` — thread hierarchy basics
- `matmul_tiled.cu` — shared memory, tiling, occupancy
- `attention_core.cu` — inner-loop only, annotated against what FlashAttention optimizes for
- Profile a real vLLM attention kernel with `nsys`/`ncu`, compare against `matmul_tiled.cu`'s profile

**Write-up (once done):** *What I Learned Writing and Profiling My First CUDA Kernels*

Status: not started.
