# Phase 4 — Multi-GPU Deployment

**Question:** Where's the real bottleneck when scaling — communication or compute?

**Plan:**
- Single-node multi-GPU first (2-4 GPUs, vLLM built-in tensor parallelism) — cheap, fast, real numbers
- Multi-node as an explicit stretch goal (real cloud cost — budget before committing)
- Scaling study: 1 vs 2 vs 4 (vs 8) GPUs, TP vs PP, isolate communication vs. compute bottleneck

**Write-up (once done):** *How vLLM Scales Across Multiple GPUs: A Bottleneck-First Scaling Study*

Status: not started.
