# Runbook

Exact steps to reproduce every result in this repo. Updated as each phase is built — nothing goes in `BENCHMARKS.md` without a corresponding entry here.

## Environment (fill in once set up)

- OS / driver / CUDA toolkit version:
- Python version + package manager (uv/poetry/pip):
- GPU(s) used, per phase (model, count, cloud provider or local):

## Phase 1 — CUDA Fundamentals + Profiling

- Setup:
- Build/run commands:
- Profiling commands (`nsys`, `ncu`) and what flags were used:

## Phase 2 — SGLang & vLLM Internals

- SGLang version / commit pinned:
- vLLM version / commit pinned:
- Benchmark harness invocation:

## Phase 3 — Disaggregated Serving

- vLLM disaggregation config used:
- Workload generator / trace source:

## Phase 4 — Multi-GPU Deployment

- Cluster config (node count, GPU count/node, interconnect):
- Terraform/manifest apply commands:
- Scaling study invocation:

## Phase 5 — Distributed Training Awareness

- Reference to Aether Control's GRPO/DeepSpeed pipeline config used for comparison:
