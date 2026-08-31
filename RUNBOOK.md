# Runbook

Exact steps to reproduce every result in this repo. Updated as each phase is built — nothing goes in `BENCHMARKS.md` without a corresponding entry here.

## Environment (fill in once set up)

- OS / driver / CUDA toolkit version:
- Python version + package manager (uv/poetry/pip):
- GPU(s) used, per phase (model, count, cloud provider or local):

## Phase 1 — CUDA Fundamentals + Profiling

- Environment: Google Colab free tier (T4 GPU). Runtime → Change runtime type → T4 GPU.
- Setup (in a Colab cell):
  ```
  !nvidia-smi          # confirm T4 is attached, check driver/CUDA version
  !nvcc --version       # confirm nvcc is available (it is, by default, on Colab GPU runtimes)
  ```
- Build/run commands (Assignment 1 — vector_add):
  ```
  # upload vector_add.cu to the Colab session, or use %%writefile to paste it into a cell
  !nvcc -O3 -o vector_add vector_add.cu
  !./vector_add
  ```
- Profiling commands (`nsys`, `ncu`) and what flags were used: TBD — added once we reach the profiling half of Assignment 1.

## Phase 2 — SGLang & vLLM Internals

- SGLang version / commit pinned:
- vLLM version / commit pinned:
- Benchmark harness invocation:

## Phase 3 — Disaggregated Serving

- Architecture: Clairvoyant v2, a two-phase predictive SJF scheduler sitting ahead of a disaggregated P/D cluster. Built and benchmarked in a standalone repo, [clairvoyant-disagg](https://github.com/Aravind0403/clairvoyant-disagg), reproduced in disaggregated-serving/ARCHITECTURE.md and disaggregated-serving/BENCHMARKS.md.
- Simulated cluster: 1 P-worker + 1 D-worker (minimal case), calibrated to NVIDIA RTX 4090/A10G/L4 rates (prefill 0.04ms/prompt-token, decode 20ms/output-token, KV transfer ~0.003ms/prompt-token over PCIe Gen4/RDMA).
- Workload generator / trace source: benchmarks/workload_generator.py in clairvoyant-disagg — synthetic 4-quadrant matrix (Q1-Q4), closed-loop burst and Poisson arrival modes. Also validated against real prompt/response data: data/dolly_labeled.csv, data/cnn_dailymail_test.csv, data/lmsys_prompts.jsonl, data/oasst1_labeled.csv (see clairvoyant-disagg repo).
- vLLM disaggregation config (production integration, not yet run on real GPUs): KV connector PyNcclConnector/MooncakeConnector, push mode with async overlap — see RUNBOOK section in clairvoyant-disagg for exact launch commands (--is-prefill-only / --is-decode-only).
- Status: simulator benchmarks (synthetic + real-dataset) complete; real-GPU run pending (Aravind will rerun on the actual multi-GPU setup and report back).

## Phase 4 — Multi-GPU Deployment

- Cluster config (node count, GPU count/node, interconnect):
- Terraform/manifest apply commands:
- Scaling study invocation:

## Phase 5 — Distributed Training Awareness

- Reference to Aether Control's GRPO/DeepSpeed pipeline config used for comparison:
