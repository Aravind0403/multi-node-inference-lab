# Multi-Node-Inference-Lab

An applied systems research lab studying where LLM inference serving techniques — scheduling, parallelism, and disaggregation — actually hold up as you scale from a single GPU to heterogeneous, multi-node deployments, and where the common assumptions break down.

This extends the serving control plane built in [Aether Control](https://github.com/Aravind0403/aether-control-llm-infra) and the scheduling research from [Clairvoyant](https://arxiv.org/html/2606.07248v1) into CUDA-level, multi-GPU, and disaggregated-serving territory.

Every phase below produces working code and a technical write-up. Raw findings are logged in [`BENCHMARKS.md`](./BENCHMARKS.md) as they happen; decisions and their trade-offs are logged in [`TRADEOFFS.md`](./TRADEOFFS.md); exact reproduction steps live in [`RUNBOOK.md`](./RUNBOOK.md).

## Phases

| # | Phase | Question being tested | Status | Write-up |
|---|-------|------------------------|--------|----------|
| 1 | [CUDA Fundamentals + Profiling](./cuda-kernels/) | Where does an inference kernel actually spend its time — compute or memory? | Done | [Write-up](./cuda-kernels/WRITEUP.md) |
| 2 | [SGLang & vLLM Internals](./sglang-internals/) | How do two production schedulers differ in design, and does it matter empirically? | Done | [Write-up](./sglang-internals/WRITEUP.md) |
| 3 | [Disaggregated Serving](./disaggregated-serving/) | Where does prefill/decode separation help, and where does it not? | In progress — Clairvoyant v2 built, simulator + real-dataset benchmarks done, interview Q&A through Hard tier; real-GPU run pending | — |
| 4 | [Multi-GPU Deployment](./multi-gpu/) | Where's the real bottleneck when scaling — communication or compute? | Not started | — |
| 5 | [Distributed Training Awareness](./training-awareness/) | How does training infra differ structurally from inference infra? | Done | [Write-up](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/blob/main/docs/explainer/training_vs_inference_infrastructure.md) |

## Repo structure

```
multi-node-inference-lab/
├── README.md              # this file — objective + phase index
├── RUNBOOK.md              # exact setup + reproduction steps, per phase
├── TRADEOFFS.md            # running decision log (lightweight ADRs)
├── BENCHMARKS.md           # dated lab notebook of raw findings
├── cuda-kernels/
├── sglang-internals/
├── disaggregated-serving/
├── multi-gpu/
└── training-awareness/
```
