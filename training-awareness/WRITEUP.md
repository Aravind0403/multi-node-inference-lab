# Training vs. Inference Infrastructure: What Building a GRPO Pipeline Taught Me

*Phase 5 of Multi-Node-Inference-Lab. Time spent: ~3 hours. No new GPU rental — this phase repackages the existing GRPO/DeepSpeed pipeline through a comparative lens rather than building new infrastructure.*

## The question

By this phase I'd spent four phases entirely inside inference infrastructure — kernels, schedulers, disaggregation, multi-GPU serving. But I'd already built a real training pipeline earlier, for a different reason (post-training a model with GRPO), without ever asking what structurally separates training infra from inference infra as *systems*. This phase answers that, using my own `rlhf-pipeline` (GRPO + DeepSpeed ZeRO-3, Qwen2.5-1.5B) and `vllm-engine` as the concrete, working example of both sides.

## Memory anatomy: the 8-9x multiplier

Inference needs one thing: the weights, resident for a forward pass (2 bytes/param, FP16), plus a KV-cache that scales with batch size × context length, not parameter count.

Training with Adam needs four things simultaneously:
- **Weights** (2 bytes/param, FP16) — still needed for the forward pass.
- **Gradients** (2 bytes/param, FP16) — `dL/dW` per parameter, computed by backprop and held until the optimizer step consumes them.
- **Adam optimizer states** (8 bytes/param) — two FP32 momentum buffers (first and second moment), kept in FP32 because the accumulation across thousands of steps needs numerical stability that FP16 can't provide.
- **FP32 master weight copy** (4 bytes/param) — the actual optimizer update is applied in FP32, not FP16; the result is cast back to FP16 for the next forward pass.

That's 2+2+8+4 = 16 bytes/param before activation memory, against inference's 2 bytes/param — an 8-9x gap that's *why* ZeRO-3 exists at all: at the memory ratio above, any model that comfortably fits in inference VRAM can outgrow a single GPU's training VRAM. The full formula-based comparison, runnable in under a second, is in [`demo/profile_training_vs_inference.py`](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/blob/main/demo/profile_training_vs_inference.py) in the pipeline repo.

## Where GRPO's rollout phase actually sits

GRPO generates G=4 completions per prompt before every policy update — and that rollout step is pure inference, memory-wise. It's a standard autoregressive forward pass wrapped in `torch.no_grad()`: no gradients, no optimizer states, no FP32 master copy. Only weights + KV-cache. The training-shaped cost (backward pass, ZeRO-3 collectives, optimizer step) only arrives in the next step, once rewards are computed from those rollouts.

The nuance that matters for a real peak-VRAM estimate: GRPO also keeps a frozen reference model (π_ref) resident for the KL-divergence term, and that model stays in inference-mode — no gradients, no optimizer state — *even during the update phase*. So actual peak VRAM during an update step is training-memory (policy) + inference-memory (reference model) at the same time, not a clean either/or between two phases.

## Communication and failure modes

ZeRO-3 shards parameters, gradients, and optimizer states across GPUs, which means training's steady-state traffic is an `All-Gather` (reconstruct full weights before computing a layer) followed by a `Reduce-Scatter` (aggregate and shard gradients after backprop) — a synchronous, whole-cluster operation every step. Inference's tensor-parallel `All-Reduce` happens per layer per token, but it's not accompanied by an optimizer-state dependency across steps, so a slow or dead worker degrades throughput rather than halting the whole job. That's the practical shape of stateful-vs-stateless: training is synchronous and cluster-wide by construction (ZeRO sharding means every GPU needs every other GPU's shard to reconstruct a layer), while inference replicas are independent and can fail in isolation.

## A real find: catching my own fabricated section before it shipped

The first draft of this phase's write-up included a "failure injection experiments" section (EXP-001 through EXP-005) with specific numbers — throughput dropping 81% under VRAM pressure, P99 latency spiking 776%, 142 preemptions under load. Those numbers were drafted as illustrative targets during early design modeling and never actually measured — no `.nsys-rep`, `.sqlite`, or benchmark JSON backing them existed anywhere in the repo, and the source file was never even git-tracked. They got carried into a polished write-up without being labeled as synthetic.

Caught during review, before publishing: verified against the actual repo (not a stale mirror — a genuine complication was that I was initially checking `aether-control-llm-infra`, a different repo than the real working one), confirmed no supporting artifacts existed anywhere, and removed the section entirely rather than caveating it, since there was no real run behind it at all. The published version — [`docs/explainer/training_vs_inference_infrastructure.md`](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/blob/main/docs/explainer/training_vs_inference_infrastructure.md) — contains only what's independently verifiable: the deterministic memory-formula math and citations to real repo assets (K8s manifests, DeepSpeed config). Full incident log in [`TRADEOFFS.md`](../TRADEOFFS.md).

This is worth having ready for an interview in its own right: "I drafted illustrative numbers during design, caught that they weren't backed by real measurements before they shipped, and pulled them" is a stronger signal than a suspiciously clean benchmark table would have been.

## What's not done yet

No real GPU profiling of the training pipeline (no `nsys`/`ncu` trace of an actual DeepSpeed ZeRO-3 step) — everything here is formula-derived, not measured on hardware, and is labeled as such throughout. A real profiling run would need a rented multi-GPU environment, which this phase deliberately didn't scope in (see the phase README — this was meant to be the low-cost phase, repackaging existing work rather than new infra spend). If Phase 4's GPU budget has room left after its own re-run, profiling one real GRPO training step there would upgrade this phase's evidence from "formula" to "measured."

## Resume-bullet material

- Analyzed the memory and communication architecture of a production GRPO/DeepSpeed ZeRO-3 training pipeline against a PagedAttention-based inference server, quantifying an 8-9x per-parameter VRAM gap (16 bytes/param training vs. 2 bytes/param inference) and identifying which components (Adam optimizer state, FP32 master weights) drive it.
- Identified that RLHF-style pipelines exercise both inference and training memory regimes within a single training step (inference-mode rollout generation and reference-model KL computation, alongside training-mode policy gradient updates), with implications for accurate peak-VRAM estimation.
- Caught and removed a fabricated "empirical results" section from a technical write-up before publishing, after auditing the repository for supporting evidence and finding none — replaced with only independently-verifiable, formula-derived analysis.

---
*Interview Q&A: [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md). Decision log: [`TRADEOFFS.md`](../TRADEOFFS.md). Source pipeline: [`Building-and-Optimizing-Production-LLM-Serving-System`](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System).*
