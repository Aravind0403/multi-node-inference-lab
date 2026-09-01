# Phase 5 — Distributed Training Awareness

**Question:** How does training infra differ structurally from inference infra?

**Plan:**
- Repackage existing GRPO/TRL/DeepSpeed ZeRO-3 pipeline through the "training vs. inference infra differences" lens (memory profile, communication pattern, failure modes)
- Short comparative note: ZeRO stages vs. what the GRPO pipeline actually uses, and why

**Write-up:** [Training vs. Inference Infrastructure](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/blob/main/docs/explainer/training_vs_inference_infrastructure.md) — memory anatomy (bytes/param math, verifiable via `demo/profile_training_vs_inference.py`), inter-GPU communication (ZeRO-3 All-Gather/Reduce-Scatter vs. vLLM Tensor-Parallel All-Reduce), and failure blast radius (stateful training vs. stateless inference), grounded in the actual GRPO/DeepSpeed ZeRO-3 pipeline and Kubernetes manifests in that repo. See TRADEOFFS.md for a verification note on this write-up's revision history.

Status: done (2026-09-01).
