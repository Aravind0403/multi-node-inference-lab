# Clairvoyant v2: Two-Phase Predictive SJF for Disaggregated LLM Serving

[![Paper](https://img.shields.io/badge/arXiv-2606.07248-b31b1b.svg)](https://arxiv.org/abs/2606.07248)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Clairvoyant v2** extends the predictive Shortest-Job-First (SJF) scheduling paradigm from serial single-worker LLM backends to modern **Prefill-Decode (P/D) Disaggregated Serving Architectures** (such as vLLM Disaggregated, SGLang HiCache, Moonshot Mooncake, and DeepSeek 3FS).

---

## 🚀 The Core Problem: The Disaggregation SJF Inversion Trap

In monolithic LLM serving, total service time is dominated by output token generation:
$$S_{\text{monolithic}} = T_{\text{prefill}}(\text{Prompt}) + T_{\text{decode}}(\text{Output})$$
Because decode takes $\approx 20\text{ ms/token}$ while prefill takes $\approx 0.04\text{ ms/token}$, output length dominates $90\%+$ of total request latency. Predicting output length ($\hat{L}_{\text{out}}$) via Clairvoyant's 0.029 ms ONNX model served as an effective proxy for total request time.

### Why Monolithic SJF Breaks in Disaggregated Backends:
In a disaggregated cluster, Prefill and Decode are physically decoupled into separate GPU pools:
* **Prefill Workers (P-Pool):** Compute-bound ($N \times D$ matrix multiplication); service time is $f(\text{Prompt})$, independent of output length.
* **Decode Workers (D-Pool):** Memory-bandwidth-bound ($1 \times D$ KV lookups); service time is $f(\text{Output})$, independent of prompt length.

> [!WARNING]
> If a Prefill worker naively uses output-length SJF, a 10,000-token prompt with a 10-token output will be prioritized ahead of a 50-token prompt with a 1,000-token output—turning SJF into **Longest-Job-First (LJF)** for prefill, causing severe Time-to-First-Token (TTFT) degradation for short requests.

---

## 🏛️ The Solution: Two-Phase Predictive SJF

Clairvoyant v2 implements a decoupled 3-layer scheduling hierarchy:

```
                            ┌──────────────────────────────────────────────┐
                            │    Layer 0: Ingress Gateway & Clairvoyant    │
                            │    Extracts L_prompt & Predicts P(Long)      │
                            │    (Sub-microsecond lexical + 0.029ms ONNX)  │
                            └──────────────────────┬───────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                     ▼
        ┌───────────────────────────────┐                     ┌───────────────────────────────┐
        │  Layer 1 P-Admission Queue   │                     │  Layer 1 D-Admission Queue   │
        │  Shortest-Prefill-First (SPF) │                     │  Shortest-Decode-First (SDF)  │
        │  Priority = Prompt Tokens     │                     │  Priority = P(Long)           │
        └───────────────┬───────────────┘                     └───────────────┬───────────────┘
                        ▼                                                     ▼
        ┌───────────────────────────────┐      KV Cache       ┌───────────────────────────────┐
        │       Prefill GPU Pool        │ ──────────────────► │        Decode GPU Pool        │
        │       (Compute-Bound)         │   (RDMA / PCIe)     │    (Memory-Bandwidth-Bound)   │
        └───────────────────────────────┘                     └───────────────────────────────┘
```

1. **Layer 0 (Global Ingress Gateway):** Extracts 19 lexical features and predicts $P(\text{Long})$ in $0.029\text{ ms}$, tagging request metadata.
2. **Layer 1 P-Stage (Shortest-Prefill-First / SPF):** P-Queue orders by prompt token count $L_{\text{prompt}}$. Short prompts clear the compute-bound P-workers in milliseconds.
3. **Layer 1 D-Stage (Shortest-Decode-First / SDF):** D-Queue orders by predicted $P(\text{Long})$ with a calibrated starvation timeout $\tau = 3 \times \mu_{\text{short}}$. Short outputs release their KV-cache slots rapidly.

---

## 📊 4-Quadrant Benchmark Matrix

| Quadrant | Prompt Length | Output Length | Scenario | Dominant Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Short-Short)** | $<200$ tok | $<50$ tok | Fast QA, intent classification | Ultra-low TTFT requirement |
| **Q2 (Short-Long)** | $<200$ tok | $>1000$ tok | Code generation, reasoning CoT | Decode-Bound |
| **Q3 (Long-Short)** | $>2000$ tok | $<50$ tok | Document QA, Summarization | Prefill-Bound |
| **Q4 (Long-Long)** | $>2000$ tok | $>1000$ tok | Repository refactor, synthesis | Both P & D Heavy |

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Aravind0403/clairvoyant-disagg.git
cd clairvoyant-disagg

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Feature & Predictor Tests
```bash
PYTHONPATH=. python3 tests/test_features_and_predictor.py
```

### 3. Run Comparative Evaluation (A vs B vs C)
```bash
PYTHONPATH=. python3 benchmarks/run_experiments.py
```

---

## 📁 Repository Structure

```text
clairvoyant-disagg/
├── model/
│   ├── predictor.onnx            # Pre-trained 0.029ms XGBoost ONNX model
│   ├── predictor.json            # Model hyperparameters and metadata
│   ├── feature_extractor.py      # 19-feature lexical extractor (sub-microsecond)
│   └── predictor_onnx.py         # ONNX Runtime CPU inference wrapper
│
├── proxy/
│   └── two_phase_dispatcher.py   # Layer 0 Ingress + Layer 1 P/D Admission Queues
│
├── benchmarks/
│   ├── workload_generator.py     # 4-quadrant synthetic trace generator (Q1-Q4)
│   ├── telemetry.py              # High-resolution TTFT, TPOT, KV-transfer logger
│   └── run_experiments.py        # Automated runner for Conditions A, B, and C
│
├── tests/
│   ├── test_features_and_predictor.py
│   └── test_poisson_ordering.py
│
├── README.md
├── RUNBOOK.md
├── RESEARCH_ROADMAP.md
└── BENCHMARKS.md
```

---

## 📜 Citation

```bibtex
@article{sundaresan2026clairvoyant,
  title={Clairvoyant: Predictive SJF Scheduling to Mitigate Head-of-Line Blocking in Serial LLM Backends},
  author={Sundaresan, Aravind},
  journal={arXiv preprint arXiv:2606.07248},
  year={2026}
}
```
