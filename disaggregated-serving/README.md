# Phase 3 — Disaggregated Serving

**Question:** Where does prefill/decode separation help, and where does it not?

**Plan:**
- Extend vLLM's real merged P/D disaggregation implementation rather than hand-rolling a custom prefill/decode split (decision locked in [`TRADEOFFS.md`](../TRADEOFFS.md), 2026-08-18).
- Instrument TTFT/TPOT separately for prefill-bound vs. decode-bound workloads, plus KV cache transfer overhead.
- Novel angle: extend Clairvoyant's predictive SJF admission control to a disaggregated backend and test whether its scheduling gains survive the P/D split — this became its own sub-project, **Clairvoyant v2**.

**What was built:** Clairvoyant v2 — a two-phase predictive SJF scheduler for disaggregated serving. Single-ranking SJF (tuned for monolithic serving, where output length dominates total cost) breaks under disaggregation because prefill cost is a function of prompt length and decode cost is a function of output length — two independent variables. Clairvoyant v2 splits admission into a Shortest-Prefill-First queue (ranked by prompt tokens) feeding the P-pool and a Shortest-Decode-First queue (ranked by predicted output length, via the existing 0.029ms ONNX predictor) feeding the D-pool, with a starvation timeout on the D-side so long jobs aren't starved indefinitely. Full architecture, the 4-quadrant workload matrix (Q1–Q4), and the 3-layer scheduling hierarchy diagram are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

**Code:** built and benchmarked in a standalone repo, [clairvoyant-disagg](https://github.com/Aravind0403/clairvoyant-disagg) (`model/`, `proxy/`, `benchmarks/`, `tests/`) — kept separate from this lab repo so it can also stand alone as a citable extension of the [Clairvoyant paper](https://arxiv.org/abs/2606.07248); results are reproduced here.

> **Note on measurement:** all results below are from a calibrated discrete-event simulator (`benchmarks/run_experiments.py` in clairvoyant-disagg) — request timings are computed analytically from measured per-token rate constants (`prefill_ms_per_tok`, `decode_ms_per_tok`), not measured on live GPU hardware. Real prompt/response *data* (Dolly, LMSYS, OASST1, CNN/DailyMail) feeds the simulation's request lengths, but the simulation itself does not execute any model or touch a GPU. A real-GPU validation run is planned (see status below) to check the simulator's calibration against actual hardware.

**Results (N=200, closed-loop burst):** Two-Phase SJF cuts Q1 (short-short) TTFT P50 from 970.25s under FIFO to 158.17s (83.7% reduction), and from 661.17s under monolithic SJF — the disaggregation-aware ranking outperforms both the naive baseline and the old single-phase scheduler. Full results, the Poisson steady-state comparison, and the queueing-theory explanation for the absolute numbers are in [`BENCHMARKS.md`](./BENCHMARKS.md).

**Write-up (once done):** *Disaggregated Serving on the Lab: Where It Helps, Where It Doesn't*

Status: Simulator benchmarks complete (synthetic + real-dataset). Interview Q&A pass complete through Hard tier (see INTERVIEW_QUESTIONS.md). Real-GPU validation run pending (planned alongside Phase 4's paid multi-GPU environment). Write-up not started.
