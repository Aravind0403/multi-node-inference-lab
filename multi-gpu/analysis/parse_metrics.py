#!/usr/bin/env python3
"""
parse_metrics.py
Parses, aggregates, and outputs statistical speedup comparisons between TP=1 vs TP=2
and vLLM vs SGLang prefix caching performance.
"""

import argparse
import glob
import json
import os
from typing import Dict, List
import pandas as pd


def load_benchmark_files(bench_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(bench_dir, "*.json"))
    all_records = []
    for fpath in files:
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict):
                    all_records.append(data)
            except Exception as e:
                print(f"Error parsing {fpath}: {e}")
    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame(all_records)


def generate_tp_scaling_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Tensor Parallelism Scaling & Speedup Analysis\n")
    if df.empty:
        return "No benchmark data found to analyze."

    # Group by engine, prompt_len, output_len, concurrency
    grouped = df.groupby(["engine", "concurrency", "tp_size"]).mean(numeric_only=True).reset_index()

    lines.append("## Throughput & Latency Scaling (TP=1 vs TP=2)\n")
    lines.append("| Engine | Concurrency | TP Size | Out Tok/s | TTFT P50 (ms) | ITL P50 (ms) | Speedup (Tok/s) |")
    lines.append("|---|---|---|---|---|---|---|")

    for (engine, conc), g in grouped.groupby(["engine", "concurrency"]):
        tp1_row = g[g["tp_size"] == 1]
        tp2_row = g[g["tp_size"] == 2]

        tp1_toks = tp1_row["output_tok_per_sec"].values[0] if not tp1_row.empty else 0.0
        tp2_toks = tp2_row["output_tok_per_sec"].values[0] if not tp2_row.empty else 0.0
        speedup_str = f"{tp2_toks / tp1_toks:.2f}x" if tp1_toks > 0 and tp2_toks > 0 else "N/A"

        for _, row in g.iterrows():
            lines.append(
                f"| {row['engine']} | {int(row['concurrency'])} | {int(row['tp_size'])} | "
                f"{row['output_tok_per_sec']:.1f} | {row['ttft_p50_ms']:.1f} | {row['itl_p50_ms']:.1f} | {speedup_str} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse multi-GPU benchmark results")
    parser.add_argument("--bench-dir", type=str, default="benchmarks")
    parser.add_argument("--output-report", type=str, default="benchmarks/scaling_summary.md")
    args = parser.parse_args()

    df = load_benchmark_files(args.bench_dir)
    report = generate_tp_scaling_report(df)
    print(report)

    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.output_report}")


if __name__ == "__main__":
    main()
