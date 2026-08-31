#!/usr/bin/env python3
"""
run_benchmarks.py
Comprehensive Async Benchmarking Engine for vLLM & SGLang Multi-GPU Deployments.
Measures TTFT, ITL/TPOT, Request & Token Throughput, and Prefix-Caching Hit Rates.
"""

import argparse
import asyncio
import json
import os
import random
import string
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class RequestMetric:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    itl_ms_list: List[float]
    total_latency_ms: float
    success: bool
    cached_prefix_hit: bool = False
    error: str = ""

@dataclass
class BenchmarkResult:
    engine: str
    tp_size: int
    concurrency: int
    prompt_len: int
    output_len: int
    prefix_caching_enabled: bool
    total_requests: int
    successful_requests: int
    duration_s: float
    req_per_sec: float
    output_tok_per_sec: float
    total_tok_per_sec: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    e2e_mean_ms: float
    e2e_p50_ms: float
    e2e_p90_ms: float
    e2e_p99_ms: float


def generate_random_prompt(num_tokens: int) -> str:
    """Approximate 1 token ~ 4 characters."""
    words = ["inference", "tensor", "parallel", "gpu", "kernel", "stream", "allreduce", "nccl", "latency", "throughput"]
    needed_words = max(1, num_tokens // 2)
    return " ".join(random.choices(words, k=needed_words))


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    is_cached_prefix: bool = False
) -> RequestMetric:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    itl_list: List[float] = []
    generated_tokens = 0

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return RequestMetric(
                    request_id=request_id,
                    prompt_tokens=len(prompt.split()),
                    output_tokens=0,
                    ttft_ms=0.0,
                    itl_ms_list=[],
                    total_latency_ms=0.0,
                    success=False,
                    cached_prefix_hit=is_cached_prefix,
                    error=f"HTTP {resp.status}: {text[:100]}"
                )

            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                if line_str == "data: [DONE]":
                    break

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft = (first_token_time - start_time) * 1000.0
                else:
                    itl_list.append((now - last_token_time) * 1000.0)

                last_token_time = now
                generated_tokens += 1

            total_latency = (time.perf_counter() - start_time) * 1000.0
            ttft_val = (first_token_time - start_time) * 1000.0 if first_token_time else total_latency

            return RequestMetric(
                request_id=request_id,
                prompt_tokens=len(prompt.split()),
                output_tokens=generated_tokens,
                ttft_ms=ttft_val,
                itl_ms_list=itl_list,
                total_latency_ms=total_latency,
                success=True,
                cached_prefix_hit=is_cached_prefix
            )

    except Exception as e:
        return RequestMetric(
            request_id=request_id,
            prompt_tokens=len(prompt.split()),
            output_tokens=0,
            ttft_ms=0.0,
            itl_ms_list=[],
            total_latency_ms=0.0,
            success=False,
            cached_prefix_hit=is_cached_prefix,
            error=str(e)
        )


async def run_concurrency_sweep(
    endpoint_url: str,
    model: str,
    engine: str,
    tp_size: int,
    concurrency: int,
    num_requests: int,
    prompt_len: int,
    output_len: int,
    prefix_caching: bool = False
) -> BenchmarkResult:
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Pre-generate prompts
        shared_prefix = generate_random_prompt(prompt_len // 2) if prefix_caching else ""
        tasks = []
        queue = asyncio.Queue()

        for i in range(num_requests):
            is_cached = prefix_caching and (i > 0)
            user_part = generate_random_prompt(prompt_len // 2 if prefix_caching else prompt_len)
            full_prompt = f"{shared_prefix} {user_part}" if prefix_caching else user_part
            await queue.put((f"req_{i}", full_prompt, is_cached))

        metrics: List[RequestMetric] = []
        bench_start = time.perf_counter()

        async def worker():
            while not queue.empty():
                try:
                    req_id, prompt_text, is_cached = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                metric = await send_streaming_request(
                    session=session,
                    url=endpoint_url,
                    model=model,
                    prompt=prompt_text,
                    max_tokens=output_len,
                    request_id=req_id,
                    is_cached_prefix=is_cached
                )
                metrics.append(metric)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        bench_duration = time.perf_counter() - bench_start

    successful = [m for m in metrics if m.success]
    if not successful:
        raise RuntimeError("All benchmark requests failed!")

    ttfts = [m.ttft_ms for m in successful]
    e2es = [m.total_latency_ms for m in successful]
    all_itls = [itl for m in successful for itl in m.itl_ms_list] if any(m.itl_ms_list for m in successful) else [0.0]
    total_out_tokens = sum(m.output_tokens for m in successful)
    total_all_tokens = sum(m.prompt_tokens + m.output_tokens for m in successful)

    return BenchmarkResult(
        engine=engine,
        tp_size=tp_size,
        concurrency=concurrency,
        prompt_len=prompt_len,
        output_len=output_len,
        prefix_caching_enabled=prefix_caching,
        total_requests=num_requests,
        successful_requests=len(successful),
        duration_s=bench_duration,
        req_per_sec=len(successful) / bench_duration,
        output_tok_per_sec=total_out_tokens / bench_duration,
        total_tok_per_sec=total_all_tokens / bench_duration,
        ttft_mean_ms=float(np.mean(ttfts)),
        ttft_p50_ms=float(np.percentile(ttfts, 50)),
        ttft_p90_ms=float(np.percentile(ttfts, 90)),
        ttft_p99_ms=float(np.percentile(ttfts, 99)),
        itl_mean_ms=float(np.mean(all_itls)),
        itl_p50_ms=float(np.percentile(all_itls, 50)),
        itl_p90_ms=float(np.percentile(all_itls, 90)),
        itl_p99_ms=float(np.percentile(all_itls, 99)),
        e2e_mean_ms=float(np.mean(e2es)),
        e2e_p50_ms=float(np.percentile(e2es, 50)),
        e2e_p90_ms=float(np.percentile(e2es, 90)),
        e2e_p99_ms=float(np.percentile(e2es, 99)),
    )


def print_summary_table(results: List[BenchmarkResult]):
    table = Table(title="Multi-GPU Tensor Parallel Benchmark Results")
    table.add_column("Engine", style="cyan")
    table.add_column("TP", justify="center")
    table.add_column("Conc", justify="center")
    table.add_column("Prompt/Out", justify="center")
    table.add_column("APC/Radix", justify="center")
    table.add_column("Req/s", justify="right", style="green")
    table.add_column("Out Tok/s", justify="right", style="green")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("ITL P50 (ms)", justify="right")
    table.add_column("E2E P50 (ms)", justify="right")

    for r in results:
        table.add_row(
            r.engine,
            str(r.tp_size),
            str(r.concurrency),
            f"{r.prompt_len}/{r.output_len}",
            "ON" if r.prefix_caching_enabled else "OFF",
            f"{r.req_per_sec:.2f}",
            f"{r.output_tok_per_sec:.1f}",
            f"{r.ttft_p50_ms:.1f}",
            f"{r.itl_p50_ms:.1f}",
            f"{r.e2e_p50_ms:.1f}"
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Benchmark Runner")
    parser.add_argument("--config", type=str, default="config/bench_config.yaml")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--output", type=str, default="benchmarks/results.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    endpoint_url = f"http://127.0.0.1:{args.port}/v1/completions"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results: List[BenchmarkResult] = []
    concurrency_list = cfg["tp_sweep"]["concurrency_levels"]
    prompt_len = cfg["tp_sweep"]["fixed_prompt_tokens"]
    output_len = cfg["tp_sweep"]["fixed_output_tokens"]

    console.print(f"[bold green]Starting Benchmark on {args.engine.upper()} (TP={args.tp_size})[/bold green]")
    for conc in concurrency_list:
        num_reqs = max(conc * 4, 16)
        console.print(f"-> Testing Concurrency = {conc} ({num_reqs} requests)...")
        res = asyncio.run(
            run_concurrency_sweep(
                endpoint_url=endpoint_url,
                model=model_name,
                engine=args.engine,
                tp_size=args.tp_size,
                concurrency=conc,
                num_requests=num_reqs,
                prompt_len=prompt_len,
                output_len=output_len,
                prefix_caching=args.prefix_caching
            )
        )
        results.append(res)

    print_summary_table(results)

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    console.print(f"[bold blue]Results successfully saved to {args.output}[/bold blue]")


if __name__ == "__main__":
    main()
