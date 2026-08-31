# Tensor Parallelism Scaling & Speedup Analysis

## Throughput & Latency Scaling (TP=1 vs TP=2)

| Engine | Concurrency | TP Size | Out Tok/s | TTFT P50 (ms) | ITL P50 (ms) | Speedup (Tok/s) |
|---|---|---|---|---|---|---|
| vllm | 1 | 2 | 35.1 | 38.6 | 31.6 | N/A |
| vllm | 2 | 2 | 47.4 | 113.0 | 39.5 | N/A |
| vllm | 4 | 2 | 112.2 | 121.7 | 39.2 | N/A |
| vllm | 8 | 2 | 176.0 | 162.7 | 44.8 | N/A |
| vllm | 16 | 2 | 498.3 | 186.9 | 32.1 | N/A |
| vllm | 32 | 2 | 1178.7 | 232.8 | 21.1 | N/A |
| vllm | 64 | 2 | 2488.2 | 282.3 | 21.2 | N/A |