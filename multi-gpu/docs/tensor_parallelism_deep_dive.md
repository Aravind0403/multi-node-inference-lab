# Tensor Parallelism & Multi-GPU Serving Deep Dive

## 1. Mathematical Foundations of Tensor Parallelism (Megatron-LM Style)

Tensor Parallelism (TP) partitions individual weight matrices across $N$ GPUs within a single node, distributing both computational FLOPs and memory footprint without introducing pipeline bubble stages.

```
+-------------------------------------------------------------------------+
|                       Transformer Layer (TP=2)                          |
+-------------------------------------------------------------------------+
| Input Activation X [B, S, H]                                            |
|                                                                         |
| [Attention Block]                                                       |
|   ├── GPU 0: QKV Proj (Col-Parallel) ──> Local Multi-Head Attention     |
|   └── GPU 1: QKV Proj (Col-Parallel) ──> Local Multi-Head Attention     |
|   ├── GPU 0: Out Proj (Row-Parallel) ─┐                                 |
|   └── GPU 1: Out Proj (Row-Parallel) ─┴─> [All-Reduce #1 (NCCL Stream)] |
|                                                    │                    |
| Residual Connection + RMSNorm <────────────────────┘                    |
|                                                                         |
| [MLP Block (SwiGLU)]                                                    |
|   ├── GPU 0: Gate/Up Proj (Col-Parallel) ──> SiLU(Gate) * Up            |
|   └── GPU 1: Gate/Up Proj (Col-Parallel) ──> SiLU(Gate) * Up            |
|   ├── GPU 0: Down Proj (Row-Parallel) ──┐                               |
|   └── GPU 1: Down Proj (Row-Parallel) ──┴─> [All-Reduce #2 (NCCL Stream)]
|                                                    │                    |
| Layer Output <─────────────────────────────────────┘                    |
+-------------------------------------------------------------------------+
```

### 1.1 Column-Parallel Linear Layer
Splits the output weight dimension $H_{out}$ across $N$ GPUs:
$$W = \begin{bmatrix} W_1 & W_2 & \dots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{H_{in} \times \frac{H_{out}}{N}}$$
Each GPU computes its local partition independently:
$$Y_i = X W_i, \quad Y = \begin{bmatrix} Y_1 & Y_2 & \dots & Y_N \end{bmatrix}$$
*Communication required:* **Zero.**

### 1.2 Row-Parallel Linear Layer
Splits the input weight dimension $H_{in}$ across $N$ GPUs:
$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{H_{in}}{N} \times H_{out}}$$
The input $X$ is partitioned along columns as $X = \begin{bmatrix} X_1 & X_2 & \dots & X_N \end{bmatrix}$. Each GPU computes:
$$Y_i = X_i W_i$$
To obtain the complete mathematical result $Y = XW = \sum_{i=1}^N X_i W_i$, an **All-Reduce (SUM)** collective is executed across all $N$ GPUs:
$$Y = \text{All-Reduce-Sum}(Y_i)$$

### 1.3 Communication Invariants & Collectives
- **Collectives Per Layer:** Exactly **2 All-Reduces** per transformer layer (one after Attention Out-Proj, one after MLP Down-Proj).
- **Communication Volume:** In Ring All-Reduce, each GPU sends and receives:
  $$\text{Bytes Transferred per GPU} = 2 \times \left(\frac{N-1}{N}\right) \times M$$
  where $M = B \times S \times H \times \text{sizeof(dtype)}$ is the activation tensor size in bytes.

---

## 2. CUDA Streams, Hardware Concurrency & Overlap

### 2.1 The CUDA Execution Engine Architecture
Modern NVIDIA GPUs (Ampere, Hopper, Blackwell) contain distinct hardware engines:
1. **Streaming Multiprocessors (SMs):** Execute compute kernels (GEMM, FlashAttention, LayerNorm).
2. **DMA Copy Engines (H2D / D2H):** Bidirectional asynchronous host-to-device and device-to-host memory copies.
3. **NVLink / High-Speed Interconnect Engines:** Inter-GPU peer-to-peer data transfers.

```
       Host CPU
          │
  ┌───────┴────────┐
  ▼                ▼
Compute Stream   NCCL Stream
  │                │
  ▼                ▼
[GEMM on SMs]    [All-Reduce on NVLink]  <── Overlapped in Hardware!
  │                │
  └───────┬────────┘
          ▼
     Synchronize
```

### 2.2 Default Stream vs Non-Default Streams
- **Default Stream (`Stream 0`):** Implicitly synchronizes with all other streams on the device. Placing both compute and communication on Stream 0 strictly serializes execution.
- **Dedicated Communication Streams:** Serving engines (vLLM, SGLang) allocate a dedicated `cudaStream_t` for NCCL collectives.
- **CUDA Events & Dependencies:** `cudaStreamWaitEvent()` allows the communication stream to wait only until the local GEMM completes, leaving other compute streams free to continue background work.

### 2.3 Profiling with Nsight Systems (`nsys`)
When inspecting an `nsys` profile trace:
- Look for `ncclKernel_AllReduce_RING_LL` or `ncclKernel_AllReduce_TREE` on the NCCL stream row.
- **Healthy Trace:** Compute kernels (`volta_gemm`, `ampere_bf16_gemm`) on Stream A overlap with NCCL ring traffic on Stream B.
- **Stall/Bubble Trace:** Long gaps where SMs are idle at 0% utilization while waiting for `cudaStreamSynchronize` or slow PCIe bus transfers.

---

## 3. Communication vs Compute: Bottleneck Analysis

| Serving Regime | Dominant Bottleneck | TP Scaling Behavior | NVLink vs PCIe Impact |
|---|---|---|---|
| **Prefill (TTFT)** | **Compute (FLOPs)** | Near-linear speedup ($~1.8\times-1.9\times$ on TP=2). Matrix multiplications have high arithmetic intensity ($O(B \cdot S \cdot H^2)$ vs $O(B \cdot S \cdot H)$ comm). | Moderate (Compute time outweighs all-reduce latency). |
| **Decode (Batch=1)** | **Memory Latency & Comm Overhead** | Minimal or negative speedup. Each token generates small all-reduce messages ($M = 1 \times 1 \times H$), where NCCL fixed launch latency (~5-15µs) dominates. | **Severe** (PCIe adds substantial latency penalty; NVLink keeps all-reduce < 8µs). |
| **Decode (High Batch $\ge 16$)** | **Memory Bandwidth (HBM)** | High speedup ($~1.7\times-1.9\times$). Aggregate HBM bandwidth doubles, allowing larger batches to fit and stream in parallel. | High (Faster all-reduce prevents collective stalls). |

---

## 4. Multi-GPU Prefix Caching: vLLM vs SGLang

When multiple requests share common prefixes (system prompts, few-shot examples, document context) across TP ranks:

### vLLM: Automatic Prefix Caching (APC)
- **Granularity:** Fixed block-level (e.g., 16 tokens per block).
- **Mechanism:** Computes hash of block tokens + parent block hash.
- **TP Execution:** Each TP rank stores a partition of the KV heads for that block. Block allocation is synchronized across ranks.

### SGLang: RadixAttention
- **Granularity:** Arbitrary token sequence matching using a Radix Tree (trie) data structure.
- **Mechanism:** Maintains an explicit Radix Tree in CPU host memory that maps prefix tokens to GPU KV memory pages. Supports tree-based branching, prefix insertion, and LRU eviction.
- **TP Execution:** Tree operations run on host rank 0; KV cache allocations are broadcasted to all TP worker ranks.

---

## 5. Interview Cheat Sheet & Key Questions

### Q1: Why are there exactly 2 All-Reduces per transformer layer in Megatron TP?
> **Answer:** In Megatron-LM tensor parallelism:
> 1. In Attention, QKV projection is column-parallel (no comm), and Output projection is row-parallel, producing partial sums that require **1 All-Reduce**.
> 2. In MLP (SwiGLU), Gate and Up projections are column-parallel (no comm), and Down projection is row-parallel, producing partial sums that require **1 All-Reduce**.
> Total: Exactly 2 All-Reduces per layer, regardless of the number of GPUs $N$.

### Q2: Why is TP rarely used across nodes over standard Ethernet?
> **Answer:** TP executes 2 All-Reduces per layer for every single generated token. In a 32-layer model generating 100 tokens, that's $6,400$ All-Reduce operations. Standard Ethernet / PCIe latency (~10-50µs per collective) accumulates massive latency bubbles. TP requires intra-node NVLink/NVSwitch bandwidth (600–900 GB/s, <5µs latency). Across nodes, Pipeline Parallelism (PP) or Data Parallelism (DP) is used because they only communicate between layer boundaries or batch steps.

### Q3: Why does TP=2 sometimes increase latency for batch size 1 decode?
> **Answer:** In batch=1 token generation, activation sizes are tiny ($B=1, S=1$), meaning compute takes only a few microseconds. The fixed overhead of launching 2 NCCL All-Reduce kernels per layer over the interconnect can exceed the compute time saved by splitting the matmul in half.
