#!/usr/bin/env python3
"""
profile_tp_kernels.py
Standalone PyTorch script that implements Megatron-style Tensor Parallelism for a Transformer Layer.
Uses NVTX annotations and separate CUDA streams for compute and NCCL All-Reduce collectives
to enable detailed Nsight Systems (nsys) timeline inspection.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer (Splits output dimension across GPUs).
    Formula: Y_i = X * W_i
    No communication required after matrix multiplication!
    """
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_push("ColumnParallelMatMul")
        out = F.linear(x, self.weight, self.bias)
        torch.cuda.nvtx.range_pop()
        return out


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer (Splits input dimension across GPUs).
    Formula: Y = Sum_i (X_i * W_i)
    Requires an All-Reduce sum across all GPUs to obtain the final output!
    """
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int, comm_stream: torch.cuda.Stream = None):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.comm_stream = comm_stream
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_push("RowParallelMatMul")
        partial_out = F.linear(x, self.weight)
        torch.cuda.nvtx.range_pop()

        # All-Reduce collective to sum partial results across GPUs
        torch.cuda.nvtx.range_push("NCCL_AllReduce")
        if self.comm_stream is not None:
            # Overlap pattern: wait for compute on compute stream, issue on comm stream
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                dist.all_reduce(partial_out, op=dist.ReduceOp.SUM)
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        else:
            dist.all_reduce(partial_out, op=dist.ReduceOp.SUM)
        torch.cuda.nvtx.range_pop()

        if self.bias is not None and self.rank == 0:
            partial_out += self.bias
        return partial_out


class TensorParallelMLP(nn.Module):
    """
    Standard SwiGLU / MLP block with TP=2 sharding:
    1. Gate/Up Proj: Column-parallel (No comm)
    2. Activation: SiLU(Gate) * Up
    3. Down Proj: Row-parallel + All-Reduce (Comm #1 in MLP)
    """
    def __init__(self, hidden_dim: int, ffn_dim: int, world_size: int, rank: int, comm_stream: torch.cuda.Stream):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(hidden_dim, ffn_dim * 2, world_size, rank)
        self.down_proj = RowParallelLinear(ffn_dim, hidden_dim, world_size, rank, comm_stream)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_push("MLP_Block")
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        out = self.down_proj(act)
        torch.cuda.nvtx.range_pop()
        return out


def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return local_rank, world_size


def run_profile(batch_size: int = 16, seq_len: int = 512, hidden_dim: int = 4096, ffn_dim: int = 14336, iterations: int = 20):
    local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    # Dedicated communication stream for NCCL collectives
    comm_stream = torch.cuda.Stream(device=device)

    mlp = TensorParallelMLP(hidden_dim, ffn_dim, world_size, local_rank, comm_stream).to(device=device, dtype=torch.bfloat16)

    # Input activation tensor
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        _ = mlp(x)
    torch.cuda.synchronize()

    print(f"[Rank {local_rank}/{world_size}] Starting {iterations} profiled iterations...")

    torch.cuda.nvtx.range_push("Profiled_Forward_Loop")
    for i in range(iterations):
        torch.cuda.nvtx.range_push(f"Iteration_{i}")
        out = mlp(x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    print(f"[Rank {local_rank}/{world_size}] Profiling complete. Output shape: {out.shape}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_profile()
