#!/usr/bin/env bash
# ==============================================================================
# Script: profile_nsys.sh
# Description: Profile Tensor Parallel Forward Pass with NVIDIA Nsight Systems
# Captures CUDA streams, cuBLAS GEMMs, NCCL All-Reduce, and NVTX markers.
# ==============================================================================

set -euo pipefail

NUM_GPUS=${1:-2}
OUTPUT_NAME=${2:-"tp2_kernel_trace"}
OUTPUT_DIR="benchmarks/nsys_traces"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Launching Nsight Systems Profiling"
echo " GPUs:       $NUM_GPUS"
echo " Output:     $OUTPUT_DIR/${OUTPUT_NAME}.nsys-rep"
echo "============================================================"

# Check if nsys is available in PATH
if ! command -v nsys &> /dev/null; then
    echo "[!] Warning: 'nsys' CLI not found in PATH."
    echo "[!] Run this script on an NVIDIA CUDA environment with Nsight Systems installed."
    echo "[!] Executing torchrun directly for syntax & execution verification..."
    torchrun --nproc_per_node="$NUM_GPUS" scripts/profile_tp_kernels.py
    exit 0
fi

# Run nsys profiling with torchrun
nsys profile \
    --output="$OUTPUT_DIR/$OUTPUT_NAME" \
    --trace=cuda,nvtx,osrt,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    torchrun --nproc_per_node="$NUM_GPUS" scripts/profile_tp_kernels.py

echo "============================================================"
echo " Trace captured: $OUTPUT_DIR/${OUTPUT_NAME}.nsys-rep"
echo " View in Nsight Systems GUI or export stats with:"
echo " nsys stats --report cuda_api_sum,cuda_gpu_kern_sum $OUTPUT_DIR/${OUTPUT_NAME}.nsys-rep"
echo "============================================================"
