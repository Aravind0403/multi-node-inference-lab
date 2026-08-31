#!/usr/bin/env bash
# ==============================================================================
# Script: run_sglang_server.sh
# Description: Launch SGLang Server with Tensor Parallelism & RadixAttention
# ==============================================================================

set -euo pipefail

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
TP_SIZE=${2:-2}
PORT=${3:-8000}
ENABLE_RADIX=${4:-"true"}
GPU_IDS=${5:-"0,1"}

echo "============================================================"
echo " Starting SGLang Server"
echo " Model:               $MODEL_NAME"
echo " Tensor Parallel:     $TP_SIZE"
echo " CUDA Visible Devices: $GPU_IDS"
echo " Radix Cache:         $ENABLE_RADIX"
echo " Port:                $PORT"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export NCCL_DEBUG=INFO

EXTRA_ARGS=""
if [ "$ENABLE_RADIX" = "false" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --disable-radix-cache"
fi

python3 -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --tp "$TP_SIZE" \
    --port "$PORT" \
    --host "0.0.0.0" \
    --mem-fraction-static 0.90 \
    --context-length 8192 \
    --trust-remote-code \
    $EXTRA_ARGS
