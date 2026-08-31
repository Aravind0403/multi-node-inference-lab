#!/usr/bin/env bash
# ==============================================================================
# Script: run_vllm_server.sh
# Description: Launch vLLM OpenAI-Compatible Server with Tensor Parallelism
# ==============================================================================

set -euo pipefail

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
TP_SIZE=${2:-2}
PORT=${3:-8000}
ENABLE_APC=${4:-"true"} # Automatic Prefix Caching
GPU_IDS=${5:-"0,1"}

echo "============================================================"
echo " Starting vLLM Server"
echo " Model:               $MODEL_NAME"
echo " Tensor Parallel:     $TP_SIZE"
echo " CUDA Visible Devices: $GPU_IDS"
echo " Prefix Caching (APC):$ENABLE_APC"
echo " Port:                $PORT"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export NCCL_DEBUG=INFO # Logs NCCL ring/tree topology, helpful for checking NVLink vs PCIe
export VLLM_LOGGING_LEVEL=INFO

EXTRA_ARGS=""
if [ "$ENABLE_APC" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --enable-prefix-caching"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    --host "0.0.0.0" \
    --dtype "bfloat16" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --trust-remote-code \
    $EXTRA_ARGS
