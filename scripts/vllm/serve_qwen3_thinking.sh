#!/bin/bash

# Script to serve Qwen3-VL-235B-A22B-Thinking-FP8 with reasoning support

CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"

mkdir -p vllm_logs

# Fix CUDA_VISIBLE_DEVICES if it contains UUIDs
if [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen3_thinking_vllm" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=30" \
    --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --api-key "qwen3" > "vllm_logs/vllm_qwen3_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi
