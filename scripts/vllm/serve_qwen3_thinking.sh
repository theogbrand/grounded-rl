#!/bin/bash

# check if argument is provided, otherwise use default
CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3

# # Check if checkpoint path exists
# if [ ! -e "$CHECKPOINT_PATH" ]; then
#     echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
#     exit 1
# fi

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

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi
