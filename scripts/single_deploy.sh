#!/bin/bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LORAS=20
ROOT="meta-llama-3-8b-instruct"
MAX_LORA_RANK=16
GPU_ID=0

PORT=8000

echo "Deploying model $MODEL with $MAX_LORAS LORAs"
echo "Starting API servers..."

mkdir -p vllm_logs/$ROOT

COMMON_ARGS="--model $MODEL \
    --trust-remote-code \
    --enable-lora \
    --seed 42 \
    --max-lora-rank $MAX_LORA_RANK \
    --gpu-memory-utilization 0.45 \
    --max-loras $MAX_LORAS \
    --max-cpu-loras $MAX_LORAS \
    --max-model-len 4096"


CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
   $COMMON_ARGS \
   --port $PORT > vllm_logs/$ROOT/port.log 2>&1 &
 

echo "Waiting for servers to start ..."
sleep 30
echo "All API servers have been started"