#!/bin/bash

# vLLM Server Launch Script for Gemma 3 27B with LoRA
# Base Model: unsloth/gemma-3-27b-it-unsloth-bnb-4bit (4-bit quantized)
# Adapter: ./gemma2_neobiotech_adapter

echo "Starting vLLM server..."
echo "Model: unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
echo "Adapter: gemma-3-finetuned"

# Note: --quantization bitsandbytes is required to load the 4-bit base model
# --tensor-parallel-size 1 is used as 27B 4-bit fits in ~16GB VRAM (one 4090 has 24GB)
# If OOM occurs, try --tensor-parallel-size 2

python -m vllm.entrypoints.openai.api_server \
    --model unsloth/gemma-3-27b-it-unsloth-bnb-4bit \
    --enable-lora \
    --lora-modules gemma-3-finetuned=./gemma2_neobiotech_adapter \
    --served-model-name gemma-3-finetuned \
    --port 8000 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --quantization bitsandbytes \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --max-lora-rank 64
