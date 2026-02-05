#!/bin/bash
# Simple eval script: start vLLM servers, run benchmarks

set -e
trap 'kill $(jobs -p) 2>/dev/null' EXIT

BENCHMARKS="bbh drop gsm8k ifeval math mmlu truthfulqa hellaswag arc_challenge winogrande"

# Start vLLM servers in background
CUDA_VISIBLE_DEVICES=1 uv run vllm serve hbfreed/Olmo-2-1B-Distilled --port 8000 --api-key "" &
sleep 15
CUDA_VISIBLE_DEVICES=2 uv run vllm serve allenai/OLMo-2-0425-1B-Instruct --port 8001 --api-key "" &

# Wait for servers to be ready
echo "Waiting for vLLM servers..."
until curl -s http://localhost:8000/health >/dev/null; do sleep 2; done
until curl -s http://localhost:8001/health >/dev/null; do sleep 2; done
echo "Servers ready!"

# Run evals
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8000/v1 uv run bench eval $BENCHMARKS --model vllm/hbfreed/Olmo-2-1B-Distilled
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8001/v1 uv run bench eval $BENCHMARKS --model vllm/allenai/OLMo-2-0425-1B-Instruct
