#!/bin/bash
# Eval script: run lm-eval benchmarks on 3 GPUs in parallel
# Phase 1: short benchmarks across all models
# Phase 2: long benchmarks on the winner

set -e

GPUS=(0 1 2)
SHORT_BENCHMARKS="gsm8k_cot,arc_easy,truthfulqa_mc2,ifeval"
LONG_BENCHMARKS="drop,minerva_math,mmlu"

MODELS=(
    "allenai/OLMo-2-0425-1B-Instruct"
    "checkpoints/offpolicy-7b-lr3e-5/latest"
    "checkpoints/offpolicy-olmo3-7b/latest"
    "checkpoints/offpolicy-olmo3-7b/prev"
    "checkpoints/offpolicy-olmo3-7b-lr3e-5/latest"
    "checkpoints/offpolicy-olmo3-7b-lr3e-5/prev"
    "checkpoints/onpolicy/latest"
    "checkpoints/onpolicy/prev"
)

log_name() {
    local model=$1
    local parent=$(basename "$(dirname "$model")")
    echo "${parent}-$(basename "$model")"
}

eval_model() {
    local model=$1
    local tasks=$2
    local gpu=$3
    local name=$(log_name "$model")

    echo "[$name] Running lm-eval on GPU $gpu: $tasks"
    CUDA_VISIBLE_DEVICES=$gpu uv run lm_eval \
        --model vllm \
        --model_args "pretrained=$model,dtype=bfloat16,gpu_memory_utilization=0.9" \
        --tasks "$tasks" \
        --apply_chat_template \
        --output_path "logs/$name" \
        --batch_size auto
}

run_batch() {
    local benchmarks=$1
    shift
    local models=("$@")
    local pids=()

    for i in "${!models[@]}"; do
        local gpu_idx=$((i % ${#GPUS[@]}))
        eval_model "${models[$i]}" "$benchmarks" "${GPUS[$gpu_idx]}" &
        pids+=($!)
        sleep 5
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "Warning: a model eval failed (pid $pid)"
    done
}

echo "========== PHASE 1: Short benchmarks (3 GPUs parallel) =========="
for ((i=0; i<${#MODELS[@]}; i+=3)); do
    run_batch "$SHORT_BENCHMARKS" "${MODELS[@]:$i:3}"
done

echo ""
echo "========== Phase 1 Results =========="
uv run python parse_results.py
echo ""
echo "Phase 1 complete! Check results, then uncomment Phase 2 for the best checkpoint."
echo ""

# --- PHASE 2: Uncomment and set BEST_MODEL after Phase 1 ---
# BEST_MODEL="checkpoints/step_200"
# echo "========== PHASE 2: Long benchmarks on $BEST_MODEL =========="
# eval_model "$BEST_MODEL" "$LONG_BENCHMARKS" 0
# eval_model "allenai/OLMo-2-0425-1B-Instruct" "$LONG_BENCHMARKS" 1
