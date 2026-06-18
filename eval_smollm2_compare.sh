#!/bin/bash
# Compare SmolLM2 distillation results via Eleuther lm-eval-harness + vllm.
# Runs 5 models on instruction-following + common-sense tasks.
# Internally consistent comparison; absolute numbers won't match HF's lighteval reports.
set -e

TASKS="ifeval,arc_easy,arc_challenge,piqa,hellaswag"

MODELS=(
    "baseline-135m|HuggingFaceTB/SmolLM2-135M-Instruct"
    "distilled-from-360m|checkpoints/smollm2-360m-to-135m-revkl/latest"
    "distilled-from-1.7b|checkpoints/smollm2-1_7b-to-135m-revkl/latest"
    "teacher-360m|HuggingFaceTB/SmolLM2-360M-Instruct"
    "teacher-1.7b|HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

eval_model() {
    local entry=$1
    local gpu=$2
    local name="${entry%%|*}"
    local path="${entry##*|}"

    echo "[$name] GPU $gpu  ←  $path"
    CUDA_VISIBLE_DEVICES=$gpu uv run lm_eval \
        --model vllm \
        --model_args "pretrained=$path,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=2048" \
        --tasks "$TASKS" \
        --apply_chat_template \
        --batch_size auto \
        --output_path "logs/eval-smollm2/$name" \
        > "logs/eval-smollm2/$name.log" 2>&1
}

mkdir -p logs/eval-smollm2

# 3 in parallel on GPUs 0,1,2
i=0
while [ $i -lt ${#MODELS[@]} ]; do
    pids=()
    for gpu in 0 1 2; do
        if [ $i -lt ${#MODELS[@]} ]; then
            eval_model "${MODELS[$i]}" "$gpu" &
            pids+=($!)
            i=$((i+1))
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
done

echo "Done! Results in logs/eval-smollm2/"
echo "Quick summary:"
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    echo "  --- $name ---"
    grep -E "^\| (ifeval|arc_easy|arc_challenge|piqa|hellaswag)" "logs/eval-smollm2/$name.log" 2>/dev/null | head -10
done
