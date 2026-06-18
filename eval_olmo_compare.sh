#!/bin/bash
# Parallel eval: baseline + distilled-from-olmo3 + teacher.
# 3 models on 3 GPUs simultaneously.
set -e

TASKS="ifeval,arc_easy,arc_challenge,piqa,hellaswag,gsm8k_cot"

# entry format: name|path|chat (chat=1 to apply chat template, 0 to skip)
MODELS=(
    "baseline-olmo-1b-base|allenai/OLMo-2-0425-1B|0"
    "baseline-olmo-1b-instruct|allenai/OLMo-2-0425-1B-Instruct|1"
    "distilled-from-base|checkpoints/olmo2-7b-to-olmo2-1b-base-revkl/latest|1"
)

eval_model() {
    local entry=$1
    local gpu=$2
    IFS='|' read -r name path chat <<< "$entry"
    local chat_flag=""
    [ "$chat" = "1" ] && chat_flag="--apply_chat_template"

    echo "[$name] GPU $gpu  ←  $path  (chat=$chat)"
    CUDA_VISIBLE_DEVICES=$gpu uv run lm_eval \
        --model vllm \
        --model_args "pretrained=$path,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=2048" \
        --tasks "$TASKS" \
        $chat_flag \
        --batch_size auto \
        --output_path "logs/eval-olmo/$name" \
        > "logs/eval-olmo/$name.log" 2>&1
}

mkdir -p logs/eval-olmo

# Launch all 3 in parallel
pids=()
i=0
for entry in "${MODELS[@]}"; do
    eval_model "$entry" "$i" &
    pids+=($!)
    i=$((i+1))
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo ""
echo "Done! Results in logs/eval-olmo/"
echo ""
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    echo "================== $name =================="
    grep -E "^\|" "logs/eval-olmo/$name.log" 2>/dev/null \
        | tr '\r' '\n' \
        | grep -E "^\|.*\|.*[0-9]" \
        | grep -vE "Tasks|none.*none" \
        | head -20
    echo ""
done
