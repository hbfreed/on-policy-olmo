#!/bin/bash
# LR sweep for on-policy distillation (Liger/DeepSeek full-vocab reverse KL).
# Known: 1e-5 too low, 3e-4 worked → loss → 0.0X. Sweeping around/above 3e-4.
set -e

for lr in 1e-4 3e-4 1e-3 3e-3; do
    echo "=== Running LR=$lr ==="
    uv run python distill_on_policy.py \
        --lr "$lr" \
        --sweep 50 \
        --wandb-project on-policy-distill-sweeps \
        --checkpoint-base "checkpoints/sweep-lr$lr"
done

echo "Done! Compare runs in wandb."
