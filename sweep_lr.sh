#!/bin/bash
set -e

for lr in 5e-5 7e-5 1e-4 3e-4; do
    echo "=== Running LR=$lr ==="
    uv run python distill_off_policy.py --lr $lr --sweep 50
done

echo "Done! Compare runs in wandb."
