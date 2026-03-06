#!/bin/bash
set -e

for lr in 1e-6 5e-6 1e-5 5e-5; do
    echo "=== Running LR=$lr ==="
    uv run python distill_on_policy.py --lr $lr --sweep 20
done

echo "Done! Compare runs in wandb."
