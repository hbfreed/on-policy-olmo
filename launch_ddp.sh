#!/usr/bin/env bash
# Launch DDP on-policy distillation with 2 student ranks on GPUs 2,3.
# Usage: uv run bash launch_ddp.sh [extra args...]
#
# Examples:
#   uv run bash launch_ddp.sh --micro-batch-size 32 --teacher-micro-batch-size 32 --lr 1e-5
#   uv run bash launch_ddp.sh --sweep 2 --micro-batch-size 4  # smoke test

torchrun --nproc_per_node=2 distill_on_policy_ddp.py "$@"
