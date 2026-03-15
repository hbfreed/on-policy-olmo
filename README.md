# On-Policy Distillation: OLMo 2

On-policy distillation of OLMo 2 1B using OLMo 3 7B as the teacher, following [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) — the student generates outputs and the teacher provides soft labels on the student's own distribution.

| Role    | Model       |
|---------|-------------|
| Student | OLMo 2 1B   |
| Teacher | OLMo 3 7B   |

### GPU layout

| GPU | Role |
|-----|------|
| 0 | vLLM student (generation) |
| 1 | HF teacher (inference) |
| 2+ | HF student (training, DDP when >1) |

Single-GPU training uses 3 GPUs total; each additional DDP rank adds one more.

### Launch

```bash
# Single-GPU student (3 GPUs total)
torchrun --nproc_per_node=1 distill_on_policy_ddp.py

# DDP with 2 student ranks (4 GPUs total)
torchrun --nproc_per_node=2 distill_on_policy_ddp.py

# Via launch script (passes extra args through)
uv run bash launch_ddp.sh --lr 1e-5 --micro-batch-size 32

# Quick smoke test
uv run bash launch_ddp.sh --sweep 2 --micro-batch-size 4
```

### CLI args

| Arg | Default | Description |
|-----|---------|-------------|
| `--lr` | 1e-5 | Learning rate |
| `--micro-batch-size` | 2 | Per-rank micro-batch size for student training |
| `--teacher-micro-batch-size` | 6 | Micro-batch size for teacher inference |
| `--sweep` | None | Limit to N training steps (for debugging) |
| `--wandb-run-id` | None | Resume a W&B run |

### Evals

`evals.py` runs mid-training benchmarks via lm-eval-harness, reusing the existing vLLM engine so no extra GPU is needed.

## Project structure

```
distill_on_policy.py        # single-GPU on-policy distillation
distill_on_policy_ddp.py    # DDP on-policy distillation (also works single-GPU)
distill_off_policy.py       # off-policy distillation
evals.py                    # mid-training eval harness
launch_ddp.sh               # DDP launch wrapper
```
