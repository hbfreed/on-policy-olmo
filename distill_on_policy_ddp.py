"""DDP on-policy distillation: N student GPUs (ranks 0..N-1) on cuda:2..2+N-1.

GPU layout (nproc_per_node=K):
  GPU 0:        vLLM student (generation)
  GPU 1:        HF teacher (inference)
  GPU 2 .. K+1: DDP student ranks (training)

Works for single-GPU training too — just set nproc_per_node=1.

Usage:
  torchrun --nproc_per_node=1 distill_on_policy_ddp.py          # single-GPU (3 GPUs total)
  torchrun --nproc_per_node=2 distill_on_policy_ddp.py          # DDP (4 GPUs total)
  uv run bash launch_ddp.sh --sweep 2 --micro-batch-size 4      # smoke test
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Queue

import bitsandbytes as bnb
import torch
import torch.distributed as dist
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from vllm import LLM

import wandb
from distill_utils import (
    BATCH_SIZE,
    CLIP_EPS,
    DATASET,
    DEBUG_MODE,
    EVAL_EVERY_N_STEPS,
    EVAL_N_SAMPLES,
    EVAL_TASKS,
    GRAD_ACCUM_STEPS,
    GROUP_SIZE,
    HUB_REPO,
    LR,
    MAX_CONTEXT_LENGTH,
    MAX_GRAD_NORM,
    N_EPOCHS,
    N_SAMPLE_PROMPTS,
    RUN_NAME,
    SAMPLE_EVERY_N_STEPS,
    STUDENT,
    SYNC_EVERY_N_STEPS,
    TEACHER,
    WANDB_PROJECT,
    WARMUP_STEPS,
    build_loss_mask,
    generate_rollouts,
    generate_samples,
    get_logprobs_at_tokens,
    get_sync_interval,
    load_checkpoint,
    prepare_prompts,
    run_teacher_pipeline,
    save_checkpoint,
    timed_generate_rollouts,
    timed_sync_weights_to_vllm,
)
from evals import run_evals

TEACHER_DEVICE = "cuda:1"
DDP_GPU_OFFSET = 2  # rank i -> cuda:{i + DDP_GPU_OFFSET}


def broadcast_rollout_data(rank, world_size, device, sequences, attention_mask,
                           old_student_logprobs, teacher_logprobs_all,
                           hit_eos, prompt_lens_t):
    """Broadcast all rollout tensors from rank 0 to all ranks via NCCL.

    Tensors come in on CPU from rank 0; moved to CUDA for broadcast, returned on CPU.
    """
    # 1) Broadcast shape info so non-zero ranks can allocate buffers
    if rank == 0:
        shape_info = torch.tensor([sequences.shape[0], sequences.shape[1]],
                                  dtype=torch.long, device=device)
    else:
        shape_info = torch.empty(2, dtype=torch.long, device=device)
    dist.broadcast(shape_info, src=0)
    n_seq, seq_len = shape_info.tolist()

    # 2) Build tensors on CUDA for NCCL broadcast
    def _to_device(t, shape, dtype):
        if rank == 0:
            return t.to(device)
        return torch.empty(shape, dtype=dtype, device=device)

    sequences = _to_device(sequences, (n_seq, seq_len), torch.long)
    attention_mask = _to_device(attention_mask, (n_seq, seq_len), torch.long)
    old_student_logprobs = _to_device(old_student_logprobs, (n_seq, seq_len), torch.float32)
    teacher_logprobs_all = _to_device(teacher_logprobs_all, (n_seq, seq_len - 1), torch.float32)
    # NCCL doesn't support bool; cast to uint8 for broadcast
    if rank == 0:
        hit_eos = hit_eos.to(torch.uint8).to(device)
    else:
        hit_eos = torch.empty(n_seq, dtype=torch.uint8, device=device)
    prompt_lens_t = _to_device(prompt_lens_t, (n_seq,), torch.long)

    # 3) Broadcast all tensors
    dist.broadcast(sequences, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(old_student_logprobs, src=0)
    dist.broadcast(teacher_logprobs_all, src=0)
    dist.broadcast(hit_eos, src=0)
    dist.broadcast(prompt_lens_t, src=0)

    # Return on CPU to avoid double-storing on GPU (we'll .to(device) selectively later)
    return (sequences.cpu(), attention_mask.cpu(), old_student_logprobs.cpu(),
            teacher_logprobs_all.cpu(), hit_eos.cpu().bool(), prompt_lens_t.cpu())


def collect_teacher_logprobs(teacher, sequences, attention_mask,
                             teacher_micro_batch_size, student_device):
    """Run teacher pipeline synchronously, return full [N, seq_len-1] tensor."""
    queue = Queue(maxsize=4)
    # Run in current thread — teacher has its own GPU, no contention
    run_teacher_pipeline(teacher, sequences, attention_mask,
                         teacher_micro_batch_size, TEACHER_DEVICE,
                         student_device, queue,
                         consumer_chunk_size=len(sequences))
    result = queue.get()
    if isinstance(result, BaseException):
        raise result
    sentinel = queue.get()
    assert sentinel is None
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="DDP on-policy distillation")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--micro-batch-size", type=int, default=2,
                        help="Sequences per student forward pass")
    parser.add_argument("--teacher-micro-batch-size", type=int, default=6,
                        help="Sequences per teacher forward pass")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps (for quick sweeps)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume")
    parser.add_argument("--checkpoint-base", type=str,
                        default="checkpoints/onpolicy-from-baseline",
                        help="Base directory for checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save rolling checkpoint every N steps")
    parser.add_argument("--milestone-every", type=int, default=500,
                        help="Save permanent milestone checkpoint every N steps")
    return parser.parse_args()


def main_ddp():
    args = parse_args()
    lr = args.lr
    micro_batch_size = args.micro_batch_size
    teacher_micro_batch_size = args.teacher_micro_batch_size
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id
    checkpoint_base = args.checkpoint_base
    resume_from = args.resume_from
    save_every = args.save_every
    milestone_every = args.milestone_every

    # DDP init
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank + DDP_GPU_OFFSET}"
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"DDP: {world_size} ranks, devices cuda:{DDP_GPU_OFFSET}..cuda:{DDP_GPU_OFFSET + world_size - 1}")

    # Tokenizer + dataset (all ranks)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds = load_dataset(DATASET, split="train")
    dataset = (
        ds.select_columns(["prompt", "input_ids_prompt"])
        .filter(lambda x: len(x["input_ids_prompt"]) < MAX_CONTEXT_LENGTH)
        .shuffle(seed=1223)
    )

    if DEBUG_MODE:
        from datasets import concatenate_datasets
        single = dataset.select(range(1))
        dataset = concatenate_datasets([single] * GRAD_ACCUM_STEPS)
        if rank == 0:
            print(f"DEBUG MODE: 1 prompt repeated {GRAD_ACCUM_STEPS}x")

    n_epochs = 20 if DEBUG_MODE else N_EPOCHS
    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * n_epochs

    # Student model (all ranks, each on its own GPU)
    if rank == 0:
        print(f"Loading student model from {resume_from or STUDENT}...")
    student_src = resume_from if resume_from else STUDENT
    student = AutoLigerKernelForCausalLM.from_pretrained(
        student_src, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    student.gradient_checkpointing_enable()
    SHARED_VOCAB_SIZE = student.config.vocab_size

    student = DDP(student, device_ids=[rank + DDP_GPU_OFFSET])

    optimizer = bnb.optim.AdamW8bit(
        student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8
    )
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    # Rank 0 only: teacher, vLLM, wandb, eval prompts
    teacher = None
    vllm_student = None
    vllm_executor = None
    checkpoint_executor = None
    eval_prompts = None

    if rank == 0:
        print(f"Loading teacher model from {TEACHER}...")
        teacher = AutoLigerKernelForCausalLM.from_pretrained(
            TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(TEACHER_DEVICE)
        teacher.eval()
        print(f"Student vocab: {SHARED_VOCAB_SIZE}, Teacher vocab: {teacher.config.vocab_size}")

        print("Loading vLLM student on cuda:0...")
        vllm_student = LLM(
            STUDENT, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
        )

        eval_prompts = [
            dataset[i]["input_ids_prompt"]
            for i in range(min(N_SAMPLE_PROMPTS, len(dataset)))
        ]

        vllm_executor = ThreadPoolExecutor(max_workers=1)
        checkpoint_executor = ThreadPoolExecutor(max_workers=1)

    # Resume from checkpoint
    start_step = 0
    if resume_from and rank == 0:
        start_step = load_checkpoint(resume_from, student.module, optimizer, vllm_student)
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step}")

    # Broadcast start_step from rank 0
    start_step_t = torch.tensor([start_step], dtype=torch.long, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = start_step_t.item()
    if start_step > 0 and rank != 0:
        for _ in range(start_step):
            scheduler.step()

    # Wandb init (rank 0 only)
    if rank == 0:
        def short_name(model_name: str) -> str:
            return model_name.split("/")[-1]

        run_name = RUN_NAME
        if run_name is None:
            student_short = short_name(STUDENT).lower().replace("olmo-2-0425-", "olmo")
            lr_str = f"{lr:.0e}".replace("-0", "-")
            sweep_tag = f"-sweep{sweep_steps}" if sweep_steps else ""
            run_name = (
                f"{student_short}-onpolicy-ddp{world_size}"
                f"-lr{lr_str}-clip{CLIP_EPS}-sync{SYNC_EVERY_N_STEPS}{sweep_tag}"
            )

        wandb.init(
            project=WANDB_PROJECT,
            id=wandb_run_id,
            name=run_name,
            config={
                "teacher": TEACHER, "student": STUDENT,
                "batch_size": BATCH_SIZE, "group_size": group_size,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "micro_batch_size": micro_batch_size,
                "teacher_micro_batch_size": teacher_micro_batch_size,
                "ddp_world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
                "n_epochs": n_epochs, "total_steps": total_steps,
                "lr": lr, "sweep_steps": sweep_steps,
                "clip_eps": CLIP_EPS, "max_grad_norm": MAX_GRAD_NORM,
                "warmup_steps": WARMUP_STEPS, "max_context_length": max_context,
            },
            resume="must" if wandb_run_id else "allow",
        )

        baseline_table = generate_samples(vllm_student, eval_prompts, tokenizer, max_context)
        wandb.log({"eval/samples": baseline_table}, step=0)

    # Training loop
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=steps_per_epoch * n_epochs - start_step, desc="Training",
                disable=(rank != 0))

    # Rank 0 async state
    checkpoint_future = None
    sync_future = None
    sample_future = None
    eval_future = None
    gen_future = None
    last_sync_duration = None
    sync_interval = SYNC_EVERY_N_STEPS
    sync_state = {"steps_since_decrease": 0}

    n_sequences_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * group_size
    n_micro_batches = n_sequences_per_step // micro_batch_size
    assert n_micro_batches % world_size == 0, (
        f"n_micro_batches ({n_micro_batches}) must be divisible by world_size ({world_size}). "
        f"Adjust --micro-batch-size or GRAD_ACCUM_STEPS."
    )
    mbs_per_rank = n_micro_batches // world_size

    for epoch in range(n_epochs):
        all_batches = list(dataset.iter(batch_size=BATCH_SIZE))

        for opt_step_idx in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + opt_step_idx
            if current_step < start_step:
                continue

            # === Rank 0: generate rollouts + teacher logprobs ===
            sequences = None
            attention_mask = None
            old_student_logprobs = None
            teacher_logprobs_all = None
            hit_eos = None
            prompt_lens_t = None

            if rank == 0:
                # Drain in-flight vLLM work
                if sync_future is not None:
                    last_sync_duration = sync_future.result()
                    sync_future = None
                if sample_future is not None:
                    wandb.log({"eval/samples": sample_future.result()})
                    sample_future = None
                if eval_future is not None:
                    wandb.log(eval_future.result())
                    eval_future = None

                opt_step_start_time = time.time()

                # Use prefetched generation or generate synchronously
                if gen_future is not None:
                    sequences, prompt_lens, old_student_logprobs, attention_mask, gen_time = (
                        gen_future.result()
                    )
                    gen_future = None
                else:
                    prompts = prepare_prompts(opt_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS)
                    gen_start = time.time()
                    sequences, prompt_lens, old_student_logprobs, attention_mask = (
                        generate_rollouts(
                            vllm_student, prompts, PAD_TOKEN_ID,
                            group_size, max_context, SHARED_VOCAB_SIZE,
                        )
                    )
                    gen_time = time.time() - gen_start

                # Prefetch next step
                next_step_idx = opt_step_idx + 1
                will_sync = (global_step + 1) % sync_interval == 0
                can_prefetch = (
                    next_step_idx < steps_per_epoch
                    and sync_interval > 1
                    and not will_sync
                )
                if can_prefetch:
                    next_prompts = prepare_prompts(
                        next_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS,
                    )
                    gen_future = vllm_executor.submit(
                        timed_generate_rollouts,
                        vllm_student, next_prompts, PAD_TOKEN_ID,
                        group_size, max_context, SHARED_VOCAB_SIZE,
                    )

                # hit_eos detection
                positions = torch.arange(sequences.shape[1]).unsqueeze(0)
                prompt_lens_tensor = torch.tensor(prompt_lens).unsqueeze(1)
                completion_mask = (positions >= prompt_lens_tensor) & (sequences != PAD_TOKEN_ID)
                hit_eos = ((sequences == tokenizer.eos_token_id) & completion_mask).any(dim=1)
                prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long)

                # Collect ALL teacher logprobs before broadcast
                teacher_logprobs_all = collect_teacher_logprobs(
                    teacher, sequences, attention_mask,
                    teacher_micro_batch_size, device,
                )
                # Move teacher logprobs to CPU for broadcast
                teacher_logprobs_all = teacher_logprobs_all.cpu().float()
                old_student_logprobs = old_student_logprobs.float()

            # === Broadcast rollout data to all ranks ===
            (sequences, attention_mask, old_student_logprobs,
             teacher_logprobs_all, hit_eos, prompt_lens_t) = broadcast_rollout_data(
                rank, world_size, device, sequences, attention_mask,
                old_student_logprobs, teacher_logprobs_all,
                hit_eos, prompt_lens_t,
            )

            prompt_lens = prompt_lens_t.tolist()

            # Pre-compute masks on each rank's device
            old_logprobs_shifted_all = old_student_logprobs[:, 1:].to(device)
            loss_mask_all = build_loss_mask(sequences.to(device), prompt_lens, PAD_TOKEN_ID)
            teacher_logprobs_all = teacher_logprobs_all.to(device)
            total_generated_tokens = loss_mask_all.sum().item()

            # === Micro-batch loop (DDP) ===
            rank_start = rank * mbs_per_rank

            for local_idx in range(mbs_per_rank):
                mb_idx = rank_start + local_idx
                seq_start = mb_idx * micro_batch_size
                seq_end = seq_start + micro_batch_size

                teacher_lp = teacher_logprobs_all[seq_start:seq_end]
                mb_old_lp = old_logprobs_shifted_all[seq_start:seq_end]
                mb_loss_mask = loss_mask_all[seq_start:seq_end]
                mb_advantage = -(mb_old_lp - teacher_lp).detach()

                student_input = sequences[seq_start:seq_end].to(device, non_blocking=True)
                student_mask = attention_mask[seq_start:seq_end].to(device, non_blocking=True)

                student_out = student(
                    input_ids=student_input,
                    attention_mask=student_mask,
                )
                current_logprobs = get_logprobs_at_tokens(
                    student_out.logits, student_input, SHARED_VOCAB_SIZE,
                )

                ratio = torch.exp(current_logprobs - mb_old_lp)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -clipped_ratio * mb_advantage
                per_token_loss = torch.max(pg_loss1, pg_loss2)
                masked_loss = (per_token_loss * mb_loss_mask).sum() / mb_loss_mask.sum()

                # Each rank divides by mbs_per_rank; DDP averages across world_size
                # -> total division = mbs_per_rank * world_size = n_micro_batches
                scaled_loss = masked_loss / mbs_per_rank
                ctx = nullcontext() if (local_idx == mbs_per_rank - 1) else student.no_sync()
                with ctx:
                    scaled_loss.backward()
                accumulated_loss += scaled_loss.item()

            # === Optimizer step ===
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # === Logging + sync (rank 0 only) ===
            if rank == 0:
                opt_step_time = time.time() - opt_step_start_time
                avg_loss = accumulated_loss
                tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0

                mask_sum_mb = mb_loss_mask.sum()
                seq_lens_all = attention_mask.to(device).sum(dim=1)
                prompt_lens_all_t = prompt_lens_t.to(device)
                avg_gen_len = (seq_lens_all - prompt_lens_all_t).float().mean()
                mb_kl = ((mb_old_lp - teacher_lp) * mb_loss_mask).sum() / mask_sum_mb

                metrics_tensor = torch.stack([
                    grad_norm,
                    (mb_advantage * mb_loss_mask).sum() / mask_sum_mb,
                    (ratio * mb_loss_mask).sum() / mask_sum_mb,
                    mb_kl,
                    ((ratio > 1.0 + CLIP_EPS) | (ratio < 1.0 - CLIP_EPS)).float().sum() / mask_sum_mb,
                    ((pg_loss2 > pg_loss1) * mb_loss_mask).sum() / mask_sum_mb,
                    ((current_logprobs - mb_old_lp).abs() * mb_loss_mask).sum() / mask_sum_mb,
                ])
                (grad_norm_val, mean_advantage, mean_ratio, mean_kl,
                 ratio_clipped_frac, clip_active_frac, approx_policy_drift) = metrics_tensor.tolist()

                log_payload = {
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm_val,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/gen_time_sec": gen_time,
                    "train/optimizer_step_time_sec": opt_step_time,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/mean_advantage": mean_advantage,
                    "train/mean_ratio": mean_ratio,
                    "train/mean_kl": mean_kl,
                    "train/ratio_clipped_frac": ratio_clipped_frac,
                    "train/clip_active_frac": clip_active_frac,
                    "train/approx_policy_drift": approx_policy_drift,
                    "train/avg_gen_length": avg_gen_len.item(),
                    "train/no_eos_frac": (~hit_eos).float().mean().item(),
                }
                if last_sync_duration is not None:
                    log_payload["train/sync_duration_sec"] = last_sync_duration
                    last_sync_duration = None
                sync_interval = get_sync_interval(
                    global_step, mean_ratio, approx_policy_drift, sync_interval, sync_state,
                )
                log_payload["train/sync_every_n_steps"] = sync_interval
                wandb.log(log_payload)

            accumulated_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Rank 0: sync weights, samples, evals, checkpoints
            if rank == 0:
                if global_step % sync_interval == 0:
                    if gen_future is not None:
                        gen_future = None  # discard stale prefetch
                    if sync_future is None or sync_future.done():
                        sync_future = vllm_executor.submit(
                            timed_sync_weights_to_vllm, student.module, vllm_student,
                        )

                if global_step % SAMPLE_EVERY_N_STEPS == 0 and sample_future is None:
                    sample_future = vllm_executor.submit(
                        generate_samples, vllm_student, eval_prompts, tokenizer, max_context,
                    )

                if global_step % EVAL_EVERY_N_STEPS == 0 and eval_future is None:
                    eval_future = vllm_executor.submit(
                        run_evals, vllm_student, tokenizer, STUDENT,
                        tasks=EVAL_TASKS, limit=EVAL_N_SAMPLES,
                    )

                hub_repo = None if DEBUG_MODE else HUB_REPO
                if global_step % save_every == 0:
                    if checkpoint_future is not None:
                        checkpoint_future.result()
                    checkpoint_future = checkpoint_executor.submit(
                        save_checkpoint, student.module, tokenizer, optimizer,
                        global_step, checkpoint_base, milestone_every, hub_repo,
                    )

            if sweep_steps and global_step >= sweep_steps:
                if rank == 0:
                    print(f"Sweep: stopping after {sweep_steps} steps")
                break

        if sweep_steps and global_step >= sweep_steps:
            break

    pbar.close()

    # Cleanup
    if rank == 0:
        if sync_future is not None:
            sync_future.result()
        if sample_future is not None:
            wandb.log({"eval/samples": sample_future.result()})
        if eval_future is not None:
            wandb.log(eval_future.result())
        if checkpoint_future is not None:
            checkpoint_future.result()

        vllm_executor.shutdown(wait=True)
        checkpoint_executor.shutdown(wait=True)

        hub_repo = None if DEBUG_MODE else HUB_REPO
        save_checkpoint(student.module, tokenizer, optimizer, global_step,
                        checkpoint_base, milestone_every, hub_repo)
        wandb.finish()
        print("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()
