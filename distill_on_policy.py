import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import bitsandbytes as bnb
import torch
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
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
    MICRO_BATCH_SIZE,
    N_EPOCHS,
    N_SAMPLE_PROMPTS,
    RUN_NAME,
    SAMPLE_EVERY_N_STEPS,
    STUDENT,
    STUDENT_DEVICE,
    SYNC_EVERY_N_STEPS,
    TEACHER,
    TEACHER_DEVICE,
    TEACHER_MICRO_BATCH_SIZE,
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

torch.manual_seed(1223)


def parse_args():
    parser = argparse.ArgumentParser(description="On-policy distillation")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps (for quick LR sweeps)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume (e.g. xhzvc6kp)")
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


def main():
    args = parse_args()
    lr = args.lr
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id
    checkpoint_base = args.checkpoint_base
    resume_from = args.resume_from
    save_every = args.save_every
    milestone_every = args.milestone_every

    # Load tokenizer
    print(f"Loading tokenizer from {TEACHER}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"  # Required for decoder-only models
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Load dataset
    print(f"Loading dataset from {DATASET}...")
    ds = load_dataset(DATASET, split="train")

    dataset = (
        ds.select_columns(["prompt", "input_ids_prompt"])
        .filter(lambda x: len(x["input_ids_prompt"]) < MAX_CONTEXT_LENGTH)
        .shuffle(seed=1223)
    )

    if DEBUG_MODE:
        # Single-prompt overfitting test, a la Thinking Machines
        single = dataset.select(range(1))
        # Repeat to fill GRAD_ACCUM_STEPS batches so the training loop works unchanged
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([single] * GRAD_ACCUM_STEPS)
        print(f"DEBUG MODE: 1 prompt repeated {GRAD_ACCUM_STEPS}x for overfitting test")

    n_epochs = 20 if DEBUG_MODE else N_EPOCHS

    # Fixed eval prompts for tracking generation quality over training
    eval_prompts = [
        dataset[i]["input_ids_prompt"]
        for i in range(min(N_SAMPLE_PROMPTS, len(dataset)))
    ]

    batch_size = BATCH_SIZE
    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // (batch_size * GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * n_epochs

    # Load models
    print(f"Loading student model from {STUDENT}...")
    if resume_from:
        student = AutoLigerKernelForCausalLM.from_pretrained(
            resume_from,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(STUDENT_DEVICE)
    else:
        student = AutoLigerKernelForCausalLM.from_pretrained(
            STUDENT, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(STUDENT_DEVICE)

    print(f"Loading teacher model from {TEACHER}...")
    teacher = AutoLigerKernelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(TEACHER_DEVICE)
    teacher.eval()

    print(f"Student vocab: {student.config.vocab_size}")
    print(f"Teacher vocab: {teacher.config.vocab_size}")
    vllm_model_path = resume_from or STUDENT
    if student.config.vocab_size != teacher.config.vocab_size:
        print(f"Resizing student embeddings {student.config.vocab_size} -> {teacher.config.vocab_size}")
        student.resize_token_embeddings(teacher.config.vocab_size)
        # Save resized model for vLLM (can't sync mismatched shapes)
        import tempfile
        vllm_model_path = tempfile.mkdtemp(prefix="olmo_resized_")
        student.save_pretrained(vllm_model_path)
        print(f"Saved resized student for vLLM at {vllm_model_path}")
    SHARED_VOCAB_SIZE = student.config.vocab_size

    # Initialize vLLM for fast generation on separate GPU
    # skip_tokenizer_init=True since we input token IDs directly
    print(f"Loading vLLM student on {STUDENT_DEVICE}...")
    vllm_student = LLM(
        vllm_model_path,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )
    if vllm_model_path != (resume_from or STUDENT):
        import shutil
        shutil.rmtree(vllm_model_path)
        print(f"Cleaned up temp model at {vllm_model_path}")

    student.gradient_checkpointing_enable()

    optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps
    )

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, student, optimizer, vllm_student)
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")

    def short_name(model_name: str) -> str:
        return model_name.split("/")[-1]

    run_name = RUN_NAME
    if run_name is None:
        student_short = short_name(STUDENT).lower().replace("olmo-2-0425-", "olmo")
        lr_str = f"{lr:.0e}".replace("-0", "-")
        sweep_tag = f"-sweep{sweep_steps}" if sweep_steps else ""
        run_name = (
            f"{student_short}-onpolicy-distill"
            f"-lr{lr_str}-clip{CLIP_EPS}-sync{SYNC_EVERY_N_STEPS}{sweep_tag}"
        )

    wandb.init(
        project=WANDB_PROJECT,
        id=wandb_run_id,
        name=run_name,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "batch_size": batch_size,
            "group_size": group_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "lr": lr,
            "sweep_steps": sweep_steps,
            "clip_eps": CLIP_EPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_steps": WARMUP_STEPS,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "resume_from": resume_from,
            "eval_every_n_steps": EVAL_EVERY_N_STEPS,
            "eval_n_samples": EVAL_N_SAMPLES,
            "eval_tasks": EVAL_TASKS,
        },
        resume="must" if wandb_run_id else "allow",
    )

    # Baseline generation samples before any training
    baseline_table = generate_samples(vllm_student, eval_prompts, tokenizer, max_context)
    wandb.log({"eval/samples": baseline_table}, step=0)

    # Training loop
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # Create progress bar
    total_optimizer_steps = steps_per_epoch * n_epochs
    pbar = tqdm(total=total_optimizer_steps - start_step, desc="Training")

    # Set once to avoid per-step device switches
    torch.cuda.set_device(STUDENT_DEVICE)

    # Use a single-threaded executor to serialize all vLLM calls (generate/sync)
    vllm_executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_executor = ThreadPoolExecutor(max_workers=1)
    teacher_executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_future = None
    sync_future = None
    sample_future = None
    eval_future = None
    gen_future = None  # Future for prefetched next-step generation
    last_sync_duration = None
    sync_interval = SYNC_EVERY_N_STEPS
    sync_state = {"steps_since_decrease": 0}

    for epoch in range(n_epochs):
        all_batches = list(dataset.iter(batch_size=batch_size))

        for opt_step_idx in range(steps_per_epoch):
            # Skip if resuming
            current_step = epoch * steps_per_epoch + opt_step_idx
            if current_step < start_step:
                continue

            # Ensure any in-flight vLLM work is done before generating —
            # vLLM is not thread-safe for concurrent generate calls
            if sync_future is not None:
                last_sync_duration = sync_future.result()
                sync_future = None

            if sample_future is not None:
                wandb.log({"eval/samples": sample_future.result()})
                sample_future = None

            # Drain eval before generate_rollouts — both use the vLLM engine
            if eval_future is not None:
                wandb.log(eval_future.result())
                eval_future = None

            opt_step_start_time = time.time()

            # Use prefetched generation if available, otherwise generate synchronously
            if gen_future is not None:
                sequences, prompt_lens, old_student_logprobs, attention_mask, gen_time = (
                    gen_future.result()
                )
                gen_future = None
            else:
                prompts = prepare_prompts(opt_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS)
                gen_start_time = time.time()
                sequences, prompt_lens, old_student_logprobs, attention_mask = (
                    generate_rollouts(
                        vllm_student, prompts, PAD_TOKEN_ID, group_size, max_context, SHARED_VOCAB_SIZE
                    )
                )
                gen_time = time.time() - gen_start_time

            # Prefetch next step's generation (overlaps with training below)
            next_step_idx = opt_step_idx + 1
            will_sync = (global_step + 1) % sync_interval == 0
            can_prefetch = (
                next_step_idx < steps_per_epoch
                and sync_interval > 1
                and not will_sync
            )
            if can_prefetch:
                next_prompts = prepare_prompts(
                    next_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS
                )
                gen_future = vllm_executor.submit(
                    timed_generate_rollouts,
                    vllm_student, next_prompts, PAD_TOKEN_ID,
                    group_size, max_context, SHARED_VOCAB_SIZE,
                )

            total_generated_tokens = 0

            # Flag sequences that hit max_length without EOS (vectorized)
            positions = torch.arange(sequences.shape[1]).unsqueeze(0)
            prompt_lens_t = torch.tensor(prompt_lens).unsqueeze(1)
            completion_mask = (positions >= prompt_lens_t) & (sequences != PAD_TOKEN_ID)
            hit_eos = ((sequences == tokenizer.eos_token_id) & completion_mask).any(dim=1)

            # Pre-compute loss mask and old logprobs for all sequences
            n_micro_batches = len(sequences) // MICRO_BATCH_SIZE
            old_logprobs_shifted_all = old_student_logprobs[:, 1:].to(STUDENT_DEVICE)
            loss_mask_all = build_loss_mask(sequences.to(STUDENT_DEVICE), prompt_lens, PAD_TOKEN_ID)
            total_generated_tokens = loss_mask_all.sum().item()

            # Teacher pipeline: batches at TEACHER_MICRO_BATCH_SIZE for throughput,
            # splits output into MICRO_BATCH_SIZE chunks for student consumption.
            teacher_queue = Queue(maxsize=4)
            teacher_thread = teacher_executor.submit(
                run_teacher_pipeline, teacher, sequences, attention_mask,
                TEACHER_MICRO_BATCH_SIZE, TEACHER_DEVICE, STUDENT_DEVICE, teacher_queue,
                MICRO_BATCH_SIZE
            )

            for mb_idx in range(n_micro_batches):
                seq_start = mb_idx * MICRO_BATCH_SIZE
                seq_end = seq_start + MICRO_BATCH_SIZE

                teacher_lp = teacher_queue.get()
                if isinstance(teacher_lp, BaseException):
                    raise teacher_lp

                mb_old_lp = old_logprobs_shifted_all[seq_start:seq_end]
                mb_loss_mask = loss_mask_all[seq_start:seq_end]

                mb_advantage = -(mb_old_lp - teacher_lp)
                mb_advantage = mb_advantage.detach()

                student_input = sequences[seq_start:seq_end].to(STUDENT_DEVICE, non_blocking=True)

                student_mask = attention_mask[seq_start:seq_end].to(STUDENT_DEVICE, non_blocking=True)

                student_out = student(
                    input_ids=student_input,
                    attention_mask=student_mask,
                )
                current_logprobs = get_logprobs_at_tokens(student_out.logits, student_input, SHARED_VOCAB_SIZE)

                ratio = torch.exp(current_logprobs - mb_old_lp)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -clipped_ratio * mb_advantage
                per_token_loss = torch.max(pg_loss1, pg_loss2)
                masked_loss = (per_token_loss * mb_loss_mask).sum() / mb_loss_mask.sum()

                scaled_loss = masked_loss / n_micro_batches
                scaled_loss.backward()
                accumulated_loss += scaled_loss.item()

            # Drain sentinel and propagate any teacher exception
            assert teacher_queue.get() is None
            teacher_thread.result()

            # --- Optimizer step (after all micro-batches) ---
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            opt_step_time = time.time() - opt_step_start_time
            avg_loss = accumulated_loss  # already averaged via scaled_loss

            tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0

            # Log metrics (last micro-batch values for ratio/drift stats)
            mask_sum_mb = mb_loss_mask.sum()
            seq_lens_all = attention_mask.to(STUDENT_DEVICE).sum(dim=1)
            prompt_lens_all_t = torch.tensor(prompt_lens, device=STUDENT_DEVICE)
            avg_gen_len = (seq_lens_all - prompt_lens_all_t).float().mean()

            # mean_kl from last micro-batch's teacher logprobs (representative sample)
            mb_kl = ((mb_old_lp - teacher_lp) * mb_loss_mask).sum() / mb_loss_mask.sum()

            metrics_tensor = torch.stack(
                [
                    grad_norm,
                    (mb_advantage * mb_loss_mask).sum() / mask_sum_mb,
                    (ratio * mb_loss_mask).sum() / mask_sum_mb,
                    mb_kl,
                    ((ratio > 1.0 + CLIP_EPS) | (ratio < 1.0 - CLIP_EPS)).float().sum() / mask_sum_mb,
                    ((pg_loss2 > pg_loss1) * mb_loss_mask).sum() / mask_sum_mb,
                    (
                        (current_logprobs - mb_old_lp).abs() * mb_loss_mask
                    ).sum()
                    / mask_sum_mb,
                ]
            )
            (
                grad_norm_val,
                mean_advantage,
                mean_ratio,
                mean_kl,
                ratio_clipped_frac,
                clip_active_frac,
                approx_policy_drift,
            ) = metrics_tensor.tolist()

            log_payload = {
                "train/loss": avg_loss,
                "train/grad_norm": grad_norm_val,
                "train/tokens_per_sec": tokens_per_sec,
                "train/gen_time_sec": gen_time,
                "train/optimizer_step_time_sec": opt_step_time,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/global_step": global_step,
                # Policy gradient diagnostics
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
                global_step, mean_ratio, approx_policy_drift, sync_interval, sync_state
            )
            log_payload["train/sync_every_n_steps"] = sync_interval
            wandb.log(log_payload)

            accumulated_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Sync updated weights to vLLM for on-policy generation
            if global_step % sync_interval == 0:
                if gen_future is not None:
                    # Sync needed — discard prefetched stale rollouts
                    gen_future = None
                if sync_future is None or sync_future.done():
                    sync_future = vllm_executor.submit(
                        timed_sync_weights_to_vllm, student, vllm_student
                    )

            # Log generation quality samples (non-blocking)
            if global_step % SAMPLE_EVERY_N_STEPS == 0 and sample_future is None:
                sample_future = vllm_executor.submit(
                    generate_samples,
                    vllm_student,
                    eval_prompts,
                    tokenizer,
                    max_context,
                )

            # Run benchmark evals (async via vllm_executor)
            if global_step % EVAL_EVERY_N_STEPS == 0 and eval_future is None:
                eval_future = vllm_executor.submit(
                    run_evals, vllm_student, tokenizer, STUDENT,
                    tasks=EVAL_TASKS, limit=EVAL_N_SAMPLES,
                )

            # Save checkpoint (async - don't block training)
            # Skip hub upload in debug mode
            hub_repo = None if DEBUG_MODE else HUB_REPO
            if global_step % save_every == 0:
                # Wait for previous upload to finish before starting new one
                if checkpoint_future is not None:
                    checkpoint_future.result()
                checkpoint_future = checkpoint_executor.submit(
                    save_checkpoint,
                    student,
                    tokenizer,
                    optimizer,
                    global_step,
                    checkpoint_base,
                    milestone_every,
                    hub_repo,
                )

            # Early stop for LR sweep
            if sweep_steps and global_step >= sweep_steps:
                print(f"Sweep: stopping after {sweep_steps} steps")
                break

        if sweep_steps and global_step >= sweep_steps:
            break

    pbar.close()

    # Wait for any pending sync/checkpoint/samples/evals
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
    teacher_executor.shutdown(wait=True)

    # Final save (synchronous - we're done anyway)
    hub_repo = None if DEBUG_MODE else HUB_REPO
    save_checkpoint(student, tokenizer, optimizer, global_step,
                    checkpoint_base, milestone_every, hub_repo)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
