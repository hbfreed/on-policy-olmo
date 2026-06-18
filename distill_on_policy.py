import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
from muon import SingleDeviceMuonWithAuxAdam
from datasets import load_dataset
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, get_wsd_schedule
from vllm import LLM

import wandb
from distill_utils import (
    HiddenCapture,
    active_lengths_from_attention,
    build_shift_labels,
    generate_rollouts,
    generate_samples,
    init_weight_transfer,
    load_checkpoint,
    prepare_prompts,
    run_teacher_pipeline_hidden,
    save_checkpoint,
    timed_generate_rollouts,
    timed_sync_weights_to_vllm,
    timed_sync_weights_to_vllm_native,
)
from evals import run_evals

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

torch.manual_seed(1223)

# ============================================================
# Config
# ============================================================

WANDB_PROJECT = "olmo-distill"
RUN_NAME = None  # set to a string to override auto naming
HUB_REPO = None

# --- Models / data ---
TEACHER = "allenai/OLMo-2-1124-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B"
DATASET = "allenai/RLVR-MATH"

# --- Device layout (3 GPUs) ---
VLLM_DEVICE = "cuda:0"     # vLLM student rollouts
TEACHER_DEVICE = "cuda:1"  # HF teacher inference
STUDENT_DEVICE = "cuda:2"  # HF student training

# --- Training ---
LR = 1e-4                     # AdamW LR for embed / lm_head / norms (aux)
MUON_LR = 2e-4                # Muon LR for hidden 2D matrices.
                              # Matches Moonshot/K2/GLM effective fine-tuning step magnitude:
                              # they use lr=2e-5 with explicit 0.2*sqrt(max(A,B)) RMS scaling;
                              # Keller Jordan's `muon` package only scales by max(1, A/B)^0.5,
                              # so we use ~10x higher lr to match effective step size.
WEIGHT_DECAY = 0.1            # Moonshot/K2/GLM standard, applied to both Muon and AdamW
N_EPOCHS = 30  # RLVR-MATH is ~7500 prompts → ~7 steps/epoch → ~210 total steps
GROUP_SIZE = 4                # rollouts per prompt
GRAD_ACCUM_STEPS = 256        # unique prompts per optimizer step
MICRO_BATCH_SIZE = 2          # OLMo-1B: mbs=2 fits per prior memory note
TEACHER_MICRO_BATCH_SIZE = 8  # OLMo-7B teacher: mbs=8 was the prior tested limit
MAX_CONTEXT_LENGTH = 2048
MAX_GRAD_NORM = 3.0
WARMUP_STEPS = 20             # cushion early grad_norm spikes seen at LR=1e-4
DECAY_STEPS = 30              # WSD: last N steps decay from peak LR to min_lr_ratio*LR
SYNC_EVERY_N_STEPS = 1        # vLLM weight sync cadence
ENABLE_GRADIENT_CHECKPOINTING = False

# --- Eval / logging ---
DEBUG_MODE = False
N_SAMPLE_PROMPTS = 4
SAMPLE_EVERY_N_STEPS = 0      # 0 disables; vLLM sampling time is training time
EVAL_EVERY_N_STEPS = 0        # 0 disables mid-run lm-eval
EVAL_N_SAMPLES = 200
EVAL_TASKS = ["gsm8k_cot", "arc_easy", "truthfulqa_mc2", "ifeval"]


def parse_args():
    parser = argparse.ArgumentParser(description="On-policy distillation")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps (for quick LR sweeps)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume (e.g. xhzvc6kp)")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                        help="Wandb project name")
    parser.add_argument("--checkpoint-base", type=str,
                        default="checkpoints/onpolicy-from-baseline",
                        help="Base directory for checkpoints")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Load model weights only (fresh optimizer, step=0)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training (model + optimizer + step)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save rolling checkpoint every N steps")
    parser.add_argument("--milestone-every", type=int, default=500,
                        help="Save permanent milestone checkpoint every N steps")
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE,
                        help="Student sequences per forward/backward microbatch")
    parser.add_argument("--teacher-micro-batch-size", type=int,
                        default=TEACHER_MICRO_BATCH_SIZE,
                        help="Teacher sequences per inference chunk")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY_N_STEPS,
                        help="Generate sample table every N steps; 0 disables")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_N_STEPS,
                        help="Run lm-eval every N steps; 0 disables")
    parser.add_argument("--eval-n-samples", type=int, default=EVAL_N_SAMPLES,
                        help="Number of examples per eval task")
    return parser.parse_args()


def main():
    args = parse_args()
    lr = args.lr
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id
    checkpoint_base = args.checkpoint_base
    init_from = args.init_from
    resume_from = args.resume_from
    save_every = args.save_every
    milestone_every = args.milestone_every
    micro_batch_size = args.micro_batch_size
    teacher_micro_batch_size = args.teacher_micro_batch_size
    sample_every = args.sample_every
    eval_every = args.eval_every
    eval_n_samples = args.eval_n_samples

    # Load tokenizer
    print(f"Loading tokenizer from {TEACHER}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"  # Required for decoder-only models
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Load dataset and retokenize prompts for the current teacher tokenizer
    print(f"Loading dataset from {DATASET}...")
    ds = load_dataset(DATASET, split="train")

    # Handle two schemas:
    #   - Dolci-Instruct-*: "prompt" field (string with "user: " prefix)
    #   - RLVR-* (Allen AI): "messages" field (list of {role, content})
    has_messages = "messages" in ds.column_names
    keep_col = "messages" if has_messages else "prompt"

    def retokenize(example):
        if has_messages:
            messages = example["messages"]
        else:
            content = example["prompt"].removeprefix("user: ")
            messages = [{"role": "user", "content": content}]
        example["input_ids_prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        return example

    dataset = (
        ds.select_columns([keep_col])
        .map(retokenize)
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

    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH
    sequences_per_step = group_size * GRAD_ACCUM_STEPS
    if sequences_per_step % micro_batch_size != 0:
        raise ValueError(
            f"micro_batch_size={micro_batch_size} must divide "
            f"{sequences_per_step} sequences per optimizer step"
        )
    if teacher_micro_batch_size < micro_batch_size:
        raise ValueError("teacher_micro_batch_size must be >= micro_batch_size")

    steps_per_epoch = len(dataset) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * n_epochs

    # Load models
    student_load_path = init_from or resume_from or STUDENT
    print(f"Loading student model from {student_load_path}...")
    student = AutoLigerKernelForCausalLM.from_pretrained(
        student_load_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(STUDENT_DEVICE)

    print(f"Loading teacher model from {TEACHER}...")
    teacher = AutoLigerKernelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(TEACHER_DEVICE)
    teacher.eval()

    # Frozen copy of teacher's lm_head on the student GPU for the fused JSD kernel.
    teacher_lm_head_weight = (
        teacher.lm_head.weight.detach().to(STUDENT_DEVICE).contiguous()
    )

    print(f"Student vocab: {student.config.vocab_size}")
    print(f"Teacher vocab: {teacher.config.vocab_size}")
    vllm_model_path = student_load_path
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
    print(f"Loading vLLM student on {VLLM_DEVICE}...")
    torch.cuda.set_device(VLLM_DEVICE)
    from vllm.config import WeightTransferConfig
    vllm_student = LLM(
        vllm_model_path,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=max_context,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )
    if vllm_model_path != student_load_path:
        import shutil
        shutil.rmtree(vllm_model_path)
        print(f"Cleaned up temp model at {vllm_model_path}")

    if ENABLE_GRADIENT_CHECKPOINTING:
        student.gradient_checkpointing_enable()

    # Muon for hidden 2D matrices, AdamW (aux) for embed / lm_head / norms / biases
    hidden_matrix_params, embed_params, head_params, scalar_params = [], [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in n.lower():
            embed_params.append(p)
        elif "lm_head" in n.lower():
            head_params.append(p)
        elif p.ndim < 2:
            scalar_params.append(p)
        else:
            hidden_matrix_params.append(p)
    adam_groups = [
        dict(params=g, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY, use_muon=False)
        for g in (head_params, embed_params, scalar_params)
        if g
    ]
    muon_group = dict(params=hidden_matrix_params, lr=MUON_LR, momentum=0.95,
                      weight_decay=WEIGHT_DECAY, use_muon=True)
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups=[*adam_groups, muon_group])
    print(f"Muon: {sum(p.numel() for p in hidden_matrix_params)/1e6:.1f}M params; "
          f"AdamW aux: {sum(p.numel() for g in (head_params, embed_params, scalar_params) for p in g)/1e6:.1f}M params")
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    decay_steps = min(DECAY_STEPS, max(1, total_steps - warmup_steps - 1))
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        num_training_steps=total_steps,
        decay_type="cosine",
        min_lr_ratio=0.1,
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
            f"{student_short}-onpolicy-revkl"
            f"-lr{lr_str}-sync{SYNC_EVERY_N_STEPS}{sweep_tag}"
        )

    wandb.init(
        project=args.wandb_project,
        id=wandb_run_id,
        name=run_name,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "group_size": group_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "micro_batch_size": micro_batch_size,
            "teacher_micro_batch_size": teacher_micro_batch_size,
            "effective_batch_size": group_size * GRAD_ACCUM_STEPS,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "lr": lr,
            "muon_lr": MUON_LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Muon+AuxAdamW",
            "sweep_steps": sweep_steps,
            "loss": "reverse_kl_full_vocab",
            "max_grad_norm": MAX_GRAD_NORM,
            "gradient_checkpointing": ENABLE_GRADIENT_CHECKPOINTING,
            "warmup_steps": WARMUP_STEPS,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "init_from": init_from,
            "resume_from": resume_from,
            "sample_every_n_steps": sample_every,
            "eval_every_n_steps": eval_every,
            "eval_n_samples": eval_n_samples,
            "eval_tasks": EVAL_TASKS,
        },
        resume="must" if wandb_run_id else "allow",
    )

    # Baseline generation samples before any training, if sample logging is enabled.
    if sample_every > 0:
        baseline_table = generate_samples(vllm_student, eval_prompts, tokenizer, max_context)
        wandb.log({"eval/samples": baseline_table}, step=0)

    # Training loop
    global_step = start_step
    optimizer.zero_grad(set_to_none=True)

    # Create progress bar
    total_optimizer_steps = steps_per_epoch * n_epochs
    pbar = tqdm(total=total_optimizer_steps - start_step, desc="Training")

    # Set once to avoid per-step device switches
    torch.cuda.set_device(STUDENT_DEVICE)

    # Native NCCL weight transfer: HF student (STUDENT_DEVICE) -> vLLM worker (VLLM_DEVICE).
    # Direct GPU->GPU broadcast replacing the BytesIO/collective_rpc sync.
    weight_transfer_group = init_weight_transfer(student, vllm_student, STUDENT_DEVICE)

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

    # Pre-hook on student lm_head: captures hidden state and skips lm_head matmul;
    # Liger fused JSD does the projection internally in chunks.
    student_capture = HiddenCapture(student)

    for epoch in range(n_epochs):
        all_batches = list(dataset.iter(batch_size=1))

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
                sequences, prompt_lens, _, attention_mask, hit_eos, gen_time = (
                    gen_future.result()
                )
                gen_future = None
            else:
                prompts = prepare_prompts(opt_step_idx, all_batches, tokenizer, GRAD_ACCUM_STEPS)
                gen_start_time = time.time()
                sequences, prompt_lens, _, attention_mask, hit_eos = (
                    generate_rollouts(
                        vllm_student, prompts, PAD_TOKEN_ID, group_size, max_context, SHARED_VOCAB_SIZE,
                        with_logprobs=False, sort_by_length=True,
                    )
                )
                gen_time = time.time() - gen_start_time

            # Prefetch next step's generation (overlaps with training below)
            next_step_idx = opt_step_idx + 1
            will_sync = (global_step + 1) % SYNC_EVERY_N_STEPS == 0
            can_prefetch = (
                next_step_idx < steps_per_epoch
                and SYNC_EVERY_N_STEPS > 1
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
                    False,  # with_logprobs
                    True,   # sort_by_length
                )

            # hit_eos comes from vLLM finish_reason (stop_token_ids strips EOS from output)

            active_lens_cpu = active_lengths_from_attention(attention_mask)
            shift_labels_cpu = build_shift_labels(attention_mask, prompt_lens)
            completion_lengths_cpu = shift_labels_cpu.ne(-100).sum(dim=1)
            total_generated_tokens = int(completion_lengths_cpu.sum().item())

            # Teacher pipeline: ships last hidden state per micro-batch.
            teacher_queue = Queue(maxsize=4)
            teacher_thread = teacher_executor.submit(
                run_teacher_pipeline_hidden, teacher, sequences, attention_mask,
                teacher_micro_batch_size, TEACHER_DEVICE, STUDENT_DEVICE, teacher_queue,
                micro_batch_size
            )

            # Stage rollout tensors on the student GPU once per optimizer step.
            n_micro_batches = len(sequences) // micro_batch_size
            sequences_student = sequences.to(STUDENT_DEVICE, non_blocking=True)
            attention_mask_student = attention_mask.to(STUDENT_DEVICE, non_blocking=True)
            shift_labels_all = shift_labels_cpu.to(STUDENT_DEVICE, non_blocking=True)
            micro_max_lens = [
                int(active_lens_cpu[i:i + micro_batch_size].max().item())
                for i in range(0, len(sequences), micro_batch_size)
            ]
            loss_terms = []

            for mb_idx in range(n_micro_batches):
                seq_start = mb_idx * micro_batch_size
                seq_end = seq_start + micro_batch_size
                active_len = micro_max_lens[mb_idx]

                teacher_hidden = teacher_queue.get()  # [B, T, H_t] on STUDENT_DEVICE
                if isinstance(teacher_hidden, BaseException):
                    raise teacher_hidden

                student_input = sequences_student[seq_start:seq_end, :active_len]
                student_mask = attention_mask_student[seq_start:seq_end, :active_len]
                teacher_hidden = teacher_hidden[:, :active_len, :]

                # Forward student; pre-hook captures hidden, lm_head call is a no-op.
                student(input_ids=student_input, attention_mask=student_mask)
                student_hidden = student_capture.hidden  # [B, T, H_s]

                s_in = (
                    student_hidden[:, :-1, :]
                    .reshape(-1, student_hidden.size(-1)).contiguous()
                )
                t_in = (
                    teacher_hidden[:, :-1, :]
                    .reshape(-1, teacher_hidden.size(-1)).contiguous()
                )

                shift_labels = (
                    shift_labels_all[seq_start:seq_end, :active_len - 1]
                    .reshape(-1).contiguous()
                )

                loss = LigerFusedLinearJSDFunction.apply(
                    s_in, student.lm_head.weight,
                    t_in, teacher_lm_head_weight,
                    shift_labels,
                    1.0,    # jsd_beta=1.0  ->  reverse KL D(pi_student || pi_teacher)
                    -100,   # ignore_index
                    1.0,    # temperature
                )
                scaled_loss = loss / n_micro_batches
                scaled_loss.backward()
                loss_terms.append(scaled_loss.detach())

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
            avg_loss = torch.stack(loss_terms).sum().item()  # averaged via scaled_loss

            tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0
            avg_gen_len = completion_lengths_cpu.float().mean().item()

            log_payload = {
                "train/loss": avg_loss,                # mean reverse KL per non-ignored token
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "train/gen_time_sec": gen_time,
                "train/optimizer_step_time_sec": opt_step_time,
                "train/lr_adamw": scheduler.get_last_lr()[0],
                "train/lr_muon": scheduler.get_last_lr()[-1],
                "train/global_step": global_step,
                "train/avg_gen_length": avg_gen_len,
                "train/no_eos_frac": (~hit_eos).float().mean().item(),
            }
            if last_sync_duration is not None:
                log_payload["train/sync_duration_sec"] = last_sync_duration
                last_sync_duration = None
            wandb.log(log_payload)

            global_step += 1
            pbar.update(1)

            # Sync updated weights to vLLM for on-policy generation
            if global_step % SYNC_EVERY_N_STEPS == 0:
                if gen_future is not None:
                    # Sync needed — discard prefetched stale rollouts
                    gen_future = None
                if sync_future is None or sync_future.done():
                    sync_future = vllm_executor.submit(
                        timed_sync_weights_to_vllm_native,
                        student, vllm_student, weight_transfer_group, STUDENT_DEVICE,
                    )

            # Log generation quality samples (non-blocking)
            if sample_every > 0 and global_step % sample_every == 0 and sample_future is None:
                sample_future = vllm_executor.submit(
                    generate_samples,
                    vllm_student,
                    eval_prompts,
                    tokenizer,
                    max_context,
                )

            # Run benchmark evals (async via vllm_executor)
            if eval_every > 0 and global_step % eval_every == 0 and eval_future is None:
                eval_future = vllm_executor.submit(
                    run_evals, vllm_student, tokenizer, STUDENT,
                    tasks=EVAL_TASKS, limit=eval_n_samples,
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
