import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import torch
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerFusedLinearJSD
from torchao.optim import AdamW4bit
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

#DATASET = "allenai/Dolci-Think-RL-7B"
DATASET = "allenai/Dolci-Instruct-RL"
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B-Instruct"
HUB_REPO = "hbfreed/Olmo-2-1B-Distilled"
WANDB_PROJECT = "olmo-2-1b-on-policy-distillation"
RUN_NAME = "instruct-student-instruct-teacher"  # set to a string to override auto naming

STUDENT_DEVICE = "cuda:2"  # HF student for training
TEACHER_DEVICE = "cuda:1"  # HF teacher for inference
VLLM_DEVICE = "cuda:0"  # vLLM student for fast generation (vLLM uses first visible GPU)

BATCH_SIZE = 3
N_EPOCHS = 1
GROUP_SIZE = 2  # number of rollouts per prompt
GRAD_ACCUM_STEPS = 4
MAX_CONTEXT_LENGTH = 1024  # Reduced to fit in GPU memory
LR = 1e-4
SYNC_EVERY_N_STEPS = 8

RESUME_FROM = None  # or "checkpoints/step_1000"
DEBUG_MODE = False
PROFILE_STEPS = 0  # Set to 0 to disable profiling (uses lots of RAM)

torch.manual_seed(1223)


def generate_rollouts(
    vllm_student, batch, pad_token_id, group_size=1, max_context_length=4096
):
    """Generate rollouts from student model using vLLM, returning sequences and prompt length."""
    prompts = batch["input_ids_prompt"]
    if not isinstance(prompts[0], list):
        prompts = [prompts]

    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    max_new_tokens = max_context_length - max_prompt_len

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=group_size,
    )

    token_prompts = [{"prompt_token_ids": p} for p in prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
    )

    # Convert vLLM outputs to tensor: each RequestOutput has `outputs` list
    # With n=group_size, we get group_size completions per prompt
    all_sequences = []
    for req_output, prompt_len in zip(outputs, prompt_lens):
        prompt_ids = req_output.prompt_token_ids
        for completion in req_output.outputs:
            # Combine prompt + generated tokens
            full_seq = list(prompt_ids) + list(completion.token_ids)
            all_sequences.append(full_seq)

    # Pad sequences to same length (right pad with pad_token_id)
    max_seq_len = max(len(seq) for seq in all_sequences)
    padded = [seq + [pad_token_id] * (max_seq_len - len(seq)) for seq in all_sequences]

    sequences = torch.tensor(padded)
    return sequences, max_prompt_len


def create_shift_labels(sequences, prompt_len, pad_token_id, ignore_index=-100):
    """Create shifted labels masking prompt and padding tokens for loss computation."""
    labels = sequences.clone()
    labels[:, :prompt_len] = ignore_index
    labels[labels == pad_token_id] = ignore_index
    return labels.view(-1)


def save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo=None):
    """Save checkpoint to disk and optionally push to hub."""
    checkpoint_dir = f"checkpoints/step_{global_step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    student.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": global_step},
        f"{checkpoint_dir}/training_state.pt",
    )

    print(f"Saved checkpoint to {checkpoint_dir}")

    if hub_repo:
        try:
            student.push_to_hub(hub_repo, commit_message=f"Step {global_step}")
            tokenizer.push_to_hub(hub_repo, commit_message=f"Step {global_step}")
            print(f"Pushed checkpoint to {hub_repo}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


def sync_weights_to_vllm(hf_model, vllm_llm):
    """Sync weights from HF model to vLLM engine for on-policy learning.

    Uses collective_rpc to update weights in V1 architecture.
    See: https://github.com/vllm-project/vllm/issues/5723
    """
    # Serialize state dict to bytes (avoids pickle issues with numpy/tensors)
    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(hf_state_dict, buffer)
    weights_bytes = buffer.getvalue()

    # Define function to run on vLLM workers
    def load_weights_on_worker(worker, serialized_weights):
        buf = io.BytesIO(serialized_weights)
        weights_dict = torch.load(buf, weights_only=True)
        weights = list(weights_dict.items())
        worker.model_runner.model.load_weights(weights=weights)

    # Call on all workers via RPC
    method_bytes = cloudpickle.dumps(load_weights_on_worker)
    vllm_llm.llm_engine.collective_rpc(method_bytes, args=(weights_bytes,))


def timed_sync_weights_to_vllm(hf_model, vllm_llm):
    """Time sync to help pick a data-driven SYNC_EVERY_N_STEPS."""
    start = time.time()
    sync_weights_to_vllm(hf_model, vllm_llm)
    return time.time() - start


def load_checkpoint(checkpoint_path, student, optimizer, vllm_student=None):
    """Load optimizer state and return the step to resume from."""
    state_path = f"{checkpoint_path}/training_state.pt"
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        # Sync loaded weights to vLLM engine
        if vllm_student is not None:
            sync_weights_to_vllm(student, vllm_student)
        return state["step"]
    return 0


def main():
    # Load tokenizer
    print(f"Loading tokenizer from {TEACHER}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"  # Required for decoder-only models
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Load dataset
    print(f"Loading dataset from {DATASET}...")
    ds = load_dataset(DATASET, split="train")

    # 86.5% of prompts are < 472 tokens
    dataset = (
        ds.select_columns(["input_ids_prompt"])
        .filter(lambda x: len(x["input_ids_prompt"]) < 472)
        .shuffle(seed=1223)
    )

    if DEBUG_MODE:
        dataset = dataset.select(range(min(32, len(dataset))))
        print(f"DEBUG MODE: Using {len(dataset)} samples")

    batch_size = BATCH_SIZE
    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // batch_size
    total_steps = steps_per_epoch * N_EPOCHS
    save_every = max(1, min(500, int(total_steps * 0.02)))

    # Load models
    print(f"Loading student model from {STUDENT}...")
    if RESUME_FROM:
        student = AutoLigerKernelForCausalLM.from_pretrained(
            RESUME_FROM,
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

    # Resize teacher vocab to match student (100352 vs 100278)
    # This adds 74 new tokens initialized randomly (won't be used in practice)
    teacher.resize_token_embeddings(100352)

    # Save references before torch.compile (compiled models need special attribute access)
    student_base_model = student.model  # Base transformer without lm_head
    teacher_base_model = teacher.model
    student_lm_head_weight = student.lm_head.weight  # Will have gradients
    teacher_lm_head_weight = teacher.lm_head.weight.to(
        STUDENT_DEVICE
    )  # Copy for fused kernel
    student_hidden_size = student.config.hidden_size
    teacher_hidden_size = teacher.config.hidden_size

    # Initialize vLLM for fast generation on separate GPU
    # skip_tokenizer_init=True since we input token IDs directly
    print(f"Loading vLLM student on {VLLM_DEVICE}...")
    vllm_student = LLM(
        STUDENT,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )

    print("Compiling models with torch.compile...")
    student_base_model = torch.compile(student_base_model)
    teacher_base_model = torch.compile(teacher_base_model)

    optimizer = AdamW4bit(student.parameters(), lr=LR)

    start_step = 0
    if RESUME_FROM:
        start_step = load_checkpoint(RESUME_FROM, student, optimizer, vllm_student)
        print(f"Resuming from step {start_step}")

    fused_jsd = LigerFusedLinearJSD(
        jsd_beta=1.0, temperature=1.0
    )  # beta=1.0 is reverse kl

    def short_name(model_name: str) -> str:
        return model_name.split("/")[-1]

    run_name = RUN_NAME
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"distill_{short_name(STUDENT)}_from_{short_name(TEACHER)}"
            f"_bs{BATCH_SIZE}_ga{GRAD_ACCUM_STEPS}_gs{GROUP_SIZE}"
            f"_ctx{MAX_CONTEXT_LENGTH}_sync{SYNC_EVERY_N_STEPS}_{timestamp}"
        )

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "batch_size": batch_size,
            "group_size": group_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": N_EPOCHS,
            "total_steps": total_steps,
            "lr": LR,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "resume_from": RESUME_FROM,
        },
        resume="allow",
    )

    # Training loop
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # Create progress bar
    total_optimizer_steps = (steps_per_epoch * N_EPOCHS) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_optimizer_steps - start_step, desc="Training")

    # Set once to avoid per-step device switches
    torch.cuda.set_device(STUDENT_DEVICE)

    # Async prefetch: generate batch N+1 while training on batch N
    # This overlaps vLLM generation (GPU 0) with HF training (GPU 1+2)
    # Use a single-threaded executor to serialize all vLLM calls (generate/sync)
    vllm_executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_future = None
    sync_future = None
    last_sync_duration = None
    opt_step_start_time = None
    prefetch_wait_accum = 0.0
    prefetch_wait_count = 0

    # Setup profiler
    profiler = None
    if PROFILE_STEPS > 0:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=PROFILE_STEPS, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()

    for epoch in range(N_EPOCHS):
        # Convert to list for indexing (needed for prefetch lookahead)
        all_batches = list(dataset.iter(batch_size=batch_size))
        prefetch_future = None

        for batch_idx, batch in enumerate(all_batches):
            # Skip batches if resuming
            current_step = epoch * steps_per_epoch + batch_idx
            if current_step < start_step:
                continue

            if sync_future is not None and sync_future.done():
                last_sync_duration = sync_future.result()
                sync_future = None

            if batch_idx % GRAD_ACCUM_STEPS == 0:
                opt_step_start_time = time.time()

            step_start_time = time.time()

            # Get sequences: either from prefetch or generate synchronously (first batch)
            if prefetch_future is not None:
                prefetch_wait_start = time.time()
                sequences, prompt_len = prefetch_future.result()
                prefetch_wait_sec = time.time() - prefetch_wait_start
                gen_time = time.time() - prefetch_start_time
            else:
                gen_start_time = time.time()
                gen_future = vllm_executor.submit(
                    generate_rollouts,
                    vllm_student, batch, PAD_TOKEN_ID, group_size, max_context
                )
                sequences, prompt_len = gen_future.result()
                gen_time = time.time() - gen_start_time
                prefetch_wait_sec = 0.0

            # Start prefetching next batch (runs in background during training)
            if batch_idx + 1 < len(all_batches):
                next_batch = all_batches[batch_idx + 1]
                prefetch_start_time = time.time()
                prefetch_future = vllm_executor.submit(
                    generate_rollouts,
                    vllm_student, next_batch, PAD_TOKEN_ID, group_size, max_context
                )
            else:
                prefetch_future = None
            num_generated_tokens = (sequences.shape[1] - prompt_len) * sequences.shape[
                0
            ]
            tokens_per_sec = num_generated_tokens / gen_time if gen_time > 0 else 0
            prefetch_wait_accum += prefetch_wait_sec
            prefetch_wait_count += 1

            # Move inputs once
            student_input = sequences.to(STUDENT_DEVICE, non_blocking=True)
            teacher_input = sequences.to(TEACHER_DEVICE, non_blocking=True)

            # Create shift labels on GPU to avoid CPU->GPU copy
            shift_labels = create_shift_labels(student_input, prompt_len, PAD_TOKEN_ID)

            # Get hidden states (not logits!) for fused kernel
            student_out = student_base_model(input_ids=student_input)
            student_hidden = student_out.last_hidden_state.view(-1, student_hidden_size)

            # Teacher forward without gradients
            with torch.inference_mode():
                teacher_out = teacher_base_model(input_ids=teacher_input)
                teacher_hidden = teacher_out.last_hidden_state.view(
                    -1, teacher_hidden_size
                )
                teacher_hidden = teacher_hidden.to(STUDENT_DEVICE, non_blocking=True).detach()

            # Fused loss (hidden states + weights -> loss directly, no logits materialized)
            loss = fused_jsd(
                student_hidden,
                student_lm_head_weight,
                teacher_hidden,
                teacher_lm_head_weight,
                shift_labels,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step every GRAD_ACCUM_STEPS
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # Verify gradients exist (first few steps, before zero_grad)
                if global_step < 3:
                    has_grads = any(
                        p.grad is not None
                        for p in student.parameters()
                        if p.requires_grad
                    )
                    print(f"Step {global_step + 1}: gradients exist = {has_grads}")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step_time = time.time() - step_start_time
                opt_step_time = None
                if opt_step_start_time is not None:
                    opt_step_time = time.time() - opt_step_start_time
                avg_loss = accumulated_loss / GRAD_ACCUM_STEPS
                avg_prefetch_wait = (
                    prefetch_wait_accum / prefetch_wait_count
                    if prefetch_wait_count > 0
                    else 0.0
                )

                # Log metrics
                log_payload = {
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time_sec": step_time,
                    "train/prefetch_wait_sec": avg_prefetch_wait,
                    "train/learning_rate": LR,
                    "train/global_step": global_step,
                }
                if opt_step_time is not None:
                    log_payload["train/optimizer_step_time_sec"] = opt_step_time
                if last_sync_duration is not None:
                    log_payload["train/sync_duration_sec"] = last_sync_duration
                    last_sync_duration = None
                log_payload["train/sync_every_n_steps"] = SYNC_EVERY_N_STEPS
                wandb.log(log_payload)

                accumulated_loss = 0.0
                prefetch_wait_accum = 0.0
                prefetch_wait_count = 0
                global_step += 1
                pbar.update(1)

                # Sync updated weights to vLLM for on-policy generation
                if global_step % SYNC_EVERY_N_STEPS == 0:
                    if sync_future is None or sync_future.done():
                        sync_future = vllm_executor.submit(
                            timed_sync_weights_to_vllm, student, vllm_student
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
                        student, tokenizer, optimizer, global_step, hub_repo
                    )

            # Profiler step
            if profiler is not None:
                profiler.step()
                if global_step >= PROFILE_STEPS + 2:  # wait + warmup + active
                    profiler.stop()
                    print(f"Profiler stopped. View with: tensorboard --logdir=./profiler_logs")
                    profiler = None

    pbar.close()

    # Wait for any pending sync/checkpoint
    if sync_future is not None:
        sync_future.result()
    if checkpoint_future is not None:
        checkpoint_future.result()

    vllm_executor.shutdown(wait=True)
    checkpoint_executor.shutdown(wait=True)

    # Final save (synchronous - we're done anyway)
    hub_repo = None if DEBUG_MODE else HUB_REPO
    save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
