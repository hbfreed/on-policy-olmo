import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import bitsandbytes as bnb
import cloudpickle
import torch
import torch.nn.functional as F
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

# DATASET = "allenai/Dolci-Think-RL-7B"
DATASET = "allenai/Dolci-Instruct-RL"
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B-Instruct"
HUB_REPO = None  # "hbfreed/Olmo-2-1B-Distilled"
WANDB_PROJECT = "olmo-2-1b-on-policy-distillation"
RUN_NAME = (
    "instruct-student-instruct-teacher"  # set to a string to override auto naming
)

STUDENT_DEVICE = "cuda:2"  # HF student for training
TEACHER_DEVICE = "cuda:1"  # HF teacher for inference
VLLM_DEVICE = "cuda:0"  # vLLM student for fast generation (vLLM uses first visible GPU)

BATCH_SIZE = 1
N_EPOCHS = 1
GROUP_SIZE = 2  # number of rollouts per prompt
GRAD_ACCUM_STEPS = 64
MAX_CONTEXT_LENGTH = 2048
LR = 1e-7
SYNC_EVERY_N_STEPS = 1
SYNC_MIN = 1
SYNC_MAX = 16

RESUME_FROM = None  # or "checkpoints/step_1000"
DEBUG_MODE = False
N_SAMPLE_PROMPTS = 4
SAMPLE_EVERY_N_STEPS = 200

steps_since_decrease = 0

torch.manual_seed(1223)


def get_sync_interval(step, mean_ratio, approx_drift, current_interval):
    global steps_since_decrease

    if step < 50:
        return 1

    # Danger — policy drifted too far, importance sampling unreliable
    if abs(mean_ratio - 1.0) > 0.2 or approx_drift > 0.25:
        steps_since_decrease = 0
        return max(SYNC_MIN, current_interval // 2)

    steps_since_decrease += 1

    # Comfortable for a while — try pushing
    if (abs(mean_ratio - 1.0) < 0.05 and approx_drift < 0.08
            and steps_since_decrease > 20):
        return min(SYNC_MAX, current_interval + 1)

    return current_interval


def get_logprobs_at_tokens(logits, tokens):
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


def run_teacher_pipeline(teacher, sequences, attention_mask, group_size,
                         device, student_device, queue):
    """Producer: compute teacher logprobs in chunks, push to queue."""
    for i in range(0, len(sequences), group_size):
        chunk_seq = sequences[i:i + group_size].to(device, non_blocking=True)
        chunk_mask = attention_mask[i:i + group_size].to(device, non_blocking=True)
        with torch.inference_mode():
            t_out = teacher(input_ids=chunk_seq, attention_mask=chunk_mask)
        logprobs = get_logprobs_at_tokens(t_out.logits, chunk_seq)
        queue.put(logprobs.to(student_device).detach())
    queue.put(None)  # sentinel


def generate_rollouts(
    vllm_student, prompts, pad_token_id, group_size=1, max_context_length=4096
):
    """Generate rollouts from student model using vLLM, returning sequences and prompt length."""
    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    max_new_tokens = max_context_length - max_prompt_len

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=group_size,
        logprobs=1,
    )

    token_prompts = [{"prompt_token_ids": p} for p in prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Convert vLLM outputs to tensor: each RequestOutput has `outputs` list
    # With n=group_size, we get group_size completions per prompt
    all_sequences = []
    all_logprobs = []
    for req_output, prompt_len in zip(outputs, prompt_lens):
        prompt_ids = req_output.prompt_token_ids
        for completion in req_output.outputs:
            # Combine prompt + generated tokens
            full_seq = list(prompt_ids) + list(completion.token_ids)
            all_sequences.append(full_seq)
            seq_logprobs = [0.0] * prompt_len
            for idx, logprob_dict in enumerate(completion.logprobs):
                token_id = completion.token_ids[idx]
                seq_logprobs.append(logprob_dict[token_id].logprob)
            all_logprobs.append(seq_logprobs)

    # Pad sequences to same length (right pad with pad_token_id)
    max_seq_len = max(len(seq) for seq in all_sequences)
    padded = [seq + [pad_token_id] * (max_seq_len - len(seq)) for seq in all_sequences]
    padded_logprobs = [
        logprob + [0.0] * (max_seq_len - len(logprob)) for logprob in all_logprobs
    ]

    sequences = torch.tensor(padded)
    attention_mask = (sequences != pad_token_id).long()
    old_logprobs = torch.tensor(padded_logprobs)
    expanded_prompt_lens = [pl for pl in prompt_lens for _ in range(group_size)]

    return sequences, expanded_prompt_lens, old_logprobs, attention_mask


def build_loss_mask(sequences, prompt_lens, pad_token_id):
    """
    Build a mask that's 1.0 for completion tokens, 0.0 for prompt and padding.

    sequences: [batch, seq_len]
    prompt_lens: list[int], length = batch (per-sequence prompt lengths)
    pad_token_id: int

    Returns: [batch, seq_len - 1] (shifted to match logprob indexing)
    """
    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
    prompt_lens_t = torch.tensor(prompt_lens, device=sequences.device).unsqueeze(1)
    mask = (positions >= prompt_lens_t).float()
    mask[sequences == pad_token_id] = 0.0
    return mask[:, 1:]


def generate_samples(vllm_student, eval_prompts, tokenizer, max_context_length=4096):
    """Generate completions for eval prompts and return a wandb.Table."""
    prompt_lens = [len(p) for p in eval_prompts]
    max_prompt_len = max(prompt_lens)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_context_length - max_prompt_len,
        n=1,
    )
    token_prompts = [{"prompt_token_ids": p} for p in eval_prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    table = wandb.Table(columns=["prompt", "completion"])
    for req_output in outputs:
        prompt_text = tokenizer.decode(
            req_output.prompt_token_ids, skip_special_tokens=True
        )
        completion_text = tokenizer.decode(
            req_output.outputs[0].token_ids, skip_special_tokens=True
        )
        table.add_data(prompt_text, completion_text)
    return table


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
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=checkpoint_dir,
                repo_id=hub_repo,
                commit_message=f"Step {global_step}",
                ignore_patterns=["training_state.pt"],
            )
            print(f"Pushed checkpoint to {hub_repo}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


def sync_weights_to_vllm(hf_model, vllm_llm):
    """Sync weights from HF model to vLLM engine for on-policy learning.

    Uses collective_rpc to update weights in V1 architecture.
    See: https://github.com/vllm-project/vllm/issues/5723
    """
    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(hf_state_dict, buffer)
    weights_bytes = buffer.getvalue()

    def load_weights_on_worker(worker, serialized_weights):
        buf = io.BytesIO(serialized_weights)
        weights_dict = torch.load(buf, weights_only=True)
        weights = list(weights_dict.items())
        worker.model_runner.model.load_weights(weights=weights)

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

    # Fixed eval prompts for tracking generation quality over training
    eval_prompts = [
        dataset[i]["input_ids_prompt"]
        for i in range(min(N_SAMPLE_PROMPTS, len(dataset)))
    ]

    batch_size = BATCH_SIZE
    group_size = GROUP_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // (batch_size * GRAD_ACCUM_STEPS)
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

    # Initialize vLLM for fast generation on separate GPU
    # skip_tokenizer_init=True since we input token IDs directly
    print(f"Loading vLLM student on {VLLM_DEVICE}...")
    vllm_student = LLM(
        STUDENT,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )

    student.gradient_checkpointing_enable()

    student_compiled = student
    teacher_compiled = teacher

    optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=LR)

    start_step = 0
    if RESUME_FROM:
        start_step = load_checkpoint(RESUME_FROM, student, optimizer, vllm_student)
        print(f"Resuming from step {start_step}")

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

    # Baseline generation samples before any training
    baseline_table = generate_samples(vllm_student, eval_prompts, tokenizer, max_context)
    wandb.log({"eval/samples": baseline_table}, step=0)

    # Training loop
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # Create progress bar
    total_optimizer_steps = steps_per_epoch * N_EPOCHS
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
    last_sync_duration = None
    sync_interval = SYNC_EVERY_N_STEPS

    for epoch in range(N_EPOCHS):
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

            opt_step_start_time = time.time()

            # Collect GRAD_ACCUM_STEPS prompts and generate all rollouts at once
            chunk_start = opt_step_idx * GRAD_ACCUM_STEPS
            chunk_end = chunk_start + GRAD_ACCUM_STEPS
            prompts = [all_batches[i]["input_ids_prompt"] for i in range(chunk_start, chunk_end)]
            # Each prompt comes as a list of token ids from the dataset;
            # with batch_size=1, iter yields single-element lists — unwrap them
            prompts = [p[0] if isinstance(p, list) and isinstance(p[0], list) else p for p in prompts]

            gen_start_time = time.time()
            sequences, prompt_lens, old_student_logprobs, attention_mask = (
                generate_rollouts(
                    vllm_student, prompts, PAD_TOKEN_ID, group_size, max_context
                )
            )
            gen_time = time.time() - gen_start_time

            # Print a decoded rollout to check if training data is coherent
            if opt_step_idx < 2:
                print(f"[opt_step {opt_step_idx}] Rollout sample: {tokenizer.decode(sequences[0].tolist()[:200])}")

            total_generated_tokens = 0

            # Producer-consumer pipeline: teacher is faster (no backward, no grad
            # checkpointing) so it races ahead by 2-3 chunks, keeping both GPUs busy.
            # maxsize=4 bounds memory so teacher doesn't cache too many logprobs.
            teacher_queue = Queue(maxsize=4)
            teacher_thread = teacher_executor.submit(
                run_teacher_pipeline, teacher_compiled, sequences, attention_mask,
                group_size, TEACHER_DEVICE, STUDENT_DEVICE, teacher_queue
            )

            # Inner loop: micro-batches over pre-generated sequences
            for micro_idx in range(GRAD_ACCUM_STEPS):
                start = micro_idx * group_size
                end = start + group_size
                mb_seq = sequences[start:end]
                mb_plens = prompt_lens[start:end]
                mb_old_lp = old_student_logprobs[start:end]
                mb_mask = attention_mask[start:end]

                # Move inputs once
                student_input = mb_seq.to(STUDENT_DEVICE, non_blocking=True)
                student_mask = mb_mask.to(STUDENT_DEVICE, non_blocking=True)

                student_out = student_compiled(
                    input_ids=student_input,
                    attention_mask=student_mask,
                )
                current_logprobs = get_logprobs_at_tokens(student_out.logits, student_input)

                teacher_logprobs = teacher_queue.get()  # blocks only if teacher behind

                loss_mask = build_loss_mask(student_input, mb_plens, PAD_TOKEN_ID)

                # Gradient chain diagnostics (first 3 steps only)
                if global_step < 3 and micro_idx == 0:
                    print(f"--- Gradient diagnostics (step {global_step}) ---")
                    print(f"logits.grad_fn: {student_out.logits.grad_fn}")
                    print(f"logits.requires_grad: {student_out.logits.requires_grad}")
                    print(f"current_logprobs.grad_fn: {current_logprobs.grad_fn}")
                    print(f"loss_mask sum: {loss_mask.sum().item()}")

                seq_lens = mb_mask.to(STUDENT_DEVICE).sum(dim=1)
                prompt_lens_t = torch.tensor(mb_plens, device=STUDENT_DEVICE)
                avg_gen_len = (seq_lens - prompt_lens_t).float().mean()

                num_generated_tokens = loss_mask.sum().item()
                total_generated_tokens += num_generated_tokens

                old_logprobs_shifted = mb_old_lp[:, 1:].to(STUDENT_DEVICE)
                advantage = -(old_logprobs_shifted - teacher_logprobs)

                ratio = torch.exp(current_logprobs - old_logprobs_shifted)
                eps = 0.2
                ratio_clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                loss_unclipped = -ratio * advantage.detach()
                loss_clipped = -ratio_clipped * advantage.detach()
                per_token_loss = torch.max(loss_unclipped, loss_clipped)  # pessimistic bound
                masked_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()

                # Gradient chain diagnostics continued
                if global_step < 3 and micro_idx == 0:
                    ms = max(loss_mask.sum().item(), 1)
                    print(f"advantage abs mean: {(advantage * loss_mask).abs().sum().item() / ms:.6f}")
                    print(f"ratio mean (should be ~1): {(ratio * loss_mask).sum().item() / ms:.6f}")
                    print(f"masked_loss: {masked_loss.item():.6f}")
                    print(f"masked_loss.grad_fn: {masked_loss.grad_fn}")
                    print("---")

                # Scale loss for gradient accumulation
                scaled_loss = masked_loss / GRAD_ACCUM_STEPS
                scaled_loss.backward()
                accumulated_loss += scaled_loss.item()

            # Drain sentinel and propagate any teacher exception
            assert teacher_queue.get() is None
            teacher_thread.result()

            # --- Optimizer step (after all micro-batches) ---
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

            opt_step_time = time.time() - opt_step_start_time
            avg_loss = accumulated_loss  # already averaged via scaled_loss

            tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0

            # Log metrics — batch all GPU->CPU transfers into one sync
            # (uses values from last micro-batch, same as before)
            mask_sum = loss_mask.sum()
            metrics_tensor = torch.stack(
                [
                    grad_norm,
                    (advantage * loss_mask).sum() / mask_sum,
                    (ratio * loss_mask).sum() / mask_sum,
                    ((old_logprobs_shifted - teacher_logprobs) * loss_mask).sum()
                    / mask_sum,
                    ((ratio > 1.2) | (ratio < 0.8)).float().sum() / mask_sum,
                    (
                        (current_logprobs - old_logprobs_shifted).abs() * loss_mask
                    ).sum()
                    / mask_sum,
                ]
            )
            (
                grad_norm_val,
                mean_advantage,
                mean_ratio,
                mean_kl,
                ratio_clipped_frac,
                approx_policy_drift,
            ) = metrics_tensor.tolist()

            log_payload = {
                "train/loss": avg_loss,
                "train/grad_norm": grad_norm_val,
                "train/tokens_per_sec": tokens_per_sec,
                "train/gen_time_sec": gen_time,
                "train/optimizer_step_time_sec": opt_step_time,
                "train/learning_rate": LR,
                "train/global_step": global_step,
                # Policy gradient diagnostics
                "train/mean_advantage": mean_advantage,
                "train/mean_ratio": mean_ratio,
                "train/mean_kl": mean_kl,
                "train/ratio_clipped_frac": ratio_clipped_frac,
                "train/approx_policy_drift": approx_policy_drift,
                "train/avg_gen_length": avg_gen_len.item(),
            }
            if last_sync_duration is not None:
                log_payload["train/sync_duration_sec"] = last_sync_duration
                last_sync_duration = None
            sync_interval = get_sync_interval(
                global_step, mean_ratio, approx_policy_drift, sync_interval
            )
            log_payload["train/sync_every_n_steps"] = sync_interval
            wandb.log(log_payload)

            accumulated_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Sync updated weights to vLLM for on-policy generation
            if global_step % sync_interval == 0:
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
                    hub_repo,
                )

    pbar.close()

    # Wait for any pending sync/checkpoint/samples
    if sync_future is not None:
        sync_future.result()
    if sample_future is not None:
        wandb.log({"eval/samples": sample_future.result()})
    if checkpoint_future is not None:
        checkpoint_future.result()

    vllm_executor.shutdown(wait=True)
    checkpoint_executor.shutdown(wait=True)
    teacher_executor.shutdown(wait=True)

    # Final save (synchronous - we're done anyway)
    hub_repo = None if DEBUG_MODE else HUB_REPO
    save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
