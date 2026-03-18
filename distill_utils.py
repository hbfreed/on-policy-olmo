"""Shared utilities for distillation scripts."""

import io
import os
import shutil
import time

import cloudpickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

# --- Model & dataset constants ---
DATASET = "allenai/Dolci-Instruct-RL"
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B-Instruct"
HUB_REPO = None  # "hbfreed/Olmo-2-1B-Distilled"
WANDB_PROJECT = "olmo-2-1b-on-policy-distillation"
RUN_NAME = None  # set to a string to override auto naming

# --- Device layout ---
STUDENT_DEVICE = "cuda:2"  # HF student for training
TEACHER_DEVICE = "cuda:1"  # HF teacher for inference
VLLM_DEVICE = "cuda:0"  # vLLM student for fast generation

# --- Training hyperparameters ---
BATCH_SIZE = 1
N_EPOCHS = 1
GROUP_SIZE = 4  # number of rollouts per prompt
MICRO_BATCH_SIZE = 2  # sequences per student/teacher forward pass
TEACHER_MICRO_BATCH_SIZE = 6
GRAD_ACCUM_STEPS = 256
MAX_CONTEXT_LENGTH = 2048
LR = 1e-5
CLIP_EPS = 0.2
MAX_GRAD_NORM = 3.0
WARMUP_STEPS = 50
SYNC_EVERY_N_STEPS = 4
SYNC_MIN = 1
SYNC_MAX = 4

# --- Eval & logging ---
DEBUG_MODE = False
N_SAMPLE_PROMPTS = 4
SAMPLE_EVERY_N_STEPS = 50
EVAL_EVERY_N_STEPS = 50
EVAL_N_SAMPLES = 200
EVAL_TASKS = ["gsm8k_cot", "arc_easy", "truthfulqa_mc2", "ifeval"]
TEACHER_TOP_K = 128
SFT_DATASET = "allenai/Dolci-Instruct-SFT"


def pack_sequences(dataset, pack_length, pad_token_id, save_path=None):
    """First-fit-decreasing bin packing into fixed-length chunks.

    Uses SortedList for O(N sqrt(B)) bin assignment, then writes packed arrays
    directly to disk via numpy memmap to avoid holding everything in RAM.

    If save_path is given, writes 4 memmap files there and returns the path.
    Otherwise returns a dict of numpy arrays (caution: large).
    """
    import numpy as np
    from sortedcontainers import SortedList

    # Phase 1: compute lengths as numpy array
    import time as _time
    print("  Computing sequence lengths...", flush=True)
    t0 = _time.time()
    lengths_col = dataset.map(
        lambda ex: {"_len": len(ex["input_ids"])},
        num_proc=24, remove_columns=dataset.column_names,
    )
    print(f"  Map done ({_time.time()-t0:.1f}s). Fetching to numpy...", flush=True)
    t0 = _time.time()
    lengths = np.array(lengths_col["_len"], dtype=np.int32)
    del lengths_col
    print(f"  Fetch done ({_time.time()-t0:.1f}s). Sorting {len(lengths)} lengths...", flush=True)
    t0 = _time.time()
    n_seqs = len(lengths)
    sorted_indices = np.argsort(-lengths)  # descending, C-level sort
    print(f"  Sort done ({_time.time()-t0:.1f}s).", flush=True)

    # Phase 2: bin assignment using SortedList (O(sqrt(N)) insert/remove)
    # Each entry is (capacity, bin_id) — SortedList sorts by capacity
    bins_sorted = SortedList()  # stores (capacity, bin_id)
    bin_contents = []           # bin_id -> [seq indices]
    next_bin_id = 0

    for idx in tqdm(sorted_indices, desc="Bin packing", unit="seq"):
        seq_len = int(lengths[idx])
        # Find smallest bin with enough capacity
        pos = bins_sorted.bisect_left((seq_len,))
        if pos < len(bins_sorted):
            old_cap, bin_id = bins_sorted.pop(pos)
            new_cap = old_cap - seq_len
            bin_contents[bin_id].append(idx)
            bins_sorted.add((new_cap, bin_id))
        else:
            # New bin
            bin_id = next_bin_id
            next_bin_id += 1
            bin_contents.append([idx])
            bins_sorted.add((pack_length - seq_len, bin_id))

    n_chunks = len(bin_contents)
    del bins_sorted, sorted_indices  # free memory

    # Phase 3: build packed arrays — write directly to memmap files
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        input_ids_mm = np.memmap(f"{save_path}/input_ids.npy", dtype=np.int32,
                                 mode="w+", shape=(n_chunks, pack_length))
        position_ids_mm = np.memmap(f"{save_path}/position_ids.npy", dtype=np.int16,
                                    mode="w+", shape=(n_chunks, pack_length))
        loss_mask_mm = np.memmap(f"{save_path}/loss_mask.npy", dtype=np.uint8,
                                 mode="w+", shape=(n_chunks, pack_length))
        pad_mask_mm = np.memmap(f"{save_path}/pad_mask.npy", dtype=np.bool_,
                                mode="w+", shape=(n_chunks, pack_length))
        # Save shape metadata
        np.save(f"{save_path}/meta.npy", np.array([n_chunks, pack_length]))
    else:
        input_ids_mm = np.full((n_chunks, pack_length), pad_token_id, dtype=np.int32)
        position_ids_mm = np.zeros((n_chunks, pack_length), dtype=np.int16)
        loss_mask_mm = np.zeros((n_chunks, pack_length), dtype=np.uint8)
        pad_mask_mm = np.zeros((n_chunks, pack_length), dtype=np.bool_)

    # Fill chunk by chunk, fetching from Arrow one sequence at a time
    for chunk_idx, seq_indices in enumerate(tqdm(bin_contents, desc="Building chunks", unit="chunk")):
        offset = 0
        for idx in seq_indices:
            row = dataset[int(idx)]
            ids = row["input_ids"]
            mask = row["loss_mask"]
            seq_len = len(ids)
            input_ids_mm[chunk_idx, offset:offset + seq_len] = ids
            position_ids_mm[chunk_idx, offset:offset + seq_len] = np.arange(seq_len, dtype=np.int16)
            loss_mask_mm[chunk_idx, offset:offset + seq_len] = mask
            pad_mask_mm[chunk_idx, offset:offset + seq_len] = True
            offset += seq_len
        # Tail is already zero/False/pad from initialization

    if save_path is not None:
        # Flush memmaps
        del input_ids_mm, position_ids_mm, loss_mask_mm, pad_mask_mm
        return save_path

    return {
        "input_ids": input_ids_mm,
        "position_ids": position_ids_mm,
        "loss_mask": loss_mask_mm,
        "pad_mask": pad_mask_mm,
    }


def fused_partial_kl(student_logits, teacher_top_ids, teacher_top_lps, loss_mask):
    """Partial forward KL without materializing full [B,T,V] log_softmax.

    All inputs are already shifted/aligned by the caller.
    student_logits: [B, T, V]
    teacher_top_ids: [B, T, K]
    teacher_top_lps: [B, T, K]
    loss_mask: [B, T]
    """
    lse = torch.logsumexp(student_logits, dim=-1)  # [B, T]
    student_at_tops = student_logits.gather(-1, teacher_top_ids) - lse.unsqueeze(-1)  # [B, T, K]

    teacher_probs = teacher_top_lps.exp()
    per_token_kl = (teacher_probs * (teacher_top_lps - student_at_tops)).sum(dim=-1)  # [B, T]
    return (per_token_kl * loss_mask).sum() / loss_mask.sum()


def get_sync_interval(step, mean_ratio, approx_drift, current_interval, sync_state):
    """Adaptive sync interval based on policy drift metrics.

    sync_state: mutable dict with key 'steps_since_decrease' (int).
    """
    if step < 5:
        return 1

    # Danger — policy drifted too far, importance sampling unreliable
    if abs(mean_ratio - 1.0) > 0.2 or approx_drift > 0.25:
        sync_state["steps_since_decrease"] = 0
        return max(SYNC_MIN, current_interval // 2)

    sync_state["steps_since_decrease"] += 1

    # Comfortable for a while — try pushing
    if (abs(mean_ratio - 1.0) < 0.05 and approx_drift < 0.08
            and sync_state["steps_since_decrease"] > 20):
        return min(SYNC_MAX, current_interval + 1)

    return current_interval


def get_logprobs_at_tokens(logits, tokens, vocab_size=None):
    if vocab_size is not None:
        logits = logits[:, :, :vocab_size]
    # Use F.cross_entropy which fuses log_softmax + gather internally,
    # avoiding materializing the full [B, T, V] log_softmax tensor (~1.5 GiB).
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return -F.cross_entropy(
        shift_logits.view(B * T, V), shift_labels.view(B * T),
        reduction="none",
    ).view(B, T)


def run_teacher_pipeline(teacher, sequences, attention_mask, chunk_size,
                         device, student_device, queue, consumer_chunk_size=None):
    """Producer: compute teacher logprobs in chunks, emit exact consumer-sized pieces.

    Buffers across teacher chunks to handle non-aligned sizes (e.g. teacher=6, consumer=4).
    """
    if consumer_chunk_size is None:
        consumer_chunk_size = chunk_size
    try:
        buffer = []
        buffered_rows = 0
        for i in range(0, len(sequences), chunk_size):
            chunk_seq = sequences[i:i + chunk_size].to(device, non_blocking=True)
            chunk_mask = attention_mask[i:i + chunk_size].to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    t_out = teacher(input_ids=chunk_seq, attention_mask=chunk_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Teacher OOM with batch_size={chunk_seq.shape[0]}, "
                    f"seq_len={chunk_seq.shape[1]}. Reduce TEACHER_MICRO_BATCH_SIZE "
                    f"(currently {chunk_size})."
                )
            logprobs = get_logprobs_at_tokens(t_out.logits, chunk_seq)
            logprobs = logprobs.to(student_device).detach()
            buffer.append(logprobs)
            buffered_rows += logprobs.shape[0]
            # Emit complete consumer-sized chunks from buffer
            while buffered_rows >= consumer_chunk_size:
                combined = torch.cat(buffer, dim=0)
                queue.put(combined[:consumer_chunk_size])
                remainder = combined[consumer_chunk_size:]
                buffer = [remainder] if remainder.shape[0] > 0 else []
                buffered_rows = remainder.shape[0]
        # Drop any leftover rows that don't fill a complete consumer chunk
        # (student loop only processes n_sequences // consumer_chunk_size chunks)
        queue.put(None)  # sentinel
    except Exception as e:
        queue.put(e)  # unblock consumer so it doesn't hang
        raise


def generate_rollouts(
    vllm_student, prompts, pad_token_id, group_size=1, max_context_length=4096, vocab_size=None
):
    """Generate rollouts from student model using vLLM, returning sequences and prompt length."""
    from vllm import SamplingParams

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
    # Replace token IDs outside the shared vocab with pad so they're masked
    # out of attention and loss (student's padded vocab > teacher's vocab)
    if vocab_size is not None:
        sequences[sequences >= vocab_size] = pad_token_id
    attention_mask = (sequences != pad_token_id).long()
    old_logprobs = torch.tensor(padded_logprobs)
    expanded_prompt_lens = [pl for pl in prompt_lens for _ in range(group_size)]

    return sequences, expanded_prompt_lens, old_logprobs, attention_mask


def prepare_prompts(opt_step_idx, all_batches, tokenizer, grad_accum_steps):
    """Get pretokenized prompts for a given optimizer step index."""
    chunk_start = opt_step_idx * grad_accum_steps
    chunk_end = chunk_start + grad_accum_steps
    return [all_batches[i]["input_ids_prompt"] for i in range(chunk_start, chunk_end)]


def timed_generate_rollouts(*args, **kwargs):
    """Wrapper that returns generate_rollouts results plus elapsed time."""
    t0 = time.time()
    result = generate_rollouts(*args, **kwargs)
    return (*result, time.time() - t0)


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
    from vllm import SamplingParams

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


def save_checkpoint(student, tokenizer, optimizer, global_step,
                    checkpoint_base, milestone_every=500, hub_repo=None,
                    state_dict_cpu=None, opt_state=None):
    """Save rolling 'latest'/'prev' checkpoints, plus a permanent one every milestone_every steps.

    If state_dict_cpu/opt_state are provided, saves from those (for async bg saves).
    Otherwise snapshots from the live model/optimizer.
    """
    latest_dir = f"{checkpoint_base}/latest"
    prev_dir = f"{checkpoint_base}/prev"

    # Rotate: latest -> prev (so we always have two recent checkpoints)
    if os.path.exists(latest_dir):
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        os.rename(latest_dir, prev_dir)

    os.makedirs(latest_dir, exist_ok=True)
    if state_dict_cpu is not None:
        # Save from pre-snapshotted CPU state (async path)
        from safetensors.torch import save_file
        student.config.save_pretrained(latest_dir)
        save_file(state_dict_cpu, f"{latest_dir}/model.safetensors")
    else:
        student.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)
    _opt = opt_state if opt_state is not None else {"optimizer": optimizer.state_dict(), "step": global_step}
    torch.save(_opt, f"{latest_dir}/training_state.pt")
    print(f"Saved latest checkpoint (step {global_step}) to {latest_dir}")

    if milestone_every > 0 and global_step % milestone_every == 0:
        milestone_dir = f"{checkpoint_base}/step_{global_step}"
        if state_dict_cpu is not None:
            shutil.copytree(latest_dir, milestone_dir)
        else:
            os.makedirs(milestone_dir, exist_ok=True)
            student.save_pretrained(milestone_dir)
            tokenizer.save_pretrained(milestone_dir)
            torch.save(
                {"optimizer": optimizer.state_dict(), "step": global_step},
                f"{milestone_dir}/training_state.pt",
            )
        print(f"Saved milestone checkpoint to {milestone_dir}")

    if hub_repo:
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=latest_dir,
                repo_id=hub_repo,
                commit_message=f"Step {global_step}",
                ignore_patterns=["training_state.pt"],
            )
            print(f"Pushed checkpoint to {hub_repo}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


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
