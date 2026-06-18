"""Shared functions for distillation scripts. Configs live in each script."""

import io
import os
import shutil
import time

import cloudpickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

# Adaptive sync bounds — used by get_sync_interval (legacy PG-shaped path).
SYNC_MIN = 1
SYNC_MAX = 1


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


def get_logprobs_at_tokens(logits, tokens, vocab_size=None, inplace=False):
    if vocab_size is not None:
        logits = logits[:, :, :vocab_size]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    B, T, V = shift_logits.shape

    if inplace:
        # In-place bf16 logsumexp: reuses logits memory instead of allocating
        # a new [B*T, V] tensor. Only safe under inference_mode() (teacher path).
        flat = shift_logits.view(B * T, V)
        target = flat.gather(1, shift_labels.view(B * T, 1)).squeeze(1)
        max_val = flat.max(dim=1, keepdim=True).values
        flat -= max_val
        flat.exp_()
        lse = flat.sum(dim=1).float().log() + max_val.squeeze(1).float()
        return (target.float() - lse).to(shift_logits.dtype).view(B, T)
    else:
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
        carry = None
        for i in range(0, len(sequences), chunk_size):
            chunk_seq = sequences[i:i + chunk_size].to(device, non_blocking=True)
            chunk_mask = attention_mask[i:i + chunk_size].to(device, non_blocking=True)
            try:
                with torch.inference_mode(), torch.cuda.device(device):
                    t_out = teacher(input_ids=chunk_seq, attention_mask=chunk_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                        f"Teacher OOM with batch_size={chunk_seq.shape[0]}, "
                        f"seq_len={chunk_seq.shape[1]}. Reduce TEACHER_MICRO_BATCH_SIZE "
                        f"(currently {chunk_size})."
                )
            logprobs = get_logprobs_at_tokens(t_out.logits, chunk_seq, inplace=True)
            logprobs = logprobs.to(student_device).detach()

            if carry is not None:
                logprobs = torch.cat((carry, logprobs), dim=0)
                carry = None

            emitted_rows = 0
            while emitted_rows + consumer_chunk_size <= logprobs.shape[0]:
                queue.put(logprobs[emitted_rows:emitted_rows + consumer_chunk_size])
                emitted_rows += consumer_chunk_size

            if emitted_rows < logprobs.shape[0]:
                carry = logprobs[emitted_rows:]
        # Drop any leftover rows that don't fill a complete consumer chunk
        # (student loop only processes n_sequences // consumer_chunk_size chunks)
        queue.put(None)  # sentinel
    except Exception as e:
        queue.put(e)  # unblock consumer so it doesn't hang
        raise


class HiddenCapture:
    """Capture lm_head input via forward pre-hook; short-circuit lm_head with a 1-token dummy.

    Avoids the [B,T,V] lm_head matmul + logits allocation when the caller only needs the
    final hidden state (e.g. to feed a fused-linear distillation kernel). The dummy output
    is discarded by the caller; the captured hidden lives on `self.hidden`.
    """

    def __init__(self, model):
        self.hidden = None
        self.dummy = None
        self._handle = model.lm_head.register_forward_pre_hook(self._pre)

    def _pre(self, module, args):
        x = args[0]
        self.hidden = x
        if (self.dummy is None
                or self.dummy.shape[0] != x.shape[0]
                or self.dummy.shape[2] != x.shape[2]
                or self.dummy.dtype != x.dtype):
            self.dummy = x.new_zeros(x.shape[0], 1, x.shape[2])
        return (self.dummy,) + args[1:]

    def close(self):
        self._handle.remove()


def active_lengths_from_attention(attention_mask):
    """Return the last non-pad position + 1 for each row."""
    if attention_mask.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=attention_mask.device)
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
        dtype=torch.long,
    ).unsqueeze(0)
    return torch.where(attention_mask.bool(), positions + 1, 0).max(dim=1).values


def _pad_seq_dim(tensor, target_len):
    if tensor.shape[1] == target_len:
        return tensor
    if tensor.shape[1] > target_len:
        return tensor[:, :target_len]
    if tensor.dim() == 2:
        pad = tensor.new_zeros(tensor.shape[0], target_len - tensor.shape[1])
    else:
        pad = tensor.new_zeros(
            tensor.shape[0], target_len - tensor.shape[1], *tensor.shape[2:]
        )
    return torch.cat((tensor, pad), dim=1)


def run_teacher_pipeline_hidden(teacher, sequences, attention_mask, chunk_size,
                                device, student_device, queue, consumer_chunk_size=None):
    """Producer: run teacher forward in chunks, ship last hidden state (post-norm).

    Uses HiddenCapture on lm_head to avoid materializing teacher logits. Buffers across
    teacher chunks to handle non-aligned consumer sizes.
    """
    if consumer_chunk_size is None:
        consumer_chunk_size = chunk_size
    cap = HiddenCapture(teacher)
    try:
        carry = None
        carry_mask = None
        for i in range(0, len(sequences), chunk_size):
            chunk_mask_cpu = attention_mask[i:i + chunk_size]
            active_len = int(active_lengths_from_attention(chunk_mask_cpu).max().item())
            chunk_seq = sequences[i:i + chunk_size, :active_len].to(
                device, non_blocking=True
            )
            chunk_mask = chunk_mask_cpu[:, :active_len].to(device, non_blocking=True)
            try:
                with torch.inference_mode(), torch.cuda.device(device):
                    teacher(input_ids=chunk_seq, attention_mask=chunk_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Teacher OOM with batch_size={chunk_seq.shape[0]}, "
                    f"seq_len={chunk_seq.shape[1]}. Reduce TEACHER_MICRO_BATCH_SIZE "
                    f"(currently {chunk_size})."
                )
            hidden = cap.hidden.to(student_device, non_blocking=True).detach()
            hidden_mask = chunk_mask_cpu[:, :active_len]

            if carry is not None:
                target_len = max(carry.shape[1], hidden.shape[1])
                carry = _pad_seq_dim(carry, target_len)
                hidden = _pad_seq_dim(hidden, target_len)
                carry_mask = _pad_seq_dim(carry_mask, target_len)
                hidden_mask = _pad_seq_dim(hidden_mask, target_len)
                hidden = torch.cat((carry, hidden), dim=0)
                hidden_mask = torch.cat((carry_mask, hidden_mask), dim=0)
                carry = None
                carry_mask = None

            emitted = 0
            while emitted + consumer_chunk_size <= hidden.shape[0]:
                piece_mask = hidden_mask[emitted:emitted + consumer_chunk_size]
                piece_len = int(active_lengths_from_attention(piece_mask).max().item())
                queue.put(hidden[emitted:emitted + consumer_chunk_size, :piece_len])
                emitted += consumer_chunk_size
            if emitted < hidden.shape[0]:
                carry = hidden[emitted:]
                carry_mask = hidden_mask[emitted:]
        queue.put(None)
    except Exception as e:
        queue.put(e)
        raise
    finally:
        cap.close()


def generate_rollouts(
    vllm_student, prompts, pad_token_id, group_size=1, max_context_length=4096, vocab_size=None,
    with_logprobs=True, sort_by_length=False,
):
    """Generate rollouts from student model using vLLM, returning sequences and prompt length.

    If `with_logprobs=False`, the returned `old_logprobs` value is None. vLLM
    `logprobs=1` adds nontrivial overhead, so callers that don't need importance
    sampling should pass False.
    """
    from vllm import SamplingParams

    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    max_new_tokens = max_context_length - max_prompt_len

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=group_size,
        logprobs=1 if with_logprobs else None,
        stop_token_ids=[pad_token_id],
        detokenize=False,
    )

    token_prompts = [{"prompt_token_ids": p} for p in prompts]
    outputs = vllm_student.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Convert vLLM outputs to tensor: each RequestOutput has `outputs` list
    # With n=group_size, we get group_size completions per prompt
    n_sequences = sum(len(req_output.outputs) for req_output in outputs)
    max_seq_len = 0
    for req_output, prompt_len in zip(outputs, prompt_lens):
        for completion in req_output.outputs:
            max_seq_len = max(max_seq_len, prompt_len + len(completion.token_ids))

    sequences = torch.full((n_sequences, max_seq_len), pad_token_id, dtype=torch.long)
    old_logprobs = (
        torch.zeros((n_sequences, max_seq_len), dtype=torch.float32)
        if with_logprobs else None
    )
    hit_eos = torch.empty(n_sequences, dtype=torch.bool)
    expanded_prompt_lens = [0] * n_sequences
    full_seq_lens = [0] * n_sequences  # per-row length used for attention_mask

    row = 0
    for req_output, prompt_len in zip(outputs, prompt_lens):
        prompt_tensor = torch.as_tensor(req_output.prompt_token_ids, dtype=torch.long)
        for completion in req_output.outputs:
            gen_len = len(completion.token_ids)
            full_seq_len = prompt_len + gen_len
            sequences[row, :prompt_len] = prompt_tensor
            if gen_len:
                sequences[row, prompt_len:full_seq_len] = torch.as_tensor(
                    completion.token_ids, dtype=torch.long
                )
                if with_logprobs and completion.logprobs is not None:
                    token_logprobs = [
                        logprob_dict[token_id].logprob
                        for token_id, logprob_dict in zip(
                            completion.token_ids, completion.logprobs
                        )
                    ]
                    old_logprobs[row, prompt_len:full_seq_len] = torch.as_tensor(
                        token_logprobs, dtype=torch.float32
                    )
            hit_eos[row] = completion.finish_reason == "stop"
            expanded_prompt_lens[row] = prompt_len
            full_seq_lens[row] = full_seq_len
            row += 1

    # Replace token IDs outside the shared vocab with pad so they're masked
    # out of attention and loss (student's padded vocab > teacher's vocab)
    if vocab_size is not None:
        sequences[sequences >= vocab_size] = pad_token_id
    # Build attention_mask from per-row sequence ends (NOT token-id equality).
    # When pad_token_id == eos_token_id (SmolLM2 case), token-equality would
    # zero out internal EOS positions, denying gradient on EOS prediction.
    attention_mask = torch.zeros_like(sequences, dtype=torch.long)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    full_lens_t = torch.tensor(full_seq_lens, dtype=torch.long).unsqueeze(1)
    attention_mask = (positions < full_lens_t).long()

    if sort_by_length and n_sequences > 1:
        lengths = active_lengths_from_attention(attention_mask)
        order = torch.argsort(lengths, descending=True)
        order_list = order.tolist()
        sequences = sequences[order]
        if old_logprobs is not None:
            old_logprobs = old_logprobs[order]
        attention_mask = attention_mask[order]
        hit_eos = hit_eos[order]
        expanded_prompt_lens = [expanded_prompt_lens[i] for i in order_list]

    if torch.cuda.is_available():
        sequences = sequences.pin_memory()
        if old_logprobs is not None:
            old_logprobs = old_logprobs.pin_memory()
        attention_mask = attention_mask.pin_memory()
        hit_eos = hit_eos.pin_memory()

    return sequences, expanded_prompt_lens, old_logprobs, attention_mask, hit_eos


def prepare_prompts(opt_step_idx, all_batches, tokenizer, grad_accum_steps):
    """Get pretokenized prompts for a given optimizer step index."""
    chunk_start = opt_step_idx * grad_accum_steps
    chunk_end = chunk_start + grad_accum_steps
    prompts = []
    for i in range(chunk_start, chunk_end):
        p = all_batches[i]["input_ids_prompt"]
        # dataset.iter(batch_size=1) wraps each row in an outer list
        if isinstance(p[0], list):
            p = p[0]
        prompts.append(p)
    return prompts


def timed_generate_rollouts(*args, **kwargs):
    """Wrapper that returns generate_rollouts results plus elapsed time."""
    t0 = time.time()
    result = generate_rollouts(*args, **kwargs)
    return (*result, time.time() - t0)


def build_loss_mask(sequences, prompt_lens, pad_token_id, attention_mask=None):
    """
    Build a mask that's 1.0 for completion tokens, 0.0 for prompt and padding.

    sequences: [batch, seq_len]
    prompt_lens: list[int], length = batch (per-sequence prompt lengths)
    pad_token_id: int
    attention_mask: [batch, seq_len], optional. If provided, uses this to
        determine valid (non-pad) positions instead of token-id equality —
        which is critical when pad_token_id == eos_token_id (SmolLM2).

    Returns: [batch, seq_len - 1] (shifted to match logprob indexing)
    """
    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
    prompt_lens_t = torch.tensor(prompt_lens, device=sequences.device).unsqueeze(1)
    mask = (positions >= prompt_lens_t).float()
    if attention_mask is not None:
        mask = mask * attention_mask.float().to(mask.device)
    else:
        mask[sequences == pad_token_id] = 0.0
    return mask[:, 1:]


def build_shift_labels(attention_mask, prompt_lens, ignore_index=-100):
    """Build labels used only for ignore masking in the fused JSD kernel."""
    batch_size, seq_len = attention_mask.shape
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
    prompt_lens_t = torch.tensor(prompt_lens, device=attention_mask.device).unsqueeze(1)
    valid = (positions >= prompt_lens_t) & attention_mask.bool()
    labels = torch.full(
        (batch_size, seq_len),
        ignore_index,
        dtype=torch.long,
        device=attention_mask.device,
    )
    labels[valid] = 0
    return labels[:, 1:].contiguous()


def generate_samples(vllm_student, eval_prompts, tokenizer, max_context_length=4096):
    """Generate completions for eval prompts and return a wandb.Table."""
    from vllm import SamplingParams

    prompt_lens = [len(p) for p in eval_prompts]
    max_prompt_len = max(prompt_lens)
    eos_id = tokenizer.eos_token_id
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_context_length - max_prompt_len,
        n=1,
        stop_token_ids=[eos_id],
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
