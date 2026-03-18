"""DDP SFT / off-policy distillation with sequence packing.

Two loss modes:
  - kld: Teacher (OLMo-3-7B-Instruct) provides soft targets via top-K logprobs,
         student minimizes partial forward KL. 1 HF teacher + 2 DDP students.
         Teacher extraction is pipelined: step N+1 prefetched during step N's student work.
  - cce: Standard supervised fine-tuning using cut-cross-entropy (no teacher).
         3 DDP students. lm_head hook avoids materializing full logits.

GPU layout:
  KLD: GPU 0 = HF teacher, GPU 1..2 = DDP student ranks (nproc_per_node=2, offset=1)
  CCE: GPU 0..2 = DDP student ranks (nproc_per_node=3, offset=0)

Usage:
  uv run bash launch_sft.sh kld --sweep 3 --total-batch-size 128
  uv run bash launch_sft.sh cce --sweep 3 --total-batch-size 128
"""

import argparse
import math
import random
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
import torch.distributed as dist
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from distill_utils import (
    EVAL_EVERY_N_STEPS,
    EVAL_N_SAMPLES,
    EVAL_TASKS,
    MAX_GRAD_NORM,
    N_EPOCHS,
    SFT_DATASET,
    STUDENT,
    TEACHER,
    TEACHER_TOP_K,
    WARMUP_STEPS,
    fused_partial_kl,
    load_checkpoint,
    pack_sequences,
    save_checkpoint,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(1223)

WANDB_PROJECT = "olmo-2-1b-sft"
MIN_LR_RATIO = 0.1
DECAY_FRACTION = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="DDP SFT / off-policy distillation")
    parser.add_argument("--loss-type", type=str, choices=["kld", "cce"], required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pack-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--teacher-micro-batch-size", type=int, default=6,
                        help="Teacher sub-chunking (KLD only)")
    parser.add_argument("--total-batch-size", type=int, default=128,
                        help="Target chunks per optimizer step (auto-rounded down to fit micro-batch-size * world_size)")
    parser.add_argument("--gpu-offset", type=int, default=None,
                        help="Rank i -> cuda:{i + offset}. Auto: 1 for KLD, 0 for CCE")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--checkpoint-base", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=None,
                        help=f"Eval frequency (default: EVAL_EVERY_N_STEPS={EVAL_EVERY_N_STEPS})")
    parser.add_argument("--milestone-every", type=int, default=500)
    return parser.parse_args()


def tokenize_and_filter(ds, tokenizer, pack_length):
    """Tokenize conversations with loss mask, filter to pack_length."""

    def tokenize_conversation(example):
        messages = example["messages"]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True)

        # Build loss mask: 1 for assistant tokens, 0 for user/system
        loss_mask = [0] * len(full_ids)
        position = 0
        for i, msg in enumerate(messages):
            partial_ids = tokenizer.apply_chat_template(messages[:i + 1], tokenize=True)
            if msg["role"] == "assistant":
                loss_mask[position:len(partial_ids)] = [1] * (len(partial_ids) - position)
            position = len(partial_ids)

        return {"input_ids": full_ids, "loss_mask": loss_mask}

    dataset = ds.map(tokenize_conversation, remove_columns=ds.column_names, num_proc=24)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= pack_length, num_proc=24)
    return dataset


def extract_teacher_topk(teacher, batch_ids, batch_pad_mask, batch_pos_ids,
                         teacher_device, teacher_micro_batch_size, vocab_size):
    """Run teacher forward on packed chunks, extract shifted top-K logprobs."""
    K = TEACHER_TOP_K
    all_top_ids = []
    all_top_lps = []
    n = batch_ids.shape[0]

    for i in range(0, n, teacher_micro_batch_size):
        chunk_ids = batch_ids[i:i + teacher_micro_batch_size].to(teacher_device)
        chunk_mask = batch_pad_mask[i:i + teacher_micro_batch_size].to(teacher_device)
        chunk_pos = batch_pos_ids[i:i + teacher_micro_batch_size].to(teacher_device)

        with torch.inference_mode():
            t_out = teacher(input_ids=chunk_ids, attention_mask=chunk_mask,
                            position_ids=chunk_pos)
        # Shift: logits[:, :-1] predicts token at position+1
        t_logits = t_out.logits[:, :-1, :vocab_size].float()
        t_lse = torch.logsumexp(t_logits, dim=-1)
        top_lps_raw, top_ids = t_logits.topk(K, dim=-1)
        top_lps = top_lps_raw - t_lse.unsqueeze(-1)

        all_top_ids.append(top_ids)
        all_top_lps.append(top_lps)
        del t_out, t_logits, t_lse, top_lps_raw

    return torch.cat(all_top_ids), torch.cat(all_top_lps)  # [N, T-1, K] on teacher_device


def main():
    args = parse_args()
    loss_type = args.loss_type

    if loss_type == "cce":
        from cut_cross_entropy import linear_cross_entropy
    from evals import run_evals_hf

    lr = args.lr
    pack_length = args.pack_length
    micro_batch_size = args.micro_batch_size
    teacher_micro_batch_size = args.teacher_micro_batch_size
    total_batch_size = args.total_batch_size
    gpu_offset = args.gpu_offset if args.gpu_offset is not None else (1 if loss_type == "kld" else 0)
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id
    checkpoint_base = args.checkpoint_base or f"checkpoints/sft-{loss_type}"
    resume_from = args.resume_from
    save_every = args.save_every
    milestone_every = args.milestone_every

    # DDP init
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank + gpu_offset}"
    torch.cuda.set_device(device)

    # Round total_batch_size to nearest multiple of micro_batch_size * world_size
    unit = micro_batch_size * world_size
    total_batch = round(total_batch_size / unit) * unit
    assert total_batch > 0, (
        f"total_batch_size ({total_batch_size}) too small for "
        f"micro_batch_size ({micro_batch_size}) * world_size ({world_size}) = {unit}"
    )
    n_micro_batches = total_batch // micro_batch_size
    mbs_per_rank = n_micro_batches // world_size

    if rank == 0:
        if total_batch != total_batch_size:
            print(f"Adjusted batch size: {total_batch_size} -> {total_batch} "
                  f"(nearest multiple of {unit})")
        print(f"SFT mode: {loss_type.upper()}, DDP: {world_size} ranks, "
              f"batch={total_batch} ({mbs_per_rank}x{micro_batch_size}/rank), "
              f"devices cuda:{gpu_offset}..cuda:{gpu_offset + world_size - 1}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Tokenize + pack (all ranks, deterministic). Cache both stages to disk.
    cache_dir = Path("data_cache")
    tokenized_path = cache_dir / f"dolci_sft_tokenized_pl{pack_length}"
    packed_path = cache_dir / f"dolci_sft_packed_pl{pack_length}"
    meta_path = packed_path / "meta.npy"

    # Stage 1: tokenized dataset (saved as HF dataset)
    if tokenized_path.exists():
        if rank == 0:
            print(f"Loading cached tokenized dataset from {tokenized_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(tokenized_path))
    else:
        from datasets import load_dataset
        if rank == 0:
            print(f"Loading dataset {SFT_DATASET}...")
        ds = load_dataset(SFT_DATASET, split="train")
        if rank == 0:
            print(f"Tokenizing {len(ds)} conversations...")
        dataset = tokenize_and_filter(ds, tokenizer, pack_length)
        if rank == 0:
            cache_dir.mkdir(exist_ok=True)
            dataset.save_to_disk(str(tokenized_path))
            print(f"Saved tokenized dataset ({len(dataset)} seqs) to {tokenized_path}")
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"Tokenized: {len(dataset)} sequences (pack_length={pack_length})")

    # Stage 2: packed chunks (saved as memmap files)
    if meta_path.exists():
        if rank == 0:
            print(f"Loading cached packed chunks from {packed_path}")
    else:
        if rank == 0:
            print("Packing sequences (writing memmap)...")
        pack_sequences(dataset, pack_length, PAD_TOKEN_ID, save_path=str(packed_path))
        if rank == 0:
            print(f"Saved packed chunks to {packed_path}")
        if world_size > 1:
            dist.barrier()

    # Load memmap arrays (read-only, near-zero RAM)
    meta = np.load(str(meta_path))
    n_chunks, pl = int(meta[0]), int(meta[1])
    packed_ids = np.memmap(f"{packed_path}/input_ids.npy", dtype=np.int32,
                           mode="r", shape=(n_chunks, pack_length))
    packed_pos = np.memmap(f"{packed_path}/position_ids.npy", dtype=np.int16,
                           mode="r", shape=(n_chunks, pack_length))
    packed_loss_mask = np.memmap(f"{packed_path}/loss_mask.npy", dtype=np.uint8,
                                 mode="r", shape=(n_chunks, pack_length))
    packed_pad_mask = np.memmap(f"{packed_path}/pad_mask.npy", dtype=np.bool_,
                                mode="r", shape=(n_chunks, pack_length))

    if rank == 0:
        total_real = int(packed_pad_mask.sum())
        total_slots = n_chunks * pack_length
        print(f"Packing: {len(dataset)} seqs -> {n_chunks} chunks, "
              f"efficiency {total_real / total_slots:.1%}")

    # Deterministic shuffle via index permutation (don't shuffle the memmap)
    random.seed(1223)
    chunk_order = list(range(n_chunks))
    random.shuffle(chunk_order)

    n_epochs = N_EPOCHS
    steps_per_epoch = n_chunks // total_batch
    total_steps = steps_per_epoch * n_epochs

    if rank == 0:
        print(f"Steps per epoch: {steps_per_epoch}, total: {total_steps}")

    # Student model
    student_src = resume_from if resume_from else STUDENT
    if rank == 0:
        print(f"Loading student from {student_src}...")
    student = AutoLigerKernelForCausalLM.from_pretrained(
        student_src, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    # NOTE: gradient checkpointing is a no-op with Liger kernels (fused ops
    # bypass HF checkpointing hooks). Profiling confirmed identical memory
    # and identical CUDA time with/without. Removed to avoid confusion.
    VOCAB_SIZE = student.config.vocab_size

    # CCE: capture last hidden state before lm_head, replace input with
    # 1-token dummy so the expensive [B*T, H] @ [H, V] matmul is skipped.
    # DDP still sees lm_head.weight as used; real gradients flow through
    # linear_cross_entropy which takes hidden + weight directly.
    _captured_hidden = [None]
    if loss_type == "cce":
        def _capture_pre_lm_head(module, args):
            _captured_hidden[0] = args[0]
            return (args[0][:, :1, :],)
        student.lm_head.register_forward_pre_hook(_capture_pre_lm_head)

    student = DDP(student, device_ids=[rank + gpu_offset],
                  static_graph=True, broadcast_buffers=False)

    optimizer = bnb.optim.AdamW8bit(
        student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8
    )

    # WSD LR schedule
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    decay_steps = int(total_steps * DECAY_FRACTION)
    stable_steps = total_steps - warmup_steps - decay_steps

    def wsd_lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if current_step < warmup_steps + stable_steps:
            return 1.0
        progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lr_lambda)

    # Teacher (KLD only, rank 0 only)
    teacher = None
    teacher_device = None
    if loss_type == "kld" and rank == 0:
        teacher_device = "cuda:0"
        print(f"Loading teacher {TEACHER} on {teacher_device}...")
        teacher = AutoLigerKernelForCausalLM.from_pretrained(
            TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(teacher_device)
        teacher.eval()
        print(f"Student vocab: {VOCAB_SIZE}, Teacher vocab: {teacher.config.vocab_size}")

    # Resume
    start_step = 0
    if resume_from and rank == 0:
        start_step = load_checkpoint(resume_from, student.module, optimizer)
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step}")
    # Broadcast start_step
    start_step_t = torch.tensor([start_step], dtype=torch.long, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = start_step_t.item()
    if start_step > 0 and rank != 0:
        for _ in range(start_step):
            scheduler.step()

    # Wandb (rank 0)
    if rank == 0:
        student_short = STUDENT.split("/")[-1].lower().replace("olmo-2-0425-", "olmo")
        lr_str = f"{lr:.0e}".replace("-0", "-")
        sweep_tag = f"-sweep{sweep_steps}" if sweep_steps else ""
        run_name = (f"{student_short}-sft-{loss_type}-ddp{world_size}"
                    f"-lr{lr_str}-pack{pack_length}{sweep_tag}")
        wandb.init(
            project=WANDB_PROJECT, id=wandb_run_id, name=run_name,
            config={
                "loss_type": loss_type, "teacher": TEACHER, "student": STUDENT,
                "pack_length": pack_length, "total_batch": total_batch,
                "micro_batch_size": micro_batch_size, "ddp_world_size": world_size,
                "steps_per_epoch": steps_per_epoch, "n_epochs": n_epochs,
                "total_steps": total_steps, "lr": lr, "sweep_steps": sweep_steps,
                "max_grad_norm": MAX_GRAD_NORM, "warmup_steps": warmup_steps,
                "teacher_top_k": TEACHER_TOP_K if loss_type == "kld" else None,
            },
            resume="must" if wandb_run_id else "allow",
        )

    # Training loop
    global_step = start_step
    _save_thread = None
    optimizer.zero_grad(set_to_none=True)

    # Micro-batch sharding (n_micro_batches, mbs_per_rank computed above)

    # KLD: teacher stream for async prefetch (teacher on GPU 0, students on GPU 1+)
    teacher_stream = None
    _prefetched_teacher = None  # (top_ids, top_lps) on student device, ready for broadcast
    if loss_type == "kld" and rank == 0:
        teacher_stream = torch.cuda.Stream(device=teacher_device)

    def _load_batch(step_idx):
        """Load packed chunks from memmap for a given optimizer step."""
        start = step_idx * total_batch
        indices = chunk_order[start:start + total_batch]
        return (
            torch.from_numpy(packed_ids[indices].astype(np.int64)),
            torch.from_numpy(packed_pos[indices].astype(np.int64)),
            torch.from_numpy(packed_loss_mask[indices].astype(np.float32)),
            torch.from_numpy(packed_pad_mask[indices].copy()),
        )

    def _teacher_extract_and_stage(batch_ids, batch_pad_mask, batch_pos_ids):
        """Extract teacher top-K and move to rank 0's student device. Runs on teacher_stream."""
        top_ids, top_lps = extract_teacher_topk(
            teacher, batch_ids, batch_pad_mask, batch_pos_ids,
            teacher_device, teacher_micro_batch_size, VOCAB_SIZE,
        )
        # Move from teacher_device (cuda:0) to rank 0's student device
        return top_ids.to(device), top_lps.to(device)

    pbar = tqdm(total=total_steps - start_step, desc=f"SFT ({loss_type.upper()})",
                disable=(rank != 0))

    for epoch in range(n_epochs):
        for opt_step_idx in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + opt_step_idx
            if current_step < start_step:
                continue

            opt_step_start = time.time()

            # Load batch from memmap
            batch_ids, batch_pos_ids, batch_loss_mask, batch_pad_mask = _load_batch(opt_step_idx)

            # === KLD: use prefetched teacher outputs or extract synchronously ===
            if loss_type == "kld":
                if _prefetched_teacher is not None:
                    # Wait for async teacher extraction to finish
                    if rank == 0:
                        teacher_stream.synchronize()
                    top_ids, top_lps = _prefetched_teacher
                    _prefetched_teacher = None
                else:
                    # First step: extract synchronously
                    if rank == 0:
                        top_ids, top_lps = _teacher_extract_and_stage(
                            batch_ids, batch_pad_mask, batch_pos_ids)
                    else:
                        top_ids = torch.empty(total_batch, pack_length - 1, TEACHER_TOP_K,
                                              dtype=torch.long, device=device)
                        top_lps = torch.empty(total_batch, pack_length - 1, TEACHER_TOP_K,
                                              dtype=torch.float32, device=device)
                dist.broadcast(top_ids, src=0)
                dist.broadcast(top_lps, src=0)

                # Prefetch teacher outputs for NEXT step (async, overlaps with student work)
                next_step_idx = opt_step_idx + 1
                if next_step_idx < steps_per_epoch and not (sweep_steps and global_step + 1 >= sweep_steps):
                    next_ids, next_pos, _, next_pad = _load_batch(next_step_idx)
                    if rank == 0:
                        with torch.cuda.stream(teacher_stream):
                            _pf_top_ids, _pf_top_lps = _teacher_extract_and_stage(
                                next_ids, next_pad, next_pos)
                        _prefetched_teacher = (_pf_top_ids, _pf_top_lps)
                    else:
                        # Rank 1+ pre-allocate receive buffers for next broadcast
                        _prefetched_teacher = (
                            torch.empty(total_batch, pack_length - 1, TEACHER_TOP_K,
                                        dtype=torch.long, device=device),
                            torch.empty(total_batch, pack_length - 1, TEACHER_TOP_K,
                                        dtype=torch.float32, device=device),
                        )

            # === Micro-batch loop ===
            accumulated_loss_gpu = torch.zeros(1, device=device)
            total_loss_tokens_gpu = torch.zeros(1, device=device)
            rank_start = rank * mbs_per_rank

            for local_idx in range(mbs_per_rank):
                mb_idx = rank_start + local_idx
                seq_start = mb_idx * micro_batch_size
                seq_end = seq_start + micro_batch_size

                mb_ids = batch_ids[seq_start:seq_end].to(device, non_blocking=True)
                mb_pad_mask = batch_pad_mask[seq_start:seq_end].to(device, non_blocking=True)
                mb_pos_ids = batch_pos_ids[seq_start:seq_end].to(device, non_blocking=True)
                mb_loss_mask = batch_loss_mask[seq_start:seq_end].to(device, non_blocking=True)

                if loss_type == "cce":
                    # Forward through DDP; pre-hook captures hidden state
                    # and feeds 1-token dummy to lm_head (nearly free)
                    student_out = student(
                        input_ids=mb_ids, attention_mask=mb_pad_mask,
                        position_ids=mb_pos_ids,
                    )
                    hidden = _captured_hidden[0]
                    _captured_hidden[0] = None
                    del student_out

                    # CCE handles shift internally, avoids .contiguous() copies
                    per_token_loss = linear_cross_entropy(
                        hidden,
                        student.module.lm_head.weight,
                        mb_ids,
                        shift=True,
                        reduction="none",
                    )  # [B, T-1]

                    combined_mask = mb_loss_mask[:, 1:] * mb_pad_mask[:, 1:].float()
                    masked_loss = (per_token_loss * combined_mask).sum() / combined_mask.sum()

                else:  # kld
                    student_out = student(
                        input_ids=mb_ids, attention_mask=mb_pad_mask,
                        position_ids=mb_pos_ids,
                    )
                    # Shift student logits to align with teacher's shifted top-K
                    s_logits = student_out.logits[:, :-1, :VOCAB_SIZE]
                    mb_top_ids = top_ids[seq_start:seq_end]
                    mb_top_lps = top_lps[seq_start:seq_end]
                    combined_mask = mb_loss_mask[:, 1:] * mb_pad_mask[:, 1:].float()

                    masked_loss = fused_partial_kl(
                        s_logits, mb_top_ids, mb_top_lps, combined_mask,
                    )

                scaled_loss = masked_loss / mbs_per_rank
                ctx = nullcontext() if (local_idx == mbs_per_rank - 1) else student.no_sync()
                with ctx:
                    scaled_loss.backward()
                accumulated_loss_gpu += scaled_loss.detach()
                total_loss_tokens_gpu += (mb_loss_mask[:, 1:] * mb_pad_mask[:, 1:].float()).sum()

            # Optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging (rank 0) — single .item() call instead of per-micro-batch
            if rank == 0:
                opt_step_time = time.time() - opt_step_start
                log_payload = {
                    "train/loss": accumulated_loss_gpu.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/optimizer_step_time_sec": opt_step_time,
                    "train/global_step": global_step,
                    "train/loss_tokens": total_loss_tokens_gpu.item(),
                }
                wandb.log(log_payload)

            global_step += 1
            pbar.update(1)

            # Eval (rank 0)
            eval_every = args.eval_every or EVAL_EVERY_N_STEPS
            if rank == 0 and global_step % eval_every == 0:
                student.eval()
                metrics = run_evals_hf(
                    student.module, tokenizer, STUDENT,
                    tasks=EVAL_TASKS, limit=EVAL_N_SAMPLES,
                )
                wandb.log(metrics)
                student.train()

            # Checkpoint (rank 0, async — snapshot state to CPU, write in background)
            if rank == 0 and global_step % save_every == 0:
                if _save_thread is not None and _save_thread.is_alive():
                    _save_thread.join()
                _snap_state = {k: v.cpu() for k, v in student.module.state_dict().items()}
                _snap_opt = {"optimizer": optimizer.state_dict(), "step": global_step}
                _snap_step = global_step
                def _bg_save(state, opt, step):
                    save_checkpoint(student.module, tokenizer, optimizer,
                                    step, checkpoint_base, milestone_every,
                                    state_dict_cpu=state, opt_state=opt)
                _save_thread = threading.Thread(
                    target=_bg_save, args=(_snap_state, _snap_opt, _snap_step), daemon=True)
                _save_thread.start()

            if sweep_steps and global_step >= sweep_steps:
                if rank == 0:
                    print(f"Sweep: stopping after {sweep_steps} steps")
                break

        if sweep_steps and global_step >= sweep_steps:
            break

    pbar.close()

    # Final save + cleanup
    if rank == 0:
        if _save_thread is not None and _save_thread.is_alive():
            _save_thread.join()
        save_checkpoint(student.module, tokenizer, optimizer,
                        global_step, checkpoint_base, milestone_every)
        wandb.finish()
        print("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
