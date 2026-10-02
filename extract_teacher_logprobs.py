"""Multi-GPU teacher logprob extraction for pre-computed KLD targets.

Extracts top-K teacher logprobs from packed SFT data and saves to memmap files.
Each GPU loads the teacher and processes a contiguous shard of chunks in parallel.

Usage:
  torchrun --nproc_per_node=3 extract_teacher_logprobs.py [--pack-length 2048] [--teacher-mbs 8]
"""

import argparse

import numpy as np
import torch
import torch.distributed as dist
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from pathlib import Path
from tqdm import tqdm
from transformers import AutoConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# Config
# ============================================================
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B-Instruct"
TEACHER_TOP_K = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Extract teacher top-K logprobs to disk")
    parser.add_argument("--pack-length", type=int, default=2048)
    parser.add_argument("--teacher-mbs", type=int, default=8,
                        help="Teacher micro-batch size (chunks per forward pass)")
    parser.add_argument("--output-base", type=str, default="/media/henry/MoreFiles",
                        help="Base path for output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed chunk")
    return parser.parse_args()


def main():
    args = parse_args()
    pack_length = args.pack_length
    teacher_mbs = args.teacher_mbs
    K = TEACHER_TOP_K

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Load packed data metadata
    packed_path = Path(f"data_cache/dolci_sft_packed_pl{pack_length}")
    meta = np.load(str(packed_path / "meta.npy"))
    n_chunks = int(meta[0])

    # Load packed input data (read-only memmaps)
    packed_ids = np.memmap(f"{packed_path}/input_ids.npy", dtype=np.int32,
                           mode="r", shape=(n_chunks, pack_length))
    packed_pos = np.memmap(f"{packed_path}/position_ids.npy", dtype=np.int16,
                           mode="r", shape=(n_chunks, pack_length))
    packed_pad_mask = np.memmap(f"{packed_path}/pad_mask.npy", dtype=np.bool_,
                                mode="r", shape=(n_chunks, pack_length))

    # Student vocab size (without loading weights)
    vocab_size = AutoConfig.from_pretrained(STUDENT).vocab_size

    # Output path
    output_path = Path(f"{args.output_base}/teacher_logprobs_pl{pack_length}")

    # Rank 0 creates output memmaps + progress file
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        meta_file = output_path / "teacher_meta.npy"
        if not args.resume or not meta_file.exists():
            np.memmap(f"{output_path}/teacher_top_ids.npy", dtype=np.int32,
                      mode="w+", shape=(n_chunks, pack_length - 1, K))
            np.memmap(f"{output_path}/teacher_top_lps.npy", dtype=np.float16,
                      mode="w+", shape=(n_chunks, pack_length - 1, K))
            np.save(str(meta_file), np.array([n_chunks, pack_length, K]))
            np.memmap(f"{output_path}/progress.npy", dtype=np.uint8,
                      mode="w+", shape=(n_chunks,))
            print(f"Created output memmaps at {output_path}")
            print(f"  teacher_top_ids: ({n_chunks}, {pack_length-1}, {K}) int32")
            print(f"  teacher_top_lps: ({n_chunks}, {pack_length-1}, {K}) float16")
        else:
            print(f"Resuming extraction at {output_path}")

    dist.barrier()

    # All ranks open memmaps in r+ mode
    top_ids_mm = np.memmap(f"{output_path}/teacher_top_ids.npy", dtype=np.int32,
                           mode="r+", shape=(n_chunks, pack_length - 1, K))
    top_lps_mm = np.memmap(f"{output_path}/teacher_top_lps.npy", dtype=np.float16,
                           mode="r+", shape=(n_chunks, pack_length - 1, K))
    progress_mm = np.memmap(f"{output_path}/progress.npy", dtype=np.uint8,
                            mode="r+", shape=(n_chunks,))

    # Contiguous shard for this rank
    chunks_per_rank = n_chunks // world_size
    rank_start = rank * chunks_per_rank
    rank_end = rank_start + chunks_per_rank
    if rank == world_size - 1:
        rank_end = n_chunks  # last rank gets remainder

    # Find resume point
    start_chunk = rank_start
    if args.resume:
        for i in range(rank_start, rank_end):
            if progress_mm[i] == 0:
                start_chunk = i
                break
        else:
            start_chunk = rank_end  # all done
        done = int(progress_mm[rank_start:rank_end].sum())
        if rank == 0:
            print(f"Rank {rank}: {done}/{rank_end - rank_start} already done, "
                  f"resuming from chunk {start_chunk}")

    n_remaining = rank_end - start_chunk
    if rank == 0:
        print(f"Rank {rank}: chunks [{rank_start}, {rank_end}), "
              f"{n_remaining} to process")
        print(f"Loading teacher {TEACHER} on {device}...")

    # Load teacher on this rank's GPU
    teacher = AutoLigerKernelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    teacher.eval()

    if rank == 0:
        print(f"Teacher loaded. Student vocab: {vocab_size}, "
              f"Teacher vocab: {teacher.config.vocab_size}")
        print(f"Starting extraction (teacher_mbs={teacher_mbs})...")

    # Process chunks
    pbar = tqdm(total=n_remaining, desc=f"Rank {rank}", disable=(rank != 0),
                unit="chunk")
    flush_every = 100

    for batch_start in range(start_chunk, rank_end, teacher_mbs):
        batch_end = min(batch_start + teacher_mbs, rank_end)
        bs = batch_end - batch_start

        ids = torch.from_numpy(packed_ids[batch_start:batch_end]).to(
            device, dtype=torch.long, non_blocking=True)
        pad_mask = torch.from_numpy(packed_pad_mask[batch_start:batch_end]).to(
            device, non_blocking=True)
        pos = torch.from_numpy(packed_pos[batch_start:batch_end]).to(
            device, dtype=torch.long, non_blocking=True)

        with torch.inference_mode():
            t_out = teacher(input_ids=ids, attention_mask=pad_mask, position_ids=pos)
            t_logits = t_out.logits[:, :-1, :vocab_size]
            del t_out

            top_lps_raw, top_ids_batch = t_logits.topk(K, dim=-1)

            # In-place bf16 logsumexp (avoids fp32 materialization OOM)
            max_val = t_logits.max(dim=-1, keepdim=True).values
            t_logits -= max_val
            t_logits.exp_()
            t_lse = max_val.squeeze(-1).float() + t_logits.sum(dim=-1).float().log()
            del t_logits, max_val

            top_lps_batch = top_lps_raw.float() - t_lse.unsqueeze(-1)
            del t_lse, top_lps_raw

        # Write to memmaps (int32 ids, float16 logprobs)
        top_ids_mm[batch_start:batch_end] = top_ids_batch.cpu().numpy().astype(np.int32)
        top_lps_mm[batch_start:batch_end] = top_lps_batch.cpu().numpy().astype(np.float16)
        progress_mm[batch_start:batch_end] = 1
        del top_ids_batch, top_lps_batch

        if (batch_end - start_chunk) % flush_every < teacher_mbs:
            top_ids_mm.flush()
            top_lps_mm.flush()
            progress_mm.flush()

        pbar.update(bs)

    # Final flush
    top_ids_mm.flush()
    top_lps_mm.flush()
    progress_mm.flush()
    pbar.close()

    if rank == 0:
        total_done = int(progress_mm.sum())
        print(f"Rank 0 done. Progress so far: {total_done}/{n_chunks} chunks")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
