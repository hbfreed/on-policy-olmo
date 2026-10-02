"""Patch pre-extracted top-K teacher logprobs to always include EOS.

Where EOS is missing from top-K, replaces the lowest-ranked (last) entry
with EOS and a conservative logprob estimate (min of existing top-K).

Modifies memmaps in-place. Safe to interrupt and re-run (idempotent).

Usage:
  uv run python patch_eos_topk.py [--pack-length 2048]
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

TEACHER = "allenai/Olmo-3-7B-Instruct"
TEACHER_TOP_K = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-length", type=int, default=2048)
    parser.add_argument("--output-base", type=str, default="/media/henry/MoreFiles")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Chunks to process at a time (RAM usage)")
    args = parser.parse_args()

    K = TEACHER_TOP_K
    path = Path(f"{args.output_base}/teacher_logprobs_pl{args.pack_length}")
    meta = np.load(str(path / "teacher_meta.npy"), allow_pickle=True)
    n_chunks, pack_length = int(meta[0]), int(meta[1])
    pl = pack_length - 1  # shifted positions

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    eos_id = tokenizer.eos_token_id
    print(f"EOS token ID: {eos_id}")
    print(f"Patching {path}: {n_chunks} chunks, {pl} positions, K={K}")

    ids = np.memmap(f"{path}/teacher_top_ids.npy", dtype=np.int32,
                    mode="r+", shape=(n_chunks, pl, K))
    lps = np.memmap(f"{path}/teacher_top_lps.npy", dtype=np.float16,
                    mode="r+", shape=(n_chunks, pl, K))

    # Back up the last rank (only slot we overwrite) for reversibility
    backup_ids_path = path / "backup_last_ids.npy"
    backup_lps_path = path / "backup_last_lps.npy"
    if not backup_ids_path.exists():
        print("Backing up rank K-1 slot (~7GB)...")
        np.save(str(backup_ids_path), np.array(ids[:, :, -1]))
        np.save(str(backup_lps_path), np.array(lps[:, :, -1]))
        print(f"Saved backups to {backup_ids_path} and {backup_lps_path}")
    else:
        print("Backups already exist, skipping.")

    patched = 0
    total = 0

    for start in tqdm(range(0, n_chunks, args.chunk_size), desc="Patching"):
        end = min(start + args.chunk_size, n_chunks)
        chunk_ids = np.array(ids[start:end])    # (cs, pl, K)
        chunk_lps = np.array(lps[start:end])

        # Find positions where EOS is missing from top-K
        has_eos = (chunk_ids == eos_id).any(axis=2)  # (cs, pl)
        missing = ~has_eos
        n_missing = missing.sum()
        total += has_eos.size
        patched += n_missing

        if n_missing > 0:
            # Replace last entry (rank K-1) with EOS
            chunk_ids[missing, -1] = eos_id
            # Use min logprob in that position's top-K as conservative estimate
            chunk_lps[missing, -1] = chunk_lps[missing].min(axis=-1)

            ids[start:end] = chunk_ids
            lps[start:end] = chunk_lps

    ids.flush()
    lps.flush()
    print(f"Done. Patched {patched}/{total} positions ({patched/total:.1%})")
    print(f"Already had EOS: {total - patched}/{total} ({(total-patched)/total:.1%})")


if __name__ == "__main__":
    main()
