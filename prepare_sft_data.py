"""Precompute tokenized + packed data for sft_ddp.py. No GPU needed.

Usage:
  uv run python prepare_sft_data.py                    # default pack_length=2048
  uv run python prepare_sft_data.py --pack-length 4096

Saves to data_cache/ (same paths sft_ddp.py looks for).
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from distill_utils import SFT_DATASET, TEACHER, pack_sequences
from sft_ddp import tokenize_and_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-length", type=int, default=2048)
    args = parser.parse_args()
    pack_length = args.pack_length

    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    tokenized_path = cache_dir / f"dolci_sft_tokenized_pl{pack_length}"
    packed_path = cache_dir / f"dolci_sft_packed_pl{pack_length}"

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Stage 1: tokenize
    if tokenized_path.exists():
        print(f"Tokenized cache exists: {tokenized_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(tokenized_path))
    else:
        print(f"Loading {SFT_DATASET}...")
        ds = load_dataset(SFT_DATASET, split="train")
        print(f"Tokenizing {len(ds)} conversations (num_proc=24)...")
        dataset = tokenize_and_filter(ds, tokenizer, pack_length)
        dataset.save_to_disk(str(tokenized_path))
        print(f"Saved {len(dataset)} tokenized sequences to {tokenized_path}")

    print(f"Tokenized: {len(dataset)} sequences")

    # Stage 2: pack (writes memmap files directly to disk)
    meta_path = Path(f"{packed_path}/meta.npy")
    if meta_path.exists():
        meta = np.load(str(meta_path))
        n_chunks, pl = int(meta[0]), int(meta[1])
        print(f"Packed cache exists: {packed_path} ({n_chunks} chunks)")
    else:
        print("Packing sequences (first-fit-decreasing, writing to memmap)...")
        pack_sequences(dataset, pack_length, PAD_TOKEN_ID, save_path=str(packed_path))
        meta = np.load(str(meta_path))
        n_chunks, pl = int(meta[0]), int(meta[1])
        print(f"Saved {n_chunks} packed chunks to {packed_path}")

    # Stats
    pad_mask = np.memmap(f"{packed_path}/pad_mask.npy", dtype=np.bool_,
                         mode="r", shape=(n_chunks, pack_length))
    total_real = int(pad_mask.sum())
    total_slots = n_chunks * pack_length
    avg_seqs = len(dataset) / n_chunks
    print(f"\nPacking stats:")
    print(f"  {len(dataset)} seqs -> {n_chunks} chunks")
    print(f"  avg {avg_seqs:.1f} seqs/chunk")
    print(f"  efficiency {total_real / total_slots:.1%}")
    print(f"\nDone! sft_ddp.py will load these caches automatically.")


if __name__ == "__main__":
    main()
