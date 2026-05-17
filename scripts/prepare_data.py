"""
scripts/prepare_data.py — Download and tokenize real training datasets.

Available datasets
------------------
  tinystories   ~500 MB, simple English stories
  openwebtext   ~40 GB, Reddit-curated web text (GPT-2 training data)
  wikipedia     ~20 GB, English Wikipedia
  ko_wikipedia  ~6 GB, Korean Wikipedia
  naver_news    ~2.5 GB, Korean news articles (requires manual download from aihub)

Usage
-----
  # Fast first run (TinyStories — English):
  python scripts/prepare_data.py --dataset tinystories --out data/

    # Korean data:
    python scripts/prepare_data.py --dataset ko_wikipedia --out data/raw/

  # Or use custom Korean text file you have:
  echo "안녕하세요 반갑습니다" > data/korean_sample.txt
  python scripts/train_tokenizer.py --data data/korean_sample.txt --output kr_tok
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from grinvi.tokenizer import GrinViTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize_and_save(
    texts,
    tokenizer: GrinViTokenizer,
    out_path: Path,
    split_name: str,
    chunk_size: int = 2048,
    max_tokens: int | None = None,
):
    """Tokenize an iterable of strings and write a flat .bin of uint32 token ids."""
    out_path.mkdir(parents=True, exist_ok=True)
    bin_file = out_path / f"{split_name}.bin"
    txt_file = out_path / f"{split_name}.txt"

    all_ids = []
    total = 0

    with open(txt_file, "w", encoding="utf-8") as tf:
        for text in tqdm(texts, desc=f"Tokenising {split_name}"):
            ids = tokenizer.encode(text, add_bos=False, add_eos=True)
            all_ids.extend(ids)
            tf.write(text + "\n")
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break

    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(str(bin_file))
    print(f"  [{split_name}] {total:,} tokens → {bin_file}  ({bin_file.stat().st_size / 1e6:.1f} MB)")
    return bin_file


def tokenize_and_save_sp(
    texts,
    tokenizer,  # GrinViTokenizerSP
    out_path: Path,
    split_name: str,
    chunk_size: int = 2048,
    max_tokens: int | None = None,
):
    """Tokenize with SentencePiece tokenizer and save."""
    out_path.mkdir(parents=True, exist_ok=True)
    bin_file = out_path / f"{split_name}.bin"
    txt_file = out_path / f"{split_name}.txt"

    all_ids = []
    total = 0

    with open(txt_file, "w", encoding="utf-8") as tf:
        for text in tqdm(texts, desc=f"Tokenising {split_name}"):
            ids = tokenizer.encode(text, add_bos=False, add_eos=True)
            all_ids.extend(ids)
            tf.write(text + "\n")
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break

    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(str(bin_file))
    print(f"  [{split_name}] {total:,} tokens → {bin_file}  ({bin_file.stat().st_size / 1e6:.1f} MB)")
    return bin_file


def write_text_corpus(texts, out_path: Path, split_name: str, max_items: int | None = None):
    """Write raw text lines to a .txt corpus file."""
    out_path.mkdir(parents=True, exist_ok=True)
    txt_file = out_path / f"{split_name}.txt"
    total = 0
    with open(txt_file, "w", encoding="utf-8") as f:
        for text in tqdm(texts, desc=f"Writing {split_name}"):
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            f.write(text + "\n")
            total += 1
            if max_items and total >= max_items:
                break
    print(f"  [{split_name}] {total:,} lines → {txt_file}  ({txt_file.stat().st_size / 1e6:.1f} MB)")
    return txt_file


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def prepare_tinystories(out_path: Path, max_tokens: int | None = None):
    print("[GrinVi] Downloading TinyStories …")
    ds = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
    tok = GrinViTokenizer()

    tokenize_and_save(
        (ex["text"] for ex in ds["train"]),
        tok, out_path, "train", max_tokens=max_tokens,
    )
    tokenize_and_save(
        (ex["text"] for ex in ds["validation"]),
        tok, out_path, "val", max_tokens=max_tokens // 20 if max_tokens else None,
    )


def prepare_openwebtext(out_path: Path, max_tokens: int | None = None):
    print("[GrinVi] Downloading OpenWebText (this may take a while) …")
    ds = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
    tok = GrinViTokenizer()

    # 95/5 train/val split
    split = ds.train_test_split(test_size=0.05, seed=42)
    tokenize_and_save(
        (ex["text"] for ex in split["train"]),
        tok, out_path, "train", max_tokens=max_tokens,
    )
    tokenize_and_save(
        (ex["text"] for ex in split["test"]),
        tok, out_path, "val",
    )


def prepare_wikipedia(out_path: Path, max_tokens: int | None = None):
    print("[GrinVi] Downloading Wikipedia (20231101.en) …")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", trust_remote_code=True)
    tok = GrinViTokenizer()

    split = ds.train_test_split(test_size=0.02, seed=42)
    tokenize_and_save(
        (ex["text"] for ex in split["train"]),
        tok, out_path, "train", max_tokens=max_tokens,
    )
    tokenize_and_save(
        (ex["text"] for ex in split["test"]),
        tok, out_path, "val",
    )


def prepare_ko_wikipedia(out_path: Path, max_tokens: int | None = None):
    """Korean Wikipedia — 한국어 위키백과"""
    print("[GrinVi] Downloading Korean Wikipedia …")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
    from grinvi.tokenizer_morph import GrinViMorphTokenizer

    split = ds.train_test_split(test_size=0.02, seed=42)
    train_texts = split["train"]["text"]
    val_texts = split["test"]["text"]

    train_txt = write_text_corpus(train_texts, out_path, "train", max_items=max_tokens)
    write_text_corpus(val_texts, out_path, "val", max_items=max_tokens // 20 if max_tokens else None)

    GrinViMorphTokenizer.train(
        str(train_txt),
        output_prefix=str(out_path / "ko_tokenizer"),
        vocab_size=80000,
    )

    print(f"[GrinVi] Korean tokenizer ready: {out_path / 'ko_tokenizer.json'}")


# ---------------------------------------------------------------------------
# BinDataset — efficient dataset over the pre-tokenized .bin files
# ---------------------------------------------------------------------------
# (This can be used directly in train.py if you pass --data_bin)

class BinDataset:
    """
    Memory-mapped dataset over a pre-tokenized .bin file.
    Much faster and lower-memory than re-tokenizing each epoch.

    Example use alongside train.py:
        from scripts.prepare_data import BinDataset
        train_ds = BinDataset("data/train.bin", seq_len=512)
    """
    def __init__(self, path: str, seq_len: int = 512):
        import torch
        from torch.utils.data import Dataset

        self.seq_len = seq_len
        data = np.fromfile(path, dtype=np.uint32).astype(np.int64)
        self.data = torch.from_numpy(data)
        self.n = (len(self.data) - 1) // seq_len
        print(f"[BinDataset] {len(self.data):,} tokens → {self.n:,} samples (seq_len={seq_len})")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import torch
        start = idx * self.seq_len
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download and tokenize datasets for GrinVi")
    p.add_argument("--dataset", choices=["tinystories", "openwebtext", "wikipedia", "ko_wikipedia"],
                   default="tinystories")
    p.add_argument("--out", default="data", help="Output directory")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Cap the number of tokens (useful for quick experiments)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out) / args.dataset

    if args.dataset == "tinystories":
        prepare_tinystories(out, args.max_tokens)
    elif args.dataset == "openwebtext":
        prepare_openwebtext(out, args.max_tokens)
    elif args.dataset == "wikipedia":
        prepare_wikipedia(out, args.max_tokens)
    elif args.dataset == "ko_wikipedia":
        prepare_ko_wikipedia(out, args.max_tokens)

    print(f"\n[GrinVi] Done! Training files are in: {out}")
    print(f"  Train with:")
    print(f"    python scripts/train.py --preset small --data {out}/train.txt")
    if args.dataset == "ko_wikipedia":
        print(f"\n  For Korean training, also use a Korean tokenizer:")
        print(f"    python scripts/train.py \\")
        print(f"        --preset small \\")
        print(f"        --tokenizer morph \\")
        print(f"        --tokenizer_model {out}/ko_tokenizer.json \\")
        print(f"        --data {out}/train.txt")


if __name__ == "__main__":
    main()

