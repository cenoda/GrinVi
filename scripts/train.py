"""
scripts/train.py — Launch a GrinVi training run.

Quick start (character-level toy data):
    python scripts/train.py --preset tiny --max_steps 1000

Full run on a text file:
    python scripts/train.py --preset small --data data/train.txt --max_steps 100000

Multi-GPU (DDP) via torchrun:
    torchrun --nproc_per_node=2 scripts/train.py --preset small --data data/train.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make sure the repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from grinvi.config import GrinViConfig
from grinvi.model import GrinViModel
from grinvi.tokenizer import GrinViTokenizer
from grinvi.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# Simple text-file dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Streaming text-file dataset — 파일을 줄 단위로 읽어 메모리 효율적으로 처리.
    전체 파일을 메모리에 올리지 않고 토큰 버퍼를 채워가며 윈도우를 반환한다.
    """
    def __init__(self, path: str, tokenizer, seq_len: int = 512,
                 max_lines: int | None = None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.path = path
        self.max_lines = max_lines

        # 토큰 버퍼를 미리 채운다 (최대 50M 토큰 = ~200MB)
        MAX_TOKENS = 50_000_000
        buf: list[int] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                buf.extend(tokenizer.encode(line, add_bos=False, add_eos=False))
                buf.append(tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2)
                if len(buf) >= MAX_TOKENS:
                    break

        n = ((len(buf) - 1) // seq_len) * seq_len
        self.ids = torch.tensor(buf[:n + 1], dtype=torch.long)
        print(f"[Dataset] {len(self.ids):,} tokens → {len(self):,} samples (seq_len={seq_len})")

    def __len__(self):
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start: start + self.seq_len]
        y = self.ids[start + 1: start + self.seq_len + 1]
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# Tiny synthetic dataset for smoke-test
# ---------------------------------------------------------------------------

class RepeatDataset(Dataset):
    """Generates random token sequences — useful for smoke-testing."""
    def __init__(self, vocab_size: int, seq_len: int = 128, n_samples: int = 10_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return ids[:-1], ids[1:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train GrinVi")
    p.add_argument("--preset", choices=["tiny", "small", "medium", "large"], default="tiny",
                   help="Model size preset")
    p.add_argument("--data", default=None, type=str,
                   help="Path to a plain-text training file. If omitted, uses synthetic data.")
    p.add_argument("--eval_data", default=None, type=str,
                   help="Path to a plain-text eval file.")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (faster after warmup)")
    p.add_argument("--grad_ckpt", action="store_true", help="Gradient checkpointing (saves ~35%% VRAM)")
    p.add_argument("--tokenizer", choices=["cl100k_base", "sentencepiece"], default="cl100k_base",
                   help="cl100k_base for English, sentencepiece for Korean/multilingual")
    p.add_argument("--tokenizer_model", default=None, type=str,
                   help="Path to SentencePiece .model file (required if --tokenizer sentencepiece)")
    p.add_argument("--resume", default=None, type=str, help="Path to checkpoint to resume from")
    p.add_argument("--keep_last_n", type=int, default=5,
                   help="보관할 최근 체크포인트 수 (0 = 무제한)")
    # Task 11.4: --scale_lr CLI argument
    p.add_argument("--scale_lr", choices=["linear", "sqrt", "none"], default="none",
                   help="LR scaling mode for multi-GPU: linear, sqrt, or none (default)")
    return p.parse_args()


def main():
    args = parse_args()

    # Task 11.1: Detect DDP mode via environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", -1))
    is_ddp = rank != -1
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if is_ddp else 1
    # DDP mode: use cuda:local_rank; single GPU: use --device or auto-detect
    device = f"cuda:{local_rank}" if is_ddp else (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build config
    config: GrinViConfig = getattr(GrinViConfig, args.preset)()

    # Choose tokenizer
    if args.tokenizer == "cl100k_base":
        tokenizer = GrinViTokenizer()
    elif args.tokenizer == "sentencepiece":
        if not args.tokenizer_model:
            print("[ERROR] --tokenizer sentencepiece requires --tokenizer_model <path_to.model>")
            sys.exit(1)
        from grinvi.tokenizer_sp import GrinViTokenizerSP
        tokenizer = GrinViTokenizerSP(args.tokenizer_model)
    else:
        raise ValueError(f"Unknown tokenizer: {args.tokenizer}")

    config.vocab_size = tokenizer.vocab_size

    # Build or restore model
    # Task 13: All ranks load the same checkpoint — GrinViModel.from_pretrained works without conversion
    if args.resume:
        model = GrinViModel.from_pretrained(args.resume)
    else:
        model = GrinViModel(config)

    if not is_ddp or rank == 0:
        print(f"[GrinVi] {args.preset} model — {model.num_parameters():,} parameters")
        print(f"[GrinVi] Tokenizer: {args.tokenizer}  vocab_size={tokenizer.vocab_size}")

    # Dataset
    if args.data:
        train_ds = TextDataset(args.data, tokenizer, args.seq_len)
        eval_ds  = TextDataset(args.eval_data, tokenizer, args.seq_len) if args.eval_data else None
    else:
        if not is_ddp or rank == 0:
            print("[GrinVi] No --data provided, using synthetic random data for smoke-test.")
        train_ds = RepeatDataset(config.vocab_size, args.seq_len)
        eval_ds  = RepeatDataset(config.vocab_size, args.seq_len, n_samples=200)

    # Task 11.2: Use DistributedSampler in DDP mode
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,          # shuffle must be False when using a sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        if eval_ds else None
    )

    # Task 11.4: Pass world_size and scale_lr to TrainerConfig
    tcfg = TrainerConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dtype=args.dtype,
        compile_model=args.compile,
        gradient_checkpointing=args.grad_ckpt,
        keep_last_n=args.keep_last_n,
        world_size=world_size,
        scale_lr=args.scale_lr,
    )

    trainer = Trainer(model, tcfg, train_loader, eval_loader)

    # Task 11.3: Track epoch for DistributedSampler.set_epoch
    # We patch the train_loader iteration to call set_epoch at each eval interval.
    # The Trainer handles the training loop internally, so we store the sampler
    # on the trainer for potential use. The set_epoch call is done via a wrapper
    # approach: we subclass or monkey-patch the loader's __iter__.
    # Since Trainer iterates train_loader internally, we attach the sampler
    # so the Trainer can call set_epoch. However, the current Trainer design
    # doesn't expose epoch hooks. We implement set_epoch by wrapping the loader.
    if train_sampler is not None:
        _original_iter = train_loader.__class__.__iter__

        # Attach sampler reference to trainer for set_epoch calls
        trainer._train_sampler = train_sampler
        trainer._sampler_epoch = 0

        # Monkey-patch _eval to call set_epoch before each eval interval
        _original_eval = trainer._eval

        def _eval_with_set_epoch(step: int):
            trainer._sampler_epoch += 1
            train_sampler.set_epoch(trainer._sampler_epoch)
            return _original_eval(step)

        trainer._eval = _eval_with_set_epoch

    trainer.train()


if __name__ == "__main__":
    main()
