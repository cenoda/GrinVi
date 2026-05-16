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

from torch.utils.data import IterableDataset

class TextDataset(IterableDataset):
    """
    Streaming text-file dataset — 25GB 같은 초대용량 텍스트를 실시간으로 읽기 위해 IterableDataset으로 변경합니다.
    """
    def __init__(self, path: str, tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.path = path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 멀티프로세싱 워커가 여러개일 경우 각자 다른 파일 위치부터 읽게 처리하면 좋지만,
        # 여기서는 단순화를 위해 워커가 하나라고 가정하거나 랜덤하게 넘기도록 처리
        buf = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            if worker_info is not None:
                # 워커마다 다른 부분을 읽도록 파일을 분할합니다.
                # 총 라인수를 알기 어려우므로 바이트 오프셋 기준으로 크게 뜁니다.
                # 예: 최대 25GB 파일이면 워커/노드별로 충분히 멀리 띄워줍니다.
                import os
                file_size = os.path.getsize(self.path)
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                
                # 시작 오프셋 계산
                chunk_size = file_size // num_workers
                start_offset = worker_id * chunk_size
                
                if start_offset > 0:
                    f.seek(start_offset)
                    # 첫 줄은 잘렸을 확률이 높으니 버립니다.
                    f.readline()
                
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buf.extend(self.tokenizer.encode(line, add_bos=False, add_eos=False))
                buf.append(self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2)

                while len(buf) > self.seq_len * 4: # 약간 여유있게 버퍼링
                    x = torch.tensor(buf[:self.seq_len], dtype=torch.long)
                    y = torch.tensor(buf[1:self.seq_len+1], dtype=torch.long)
                    buf = buf[self.seq_len:]
                    yield x, y

    def __getstate__(self):
        state = self.__dict__.copy()
        if "tokenizer" in state:
            del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 형태소 토크나이저 복원
        from grinvi.tokenizer_morph import GrinViMorphTokenizer
        self.tokenizer = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")


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
    p.add_argument("--tokenizer", choices=["cl100k_base", "sentencepiece", "morph"], default="cl100k_base",
                   help="cl100k_base for English, sentencepiece or morph for Korean/multilingual")
    p.add_argument("--tokenizer_model", default=None, type=str,
                   help="Path to tokenizer model file (required if --tokenizer sentencepiece or morph)")
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
    elif args.tokenizer == "morph":
        if not args.tokenizer_model:
            print("[ERROR] --tokenizer morph requires --tokenizer_model <path_to.json>")
            sys.exit(1)
        from grinvi.tokenizer_morph import GrinViMorphTokenizer
        tokenizer = GrinViMorphTokenizer(args.tokenizer_model)
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
        # IterableDataset은 DistributedSampler를 직접 물릴 수 없습니다.
        train_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=16,
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
