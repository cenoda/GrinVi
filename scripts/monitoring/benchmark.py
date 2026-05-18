"""
scripts/benchmark.py — Measure tokens/sec on your GPU for each model preset.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --preset medium --batch_size 4 --seq_len 1024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from rich.console import Console
from rich.table import Table

from grinvi.config import GrinViConfig
from grinvi.model import GrinViModel

console = Console()


def benchmark(preset: str, batch_size: int, seq_len: int, dtype_str: str,
              grad_ckpt: bool, n_iters: int = 20, device: str = "cuda"):
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    cfg = getattr(GrinViConfig, preset)()
    # Use config's default vocab size (which is now 80,000)
    model = GrinViModel(cfg).to(device).to(dtype)
    if grad_ckpt:
        model.enable_gradient_checkpointing()

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    labels    = input_ids.clone()

    # Warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_iter = batch_size * seq_len
    total_tokens    = tokens_per_iter * n_iters
    tok_per_sec     = total_tokens / elapsed
    vram_used       = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    params = model.num_parameters()
    return {
        "preset":      preset,
        "params":      f"{params/1e6:.0f} M",
        "batch×seq":   f"{batch_size}×{seq_len}",
        "dtype":       dtype_str,
        "grad_ckpt":   "✓" if grad_ckpt else "✗",
        "tok/s":       f"{tok_per_sec:,.0f}",
        "VRAM (GB)":   f"{vram_used:.2f}",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default=None, help="Run only this preset (tiny/small/medium/large)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--n_iters", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"\n[bold cyan]GrinVi GPU Benchmark[/bold cyan]  —  {torch.cuda.get_device_name(0)}\n")

    runs = []

    if args.preset:
        presets = [args.preset]
    else:
        presets = ["tiny", "small", "medium", "large"]

    batch_defaults = {"tiny": 32, "small": 16, "medium": 6, "large": 4}

    for preset in presets:
        bs = args.batch_size or batch_defaults[preset]
        for grad_ckpt in ([args.grad_ckpt] if args.preset else [False, True]):
            try:
                console.print(f"  Benchmarking [bold]{preset}[/bold]  (batch={bs}, seq={args.seq_len}, grad_ckpt={grad_ckpt}) …", end="")
                r = benchmark(preset, bs, args.seq_len, args.dtype, grad_ckpt,
                               n_iters=args.n_iters, device=device)
                runs.append(r)
                console.print(f"  {r['tok/s']} tok/s  —  {r['VRAM (GB)']} GB VRAM")
            except torch.cuda.OutOfMemoryError:
                console.print("  [red]OOM — skip[/red]")
                torch.cuda.empty_cache()

    if not runs:
        return

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    for col in runs[0].keys():
        table.add_column(col, justify="right" if col not in ("preset", "dtype") else "left")
    for r in runs:
        table.add_row(*r.values())

    console.print()
    console.print(table)


if __name__ == "__main__":
    main()

