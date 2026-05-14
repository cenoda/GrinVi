#!/usr/bin/env python3
"""
GrinVi Training Monitor — Live CLI dashboard
Shows real-time metrics while training runs.

Usage:
    python scripts/monitor.py [--log training.log] [--refresh 1]
"""
import argparse
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text


console = Console()


def parse_log_file(log_path: str) -> dict:
    """Parse the training.log file and extract latest metrics."""
    if not Path(log_path).exists():
        return {"error": f"Log file not found: {log_path}"}

    metrics = {
        "step": 0,
        "loss": 0.0,
        "lr": 0.0,
        "tok_per_sec": 0,
        "eval_loss": None,
        "eval_ppl": None,
        "checkpoint_saved": False,
        "last_update": None,
        "errors": 0,
    }

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse from last lines (reverse order for efficiency)
        for line in reversed(lines[-100:]):
            line = line.strip()
            if not line:
                continue

            # Parse training step: "step 500 | loss 7.2013 | lr 7.50e-05 | 453,986 tok/s"
            step_match = re.search(r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\de.-]+)\s*\|\s*([\d,]+)\s*tok/s", line)
            if step_match:
                metrics["step"] = int(step_match.group(1))
                metrics["loss"] = float(step_match.group(2))
                metrics["lr"] = float(step_match.group(3))
                metrics["tok_per_sec"] = int(step_match.group(4).replace(",", ""))

            # Parse eval: "eval  step=500  loss=7.1338  ppl=1253.63"
            eval_match = re.search(r"eval\s+step=(\d+)\s+loss=([\d.]+)\s+ppl=([\d.]+)", line)
            if eval_match:
                metrics["eval_loss"] = float(eval_match.group(2))
                metrics["eval_ppl"] = float(eval_match.group(3))

            # Parse checkpoint: "Checkpoint saved → checkpoints/step-500"
            if "Checkpoint saved" in line:
                metrics["checkpoint_saved"] = True

            # Count errors
            if "⚠" in line or "✗" in line:
                metrics["errors"] += 1

        metrics["last_update"] = datetime.now()
    except Exception as e:
        metrics["error"] = str(e)

    return metrics


def create_dashboard(metrics: dict, max_steps: int = 4000) -> Layout:
    """Create a Rich layout dashboard."""

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="stats"),
        Layout(name="progress"),
        Layout(name="footer", size=3),
    )

    # Header
    title = Text("🚀 GrinVi Training Monitor", style="bold cyan")
    layout["header"].update(Panel(title, style="blue"))

    # Stats table
    stats_table = Table(show_header=False, box=None, padding=(0, 2))

    step = metrics.get("step", 0)
    loss = metrics.get("loss", 0.0)
    lr = metrics.get("lr", 0.0)
    tok_per_sec = metrics.get("tok_per_sec", 0)
    eval_loss = metrics.get("eval_loss")
    eval_ppl = metrics.get("eval_ppl")
    errors = metrics.get("errors", 0)

    # Color based on values
    loss_color = "green" if loss < 5 else "yellow" if loss < 7 else "red"
    lr_color = "cyan"

    stats_table.add_row(
        f"[bold cyan]Step[/bold cyan]",
        f"[{loss_color}]{step:,} / {max_steps:,}[/{loss_color}]"
    )
    stats_table.add_row(
        f"[bold cyan]Loss[/bold cyan]",
        f"[{loss_color}]{loss:.4f}[/{loss_color}]"
    )
    stats_table.add_row(
        f"[bold cyan]Learning Rate[/bold cyan]",
        f"[{lr_color}]{lr:.2e}[/{lr_color}]"
    )
    stats_table.add_row(
        f"[bold cyan]Speed[/bold cyan]",
        f"[yellow]{tok_per_sec:,} tokens/sec[/yellow]"
    )

    if eval_loss is not None:
        stats_table.add_row(
            f"[bold cyan]Eval Loss[/bold cyan]",
            f"[magenta]{eval_loss:.4f}[/magenta]"
        )
    if eval_ppl is not None:
        stats_table.add_row(
            f"[bold cyan]Eval Perplexity[/bold cyan]",
            f"[magenta]{eval_ppl:.2f}[/magenta]"
        )

    if errors > 0:
        stats_table.add_row(
            f"[bold cyan]Errors[/bold cyan]",
            f"[yellow]{errors}[/yellow]"
        )

    layout["stats"].update(Panel(stats_table, title="📊 Metrics", style="green"))

    # Progress bar
    progress_pct = (step / max_steps) * 100 if max_steps > 0 else 0

    if step > 0 and tok_per_sec > 0:
        # Estimate time remaining
        tokens_remaining = (max_steps - step) * 512 * 8 / 4  # seq_len * batch_size / grad_accum
        time_remaining_sec = tokens_remaining / tok_per_sec
        time_remaining = timedelta(seconds=int(time_remaining_sec))
        time_str = f"ETA: {time_remaining}"
    else:
        time_str = "ETA: calculating..."

    progress_layout = Layout()
    progress_layout.split_row(
        Layout(Text(f"{progress_pct:.1f}%", style="bold cyan", justify="right"), size=6),
        Layout(),
    )

    layout["progress"].update(Panel(
        progress_layout,
        title=f"⏱️  Progress {time_str}",
        style="magenta"
    ))

    # Footer with timestamp
    update_time = metrics.get("last_update")
    if update_time:
        time_text = f"Last update: {update_time.strftime('%H:%M:%S')}"
    else:
        time_text = "Waiting for data..."

    layout["footer"].update(Panel(time_text, style="dim white"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Monitor GrinVi training")
    parser.add_argument("--log", default="training.log", help="Path to training log file")
    parser.add_argument("--refresh", type=float, default=2, help="Refresh interval in seconds")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max training steps")
    args = parser.parse_args()

    try:
        with Live(
            create_dashboard({"error": "Initializing..."}, args.max_steps),
            refresh_per_second=1/args.refresh,
            console=console,
        ) as live:
            while True:
                metrics = parse_log_file(args.log)
                dashboard = create_dashboard(metrics, args.max_steps)
                live.update(dashboard)
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()

