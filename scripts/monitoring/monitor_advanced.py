#!/usr/bin/env python3
"""
GrinVi Advanced Monitor — Pretty console dashboard with charts
Shows training metrics in a beautiful layout.

Usage:
    python scripts/monitor_advanced.py [--log training.log]
"""
import argparse
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text


console = Console()


# Simple sparkline replacement
def simple_sparkline(values: list, max_value: float = None) -> str:
    """Create a text-based sparkline."""
    if not values:
        return "▁"

    if max_value is None:
        max_value = max(values) if values else 1

    if max_value == 0:
        max_value = 1

    chars = "▁▂▃▄▅▆▇█"
    spark = ""
    for v in values:
        idx = int((v / max_value) * (len(chars) - 1))
        spark += chars[idx]
    return spark


class TrainingMetrics:
    """Track and store training metrics."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.steps = deque(maxlen=max_history)
        self.losses = deque(maxlen=max_history)
        self.lrs = deque(maxlen=max_history)
        self.speeds = deque(maxlen=max_history)
        self.eval_losses = []
        self.eval_ppls = []
        self.checkpoints = []
        self.errors = 0
        self.last_step = 0
        self.start_time = datetime.now()

    def add_step(self, step: int, loss: float, lr: float, speed: int):
        """Add a training step."""
        self.steps.append(step)
        self.losses.append(loss)
        self.lrs.append(lr)
        self.speeds.append(speed)
        self.last_step = step

    def add_eval(self, loss: float, ppl: float):
        """Add eval metrics."""
        self.eval_losses.append(loss)
        self.eval_ppls.append(ppl)

    def add_checkpoint(self, checkpoint: str):
        """Add checkpoint info."""
        self.checkpoints.append(checkpoint)

    def get_stats(self) -> dict:
        """Get current stats."""
        if not self.losses:
            return {}

        return {
            "step": self.steps[-1] if self.steps else 0,
            "loss": self.losses[-1],
            "loss_min": min(self.losses),
            "loss_avg": sum(self.losses) / len(self.losses),
            "lr": self.lrs[-1] if self.lrs else 0,
            "speed": self.speeds[-1] if self.speeds else 0,
            "speed_avg": sum(self.speeds) / len(self.speeds) if self.speeds else 0,
            "eval_loss": self.eval_losses[-1] if self.eval_losses else None,
            "eval_ppl": self.eval_ppls[-1] if self.eval_ppls else None,
            "checkpoints": len(self.checkpoints),
            "errors": self.errors,
            "elapsed": (datetime.now() - self.start_time).total_seconds() / 3600,
        }


def parse_log_file(log_path: str, metrics: TrainingMetrics) -> TrainingMetrics:
    """Parse the training.log file and update metrics."""
    if not Path(log_path).exists():
        return metrics

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            new_lines = f.readlines()

        # Parse lines
        for line in new_lines[-50:]:  # Only parse recent lines
            line = line.strip()
            if not line:
                continue

            # Parse training step
            step_match = re.search(r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\de.-]+)\s*\|\s*([\d,]+)\s*tok/s", line)
            if step_match:
                step = int(step_match.group(1))
                if step > metrics.last_step:  # Only add new steps
                    loss = float(step_match.group(2))
                    lr = float(step_match.group(3))
                    speed = int(step_match.group(4).replace(",", ""))
                    metrics.add_step(step, loss, lr, speed)

            # Parse eval
            eval_match = re.search(r"eval\s+step=(\d+)\s+loss=([\d.]+)\s+ppl=([\d.]+)", line)
            if eval_match:
                loss = float(eval_match.group(2))
                ppl = float(eval_match.group(3))
                metrics.add_eval(loss, ppl)

            # Parse checkpoint
            ckpt_match = re.search(r"Checkpoint saved → (.+)", line)
            if ckpt_match:
                metrics.add_checkpoint(ckpt_match.group(1))

            # Count errors
            if "⚠" in line or "✗" in line:
                metrics.errors += 1

    except Exception as e:
        console.print(f"[red]Error parsing log: {e}[/red]")

    return metrics


def create_dashboard(metrics: TrainingMetrics, max_steps: int = 4000) -> Layout:
    """Create a beautiful dashboard."""

    layout = Layout()
    layout.split_column(
        Layout(name="title", size=3),
        Layout(name="main"),
        Layout(name="bottom", size=2),
    )

    # Title
    stats = metrics.get_stats()
    progress_pct = (stats.get("step", 0) / max_steps * 100) if max_steps > 0 else 0

    title_text = Text()
    title_text.append("🚀 ", style="bold yellow")
    title_text.append("GrinVi Training ", style="bold cyan")
    title_text.append(f"| Step: {stats.get('step', 0):,}/{max_steps:,} ({progress_pct:.1f}%)", style="bold green")

    layout["title"].update(Panel(title_text, style="blue"))

    # Main layout - split into left and right
    layout["main"].split_row(
        Layout(name="metrics"),
        Layout(name="charts"),
    )

    # Metrics panel
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))

    loss = stats.get("loss", 0.0)
    loss_color = "green" if loss < 5 else "yellow" if loss < 7 else "red"

    metrics_table.add_row(
        "[bold cyan]Loss[/bold cyan]",
        f"[{loss_color}]{loss:.4f}[/{loss_color}]",
        f"[dim](min: {stats.get('loss_min', 0):.4f}, avg: {stats.get('loss_avg', 0):.4f})[/dim]"
    )
    metrics_table.add_row(
        "[bold cyan]LR[/bold cyan]",
        f"[magenta]{stats.get('lr', 0):.2e}[/magenta]"
    )
    metrics_table.add_row(
        "[bold cyan]Speed[/bold cyan]",
        f"[yellow]{stats.get('speed', 0):,} tok/s[/yellow]",
        f"[dim](avg: {stats.get('speed_avg', 0):,.0f})[/dim]"
    )

    if stats.get("eval_loss"):
        metrics_table.add_row(
            "[bold cyan]Eval Loss[/bold cyan]",
            f"[magenta]{stats.get('eval_loss'):.4f}[/magenta]"
        )
        metrics_table.add_row(
            "[bold cyan]Perplexity[/bold cyan]",
            f"[magenta]{stats.get('eval_ppl'):.2f}[/magenta]"
        )

    metrics_table.add_row(
        "[bold cyan]Checkpoints[/bold cyan]",
        f"[cyan]{stats.get('checkpoints', 0)}[/cyan]"
    )

    if stats.get("errors", 0) > 0:
        metrics_table.add_row(
            "[bold cyan]Errors[/bold cyan]",
            f"[yellow]{stats.get('errors')}[/yellow]"
        )

    metrics_table.add_row(
        "[bold cyan]Elapsed[/bold cyan]",
        f"[cyan]{stats.get('elapsed', 0):.2f}h[/cyan]"
    )

    layout["metrics"].update(Panel(metrics_table, title="📊 Metrics", style="green"))

    # Charts panel
    charts_table = Table(show_header=False, box=None, padding=(0, 1))

    if metrics.losses:
        loss_spark = simple_sparkline([float(x) for x in list(metrics.losses)[-20:]], max_value=max(metrics.losses) * 1.1 or 10)
        charts_table.add_row(
            "[bold cyan]Loss Trend[/bold cyan]",
            loss_spark
        )

    if metrics.speeds:
        speed_spark = simple_sparkline([float(x) for x in list(metrics.speeds)[-20:]], max_value=max(metrics.speeds) * 1.1 or 1)
        charts_table.add_row(
            "[bold cyan]Speed Trend[/bold cyan]",
            speed_spark
        )

    layout["charts"].update(Panel(charts_table, title="📈 Trends", style="cyan"))

    # Bottom status
    status_text = Text()
    status_text.append(f"Last updated: ", style="dim")
    status_text.append(datetime.now().strftime("%H:%M:%S"), style="cyan")
    if metrics.checkpoints:
        status_text.append(f"  |  Latest: {metrics.checkpoints[-1].split('/')[-1]}", style="dim yellow")

    layout["bottom"].update(Panel(status_text, style="dim white"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Advanced GrinVi monitor")
    parser.add_argument("--log", default="training.log", help="Path to training log")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max steps")
    args = parser.parse_args()

    metrics = TrainingMetrics()

    try:
        with Live(
            create_dashboard(metrics, args.max_steps),
            refresh_per_second=0.5,
            console=console,
        ) as live:
            while True:
                metrics = parse_log_file(args.log, metrics)
                dashboard = create_dashboard(metrics, args.max_steps)
                live.update(dashboard)
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]✓ Monitor stopped.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()

