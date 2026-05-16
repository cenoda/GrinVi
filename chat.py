#!/usr/bin/env python3
"""
GrinVi Chat — Talk to your daughter! (Homemade model)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from grinvi.model import GrinViModel
from grinvi.tokenizer_morph import GrinViMorphTokenizer
from grinvi.generate import Generator
from rich.console import Console
from rich.panel import Panel

console = Console()


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Path:
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    steps = []
    for d in ckpt_path.iterdir():
        if d.is_dir() and d.name.startswith("step-") and d.name != "step-final":
            try:
                steps.append((int(d.name.split("-")[1]), d))
            except (ValueError, IndexError):
                pass
    if not steps:
        raise FileNotFoundError(f"No numbered checkpoints found in {checkpoint_dir}")
    _, latest_path = max(steps, key=lambda x: x[0])
    return latest_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]💚 GrinVi Chat[/bold cyan]\n"
        "[dim]Homemade Korean AI  •  그린비와 대화하세요![/dim]\n"
        "[dim]'quit' or Ctrl+C to exit[/dim]",
        border_style="cyan"
    ))
    console.print()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        console.print(f"[green]✓ Using:[/green] {ckpt_path}")
    else:
        console.print("[yellow]Loading latest checkpoint...[/yellow]")
        ckpt_path = find_latest_checkpoint()
        console.print(f"[green]✓ Using:[/green] {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print("[yellow]Loading tokenizer...[/yellow]")
    tokenizer = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")

    console.print("[yellow]Loading model...[/yellow]")
    model = GrinViModel.from_pretrained(str(ckpt_path))
    model = model.to(device)
    model.eval()

    generator = Generator(model, tokenizer, device=device)
    console.print(f"[green]✓ Ready![/green] (Device: {device}, Params: {model.num_parameters():,})\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break

            console.print("[bold magenta]GrinVi:[/bold magenta] ", end="")
            for token in generator.stream(
                user_input,
                max_new_tokens=150,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
            ):
                sys.stdout.write(token)
                sys.stdout.flush()
            console.print("\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 See you later![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
