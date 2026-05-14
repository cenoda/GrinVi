"""
scripts/download_korean_data.py — Automatically download & combine Korean training data

Usage:
    python scripts/download_korean_data.py --sources wikipedia,nsmc,huggingface --out data/korean_combined.txt
    python scripts/download_korean_data.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


def download_wikipedia(out_dir: Path) -> Generator[str, None, None]:
    """Download Korean Wikipedia."""
    console.print("[bold cyan]📚 Downloading Korean Wikipedia...[/bold cyan]")
    try:
        from grinvi.tokenizer_sp import GrinViTokenizerSP
        import subprocess

        # Use the existing data download mechanism
        result = subprocess.run(
            ["python", "scripts/prepare_data.py", "--dataset", "ko_wikipedia", "--out", str(out_dir)],
            capture_output=False
        )

        if result.returncode == 0:
            wiki_file = out_dir / "ko_wikipedia" / "train.txt"
            if wiki_file.exists():
                console.print(f"✓ Wikipedia ready at {wiki_file}")
                with open(wiki_file, encoding='utf-8') as f:
                    for line in f:
                        yield line.strip()
    except Exception as e:
        console.print(f"[red]✗ Failed to download Wikipedia: {e}[/red]")


def download_nsmc(out_dir: Path) -> Generator[str, None, None]:
    """Download NSMC (Naver Sentiment Movie Corpus)."""
    console.print("[bold cyan]🎬 Downloading NSMC (Naver Movie Reviews)...[/bold cyan]")
    try:
        import subprocess
        import tarfile
        import urllib.request

        out_dir.mkdir(parents=True, exist_ok=True)
        nsmc_path = out_dir / "nsmc"

        if not nsmc_path.exists():
            # Clone from GitHub
            console.print("  Cloning NSMC repository...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/e9t/nsmc.git"],
                cwd=str(out_dir),
                capture_output=True
            )

        if nsmc_path.exists():
            rating_file = nsmc_path / "ratings_train.txt"
            if rating_file.exists():
                console.print(f"✓ NSMC ready at {rating_file}")
                with open(rating_file, encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            yield parts[1]  # Extract review text
    except Exception as e:
        console.print(f"[red]✗ Failed to download NSMC: {e}[/red]")


def download_huggingface(out_dir: Path, dataset_name: str = "koreanlanguageprocessing/korean-corpus") -> Generator[str, None, None]:
    """Download from HuggingFace Datasets."""
    console.print(f"[bold cyan]🤗 Downloading from HuggingFace: {dataset_name}[/bold cyan]")
    try:
        from datasets import load_dataset

        console.print(f"  Loading {dataset_name}...")
        ds = load_dataset(dataset_name, trust_remote_code=True)

        # Try different column names
        text_col = None
        for col in ["text", "content", "review", "comment"]:
            if col in ds["train"].column_names:
                text_col = col
                break

        if text_col:
            console.print(f"✓ Using text column: {text_col}")
            for ex in ds["train"]:
                text = ex[text_col]
                if isinstance(text, str) and text.strip():
                    yield text
        else:
            console.print(f"[yellow]⚠ No text column found in dataset[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed to download from HuggingFace: {e}[/red]")


def download_korean_news(out_dir: Path) -> Generator[str, None, None]:
    """Download Korean news dataset."""
    console.print("[bold cyan]📰 Downloading Korean News Corpus...[/bold cyan]")
    try:
        from datasets import load_dataset

        console.print("  Loading 'wnut_17_korean'...")
        ds = load_dataset("wnut_17", "korean", trust_remote_code=True)

        for ex in ds["train"]:
            if "tokens" in ex:
                text = " ".join(ex["tokens"])
                if text.strip():
                    yield text
    except Exception as e:
        console.print(f"[yellow]⚠ Korean news not available: {e}[/yellow]")


def combine_sources(
    sources: list[str],
    out_dir: Path,
    output_file: Path,
    max_lines: int | None = None,
) -> int:
    """Combine data from multiple sources."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = 0

    console.print(f"\n[bold]Combining sources: {', '.join(sources)}[/bold]\n")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Combining data...", total=None)

            for source in sources:
                if source == "wikipedia":
                    gen = download_wikipedia(out_dir)
                elif source == "nsmc":
                    gen = download_nsmc(out_dir)
                elif source == "huggingface":
                    gen = download_huggingface(out_dir)
                elif source == "news":
                    gen = download_korean_news(out_dir)
                else:
                    console.print(f"[yellow]⚠ Unknown source: {source}[/yellow]")
                    continue

                for line in gen:
                    if line.strip():
                        out_f.write(line + "\n")
                        total_lines += 1
                        progress.update(task, advance=1)

                    if max_lines and total_lines >= max_lines:
                        break

                if max_lines and total_lines >= max_lines:
                    break

    return total_lines


def parse_args():
    p = argparse.ArgumentParser(
        description="Download and combine Korean training data from multiple sources"
    )
    p.add_argument(
        "--sources",
        default="wikipedia,nsmc",
        help="Comma-separated list of sources: wikipedia, nsmc, huggingface, news (default: wikipedia,nsmc)"
    )
    p.add_argument("--out", default="data", help="Output directory (default: data)")
    p.add_argument("--output", default="korean_combined.txt", help="Output filename (default: korean_combined.txt)")
    p.add_argument("--max_lines", type=int, default=None, help="Maximum lines to download per source")
    return p.parse_args()


def main():
    args = parse_args()
    sources = [s.strip() for s in args.sources.split(",")]
    out_dir = Path(args.out)
    output_file = out_dir / args.output

    console.rule("[bold cyan]Korean Data Downloader[/bold cyan]")
    console.print(f"Sources: {', '.join(sources)}")
    console.print(f"Output: {output_file}\n")

    total = combine_sources(sources, out_dir, output_file, args.max_lines)

    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"  Lines downloaded: {total:,}")
    console.print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")
    console.print(f"  Location: {output_file}")

    console.print(f"\n[bold]Next: Train your Korean tokenizer[/bold]")
    console.print(f"  python scripts/train_tokenizer.py --data {output_file} --output korean_tok")

    console.print(f"\n[bold]Then: Train your model[/bold]")
    console.print(f"  python scripts/train.py --preset medium \\")
    console.print(f"      --tokenizer sentencepiece \\")
    console.print(f"      --tokenizer_model korean_tok.model \\")
    console.print(f"      --data {output_file}")


if __name__ == "__main__":
    main()

