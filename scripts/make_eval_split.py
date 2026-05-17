"""
scripts/make_eval_split.py — build a deterministic train/eval split for large text corpora.

Designed for GrinVi's chat-style datasets:
- `질문:` / `답변:` blocks are kept together in chat mode
- falls back to line mode for plain text corpora
- deterministic hashing avoids loading the full file into memory
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

QUESTION_MARKERS = ("질문:", "<usr>")
ANSWER_MARKERS = ("답변:", "<bot>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a deterministic train/eval split")
    p.add_argument("--input", required=True, help="Source text file")
    p.add_argument("--train-output", required=True, help="Output training file")
    p.add_argument("--eval-output", required=True, help="Output eval file")
    p.add_argument("--eval-ratio", type=float, default=0.01, help="Fraction of blocks for eval")
    p.add_argument("--seed", type=int, default=1337, help="Deterministic split seed")
    p.add_argument(
        "--mode",
        choices=["auto", "chat", "line"],
        default="auto",
        help="Split by chat blocks or by individual lines",
    )
    p.add_argument("--probe-lines", type=int, default=2000, help="Non-empty lines to inspect in auto mode")
    p.add_argument("--dry-run", action="store_true", help="Analyze and report without writing output files")
    return p.parse_args()


def is_question_line(text: str) -> bool:
    return text.startswith(QUESTION_MARKERS)


def is_answer_line(text: str) -> bool:
    return text.startswith(ANSWER_MARKERS)


def detect_mode(path: Path, probe_lines: int) -> str:
    nonempty = 0
    questions = 0
    answers = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            nonempty += 1
            if is_question_line(text):
                questions += 1
            if is_answer_line(text):
                answers += 1
            if nonempty >= probe_lines:
                break

    if questions >= 5 and answers >= 5:
        return "chat"
    return "line"


def iter_blocks(path: Path, mode: str) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        if mode == "line":
            for raw in f:
                if raw.strip():
                    yield [raw]
            return

        current: list[str] = []
        for raw in f:
            stripped = raw.strip()

            if not stripped:
                if current:
                    current.append(raw)
                continue

            if is_question_line(stripped) and current:
                yield current
                current = [raw]
            else:
                current.append(raw)

        if current:
            yield current


def should_go_to_eval(block_index: int, block: Iterable[str], ratio: float, seed: int) -> bool:
    first_nonempty = next((line.strip() for line in block if line.strip()), "")
    payload = f"{seed}|{block_index}|{first_nonempty}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / 2**64
    return value < ratio


def write_split(
    input_path: Path,
    train_output: Path,
    eval_output: Path,
    mode: str,
    eval_ratio: float,
    seed: int,
    dry_run: bool,
) -> dict[str, int | str]:
    train_blocks = eval_blocks = 0
    train_lines = eval_lines = 0
    total_blocks = total_lines = 0

    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    train_handle = None
    eval_handle = None
    try:
        if not dry_run:
            train_handle = train_output.open("w", encoding="utf-8")
            eval_handle = eval_output.open("w", encoding="utf-8")

        for block_index, block in enumerate(iter_blocks(input_path, mode)):
            total_blocks += 1
            total_lines += len(block)
            to_eval = should_go_to_eval(block_index, block, eval_ratio, seed)

            if to_eval:
                eval_blocks += 1
                eval_lines += len(block)
                if eval_handle is not None:
                    eval_handle.writelines(block)
            else:
                train_blocks += 1
                train_lines += len(block)
                if train_handle is not None:
                    train_handle.writelines(block)
    finally:
        if train_handle is not None:
            train_handle.close()
        if eval_handle is not None:
            eval_handle.close()

    return {
        "mode": mode,
        "total_blocks": total_blocks,
        "total_lines": total_lines,
        "train_blocks": train_blocks,
        "train_lines": train_lines,
        "eval_blocks": eval_blocks,
        "eval_lines": eval_lines,
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.eval_ratio < 1.0):
        raise SystemExit("--eval-ratio must be between 0 and 1")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    mode = args.mode if args.mode != "auto" else detect_mode(input_path, args.probe_lines)
    stats = write_split(
        input_path=input_path,
        train_output=Path(args.train_output),
        eval_output=Path(args.eval_output),
        mode=mode,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print("[make_eval_split] complete")
    print(f"  input       : {input_path}")
    print(f"  mode        : {stats['mode']}")
    print(f"  total blocks: {stats['total_blocks']:,}")
    print(f"  total lines : {stats['total_lines']:,}")
    print(f"  train blocks: {stats['train_blocks']:,}")
    print(f"  eval blocks : {stats['eval_blocks']:,}")
    print(f"  train lines : {stats['train_lines']:,}")
    print(f"  eval lines  : {stats['eval_lines']:,}")
    if args.dry_run:
        print("  dry-run     : yes (no files written)")
    else:
        print(f"  train output: {args.train_output}")
        print(f"  eval output : {args.eval_output}")


if __name__ == "__main__":
    main()
