"""
scripts/preflight_train.py — quick safety checks before launching a costly training run.

Examples
--------
python scripts/preflight_train.py \
    --data data/processed/train.txt \
    --tokenizer morph \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json

python scripts/preflight_train.py \
    --data data/processed/train.txt \
    --tokenizer morph \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
    --resume checkpoints_medium/step-18000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from grinvi.tokenizer import GrinViTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Preflight checks for GrinVi training runs")
    p.add_argument("--data", required=True, help="Training text file")
    p.add_argument("--eval_data", default=None, help="Optional eval text file")
    p.add_argument("--tokenizer", choices=["cl100k_base", "sentencepiece", "morph"], default="morph")
    p.add_argument("--tokenizer_model", default=None, help="Tokenizer model path for sentencepiece/morph")
    p.add_argument("--resume", default=None, help="Checkpoint directory to resume from")
    p.add_argument("--sample_lines", type=int, default=128, help="Number of lines to sample per file position")
    return p.parse_args()


def build_tokenizer(args):
    tokenizer_model = args.tokenizer_model
    if not tokenizer_model and args.resume:
        resume_dir = Path(args.resume)
        if (resume_dir / "tokenizer.json").exists():
            tokenizer_model = str(resume_dir / "tokenizer.json")
        elif (resume_dir / "tokenizer.model").exists():
            tokenizer_model = str(resume_dir / "tokenizer.model")

    if args.tokenizer == "cl100k_base":
        return GrinViTokenizer()

    if not tokenizer_model:
        raise SystemExit(f"--tokenizer_model is required when --tokenizer {args.tokenizer} (and not found in resume dir)")

    if args.tokenizer == "sentencepiece":
        from grinvi.tokenizer_sp import GrinViTokenizerSP
        return GrinViTokenizerSP(tokenizer_model)

    from grinvi.tokenizer_morph import GrinViMorphTokenizer
    return GrinViMorphTokenizer(tokenizer_model)


def sample_lines(path: Path, lines_per_probe: int) -> list[str]:
    size = path.stat().st_size
    probes = [0.0, 0.25, 0.5, 0.75, 0.95]
    out: list[str] = []
    with open(path, "rb") as f:
        for frac in probes:
            pos = int(size * frac)
            f.seek(pos)
            if pos > 0:
                f.readline()  # drop partial line
            for _ in range(lines_per_probe):
                line = f.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    out.append(text)
    return out


def count_contains(lines: Iterable[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def count_unk_ids(tokenizer, text: str) -> int:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        return 0
    return tokenizer.encode(text, add_bos=False, add_eos=False).count(unk_id)


def check_resume(resume_dir: Path, errors: list[str], notes: list[str]):
    if not resume_dir.exists():
        errors.append(f"resume checkpoint does not exist: {resume_dir}")
        return

    required = [resume_dir / "config.json", resume_dir / "model.safetensors"]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        errors.append(f"resume checkpoint is missing required files: {', '.join(missing)}")

    trainer_state = resume_dir / "trainer_state.pt"
    if not trainer_state.exists():
        errors.append(
            "resume checkpoint has no trainer_state.pt — this is not a true optimizer/scheduler resume and may spike loss"
        )
    else:
        notes.append(f"resume checkpoint includes trainer_state.pt: {trainer_state}")


def main():
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"training file not found: {data_path}")

    if args.eval_data:
        eval_path = Path(args.eval_data)
        if not eval_path.exists():
            errors.append(f"eval file not found: {eval_path}")
        else:
            notes.append(f"eval file found: {eval_path} ({eval_path.stat().st_size:,} bytes)")
    else:
        warnings.append("no --eval_data provided; you will not be able to detect overfitting reliably")

    tokenizer = build_tokenizer(args)
    notes.append(f"training file: {data_path} ({data_path.stat().st_size:,} bytes)")
    notes.append(f"tokenizer: {args.tokenizer}")
    if args.tokenizer_model:
        notes.append(f"tokenizer model: {args.tokenizer_model}")

    lines = sample_lines(data_path, args.sample_lines)
    if not lines:
        errors.append("failed to sample non-empty lines from training file")
    else:
        legacy_usr = count_contains(lines, "<usr>")
        legacy_bot = count_contains(lines, "<bot>")
        q_count = count_contains(lines, "질문:")
        a_count = count_contains(lines, "답변:")
        notes.append(
            f"sample markers — <usr>: {legacy_usr}, <bot>: {legacy_bot}, 질문:: {q_count}, 답변:: {a_count}"
        )

        legacy_unk = count_unk_ids(tokenizer, "<usr>") + count_unk_ids(tokenizer, "<bot>")
        qa_unk = count_unk_ids(tokenizer, "질문:") + count_unk_ids(tokenizer, "답변:")
        notes.append(f"tokenizer probe — legacy tag unk count: {legacy_unk}, 질문/답변 unk count: {qa_unk}")

        if (legacy_usr or legacy_bot) and legacy_unk > 0:
            errors.append(
                "sampled data still contains <usr>/<bot>, and the current tokenizer maps them to unknown tokens"
            )
        if (q_count + a_count) == 0 and (legacy_usr + legacy_bot) == 0:
            warnings.append("sampled data does not show obvious chat markers; verify your dataset format intentionally")
        if qa_unk > 0:
            warnings.append("질문:/답변: still produce UNK tokens with the selected tokenizer")

    if args.resume:
        check_resume(Path(args.resume), errors, notes)

    print("[preflight] notes")
    for item in notes:
        print(f"  - {item}")

    if warnings:
        print("[preflight] warnings")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        print("[preflight] errors")
        for item in errors:
            print(f"  - {item}")
        print("[preflight] FAILED")
        raise SystemExit(1)

    print("[preflight] OK")


if __name__ == "__main__":
    main()


