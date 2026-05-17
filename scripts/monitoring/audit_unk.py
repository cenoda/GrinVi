"""
scripts/audit_unk.py — Measure UNK ratio across many lines of training data.

Example:
    python scripts/audit_unk.py \
        --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
        --data data/raw/ko_wikipedia/train.txt \
        --max_lines 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from grinvi.tokenizer_morph import GrinViMorphTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_lines", type=int, default=20000)
    args = ap.parse_args()

    tok = GrinViMorphTokenizer(args.tokenizer_model)
    print(f"vocab_size = {tok.vocab_size:,}")

    total_tokens = 0
    total_unk = 0
    examples_with_unk: list = []

    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line, add_bos=False, add_eos=False)
            n_unk = sum(1 for tid in ids if tid == tok.unk_token_id)
            total_tokens += len(ids)
            total_unk += n_unk
            if n_unk > 0 and len(examples_with_unk) < 5:
                examples_with_unk.append((n_unk, len(ids), line[:200]))

    ratio = total_unk / max(total_tokens, 1)
    print(f"\nScanned: {min(i + 1, args.max_lines):,} lines")
    print(f"Total tokens : {total_tokens:,}")
    print(f"UNK tokens   : {total_unk:,}")
    print(f"UNK ratio    : {ratio:.4%}")
    if ratio < 0.001:
        print("  >> ✅ EXCELLENT (< 0.1%)")
    elif ratio < 0.01:
        print("  >> ✅ GOOD (< 1%)")
    elif ratio < 0.05:
        print("  >> ⚠️  MARGINAL (1-5%)")
    else:
        print("  >> 🚨 BAD (≥ 5%) — tokenizer is dropping real content")

    if examples_with_unk:
        print("\nFirst lines that still produced UNK (should be ~empty after fix):")
        for n_unk, n_total, sample in examples_with_unk:
            print(f"  [{n_unk}/{n_total}] {sample!r}")


if __name__ == "__main__":
    main()

