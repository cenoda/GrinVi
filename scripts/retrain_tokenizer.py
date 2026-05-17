"""
scripts/retrain_tokenizer.py — Retrain the Korean morph tokenizer in place.

Uses an existing train.txt corpus (no re-download needed).

Example:
    python scripts/retrain_tokenizer.py \
        --data data/raw/ko_wikipedia/train.txt \
        --output_prefix data/raw/ko_wikipedia/ko_tokenizer \
        --vocab_size 80000

After retraining, the resulting ko_tokenizer.json has built-in character
fallback so encoding any Korean text produces ~0% UNK.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grinvi.tokenizer_morph import GrinViMorphTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to train.txt (one document per line)")
    ap.add_argument("--output_prefix", required=True,
                    help="Output prefix; writes <prefix>.json and <prefix>.vocab")
    ap.add_argument("--vocab_size", type=int, default=80000)
    ap.add_argument("--extra_char_top_n", type=int, default=4000,
                    help="How many non-Hangul chars (hanja, latin, etc.) to include")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Parallel worker count (default: all CPUs)")
    ap.add_argument("--chunk_size", type=int, default=10000,
                    help="Lines per work unit (larger = less overhead, default 10000)")
    args = ap.parse_args()

    GrinViMorphTokenizer.train(
        args.data,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        extra_char_top_n=args.extra_char_top_n,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    print(f"[OK] Tokenizer written to {args.output_prefix}.json")


if __name__ == "__main__":
    main()

