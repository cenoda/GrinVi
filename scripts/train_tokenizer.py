"""
scripts/train_tokenizer.py — Train a custom SentencePiece tokenizer for Korean/CJK languages.

Examples:

  # Train on a single Korean text file:
  python scripts/train_tokenizer.py \
      --data data/korean_corpus.txt \
      --output grinvi_korean \
    --vocab_size 64000

  # Train and verify:
  python scripts/train_tokenizer.py \
      --data data/korean.txt \
      --output my_korean_tok \
      --test "안녕하세요"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grinvi.tokenizer_sp import GrinViTokenizerSP


def parse_args():
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for GrinVi")
    p.add_argument("--data", required=True, help="Path to training text file")
    p.add_argument("--output", default="grinvi_tok", help="Output model prefix (will create .model and .vocab)")
    p.add_argument("--vocab_size", type=int, default=64000, help="Vocabulary size")
    p.add_argument("--model_type", choices=["bpe", "unigram", "char", "word"], default="bpe")
    p.add_argument("--character_coverage", type=float, default=0.9995,
                   help="Character coverage (0.9995 recommended for CJK)")
    p.add_argument("--test", default=None, help="Test string to tokenize after training")
    return p.parse_args()


def main():
    args = parse_args()

    # Train tokenizer on the data file
    tok = GrinViTokenizerSP.train(
        args.data,
        output_prefix=args.output,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )

    print(f"\nTokenizer stats:")
    print(f"  Vocab size: {tok.vocab_size}")

    # Test if requested
    if args.test:
        ids = tok.encode(args.test, add_bos=False, add_eos=False)
        decoded = tok.decode(ids)
        print(f"\nTest encode:")
        print(f"  Input:   {args.test}")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded:  {decoded}")

    print(f"\n✓ Tokenizer ready to use!")
    print(f"  Use in training with:")
    print(f"    python scripts/train.py \\")
    print(f"        --preset small \\")
    print(f"        --tokenizer sentencepiece \\")
    print(f"        --tokenizer_model {args.output}.model \\")
    print(f"        --data data/korean_train.txt")


if __name__ == "__main__":
    main()

