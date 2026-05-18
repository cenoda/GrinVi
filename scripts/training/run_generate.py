"""
scripts/training/run_generate.py — CLI entry point for GrinVi text generation.

Wraps grinvi.generate.Generator with argparse for easy CLI use.

Usage:
    python scripts/training/run_generate.py \
        --checkpoint checkpoints/v2_137m_80k/step-final \
        --tokenizer morph \
        --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer_80k.json \
        --prompt $'질문: 안녕하세요\\n답변:' \
        --max_new_tokens 80 --min_new_tokens 30 --ban_special \
        --temperature 0.8 --top_p 0.9 --repetition_penalty 1.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from grinvi.generate import Generator
from grinvi.model import GrinViModel
from grinvi.tokenizer_morph import GrinViMorphTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="morph", choices=["morph"])
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_sample", action="store_true", help="Greedy decoding")
    ap.add_argument("--ban_special", action="store_true",
                    help="Ban PAD/BOS/UNK from sampling")
    ap.add_argument("--ban_tokens", type=int, nargs="*", default=None,
                    help="Additional token IDs to ban")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = GrinViMorphTokenizer(args.tokenizer_model)
    model = GrinViModel.from_pretrained(args.checkpoint, device=args.device)
    model.eval()

    cfg = getattr(model, "config", None)
    m_vocab = getattr(cfg, "vocab_size", None) if cfg else None
    if m_vocab != tok.vocab_size:
        print(f"🚨 vocab mismatch: model={m_vocab}, tokenizer={tok.vocab_size}",
              file=sys.stderr)
        return 2

    ban = list(args.ban_tokens or [])
    if args.ban_special:
        for name in ("pad_token_id", "bos_token_id", "unk_token_id"):
            tid = getattr(tok, name, None)
            if isinstance(tid, int) and tid >= 0:
                ban.append(tid)

    gen = Generator(model=model, tokenizer=tok, device=args.device)
    output = gen.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        ban_tokens=ban or None,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

