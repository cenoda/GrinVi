"""
scripts/generate.py — Generate text with a trained GrinVi checkpoint.

Examples
--------
Interactive REPL:
    python scripts/generate.py --checkpoint checkpoints/step-final

Single prompt:
    python scripts/generate.py --checkpoint checkpoints/step-final \
        --prompt "Once upon a time" --max_new_tokens 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from grinvi.model import GrinViModel
from grinvi.tokenizer import GrinViTokenizer
from grinvi.generate import Generator


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with GrinVi")
    p.add_argument("--checkpoint", required=True, help="Path to saved checkpoint directory")
    p.add_argument("--prompt", default=None, type=str, help="Prompt string (omit for interactive mode)")
    p.add_argument(
        "--tokenizer",
        choices=["cl100k_base", "sentencepiece", "morph"],
        default="cl100k_base",
        help="Tokenizer used for the checkpoint (use morph for Korean checkpoints)",
    )
    p.add_argument(
        "--tokenizer_model",
        default=None,
        help="Path to the tokenizer model file (required when --tokenizer sentencepiece or morph)",
    )
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding (overrides temperature/top-k/top-p)")
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    p.add_argument("--min_new_tokens", type=int, default=0,
                   help="Suppress EOS until at least this many new tokens are generated")
    p.add_argument("--ban_special", action="store_true",
                   help="Ban PAD/BOS/UNK during generation (EOS still allowed unless --min_new_tokens)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[GrinVi] Loading model from '{args.checkpoint}' on {device} …")
    model = GrinViModel.from_pretrained(args.checkpoint, device=device)
    if args.tokenizer == "sentencepiece":
        if not args.tokenizer_model:
            raise SystemExit("--tokenizer_model is required when --tokenizer sentencepiece")
        from grinvi.tokenizer_sp import GrinViTokenizerSP

        tokenizer = GrinViTokenizerSP(args.tokenizer_model)
    elif args.tokenizer == "morph":
        if not args.tokenizer_model:
            raise SystemExit("--tokenizer_model is required when --tokenizer morph")
        from grinvi.tokenizer_morph import GrinViMorphTokenizer

        tokenizer = GrinViMorphTokenizer(args.tokenizer_model)
    else:
        tokenizer = GrinViTokenizer()
    gen = Generator(model, tokenizer, device=device)

    def run_prompt(prompt: str):
        if args.stream:
            print(prompt, end="", flush=True)
            for tok in gen.stream(prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p):
                print(tok, end="", flush=True)
            print()
        else:
            ban = None
            if args.ban_special:
                ban = []
                for attr in ("pad_token_id", "bos_token_id", "unk_token_id"):
                    tid = getattr(tokenizer, attr, None)
                    if tid is not None:
                        ban.append(tid)
            output = gen.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=None if args.greedy else args.top_k,
                top_p=None if args.greedy else args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=not args.greedy,
                min_new_tokens=args.min_new_tokens,
                ban_tokens=ban,
            )
            print(output)

    if args.prompt:
        run_prompt(args.prompt)
    else:
        # Interactive REPL
        print("[GrinVi] Interactive mode — type a prompt and press Enter. Ctrl-C to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if not prompt.strip():
                    continue
                run_prompt(prompt)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\n[GrinVi] Bye!")
                break


if __name__ == "__main__":
    main()

