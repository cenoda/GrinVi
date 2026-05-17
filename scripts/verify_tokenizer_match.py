"""
scripts/verify_tokenizer_match.py — After retraining the tokenizer locally,
this script tests whether the new tokenizer matches what the server's
checkpoint was trained with.

Strategy:
  1. Load model from checkpoint
  2. Encode a known prompt with the new tokenizer
  3. Pass through model — if top-k predictions are coherent Korean tokens
     (not chaotic noise), the tokenizer matches. If the IDs are scrambled
     relative to the model's learned embeddings, top-k will be junk.

Run after retrain completes:
    python scripts/verify_tokenizer_match.py \
        --checkpoint checkpoints/v2_137m_80k/step-final \
        --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer_80k.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from grinvi.model import GrinViModel
from grinvi.tokenizer_morph import GrinViMorphTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 60)
    print("Tokenizer ↔ Model match verification")
    print("=" * 60)

    tok = GrinViMorphTokenizer(args.tokenizer_model)
    print(f"Tokenizer loaded: vocab_size={tok.vocab_size:,}")

    model = GrinViModel.from_pretrained(args.checkpoint, device=args.device)
    model.eval()
    cfg = getattr(model, "config", None)
    m_vocab = getattr(cfg, "vocab_size", None) if cfg else None
    print(f"Model loaded: vocab_size={m_vocab}")

    if m_vocab != tok.vocab_size:
        print("🚨 vocab_size mismatch — inference impossible.")
        return 2

    # Run a few common prompts and inspect top-10 predictions.
    prompts = [
        "질문: 안녕하세요\n답변:",
        "질문: 한국에서 가장 높은 산은 무엇인가요?\n답변:",
        "질문: 라면을 끓이는 방법을 알려주세요.\n답변:",
        "질문: 인공지능이란 무엇인가요?\n답변:",
    ]

    print()
    coherent_count = 0
    for prompt in prompts:
        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        with torch.inference_mode():
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            logits, _ = model(x)
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            top = torch.topk(probs, 10)

        pieces = [tok.id_to_token[i] if 0 <= i < tok.vocab_size else "?"
                  for i in top.indices.tolist()]
        probs_top = top.values.tolist()

        # Heuristic: a healthy match has top-1 prob > 0.05 and most top-10
        # pieces are Korean / sensible. A mismatch produces ~uniform-ish
        # noise with top-1 << 0.01.
        is_healthy = probs_top[0] > 0.02
        # Count Korean pieces in top-10 (rough indicator)
        ko_count = sum(1 for p in pieces
                       if any(0xAC00 <= ord(c) <= 0xD7A3 for c in str(p)))

        marker = "✅" if is_healthy and ko_count >= 3 else "⚠️"
        if is_healthy and ko_count >= 3:
            coherent_count += 1
        print(f"{marker} {prompt!r}")
        print(f"   top-1 prob = {probs_top[0]:.4f}, Korean pieces in top-10 = {ko_count}")
        print(f"   top-3 pieces: {pieces[:3]}")
        print()

    print("=" * 60)
    if coherent_count >= 3:
        print(f"✅ Coherent on {coherent_count}/{len(prompts)} prompts — "
              "tokenizer matches the model.")
        return 0
    elif coherent_count >= 1:
        print(f"⚠️  Coherent on only {coherent_count}/{len(prompts)} — "
              "PARTIAL match. Inference may produce mixed results.")
        return 1
    else:
        print("🚨 No prompts show coherent predictions — "
              "tokenizer LIKELY MISMATCH. Inference will produce garbage.")
        return 2


if __name__ == "__main__":
    sys.exit(main())

