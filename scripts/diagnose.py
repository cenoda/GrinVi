"""
scripts/diagnose.py — 학습이 정상인지 토크나이저/모델을 직접 까서 검증.

사용 예:
    python scripts/diagnose.py \
        --checkpoint checkpoints/step-4000 \
        --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
        --prompt $'질문: 어디인가요?\n답변:'

검증 항목:
  [1] 토크나이저 vocab 크기/특수토큰
  [2] 프롬프트 인코드 → 토큰 ID/조각/UNK 비율
  [3] 디코드 라운드트립 (원문 보존 여부)
  [4] 학습데이터(train.txt) 첫 줄 인코드 시 UNK 비율 (이게 높으면 토크나이저 자체가 문제)
  [5] 모델 top-10 다음 토큰 예측 (모델이 의미있는 분포를 학습했는지)
  [6] 모델 vocab_size vs 토크나이저 vocab_size 일치 여부 (제일 흔한 사일런트 버그)
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


def hr(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--prompt", default="질문: 어디인가요?\n답변:")
    ap.add_argument("--train_txt", default="data/raw/ko_wikipedia/train.txt",
                    help="학습 텍스트 첫 줄 인코드 검증용 (없으면 스킵)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    hr("[1] 토크나이저 로드")
    tok = GrinViMorphTokenizer(args.tokenizer_model)
    print(f"vocab_size        = {tok.vocab_size:,}")
    print(f"PAD/BOS/EOS/UNK   = {tok.pad_token_id} / {tok.bos_token_id} / "
          f"{tok.eos_token_id} / {tok.unk_token_id}")
    print(f"include_pos       = {tok.include_pos}")
    # 샘플 vocab 일부
    print(f"vocab[0:6]        = {tok.id_to_token[0:6]}")
    print(f"vocab[1000:1006]  = {tok.id_to_token[1000:1006]}")

    # ------------------------------------------------------------------
    hr("[2] 프롬프트 인코드")
    print(f"PROMPT (repr): {args.prompt!r}")
    pieces = tok._tokenize_to_pieces(args.prompt)
    ids = tok.encode(args.prompt, add_bos=True, add_eos=False)
    n_unk = sum(1 for p in pieces if p not in tok.token_to_id)
    unk_pieces = [p for p in pieces if p not in tok.token_to_id]
    print(f"morpheme pieces ({len(pieces)}개): {pieces}")
    print(f"token ids        : {ids}")
    print(f"UNK 조각 수      : {n_unk} / {len(pieces)}  (비율 {n_unk / max(len(pieces),1):.1%})")
    if unk_pieces:
        print(f"  >> UNK가 된 조각들: {unk_pieces}")
        print("  >> 이게 많으면 '학습 데이터에 이 형태소가 없었다' 또는 'vocab_size가 작다'는 뜻.")

    # ------------------------------------------------------------------
    hr("[3] 디코드 라운드트립 (인코드→디코드가 원문을 보존하는가)")
    decoded = tok.decode(ids, skip_special_tokens=True)
    print(f"원본    : {args.prompt!r}")
    print(f"디코드  : {decoded!r}")
    if decoded.replace(" ", "") == args.prompt.replace(" ", "").replace("\n", ""):
        print("  >> OK: 라운드트립 보존됨.")
    else:
        print("  >> 주의: 라운드트립에서 표면형이 변형됨 (예: '어디인가요?' → '어딘가요?').")
        print("     이건 kiwi.join이 형태소 (이/VCP + ㄴ가/EF)를 합칠 때 'ㄴ'을 종성으로 흡수해서 그렇습니다.")
        print("     → 모델이 못 배운 게 아니라 디코더가 표면형을 재조합하면서 생기는 정상 현상.")

    # ------------------------------------------------------------------
    hr("[4] 학습 데이터 첫 줄 인코드 UNK 비율 (토크나이저 자체 검증)")
    train_path = Path(args.train_txt)
    if not train_path.exists():
        print(f"[skip] {train_path} 없음 — 이 단계 건너뜀.")
    else:
        with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = ""
            for line in f:
                if line.strip():
                    sample = line.strip()
                    break
        sample = sample[:500]
        print(f"학습 샘플(500자): {sample!r}")
        sp = tok._tokenize_to_pieces(sample)
        n_unk = sum(1 for p in sp if p not in tok.token_to_id)
        print(f"학습 텍스트 조각 수: {len(sp)}, UNK: {n_unk} ({n_unk / max(len(sp),1):.2%})")
        if n_unk / max(len(sp), 1) > 0.05:
            print("  >> 경고: 학습 데이터에서조차 UNK가 5% 이상. 토크나이저 vocab이 너무 작거나 학습-사용 불일치.")
        else:
            print("  >> OK: 학습 데이터는 토크나이저로 거의 다 커버됨.")

    # ------------------------------------------------------------------
    hr("[5] 모델 로드 & vocab 크기 일치 검사")
    print(f"[load] {args.checkpoint}")
    model = GrinViModel.from_pretrained(args.checkpoint, device=args.device)
    model.eval()
    # vocab_size 추출 시도
    cfg = getattr(model, "config", None)
    m_vocab = getattr(cfg, "vocab_size", None) if cfg else None
    if m_vocab is None:
        # lm_head 추정
        for name, p in model.named_parameters():
            if "lm_head" in name and p.ndim == 2:
                m_vocab = p.shape[0]
                break
    print(f"model vocab_size      = {m_vocab}")
    print(f"tokenizer vocab_size  = {tok.vocab_size}")
    if m_vocab is not None and m_vocab != tok.vocab_size:
        print("  >> 🚨 치명적 불일치! 모델 출력 차원과 토크나이저 vocab이 다름 → 디코드 결과가 완전 엉뚱하게 나옵니다.")
    else:
        print("  >> OK: 일치.")

    # ------------------------------------------------------------------
    hr("[6] 모델 top-10 다음 토큰 예측 (학습이 진행됐는지의 직접 증거)")
    with torch.inference_mode():
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        logits, _ = model(x)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        top = torch.topk(probs, 10)
        print(f"프롬프트 마지막 위치에서 다음 토큰 top-10:")
        for rank, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            piece = tok.id_to_token[i] if 0 <= i < tok.vocab_size else "?"
            print(f"  {rank:2d}. id={i:6d}  p={p:.4f}  piece={piece!r}")
        # 균등분포(=학습 안 됨)인지 비교
        uniform_p = 1.0 / tok.vocab_size
        print(f"\n참고: 균등분포 확률 = {uniform_p:.2e}")
        print(f"top-1 확률          = {top.values[0].item():.4f}")
        if top.values[0].item() < uniform_p * 5:
            print("  >> 🚨 모델이 거의 균등분포 — 학습이 거의 안 됨.")
        elif top.values[0].item() < 0.01:
            print("  >> 주의: top-1 확률이 1% 미만 — 매우 초기 단계.")
        else:
            print("  >> OK: 모델이 명확한 분포를 형성함 (학습은 진행되고 있음).")

    # ------------------------------------------------------------------
    hr("요약")
    print("[2]의 UNK 비율, [4]의 학습데이터 UNK 비율, [5]의 vocab 일치, [6]의 top-1 확률")
    print("→ 이 4가지가 모두 OK면 '학습은 정상, 그냥 아직 스텝이 적음'.")
    print("→ 하나라도 빨간불이면 그게 진짜 원인입니다.")


if __name__ == "__main__":
    main()

