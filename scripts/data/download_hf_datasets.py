#!/usr/bin/env python3
"""
scripts/download_hf_datasets.py

HuggingFace에서 한국어 데이터셋을 스트리밍으로 다운로드하여
data/processed/train.txt에 append합니다.

라이센스 정책:
  - Apache 2.0, MIT, CC BY 계열만 포함 (상업적 이용 가능)
  - CC BY-NC (비상업적 전용) 제외
  - 라이센스 불명확한 데이터셋 제외

사용법:
    python scripts/download_hf_datasets.py --list
    python scripts/download_hf_datasets.py --datasets all
    python scripts/download_hf_datasets.py --datasets wiki2024,petitions,textbooks
    python scripts/download_hf_datasets.py --datasets wiki2024 --max_samples 100000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# 데이터셋 정의 (라이센스 안전한 것만)
# ---------------------------------------------------------------------------

DATASETS = {
    # ── Apache 2.0 ──────────────────────────────────────────────────────────
    "wiki2024": {
        "hf_name": "lcw99/wikipedia-korean-20240501",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 위키피디아 2024 (515K 문서, ~3.2GB)",
        "max_default": 515_000,
    },
    "textbooks_wikidata": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "ko_wikidata",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 위키데이터 기반 교과서 합성 데이터 (127K)",
        "max_default": 127_000,
    },
    "textbooks_claude": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "claude_evol",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 교과서 합성 데이터 - Claude evol (239K)",
        "max_default": 239_000,
    },
    "textbooks_code": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "code-alpaca",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 코드 알파카 합성 데이터 (64K)",
        "max_default": 64_000,
    },
    "textbooks_helpsteer": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "helpsteer",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 HelpSteer 합성 데이터 (25K)",
        "max_default": 25_000,
    },
    "textbooks_mmlu": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "mmlu_all",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 MMLU 전체 합성 데이터 (97K, 다양한 학문 분야)",
        "max_default": 97_000,
    },
    # ── MIT ─────────────────────────────────────────────────────────────────
    "petitions": {
        "hf_name": "heegyu/korean-petitions",
        "text_col": "content",
        "split": "train",
        "license": "MIT",
        "desc": "청와대 국민청원 (436K 건, 실제 시민 글쓰기)",
        "max_default": 400_000,
    },
    # ── CC BY 2.0 (출처 표기 필요, 상업적 이용 가능) ────────────────────────
    "nsmc": {
        "hf_name": "e9t/nsmc",
        "text_col": "document",
        "split": "train",
        "license": "CC BY 2.0",
        "desc": "네이버 영화 리뷰 NSMC (150K, 구어체)",
        "max_default": 150_000,
    },
    # ── CC BY-SA 3.0 ────────────────────────────────────────────────────────
    "kowikitext": {
        "hf_name": "heegyu/kowikitext",
        "hf_config": "default",
        "text_col": "text",
        "split": "train",
        "license": "CC BY-SA 3.0",
        "desc": "한국어 위키피디아 전체 텍스트 (1.33M 문서, ~6.5GB)",
        "max_default": 1_330_000,
        "parquet_direct": "kowikitext-20221001.parquet",
    },
    # ── CC BY-SA 4.0 ────────────────────────────────────────────────────────
    "klue_ynat": {
        "hf_name": "klue/klue",
        "hf_config": "ynat",
        "text_col": "title",
        "split": "train",
        "license": "CC BY-SA 4.0",
        "desc": "KLUE 뉴스 헤드라인 (45K)",
        "max_default": 45_000,
    },
    "klue_mrc": {
        "hf_name": "klue/klue",
        "hf_config": "mrc",
        "text_col": "context",
        "split": "train",
        "license": "CC BY-SA 4.0",
        "desc": "KLUE 기계독해 지문 (18K, 뉴스/위키 기반)",
        "max_default": 18_000,
    },
    # ── Apache 2.0 (대화/instruction) ────────────────────────────────────────
    "smol_koreantalk": {
        "hf_name": "lemon-mint/smol-koreantalk",
        "text_col": "messages",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "SmolLM2 한국어 번역 멀티턴 대화 (460K)",
        "max_default": 460_000,
        "chat_mode": True,
    },
    "coastral_writing": {
        "hf_name": "coastral/korean-writing-style-instruct",
        "text_col": "conversations",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 문체 합성 데이터 (29K, 문학/일상/고전)",
        "max_default": 29_000,
        "chat_mode": True,
    },
    "carrot_instruct": {
        "hf_name": "CarrotAI/ko-instruction-dataset",
        "text_col": "output",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "고품질 한국어 instruction (5K, WizardLM 방식)",
        "max_default": 5_000,
        "qa_mode": True,
        "instruction_col": "instruction",
    },
    "guanaco_ko": {
        "hf_name": "nlpai-lab/openassistant-guanaco-ko",
        "text_col": "text",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "OpenAssistant Guanaco 한국어 번역 (10K)",
        "max_default": 10_000,
    },
    # ── MIT (대화/instruction) ────────────────────────────────────────────────
    "kovast": {
        "hf_name": "maywell/koVast",
        "text_col": "conversations",
        "split": "train",
        "license": "MIT",
        "desc": "대규모 한국어 멀티턴 대화 (684K)",
        "max_default": 684_000,
        "chat_mode": True,
    },
    "open_korean_instructions": {
        "hf_name": "heegyu/open-korean-instructions",
        "text_col": "text",
        "split": "train",
        "license": "MIT",
        "desc": "KoAlpaca+ShareGPT+OIG 한국어 합본 (~1M)",
        "max_default": 1_000_000,
    },
    # ── CC BY-SA 3.0 ────────────────────────────────────────────────────────
    "korean_dict": {
        "hf_name": "hac541309/basic_korean_dict",
        "text_col": "text",
        "split": "train",
        "license": "CC BY-SA 3.0",
        "desc": "한국어 기초 사전 (75K, 정의+예문)",
        "max_default": 75_000,
    },
    # ── CC BY 4.0 ────────────────────────────────────────────────────────────
    "kopen_platypus": {
        "hf_name": "kyujinpy/KOpen-platypus",
        "text_col": "output",
        "split": "train",
        "license": "CC BY 4.0",
        "desc": "KOpen-Platypus 한국어 번역 (25K, 고품질 instruction)",
        "max_default": 25_000,
        "qa_mode": True,
        "instruction_col": "instruction",
    },
    # ── Apache 2.0 (추가) ────────────────────────────────────────────────────
    "kullm_v2": {
        "hf_name": "nlpai-lab/kullm-v2",
        "text_col": "output",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "KULLM-v2: GPT4ALL+Dolly+Vicuna 한국어 번역 (150K)",
        "max_default": 150_000,
        "qa_mode": True,
        "instruction_col": "instruction",
    },
    "korean_safe_conv": {
        "hf_name": "jojo0217/korean_safe_conversation",
        "text_col": "output",
        "split": "train",
        "license": "Apache 2.0",
        "desc": "한국어 안전 일상대화 (5K, 성균관대-VAIV)",
        "max_default": 5_000,
        "qa_mode": True,
        "instruction_col": "instruction",
    },
    # ── MIT (추가) ────────────────────────────────────────────────────────────
    "korquad_chat": {
        "hf_name": "heegyu/korquad-chat-v1",
        "text_col": "text",
        "split": "train",
        "license": "MIT",
        "desc": "KorQuAD 기반 지식 대화 (9.6K)",
        "max_default": 10_000,
    },
}

# ---------------------------------------------------------------------------
# 제거된 데이터셋 (라이센스 문제)
# ---------------------------------------------------------------------------
REMOVED_DATASETS = {
    "namuwiki":    "CC BY-NC-SA 2.0 — 비상업적 전용",
    "webtext":     "라이센스 불명확 (OSCAR/CC100 혼합 웹크롤)",
    "oscar":       "라이센스 불명확 (원본 웹크롤 저작권 불명확)",
    "magpie":      "라이센스 불명확",
    "textbooks":   "이름 변경됨 → textbooks_wikidata 사용",
    "ko_alpaca":   "CC BY-NC 4.0 — 비상업적 전용",
    "lbox_open":   "CC BY-NC 4.0 — 비상업적 전용",
    "kor_openorca":"CC BY-NC 4.0 — 비상업적 전용",
    "koalpaca_v1": "라이센스 불명확",
    "k2_feedback": "라이센스 불명확",
    "ko_genstruct":"라이센스 불명확",
}


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        import os
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def download_dataset(
    name: str,
    cfg: dict,
    out_file: Path,
    max_samples: int | None = None,
    min_length: int = 20,
) -> int:
    from datasets import load_dataset

    limit = max_samples or cfg["max_default"]
    print(f"\n📥 [{name}] {cfg['desc']}")
    print(f"   라이센스: {cfg['license']} | HF: {cfg['hf_name']}")
    print(f"   최대 {limit:,}개 | 최소 {min_length}자")

    try:
        hf_config = cfg.get("hf_config", None)
        parquet_direct = cfg.get("parquet_direct", None)
        if parquet_direct:
            # 스크립트 기반 데이터셋 — parquet 직접 로드
            url = f"https://huggingface.co/datasets/{cfg['hf_name']}/resolve/main/{parquet_direct}"
            ds = load_dataset("parquet", data_files={"train": url}, split="train", streaming=True)
        elif hf_config:
            ds = load_dataset(cfg["hf_name"], hf_config, split=cfg["split"], streaming=True)
        else:
            ds = load_dataset(cfg["hf_name"], split=cfg["split"], streaming=True)
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return 0

    text_col = cfg["text_col"]
    qa_mode = cfg.get("qa_mode", False)
    chat_mode = cfg.get("chat_mode", False)
    inst_col = cfg.get("instruction_col", None)

    count = 0
    skipped = 0
    start = time.time()

    with open(out_file, "a", encoding="utf-8") as f:
        for sample in ds:
            if count >= limit:
                break

            if chat_mode:
                # conversations/messages 리스트 → 텍스트로 변환
                turns = sample.get(text_col, []) or []
                parts = []
                for turn in turns:
                    role = turn.get("role") or turn.get("from") or ""
                    content = turn.get("content") or turn.get("value") or ""
                    content = str(content).strip()
                    if content:
                        parts.append(content)
                text = "\n".join(parts)
            elif qa_mode and inst_col:
                instruction = str(sample.get(inst_col, "")).strip()
                answer = str(sample.get(text_col, "")).strip()
                if not instruction or not answer:
                    skipped += 1
                    continue
                text = f"질문: {instruction}\n답변: {answer}"
            else:
                text = str(sample.get(text_col, "")).strip()

            if len(text) < min_length:
                skipped += 1
                continue

            f.write(text + "\n")
            count += 1

            if count % 50_000 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                eta = (limit - count) / rate if rate > 0 else 0
                print(f"   [{name}] {count:,}/{limit:,} ({rate:.0f}/s, ETA {eta/60:.1f}분)")

    elapsed = time.time() - start
    print(f"   ✅ {count:,}개 저장 (건너뜀 {skipped:,}개, {elapsed:.1f}초)")
    return count


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="HuggingFace 한국어 데이터셋 다운로드 (라이센스 안전한 것만)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/download_hf_datasets.py --list
  python scripts/download_hf_datasets.py --datasets all
  python scripts/download_hf_datasets.py --datasets wiki2024,petitions
  python scripts/download_hf_datasets.py --datasets wiki2024 --max_samples 50000
        """,
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"다운로드할 데이터셋 (쉼표 구분 또는 'all'). 선택지: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--out",
        default="data/processed/train.txt",
        help="출력 파일 (기본: data/processed/train.txt, append 모드)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="데이터셋당 최대 샘플 수 (기본: 각 데이터셋 설정값)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="최소 텍스트 길이 (기본: 20자)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 데이터셋 목록 출력",
    )
    args = parser.parse_args()

    if args.list:
        print("\n✅ 사용 가능한 데이터셋 (라이센스 검증 완료):")
        print(f"  {'이름':<20} {'라이센스':<18} 설명")
        print("  " + "-" * 70)
        for k, v in DATASETS.items():
            print(f"  {k:<20} {v['license']:<18} {v['desc']}")
        print()
        print("🚫 제거된 데이터셋 (라이센스 문제):")
        for k, reason in REMOVED_DATASETS.items():
            print(f"  {k:<20} {reason}")
        return

    if args.datasets == "all":
        selected = list(DATASETS.keys())
    else:
        selected = [s.strip() for s in args.datasets.split(",")]

    # 제거된 데이터셋 요청 시 경고
    for name in selected:
        if name in REMOVED_DATASETS:
            print(f"❌ '{name}'은 라이센스 문제로 제거됨: {REMOVED_DATASETS[name]}")
            print("   다른 데이터셋을 선택하세요.")
            sys.exit(1)

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    before_lines = 0
    if out_file.exists():
        with open(out_file, "rb") as f:
            before_lines = sum(1 for _ in f)

    print(f"🇰🇷 한국어 데이터셋 다운로더 (라이센스 안전)")
    print(f"   대상: {', '.join(selected)}")
    print(f"   출력: {out_file} (현재 {before_lines:,}줄)")

    total = 0
    for name in selected:
        if name not in DATASETS:
            print(f"⚠ 알 수 없는 데이터셋: '{name}' (건너뜀)")
            print(f"  사용 가능: {', '.join(DATASETS.keys())}")
            continue
        total += download_dataset(
            name,
            DATASETS[name],
            out_file,
            max_samples=args.max_samples,
            min_length=args.min_length,
        )

    after_lines = 0
    if out_file.exists():
        with open(out_file, "rb") as f:
            after_lines = sum(1 for _ in f)

    size_mb = out_file.stat().st_size / 1e6 if out_file.exists() else 0
    print(f"\n{'='*50}")
    print(f"✅ 완료!")
    print(f"   추가된 샘플: {total:,}개")
    print(f"   전체 줄 수: {before_lines:,} → {after_lines:,}")
    print(f"   파일 크기: {size_mb:.1f} MB")
    print(f"   위치: {out_file}")


if __name__ == "__main__":
    main()
