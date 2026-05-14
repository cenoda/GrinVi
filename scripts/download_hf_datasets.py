#!/usr/bin/env python3
"""
scripts/download_hf_datasets.py

HuggingFace에서 한국어 데이터셋을 스트리밍으로 다운로드하여
data/processed/train.txt에 append합니다.

사용법:
    python scripts/download_hf_datasets.py --datasets all
    python scripts/download_hf_datasets.py --datasets namuwiki,webtext
    python scripts/download_hf_datasets.py --datasets oscar --max_samples 500000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 데이터셋 정의
# ---------------------------------------------------------------------------

DATASETS = {
    "namuwiki": {
        "hf_name": "heegyu/namuwiki-extracted",
        "text_col": "text",
        "split": "train",
        "desc": "나무위키 (565K 문서, ~9.7GB)",
        "max_default": 500_000,
    },
    "webtext": {
        "hf_name": "HAERAE-HUB/KOREAN-WEBTEXT",
        "text_col": "text",
        "split": "train",
        "desc": "한국어 웹텍스트 (1.28M 문서, ~8.5GB)",
        "max_default": 800_000,
    },
    "oscar": {
        "hf_name": "lcw99/oscar-ko-only",
        "text_col": "text",
        "split": "train",
        "desc": "OSCAR 한국어 크롤 (3.67M 문서, ~12.5GB)",
        "max_default": 1_000_000,
    },
    "petitions": {
        "hf_name": "heegyu/korean-petitions",
        "text_col": "content",
        "split": "train",
        "desc": "청와대 국민청원 (436K 건, ~609MB)",
        "max_default": 400_000,
    },
    "magpie": {
        "hf_name": "channelcorp/KoMagpie-raw",
        "text_col": "output",
        "split": "train",
        "desc": "KoMagpie 한국어 instruction (2.57M 쌍, ~3.7GB)",
        "max_default": 500_000,
        "qa_mode": True,   # instruction+output 형식
        "instruction_col": "instruction",
    },
    "nsmc": {
        "hf_name": "e9t/nsmc",
        "text_col": "document",
        "split": "train",
        "desc": "네이버 영화 리뷰 (150K, 구어체)",
        "max_default": 150_000,
    },
    "wiki2024": {
        "hf_name": "lcw99/wikipedia-korean-20240501",
        "text_col": "text",
        "split": "train",
        "desc": "한국어 위키피디아 2024 (515K 문서, CC BY-SA)",
        "max_default": 515_000,
    },
    "ko_wikidata": {
        "hf_name": "maywell/korean_textbooks",
        "hf_config": "ko_wikidata",
        "text_col": "text",
        "split": "train",
        "desc": "한국어 위키데이터 교과서 (127K, CC BY-SA)",
        "max_default": 127_000,
    },
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
    print(f"   HF: {cfg['hf_name']} | 최대 {limit:,}개 | 최소 {min_length}자")

    try:
        hf_config = cfg.get("hf_config", None)
        if hf_config:
            ds = load_dataset(cfg["hf_name"], hf_config, split=cfg["split"], streaming=True)
        else:
            ds = load_dataset(cfg["hf_name"], split=cfg["split"], streaming=True)
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return 0

    text_col = cfg["text_col"]
    qa_mode = cfg.get("qa_mode", False)
    inst_col = cfg.get("instruction_col", None)

    count = 0
    skipped = 0
    start = time.time()

    with open(out_file, "a", encoding="utf-8") as f:
        for sample in ds:
            if count >= limit:
                break

            if qa_mode and inst_col:
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

    parser = argparse.ArgumentParser(description="HuggingFace 한국어 데이터셋 다운로드")
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
        print("\n사용 가능한 데이터셋:")
        for k, v in DATASETS.items():
            print(f"  {k:12s} — {v['desc']}")
        return

    if args.datasets == "all":
        selected = list(DATASETS.keys())
    else:
        selected = [s.strip() for s in args.datasets.split(",")]

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    before_lines = 0
    if out_file.exists():
        with open(out_file, "rb") as f:
            before_lines = sum(1 for _ in f)

    print(f"🇰🇷 한국어 데이터셋 다운로더")
    print(f"   대상: {', '.join(selected)}")
    print(f"   출력: {out_file} (현재 {before_lines:,}줄)")

    total = 0
    for name in selected:
        if name not in DATASETS:
            print(f"⚠ 알 수 없는 데이터셋: {name} (건너뜀)")
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
