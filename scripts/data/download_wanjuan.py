#!/usr/bin/env python3
"""
scripts/download_wanjuan.py

WanJuan-Korean (CC BY 4.0) 데이터셋을 HuggingFace에서 직접 다운로드하여
data/processed/train.txt에 append합니다.

총 115.9GB / 47개 파일 / 7개 카테고리
- culture, encyclopedia, general, history_policy
- local_life, news, professional_field
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

HF_BASE = "https://huggingface.co/datasets/opendatalab/WanJuan-Korean/resolve/main"

# 47개 파일 전체 목록
FILES = [
    "raw/culture/common_crawl/part-677f75d865d8-000260.jsonl.gz",
    "raw/culture/common_crawl/part-677f75d865d8-001108.jsonl.gz",
    "raw/culture/ebooks/part-677f75d865d8-000303.jsonl.gz",
    "raw/culture/general/part-677f75d865d8-000616.jsonl.gz",
    "raw/encyclopedia/common_crawl/part-677f75d865d8-000693.jsonl.gz",
    "raw/encyclopedia/general/part-677f75d865d8-001483.jsonl.gz",
    "raw/general/common_crawl/part-677f75d865d8-000020.jsonl.gz",
    "raw/general/common_crawl/part-677f75d865d8-000732.jsonl.gz",
    "raw/general/common_crawl/part-677f75d865d8-000965.jsonl.gz",
    "raw/general/general/part-677f75d865d8-000695.jsonl.gz",
    "raw/history_policy/common_crawl/part-677f75d865d8-000002.jsonl.gz",
    "raw/history_policy/common_crawl/part-677f75d865d8-000573.jsonl.gz",
    "raw/history_policy/general/part-677f75d865d8-001138.jsonl.gz",
    "raw/local_life/common_crawl/part-677f75d865d8-000478.jsonl.gz",
    "raw/local_life/common_crawl/part-677f75d865d8-000736.jsonl.gz",
    "raw/local_life/common_crawl/part-677f75d865d8-000798.jsonl.gz",
    "raw/local_life/general/part-677f75d865d8-000466.jsonl.gz",
    "raw/news/common_crawl/part-677f75d865d8-000392.jsonl.gz",
    "raw/news/common_crawl/part-677f75d865d8-001334.jsonl.gz",
    "raw/news/general/part-677f75d865d8-000022.jsonl.gz",
    "raw/news/general/part-677f75d865d8-000372.jsonl.gz",
    "raw/news/general/part-677f75d865d8-001674.jsonl.gz",
    "raw/news/general/part-677f75d865d8-001698.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-000171.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-000306.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-000832.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-000918.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-000959.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-001139.jsonl.gz",
    "raw/professional_field/common_crawl/part-677f75d865d8-001396.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000048.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000202.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000221.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000233.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000350.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000389.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000495.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000571.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000833.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-000972.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001022.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001077.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001109.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001433.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001549.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001587.jsonl.gz",
    "raw/professional_field/general/part-677f75d865d8-001665.jsonl.gz",
]


def stream_and_append(file_path: str, out_file: Path, min_length: int = 20) -> int:
    """HF에서 스트리밍으로 받아 바로 train.txt에 append. 디스크에 gz 저장 안 함."""
    import requests

    url = f"{HF_BASE}/{file_path}"
    category = "/".join(file_path.split("/")[1:3])
    count = 0
    skipped = 0

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        with gzip.open(resp.raw, "rt", encoding="utf-8", errors="ignore") as gz, \
             open(out_file, "a", encoding="utf-8") as fout:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("content", obj.get("text", "")).strip()
                    if len(text) >= min_length:
                        fout.write(text + "\n")
                        count += 1
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1

    except Exception as e:
        print(f"  ❌ [{category}] {Path(file_path).name}: {e}")
        return count

    return count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/train.txt")
    parser.add_argument("--min_length", type=int, default=20)
    parser.add_argument("--files", default="all", help="all 또는 쉼표구분 인덱스 (0-46)")
    args = parser.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.files == "all":
        targets = FILES
    else:
        idxs = [int(i) for i in args.files.split(",")]
        targets = [FILES[i] for i in idxs]

    before = sum(1 for _ in open(out_file, "rb")) if out_file.exists() else 0

    print(f"🇰🇷 WanJuan-Korean 다운로드 (CC BY 4.0)")
    print(f"   파일 수: {len(targets)}/47")
    print(f"   출력: {out_file} (현재 {before:,}줄)")
    print()

    total = 0
    start_all = time.time()

    for i, fpath in enumerate(targets):
        category = "/".join(fpath.split("/")[1:3])
        fname = Path(fpath).name
        print(f"[{i+1:2d}/{len(targets)}] {category}/{fname} ...", flush=True)
        t0 = time.time()
        n = stream_and_append(fpath, out_file, args.min_length)
        elapsed = time.time() - t0
        total += n
        print(f"         ✅ {n:,}개 ({elapsed:.0f}s) | 누적 {total:,}개")

    after = sum(1 for _ in open(out_file, "rb")) if out_file.exists() else 0
    size_gb = out_file.stat().st_size / 1e9
    elapsed_all = time.time() - start_all

    print(f"\n{'='*55}")
    print(f"✅ WanJuan-Korean 완료!")
    print(f"   추가 샘플: {total:,}개")
    print(f"   전체 줄:   {before:,} → {after:,}")
    print(f"   파일 크기: {size_gb:.1f} GB")
    print(f"   소요 시간: {elapsed_all/60:.1f}분")


if __name__ == "__main__":
    main()
