"""HuggingFace 데이터셋 수집기."""

import json
import logging
import sys
from pathlib import Path
from typing import Iterator

from .base import BaseCollector

logger = logging.getLogger(__name__)

HF_DATASETS = [
    {
        "id": "csatqa",
        "hf_id": "HAERAE-HUB/csatqa",
        "method": "url",
        "url": "https://huggingface.co/datasets/HAERAE-HUB/csatqa/resolve/main/data/csatqa.json",
        "url_eval": "https://huggingface.co/datasets/HAERAE-HUB/csatqa/resolve/main/data/csatqa_eval.json",
    },
    {
        "id": "csat_2025",
        "hf_id": "KKACHI-HUB/CSAT-KOREAN-2025",
        "method": "load_dataset",
        "split": "train",
    },
    {
        "id": "csat_sft",
        "hf_id": "LLMin/final_csat_sft",
        "method": "load_dataset",
        "split": "train",
    },
    {
        "id": "civil_local",
        "hf_id": "kikikara/Korean-Civil-Service-Examination-Train",
        "method": "load_dataset",
        "split": "train",
        "filter": {"task": "국어"},
    },
    {
        "id": "civil_national",
        "hf_id": "kikikara/Korean-Civil-Service-Examination-National",
        "method": "load_dataset",
        "split": "공무원_국가직",
        "filter": {"task": "국어"},
    },
]


class HFCollector(BaseCollector):
    """HuggingFace 데이터셋 수집기.

    5개 HuggingFace 데이터셋을 순서대로 수집합니다.
    - csatqa: URL 직접 다운로드
    - 나머지: datasets.load_dataset API 사용
    - kikikara 데이터셋: task='국어' 필터링 적용
    """

    def collect(self) -> Iterator[Path]:
        """모든 HF 데이터셋을 수집하고 저장된 파일 경로를 yield.

        Raises:
            SystemExit: 네트워크 오류 발생 시 비정상 종료
        """
        for ds_config in HF_DATASETS:
            ds_id = ds_config["id"]
            out_path = self.out_dir / "huggingface" / ds_id / "raw.jsonl"

            if self._is_cached(out_path):
                logger.info(f"캐시 사용: {out_path}")
                yield out_path
                continue

            try:
                if ds_config["method"] == "url":
                    records = self._collect_url(ds_config)
                else:
                    records = self._collect_load_dataset(ds_config)

                self._save_jsonl(records, out_path)
                logger.info(f"수집 완료: {ds_id} ({len(records)}건) -> {out_path}")
                yield out_path

            except Exception as e:
                logger.error(
                    f"데이터셋 수집 실패: {ds_config['hf_id']} - {e}\n"
                    f"재시도: python scripts/collect_korean_exam.py --source huggingface"
                )
                sys.exit(1)

    def _collect_url(self, ds_config: dict) -> list:
        """URL 직접 다운로드로 데이터 수집 (csatqa용).

        Args:
            ds_config: 데이터셋 설정 dict

        Returns:
            list: 수집된 레코드 리스트
        """
        import requests

        records = []
        urls = [ds_config["url"]]
        if "url_eval" in ds_config:
            urls.append(ds_config["url_eval"])

        for url in urls:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()

            # 먼저 단일 JSON으로 파싱 시도, 실패하면 JSONL(줄별 JSON)로 처리
            try:
                data = resp.json()
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    # JSON 파일이 dict 형태일 경우 (예: {key: [items]})
                    for value in data.values():
                        if isinstance(value, list):
                            records.extend(value)
                        else:
                            records.append(value)
                else:
                    records.append(data)
            except json.JSONDecodeError:
                # JSONL 형식: 줄마다 JSON 객체
                for line in resp.text.strip().split("\n"):
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        return records

    def _collect_load_dataset(self, ds_config: dict) -> list:
        """datasets.load_dataset API로 데이터 수집.

        Args:
            ds_config: 데이터셋 설정 dict

        Returns:
            list: 수집된 레코드 리스트
        """
        from datasets import load_dataset

        dataset = load_dataset(
            ds_config["hf_id"],
            split=ds_config["split"],
            trust_remote_code=True,
        )

        records = []
        filter_config = ds_config.get("filter")

        for row in dataset:
            # dict로 변환
            record = dict(row)

            # 필터 적용 (kikikara 데이터셋의 task='국어' 필터)
            if filter_config:
                skip = False
                for key, value in filter_config.items():
                    if record.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            records.append(record)

        return records

    def _save_jsonl(self, records: list, path: Path) -> None:
        """레코드를 JSONL 형식으로 저장.

        Args:
            records: 저장할 레코드 리스트
            path: 저장 경로
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
