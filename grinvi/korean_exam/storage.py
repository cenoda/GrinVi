"""JSONL 저장 및 train.txt 병합 스토리지."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class Storage:
    """변환된 레코드를 JSONL 파일로 저장하고, 선택적으로 train.txt에 병합한다."""

    def save(self, records: List[dict], out_dir: Path, merge: bool = False, repeat: int = 1) -> Path:
        """레코드를 JSONL 파일로 저장한다.

        Args:
            records: TrainingRecord dict 리스트.
            out_dir: 출력 기본 디렉토리 (예: data/).
            merge: True이면 data/processed/train.txt에 text 필드를 append.
            repeat: merge 시 반복 횟수 (가중치). 기본값 1.

        Returns:
            저장된 JSONL 파일 경로.

        Raises:
            OSError: 디스크 쓰기 오류 시 부분 저장 파일을 삭제하고 예외를 전파한다.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        jsonl_path = out_dir / "raw" / "korean_exam" / f"korean_exam_{date_str}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("저장 실패: %s", e)
            jsonl_path.unlink(missing_ok=True)
            raise

        # 저장 완료 후 통계 출력
        file_size = jsonl_path.stat().st_size
        logger.info(
            "저장 완료: %d건, %s, %s",
            len(records),
            _format_size(file_size),
            jsonl_path,
        )

        if merge:
            train_path = out_dir / "processed" / "train.txt"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            with train_path.open("a", encoding="utf-8") as f:
                for _ in range(repeat):
                    for rec in records:
                        f.write(rec["text"] + "\n")
            logger.info("병합 완료: %s (x%d회 반복)", train_path, repeat)

        return jsonl_path


def _format_size(size_bytes: int) -> str:
    """바이트 수를 사람이 읽기 쉬운 형태로 변환한다."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
