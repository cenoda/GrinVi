"""BaseParser 추상 클래스."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import ExamItem

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """파서의 기본 추상 클래스.

    모든 Parser는 이 클래스를 상속하여 parse() 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def parse(self, raw: dict) -> ExamItem:
        """원시 데이터 dict를 ExamItem으로 변환.

        Args:
            raw: 원시 데이터 딕셔너리

        Returns:
            ExamItem: 변환된 기출문제 객체
        """
        ...

    def parse_file(self, path: Path) -> List[ExamItem]:
        """JSONL 파일을 읽어 각 줄을 parse()로 변환.

        Args:
            path: JSONL 파일 경로

        Returns:
            List[ExamItem]: 변환된 ExamItem 리스트
        """
        items: List[ExamItem] = []
        skipped = 0

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    item = self.parse(raw)
                    items.append(item)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    skipped += 1
                    logger.warning(
                        f"{path.name}:{line_num} 파싱 실패 - {e}"
                    )

        if skipped > 0:
            logger.info(f"{path.name}: {len(items)}건 파싱, {skipped}건 건너뜀")

        return items
