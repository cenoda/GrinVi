"""LocalParser: 로컬 JSON, CSV, TXT 파일 파서."""

import csv
import json
import logging
from pathlib import Path
from typing import List

from ..models import ExamItem, ExamMetadata
from .base import BaseParser

logger = logging.getLogger(__name__)


class LocalParser(BaseParser):
    """로컬 파일(JSON, CSV, TXT) 파서.

    다양한 형식의 로컬 파일을 읽어 ExamItem으로 변환합니다.
    """

    def parse(self, raw: dict) -> ExamItem:
        """원시 데이터 dict를 ExamItem으로 변환.

        다양한 필드명을 지원합니다:
        - question/문제/문제 내용/q
        - passage/지문/context/paragraph
        - choices/선지/선택지/options
        - answer/정답/gold
        - explanation/해설/설명

        Args:
            raw: 원시 데이터 딕셔너리

        Returns:
            ExamItem: 변환된 기출문제 객체

        Raises:
            ValueError: question 필드가 비어있는 경우
        """
        # question 필드 추출 (다양한 필드명 지원)
        question = (
            raw.get("question")
            or raw.get("문제")
            or raw.get("문제 내용")
            or raw.get("q")
            or ""
        )

        if not question.strip():
            raise ValueError("question 필드가 비어있습니다.")

        # passage 필드 추출
        passage = (
            raw.get("passage")
            or raw.get("지문")
            or raw.get("context")
            or raw.get("paragraph")
            or ""
        )

        # choices 필드 추출
        choices = self._extract_choices(raw)

        # answer 필드 추출
        answer_raw = raw.get("answer") or raw.get("정답") or raw.get("gold")
        answer = int(answer_raw) if answer_raw is not None else None

        # explanation 필드 추출
        explanation = (
            raw.get("explanation")
            or raw.get("해설")
            or raw.get("설명")
        )

        # metadata 추출
        exam_type = raw.get("exam_type", raw.get("시험종류", "local"))
        year = int(raw.get("year", raw.get("연도", 0)))
        subject = raw.get("subject", raw.get("과목", "국어"))

        metadata = ExamMetadata(
            exam_type=exam_type,
            year=year,
            subject=subject,
            question_number=raw.get("question_number", raw.get("문제 번호")),
        )

        return ExamItem(
            passage=passage,
            question=question,
            choices=choices,
            answer=answer,
            explanation=explanation,
            metadata=metadata,
        )

    def _extract_choices(self, raw: dict) -> List[str] | None:
        """다양한 형식의 선지를 추출."""
        # 리스트 형태
        choices = raw.get("choices") or raw.get("선지") or raw.get("options")
        if choices and isinstance(choices, list):
            return choices

        # 개별 필드 형태 (선택지 1~5 또는 A~E)
        numbered_choices = []
        for i in range(1, 6):
            val = raw.get(f"선택지 {i}") or raw.get(f"choice_{i}")
            if val:
                numbered_choices.append(str(val))

        if numbered_choices:
            return numbered_choices

        # A~E 형태
        letter_choices = []
        for letter in "ABCDE":
            val = raw.get(letter)
            if val:
                letter_choices.append(str(val))

        if letter_choices:
            return letter_choices

        return None

    def parse_file(self, path: Path) -> List[ExamItem]:
        """파일 형식에 따라 적절한 파싱 메서드를 호출.

        Args:
            path: 파일 경로

        Returns:
            List[ExamItem]: 파싱된 ExamItem 리스트
        """
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            return self._parse_jsonl(path)
        elif suffix == ".json":
            return self._parse_json(path)
        elif suffix == ".csv":
            return self._parse_csv(path)
        elif suffix == ".txt":
            return self._parse_txt(path)
        else:
            logger.error(f"지원하지 않는 파일 형식: {suffix} ({path})")
            print(f"❌ 오류: 지원하지 않는 파일 형식: {suffix}")
            return []

    def _parse_jsonl(self, path: Path) -> List[ExamItem]:
        """JSONL 파일 파싱 (BaseParser의 기본 구현 사용)."""
        return super().parse_file(path)

    def _parse_json(self, path: Path) -> List[ExamItem]:
        """JSON 파일 파싱 (배열 또는 단일 객체)."""
        items = []
        skipped = 0

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"JSON 파일 읽기 실패: {path} - {e}")
            return []

        # 배열인 경우
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # 단일 객체이거나 data 키 아래 배열
            if "data" in data and isinstance(data["data"], list):
                records = data["data"]
            else:
                records = [data]
        else:
            logger.warning(f"예상치 못한 JSON 구조: {path}")
            return []

        for i, raw in enumerate(records):
            try:
                item = self.parse(raw)
                items.append(item)
            except (KeyError, ValueError) as e:
                skipped += 1
                logger.warning(f"{path.name}:{i + 1} 파싱 실패 - {e}")

        if skipped > 0:
            logger.info(f"{path.name}: {len(items)}건 파싱, {skipped}건 건너뜀")

        return items

    def _parse_csv(self, path: Path) -> List[ExamItem]:
        """CSV 파일 파싱."""
        items = []
        skipped = 0

        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, start=1):
                    try:
                        item = self.parse(dict(row))
                        items.append(item)
                    except (KeyError, ValueError) as e:
                        skipped += 1
                        logger.warning(f"{path.name}:{i} 파싱 실패 - {e}")
        except (OSError, csv.Error) as e:
            logger.error(f"CSV 파일 읽기 실패: {path} - {e}")
            return []

        if skipped > 0:
            logger.info(f"{path.name}: {len(items)}건 파싱, {skipped}건 건너뜀")

        return items

    def _parse_txt(self, path: Path) -> List[ExamItem]:
        """TXT 파일 파싱.

        각 줄을 하나의 문제 텍스트로 처리합니다.
        빈 줄로 구분된 블록을 하나의 문항으로 처리할 수도 있습니다.
        """
        items = []

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(f"TXT 파일 읽기 실패: {path} - {e}")
            return []

        # 빈 줄로 구분된 블록 단위로 처리
        blocks = text.split("\n\n")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # 각 블록을 하나의 문항으로 처리
            metadata = ExamMetadata(
                exam_type="local",
                year=0,
                subject="국어",
                source_file=str(path),
            )

            item = ExamItem(
                passage="",
                question=block,
                choices=None,
                answer=None,
                explanation=None,
                metadata=metadata,
            )
            items.append(item)

        return items
