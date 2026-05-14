"""HuggingFace 포맷별 파서 (CSATQAParser, CSATKKACHIParser, SFTParser, CivilParser)."""

import logging
import re
from typing import List, Optional, Tuple

from ..models import ExamItem, ExamMetadata
from .base import BaseParser

logger = logging.getLogger(__name__)


class CSATQAParser(BaseParser):
    """HAERAE-HUB/csatqa 포맷 파서.

    입력 필드: test_name, context, question, gold, option#1~5
    test_name 예: "2022년 2023 대학수학능력시험 국어  홀수형"
    """

    # 연도 및 시험 종류 추출 정규식
    _YEAR_PATTERN = re.compile(r"(\d{4})\s*대학수학능력시험")
    _MOCK_JUNE_PATTERN = re.compile(r"(\d{4}).*6월.*모의고사")
    _MOCK_SEPT_PATTERN = re.compile(r"(\d{4}).*9월.*모의고사")

    def _parse_metadata(self, test_name: str) -> ExamMetadata:
        """test_name에서 연도와 시험 종류를 추출.

        Args:
            test_name: 시험 이름 문자열

        Returns:
            ExamMetadata: 추출된 메타데이터
        """
        # 수능 패턴
        match = self._YEAR_PATTERN.search(test_name)
        if match:
            return ExamMetadata(
                exam_type="csat",
                year=int(match.group(1)),
                month=11,
                subject="국어",
                source_dataset="HAERAE-HUB/csatqa",
            )

        # 6월 모의고사 패턴
        match = self._MOCK_JUNE_PATTERN.search(test_name)
        if match:
            return ExamMetadata(
                exam_type="mock_june",
                year=int(match.group(1)),
                month=6,
                subject="국어",
                source_dataset="HAERAE-HUB/csatqa",
            )

        # 9월 모의고사 패턴
        match = self._MOCK_SEPT_PATTERN.search(test_name)
        if match:
            return ExamMetadata(
                exam_type="mock_sept",
                year=int(match.group(1)),
                month=9,
                subject="국어",
                source_dataset="HAERAE-HUB/csatqa",
            )

        # 패턴 매칭 실패 시 test_name에서 연도만 추출 시도
        year_match = re.search(r"(\d{4})", test_name)
        year = int(year_match.group(1)) if year_match else 0

        return ExamMetadata(
            exam_type="csat",
            year=year,
            subject="국어",
            source_dataset="HAERAE-HUB/csatqa",
        )

    def parse(self, raw: dict) -> ExamItem:
        """csatqa 레코드를 ExamItem으로 변환.

        Args:
            raw: 원시 데이터 dict

        Returns:
            ExamItem: 변환된 기출문제

        Raises:
            ValueError: question이 비어있는 경우
        """
        question = raw.get("question", "")
        if not question or not question.strip():
            logger.warning(f"question 비어있음: test_name={raw.get('test_name', '?')}")
            raise ValueError("question이 비어있습니다")

        choices = [raw.get(f"option#{i}", "") for i in range(1, 6)]
        # 빈 선지 제거하지 않음 (5지선다 구조 유지)

        gold = raw.get("gold")
        answer = int(gold) if gold is not None else None

        return ExamItem(
            passage=raw.get("context", ""),
            question=question,
            choices=choices,
            answer=answer,
            explanation=None,
            metadata=self._parse_metadata(raw.get("test_name", "")),
        )


class CSATKKACHIParser(BaseParser):
    """KKACHI-HUB/CSAT-KOREAN-2025 포맷 파서.

    입력 필드: idx, paragraph, question, question_plus, A~E, answer, point
    """

    def parse(self, raw: dict) -> ExamItem:
        """CSAT-KOREAN-2025 레코드를 ExamItem으로 변환.

        Args:
            raw: 원시 데이터 dict

        Returns:
            ExamItem: 변환된 기출문제

        Raises:
            ValueError: question이 비어있는 경우
        """
        question = raw.get("question", "")
        if not question or not question.strip():
            logger.warning(f"question 비어있음: idx={raw.get('idx', '?')}")
            raise ValueError("question이 비어있습니다")

        # question_plus가 있으면 question에 추가
        question_text = question
        question_plus = raw.get("question_plus")
        if question_plus and question_plus.strip():
            question_text += "\n" + question_plus

        choices = [raw.get(k, "") for k in ["A", "B", "C", "D", "E"]]

        answer = raw.get("answer")
        answer_int = int(answer) if answer is not None else None

        return ExamItem(
            passage=raw.get("paragraph", ""),
            question=question_text,
            choices=choices,
            answer=answer_int,
            explanation=None,
            metadata=ExamMetadata(
                exam_type="csat",
                year=2025,
                month=11,
                subject="국어",
                source_dataset="KKACHI-HUB/CSAT-KOREAN-2025",
                question_number=raw.get("idx"),
            ),
        )


class SFTParser(BaseParser):
    """LLMin/final_csat_sft 포맷 파서.

    입력 필드: messages (system/user/assistant)
    user 메시지에서 지문/문제/선지 추출, assistant 메시지에서 해설 추출
    """

    # 선지 패턴: ①②③④⑤ 또는 1) 2) 3) 4) 5)
    _CHOICE_PATTERN = re.compile(r"[①②③④⑤]")
    _NUMBERED_CHOICE_PATTERN = re.compile(r"^\s*[1-5]\)")

    def _split_user_message(
        self, user_msg: str
    ) -> Tuple[str, str, Optional[List[str]]]:
        """user 메시지에서 지문, 문제, 선지를 분리.

        Args:
            user_msg: user role의 메시지 내용

        Returns:
            Tuple[passage, question, choices]
        """
        lines = user_msg.split("\n")

        # 선지 시작 위치 찾기
        choice_start = -1
        for i, line in enumerate(lines):
            if self._CHOICE_PATTERN.search(line) or self._NUMBERED_CHOICE_PATTERN.match(
                line
            ):
                choice_start = i
                break

        if choice_start >= 0:
            # 선지 추출
            choice_lines = lines[choice_start:]
            choices = self._extract_choices(choice_lines)
            pre_choice = lines[:choice_start]
        else:
            choices = None
            pre_choice = lines

        # 지문과 문제 분리: 마지막 비어있지 않은 줄을 문제로 간주
        passage_lines = []
        question = ""

        # 뒤에서부터 문제 텍스트 찾기
        non_empty = [l for l in pre_choice if l.strip()]
        if non_empty:
            question = non_empty[-1]
            # 나머지는 지문
            found_question = False
            for line in reversed(pre_choice):
                if not found_question and line.strip() == question:
                    found_question = True
                    continue
                if found_question:
                    passage_lines.insert(0, line)

        passage = "\n".join(passage_lines).strip()
        return passage, question, choices

    def _extract_choices(self, lines: List[str]) -> List[str]:
        """선지 라인들에서 선지 텍스트를 추출.

        Args:
            lines: 선지가 포함된 라인 리스트

        Returns:
            List[str]: 추출된 선지 리스트
        """
        CIRCLES = "①②③④⑤"
        choices = []
        current = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 원문자 패턴으로 분리
            for i, circle in enumerate(CIRCLES):
                if circle in line:
                    if current:
                        choices.append(current.strip())
                    # 원문자 이후 텍스트
                    idx = line.index(circle)
                    current = line[idx + 1 :].strip()
                    break
            else:
                # 원문자가 없으면 현재 선지에 이어붙임
                if current:
                    current += " " + line
                else:
                    # 번호 패턴 (1), 2), ...) 시도
                    match = re.match(r"^\s*[1-5]\)\s*(.*)", line)
                    if match:
                        if current:
                            choices.append(current.strip())
                        current = match.group(1)
                    else:
                        current += " " + line if current else line

        if current:
            choices.append(current.strip())

        return choices if choices else None

    def parse(self, raw: dict) -> ExamItem:
        """SFT 레코드를 ExamItem으로 변환.

        Args:
            raw: 원시 데이터 dict

        Returns:
            ExamItem: 변환된 기출문제

        Raises:
            ValueError: question이 비어있는 경우
        """
        messages = raw.get("messages", [])

        # user 메시지 찾기
        user_msg = ""
        assistant_msg = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_msg = content
            elif role == "assistant":
                assistant_msg = content

        if not user_msg.strip():
            logger.warning("user 메시지가 비어있음")
            raise ValueError("question이 비어있습니다")

        passage, question, choices = self._split_user_message(user_msg)

        if not question.strip():
            # user_msg 전체를 question으로 사용
            question = user_msg.strip()
            passage = ""

        if not question.strip():
            logger.warning("question 추출 실패")
            raise ValueError("question이 비어있습니다")

        return ExamItem(
            passage=passage,
            question=question,
            choices=choices,
            answer=None,
            explanation=assistant_msg,
            metadata=ExamMetadata(
                exam_type="csat_sft",
                year=0,
                subject="국어",
                source_dataset="LLMin/final_csat_sft",
            ),
        )


class CivilParser(BaseParser):
    """kikikara 공무원 시험 포맷 파서.

    입력 필드: 문제 번호, 문제 내용, 선택지 1~4, 정답, year, task, source
    """

    def parse(self, raw: dict) -> ExamItem:
        """공무원 시험 레코드를 ExamItem으로 변환.

        Args:
            raw: 원시 데이터 dict

        Returns:
            ExamItem: 변환된 기출문제

        Raises:
            ValueError: question이 비어있는 경우
        """
        question = raw.get("문제 내용", "")
        if not question or not question.strip():
            logger.warning(
                f"question 비어있음: 문제 번호={raw.get('문제 번호', '?')}, "
                f"year={raw.get('year', '?')}"
            )
            raise ValueError("question이 비어있습니다")

        choices = [raw.get(f"선택지 {i}", "") for i in range(1, 5)]

        # 정답 처리 (값이 '[1]', '<NA>', '사' 등일 수 있음)
        answer_raw = raw.get("정답")
        answer = self._parse_answer(answer_raw)

        # exam_type 결정: source에 "국가직" 포함 여부
        source = raw.get("source", "")
        exam_type = "civil_national" if "국가직" in source else "civil_local"

        # 연도 처리
        year_raw = raw.get("year", 0)
        year = int(year_raw) if year_raw else 0

        # 문제 번호 처리
        q_num_raw = raw.get("문제 번호")
        question_number = int(q_num_raw) if q_num_raw is not None else None

        return ExamItem(
            passage="",
            question=question,
            choices=choices,
            answer=answer,
            explanation=None,
            metadata=ExamMetadata(
                exam_type=exam_type,
                year=year,
                subject="국어",
                question_number=question_number,
                source_dataset=raw.get("hf_id", "kikikara"),
            ),
        )

    @staticmethod
    def _parse_answer(value) -> Optional[int]:
        """정답 값을 int로 변환. '[1]', '<NA>' 등 비표준 형식 처리."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s == "<NA>":
            return None
        # '[1]' → '1' 형태 처리
        match = re.match(r"^\[?(\d+)\]?$", s)
        if match:
            return int(match.group(1))
        try:
            return int(s)
        except (ValueError, TypeError):
            return None