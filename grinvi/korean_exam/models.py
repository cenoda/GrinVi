"""ExamItem, ExamMetadata 데이터 모델."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExamMetadata:
    """기출문제 메타데이터."""

    exam_type: str  # "csat", "mock_june", "mock_sept", "teacher",
    # "leet", "district", "civil_local", "civil_national"
    year: int  # 시행 연도 (예: 2024)
    month: Optional[int] = None  # 시행 월 (수능=11, 6월모의=6, ...)
    question_number: Optional[int] = None  # 문제 번호
    subject: str = "국어"  # "국어", "언어", "언어이해" 등
    source_dataset: Optional[str] = None  # HF 데이터셋 ID (HF 수집 시)
    source_file: Optional[str] = None  # 원본 파일 경로 (PDF 수집 시)


@dataclass
class ExamItem:
    """단일 기출문제 데이터 구조."""

    passage: str  # 지문 (없으면 빈 문자열)
    question: str  # 문제 텍스트 (필수)
    choices: Optional[List[str]]  # 선지 목록 (없으면 None)
    answer: Optional[int]  # 정답 번호 1-based (없으면 None)
    explanation: Optional[str]  # 해설 (없으면 None)
    metadata: ExamMetadata

    def is_valid(self) -> bool:
        """question이 비어있지 않고 한국어 비율 50% 이상인지 검사."""
        if not self.question.strip():
            return False
        return _korean_ratio(self.question) >= 0.5


def _korean_ratio(text: str) -> float:
    """한국어 문자(가-힣) 비율 계산."""
    if not text:
        return 0.0
    korean_count = sum(1 for c in text if "가" <= c <= "힣")
    return korean_count / len(text)
