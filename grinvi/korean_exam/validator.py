"""데이터 품질 검증 모듈."""

from typing import List

from grinvi.korean_exam.models import ExamItem


class Validator:
    """기출문제 데이터 품질 검증기."""

    def korean_ratio(self, text: str) -> float:
        """한국어 문자(가-힣) 비율 계산.

        Args:
            text: 검사할 텍스트.

        Returns:
            한국어 문자 비율 (0.0 ~ 1.0). 빈 문자열이면 0.0.
        """
        if not text:
            return 0.0
        korean = sum(1 for c in text if "가" <= c <= "힣")
        return korean / len(text)

    def validate(self, item: ExamItem) -> bool:
        """ExamItem의 유효성 검사.

        question이 비어있지 않고, passage + question의 한국어 비율이 50% 이상이면 유효.

        Args:
            item: 검사할 ExamItem.

        Returns:
            유효하면 True, 아니면 False.
        """
        if not item.question.strip():
            return False
        full_text = (item.passage or "") + item.question
        return self.korean_ratio(full_text) >= 0.5

    def deduplicate(self, items: List[ExamItem]) -> List[ExamItem]:
        """question 텍스트 기준 중복 제거.

        동일한 question(strip 후 비교)을 가진 항목 중 첫 번째만 유지.

        Args:
            items: ExamItem 리스트.

        Returns:
            중복이 제거된 ExamItem 리스트.
        """
        seen: set = set()
        result: List[ExamItem] = []
        for item in items:
            key = item.question.strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
