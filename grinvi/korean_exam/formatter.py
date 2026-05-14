"""ExamItem → TrainingRecord 변환 포매터."""

import logging
from typing import Optional

from grinvi.korean_exam.models import ExamItem

logger = logging.getLogger(__name__)


class Formatter:
    """ExamItem을 학습용 TrainingRecord dict로 변환한다."""

    MIN_TEXT_LENGTH = 50  # 이 미만이면 필터링

    def format(self, item: ExamItem) -> Optional[dict]:
        """ExamItem → TrainingRecord dict 변환.

        text 필드 길이가 MIN_TEXT_LENGTH 미만이면 None을 반환하고
        필터링 사실을 로그에 기록한다.
        """
        text = self._build_text(item)
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.info(
                "필터링: text 길이 %d자 < %d (question_number=%s, year=%d)",
                len(text),
                self.MIN_TEXT_LENGTH,
                item.metadata.question_number,
                item.metadata.year,
            )
            return None
        return {
            "text": text,
            "source": "korean_exam",
            "exam_type": item.metadata.exam_type,
            "year": item.metadata.year,
            "subject": item.metadata.subject,
            "question_number": item.metadata.question_number,
        }

    def _build_text(self, item: ExamItem) -> str:
        """[지문], [문제], 선지(①②③④⑤), [해설] 구조로 텍스트를 조합한다."""
        parts: list[str] = []
        if item.passage:
            parts.append(f"[지문]\n{item.passage}")
        parts.append(f"[문제]\n{item.question}")
        if item.choices:
            CIRCLE = "①②③④⑤"
            for i, choice in enumerate(item.choices):
                parts.append(f"{CIRCLE[i]} {choice}")
        if item.explanation:
            parts.append(f"[해설]\n{item.explanation}")
        return "\n\n".join(parts)
