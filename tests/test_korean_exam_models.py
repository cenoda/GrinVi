"""Unit tests for grinvi.korean_exam.models."""

import sys
import importlib.util
from pathlib import Path

# Load models directly to avoid torch dependency in grinvi/__init__.py
_spec = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.models",
    Path(__file__).parent.parent / "grinvi" / "korean_exam" / "models.py",
)
_models = importlib.util.module_from_spec(_spec)
sys.modules["grinvi.korean_exam.models"] = _models
_spec.loader.exec_module(_models)

ExamMetadata = _models.ExamMetadata
ExamItem = _models.ExamItem
_korean_ratio = _models._korean_ratio


class TestExamMetadata:
    def test_required_fields(self):
        meta = ExamMetadata(exam_type="csat", year=2024)
        assert meta.exam_type == "csat"
        assert meta.year == 2024
        assert meta.subject == "국어"

    def test_optional_fields_default_none(self):
        meta = ExamMetadata(exam_type="leet", year=2020)
        assert meta.month is None
        assert meta.question_number is None
        assert meta.source_dataset is None
        assert meta.source_file is None

    def test_all_fields(self):
        meta = ExamMetadata(
            exam_type="mock_june",
            year=2023,
            month=6,
            question_number=15,
            subject="국어",
            source_dataset="HAERAE-HUB/csatqa",
            source_file=None,
        )
        assert meta.exam_type == "mock_june"
        assert meta.year == 2023
        assert meta.month == 6
        assert meta.question_number == 15
        assert meta.source_dataset == "HAERAE-HUB/csatqa"


class TestKoreanRatio:
    def test_empty_string(self):
        assert _korean_ratio("") == 0.0

    def test_all_korean(self):
        assert _korean_ratio("가나다라마") == 1.0

    def test_no_korean(self):
        assert _korean_ratio("abcdef") == 0.0

    def test_mixed(self):
        # 4 Korean out of 8 chars = 0.5
        assert _korean_ratio("가나다라abcd") == 0.5

    def test_spaces_count_as_non_korean(self):
        # "가 나" = 2 Korean out of 3 chars
        ratio = _korean_ratio("가 나")
        assert abs(ratio - 2 / 3) < 1e-9


class TestExamItemIsValid:
    def _make_item(self, question: str) -> ExamItem:
        meta = ExamMetadata(exam_type="csat", year=2024)
        return ExamItem(
            passage="",
            question=question,
            choices=None,
            answer=None,
            explanation=None,
            metadata=meta,
        )

    def test_valid_korean_question(self):
        item = self._make_item("이 글의 주제로 가장 적절한 것은?")
        assert item.is_valid() is True

    def test_empty_question(self):
        item = self._make_item("")
        assert item.is_valid() is False

    def test_whitespace_only_question(self):
        item = self._make_item("   \t\n  ")
        assert item.is_valid() is False

    def test_english_only_question(self):
        item = self._make_item("What is the main idea of this passage?")
        assert item.is_valid() is False

    def test_exactly_50_percent_korean(self):
        # 4 Korean / 8 total = 50%
        item = self._make_item("가나다라abcd")
        assert item.is_valid() is True

    def test_below_50_percent_korean(self):
        # 3 Korean / 8 total = 37.5%
        item = self._make_item("가나다abcde")
        assert item.is_valid() is False

    def test_full_exam_item_with_all_fields(self):
        meta = ExamMetadata(
            exam_type="csat",
            year=2022,
            month=11,
            question_number=15,
            subject="국어",
            source_dataset="HAERAE-HUB/csatqa",
        )
        item = ExamItem(
            passage="다음 글을 읽고 물음에 답하시오.",
            question="윗글에 대한 설명으로 적절하지 않은 것은?",
            choices=["첫째", "둘째", "셋째", "넷째", "다섯째"],
            answer=3,
            explanation="정답은 3번입니다.",
            metadata=meta,
        )
        assert item.is_valid() is True
        assert item.answer == 3
        assert len(item.choices) == 5
