"""Unit tests for grinvi.korean_exam.validator."""

import sys
import importlib.util
from pathlib import Path

# Load models directly to avoid torch dependency in grinvi/__init__.py
_models_spec = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.models",
    Path(__file__).parent.parent / "grinvi" / "korean_exam" / "models.py",
)
_models = importlib.util.module_from_spec(_models_spec)
sys.modules["grinvi.korean_exam.models"] = _models
_models_spec.loader.exec_module(_models)

# Load validator module
_validator_spec = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.validator",
    Path(__file__).parent.parent / "grinvi" / "korean_exam" / "validator.py",
)
_validator_mod = importlib.util.module_from_spec(_validator_spec)
sys.modules["grinvi.korean_exam.validator"] = _validator_mod
_validator_spec.loader.exec_module(_validator_mod)

ExamMetadata = _models.ExamMetadata
ExamItem = _models.ExamItem
Validator = _validator_mod.Validator


def _make_item(question: str, passage: str = "") -> ExamItem:
    """Helper to create an ExamItem with minimal metadata."""
    meta = ExamMetadata(exam_type="csat", year=2024)
    return ExamItem(
        passage=passage,
        question=question,
        choices=None,
        answer=None,
        explanation=None,
        metadata=meta,
    )


class TestKoreanRatio:
    def setup_method(self):
        self.v = Validator()

    def test_empty_string_returns_zero(self):
        assert self.v.korean_ratio("") == 0.0

    def test_all_korean_returns_one(self):
        assert self.v.korean_ratio("가나다라마바사") == 1.0

    def test_no_korean_returns_zero(self):
        assert self.v.korean_ratio("abcdefgh") == 0.0

    def test_mixed_text(self):
        # 4 Korean out of 8 total = 0.5
        assert self.v.korean_ratio("가나다라abcd") == 0.5

    def test_spaces_are_non_korean(self):
        # "가 나" = 2 Korean out of 3 chars
        ratio = self.v.korean_ratio("가 나")
        assert abs(ratio - 2 / 3) < 1e-9

    def test_numbers_are_non_korean(self):
        # "가나123" = 2 Korean out of 5 chars = 0.4
        assert self.v.korean_ratio("가나123") == 0.4

    def test_single_korean_char(self):
        assert self.v.korean_ratio("한") == 1.0

    def test_single_non_korean_char(self):
        assert self.v.korean_ratio("x") == 0.0


class TestValidate:
    def setup_method(self):
        self.v = Validator()

    def test_valid_korean_question(self):
        item = _make_item("이 글의 주제로 가장 적절한 것은?")
        assert self.v.validate(item) is True

    def test_empty_question_is_invalid(self):
        item = _make_item("")
        assert self.v.validate(item) is False

    def test_whitespace_only_question_is_invalid(self):
        item = _make_item("   \t\n  ")
        assert self.v.validate(item) is False

    def test_english_only_question_is_invalid(self):
        item = _make_item("What is the main idea of this passage?")
        assert self.v.validate(item) is False

    def test_exactly_50_percent_korean_is_valid(self):
        # 4 Korean / 8 total = 50%
        item = _make_item("가나다라abcd")
        assert self.v.validate(item) is True

    def test_below_50_percent_korean_is_invalid(self):
        # 3 Korean / 8 total = 37.5%
        item = _make_item("가나다abcde")
        assert self.v.validate(item) is False

    def test_passage_contributes_to_ratio(self):
        # question alone: "abc" = 0% Korean → invalid
        # but passage "가나다라마바사" + question "abc" = 7/10 = 70% → valid
        item = _make_item("abc", passage="가나다라마바사")
        assert self.v.validate(item) is True

    def test_passage_can_dilute_ratio(self):
        # question "가나" = 100% Korean alone
        # but passage "abcdefgh" + question "가나" = 2/10 = 20% → invalid
        item = _make_item("가나", passage="abcdefgh")
        assert self.v.validate(item) is False

    def test_empty_passage_uses_question_only(self):
        item = _make_item("한국어문장입니다", passage="")
        assert self.v.validate(item) is True


class TestDeduplicate:
    def setup_method(self):
        self.v = Validator()

    def test_empty_list(self):
        assert self.v.deduplicate([]) == []

    def test_no_duplicates(self):
        items = [
            _make_item("질문 하나"),
            _make_item("질문 둘"),
            _make_item("질문 셋"),
        ]
        result = self.v.deduplicate(items)
        assert len(result) == 3

    def test_removes_exact_duplicates(self):
        items = [
            _make_item("동일한 질문"),
            _make_item("동일한 질문"),
            _make_item("다른 질문"),
        ]
        result = self.v.deduplicate(items)
        assert len(result) == 2
        assert result[0].question == "동일한 질문"
        assert result[1].question == "다른 질문"

    def test_strips_whitespace_for_comparison(self):
        items = [
            _make_item("  질문  "),
            _make_item("질문"),
            _make_item("질문\n"),
        ]
        result = self.v.deduplicate(items)
        assert len(result) == 1
        # First item is kept
        assert result[0].question == "  질문  "

    def test_preserves_order(self):
        items = [
            _make_item("첫번째"),
            _make_item("두번째"),
            _make_item("첫번째"),
            _make_item("세번째"),
            _make_item("두번째"),
        ]
        result = self.v.deduplicate(items)
        assert len(result) == 3
        assert result[0].question == "첫번째"
        assert result[1].question == "두번째"
        assert result[2].question == "세번째"

    def test_single_item(self):
        items = [_make_item("유일한 질문")]
        result = self.v.deduplicate(items)
        assert len(result) == 1

    def test_all_duplicates(self):
        items = [_make_item("같은 질문")] * 5
        result = self.v.deduplicate(items)
        assert len(result) == 1

    def test_idempotent(self):
        """deduplicate(deduplicate(items)) == deduplicate(items)."""
        items = [
            _make_item("질문 하나"),
            _make_item("질문 둘"),
            _make_item("질문 하나"),
            _make_item("질문 셋"),
            _make_item("질문 둘"),
        ]
        once = self.v.deduplicate(items)
        twice = self.v.deduplicate(once)
        assert once == twice
