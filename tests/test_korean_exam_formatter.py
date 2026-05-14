"""Unit tests for grinvi.korean_exam.formatter."""

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

# Load formatter module
_formatter_spec = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.formatter",
    Path(__file__).parent.parent / "grinvi" / "korean_exam" / "formatter.py",
)
_formatter_mod = importlib.util.module_from_spec(_formatter_spec)
sys.modules["grinvi.korean_exam.formatter"] = _formatter_mod
_formatter_spec.loader.exec_module(_formatter_mod)

ExamMetadata = _models.ExamMetadata
ExamItem = _models.ExamItem
Formatter = _formatter_mod.Formatter


def _make_item(
    question: str = "이 글의 주제로 가장 적절한 것은?",
    passage: str = "",
    choices=None,
    explanation=None,
    exam_type: str = "csat",
    year: int = 2024,
    subject: str = "국어",
    question_number=15,
) -> ExamItem:
    """Helper to create an ExamItem with configurable fields."""
    meta = ExamMetadata(
        exam_type=exam_type,
        year=year,
        subject=subject,
        question_number=question_number,
    )
    return ExamItem(
        passage=passage,
        question=question,
        choices=choices,
        answer=None,
        explanation=explanation,
        metadata=meta,
    )


class TestBuildText:
    def setup_method(self):
        self.f = Formatter()

    def test_question_only(self):
        item = _make_item(question="다음 중 올바른 것은?")
        text = self.f._build_text(item)
        assert text == "[문제]\n다음 중 올바른 것은?"

    def test_passage_and_question(self):
        item = _make_item(
            passage="한국의 역사는 오래되었다.",
            question="이 글의 주제는?",
        )
        text = self.f._build_text(item)
        assert "[지문]\n한국의 역사는 오래되었다." in text
        assert "[문제]\n이 글의 주제는?" in text
        # passage comes before question
        assert text.index("[지문]") < text.index("[문제]")

    def test_choices_with_circle_numbers(self):
        item = _make_item(
            question="다음 중 올바른 것은?",
            choices=["첫번째", "두번째", "세번째", "네번째", "다섯번째"],
        )
        text = self.f._build_text(item)
        assert "① 첫번째" in text
        assert "② 두번째" in text
        assert "③ 세번째" in text
        assert "④ 네번째" in text
        assert "⑤ 다섯번째" in text

    def test_four_choices(self):
        """4지선다 (공무원 시험 등)도 처리 가능."""
        item = _make_item(
            question="다음 중 올바른 것은?",
            choices=["가", "나", "다", "라"],
        )
        text = self.f._build_text(item)
        assert "① 가" in text
        assert "④ 라" in text
        assert "⑤" not in text

    def test_explanation_included(self):
        item = _make_item(
            question="다음 중 올바른 것은?",
            explanation="정답은 ③번이다. 왜냐하면...",
        )
        text = self.f._build_text(item)
        assert "[해설]\n정답은 ③번이다. 왜냐하면..." in text

    def test_full_item_structure(self):
        """passage + question + choices + explanation 전체 구조."""
        item = _make_item(
            passage="지문 내용입니다.",
            question="문제 내용입니다.",
            choices=["선지1", "선지2", "선지3", "선지4", "선지5"],
            explanation="해설 내용입니다.",
        )
        text = self.f._build_text(item)
        parts = text.split("\n\n")
        assert parts[0] == "[지문]\n지문 내용입니다."
        assert parts[1] == "[문제]\n문제 내용입니다."
        assert parts[2] == "① 선지1"
        assert parts[3] == "② 선지2"
        assert parts[4] == "③ 선지3"
        assert parts[5] == "④ 선지4"
        assert parts[6] == "⑤ 선지5"
        assert parts[7] == "[해설]\n해설 내용입니다."

    def test_empty_passage_not_included(self):
        """빈 문자열 passage는 [지문] 섹션을 생성하지 않는다."""
        item = _make_item(passage="", question="질문입니다.")
        text = self.f._build_text(item)
        assert "[지문]" not in text

    def test_none_choices_not_included(self):
        """choices가 None이면 선지를 생성하지 않는다."""
        item = _make_item(question="서술형 문제입니다.", choices=None)
        text = self.f._build_text(item)
        assert "①" not in text

    def test_none_explanation_not_included(self):
        """explanation이 None이면 [해설] 섹션을 생성하지 않는다."""
        item = _make_item(question="질문입니다.", explanation=None)
        text = self.f._build_text(item)
        assert "[해설]" not in text

    def test_parts_separated_by_double_newline(self):
        """각 파트는 \\n\\n으로 구분된다."""
        item = _make_item(
            passage="지문",
            question="문제",
            explanation="해설",
        )
        text = self.f._build_text(item)
        assert "\n\n" in text
        # No triple newlines
        assert "\n\n\n" not in text


class TestFormat:
    def setup_method(self):
        self.f = Formatter()

    def test_returns_dict_with_required_fields(self):
        item = _make_item(
            passage="긴 지문 내용이 여기에 들어갑니다. 충분히 길어야 합니다.",
            question="이 글의 주제로 가장 적절한 것은?",
            exam_type="csat",
            year=2022,
            subject="국어",
            question_number=15,
        )
        result = self.f.format(item)
        assert result is not None
        assert result["source"] == "korean_exam"
        assert result["exam_type"] == "csat"
        assert result["year"] == 2022
        assert result["subject"] == "국어"
        assert result["question_number"] == 15
        assert "text" in result

    def test_text_field_contains_passage(self):
        """요구사항 6.2: passage가 존재하면 text에 지문을 먼저 포함."""
        item = _make_item(
            passage="한국어 지문 내용이 여기에 들어갑니다. 충분히 길어야 합니다.",
            question="이 글의 주제로 가장 적절한 것은?",
        )
        result = self.f.format(item)
        assert result is not None
        assert "[지문]" in result["text"]
        assert "한국어 지문 내용" in result["text"]

    def test_text_field_contains_explanation(self):
        """요구사항 6.3: explanation이 존재하면 text 끝에 해설 포함."""
        item = _make_item(
            passage="긴 지문 내용이 여기에 들어갑니다. 충분히 길어야 합니다.",
            question="이 글의 주제로 가장 적절한 것은?",
            explanation="정답은 ③번이다.",
        )
        result = self.f.format(item)
        assert result is not None
        assert "[해설]" in result["text"]
        assert result["text"].endswith("정답은 ③번이다.")

    def test_filters_short_text(self):
        """요구사항 6.4: text 길이 50자 미만 필터링."""
        item = _make_item(question="짧은 질문")  # Very short
        result = self.f.format(item)
        assert result is None

    def test_exactly_50_chars_not_filtered(self):
        """정확히 50자인 text는 필터링되지 않는다."""
        # "[문제]\n" is 5 chars, so question needs to be 45 chars
        question = "가" * 45  # "[문제]\n" + 45 = 50 chars total
        item = _make_item(question=question)
        text = self.f._build_text(item)
        assert len(text) == 50
        result = self.f.format(item)
        assert result is not None

    def test_49_chars_filtered(self):
        """49자인 text는 필터링된다."""
        question = "가" * 44  # "[문제]\n" + 44 = 49 chars total
        item = _make_item(question=question)
        text = self.f._build_text(item)
        assert len(text) == 49
        result = self.f.format(item)
        assert result is None

    def test_source_always_korean_exam(self):
        """요구사항 6.1: source는 항상 'korean_exam'."""
        item = _make_item(
            passage="충분히 긴 지문 내용이 여기에 들어갑니다.",
            question="이 글의 주제로 가장 적절한 것은?",
        )
        result = self.f.format(item)
        assert result is not None
        assert result["source"] == "korean_exam"

    def test_metadata_fields_mapped_correctly(self):
        """요구사항 6.1: exam_type, year, subject 필드 매핑."""
        item = _make_item(
            passage="충분히 긴 지문 내용이 여기에 들어갑니다.",
            question="이 글의 주제로 가장 적절한 것은?",
            exam_type="mock_june",
            year=2023,
            subject="언어",
            question_number=3,
        )
        result = self.f.format(item)
        assert result is not None
        assert result["exam_type"] == "mock_june"
        assert result["year"] == 2023
        assert result["subject"] == "언어"
        assert result["question_number"] == 3

    def test_question_number_none(self):
        """question_number가 None인 경우도 정상 처리."""
        item = _make_item(
            passage="충분히 긴 지문 내용이 여기에 들어갑니다.",
            question="이 글의 주제로 가장 적절한 것은?",
            question_number=None,
        )
        result = self.f.format(item)
        assert result is not None
        assert result["question_number"] is None

    def test_filters_logs_message(self, caplog):
        """요구사항 6.4: 필터링 시 로그에 기록."""
        import logging

        with caplog.at_level(logging.INFO):
            item = _make_item(question="짧음")
            self.f.format(item)
        assert "필터링" in caplog.text
