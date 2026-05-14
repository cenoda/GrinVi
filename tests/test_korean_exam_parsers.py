"""Unit tests for grinvi.korean_exam.parsers (BaseParser, HF parsers)."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Load modules directly to avoid torch dependency in grinvi/__init__.py
_base_dir = Path(__file__).parent.parent / "grinvi" / "korean_exam"

# Load models first (dependency)
_spec_models = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.models",
    _base_dir / "models.py",
)
_models = importlib.util.module_from_spec(_spec_models)
sys.modules["grinvi.korean_exam.models"] = _models
_spec_models.loader.exec_module(_models)

# Ensure the package path is set for relative imports
sys.modules["grinvi"] = type(sys)("grinvi")
sys.modules["grinvi"].korean_exam = type(sys)("grinvi.korean_exam")
sys.modules["grinvi.korean_exam"] = sys.modules["grinvi"].korean_exam
sys.modules["grinvi.korean_exam"].models = _models

# Load parsers.base
_spec_base = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.parsers.base",
    _base_dir / "parsers" / "base.py",
)
_base = importlib.util.module_from_spec(_spec_base)
sys.modules["grinvi.korean_exam.parsers.base"] = _base
_spec_base.loader.exec_module(_base)

# Load parsers.hf_parsers
_spec_hf = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.parsers.hf_parsers",
    _base_dir / "parsers" / "hf_parsers.py",
)
_hf = importlib.util.module_from_spec(_spec_hf)
sys.modules["grinvi.korean_exam.parsers.hf_parsers"] = _hf
_spec_hf.loader.exec_module(_hf)

BaseParser = _base.BaseParser
ExamItem = _models.ExamItem
ExamMetadata = _models.ExamMetadata
CSATQAParser = _hf.CSATQAParser
CSATKKACHIParser = _hf.CSATKKACHIParser
SFTParser = _hf.SFTParser
CivilParser = _hf.CivilParser


class TestBaseParser:
    def test_is_abstract(self):
        """BaseParser는 직접 인스턴스화할 수 없다."""
        with pytest.raises(TypeError):
            BaseParser()

    def test_parse_file_reads_jsonl(self, tmp_path):
        """parse_file이 JSONL 파일을 올바르게 읽는다."""

        class SimpleParser(BaseParser):
            def parse(self, raw: dict) -> ExamItem:
                return ExamItem(
                    passage="",
                    question=raw["q"],
                    choices=None,
                    answer=None,
                    explanation=None,
                    metadata=ExamMetadata(exam_type="test", year=2024),
                )

        parser = SimpleParser()
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"q": "문제 하나"}\n{"q": "문제 둘"}\n', encoding="utf-8"
        )

        items = parser.parse_file(jsonl_file)
        assert len(items) == 2
        assert items[0].question == "문제 하나"
        assert items[1].question == "문제 둘"

    def test_parse_file_skips_invalid_lines(self, tmp_path):
        """parse_file이 잘못된 줄을 건너뛴다."""

        class SimpleParser(BaseParser):
            def parse(self, raw: dict) -> ExamItem:
                if "q" not in raw:
                    raise KeyError("q")
                return ExamItem(
                    passage="",
                    question=raw["q"],
                    choices=None,
                    answer=None,
                    explanation=None,
                    metadata=ExamMetadata(exam_type="test", year=2024),
                )

        parser = SimpleParser()
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"q": "유효"}\n{invalid json}\n{"no_q": true}\n{"q": "유효2"}\n',
            encoding="utf-8",
        )

        items = parser.parse_file(jsonl_file)
        assert len(items) == 2
        assert items[0].question == "유효"
        assert items[1].question == "유효2"

    def test_parse_file_skips_empty_lines(self, tmp_path):
        """parse_file이 빈 줄을 건너뛴다."""

        class SimpleParser(BaseParser):
            def parse(self, raw: dict) -> ExamItem:
                return ExamItem(
                    passage="",
                    question=raw["q"],
                    choices=None,
                    answer=None,
                    explanation=None,
                    metadata=ExamMetadata(exam_type="test", year=2024),
                )

        parser = SimpleParser()
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"q": "문제"}\n\n\n{"q": "문제2"}\n', encoding="utf-8"
        )

        items = parser.parse_file(jsonl_file)
        assert len(items) == 2


class TestCSATQAParser:
    def setup_method(self):
        self.parser = CSATQAParser()

    def test_parse_basic(self):
        """기본 csatqa 레코드 파싱."""
        raw = {
            "test_name": "2022년 2023 대학수학능력시험 국어  홀수형",
            "context": "다음 글을 읽고 물음에 답하시오.",
            "question": "윗글의 내용과 일치하는 것은?",
            "gold": 3,
            "option#1": "첫 번째 선지",
            "option#2": "두 번째 선지",
            "option#3": "세 번째 선지",
            "option#4": "네 번째 선지",
            "option#5": "다섯 번째 선지",
        }
        item = self.parser.parse(raw)

        assert item.passage == "다음 글을 읽고 물음에 답하시오."
        assert item.question == "윗글의 내용과 일치하는 것은?"
        assert item.answer == 3
        assert len(item.choices) == 5
        assert item.choices[0] == "첫 번째 선지"
        assert item.metadata.exam_type == "csat"
        assert item.metadata.year == 2023
        assert item.metadata.month == 11

    def test_parse_metadata_csat(self):
        """수능 시험 메타데이터 추출."""
        meta = self.parser._parse_metadata("2022년 2023 대학수학능력시험 국어  홀수형")
        assert meta.exam_type == "csat"
        assert meta.year == 2023
        assert meta.month == 11

    def test_parse_metadata_mock_june(self):
        """6월 모의고사 메타데이터 추출."""
        meta = self.parser._parse_metadata("2022년 6월 모의고사 국어")
        assert meta.exam_type == "mock_june"
        assert meta.year == 2022
        assert meta.month == 6

    def test_parse_metadata_mock_sept(self):
        """9월 모의고사 메타데이터 추출."""
        meta = self.parser._parse_metadata("2023년 9월 모의고사 국어")
        assert meta.exam_type == "mock_sept"
        assert meta.year == 2023
        assert meta.month == 9

    def test_parse_metadata_fallback(self):
        """패턴 매칭 실패 시 연도만 추출."""
        meta = self.parser._parse_metadata("2020년 기타 시험")
        assert meta.year == 2020
        assert meta.exam_type == "csat"

    def test_parse_empty_question_raises(self):
        """question이 비어있으면 ValueError."""
        raw = {
            "test_name": "2023 대학수학능력시험 국어",
            "context": "지문",
            "question": "",
            "gold": 1,
            "option#1": "a",
            "option#2": "b",
            "option#3": "c",
            "option#4": "d",
            "option#5": "e",
        }
        with pytest.raises(ValueError):
            self.parser.parse(raw)

    def test_parse_no_context(self):
        """context가 없는 경우 빈 문자열."""
        raw = {
            "test_name": "2023 대학수학능력시험 국어",
            "question": "독립 문제입니다",
            "gold": 2,
            "option#1": "a",
            "option#2": "b",
            "option#3": "c",
            "option#4": "d",
            "option#5": "e",
        }
        item = self.parser.parse(raw)
        assert item.passage == ""

    def test_parse_no_gold(self):
        """gold가 없는 경우 answer=None."""
        raw = {
            "test_name": "2023 대학수학능력시험 국어",
            "context": "지문",
            "question": "문제 텍스트",
            "option#1": "a",
            "option#2": "b",
            "option#3": "c",
            "option#4": "d",
            "option#5": "e",
        }
        item = self.parser.parse(raw)
        assert item.answer is None


class TestCSATKKACHIParser:
    def setup_method(self):
        self.parser = CSATKKACHIParser()

    def test_parse_basic(self):
        """기본 KKACHI 레코드 파싱."""
        raw = {
            "idx": 1,
            "paragraph": "지문 내용입니다.",
            "question": "다음 중 적절한 것은?",
            "question_plus": "추가 조건입니다.",
            "A": "선지 A",
            "B": "선지 B",
            "C": "선지 C",
            "D": "선지 D",
            "E": "선지 E",
            "answer": 3,
            "point": 2,
        }
        item = self.parser.parse(raw)

        assert item.passage == "지문 내용입니다."
        assert "다음 중 적절한 것은?" in item.question
        assert "추가 조건입니다." in item.question
        assert len(item.choices) == 5
        assert item.choices[0] == "선지 A"
        assert item.answer == 3
        assert item.metadata.exam_type == "csat"
        assert item.metadata.year == 2025
        assert item.metadata.source_dataset == "KKACHI-HUB/CSAT-KOREAN-2025"

    def test_parse_no_question_plus(self):
        """question_plus가 없는 경우."""
        raw = {
            "idx": 2,
            "paragraph": "지문",
            "question": "문제 텍스트",
            "question_plus": "",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "E": "e",
            "answer": 1,
            "point": 3,
        }
        item = self.parser.parse(raw)
        assert item.question == "문제 텍스트"

    def test_parse_empty_question_raises(self):
        """question이 비어있으면 ValueError."""
        raw = {
            "idx": 3,
            "paragraph": "지문",
            "question": "",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "E": "e",
            "answer": 1,
        }
        with pytest.raises(ValueError):
            self.parser.parse(raw)

    def test_parse_no_paragraph(self):
        """paragraph가 없는 경우 빈 문자열."""
        raw = {
            "idx": 4,
            "question": "독립 문제",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "E": "e",
            "answer": 2,
        }
        item = self.parser.parse(raw)
        assert item.passage == ""


class TestSFTParser:
    def setup_method(self):
        self.parser = SFTParser()

    def test_parse_basic(self):
        """기본 SFT 레코드 파싱."""
        raw = {
            "messages": [
                {"role": "system", "content": "당신은 수능 국어 전문가입니다."},
                {
                    "role": "user",
                    "content": "다음 글을 읽고 물음에 답하시오.\n\n지문 내용입니다.\n\n윗글의 주제로 적절한 것은?\n①첫째\n②둘째\n③셋째\n④넷째\n⑤다섯째",
                },
                {"role": "assistant", "content": "정답은 3번입니다. 해설..."},
            ]
        }
        item = self.parser.parse(raw)

        assert item.explanation == "정답은 3번입니다. 해설..."
        assert item.metadata.exam_type == "csat_sft"
        assert item.metadata.source_dataset == "LLMin/final_csat_sft"
        # question should be extracted
        assert item.question != ""

    def test_parse_no_assistant(self):
        """assistant 메시지가 없는 경우."""
        raw = {
            "messages": [
                {"role": "user", "content": "문제 텍스트입니다."},
            ]
        }
        item = self.parser.parse(raw)
        assert item.explanation is None
        assert item.question != ""

    def test_parse_empty_user_raises(self):
        """user 메시지가 비어있으면 ValueError."""
        raw = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "답변"},
            ]
        }
        with pytest.raises(ValueError):
            self.parser.parse(raw)

    def test_parse_no_messages_raises(self):
        """messages가 없으면 ValueError."""
        raw = {"messages": []}
        with pytest.raises(ValueError):
            self.parser.parse(raw)

    def test_parse_with_choices(self):
        """선지가 포함된 메시지 파싱."""
        raw = {
            "messages": [
                {
                    "role": "user",
                    "content": "문제입니다.\n①선지1\n②선지2\n③선지3\n④선지4\n⑤선지5",
                },
            ]
        }
        item = self.parser.parse(raw)
        assert item.choices is not None
        assert len(item.choices) == 5


class TestCivilParser:
    def setup_method(self):
        self.parser = CivilParser()

    def test_parse_basic(self):
        """기본 공무원 시험 레코드 파싱."""
        raw = {
            "문제 번호": 5,
            "문제 내용": "다음 중 맞춤법이 올바른 것은?",
            "선택지 1": "첫 번째",
            "선택지 2": "두 번째",
            "선택지 3": "세 번째",
            "선택지 4": "네 번째",
            "정답": 2,
            "year": 2023,
            "task": "국어",
            "source": "지방직",
        }
        item = self.parser.parse(raw)

        assert item.question == "다음 중 맞춤법이 올바른 것은?"
        assert len(item.choices) == 4
        assert item.choices[0] == "첫 번째"
        assert item.answer == 2
        assert item.metadata.exam_type == "civil_local"
        assert item.metadata.year == 2023
        assert item.metadata.question_number == 5

    def test_parse_national(self):
        """국가직 공무원 시험 레코드 파싱."""
        raw = {
            "문제 번호": 1,
            "문제 내용": "국어 문제입니다",
            "선택지 1": "a",
            "선택지 2": "b",
            "선택지 3": "c",
            "선택지 4": "d",
            "정답": 1,
            "year": 2024,
            "task": "국어",
            "source": "국가직",
        }
        item = self.parser.parse(raw)
        assert item.metadata.exam_type == "civil_national"

    def test_parse_empty_question_raises(self):
        """문제 내용이 비어있으면 ValueError."""
        raw = {
            "문제 번호": 1,
            "문제 내용": "",
            "선택지 1": "a",
            "선택지 2": "b",
            "선택지 3": "c",
            "선택지 4": "d",
            "정답": 1,
            "year": 2024,
            "task": "국어",
            "source": "지방직",
        }
        with pytest.raises(ValueError):
            self.parser.parse(raw)

    def test_parse_no_answer(self):
        """정답이 없는 경우 answer=None."""
        raw = {
            "문제 번호": 1,
            "문제 내용": "문제 텍스트입니다",
            "선택지 1": "a",
            "선택지 2": "b",
            "선택지 3": "c",
            "선택지 4": "d",
            "year": 2022,
            "task": "국어",
            "source": "지방직",
        }
        item = self.parser.parse(raw)
        assert item.answer is None

    def test_parse_no_year(self):
        """year가 없는 경우 0."""
        raw = {
            "문제 번호": 1,
            "문제 내용": "문제 텍스트입니다",
            "선택지 1": "a",
            "선택지 2": "b",
            "선택지 3": "c",
            "선택지 4": "d",
            "정답": 3,
            "task": "국어",
            "source": "지방직",
        }
        item = self.parser.parse(raw)
        assert item.metadata.year == 0

    def test_deterministic_parsing(self):
        """동일한 입력을 두 번 파싱하면 동일한 결과."""
        raw = {
            "문제 번호": 10,
            "문제 내용": "다음 중 올바른 것은?",
            "선택지 1": "가",
            "선택지 2": "나",
            "선택지 3": "다",
            "선택지 4": "라",
            "정답": 2,
            "year": 2021,
            "task": "국어",
            "source": "국가직",
        }
        item1 = self.parser.parse(raw)
        item2 = self.parser.parse(raw)
        assert item1 == item2
