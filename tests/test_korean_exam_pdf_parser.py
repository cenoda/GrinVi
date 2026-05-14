"""Unit tests for grinvi.korean_exam.parsers.pdf_parser."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

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
sys.modules.setdefault("grinvi", type(sys)("grinvi"))
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

# Load parsers.pdf_parser
_spec_pdf = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.parsers.pdf_parser",
    _base_dir / "parsers" / "pdf_parser.py",
)
_pdf = importlib.util.module_from_spec(_spec_pdf)
sys.modules["grinvi.korean_exam.parsers.pdf_parser"] = _pdf
_spec_pdf.loader.exec_module(_pdf)

PDFParser = _pdf.PDFParser
CSATPDFParser = _pdf.CSATPDFParser
TeacherPDFParser = _pdf.TeacherPDFParser
LEETPDFParser = _pdf.LEETPDFParser
ExamItem = _models.ExamItem
ExamMetadata = _models.ExamMetadata


class TestPDFParser:
    def test_remove_noise_page_numbers(self):
        """페이지 번호를 제거한다."""
        parser = PDFParser()
        text = "문제 내용입니다.\n  3  \n다음 줄입니다."
        cleaned = parser._remove_noise(text)
        assert "  3  " not in cleaned
        assert "문제 내용입니다." in cleaned

    def test_remove_noise_header(self):
        """시험지 헤더를 제거한다."""
        parser = PDFParser()
        text = "대학수학능력시험 국어 홀수형\n1. 문제입니다."
        cleaned = parser._remove_noise(text)
        assert "대학수학능력시험" not in cleaned
        assert "1. 문제입니다." in cleaned

    def test_remove_noise_copyright(self):
        """저작권 표시를 제거한다."""
        parser = PDFParser()
        text = "문제 내용\n이 문제지에 관한 저작권은\n다음 문제"
        cleaned = parser._remove_noise(text)
        assert "이 문제지에 관한" not in cleaned

    def test_remove_noise_page_separator(self):
        """페이지 구분선을 제거한다."""
        parser = PDFParser()
        text = "문제 내용\n  - 5 -  \n다음 내용"
        cleaned = parser._remove_noise(text)
        assert "- 5 -" not in cleaned

    def test_remove_noise_preserves_content(self):
        """문제 내용은 보존한다."""
        parser = PDFParser()
        text = "1. 다음 글을 읽고 물음에 답하시오.\n지문 내용입니다."
        cleaned = parser._remove_noise(text)
        assert "1. 다음 글을 읽고 물음에 답하시오." in cleaned
        assert "지문 내용입니다." in cleaned

    def test_select_sub_parser_teacher(self):
        """teacher 경로면 TeacherPDFParser를 선택."""
        parser = PDFParser()
        path = Path("data/raw/korean_exam/pdf/teacher/2024/test.pdf")
        sub = parser._select_sub_parser(path, [])
        assert isinstance(sub, TeacherPDFParser)

    def test_select_sub_parser_leet(self):
        """leet 경로면 LEETPDFParser를 선택."""
        parser = PDFParser()
        path = Path("data/raw/korean_exam/pdf/leet/2024/test.pdf")
        sub = parser._select_sub_parser(path, [])
        assert isinstance(sub, LEETPDFParser)

    def test_select_sub_parser_default(self):
        """기본 경로면 CSATPDFParser를 선택."""
        parser = PDFParser()
        path = Path("data/raw/korean_exam/pdf/csat/2024/test.pdf")
        sub = parser._select_sub_parser(path, [])
        assert isinstance(sub, CSATPDFParser)

    def test_parse_file_no_pdfplumber(self, tmp_path):
        """pdfplumber가 없으면 빈 리스트 반환."""
        parser = PDFParser()
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4")

        # Simulate pdfplumber not being installed
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("No module named 'pdfplumber'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            items = parser.parse_file(path)
            assert items == []

    def test_extract_metadata_from_path_csat(self):
        """경로에서 csat 메타데이터를 추출."""
        parser = PDFParser()
        path = Path("data/raw/korean_exam/pdf/csat/2024/test.pdf")
        meta = parser._extract_metadata_from_path(path)
        assert meta.exam_type == "csat"
        assert meta.year == 2024
        assert meta.subject == "국어"

    def test_extract_metadata_from_path_teacher(self):
        """경로에서 teacher 메타데이터를 추출."""
        parser = PDFParser()
        path = Path("data/raw/korean_exam/pdf/teacher/2022/exam.pdf")
        meta = parser._extract_metadata_from_path(path)
        assert meta.exam_type == "teacher"
        assert meta.year == 2022


class TestCSATPDFParser:
    def test_split_by_questions(self):
        """문항 번호로 텍스트를 분할."""
        parser = CSATPDFParser()
        text = "1. 첫 번째 문제입니다.\n내용\n2. 두 번째 문제입니다.\n내용"
        blocks = parser._split_by_questions(text)
        assert len(blocks) == 2
        assert blocks[0][0] == 1
        assert blocks[1][0] == 2

    def test_extract_choices_5(self):
        """5지선다 선지를 추출."""
        parser = CSATPDFParser()
        text = "문제 내용 ①첫째 ②둘째 ③셋째 ④넷째 ⑤다섯째"
        choices = parser._extract_choices(text, num_choices=5)
        assert choices is not None
        assert len(choices) == 5
        assert choices[0] == "첫째"
        assert choices[4] == "다섯째"

    def test_extract_choices_incomplete(self):
        """선지가 불완전하면 None 반환."""
        parser = CSATPDFParser()
        text = "문제 내용 ①첫째 ②둘째 ③셋째"
        choices = parser._extract_choices(text, num_choices=5)
        assert choices is None

    def test_split_passage_question(self):
        """지문과 문제를 분리."""
        parser = CSATPDFParser()
        text = "지문 내용입니다.\n여러 줄의 지문.\n윗글의 내용과 일치하는 것은?"
        passage, question = parser._split_passage_question(text)
        assert "지문 내용입니다." in passage
        assert "일치하는 것은?" in question

    def test_split_passage_question_no_passage(self):
        """지문 없이 문제만 있는 경우."""
        parser = CSATPDFParser()
        text = "다음 중 올바른 것은?"
        passage, question = parser._split_passage_question(text)
        assert passage == ""
        assert "올바른 것은?" in question

    def test_parse_pages_basic(self):
        """기본 페이지 파싱."""
        parser = CSATPDFParser()
        pages = [
            "1. 다음 글을 읽고 물음에 답하시오.\n"
            "지문 내용입니다.\n"
            "윗글의 내용과 일치하는 것은?\n"
            "①첫째 ②둘째 ③셋째 ④넷째 ⑤다섯째"
        ]
        path = Path("data/raw/korean_exam/pdf/csat/2024/test.pdf")
        items = parser._parse_pages(pages, path)
        assert len(items) >= 1
        assert items[0].metadata.exam_type == "csat"
        assert items[0].choices is not None


class TestTeacherPDFParser:
    def test_extract_choices_4(self):
        """4지선다 선지를 추출."""
        parser = TeacherPDFParser()
        text = "문제 내용 ①첫째 ②둘째 ③셋째 ④넷째"
        choices = parser._extract_choices_4(text)
        assert choices is not None
        assert len(choices) == 4
        assert choices[0] == "첫째"
        assert choices[3] == "넷째"

    def test_extract_choices_4_incomplete(self):
        """4지선다 선지가 불완전하면 None."""
        parser = TeacherPDFParser()
        text = "문제 내용 ①첫째 ②둘째"
        choices = parser._extract_choices_4(text)
        assert choices is None

    def test_parse_pages_with_choices(self):
        """선지가 있는 임용고시 문항 파싱."""
        parser = TeacherPDFParser()
        pages = [
            "1. 다음 중 올바른 것은?\n"
            "①첫째 ②둘째 ③셋째 ④넷째"
        ]
        path = Path("data/raw/korean_exam/pdf/teacher/2024/test.pdf")
        items = parser._parse_pages(pages, path)
        assert len(items) >= 1
        assert items[0].metadata.exam_type == "teacher"
        assert items[0].choices is not None
        assert len(items[0].choices) == 4

    def test_parse_pages_descriptive(self):
        """서술형 문항 파싱 (선지 없음)."""
        parser = TeacherPDFParser()
        pages = [
            "1. 다음 지문을 읽고 서술하시오.\n"
            "지문 내용입니다.\n"
            "위 지문의 주제를 서술하시오."
        ]
        path = Path("data/raw/korean_exam/pdf/teacher/2024/test.pdf")
        items = parser._parse_pages(pages, path)
        assert len(items) >= 1
        assert items[0].choices is None  # 서술형


class TestLEETPDFParser:
    def test_split_by_passages(self):
        """[지문 N] 패턴으로 분할."""
        parser = LEETPDFParser()
        text = (
            "[지문 1]\n"
            "첫 번째 지문 내용입니다.\n"
            "1. 문제 하나\n"
            "[지문 2]\n"
            "두 번째 지문 내용입니다.\n"
            "2. 문제 둘\n"
        )
        blocks = parser._split_by_passages(text)
        assert len(blocks) == 2
        assert "첫 번째 지문" in blocks[0][0]
        assert "두 번째 지문" in blocks[1][0]

    def test_split_by_passages_no_pattern(self):
        """[지문] 패턴이 없으면 빈 리스트."""
        parser = LEETPDFParser()
        text = "1. 일반 문제입니다.\n2. 또 다른 문제."
        blocks = parser._split_by_passages(text)
        assert blocks == []

    def test_extract_choices_5(self):
        """5지선다 선지를 추출."""
        parser = LEETPDFParser()
        text = "문제 내용 ①가 ②나 ③다 ④라 ⑤마"
        choices = parser._extract_choices_5(text)
        assert choices is not None
        assert len(choices) == 5

    def test_parse_pages_with_passage_blocks(self):
        """지문 세트 구조 파싱."""
        parser = LEETPDFParser()
        pages = [
            "[지문 1]\n"
            "법학 관련 지문입니다.\n"
            "1. 윗글의 내용과 일치하는 것은?\n"
            "①가 ②나 ③다 ④라 ⑤마\n"
            "2. 윗글에 대한 설명으로 적절한 것은?\n"
            "①가 ②나 ③다 ④라 ⑤마"
        ]
        path = Path("data/raw/korean_exam/pdf/leet/2024/test.pdf")
        items = parser._parse_pages(pages, path)
        assert len(items) >= 1
        assert items[0].metadata.exam_type == "leet"
        assert items[0].metadata.subject == "언어이해"
        # 지문이 공유됨
        assert "법학 관련 지문" in items[0].passage

    def test_parse_pages_without_passage_blocks(self):
        """지문 세트 구조 없이 일반 문항 파싱."""
        parser = LEETPDFParser()
        pages = [
            "1. 다음 중 올바른 것은?\n"
            "①가 ②나 ③다 ④라 ⑤마"
        ]
        path = Path("data/raw/korean_exam/pdf/leet/2024/test.pdf")
        items = parser._parse_pages(pages, path)
        assert len(items) >= 1
