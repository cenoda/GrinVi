"""Unit tests for LocalCollector and LocalParser."""

import csv
import importlib.util
import json
import sys
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
sys.modules.setdefault("grinvi", type(sys)("grinvi"))
sys.modules["grinvi"].korean_exam = type(sys)("grinvi.korean_exam")
sys.modules["grinvi.korean_exam"] = sys.modules["grinvi"].korean_exam
sys.modules["grinvi.korean_exam"].models = _models

# Load collectors.base
_spec_base_col = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.base",
    _base_dir / "collectors" / "base.py",
)
_base_col = importlib.util.module_from_spec(_spec_base_col)
sys.modules["grinvi.korean_exam.collectors.base"] = _base_col
_spec_base_col.loader.exec_module(_base_col)

# Load collectors.local_collector
_spec_local_col = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.local_collector",
    _base_dir / "collectors" / "local_collector.py",
)
_local_col = importlib.util.module_from_spec(_spec_local_col)
sys.modules["grinvi.korean_exam.collectors.local_collector"] = _local_col
_spec_local_col.loader.exec_module(_local_col)

# Load parsers.base
_spec_base_par = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.parsers.base",
    _base_dir / "parsers" / "base.py",
)
_base_par = importlib.util.module_from_spec(_spec_base_par)
sys.modules["grinvi.korean_exam.parsers.base"] = _base_par
_spec_base_par.loader.exec_module(_base_par)

# Load parsers.local_parser
_spec_local_par = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.parsers.local_parser",
    _base_dir / "parsers" / "local_parser.py",
)
_local_par = importlib.util.module_from_spec(_spec_local_par)
sys.modules["grinvi.korean_exam.parsers.local_parser"] = _local_par
_spec_local_par.loader.exec_module(_local_par)

LocalCollector = _local_col.LocalCollector
LocalParser = _local_par.LocalParser
ExamItem = _models.ExamItem
ExamMetadata = _models.ExamMetadata


class TestLocalCollector:
    def test_collect_single_json_file(self, tmp_path):
        """단일 JSON 파일 수집."""
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"question": "test"}]')

        collector = LocalCollector(out_dir=tmp_path, input_path=str(json_file))
        paths = list(collector.collect())
        assert len(paths) == 1
        assert paths[0] == json_file

    def test_collect_single_csv_file(self, tmp_path):
        """단일 CSV 파일 수집."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("question,answer\ntest,1\n")

        collector = LocalCollector(out_dir=tmp_path, input_path=str(csv_file))
        paths = list(collector.collect())
        assert len(paths) == 1
        assert paths[0] == csv_file

    def test_collect_single_txt_file(self, tmp_path):
        """단일 TXT 파일 수집."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("문제 내용입니다.")

        collector = LocalCollector(out_dir=tmp_path, input_path=str(txt_file))
        paths = list(collector.collect())
        assert len(paths) == 1
        assert paths[0] == txt_file

    def test_collect_single_jsonl_file(self, tmp_path):
        """단일 JSONL 파일 수집."""
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"question": "test"}\n')

        collector = LocalCollector(out_dir=tmp_path, input_path=str(jsonl_file))
        paths = list(collector.collect())
        assert len(paths) == 1
        assert paths[0] == jsonl_file

    def test_collect_unsupported_format(self, tmp_path):
        """미지원 형식은 에러 메시지 출력."""
        pdf_file = tmp_path / "data.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        collector = LocalCollector(out_dir=tmp_path, input_path=str(pdf_file))
        paths = list(collector.collect())
        assert len(paths) == 0

    def test_collect_nonexistent_path(self, tmp_path):
        """존재하지 않는 경로는 에러 메시지 출력."""
        collector = LocalCollector(
            out_dir=tmp_path, input_path=str(tmp_path / "nonexistent")
        )
        paths = list(collector.collect())
        assert len(paths) == 0

    def test_collect_no_input_path(self, tmp_path):
        """input_path가 None이면 에러 메시지 출력."""
        collector = LocalCollector(out_dir=tmp_path, input_path=None)
        paths = list(collector.collect())
        assert len(paths) == 0

    def test_collect_directory_recursive(self, tmp_path):
        """디렉토리 재귀 탐색."""
        # 중첩 디렉토리 구조 생성
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        (tmp_path / "file1.json").write_text('[{"q": "1"}]')
        (sub_dir / "file2.csv").write_text("question\ntest\n")
        (sub_dir / "file3.txt").write_text("문제")
        (sub_dir / "ignored.pdf").write_bytes(b"PDF")  # 무시됨

        collector = LocalCollector(out_dir=tmp_path / "out", input_path=str(tmp_path))
        paths = list(collector.collect())
        assert len(paths) == 3
        extensions = {p.suffix for p in paths}
        assert ".json" in extensions
        assert ".csv" in extensions
        assert ".txt" in extensions
        assert ".pdf" not in extensions

    def test_collect_empty_directory(self, tmp_path):
        """빈 디렉토리는 경고 메시지 출력."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        collector = LocalCollector(out_dir=tmp_path, input_path=str(empty_dir))
        paths = list(collector.collect())
        assert len(paths) == 0


class TestLocalParser:
    def test_parse_basic_dict(self):
        """기본 dict 파싱."""
        parser = LocalParser()
        raw = {
            "question": "다음 중 올바른 것은?",
            "passage": "지문 내용",
            "choices": ["가", "나", "다", "라", "마"],
            "answer": 3,
            "year": 2024,
            "exam_type": "csat",
        }
        item = parser.parse(raw)
        assert item.question == "다음 중 올바른 것은?"
        assert item.passage == "지문 내용"
        assert item.choices == ["가", "나", "다", "라", "마"]
        assert item.answer == 3
        assert item.metadata.year == 2024
        assert item.metadata.exam_type == "csat"

    def test_parse_korean_field_names(self):
        """한국어 필드명 지원."""
        parser = LocalParser()
        raw = {
            "문제": "맞춤법이 올바른 것은?",
            "지문": "지문입니다",
            "정답": 2,
            "연도": 2023,
            "시험종류": "mock",
        }
        item = parser.parse(raw)
        assert item.question == "맞춤법이 올바른 것은?"
        assert item.passage == "지문입니다"
        assert item.answer == 2
        assert item.metadata.year == 2023
        assert item.metadata.exam_type == "mock"

    def test_parse_empty_question_raises(self):
        """question이 비어있으면 ValueError."""
        parser = LocalParser()
        raw = {"question": "", "answer": 1}
        with pytest.raises(ValueError):
            parser.parse(raw)

    def test_parse_no_question_raises(self):
        """question 필드가 없으면 ValueError."""
        parser = LocalParser()
        raw = {"answer": 1, "passage": "지문"}
        with pytest.raises(ValueError):
            parser.parse(raw)

    def test_parse_numbered_choices(self):
        """선택지 1~4 형태의 선지 추출."""
        parser = LocalParser()
        raw = {
            "question": "문제입니다",
            "선택지 1": "가",
            "선택지 2": "나",
            "선택지 3": "다",
            "선택지 4": "라",
        }
        item = parser.parse(raw)
        assert item.choices == ["가", "나", "다", "라"]

    def test_parse_letter_choices(self):
        """A~E 형태의 선지 추출."""
        parser = LocalParser()
        raw = {
            "question": "문제입니다",
            "A": "첫째",
            "B": "둘째",
            "C": "셋째",
            "D": "넷째",
            "E": "다섯째",
        }
        item = parser.parse(raw)
        assert item.choices == ["첫째", "둘째", "셋째", "넷째", "다섯째"]

    def test_parse_no_choices(self):
        """선지가 없으면 None."""
        parser = LocalParser()
        raw = {"question": "서술형 문제입니다"}
        item = parser.parse(raw)
        assert item.choices is None

    def test_parse_file_json(self, tmp_path):
        """JSON 파일 파싱."""
        parser = LocalParser()
        json_file = tmp_path / "data.json"
        data = [
            {"question": "문제 하나", "answer": 1},
            {"question": "문제 둘", "answer": 2},
        ]
        json_file.write_text(json.dumps(data, ensure_ascii=False))

        items = parser.parse_file(json_file)
        assert len(items) == 2
        assert items[0].question == "문제 하나"
        assert items[1].question == "문제 둘"

    def test_parse_file_json_single_object(self, tmp_path):
        """단일 객체 JSON 파일 파싱."""
        parser = LocalParser()
        json_file = tmp_path / "data.json"
        data = {"question": "단일 문제", "answer": 3}
        json_file.write_text(json.dumps(data, ensure_ascii=False))

        items = parser.parse_file(json_file)
        assert len(items) == 1
        assert items[0].question == "단일 문제"

    def test_parse_file_json_with_data_key(self, tmp_path):
        """data 키 아래 배열이 있는 JSON 파싱."""
        parser = LocalParser()
        json_file = tmp_path / "data.json"
        data = {"data": [{"question": "문제1"}, {"question": "문제2"}]}
        json_file.write_text(json.dumps(data, ensure_ascii=False))

        items = parser.parse_file(json_file)
        assert len(items) == 2

    def test_parse_file_jsonl(self, tmp_path):
        """JSONL 파일 파싱."""
        parser = LocalParser()
        jsonl_file = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"question": "문제 하나"}, ensure_ascii=False),
            json.dumps({"question": "문제 둘"}, ensure_ascii=False),
        ]
        jsonl_file.write_text("\n".join(lines))

        items = parser.parse_file(jsonl_file)
        assert len(items) == 2

    def test_parse_file_csv(self, tmp_path):
        """CSV 파일 파싱."""
        parser = LocalParser()
        csv_file = tmp_path / "data.csv"
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "CSV 문제 하나", "answer": "1"})
            writer.writerow({"question": "CSV 문제 둘", "answer": "2"})

        items = parser.parse_file(csv_file)
        assert len(items) == 2
        assert items[0].question == "CSV 문제 하나"
        assert items[0].answer == 1

    def test_parse_file_txt(self, tmp_path):
        """TXT 파일 파싱 (빈 줄로 구분된 블록)."""
        parser = LocalParser()
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("첫 번째 문제입니다.\n\n두 번째 문제입니다.")

        items = parser.parse_file(txt_file)
        assert len(items) == 2
        assert items[0].question == "첫 번째 문제입니다."
        assert items[1].question == "두 번째 문제입니다."

    def test_parse_file_unsupported(self, tmp_path):
        """미지원 형식은 빈 리스트."""
        parser = LocalParser()
        xml_file = tmp_path / "data.xml"
        xml_file.write_text("<data></data>")

        items = parser.parse_file(xml_file)
        assert items == []

    def test_parse_file_json_skips_invalid(self, tmp_path):
        """JSON 파일에서 잘못된 레코드를 건너뛴다."""
        parser = LocalParser()
        json_file = tmp_path / "data.json"
        data = [
            {"question": "유효한 문제"},
            {"no_question": "무효"},  # question 없음
            {"question": "또 유효한 문제"},
        ]
        json_file.write_text(json.dumps(data, ensure_ascii=False))

        items = parser.parse_file(json_file)
        assert len(items) == 2

    def test_parse_file_csv_skips_invalid(self, tmp_path):
        """CSV 파일에서 잘못된 행을 건너뛴다."""
        parser = LocalParser()
        csv_file = tmp_path / "data.csv"
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "유효", "answer": "1"})
            writer.writerow({"question": "", "answer": "2"})  # 빈 question
            writer.writerow({"question": "유효2", "answer": "3"})

        items = parser.parse_file(csv_file)
        assert len(items) == 2

    def test_deterministic_parsing(self):
        """동일한 입력을 두 번 파싱하면 동일한 결과."""
        parser = LocalParser()
        raw = {
            "question": "다음 중 올바른 것은?",
            "passage": "지문",
            "choices": ["가", "나", "다"],
            "answer": 1,
            "year": 2024,
        }
        item1 = parser.parse(raw)
        item2 = parser.parse(raw)
        assert item1 == item2
