"""Pipeline 클래스 단위 테스트.

importlib을 사용하여 torch 의존성을 회피합니다.
"""

import importlib
import importlib.util
import json
import logging
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- Module loading helpers (avoid torch dependency) ---


def _load_module(module_name: str, file_path: str):
    """importlib으로 모듈을 직접 로드한다."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # 이미 로드된 모듈이 있으면 재사용
    if module_name not in sys.modules:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# grinvi.korean_exam 서브패키지를 torch 없이 로드
_BASE = Path(__file__).parent.parent / "grinvi" / "korean_exam"

# 먼저 grinvi 패키지를 빈 모듈로 등록 (torch import 방지)
if "grinvi" not in sys.modules:
    import types

    grinvi_mock = types.ModuleType("grinvi")
    grinvi_mock.__path__ = [str(Path(__file__).parent.parent / "grinvi")]
    sys.modules["grinvi"] = grinvi_mock

if "grinvi.korean_exam" not in sys.modules:
    import types

    ke_mock = types.ModuleType("grinvi.korean_exam")
    ke_mock.__path__ = [str(_BASE)]
    sys.modules["grinvi.korean_exam"] = ke_mock

# 필요한 모듈들을 순서대로 로드
_load_module("grinvi.korean_exam.models", str(_BASE / "models.py"))
_load_module("grinvi.korean_exam.validator", str(_BASE / "validator.py"))
_load_module("grinvi.korean_exam.formatter", str(_BASE / "formatter.py"))
_load_module("grinvi.korean_exam.storage", str(_BASE / "storage.py"))

# collectors 패키지
if "grinvi.korean_exam.collectors" not in sys.modules:
    import types

    coll_mock = types.ModuleType("grinvi.korean_exam.collectors")
    coll_mock.__path__ = [str(_BASE / "collectors")]
    sys.modules["grinvi.korean_exam.collectors"] = coll_mock

_load_module(
    "grinvi.korean_exam.collectors.base", str(_BASE / "collectors" / "base.py")
)
_load_module(
    "grinvi.korean_exam.collectors.hf_collector",
    str(_BASE / "collectors" / "hf_collector.py"),
)

# parsers 패키지
if "grinvi.korean_exam.parsers" not in sys.modules:
    import types

    parsers_mock = types.ModuleType("grinvi.korean_exam.parsers")
    parsers_mock.__path__ = [str(_BASE / "parsers")]
    sys.modules["grinvi.korean_exam.parsers"] = parsers_mock

_load_module("grinvi.korean_exam.parsers.base", str(_BASE / "parsers" / "base.py"))
_load_module(
    "grinvi.korean_exam.parsers.hf_parsers", str(_BASE / "parsers" / "hf_parsers.py")
)

# pipeline 모듈 로드
_load_module("grinvi.korean_exam.pipeline", str(_BASE / "pipeline.py"))

from grinvi.korean_exam.models import ExamItem, ExamMetadata
from grinvi.korean_exam.pipeline import Pipeline, PipelineResult, _HFCompositeParser


# --- Fixtures ---


def _make_item(question: str, exam_type: str = "csat", year: int = 2023) -> ExamItem:
    """테스트용 ExamItem 생성 헬퍼."""
    return ExamItem(
        passage="다음 글을 읽고 물음에 답하시오.",
        question=question,
        choices=["선택지 하나", "선택지 둘", "선택지 셋", "선택지 넷", "선택지 다섯"],
        answer=1,
        explanation=None,
        metadata=ExamMetadata(
            exam_type=exam_type,
            year=year,
            subject="국어",
        ),
    )


def _make_args(**kwargs) -> Namespace:
    """테스트용 args Namespace 생성."""
    defaults = {
        "source": "huggingface",
        "out": None,  # will be set per test
        "merge": False,
        "validate": False,
        "max_items": None,
        "exam": None,
        "input": None,
        "format": "jsonl",
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _write_jsonl(path: Path, records: list) -> None:
    """JSONL 파일 작성 헬퍼."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- PipelineResult Tests ---


class TestPipelineResult:
    def test_creation(self):
        result = PipelineResult(total=100, saved=80, filtered=20)
        assert result.total == 100
        assert result.saved == 80
        assert result.filtered == 20
        assert result.exam_type_dist == {}
        assert result.year_dist == {}

    def test_with_distributions(self):
        result = PipelineResult(
            total=50,
            saved=45,
            filtered=5,
            exam_type_dist={"csat": 30, "civil_local": 20},
            year_dist={2022: 25, 2023: 25},
        )
        assert result.exam_type_dist["csat"] == 30
        assert result.year_dist[2023] == 25


# --- _HFCompositeParser Tests ---


class TestHFCompositeParser:
    def test_dispatch_csatqa(self, tmp_path):
        """csatqa 데이터셋 경로에서 CSATQAParser로 디스패치."""
        data_dir = tmp_path / "huggingface" / "csatqa"
        record = {
            "test_name": "2022년 2023 대학수학능력시험 국어 홀수형",
            "context": "다음 글을 읽고 물음에 답하시오. 한국어 지문입니다.",
            "question": "윗글에 대한 설명으로 적절한 것은 무엇인가요?",
            "gold": 3,
            "option#1": "첫 번째 선택지입니다",
            "option#2": "두 번째 선택지입니다",
            "option#3": "세 번째 선택지입니다",
            "option#4": "네 번째 선택지입니다",
            "option#5": "다섯 번째 선택지입니다",
        }
        _write_jsonl(data_dir / "raw.jsonl", [record])

        parser = _HFCompositeParser()
        items = parser.parse_file(data_dir / "raw.jsonl")

        assert len(items) == 1
        assert items[0].metadata.exam_type == "csat"
        assert items[0].metadata.year == 2023

    def test_dispatch_civil(self, tmp_path):
        """civil_local 데이터셋 경로에서 CivilParser로 디스패치."""
        data_dir = tmp_path / "huggingface" / "civil_local"
        record = {
            "문제 번호": 1,
            "문제 내용": "다음 중 맞춤법이 올바른 것은 무엇인가요?",
            "선택지 1": "첫 번째 선택지",
            "선택지 2": "두 번째 선택지",
            "선택지 3": "세 번째 선택지",
            "선택지 4": "네 번째 선택지",
            "정답": 2,
            "year": 2022,
            "task": "국어",
            "source": "지방직 9급",
        }
        _write_jsonl(data_dir / "raw.jsonl", [record])

        parser = _HFCompositeParser()
        items = parser.parse_file(data_dir / "raw.jsonl")

        assert len(items) == 1
        assert items[0].metadata.exam_type == "civil_local"
        assert items[0].metadata.year == 2022

    def test_dispatch_unknown_falls_back(self, tmp_path):
        """알 수 없는 데이터셋 ID는 CSATQAParser로 폴백."""
        data_dir = tmp_path / "huggingface" / "unknown_dataset"
        record = {
            "test_name": "2022년 2023 대학수학능력시험 국어 홀수형",
            "context": "한국어 지문 텍스트입니다.",
            "question": "윗글에 대한 설명으로 적절한 것은?",
            "gold": 1,
            "option#1": "선택지 하나",
            "option#2": "선택지 둘",
            "option#3": "선택지 셋",
            "option#4": "선택지 넷",
            "option#5": "선택지 다섯",
        }
        _write_jsonl(data_dir / "raw.jsonl", [record])

        parser = _HFCompositeParser()
        items = parser.parse_file(data_dir / "raw.jsonl")
        assert len(items) == 1


# --- Pipeline Integration Tests ---


class TestPipeline:
    def test_run_basic(self, tmp_path):
        """기본 파이프라인 실행 (validate=False)."""
        hf_dir = tmp_path / "raw" / "korean_exam" / "huggingface" / "csatqa"
        records = [
            {
                "test_name": "2022년 2023 대학수학능력시험 국어 홀수형",
                "context": "다음 글을 읽고 물음에 답하시오. 한국어 지문입니다.",
                "question": "윗글에 대한 설명으로 적절한 것은 무엇인가요?",
                "gold": 3,
                "option#1": "첫 번째 선택지입니다",
                "option#2": "두 번째 선택지입니다",
                "option#3": "세 번째 선택지입니다",
                "option#4": "네 번째 선택지입니다",
                "option#5": "다섯 번째 선택지입니다",
            },
            {
                "test_name": "2021년 2022 대학수학능력시험 국어 홀수형",
                "context": "한국어 지문 텍스트 내용입니다.",
                "question": "윗글의 내용과 일치하는 것은 무엇인가요?",
                "gold": 2,
                "option#1": "선택지 가입니다",
                "option#2": "선택지 나입니다",
                "option#3": "선택지 다입니다",
                "option#4": "선택지 라입니다",
                "option#5": "선택지 마입니다",
            },
        ]
        _write_jsonl(hf_dir / "raw.jsonl", records)

        args = _make_args(out=str(tmp_path))

        mock_collector = MagicMock()
        mock_collector.collect.return_value = iter([hf_dir / "raw.jsonl"])

        pipeline = Pipeline()
        with patch.object(pipeline, "_build_collector", return_value=mock_collector):
            result = pipeline.run(args)

        assert result.total == 2
        assert result.saved == 2
        assert result.filtered == 0
        assert "csat" in result.exam_type_dist

    def test_run_with_validate(self, tmp_path):
        """--validate 옵션으로 한국어 비율 검증 적용."""
        hf_dir = tmp_path / "raw" / "korean_exam" / "huggingface" / "csatqa"
        records = [
            {
                "test_name": "2023 대학수학능력시험 국어",
                "context": "한국어 지문입니다.",
                "question": "이 문제는 한국어로 작성되었습니다.",
                "gold": 1,
                "option#1": "선택지 하나",
                "option#2": "선택지 둘",
                "option#3": "선택지 셋",
                "option#4": "선택지 넷",
                "option#5": "선택지 다섯",
            },
            {
                "test_name": "2023 test english",
                "context": "This is English text only.",
                "question": "This question is in English only.",
                "gold": 2,
                "option#1": "Option A",
                "option#2": "Option B",
                "option#3": "Option C",
                "option#4": "Option D",
                "option#5": "Option E",
            },
        ]
        _write_jsonl(hf_dir / "raw.jsonl", records)

        args = _make_args(out=str(tmp_path), validate=True)

        mock_collector = MagicMock()
        mock_collector.collect.return_value = iter([hf_dir / "raw.jsonl"])

        pipeline = Pipeline()
        with patch.object(pipeline, "_build_collector", return_value=mock_collector):
            result = pipeline.run(args)

        # 영어 문제는 필터링됨
        assert result.total == 1
        assert result.exam_type_dist.get("csat", 0) == 1

    def test_run_with_max_items(self, tmp_path):
        """--max_items 옵션으로 처리 항목 수 제한."""
        hf_dir = tmp_path / "raw" / "korean_exam" / "huggingface" / "csatqa"
        records = [
            {
                "test_name": "2023 대학수학능력시험 국어",
                "context": f"한국어 지문 {i}번입니다.",
                "question": f"문제 {i}번: 윗글에 대한 설명으로 적절한 것은?",
                "gold": 1,
                "option#1": "선택지 하나입니다",
                "option#2": "선택지 둘입니다",
                "option#3": "선택지 셋입니다",
                "option#4": "선택지 넷입니다",
                "option#5": "선택지 다섯입니다",
            }
            for i in range(10)
        ]
        _write_jsonl(hf_dir / "raw.jsonl", records)

        args = _make_args(out=str(tmp_path), max_items=3)

        mock_collector = MagicMock()
        mock_collector.collect.return_value = iter([hf_dir / "raw.jsonl"])

        pipeline = Pipeline()
        with patch.object(pipeline, "_build_collector", return_value=mock_collector):
            result = pipeline.run(args)

        assert result.total == 3
        assert result.saved <= 3

    def test_run_deduplication(self, tmp_path):
        """중복 제거가 항상 적용되는지 확인."""
        hf_dir = tmp_path / "raw" / "korean_exam" / "huggingface" / "csatqa"
        record = {
            "test_name": "2023 대학수학능력시험 국어",
            "context": "한국어 지문입니다.",
            "question": "동일한 문제 텍스트입니다. 이것은 중복입니다.",
            "gold": 1,
            "option#1": "선택지 하나입니다",
            "option#2": "선택지 둘입니다",
            "option#3": "선택지 셋입니다",
            "option#4": "선택지 넷입니다",
            "option#5": "선택지 다섯입니다",
        }
        _write_jsonl(hf_dir / "raw.jsonl", [record, record])

        args = _make_args(out=str(tmp_path))

        mock_collector = MagicMock()
        mock_collector.collect.return_value = iter([hf_dir / "raw.jsonl"])

        pipeline = Pipeline()
        with patch.object(pipeline, "_build_collector", return_value=mock_collector):
            result = pipeline.run(args)

        # 중복 제거로 1건만 남음
        assert result.total == 1

    def test_run_warning_on_low_valid_ratio(self, tmp_path, caplog):
        """유효 항목 50% 미만 시 경고 메시지 출력."""
        hf_dir = tmp_path / "raw" / "korean_exam" / "huggingface" / "csatqa"
        # 대부분 영어 문제 (validate=True 시 필터링됨)
        records = [
            {
                "test_name": "2023 test",
                "context": "English context text here.",
                "question": f"English question number {i} here.",
                "gold": 1,
                "option#1": "Option A",
                "option#2": "Option B",
                "option#3": "Option C",
                "option#4": "Option D",
                "option#5": "Option E",
            }
            for i in range(8)
        ] + [
            {
                "test_name": "2023 대학수학능력시험 국어",
                "context": "한국어 지문입니다.",
                "question": "한국어 문제입니다. 적절한 것을 고르시오.",
                "gold": 1,
                "option#1": "선택지 하나입니다",
                "option#2": "선택지 둘입니다",
                "option#3": "선택지 셋입니다",
                "option#4": "선택지 넷입니다",
                "option#5": "선택지 다섯입니다",
            }
        ]
        _write_jsonl(hf_dir / "raw.jsonl", records)

        args = _make_args(out=str(tmp_path), validate=True)

        mock_collector = MagicMock()
        mock_collector.collect.return_value = iter([hf_dir / "raw.jsonl"])

        pipeline = Pipeline()
        with patch.object(pipeline, "_build_collector", return_value=mock_collector):
            with caplog.at_level(logging.WARNING):
                result = pipeline.run(args)

        # 유효 항목이 50% 미만이므로 경고 발생
        assert any("50%" in record.message for record in caplog.records)

    def test_build_collector_huggingface(self):
        """_build_collector가 huggingface 소스에 대해 HFCollector를 반환."""
        from grinvi.korean_exam.collectors.hf_collector import HFCollector

        pipeline = Pipeline()
        args = _make_args(source="huggingface", out="data")
        collector = pipeline._build_collector(args)

        assert isinstance(collector, HFCollector)

    def test_build_collector_invalid_source(self):
        """지원하지 않는 소스에 대해 ValueError 발생."""
        pipeline = Pipeline()
        args = _make_args(source="invalid", out="data")

        with pytest.raises(ValueError, match="지원하지 않는 소스"):
            pipeline._build_collector(args)

    def test_build_parser_huggingface(self):
        """_build_parser가 huggingface 소스에 대해 _HFCompositeParser를 반환."""
        pipeline = Pipeline()
        args = _make_args(source="huggingface")
        parser = pipeline._build_parser(args)

        assert isinstance(parser, _HFCompositeParser)

    def test_build_parser_invalid_source(self):
        """지원하지 않는 소스에 대해 ValueError 발생."""
        pipeline = Pipeline()
        args = _make_args(source="invalid")

        with pytest.raises(ValueError, match="지원하지 않는 소스"):
            pipeline._build_parser(args)
