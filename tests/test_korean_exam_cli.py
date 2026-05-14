"""CLI 스크립트 (scripts/collect_korean_exam.py) 단위 테스트.

importlib을 사용하여 torch 의존성을 회피합니다.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- Module loading helpers (avoid torch dependency) ---

_BASE = Path(__file__).parent.parent / "grinvi" / "korean_exam"

# grinvi 패키지를 빈 모듈로 등록 (torch import 방지)
if "grinvi" not in sys.modules:
    grinvi_mock = types.ModuleType("grinvi")
    grinvi_mock.__path__ = [str(Path(__file__).parent.parent / "grinvi")]
    sys.modules["grinvi"] = grinvi_mock

if "grinvi.korean_exam" not in sys.modules:
    ke_mock = types.ModuleType("grinvi.korean_exam")
    ke_mock.__path__ = [str(_BASE)]
    sys.modules["grinvi.korean_exam"] = ke_mock


def _load_module(module_name: str, file_path: str):
    """importlib으로 모듈을 직접 로드한다."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if module_name not in sys.modules:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 필요한 모듈들을 순서대로 로드
_load_module("grinvi.korean_exam.models", str(_BASE / "models.py"))
_load_module("grinvi.korean_exam.validator", str(_BASE / "validator.py"))
_load_module("grinvi.korean_exam.formatter", str(_BASE / "formatter.py"))
_load_module("grinvi.korean_exam.storage", str(_BASE / "storage.py"))

if "grinvi.korean_exam.collectors" not in sys.modules:
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

if "grinvi.korean_exam.parsers" not in sys.modules:
    parsers_mock = types.ModuleType("grinvi.korean_exam.parsers")
    parsers_mock.__path__ = [str(_BASE / "parsers")]
    sys.modules["grinvi.korean_exam.parsers"] = parsers_mock

_load_module("grinvi.korean_exam.parsers.base", str(_BASE / "parsers" / "base.py"))
_load_module(
    "grinvi.korean_exam.parsers.hf_parsers", str(_BASE / "parsers" / "hf_parsers.py")
)

# pipeline 모듈 로드
_load_module("grinvi.korean_exam.pipeline", str(_BASE / "pipeline.py"))


def _load_cli_module():
    """importlib으로 CLI 모듈을 로드한다 (torch 의존성 회피)."""
    spec = importlib.util.spec_from_file_location(
        "collect_korean_exam",
        Path(__file__).parent.parent / "scripts" / "collect_korean_exam.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


class TestBuildParser:
    """argparse 파서 생성 테스트."""

    def test_parser_creation(self):
        parser = cli.build_parser()
        assert parser is not None
        assert parser.prog == "collect_korean_exam"

    def test_parse_huggingface_source(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "huggingface"])
        assert args.source == "huggingface"
        assert args.out == "data"
        assert args.merge is False
        assert args.validate is False
        assert args.format == "jsonl"
        assert args.max_items is None

    def test_parse_pdf_source_with_exam(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "pdf", "--exam", "csat"])
        assert args.source == "pdf"
        assert args.exam == "csat"

    def test_parse_local_source_with_input(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "local", "--input", "/tmp/data"])
        assert args.source == "local"
        assert args.input == "/tmp/data"

    def test_parse_all_options(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "--source", "huggingface",
            "--out", "/tmp/output",
            "--merge",
            "--format", "plain_text",
            "--max_items", "50",
            "--validate",
        ])
        assert args.source == "huggingface"
        assert args.out == "/tmp/output"
        assert args.merge is True
        assert args.format == "plain_text"
        assert args.max_items == 50
        assert args.validate is True

    def test_parse_exam_all(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "pdf", "--exam", "all"])
        assert args.exam == "all"

    def test_missing_source_raises(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_source_raises(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--source", "invalid"])

    def test_invalid_format_raises(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--source", "huggingface", "--format", "csv"])


class TestValidateArgs:
    """인수 유효성 검사 테스트."""

    def test_local_without_input_fails(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "local"])
        assert cli.validate_args(args) is False

    def test_local_with_input_passes(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "local", "--input", "/tmp/data"])
        assert cli.validate_args(args) is True

    def test_pdf_without_exam_fails(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "pdf"])
        assert cli.validate_args(args) is False

    def test_pdf_with_exam_passes(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "pdf", "--exam", "csat"])
        assert cli.validate_args(args) is True

    def test_huggingface_passes(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "huggingface"])
        assert cli.validate_args(args) is True


class TestMain:
    """main() 함수 통합 테스트."""

    def test_missing_required_args_returns_1(self):
        """필수 인수 누락 시 종료 코드 1."""
        with patch("sys.argv", ["collect_korean_exam"]):
            # argparse will call sys.exit(2) for missing required args
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 2

    def test_local_without_input_returns_1(self):
        """--source local에서 --input 누락 시 종료 코드 1."""
        with patch("sys.argv", ["collect_korean_exam", "--source", "local"]):
            result = cli.main()
            assert result == 1

    def test_pdf_without_exam_returns_1(self):
        """--source pdf에서 --exam 누락 시 종료 코드 1."""
        with patch("sys.argv", ["collect_korean_exam", "--source", "pdf"]):
            result = cli.main()
            assert result == 1

    def test_successful_run_returns_0(self):
        """정상 실행 시 종료 코드 0."""
        mock_result = MagicMock()
        mock_result.total = 10
        mock_result.saved = 8
        mock_result.filtered = 2

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = mock_result

        # Patch within the module loaded by importlib
        import grinvi.korean_exam.pipeline as pipeline_mod

        original_pipeline = pipeline_mod.Pipeline
        try:
            pipeline_mod.Pipeline = mock_pipeline_cls
            with patch("sys.argv", ["collect_korean_exam", "--source", "huggingface"]):
                result = cli.main()
                assert result == 0
        finally:
            pipeline_mod.Pipeline = original_pipeline

    def test_pipeline_exception_returns_1(self):
        """파이프라인 예외 발생 시 종료 코드 1."""
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.side_effect = RuntimeError("테스트 오류")

        import grinvi.korean_exam.pipeline as pipeline_mod

        original_pipeline = pipeline_mod.Pipeline
        try:
            pipeline_mod.Pipeline = mock_pipeline_cls
            with patch("sys.argv", ["collect_korean_exam", "--source", "huggingface"]):
                result = cli.main()
                assert result == 1
        finally:
            pipeline_mod.Pipeline = original_pipeline


class TestPrintNextSteps:
    """다음 단계 출력 테스트."""

    def test_prints_next_steps(self, capsys):
        cli.print_next_steps()
        captured = capsys.readouterr()
        assert "수집 완료" in captured.out
        assert "토크나이저 훈련" in captured.out
        assert "모델 훈련" in captured.out
        assert "train_tokenizer.py" in captured.out
        assert "train.py" in captured.out
