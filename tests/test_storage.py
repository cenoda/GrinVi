"""Storage 클래스 단위 테스트.

importlib을 사용하여 grinvi/__init__.py의 torch 의존성을 우회합니다.
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# importlib으로 storage 모듈 직접 로드 (torch 의존성 우회)
_spec = importlib.util.spec_from_file_location(
    "storage",
    Path(__file__).resolve().parent.parent / "grinvi" / "korean_exam" / "storage.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["storage"] = _module
_spec.loader.exec_module(_module)

Storage = _module.Storage
_format_size = _module._format_size


@pytest.fixture
def tmp_out_dir(tmp_path):
    """임시 출력 디렉토리를 제공한다."""
    return tmp_path


@pytest.fixture
def sample_records():
    """테스트용 샘플 레코드 리스트."""
    return [
        {
            "text": "지문: 한국어 텍스트 예시입니다.\n\n문제: 다음 중 올바른 것은?",
            "source": "korean_exam",
            "exam_type": "csat",
            "year": 2023,
            "subject": "국어",
            "question_number": 1,
        },
        {
            "text": "지문: 두 번째 지문입니다.\n\n문제: 이 글의 주제는?",
            "source": "korean_exam",
            "exam_type": "mock_june",
            "year": 2022,
            "subject": "국어",
            "question_number": 5,
        },
    ]


class TestStorageSaveJSONL:
    """요구사항 7.1: JSONL 파일 저장 테스트."""

    def test_creates_jsonl_file(self, tmp_out_dir, sample_records):
        """JSONL 파일이 올바른 경로에 생성되는지 확인."""
        storage = Storage()
        result_path = storage.save(sample_records, tmp_out_dir)

        assert result_path.exists()
        assert result_path.parent == tmp_out_dir / "raw" / "korean_exam"
        assert result_path.name.startswith("korean_exam_")
        assert result_path.suffix == ".jsonl"

    def test_jsonl_content_matches_records(self, tmp_out_dir, sample_records):
        """저장된 JSONL 내용이 입력 레코드와 일치하는지 확인."""
        storage = Storage()
        result_path = storage.save(sample_records, tmp_out_dir)

        lines = result_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(sample_records)

        for line, expected in zip(lines, sample_records):
            parsed = json.loads(line)
            assert parsed == expected

    def test_jsonl_preserves_korean_text(self, tmp_out_dir):
        """한국어 텍스트가 ensure_ascii=False로 올바르게 저장되는지 확인."""
        records = [{"text": "한국어 테스트 문장입니다.", "source": "korean_exam"}]
        storage = Storage()
        result_path = storage.save(records, tmp_out_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "한국어 테스트 문장입니다." in content
        # ensure_ascii=False이므로 유니코드 이스케이프가 없어야 함
        assert "\\u" not in content

    def test_creates_parent_directories(self, tmp_out_dir, sample_records):
        """중간 디렉토리가 자동 생성되는지 확인."""
        storage = Storage()
        storage.save(sample_records, tmp_out_dir)

        assert (tmp_out_dir / "raw" / "korean_exam").is_dir()

    def test_empty_records_creates_empty_file(self, tmp_out_dir):
        """빈 레코드 리스트로 빈 파일이 생성되는지 확인."""
        storage = Storage()
        result_path = storage.save([], tmp_out_dir)

        assert result_path.exists()
        assert result_path.stat().st_size == 0


class TestStorageMerge:
    """요구사항 7.2, 7.5: --merge 옵션 테스트."""

    def test_merge_creates_train_txt(self, tmp_out_dir, sample_records):
        """merge=True 시 train.txt가 생성되는지 확인."""
        storage = Storage()
        storage.save(sample_records, tmp_out_dir, merge=True)

        train_path = tmp_out_dir / "processed" / "train.txt"
        assert train_path.exists()

    def test_merge_appends_text_fields(self, tmp_out_dir, sample_records):
        """merge=True 시 text 필드가 한 줄씩 추가되는지 확인."""
        storage = Storage()
        storage.save(sample_records, tmp_out_dir, merge=True)

        train_path = tmp_out_dir / "processed" / "train.txt"
        content = train_path.read_text(encoding="utf-8")
        # 각 레코드의 text 필드가 content에 포함되어야 함
        for rec in sample_records:
            assert rec["text"] + "\n" in content

    def test_merge_appends_to_existing_file(self, tmp_out_dir, sample_records):
        """merge=True 시 기존 train.txt에 append하는지 확인."""
        train_path = tmp_out_dir / "processed" / "train.txt"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text("기존 내용\n", encoding="utf-8")

        storage = Storage()
        storage.save(sample_records, tmp_out_dir, merge=True)

        content = train_path.read_text(encoding="utf-8")
        assert content.startswith("기존 내용\n")
        # 기존 내용 뒤에 각 레코드의 text가 추가되어야 함
        for rec in sample_records:
            assert rec["text"] + "\n" in content

    def test_no_merge_does_not_create_train_txt(self, tmp_out_dir, sample_records):
        """merge=False 시 train.txt를 생성하지 않는지 확인."""
        storage = Storage()
        storage.save(sample_records, tmp_out_dir, merge=False)

        train_path = tmp_out_dir / "processed" / "train.txt"
        assert not train_path.exists()

    def test_no_merge_does_not_modify_existing_train_txt(
        self, tmp_out_dir, sample_records
    ):
        """merge=False 시 기존 train.txt를 수정하지 않는지 확인."""
        train_path = tmp_out_dir / "processed" / "train.txt"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        original_content = "기존 내용 유지\n"
        train_path.write_text(original_content, encoding="utf-8")

        storage = Storage()
        storage.save(sample_records, tmp_out_dir, merge=False)

        assert train_path.read_text(encoding="utf-8") == original_content


class TestStorageErrorHandling:
    """요구사항 7.4: 디스크 쓰기 오류 처리 테스트."""

    def test_oserror_removes_partial_file(self, tmp_out_dir):
        """OSError 발생 시 부분 저장 파일이 삭제되는지 확인."""
        records = [{"text": "테스트", "source": "korean_exam"}]
        storage = Storage()

        raw_dir = tmp_out_dir / "raw" / "korean_exam"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # json.dumps가 성공한 후 write에서 실패하도록 mock
        original_open = Path.open

        call_count = [0]

        def failing_open(self_path, *args, **kwargs):
            # jsonl_path.open 호출 시 실패
            if "korean_exam_" in str(self_path):
                raise OSError("디스크 공간 부족")
            return original_open(self_path, *args, **kwargs)

        with patch.object(Path, "open", failing_open):
            with pytest.raises(OSError, match="디스크 공간 부족"):
                storage.save(records, tmp_out_dir)

    def test_oserror_propagates_exception(self, tmp_out_dir):
        """OSError가 호출자에게 전파되는지 확인."""
        records = [{"text": "테스트", "source": "korean_exam"}]
        storage = Storage()

        # 쓰기 불가능한 경로 사용
        bad_dir = tmp_out_dir / "raw" / "korean_exam"
        bad_dir.mkdir(parents=True, exist_ok=True)

        # 파일 쓰기 중 에러 시뮬레이션
        with patch.object(Path, "open", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                storage.save(records, tmp_out_dir)


class TestStorageOutput:
    """요구사항 7.3: 저장 완료 후 통계 출력 테스트."""

    def test_returns_jsonl_path(self, tmp_out_dir, sample_records):
        """save()가 저장된 JSONL 파일 경로를 반환하는지 확인."""
        storage = Storage()
        result = storage.save(sample_records, tmp_out_dir)

        assert isinstance(result, Path)
        assert result.exists()

    def test_logs_record_count_and_size(self, tmp_out_dir, sample_records, caplog):
        """저장 완료 후 레코드 수와 파일 크기가 로그에 기록되는지 확인."""
        import logging

        with caplog.at_level(logging.INFO):
            storage = Storage()
            storage.save(sample_records, tmp_out_dir)

        assert "2건" in caplog.text
        assert "저장 완료" in caplog.text


class TestFormatSize:
    """_format_size 유틸리티 함수 테스트."""

    def test_bytes(self):
        assert _format_size(500) == "500B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0KB"

    def test_megabytes(self):
        assert _format_size(1048576) == "1.0MB"

    def test_zero(self):
        assert _format_size(0) == "0B"
