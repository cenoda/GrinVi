"""Unit tests for grinvi.korean_exam.collectors (BaseCollector, HFCollector)."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# Load collectors.base
_spec_base = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.base",
    _base_dir / "collectors" / "base.py",
)
_base = importlib.util.module_from_spec(_spec_base)
sys.modules["grinvi.korean_exam.collectors.base"] = _base
_spec_base.loader.exec_module(_base)

# Load collectors.hf_collector
_spec_hf = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.hf_collector",
    _base_dir / "collectors" / "hf_collector.py",
)
_hf = importlib.util.module_from_spec(_spec_hf)
sys.modules["grinvi.korean_exam.collectors.hf_collector"] = _hf
_spec_hf.loader.exec_module(_hf)

BaseCollector = _base.BaseCollector
HFCollector = _hf.HFCollector
HF_DATASETS = _hf.HF_DATASETS


class TestBaseCollector:
    def test_is_abstract(self):
        """BaseCollector는 직접 인스턴스화할 수 없다."""
        import pytest

        with pytest.raises(TypeError):
            BaseCollector(Path("/tmp"))

    def test_is_cached_nonexistent_file(self, tmp_path):
        """존재하지 않는 파일은 캐시되지 않은 것으로 판단."""

        class ConcreteCollector(BaseCollector):
            def collect(self):
                yield from []

        collector = ConcreteCollector(tmp_path)
        assert collector._is_cached(tmp_path / "nonexistent.jsonl") is False

    def test_is_cached_empty_file(self, tmp_path):
        """빈 파일은 캐시되지 않은 것으로 판단."""

        class ConcreteCollector(BaseCollector):
            def collect(self):
                yield from []

        collector = ConcreteCollector(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        assert collector._is_cached(empty_file) is False

    def test_is_cached_valid_file(self, tmp_path):
        """내용이 있는 파일은 캐시된 것으로 판단."""

        class ConcreteCollector(BaseCollector):
            def collect(self):
                yield from []

        collector = ConcreteCollector(tmp_path)
        valid_file = tmp_path / "data.jsonl"
        valid_file.write_text('{"key": "value"}\n')
        assert collector._is_cached(valid_file) is True

    def test_out_dir_stored(self, tmp_path):
        """out_dir이 올바르게 저장된다."""

        class ConcreteCollector(BaseCollector):
            def collect(self):
                yield from []

        collector = ConcreteCollector(tmp_path)
        assert collector.out_dir == tmp_path


class TestHFCollector:
    def test_hf_datasets_config_has_5_entries(self):
        """HF_DATASETS에 5개 데이터셋이 정의되어 있다."""
        assert len(HF_DATASETS) == 5

    def test_hf_datasets_ids(self):
        """모든 데이터셋 ID가 올바르게 정의되어 있다."""
        ids = [ds["id"] for ds in HF_DATASETS]
        assert "csatqa" in ids
        assert "csat_2025" in ids
        assert "csat_sft" in ids
        assert "civil_local" in ids
        assert "civil_national" in ids

    def test_csatqa_uses_url_method(self):
        """csatqa는 url 방식으로 수집한다."""
        csatqa = next(ds for ds in HF_DATASETS if ds["id"] == "csatqa")
        assert csatqa["method"] == "url"
        assert "url" in csatqa
        assert "url_eval" in csatqa

    def test_kikikara_has_filter(self):
        """kikikara 데이터셋은 task='국어' 필터가 있다."""
        civil_local = next(ds for ds in HF_DATASETS if ds["id"] == "civil_local")
        assert civil_local["filter"] == {"task": "국어"}

        civil_national = next(
            ds for ds in HF_DATASETS if ds["id"] == "civil_national"
        )
        assert civil_national["filter"] == {"task": "국어"}

    def test_civil_national_split(self):
        """civil_national은 '공무원_국가직' split을 사용한다."""
        civil_national = next(
            ds for ds in HF_DATASETS if ds["id"] == "civil_national"
        )
        assert civil_national["split"] == "공무원_국가직"

    def test_collect_uses_cache(self, tmp_path):
        """이미 캐시된 파일이 있으면 재다운로드하지 않는다."""
        out_dir = tmp_path / "raw" / "korean_exam"
        collector = HFCollector(out_dir)

        # 모든 데이터셋에 대해 캐시 파일 생성
        for ds in HF_DATASETS:
            cache_path = out_dir / "huggingface" / ds["id"] / "raw.jsonl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('{"test": true}\n')

        # collect 실행 - 네트워크 호출 없이 캐시 사용
        paths = list(collector.collect())
        assert len(paths) == 5
        for p in paths:
            assert p.exists()

    @patch("requests.get")
    def test_collect_url_method(self, mock_get, tmp_path):
        """URL 방식 수집이 올바르게 동작한다."""
        out_dir = tmp_path / "raw" / "korean_exam"
        collector = HFCollector(out_dir)

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"question": "테스트 문제", "context": "지문", "gold": 1}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        records = collector._collect_url(HF_DATASETS[0])
        assert len(records) == 2  # url + url_eval 각 1건
        assert mock_get.call_count == 2

    def test_save_jsonl(self, tmp_path):
        """JSONL 저장이 올바르게 동작한다."""
        out_dir = tmp_path / "raw" / "korean_exam"
        collector = HFCollector(out_dir)

        records = [
            {"question": "문제1", "answer": 1},
            {"question": "문제2", "answer": 2},
        ]
        out_path = tmp_path / "test" / "raw.jsonl"
        collector._save_jsonl(records, out_path)

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["question"] == "문제1"
        assert json.loads(lines[1])["question"] == "문제2"

    @patch("requests.get")
    def test_collect_network_error_exits(self, mock_get, tmp_path):
        """네트워크 오류 시 sys.exit(1)을 호출한다."""
        import pytest

        out_dir = tmp_path / "raw" / "korean_exam"
        collector = HFCollector(out_dir)

        mock_get.side_effect = Exception("Connection timeout")

        with pytest.raises(SystemExit) as exc_info:
            list(collector.collect())
        assert exc_info.value.code == 1

    def test_collect_load_dataset_with_filter(self, tmp_path):
        """load_dataset 방식에서 필터가 올바르게 적용된다."""
        out_dir = tmp_path / "raw" / "korean_exam"
        collector = HFCollector(out_dir)

        # Mock dataset rows (simulating what load_dataset returns)
        mock_dataset = [
            {"문제 내용": "국어 문제", "task": "국어", "year": 2024},
            {"문제 내용": "영어 문제", "task": "영어", "year": 2024},
            {"문제 내용": "국어 문제2", "task": "국어", "year": 2023},
        ]

        ds_config = {
            "id": "test",
            "hf_id": "test/dataset",
            "method": "load_dataset",
            "split": "train",
            "filter": {"task": "국어"},
        }

        # Mock the datasets module's load_dataset function
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset = mock_load_dataset

        with patch.dict(sys.modules, {"datasets": mock_datasets_module}):
            records = collector._collect_load_dataset(ds_config)
            assert len(records) == 2
            assert all(r["task"] == "국어" for r in records)
