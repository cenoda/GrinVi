"""Unit tests for grinvi.korean_exam.collectors.pdf_collector."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
_spec_base = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.base",
    _base_dir / "collectors" / "base.py",
)
_base = importlib.util.module_from_spec(_spec_base)
sys.modules["grinvi.korean_exam.collectors.base"] = _base
_spec_base.loader.exec_module(_base)

# Mock requests and bs4 before loading pdf_collector
mock_requests = MagicMock()
mock_bs4 = MagicMock()
sys.modules.setdefault("requests", mock_requests)
sys.modules.setdefault("bs4", mock_bs4)

# Load collectors.pdf_collector
_spec_pdf = importlib.util.spec_from_file_location(
    "grinvi.korean_exam.collectors.pdf_collector",
    _base_dir / "collectors" / "pdf_collector.py",
)
_pdf = importlib.util.module_from_spec(_spec_pdf)
sys.modules["grinvi.korean_exam.collectors.pdf_collector"] = _pdf

# Need to reload with real imports for testing
# Remove mocks and use actual imports
del sys.modules["requests"]
del sys.modules["bs4"]

_spec_pdf.loader.exec_module(_pdf)

PDFCollector = _pdf.PDFCollector
CSATCrawler = _pdf.CSATCrawler
MockCrawler = _pdf.MockCrawler
TeacherCrawler = _pdf.TeacherCrawler
LEETCrawler = _pdf.LEETCrawler
DistrictCrawler = _pdf.DistrictCrawler
BaseCrawler = _pdf.BaseCrawler


class TestPDFCollector:
    def test_init_with_exam_all(self, tmp_path):
        """exam='all'로 초기화하면 모든 크롤러를 생성."""
        collector = PDFCollector(out_dir=tmp_path, exam="all")
        assert collector.exam == "all"
        crawlers = collector._build_crawlers()
        assert len(crawlers) == 5

    def test_init_with_specific_exam(self, tmp_path):
        """특정 exam으로 초기화하면 해당 크롤러만 생성."""
        collector = PDFCollector(out_dir=tmp_path, exam="csat")
        crawlers = collector._build_crawlers()
        assert len(crawlers) == 1
        assert isinstance(crawlers[0], CSATCrawler)

    def test_init_with_invalid_exam(self, tmp_path):
        """잘못된 exam 값이면 빈 크롤러 목록."""
        collector = PDFCollector(out_dir=tmp_path, exam="invalid")
        crawlers = collector._build_crawlers()
        assert len(crawlers) == 0

    def test_crawler_map_has_all_types(self):
        """CRAWLER_MAP에 5개 시험 종류가 정의되어 있다."""
        assert "csat" in PDFCollector.CRAWLER_MAP
        assert "mock" in PDFCollector.CRAWLER_MAP
        assert "teacher" in PDFCollector.CRAWLER_MAP
        assert "leet" in PDFCollector.CRAWLER_MAP
        assert "district" in PDFCollector.CRAWLER_MAP

    def test_out_dir_stored(self, tmp_path):
        """out_dir이 올바르게 저장된다."""
        collector = PDFCollector(out_dir=tmp_path, exam="csat")
        assert collector.out_dir == tmp_path


class TestBaseCrawler:
    def test_download_pdf_cached(self, tmp_path):
        """캐시된 파일은 다운로드하지 않는다."""
        crawler = CSATCrawler(tmp_path)
        dest = tmp_path / "pdf" / "csat" / "test.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PDF content")

        result = crawler._download_pdf("http://example.com/test.pdf", dest)
        assert result is True
        assert crawler.skip_count == 1
        assert crawler.success_count == 0

    def test_download_pdf_success(self, tmp_path):
        """PDF 다운로드 성공."""
        import requests

        crawler = CSATCrawler(tmp_path)
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 test content"
        mock_response.raise_for_status = MagicMock()

        with patch.object(requests, "get", return_value=mock_response):
            dest = tmp_path / "pdf" / "csat" / "2024" / "test.pdf"
            result = crawler._download_pdf("http://example.com/test.pdf", dest)

        assert result is True
        assert dest.exists()
        assert crawler.success_count == 1
        assert crawler.total_size == len(b"%PDF-1.4 test content")

    def test_download_pdf_timeout(self, tmp_path):
        """타임아웃 시 실패 처리."""
        import requests

        crawler = CSATCrawler(tmp_path)

        with patch.object(
            requests, "get", side_effect=requests.Timeout("Connection timed out")
        ):
            dest = tmp_path / "pdf" / "csat" / "2024" / "test.pdf"
            result = crawler._download_pdf("http://example.com/test.pdf", dest)

        assert result is False
        assert crawler.fail_count == 1
        assert not dest.exists()

    def test_download_pdf_http_error(self, tmp_path):
        """HTTP 에러 시 실패 처리."""
        import requests

        crawler = CSATCrawler(tmp_path)

        with patch.object(
            requests, "get", side_effect=requests.HTTPError("404 Not Found")
        ):
            dest = tmp_path / "pdf" / "csat" / "2024" / "test.pdf"
            result = crawler._download_pdf("http://example.com/test.pdf", dest)

        assert result is False
        assert crawler.fail_count == 1

    def test_is_cached_nonexistent(self, tmp_path):
        """존재하지 않는 파일은 캐시되지 않음."""
        crawler = CSATCrawler(tmp_path)
        assert crawler._is_cached(tmp_path / "nonexistent.pdf") is False

    def test_is_cached_empty_file(self, tmp_path):
        """빈 파일은 캐시되지 않음."""
        crawler = CSATCrawler(tmp_path)
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        assert crawler._is_cached(empty) is False

    def test_is_cached_valid_file(self, tmp_path):
        """내용이 있는 파일은 캐시됨."""
        crawler = CSATCrawler(tmp_path)
        valid = tmp_path / "valid.pdf"
        valid.write_bytes(b"PDF content")
        assert crawler._is_cached(valid) is True

    def test_print_stats(self, tmp_path, capsys):
        """통계 출력이 올바르게 동작."""
        crawler = CSATCrawler(tmp_path)
        crawler.success_count = 5
        crawler.fail_count = 2
        crawler.skip_count = 3
        crawler.total_size = 1024 * 1024 * 2  # 2MB

        crawler.print_stats()
        captured = capsys.readouterr()
        assert "5건" in captured.out
        assert "2건" in captured.out
        assert "3건" in captured.out
        assert "2.0MB" in captured.out


class TestCSATCrawler:
    def test_init_sets_correct_out_dir(self, tmp_path):
        """CSATCrawler의 out_dir이 올바르게 설정된다."""
        crawler = CSATCrawler(tmp_path)
        assert crawler.out_dir == tmp_path / "pdf" / "csat"

    def test_extract_pdf_links_non_korean(self, tmp_path):
        """국어가 아닌 행은 빈 리스트 반환."""
        from bs4 import BeautifulSoup

        html = '<tr><td class="title"><a href="/test">2024 영어 영역</a></td></tr>'
        soup = BeautifulSoup(html, "html.parser")
        row = soup.find("tr")

        crawler = CSATCrawler(tmp_path)
        links = crawler._extract_pdf_links(row)
        assert links == []

    def test_extract_pdf_links_korean(self, tmp_path):
        """국어 영역 행에서 PDF 링크를 추출."""
        from bs4 import BeautifulSoup

        html = (
            '<tr>'
            '<td class="title"><a href="/view/123">2024학년도 국어 영역</a></td>'
            '<td><a href="/download/test.pdf">문제지</a></td>'
            '</tr>'
        )
        soup = BeautifulSoup(html, "html.parser")
        row = soup.find("tr")

        crawler = CSATCrawler(tmp_path)
        links = crawler._extract_pdf_links(row)
        assert len(links) == 1
        assert links[0][2] == 2024  # year


class TestMockCrawler:
    def test_init_sets_correct_out_dir(self, tmp_path):
        """MockCrawler의 out_dir이 올바르게 설정된다."""
        crawler = MockCrawler(tmp_path)
        assert crawler.out_dir == tmp_path / "pdf" / "mock"


class TestTeacherCrawler:
    def test_init_sets_correct_out_dir(self, tmp_path):
        """TeacherCrawler의 out_dir이 올바르게 설정된다."""
        crawler = TeacherCrawler(tmp_path)
        assert crawler.out_dir == tmp_path / "pdf" / "teacher"

    def test_extract_pdf_links_old_year(self, tmp_path):
        """2019년 이전 데이터는 건너뛴다."""
        from bs4 import BeautifulSoup

        html = (
            '<tr>'
            '<td class="title"><a href="/view/123">2017학년도 국어 1차</a></td>'
            '<td><a href="/download/test.pdf">문제지</a></td>'
            '</tr>'
        )
        soup = BeautifulSoup(html, "html.parser")
        row = soup.find("tr")

        crawler = TeacherCrawler(tmp_path)
        links = crawler._extract_pdf_links(row)
        assert links == []


class TestLEETCrawler:
    def test_init_sets_correct_out_dir(self, tmp_path):
        """LEETCrawler의 out_dir이 올바르게 설정된다."""
        crawler = LEETCrawler(tmp_path)
        assert crawler.out_dir == tmp_path / "pdf" / "leet"


class TestDistrictCrawler:
    def test_init_sets_correct_out_dir(self, tmp_path):
        """DistrictCrawler의 out_dir이 올바르게 설정된다."""
        crawler = DistrictCrawler(tmp_path)
        assert crawler.out_dir == tmp_path / "pdf" / "district"

    def test_districts_config(self):
        """3개 교육청이 설정되어 있다."""
        assert len(DistrictCrawler.DISTRICTS) == 3
        assert "seoul" in DistrictCrawler.DISTRICTS
        assert "gyeonggi" in DistrictCrawler.DISTRICTS
        assert "incheon" in DistrictCrawler.DISTRICTS

    def test_exam_months(self):
        """모의고사 시행 월이 올바르게 설정."""
        assert DistrictCrawler.EXAM_MONTHS == [3, 4, 7, 10]
