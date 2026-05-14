"""PDFCollector 및 사이트별 크롤러 구현."""

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Tuple

import requests
from bs4 import BeautifulSoup

from .base import BaseCollector

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 요청 간 대기 시간 (초)
REQUEST_DELAY = 1.0


class BaseCrawler(ABC):
    """크롤러 기본 추상 클래스."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0
        self.total_size = 0

    @abstractmethod
    def crawl(self) -> Iterator[Path]:
        """PDF를 크롤링하고 다운로드하여 경로를 yield."""
        ...

    def _download_pdf(self, url: str, dest: Path) -> bool:
        """PDF를 다운로드한다. 캐시 히트 시 건너뜀.

        Args:
            url: 다운로드 URL
            dest: 저장 경로

        Returns:
            bool: 성공 여부
        """
        if self._is_cached(dest):
            self.skip_count += 1
            return True
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            self.success_count += 1
            self.total_size += len(resp.content)
            return True
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"다운로드 실패: {url} - {e}")
            self.fail_count += 1
            return False

    def _is_cached(self, path: Path) -> bool:
        """파일이 이미 캐시되어 있는지 확인."""
        return path.exists() and path.stat().st_size > 0

    def _get_page(self, url: str) -> BeautifulSoup | None:
        """URL에서 HTML 페이지를 가져와 BeautifulSoup 객체로 반환."""
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"페이지 로드 실패: {url} - {e}")
            return None

    def print_stats(self) -> None:
        """크롤링 통계를 출력한다."""
        size_mb = self.total_size / (1024 * 1024)
        print(
            f"  성공: {self.success_count}건, "
            f"실패: {self.fail_count}건, "
            f"건너뜀(캐시): {self.skip_count}건, "
            f"총 크기: {size_mb:.1f}MB"
        )


class CSATCrawler(BaseCrawler):
    """수능 기출 PDF 크롤러 (suneung.re.kr)."""

    BASE_URL = "https://www.suneung.re.kr"
    BOARD_URL = (
        "https://www.suneung.re.kr/boardCnts/list.do"
        "?boardID=1500234&m=0403&s=suneung"
    )

    def __init__(self, out_dir: Path):
        super().__init__(out_dir / "pdf" / "csat")

    def crawl(self) -> Iterator[Path]:
        """수능 국어 PDF를 크롤링하여 다운로드."""
        logger.info("수능 기출 PDF 크롤링 시작 (suneung.re.kr)")
        page_num = 1
        max_pages = 20  # 안전 제한

        while page_num <= max_pages:
            page_url = f"{self.BOARD_URL}&page={page_num}"
            soup = self._get_page(page_url)
            if soup is None:
                break

            rows = soup.select("table tbody tr")
            if not rows:
                break

            for row in rows:
                links = self._extract_pdf_links(row)
                for url, filename, year in links:
                    dest = self.out_dir / str(year) / filename
                    if self._download_pdf(url, dest):
                        yield dest

            page_num += 1

    def _extract_pdf_links(
        self, row
    ) -> List[Tuple[str, str, int]]:
        """게시판 행에서 국어 영역 PDF 링크를 추출."""
        results = []
        # 제목에서 연도 추출 시도
        title_cell = row.select_one("td.title a, td a")
        if not title_cell:
            return results

        title_text = title_cell.get_text(strip=True)

        # 국어 영역 관련 키워드 확인
        if not any(kw in title_text for kw in ["국어", "언어"]):
            return results

        # 연도 추출
        year_match = re.search(r"(\d{4})", title_text)
        year = int(year_match.group(1)) if year_match else 0

        # 첨부파일 링크 추출
        file_links = row.select("a[href*='.pdf'], a[href*='download']")
        for link in file_links:
            href = link.get("href", "")
            if href:
                full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                filename = link.get_text(strip=True) or f"csat_{year}.pdf"
                if not filename.endswith(".pdf"):
                    filename += ".pdf"
                results.append((full_url, filename, year))

        return results


class MockCrawler(BaseCrawler):
    """평가원 모의고사 PDF 크롤러 (kice.re.kr)."""

    BASE_URL = "https://www.kice.re.kr"
    BOARD_URL = (
        "https://www.kice.re.kr/boardCnts/list.do"
        "?boardID=1500212&m=030306&s=kice"
    )

    def __init__(self, out_dir: Path):
        super().__init__(out_dir / "pdf" / "mock")

    def crawl(self) -> Iterator[Path]:
        """모의고사 국어 PDF를 크롤링하여 다운로드."""
        logger.info("평가원 모의고사 PDF 크롤링 시작 (kice.re.kr)")
        page_num = 1
        max_pages = 15

        while page_num <= max_pages:
            page_url = f"{self.BOARD_URL}&page={page_num}"
            soup = self._get_page(page_url)
            if soup is None:
                break

            rows = soup.select("table tbody tr")
            if not rows:
                break

            found_any = False
            for row in rows:
                links = self._extract_pdf_links(row)
                for url, filename, year in links:
                    dest = self.out_dir / str(year) / filename
                    if self._download_pdf(url, dest):
                        yield dest
                    found_any = True

            if not found_any:
                break
            page_num += 1

    def _extract_pdf_links(
        self, row
    ) -> List[Tuple[str, str, int]]:
        """게시판 행에서 모의고사 국어 PDF 링크를 추출."""
        results = []
        title_cell = row.select_one("td.title a, td a")
        if not title_cell:
            return results

        title_text = title_cell.get_text(strip=True)

        # 국어 과목 확인
        if not any(kw in title_text for kw in ["국어", "언어"]):
            return results

        # 연도 추출
        year_match = re.search(r"(\d{4})", title_text)
        year = int(year_match.group(1)) if year_match else 0

        # 월 추출 (6월/9월)
        month_match = re.search(r"(\d{1,2})월", title_text)
        month = int(month_match.group(1)) if month_match else 0

        # 첨부파일 링크
        file_links = row.select("a[href*='.pdf'], a[href*='download']")
        for link in file_links:
            href = link.get("href", "")
            if href:
                full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                filename = f"mock_{year}_{month:02d}.pdf"
                results.append((full_url, filename, year))

        return results


class TeacherCrawler(BaseCrawler):
    """임용고시 PDF 크롤러 (kice.re.kr)."""

    BASE_URL = "https://www.kice.re.kr"
    BOARD_URL = (
        "https://www.kice.re.kr/boardCnts/list.do"
        "?boardID=1500212&m=030306&s=kice"
    )

    def __init__(self, out_dir: Path):
        super().__init__(out_dir / "pdf" / "teacher")

    def crawl(self) -> Iterator[Path]:
        """임용고시 국어(1차) PDF를 크롤링하여 다운로드."""
        logger.info("임용고시 PDF 크롤링 시작 (kice.re.kr)")
        # 국어(1차) 검색
        search_url = f"{self.BOARD_URL}&searchStr=국어"
        page_num = 1
        max_pages = 10

        while page_num <= max_pages:
            page_url = f"{search_url}&page={page_num}"
            soup = self._get_page(page_url)
            if soup is None:
                break

            rows = soup.select("table tbody tr")
            if not rows:
                break

            found_any = False
            for row in rows:
                links = self._extract_pdf_links(row)
                for url, filename, year in links:
                    dest = self.out_dir / str(year) / filename
                    if self._download_pdf(url, dest):
                        yield dest
                    found_any = True

            if not found_any:
                break
            page_num += 1

    def _extract_pdf_links(
        self, row
    ) -> List[Tuple[str, str, int]]:
        """게시판 행에서 임용고시 국어 PDF 링크를 추출."""
        results = []
        title_cell = row.select_one("td.title a, td a")
        if not title_cell:
            return results

        title_text = title_cell.get_text(strip=True)

        # 임용고시 국어(1차) 확인
        if "국어" not in title_text:
            return results

        # 연도 추출
        year_match = re.search(r"(\d{4})", title_text)
        year = int(year_match.group(1)) if year_match else 0

        # 2019년 이후만 수집
        if year > 0 and year < 2019:
            return results

        # 첨부파일 링크
        file_links = row.select("a[href*='.pdf'], a[href*='download']")
        for link in file_links:
            href = link.get("href", "")
            if href:
                full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                filename = f"teacher_{year}.pdf"
                results.append((full_url, filename, year))

        return results


class LEETCrawler(BaseCrawler):
    """LEET 언어이해 PDF 크롤러 (moja.uwayapply.com)."""

    BASE_URL = "https://moja.uwayapply.com"

    def __init__(self, out_dir: Path):
        super().__init__(out_dir / "pdf" / "leet")

    def crawl(self) -> Iterator[Path]:
        """LEET 언어이해 PDF를 크롤링하여 다운로드."""
        logger.info("LEET 언어이해 PDF 크롤링 시작 (moja.uwayapply.com)")
        # iframe 기반 사이트: center.htm → 메뉴 → 기출 목록
        center_url = f"{self.BASE_URL}/center.htm"
        soup = self._get_page(center_url)
        if soup is None:
            logger.warning("LEET 사이트 접근 실패")
            return

        # 기출문제 메뉴 링크 탐색
        menu_links = soup.select("a[href]")
        exam_links = []
        for link in menu_links:
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "기출" in text or "언어" in text or "이해" in text:
                full_url = (
                    href if href.startswith("http") else f"{self.BASE_URL}/{href}"
                )
                exam_links.append(full_url)

        # 기출 목록 페이지에서 PDF 링크 추출
        for exam_url in exam_links:
            soup = self._get_page(exam_url)
            if soup is None:
                continue

            pdf_links = soup.select("a[href*='.pdf']")
            for pdf_link in pdf_links:
                href = pdf_link.get("href", "")
                text = pdf_link.get_text(strip=True)

                # 언어이해 영역만 필터링
                if not any(kw in text for kw in ["언어이해", "언어", "이해"]):
                    continue

                full_url = (
                    href if href.startswith("http") else f"{self.BASE_URL}/{href}"
                )

                # 연도 추출
                year_match = re.search(r"(\d{4})", text) or re.search(
                    r"(\d{4})", href
                )
                year = int(year_match.group(1)) if year_match else 0

                filename = f"leet_{year}.pdf"
                dest = self.out_dir / str(year) / filename
                if self._download_pdf(full_url, dest):
                    yield dest


class DistrictCrawler(BaseCrawler):
    """교육청 모의고사 PDF 크롤러 (서울·경기·인천)."""

    DISTRICTS = {
        "seoul": {
            "name": "서울",
            "base_url": "https://www.sen.go.kr",
            "board_path": "/web/services/brd/selectBrdList.do",
        },
        "gyeonggi": {
            "name": "경기",
            "base_url": "https://www.goe.go.kr",
            "board_path": "/web/services/brd/selectBrdList.do",
        },
        "incheon": {
            "name": "인천",
            "base_url": "https://www.ice.go.kr",
            "board_path": "/web/services/brd/selectBrdList.do",
        },
    }

    # 모의고사 시행 월
    EXAM_MONTHS = [3, 4, 7, 10]

    def __init__(self, out_dir: Path):
        super().__init__(out_dir / "pdf" / "district")

    def crawl(self) -> Iterator[Path]:
        """교육청 모의고사 국어 PDF를 크롤링하여 다운로드."""
        logger.info("교육청 모의고사 PDF 크롤링 시작 (서울·경기·인천)")

        for region_key, config in self.DISTRICTS.items():
            logger.info(f"  {config['name']} 교육청 크롤링...")
            board_url = f"{config['base_url']}{config['board_path']}"

            soup = self._get_page(board_url)
            if soup is None:
                continue

            # 게시판에서 국어 모의고사 PDF 링크 추출
            rows = soup.select("table tbody tr, li.board-item, div.board-list-item")
            for row in rows:
                links = self._extract_pdf_links(row, config)
                for url, filename, year in links:
                    dest = self.out_dir / region_key / str(year) / filename
                    if self._download_pdf(url, dest):
                        yield dest

    def _extract_pdf_links(
        self, row, config: dict
    ) -> List[Tuple[str, str, int]]:
        """게시판 행에서 교육청 모의고사 국어 PDF 링크를 추출."""
        results = []
        title_el = row.select_one("a, td.title a")
        if not title_el:
            return results

        title_text = title_el.get_text(strip=True)

        # 국어 과목 확인
        if "국어" not in title_text:
            return results

        # 연도 추출
        year_match = re.search(r"(\d{4})", title_text)
        year = int(year_match.group(1)) if year_match else 0

        # 월 추출
        month_match = re.search(r"(\d{1,2})월", title_text)
        month = int(month_match.group(1)) if month_match else 0

        # 첨부파일 링크
        file_links = row.select("a[href*='.pdf'], a[href*='download']")
        for link in file_links:
            href = link.get("href", "")
            if href:
                full_url = (
                    href
                    if href.startswith("http")
                    else f"{config['base_url']}{href}"
                )
                filename = f"district_{config['name']}_{year}_{month:02d}.pdf"
                results.append((full_url, filename, year))

        return results


class PDFCollector(BaseCollector):
    """PDF 크롤링 기반 수집기.

    여러 크롤러를 조합하여 PDF를 다운로드하고 경로를 yield합니다.
    """

    # exam 파라미터 → 크롤러 매핑
    CRAWLER_MAP = {
        "csat": CSATCrawler,
        "mock": MockCrawler,
        "teacher": TeacherCrawler,
        "leet": LEETCrawler,
        "district": DistrictCrawler,
    }

    def __init__(self, out_dir: Path, exam: str = "all"):
        """PDFCollector 초기화.

        Args:
            out_dir: 출력 디렉토리 (data/raw/korean_exam/)
            exam: 수집할 시험 종류 ("csat", "mock", "teacher", "leet", "district", "all")
        """
        super().__init__(out_dir)
        self.exam = exam

    def collect(self) -> Iterator[Path]:
        """PDF를 크롤링하고 다운로드하여 경로를 yield.

        Returns:
            Iterator[Path]: 다운로드된 PDF 파일 경로들
        """
        crawlers = self._build_crawlers()

        print("\n" + "=" * 50)
        print("📥 PDF 크롤링 시작")
        print("=" * 50)

        total_success = 0
        total_fail = 0
        total_skip = 0
        total_size = 0

        for crawler in crawlers:
            crawler_name = type(crawler).__name__
            logger.info(f"{crawler_name} 실행 중...")

            for path in crawler.crawl():
                yield path

            crawler.print_stats()
            total_success += crawler.success_count
            total_fail += crawler.fail_count
            total_skip += crawler.skip_count
            total_size += crawler.total_size

        # 전체 통계 출력
        size_mb = total_size / (1024 * 1024)
        print("\n" + "-" * 50)
        print("📊 PDF 크롤링 완료")
        print(f"  총 성공: {total_success}건")
        print(f"  총 실패: {total_fail}건")
        print(f"  총 건너뜀(캐시): {total_skip}건")
        print(f"  총 파일 크기: {size_mb:.1f}MB")
        print("=" * 50)

    def _build_crawlers(self) -> List[BaseCrawler]:
        """exam 파라미터에 따라 크롤러 목록을 생성."""
        if self.exam == "all":
            return [cls(self.out_dir) for cls in self.CRAWLER_MAP.values()]
        elif self.exam in self.CRAWLER_MAP:
            return [self.CRAWLER_MAP[self.exam](self.out_dir)]
        else:
            logger.error(f"지원하지 않는 시험 종류: {self.exam}")
            return []
