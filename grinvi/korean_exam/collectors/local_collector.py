"""LocalCollector: 로컬 파일/디렉토리 수집기."""

import logging
from pathlib import Path
from typing import Iterator

from .base import BaseCollector

logger = logging.getLogger(__name__)

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}


class LocalCollector(BaseCollector):
    """로컬 파일 시스템에서 데이터를 수집하는 Collector.

    지정된 경로의 파일 또는 디렉토리를 읽어 파일 경로를 yield합니다.
    """

    def __init__(self, out_dir: Path, input_path: str = None):
        """LocalCollector 초기화.

        Args:
            out_dir: 출력 디렉토리
            input_path: 입력 파일 또는 디렉토리 경로
        """
        super().__init__(out_dir)
        self.input_path = input_path

    def collect(self) -> Iterator[Path]:
        """로컬 파일을 수집하여 경로를 yield.

        Returns:
            Iterator[Path]: 수집된 파일 경로들

        Raises:
            SystemExit: 경로가 존재하지 않거나 미지원 형식인 경우
        """
        if not self.input_path:
            logger.error("입력 경로가 지정되지 않았습니다. --input 옵션을 사용하세요.")
            print("❌ 오류: 입력 경로가 지정되지 않았습니다. --input 옵션을 사용하세요.")
            return

        path = Path(self.input_path)

        if not path.exists():
            logger.error(f"경로가 존재하지 않습니다: {path}")
            print(f"❌ 오류: 경로가 존재하지 않습니다: {path}")
            return

        if path.is_file():
            # 단일 파일 처리
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.error(
                    f"지원하지 않는 파일 형식입니다: {path.suffix} "
                    f"(지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
                )
                print(
                    f"❌ 오류: 지원하지 않는 파일 형식입니다: {path.suffix}\n"
                    f"   지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
                return
            yield path

        elif path.is_dir():
            # 디렉토리 재귀 탐색
            found_count = 0
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield file_path
                    found_count += 1

            if found_count == 0:
                logger.warning(
                    f"디렉토리에서 지원 형식의 파일을 찾을 수 없습니다: {path}"
                )
                print(
                    f"⚠️  경고: 디렉토리에서 지원 형식의 파일을 찾을 수 없습니다: {path}\n"
                    f"   지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            else:
                logger.info(f"로컬 수집 완료: {found_count}개 파일")
