"""BaseCollector 추상 클래스."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class BaseCollector(ABC):
    """데이터 수집기의 기본 추상 클래스.

    모든 Collector는 이 클래스를 상속하여 collect() 메서드를 구현해야 합니다.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    @abstractmethod
    def collect(self) -> Iterator[Path]:
        """수집된 원시 데이터 파일 경로를 yield.

        Returns:
            Iterator[Path]: 수집된 파일 경로들
        """
        ...

    def _is_cached(self, path: Path) -> bool:
        """파일이 이미 캐시되어 있는지 확인.

        Args:
            path: 확인할 파일 경로

        Returns:
            bool: 파일이 존재하고 크기가 0보다 크면 True
        """
        return path.exists() and path.stat().st_size > 0
