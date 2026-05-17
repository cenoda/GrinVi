"""
데이터 마이그레이션 스크립트 (DataManager)

기존 data/ 루트의 임시 폴더들을 표준 구조로 이동합니다.

사용법:
    python scripts/migrate_data.py
"""

import shutil
from pathlib import Path


class DataManager:
    """데이터 폴더 구조를 관리하는 클래스."""

    BASE = Path("data")

    # 표준 경로 상수
    RAW_DIR       = BASE / "raw"
    GENERATED_DIR = BASE / "generated"
    PROCESSED_DIR = BASE / "processed"
    ARCHIVE_DIR   = BASE / "archive"

    # 마이그레이션 대상: (원본 경로, 목적지 경로)
    MIGRATION_MAP: list[tuple[Path, Path]] = [
        # 원본 데이터 → raw/
        (BASE / "ko_wikipedia",       RAW_DIR   / "ko_wikipedia"),
        # 임시 폴더들 → archive/
        (BASE / "test_degs",          ARCHIVE_DIR / "test_degs"),
        (BASE / "test_degs_2",        ARCHIVE_DIR / "test_degs_2"),
        (BASE / "test_degs_3",        ARCHIVE_DIR / "test_degs_3"),
        (BASE / "test_degs_4",        ARCHIVE_DIR / "test_degs_4"),
        (BASE / "test_degs_5",        ARCHIVE_DIR / "test_degs_5"),
        (BASE / "test_degs_6",        ARCHIVE_DIR / "test_degs_6"),
        (BASE / "test_qa",            ARCHIVE_DIR / "test_qa"),
        (BASE / "test_qa_final",      ARCHIVE_DIR / "test_qa_final"),
        (BASE / "test_qa_final_v2",   ARCHIVE_DIR / "test_qa_final_v2"),
        (BASE / "test_qa_final_v3",   ARCHIVE_DIR / "test_qa_final_v3"),
        (BASE / "test_qa_fixed",      ARCHIVE_DIR / "test_qa_fixed"),
        (BASE / "korean_training",    ARCHIVE_DIR / "korean_training"),
        (BASE / "qa_production_run",  ARCHIVE_DIR / "qa_production_run"),
    ]

    def setup_dirs(self) -> None:
        """표준 디렉토리 구조를 생성한다.

        생성 대상:
            data/raw/
            data/generated/
            data/processed/
            data/archive/
        """
        for directory in (self.RAW_DIR, self.GENERATED_DIR, self.PROCESSED_DIR, self.ARCHIVE_DIR):
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[setup] 디렉토리 준비: {directory}")

    def migrate(self) -> dict[str, list[str]]:
        """기존 임시 폴더를 archive/로, ko_wikipedia를 raw/로 이동한다.

        Returns:
            {"moved": [...], "skipped": [...]}
            - moved:   실제로 이동된 폴더 경로 목록
            - skipped: 원본이 없어 건너뛴 폴더 경로 목록

        Raises:
            FileExistsError: 목적지에 동일 이름 폴더가 이미 존재하는 경우
        """
        moved: list[str] = []
        skipped: list[str] = []

        for src, dst in self.MIGRATION_MAP:
            if not src.exists():
                # 원본 폴더가 없으면 건너뜀
                skipped.append(str(src))
                continue

            if dst.exists():
                # 목적지에 이미 동일 이름 폴더가 존재하면 오류 발생
                raise FileExistsError(
                    f"목적지 폴더가 이미 존재합니다: {dst}\n"
                    f"  원본: {src}\n"
                    f"  수동으로 처리한 후 다시 실행하세요.\n"
                    f"  예) mv {dst} {dst}_backup  또는  rm -rf {dst}"
                )

            # 부모 디렉토리가 없으면 생성
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(src), str(dst))
            moved.append(str(src))
            print(f"[migrate] 이동: {src} → {dst}")

        return {"moved": moved, "skipped": skipped}


if __name__ == "__main__":
    manager = DataManager()

    print("=== 표준 디렉토리 생성 ===")
    manager.setup_dirs()

    print("\n=== 데이터 마이그레이션 시작 ===")
    result = manager.migrate()

    print("\n=== 마이그레이션 결과 ===")
    if result["moved"]:
        print(f"이동 완료 ({len(result['moved'])}개):")
        for path in result["moved"]:
            print(f"  ✓ {path}")
    else:
        print("이동된 폴더 없음")

    if result["skipped"]:
        print(f"\n건너뜀 ({len(result['skipped'])}개, 원본 폴더 없음):")
        for path in result["skipped"]:
            print(f"  - {path}")
