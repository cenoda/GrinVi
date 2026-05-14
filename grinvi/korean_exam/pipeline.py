"""Pipeline 오케스트레이터: Collector → Parser → Validator → Formatter → Storage."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from grinvi.korean_exam.formatter import Formatter
from grinvi.korean_exam.models import ExamItem
from grinvi.korean_exam.storage import Storage
from grinvi.korean_exam.validator import Validator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과."""

    total: int  # 총 수집 항목 수 (중복 제거 후)
    saved: int  # 저장된 레코드 수
    filtered: int  # 필터링된 항목 수
    exam_type_dist: dict = field(default_factory=dict)  # 시험 종류별 분포
    year_dist: dict = field(default_factory=dict)  # 연도별 분포


class Pipeline:
    """Collector → Parser → Validator → Formatter → Storage 순서로 실행하는 오케스트레이터."""

    def run(self, args) -> PipelineResult:
        """파이프라인 전체를 실행한다.

        Args:
            args: CLI 인수 (argparse.Namespace 또는 동일 인터페이스 객체).
                필수 속성: source, out, merge, validate
                선택 속성: max_items, exam, input, format

        Returns:
            PipelineResult: 처리 통계.
        """
        # 1. Collector 선택 및 실행
        collector = self._build_collector(args)
        raw_files = list(collector.collect())
        logger.info("수집 완료: %d개 파일", len(raw_files))

        # 2. Parser 선택 및 실행
        parser = self._build_parser(args)
        items: List[ExamItem] = []
        for f in raw_files:
            items.extend(parser.parse_file(f))

        total_parsed = len(items)
        logger.info("파싱 완료: %d건", total_parsed)

        # 3. 검증 (--validate 옵션 시)
        validator = Validator()
        if getattr(args, "validate", False):
            items = [i for i in items if validator.validate(i)]
            validated_count = len(items)
            logger.info(
                "검증 완료: %d건 유효 / %d건 필터링",
                validated_count,
                total_parsed - validated_count,
            )

        # 4. 중복 제거 (항상 적용)
        before_dedup = len(items)
        items = validator.deduplicate(items)
        dedup_removed = before_dedup - len(items)
        if dedup_removed > 0:
            logger.info("중복 제거: %d건 제거", dedup_removed)

        # 5. --max_items 옵션 처리
        max_items = getattr(args, "max_items", None)
        if max_items is not None and max_items > 0:
            items = items[:max_items]
            logger.info("max_items 적용: %d건으로 제한", len(items))

        # 6. Formatter
        formatter = Formatter()
        records = [r for i in items if (r := formatter.format(i)) is not None]

        # 7. Storage
        storage = Storage()
        out_dir = Path(getattr(args, "out", "data"))
        merge = getattr(args, "merge", False)
        repeat = getattr(args, "repeat", 1) or 1
        storage.save(records, out_dir, merge=merge, repeat=repeat)

        # 8. 통계 계산
        exam_type_dist = dict(Counter(i.metadata.exam_type for i in items))
        year_dist = dict(
            sorted(Counter(i.metadata.year for i in items).items())
        )

        result = PipelineResult(
            total=len(items),
            saved=len(records),
            filtered=len(items) - len(records),
            exam_type_dist=exam_type_dist,
            year_dist=year_dist,
        )

        # 9. 통계 출력
        self._print_stats(result, total_parsed)

        # 10. 유효 항목 50% 미만 시 경고
        if total_parsed > 0 and result.total < total_parsed * 0.5:
            logger.warning(
                "⚠️  유효 항목이 전체의 50%% 미만입니다 (%d/%d). "
                "입력 데이터 품질을 확인하세요.",
                result.total,
                total_parsed,
            )

        return result

    def _build_collector(self, args):
        """소스 유형에 따라 적절한 Collector를 생성한다."""
        source = args.source
        out_dir = Path(getattr(args, "out", "data")) / "raw" / "korean_exam"

        if source == "huggingface":
            from grinvi.korean_exam.collectors.hf_collector import HFCollector

            return HFCollector(out_dir=out_dir)
        elif source == "pdf":
            from grinvi.korean_exam.collectors.pdf_collector import PDFCollector

            exam = getattr(args, "exam", "all")
            return PDFCollector(out_dir=out_dir, exam=exam)
        elif source == "local":
            from grinvi.korean_exam.collectors.local_collector import (
                LocalCollector,
            )

            input_path = getattr(args, "input", None)
            return LocalCollector(out_dir=out_dir, input_path=input_path)
        else:
            raise ValueError(f"지원하지 않는 소스: {source}")

    def _build_parser(self, args):
        """소스 유형에 따라 적절한 Parser를 생성한다."""
        source = args.source

        if source == "huggingface":
            return _HFCompositeParser()
        elif source == "pdf":
            from grinvi.korean_exam.parsers.pdf_parser import PDFParser

            return PDFParser()
        elif source == "local":
            from grinvi.korean_exam.parsers.local_parser import LocalParser

            return LocalParser()
        else:
            raise ValueError(f"지원하지 않는 소스: {source}")

    def _print_stats(self, result: PipelineResult, total_parsed: int) -> None:
        """처리 통계를 출력한다."""
        print("\n" + "=" * 50)
        print("📊 처리 통계")
        print("=" * 50)
        print(f"  총 파싱 수: {total_parsed}건")
        print(f"  유효 항목 수: {result.total}건")
        print(f"  저장된 레코드 수: {result.saved}건")
        print(f"  필터링된 항목 수: {result.filtered}건")

        if result.exam_type_dist:
            print("\n  📋 시험 종류별 분포:")
            for exam_type, count in sorted(
                result.exam_type_dist.items(), key=lambda x: -x[1]
            ):
                print(f"    {exam_type}: {count}건")

        if result.year_dist:
            print("\n  📅 연도별 분포:")
            for year, count in sorted(result.year_dist.items()):
                if year > 0:
                    print(f"    {year}년: {count}건")
                else:
                    print(f"    (연도 미상): {count}건")

        print("=" * 50)


class _HFCompositeParser:
    """HuggingFace 데이터셋별로 적절한 파서를 디스패치하는 복합 파서."""

    # 데이터셋 ID → 파서 매핑
    _PARSER_MAP = {
        "csatqa": "CSATQAParser",
        "csat_2025": "CSATKKACHIParser",
        "csat_sft": "SFTParser",
        "civil_local": "CivilParser",
        "civil_national": "CivilParser",
    }

    def parse_file(self, path: Path) -> List[ExamItem]:
        """파일 경로에서 데이터셋 ID를 추출하고 적절한 파서로 디스패치한다.

        경로 형식: .../huggingface/{dataset_id}/raw.jsonl
        """
        from grinvi.korean_exam.parsers.hf_parsers import (
            CivilParser,
            CSATKKACHIParser,
            CSATQAParser,
            SFTParser,
        )

        parser_classes = {
            "CSATQAParser": CSATQAParser,
            "CSATKKACHIParser": CSATKKACHIParser,
            "SFTParser": SFTParser,
            "CivilParser": CivilParser,
        }

        # 경로에서 데이터셋 ID 추출
        # 예: data/raw/korean_exam/huggingface/csatqa/raw.jsonl
        parts = path.parts
        dataset_id = None
        for i, part in enumerate(parts):
            if part == "huggingface" and i + 1 < len(parts):
                dataset_id = parts[i + 1]
                break

        if dataset_id is None:
            # 폴백: 파일명이나 부모 디렉토리에서 추론
            dataset_id = path.parent.name

        parser_name = self._PARSER_MAP.get(dataset_id)
        if parser_name is None:
            logger.warning(
                "알 수 없는 데이터셋 ID: %s, CSATQAParser로 폴백", dataset_id
            )
            parser_name = "CSATQAParser"

        parser = parser_classes[parser_name]()
        return parser.parse_file(path)
