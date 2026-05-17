#!/usr/bin/env python3
"""국어 기출문제 데이터 수집 CLI.

사용법:
  python scripts/collect_korean_exam.py --source huggingface [옵션]
  python scripts/collect_korean_exam.py --source pdf --exam csat [옵션]
  python scripts/collect_korean_exam.py --source pdf --exam all [옵션]
  python scripts/collect_korean_exam.py --source local --input <경로> [옵션]

예시:
  # HuggingFace 데이터셋 전체 수집
  python scripts/collect_korean_exam.py --source huggingface --out data --validate

  # 수능 PDF 크롤링
  python scripts/collect_korean_exam.py --source pdf --exam csat --out data

  # 로컬 파일 처리
  python scripts/collect_korean_exam.py --source local --input ./my_data --out data

  # 최대 100건만 처리하고 train.txt에 병합
  python scripts/collect_korean_exam.py --source huggingface --max_items 100 --merge
"""

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    """CLI 인수 파서를 생성한다."""
    parser = argparse.ArgumentParser(
        prog="collect_korean_exam",
        description="국어 기출문제 데이터 수집 파이프라인",
        epilog=(
            "사용 예시:\n"
            "  python scripts/collect_korean_exam.py --source huggingface --out data --validate\n"
            "  python scripts/collect_korean_exam.py --source pdf --exam csat --out data\n"
            "  python scripts/collect_korean_exam.py --source pdf --exam all --out data\n"
            "  python scripts/collect_korean_exam.py --source local --input ./my_data --out data\n"
            "  python scripts/collect_korean_exam.py --source huggingface --max_items 100 --merge\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수 인수
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "pdf", "local"],
        required=True,
        help="데이터 소스 선택 (필수)",
    )

    # 소스별 추가 인수
    parser.add_argument(
        "--exam",
        type=str,
        choices=["csat", "mock", "teacher", "leet", "district", "all"],
        default=None,
        help="PDF 소스 시험 종류 선택 (--source pdf 시 사용)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="로컬 파일/디렉토리 경로 (--source local 시 필수)",
    )

    # 공통 옵션
    parser.add_argument(
        "--out",
        type=str,
        default="data",
        help="출력 디렉토리 (기본값: data)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="data/processed/train.txt에 병합",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "plain_text"],
        default="jsonl",
        help="출력 포맷 (기본값: jsonl)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="최대 처리 항목 수",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="한국어 비율 검증 활성화",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="--merge 시 반복 횟수 (가중치, 기본값: 1)",
    )

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """인수 유효성 검사. 실패 시 에러 메시지를 출력하고 False를 반환한다."""
    if args.source == "local" and not args.input:
        print(
            "오류: --source local 사용 시 --input 인수가 필수입니다.",
            file=sys.stderr,
        )
        print(
            "사용법: python scripts/collect_korean_exam.py --source local --input <경로>",
            file=sys.stderr,
        )
        return False

    if args.source == "pdf" and not args.exam:
        print(
            "오류: --source pdf 사용 시 --exam 인수가 필수입니다.",
            file=sys.stderr,
        )
        print(
            "사용법: python scripts/collect_korean_exam.py --source pdf --exam {csat,mock,teacher,leet,district,all}",
            file=sys.stderr,
        )
        return False

    return True


def print_next_steps() -> None:
    """정상 완료 후 다음 단계 명령어 예시를 출력한다."""
    print("\n✅ 수집 완료!")
    print("\n다음 단계:")
    print("  1. 토크나이저 훈련:")
    print("     python scripts/train_tokenizer.py --input data/processed/train.txt")
    print("  2. 모델 훈련:")
    print("     python scripts/train.py --data data/processed/train.txt")
    print("  3. 데이터 확인:")
    print("     head -5 data/raw/korean_exam/korean_exam_*.jsonl")


def main() -> int:
    """CLI 메인 함수.

    Returns:
        int: 종료 코드 (0=성공, 1=실패)
    """
    parser = build_parser()
    args = parser.parse_args()

    # 인수 유효성 검사
    if not validate_args(args):
        return 1

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from grinvi.korean_exam.pipeline import Pipeline

        pipeline = Pipeline()
        result = pipeline.run(args)

        # 정상 완료 시 다음 단계 출력
        print_next_steps()
        return 0

    except KeyboardInterrupt:
        print("\n중단됨.", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).error("파이프라인 실행 실패: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
