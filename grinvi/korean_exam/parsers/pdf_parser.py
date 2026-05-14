"""PDF 파서 구현: pdfplumber 기반 텍스트 추출 및 구조화."""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from ..models import ExamItem, ExamMetadata
from .base import BaseParser

logger = logging.getLogger(__name__)

# 원형 숫자 문자
CIRCLE_NUMBERS = "①②③④⑤"
CIRCLE_NUMBERS_4 = "①②③④"


class PDFParser(BaseParser):
    """PDF 텍스트 추출 및 기출문제 구조화 기본 클래스.

    pdfplumber를 사용하여 PDF에서 텍스트를 추출하고,
    노이즈 패턴을 제거한 후 문항을 파싱합니다.
    """

    # 제거할 비문제 텍스트 패턴
    NOISE_PATTERNS = [
        r"^\s*\d+\s*$",  # 페이지 번호
        r"대학수학능력시험.*홀수형",  # 시험지 헤더
        r"대학수학능력시험.*짝수형",  # 시험지 헤더
        r"이 문제지에 관한.*",  # 저작권 표시
        r"^\s*-\s*\d+\s*-\s*$",  # 페이지 구분선
        r"^\s*제\s*\d+\s*교시\s*$",  # 교시 표시
        r"성명\s*수험\s*번호",  # 수험 정보 헤더
    ]

    def __init__(self):
        self._compiled_noise = [re.compile(p) for p in self.NOISE_PATTERNS]

    def parse(self, raw: dict) -> ExamItem:
        """raw dict를 ExamItem으로 변환 (PDF 파서에서는 사용하지 않음).

        PDF 파서는 parse_file()을 직접 오버라이드합니다.
        """
        # PDF 파서는 parse_file을 직접 구현하므로 이 메서드는 폴백용
        raise NotImplementedError("PDF 파서는 parse_file()을 사용합니다.")

    def parse_file(self, path: Path) -> List[ExamItem]:
        """PDF 파일을 파싱하여 ExamItem 리스트를 반환.

        Args:
            path: PDF 파일 경로

        Returns:
            List[ExamItem]: 추출된 기출문제 리스트
        """
        start_time = time.time()
        failed_pages = 0

        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber가 설치되지 않았습니다. pip install pdfplumber")
            return []

        pages_text = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text is None:
                        logger.warning(
                            f"{path.name} 페이지 {i + 1}: 텍스트 추출 불가 (스캔 이미지)"
                        )
                        failed_pages += 1
                        continue
                    cleaned = self._remove_noise(text)
                    if cleaned.strip():
                        pages_text.append(cleaned)
        except Exception as e:
            logger.error(f"PDF 열기 실패: {path} - {e}")
            return []

        # 시험 종류 판별 및 적절한 파서 선택
        sub_parser = self._select_sub_parser(path, pages_text)
        items = sub_parser._parse_pages(pages_text, path)

        elapsed = time.time() - start_time
        logger.info(
            f"{path.name}: {len(items)}문항 추출, "
            f"실패 페이지 {failed_pages}개, "
            f"소요 시간 {elapsed:.1f}초"
        )

        return items

    def _remove_noise(self, text: str) -> str:
        """노이즈 패턴을 제거한 텍스트를 반환."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            is_noise = False
            for pattern in self._compiled_noise:
                if pattern.search(line):
                    is_noise = True
                    break
            if not is_noise:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _select_sub_parser(self, path: Path, pages_text: List[str]) -> "PDFParser":
        """파일 경로와 내용을 기반으로 적절한 서브 파서를 선택."""
        path_str = str(path).lower()

        if "teacher" in path_str or "임용" in path_str:
            return TeacherPDFParser()
        elif "leet" in path_str or "언어이해" in path_str:
            return LEETPDFParser()
        else:
            # 기본: 수능/모의고사 파서
            return CSATPDFParser()

    def _parse_pages(self, pages_text: List[str], path: Path) -> List[ExamItem]:
        """페이지 텍스트에서 문항을 추출 (서브클래스에서 오버라이드)."""
        # 기본 구현: CSATPDFParser로 위임
        parser = CSATPDFParser()
        return parser._parse_pages(pages_text, path)

    def _extract_metadata_from_path(self, path: Path) -> ExamMetadata:
        """파일 경로에서 메타데이터를 추출."""
        parts = path.parts
        exam_type = "csat"
        year = 0

        for part in parts:
            if part in ("csat", "mock", "teacher", "leet", "district"):
                exam_type = part
            # 연도 추출
            if re.match(r"^\d{4}$", part):
                year = int(part)

        return ExamMetadata(
            exam_type=exam_type,
            year=year,
            subject="국어",
            source_file=str(path),
        )


class CSATPDFParser(PDFParser):
    """수능/모의고사 PDF 파서.

    문항 번호 패턴으로 문항 경계를 탐지하고,
    5지선다 선지를 추출합니다.
    """

    # 문항 번호 패턴: "1." ~ "45."
    QUESTION_PATTERN = re.compile(r"^(\d{1,2})\.\s*(.+)", re.MULTILINE)

    # 선지 패턴: ①②③④⑤
    CHOICES_PATTERN = re.compile(r"[①②③④⑤]\s*(.+?)(?=[①②③④⑤]|$)")

    def __init__(self):
        super().__init__()

    def _parse_pages(self, pages_text: List[str], path: Path) -> List[ExamItem]:
        """수능/모의고사 PDF 페이지에서 문항을 추출."""
        full_text = "\n".join(pages_text)
        metadata_base = self._extract_metadata_from_path(path)
        items = []

        # 문항 번호로 텍스트 분할
        question_blocks = self._split_by_questions(full_text)

        for q_num, block in question_blocks:
            item = self._parse_question_block(block, q_num, metadata_base)
            if item:
                items.append(item)

        return items

    def _split_by_questions(self, text: str) -> List[tuple]:
        """문항 번호 패턴으로 텍스트를 분할."""
        matches = list(self.QUESTION_PATTERN.finditer(text))
        blocks = []

        for i, match in enumerate(matches):
            q_num = int(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block_text = text[start:end]
            blocks.append((q_num, block_text))

        return blocks

    def _parse_question_block(
        self, block: str, q_num: int, metadata_base: ExamMetadata
    ) -> Optional[ExamItem]:
        """단일 문항 블록을 파싱."""
        # 문항 번호 제거
        block = re.sub(r"^\d{1,2}\.\s*", "", block, count=1)

        # 선지 추출
        choices = self._extract_choices(block, num_choices=5)

        # 선지 이전 텍스트를 문제+지문으로 분리
        if choices:
            # 첫 번째 선지 위치 찾기
            first_choice_pos = block.find("①")
            if first_choice_pos > 0:
                question_text = block[:first_choice_pos].strip()
            else:
                question_text = block.strip()
        else:
            question_text = block.strip()

        if not question_text:
            return None

        # 지문과 문제 분리 (간단한 휴리스틱)
        passage, question = self._split_passage_question(question_text)

        metadata = ExamMetadata(
            exam_type=metadata_base.exam_type,
            year=metadata_base.year,
            month=metadata_base.month,
            question_number=q_num,
            subject="국어",
            source_file=metadata_base.source_file,
        )

        return ExamItem(
            passage=passage,
            question=question,
            choices=choices,
            answer=None,  # 정답은 별도 정답표에서 추출
            explanation=None,
            metadata=metadata,
        )

    def _extract_choices(self, text: str, num_choices: int = 5) -> Optional[List[str]]:
        """텍스트에서 선지를 추출."""
        circles = CIRCLE_NUMBERS[:num_choices]
        choices = []

        for i, circle in enumerate(circles):
            if circle not in text:
                return None  # 선지가 불완전하면 None

            start = text.index(circle) + 1
            # 다음 선지 또는 텍스트 끝까지
            if i + 1 < num_choices and circles[i + 1] in text[start:]:
                end = text.index(circles[i + 1], start)
            else:
                end = len(text)

            choice_text = text[start:end].strip()
            # 줄바꿈 정리
            choice_text = re.sub(r"\s+", " ", choice_text)
            choices.append(choice_text)

        return choices if len(choices) == num_choices else None

    def _split_passage_question(self, text: str) -> tuple:
        """텍스트를 지문과 문제로 분리.

        마지막 문장(물음표로 끝나는)을 문제로, 나머지를 지문으로 처리.
        """
        lines = text.strip().split("\n")

        # 물음표로 끝나는 마지막 줄을 문제로
        question_idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            if "?" in lines[i] or "는?" in lines[i] or "시오." in lines[i]:
                question_idx = i
                break

        if question_idx == 0:
            return "", text.strip()

        passage = "\n".join(lines[:question_idx]).strip()
        question = "\n".join(lines[question_idx:]).strip()

        return passage, question


class TeacherPDFParser(PDFParser):
    """임용고시 PDF 파서.

    4지선다 패턴과 서술형 문항을 처리합니다.
    """

    QUESTION_PATTERN = re.compile(r"^(\d{1,2})\.\s*(.+)", re.MULTILINE)

    def __init__(self):
        super().__init__()

    def _parse_pages(self, pages_text: List[str], path: Path) -> List[ExamItem]:
        """임용고시 PDF 페이지에서 문항을 추출."""
        full_text = "\n".join(pages_text)
        metadata_base = self._extract_metadata_from_path(path)
        items = []

        # 문항 번호로 텍스트 분할
        matches = list(self.QUESTION_PATTERN.finditer(full_text))

        for i, match in enumerate(matches):
            q_num = int(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            block = full_text[start:end]

            item = self._parse_question_block(block, q_num, metadata_base)
            if item:
                items.append(item)

        return items

    def _parse_question_block(
        self, block: str, q_num: int, metadata_base: ExamMetadata
    ) -> Optional[ExamItem]:
        """단일 문항 블록을 파싱 (4지선다 또는 서술형)."""
        # 문항 번호 제거
        block = re.sub(r"^\d{1,2}\.\s*", "", block, count=1)

        # 4지선다 선지 추출 시도
        choices = self._extract_choices_4(block)

        # 선지가 없으면 서술형
        if choices:
            first_choice_pos = block.find("①")
            if first_choice_pos > 0:
                question_text = block[:first_choice_pos].strip()
            else:
                question_text = block.strip()
        else:
            question_text = block.strip()

        if not question_text:
            return None

        passage, question = self._split_passage_question(question_text)

        metadata = ExamMetadata(
            exam_type="teacher",
            year=metadata_base.year,
            question_number=q_num,
            subject="국어",
            source_file=metadata_base.source_file,
        )

        return ExamItem(
            passage=passage,
            question=question,
            choices=choices,  # 서술형이면 None
            answer=None,
            explanation=None,
            metadata=metadata,
        )

    def _extract_choices_4(self, text: str) -> Optional[List[str]]:
        """4지선다 선지를 추출."""
        circles = CIRCLE_NUMBERS_4
        choices = []

        for i, circle in enumerate(circles):
            if circle not in text:
                return None

            start = text.index(circle) + 1
            if i + 1 < len(circles) and circles[i + 1] in text[start:]:
                end = text.index(circles[i + 1], start)
            else:
                # ⑤가 있으면 그 전까지, 없으면 줄 끝까지
                if "⑤" in text[start:]:
                    end = text.index("⑤", start)
                else:
                    # 다음 줄바꿈 또는 텍스트 끝
                    remaining = text[start:]
                    newline_pos = remaining.find("\n\n")
                    end = start + newline_pos if newline_pos > 0 else len(text)

            choice_text = text[start:end].strip()
            choice_text = re.sub(r"\s+", " ", choice_text)
            choices.append(choice_text)

        return choices if len(choices) == 4 else None

    def _split_passage_question(self, text: str) -> tuple:
        """텍스트를 지문과 문제로 분리."""
        lines = text.strip().split("\n")

        question_idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            if "?" in lines[i] or "시오." in lines[i] or "것은" in lines[i]:
                question_idx = i
                break

        if question_idx == 0:
            return "", text.strip()

        passage = "\n".join(lines[:question_idx]).strip()
        question = "\n".join(lines[question_idx:]).strip()

        return passage, question


class LEETPDFParser(PDFParser):
    """LEET 언어이해 PDF 파서.

    지문 세트 구조([지문 1]~[지문 N])를 파싱하고
    각 지문 블록 내 문항들을 그룹핑합니다.
    """

    # 지문 블록 패턴
    PASSAGE_BLOCK_PATTERN = re.compile(r"\[지문\s*(\d+)\]")
    QUESTION_PATTERN = re.compile(r"^(\d{1,2})\.\s*(.+)", re.MULTILINE)

    def __init__(self):
        super().__init__()

    def _parse_pages(self, pages_text: List[str], path: Path) -> List[ExamItem]:
        """LEET PDF 페이지에서 문항을 추출."""
        full_text = "\n".join(pages_text)
        metadata_base = self._extract_metadata_from_path(path)
        items = []

        # 지문 블록으로 분할
        passage_blocks = self._split_by_passages(full_text)

        if passage_blocks:
            # 지문 세트 구조가 있는 경우
            for passage_text, questions_text in passage_blocks:
                block_items = self._parse_passage_block(
                    passage_text, questions_text, metadata_base
                )
                items.extend(block_items)
        else:
            # 지문 세트 구조가 없는 경우: 일반 문항 파싱
            question_blocks = self._split_by_questions(full_text)
            for q_num, block in question_blocks:
                item = self._parse_single_question(block, q_num, metadata_base)
                if item:
                    items.append(item)

        return items

    def _split_by_passages(self, text: str) -> List[tuple]:
        """[지문 N] 패턴으로 텍스트를 분할."""
        matches = list(self.PASSAGE_BLOCK_PATTERN.finditer(text))
        if not matches:
            return []

        blocks = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block_text = text[start:end]

            # 지문과 문항 분리
            question_matches = list(self.QUESTION_PATTERN.finditer(block_text))
            if question_matches:
                passage_text = block_text[: question_matches[0].start()].strip()
                questions_text = block_text[question_matches[0].start() :]
            else:
                passage_text = block_text.strip()
                questions_text = ""

            blocks.append((passage_text, questions_text))

        return blocks

    def _parse_passage_block(
        self, passage_text: str, questions_text: str, metadata_base: ExamMetadata
    ) -> List[ExamItem]:
        """지문 블록 내 문항들을 파싱."""
        items = []
        question_blocks = self._split_by_questions(questions_text)

        for q_num, block in question_blocks:
            item = self._parse_single_question(
                block, q_num, metadata_base, passage=passage_text
            )
            if item:
                items.append(item)

        return items

    def _split_by_questions(self, text: str) -> List[tuple]:
        """문항 번호 패턴으로 텍스트를 분할."""
        matches = list(self.QUESTION_PATTERN.finditer(text))
        blocks = []

        for i, match in enumerate(matches):
            q_num = int(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append((q_num, text[start:end]))

        return blocks

    def _parse_single_question(
        self,
        block: str,
        q_num: int,
        metadata_base: ExamMetadata,
        passage: str = "",
    ) -> Optional[ExamItem]:
        """단일 문항을 파싱."""
        # 문항 번호 제거
        block = re.sub(r"^\d{1,2}\.\s*", "", block, count=1)

        # 5지선다 선지 추출
        choices = self._extract_choices_5(block)

        if choices:
            first_choice_pos = block.find("①")
            if first_choice_pos > 0:
                question_text = block[:first_choice_pos].strip()
            else:
                question_text = block.strip()
        else:
            question_text = block.strip()

        if not question_text:
            return None

        metadata = ExamMetadata(
            exam_type="leet",
            year=metadata_base.year,
            question_number=q_num,
            subject="언어이해",
            source_file=metadata_base.source_file,
        )

        return ExamItem(
            passage=passage,
            question=question_text,
            choices=choices,
            answer=None,
            explanation=None,
            metadata=metadata,
        )

    def _extract_choices_5(self, text: str) -> Optional[List[str]]:
        """5지선다 선지를 추출."""
        circles = CIRCLE_NUMBERS
        choices = []

        for i, circle in enumerate(circles):
            if circle not in text:
                return None

            start = text.index(circle) + 1
            if i + 1 < len(circles) and circles[i + 1] in text[start:]:
                end = text.index(circles[i + 1], start)
            else:
                end = len(text)

            choice_text = text[start:end].strip()
            choice_text = re.sub(r"\s+", " ", choice_text)
            choices.append(choice_text)

        return choices if len(choices) == 5 else None
