# 설계 문서: 국어 기출문제 데이터 수집 파이프라인

## 개요

이 문서는 역대 국어 기출문제(수능, 모의고사, 공무원 시험, LEET, 임용고시)를 수집하여 GrinVi 학습 데이터로 변환하는 파이프라인의 기술 설계를 기술합니다.

파이프라인은 **Collector → Parser → Formatter → Storage** 4단계로 구성되며, 각 단계는 독립적으로 실행하거나 전체를 한 번에 실행할 수 있습니다.

---

## 시스템 아키텍처

```
scripts/collect_korean_exam.py  (CLI 진입점)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                     Pipeline                          │
│                                                       │
│  ┌─────────────┐   ┌──────────┐   ┌───────────────┐  │
│  │  Collector  │──▶│  Parser  │──▶│   Formatter   │  │
│  └─────────────┘   └──────────┘   └───────────────┘  │
│         │                                  │          │
│         ▼                                  ▼          │
│  data/raw/korean_exam/          data/raw/korean_exam/ │
│    huggingface/                   korean_exam_*.jsonl │
│    pdf/{type}/{year}/                                 │
│                                   (--merge 시)        │
│                                 data/processed/       │
│                                   train.txt           │
└───────────────────────────────────────────────────────┘
```

### Collector 계층 구조

```
BaseCollector (ABC)
├── HFCollector          # HuggingFace 데이터셋 수집
├── LocalCollector       # 로컬 파일 수집
└── PDFCollector         # 공식 사이트 PDF 크롤링
    ├── CSATCrawler      # suneung.re.kr (수능)
    ├── MockCrawler      # kice.re.kr (평가원 모의고사)
    ├── TeacherCrawler   # kice.re.kr (임용고시)
    ├── LEETCrawler      # moja.uwayapply.com (LEET)
    └── DistrictCrawler  # 각 시도 교육청
```

### Parser 계층 구조

```
BaseParser (ABC)
├── HFParser             # HuggingFace 포맷별 파서
│   ├── CSATQAParser     # HAERAE-HUB/csatqa 포맷
│   ├── CSATKKACHIParser # KKACHI-HUB/CSAT-KOREAN-2025 포맷
│   ├── SFTParser        # LLMin/final_csat_sft 포맷
│   └── CivilParser      # kikikara 공무원 시험 포맷
├── PDFParser            # PDF 텍스트 추출 및 구조화
│   ├── CSATPDFParser    # 수능/모의고사 PDF 파서
│   ├── TeacherPDFParser # 임용고시 PDF 파서
│   └── LEETPDFParser    # LEET 언어이해 PDF 파서
└── LocalParser          # 로컬 JSON/CSV/TXT 파서
```

---

## 디렉토리 구조

```
scripts/
└── collect_korean_exam.py      # CLI 진입점

grinvi/
└── korean_exam/
    ├── __init__.py
    ├── pipeline.py             # Pipeline 오케스트레이터
    ├── models.py               # ExamItem, ExamMetadata 데이터 모델
    ├── collectors/
    │   ├── __init__.py
    │   ├── base.py             # BaseCollector ABC
    │   ├── hf_collector.py     # HFCollector
    │   ├── local_collector.py  # LocalCollector
    │   └── pdf_collector.py    # PDFCollector + 크롤러들
    ├── parsers/
    │   ├── __init__.py
    │   ├── base.py             # BaseParser ABC
    │   ├── hf_parsers.py       # HF 포맷별 파서들
    │   ├── pdf_parser.py       # PDFParser + 시험별 파서들
    │   └── local_parser.py     # LocalParser
    ├── formatter.py            # Formatter
    ├── storage.py              # Storage
    └── validator.py            # 데이터 품질 검증
```

---

## 데이터 모델

### ExamMetadata

```python
@dataclass
class ExamMetadata:
    exam_type: str      # "csat", "mock_june", "mock_sept", "teacher",
                        # "leet", "district", "civil_local", "civil_national"
    year: int           # 시행 연도 (예: 2024)
    month: Optional[int]        # 시행 월 (수능=11, 6월모의=6, ...)
    question_number: Optional[int]  # 문제 번호
    subject: str        # "국어", "언어", "언어이해" 등
    source_dataset: Optional[str]   # HF 데이터셋 ID (HF 수집 시)
    source_file: Optional[str]      # 원본 파일 경로 (PDF 수집 시)
```

### ExamItem

```python
@dataclass
class ExamItem:
    passage: str                    # 지문 (없으면 빈 문자열)
    question: str                   # 문제 텍스트 (필수)
    choices: Optional[List[str]]    # 선지 목록 (없으면 None)
    answer: Optional[int]           # 정답 번호 1-based (없으면 None)
    explanation: Optional[str]      # 해설 (없으면 None)
    metadata: ExamMetadata

    def is_valid(self) -> bool:
        """question이 비어있지 않고 한국어 비율 50% 이상"""
        ...
```

### TrainingRecord (JSONL 출력 포맷)

```json
{
  "text": "지문: ...\n\n문제: ...\n\n① ...\n② ...\n③ ...\n④ ...\n⑤ ...",
  "source": "korean_exam",
  "exam_type": "csat",
  "year": 2022,
  "subject": "국어",
  "question_number": 15
}
```

---

## Collector 상세 설계

### BaseCollector

```python
from abc import ABC, abstractmethod
from typing import Iterator
from pathlib import Path

class BaseCollector(ABC):
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    @abstractmethod
    def collect(self) -> Iterator[Path]:
        """수집된 원시 데이터 파일 경로를 yield"""
        ...

    def _is_cached(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0
```

### HFCollector

HuggingFace 5개 데이터셋을 순서대로 수집합니다. 각 데이터셋은 수집 방법이 다릅니다.

```python
HF_DATASETS = [
    {
        "id": "csatqa",
        "hf_id": "HAERAE-HUB/csatqa",
        "method": "url",   # load_dataset 불가, URL 직접 다운로드
        "url": "https://huggingface.co/datasets/HAERAE-HUB/csatqa/resolve/main/data/csatqa.json",
        "url_eval": "https://huggingface.co/datasets/HAERAE-HUB/csatqa/resolve/main/data/csatqa_eval.json",
    },
    {
        "id": "csat_2025",
        "hf_id": "KKACHI-HUB/CSAT-KOREAN-2025",
        "method": "load_dataset",
        "split": "train",
    },
    {
        "id": "csat_sft",
        "hf_id": "LLMin/final_csat_sft",
        "method": "load_dataset",
        "split": "train",
    },
    {
        "id": "civil_local",
        "hf_id": "kikikara/Korean-Civil-Service-Examination-Train",
        "method": "load_dataset",
        "split": "train",
        "filter": {"task": "국어"},
    },
    {
        "id": "civil_national",
        "hf_id": "kikikara/Korean-Civil-Service-Examination-National",
        "method": "load_dataset",
        "split": "공무원_국가직",
        "filter": {"task": "국어"},
    },
]
```

수집된 데이터는 `data/raw/korean_exam/huggingface/{id}/raw.jsonl` 로 저장합니다.

### PDFCollector

공식 사이트별 크롤러를 조합하여 PDF를 다운로드합니다.

**CSATCrawler** (`suneung.re.kr`):
- 게시판 목록 페이지를 순회하며 `boardID=1500234` 게시글 중 `영역=국어` 행의 첨부파일 링크 추출
- 각 게시글 상세 페이지에서 PDF 다운로드 URL 파싱
- 저장 경로: `data/raw/korean_exam/pdf/csat/{year}/`

**MockCrawler** (`kice.re.kr`):
- `boardID=1500212` 게시판에서 국어 과목 PDF 목록 크롤링
- 저장 경로: `data/raw/korean_exam/pdf/mock/{year}/`

**TeacherCrawler** (`kice.re.kr`):
- `boardID=1500212` 게시판에서 `국어(1차)` 검색 결과 크롤링
- 저장 경로: `data/raw/korean_exam/pdf/teacher/{year}/`

**LEETCrawler** (`moja.uwayapply.com`):
- 사이트 구조가 iframe 기반이므로 `center.htm` → 메뉴 링크 → 기출 목록 순으로 탐색
- 언어이해 영역 PDF만 필터링
- 저장 경로: `data/raw/korean_exam/pdf/leet/{year}/`

**DistrictCrawler** (각 시도 교육청):
- 서울(sen.go.kr), 경기(goe.go.kr), 인천(ice.go.kr) 교육청 사이트 순회
- 3월·4월·7월·10월 모의고사 국어 PDF 수집
- 저장 경로: `data/raw/korean_exam/pdf/district/{region}/{year}/`

**공통 다운로드 로직**:
```python
def _download_pdf(self, url: str, dest: Path) -> bool:
    if self._is_cached(dest):
        return True  # 캐시 히트, 건너뜀
    try:
        resp = requests.get(url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except (requests.HTTPError, requests.Timeout) as e:
        logger.warning(f"다운로드 실패: {url} - {e}")
        return False
```

---

## Parser 상세 설계

### HF 포맷별 파서

각 HuggingFace 데이터셋은 필드 구조가 달라 전용 파서가 필요합니다.

**CSATQAParser** (`HAERAE-HUB/csatqa`):
```python
# 입력 필드: test_name, context, question, gold, option#1~5
# test_name 예: "2022년 2023 대학수학능력시험 국어  홀수형"
def _parse_metadata(self, test_name: str) -> ExamMetadata:
    # 정규식으로 연도, 시험 종류 추출
    # "2022년 2023 대학수학능력시험 국어" → year=2023, exam_type="csat"
    ...

def parse(self, raw: dict) -> ExamItem:
    return ExamItem(
        passage=raw.get("context", ""),
        question=raw["question"],
        choices=[raw[f"option#{i}"] for i in range(1, 6)],
        answer=raw.get("gold"),
        metadata=self._parse_metadata(raw["test_name"]),
    )
```

**CSATKKACHIParser** (`KKACHI-HUB/CSAT-KOREAN-2025`):
```python
# 입력 필드: idx, paragraph, question, question_plus, A~E, answer, point
def parse(self, raw: dict) -> ExamItem:
    question_text = raw["question"]
    if raw.get("question_plus"):
        question_text += "\n" + raw["question_plus"]
    return ExamItem(
        passage=raw.get("paragraph", ""),
        question=question_text,
        choices=[raw[k] for k in ["A", "B", "C", "D", "E"]],
        answer=raw.get("answer"),
        metadata=ExamMetadata(exam_type="csat", year=2025, subject="국어"),
    )
```

**SFTParser** (`LLMin/final_csat_sft`):
```python
# 입력 필드: messages (system/user/assistant)
# user 메시지에서 지문/문제 추출, assistant 메시지에서 해설 추출
def parse(self, raw: dict) -> ExamItem:
    user_msg = next(m["content"] for m in raw["messages"] if m["role"] == "user")
    assistant_msg = next(
        (m["content"] for m in raw["messages"] if m["role"] == "assistant"), None
    )
    passage, question, choices = self._split_user_message(user_msg)
    return ExamItem(
        passage=passage,
        question=question,
        choices=choices,
        explanation=assistant_msg,
        metadata=ExamMetadata(exam_type="csat_sft", year=0, subject="국어"),
    )
```

**CivilParser** (`kikikara` 공무원 시험):
```python
# 입력 필드: 문제번호, 문제내용, 선택지1~4, 정답, year, task, source
def parse(self, raw: dict) -> ExamItem:
    exam_type = "civil_national" if "국가직" in raw.get("source", "") else "civil_local"
    return ExamItem(
        passage="",
        question=raw["문제 내용"],
        choices=[raw[f"선택지 {i}"] for i in range(1, 5)],
        answer=int(raw["정답"]) if raw.get("정답") else None,
        metadata=ExamMetadata(
            exam_type=exam_type,
            year=int(raw["year"]),
            subject="국어",
            question_number=raw.get("문제 번호"),
        ),
    )
```

### PDFParser

`pdfplumber` 라이브러리를 사용하여 PDF에서 텍스트를 추출합니다.

```python
import pdfplumber

class PDFParser:
    # 제거할 비문제 텍스트 패턴
    NOISE_PATTERNS = [
        r'^\s*\d+\s*$',           # 페이지 번호
        r'대학수학능력시험.*홀수형',  # 시험지 헤더
        r'이 문제지에 관한.*',       # 저작권 표시
        r'^\s*-\s*\d+\s*-\s*$',   # 페이지 구분선
    ]

    def extract_text(self, pdf_path: Path) -> List[str]:
        """페이지별 텍스트 추출, 노이즈 제거"""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text is None:
                    logger.warning(f"{pdf_path.name} 페이지 {i+1}: 텍스트 추출 불가 (스캔 이미지)")
                    continue
                cleaned = self._remove_noise(text)
                pages.append(cleaned)
        return pages
```

**CSATPDFParser** (수능/모의고사):
- 문항 번호 패턴 `^\d{1,2}\.` 으로 문항 경계 탐지
- 지문은 문항 번호 앞 블록, 선지는 `①②③④⑤` 패턴으로 추출
- 정답은 별도 정답표 PDF에서 추출 (같은 디렉토리의 `*정답*` 파일)

**TeacherPDFParser** (임용고시):
- 문항 번호 패턴 `^\d{1,2}\.` 동일
- 선지는 `①②③④` 4지선다 패턴
- 서술형 문항은 `choices=None`으로 처리

**LEETPDFParser** (LEET 언어이해):
- 지문 세트 구조: `[지문 1]` ~ `[지문 N]` 블록 탐지
- 각 지문 블록 내 문항들을 묶어서 처리
- 문항 번호 `^\d{1,2}\.` 패턴

---

## Formatter 설계

`ExamItem` → `TrainingRecord` 변환을 담당합니다.

```python
class Formatter:
    MIN_TEXT_LENGTH = 50  # 이 미만이면 필터링

    def format(self, item: ExamItem) -> Optional[dict]:
        text = self._build_text(item)
        if len(text) < self.MIN_TEXT_LENGTH:
            return None  # 필터링
        return {
            "text": text,
            "source": "korean_exam",
            "exam_type": item.metadata.exam_type,
            "year": item.metadata.year,
            "subject": item.metadata.subject,
            "question_number": item.metadata.question_number,
        }

    def _build_text(self, item: ExamItem) -> str:
        parts = []
        if item.passage:
            parts.append(f"[지문]\n{item.passage}")
        parts.append(f"[문제]\n{item.question}")
        if item.choices:
            CIRCLE = "①②③④⑤"
            for i, choice in enumerate(item.choices):
                parts.append(f"{CIRCLE[i]} {choice}")
        if item.explanation:
            parts.append(f"[해설]\n{item.explanation}")
        return "\n\n".join(parts)
```

**plain_text 모드**: `text` 필드 값만 줄바꿈으로 구분하여 출력합니다.

---

## Storage 설계

```python
class Storage:
    def save(self, records: List[dict], out_dir: Path, merge: bool = False):
        date_str = datetime.now().strftime("%Y%m%d")
        jsonl_path = out_dir / "raw" / "korean_exam" / f"korean_exam_{date_str}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"저장 실패: {e}")
            jsonl_path.unlink(missing_ok=True)  # 부분 저장 파일 삭제
            raise

        if merge:
            train_path = out_dir / "processed" / "train.txt"
            with train_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(rec["text"] + "\n")
```

---

## Validator 설계

```python
import unicodedata

class Validator:
    def korean_ratio(self, text: str) -> float:
        """한국어 문자(가-힣) 비율 계산"""
        if not text:
            return 0.0
        korean = sum(1 for c in text if '가' <= c <= '힣')
        return korean / len(text)

    def validate(self, item: ExamItem) -> bool:
        if not item.question.strip():
            return False
        full_text = item.passage + item.question
        return self.korean_ratio(full_text) >= 0.5

    def deduplicate(self, items: List[ExamItem]) -> List[ExamItem]:
        """question 텍스트 기준 중복 제거"""
        seen = set()
        result = []
        for item in items:
            key = item.question.strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
```

---

## CLI 설계

```
scripts/collect_korean_exam.py

사용법:
  python scripts/collect_korean_exam.py --source huggingface [옵션]
  python scripts/collect_korean_exam.py --source pdf --exam csat [옵션]
  python scripts/collect_korean_exam.py --source pdf --exam all [옵션]
  python scripts/collect_korean_exam.py --source local --input <경로> [옵션]

필수 인수:
  --source {huggingface,pdf,local}   데이터 소스 선택

소스별 추가 인수:
  --exam {csat,mock,teacher,leet,district,all}
                                     PDF 소스 선택 (--source pdf 시 사용)
  --input PATH                       로컬 파일/디렉토리 경로 (--source local 시 필수)

공통 옵션:
  --out DIR          출력 디렉토리 (기본값: data)
  --merge            data/processed/train.txt에 병합
  --format {jsonl,plain_text}        출력 포맷 (기본값: jsonl)
  --max_items N      최대 처리 항목 수
  --validate         한국어 비율 검증 활성화
  --help             도움말 출력
```

**`--exam all`**: 5개 PDF 소스(csat, mock, teacher, leet, district)를 모두 순서대로 실행합니다.

---

## Pipeline 오케스트레이터

```python
class Pipeline:
    def run(self, args) -> PipelineResult:
        # 1. Collector 선택 및 실행
        collector = self._build_collector(args)
        raw_files = list(collector.collect())

        # 2. Parser 선택 및 실행
        parser = self._build_parser(args)
        items = []
        for f in raw_files:
            items.extend(parser.parse_file(f))

        # 3. 검증 및 중복 제거
        validator = Validator()
        if args.validate:
            items = [i for i in items if validator.validate(i)]
        items = validator.deduplicate(items)

        # 4. Formatter
        formatter = Formatter()
        records = [r for i in items if (r := formatter.format(i)) is not None]

        # 5. Storage
        storage = Storage()
        storage.save(records, Path(args.out), merge=args.merge)

        return PipelineResult(
            total=len(items),
            saved=len(records),
            filtered=len(items) - len(records),
        )
```

---

## 의존성

```toml
# pyproject.toml에 추가
[project.optional-dependencies]
korean_exam = [
    "datasets>=2.14.0",      # HuggingFace datasets
    "huggingface-hub>=0.20.0",
    "pdfplumber>=0.10.0",    # PDF 텍스트 추출
    "requests>=2.31.0",      # HTTP 크롤링
    "beautifulsoup4>=4.12.0", # HTML 파싱
]
```

`pdfplumber`는 `pdfminer.six` 기반으로 텍스트 레이어가 있는 PDF에서 정확한 추출이 가능합니다. 스캔 이미지 PDF는 현재 범위에서 제외합니다(경고 로그 후 건너뜀).

---

## 정확성 속성 (Property-Based Testing)

구현 시 다음 속성을 검증하는 PBT 테스트를 작성합니다.

1. **결정론적 파싱**: 동일한 입력을 두 번 파싱하면 동일한 `ExamItem`을 반환한다.
2. **라운드트립**: `Formatter.format(item)`의 `text` 필드에서 원본 `question`과 `passage`의 핵심 내용을 복원할 수 있다.
3. **필터링 단조성**: `min_length`를 높일수록 출력 레코드 수는 단조 감소한다.
4. **중복 제거 멱등성**: `deduplicate(deduplicate(items)) == deduplicate(items)`.
5. **한국어 비율 정확성**: `korean_ratio(text) >= 0.5`인 텍스트만 통과한다.
