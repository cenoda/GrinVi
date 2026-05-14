# 설계 문서: GrinVi 데이터 리팩토링

## 개요

GrinVi 데이터 리팩토링은 세 가지 독립적이지만 연관된 문제를 해결합니다.

1. **데이터 폴더 구조 혼란**: 현재 `data/` 루트에 `test_degs*`, `korean_training`, `qa_production_run` 등 임시 폴더가 난립하고 있으며, 원본 데이터와 생성 데이터가 구분되지 않습니다.
2. **체크포인트 무한 증가**: `checkpoints/step-*` 디렉토리가 171개 이상 쌓여 디스크를 낭비하고 있으며, 최고 성능 모델을 별도로 보관하는 메커니즘이 없습니다.
3. **Generator 출력 경로 문제**: `--output-dir` 기본값이 `data/korean_training`으로 하드코딩되어 있고, `.txt`와 `.jsonl`이 중복 저장되며, 병합 자동화가 없습니다.

이 리팩토링은 기존 학습 로직을 변경하지 않고 데이터 관리 레이어만 개선합니다.

---

## 아키텍처

### 변경 범위

```
grinvi-data-refactor
├── grinvi/
│   └── trainer.py          ← keep_last_n, best checkpoint 추가
├── scripts/
│   ├── train.py            ← --keep_last_n CLI 인수 추가
│   └── generate_training_data.py  ← 출력 경로, JSONL 전용, --merge 추가
└── scripts/
    └── migrate_data.py     ← 신규: 데이터 마이그레이션 스크립트 (DataManager)
```

### 데이터 폴더 구조 (목표 상태)

```
data/
├── raw/                    ← 원본 소스 데이터 (변경 없이 보관)
│   └── ko_wikipedia/
│       ├── train.txt
│       ├── val.txt
│       ├── ko_tokenizer.model
│       └── ko_tokenizer.vocab
├── generated/              ← Generator 출력
│   ├── text/               ← text 모드 결과
│   │   └── run_20260514/
│   │       ├── gemini_20260514_082056.jsonl
│   │       └── deepseek_20260514_082056.jsonl
│   └── qa/                 ← qa 모드 결과
│       └── run_20260514/
│           └── gemini_20260514_082056.jsonl
├── processed/              ← 학습용 병합 파일
│   ├── train.txt
│   └── val.txt
└── archive/                ← 기존 임시 폴더 보관
    ├── test_degs/
    ├── test_degs_2/
    ├── korean_training/
    └── qa_production_run/
```

### 체크포인트 구조 (목표 상태)

```
checkpoints/
├── step-170000/            ← 최근 N개만 보관
├── step-169000/
├── ...
└── best/                   ← 최저 eval loss 체크포인트 (자동 삭제 제외)
    ├── model.safetensors
    └── config.json
```

---

## 컴포넌트 및 인터페이스

### 1. DataManager (`scripts/migrate_data.py`)

마이그레이션을 일회성으로 실행하는 독립 스크립트입니다. 기존 데이터를 새 구조로 이동합니다.

```python
class DataManager:
    BASE = Path("data")

    # 표준 경로 상수
    RAW_DIR       = BASE / "raw"
    GENERATED_DIR = BASE / "generated"
    PROCESSED_DIR = BASE / "processed"
    ARCHIVE_DIR   = BASE / "archive"

    def setup_dirs(self) -> None:
        """표준 디렉토리 구조를 생성한다."""

    def migrate(self) -> dict[str, list[str]]:
        """기존 임시 폴더를 archive/로, ko_wikipedia를 raw/로 이동한다.
        반환값: {"moved": [...], "skipped": [...]}
        """
```

**마이그레이션 대상:**

| 원본 경로 | 목적지 경로 |
|---|---|
| `data/ko_wikipedia/` | `data/raw/ko_wikipedia/` |
| `data/test_degs/` | `data/archive/test_degs/` |
| `data/test_degs_2/` ~ `data/test_degs_6/` | `data/archive/test_degs_2/` ~ |
| `data/test_qa_final_v2/` | `data/archive/test_qa_final_v2/` |
| `data/test_qa_fixed/` | `data/archive/test_qa_fixed/` |
| `data/korean_training/` | `data/archive/korean_training/` |
| `data/qa_production_run/` | `data/archive/qa_production_run/` |

### 2. Generator 출력 경로 로직 (`scripts/generate_training_data.py`)

#### `resolve_output_dir(mode, output_dir_arg)` 함수

```python
def resolve_output_dir(mode: str, output_dir_arg: str | None) -> Path:
    """
    출력 디렉토리를 결정한다.
    - output_dir_arg가 있으면 해당 경로 사용
    - 없으면 data/generated/{mode}/run_{YYYYMMDD}/ 생성
      동일 날짜에 이미 존재하면 run_{YYYYMMDD}_2, _3, ... 으로 증가
    """
```

#### `merge_to_processed(run_dir, processed_dir)` 함수

```python
def merge_to_processed(run_dir: Path, processed_dir: Path) -> int:
    """
    run_dir의 모든 .jsonl 파일에서 text 필드를 추출하여
    processed_dir/train.txt에 append한다.
    반환값: 병합된 항목 수
    """
```

#### `run_worker` 함수 변경 사항

- `.txt` 파일 저장 코드 제거
- `.jsonl` 파일만 저장
- qa 모드 시 `question`, `answer` 필드 추가

### 3. TrainerConfig 변경 (`grinvi/trainer.py`)

```python
class TrainerConfig:
    def __init__(
        self,
        ...
        keep_last_n: int = 5,   # 신규: 보관할 최근 체크포인트 수 (0 = 무제한)
    ):
```

### 4. Trainer 체크포인트 관리 (`grinvi/trainer.py`)

#### `_save` 메서드 확장

```python
def _save(self, tag):
    # 1. 기존 체크포인트 저장 (변경 없음)
    out = self.cfg.checkpoint_dir / f"step-{tag}"
    self.model.save_pretrained(str(out))

    # 2. keep_last_n 정리 (신규)
    self._cleanup_checkpoints()

def _cleanup_checkpoints(self):
    """keep_last_n에 따라 오래된 체크포인트를 삭제한다."""
    if self.cfg.keep_last_n == 0:
        return
    # step-숫자 패턴만 수집 (best/ 제외)
    # 스텝 번호 기준 오름차순 정렬
    # 초과분 삭제
```

#### `_eval` 메서드 확장

```python
def _eval(self, step: int):
    ...
    avg = sum(losses) / len(losses)

    # Best checkpoint 저장 (신규)
    if avg < self.best_eval_loss:
        self.best_eval_loss = avg
        self._save_best()

def _save_best(self):
    """현재 모델을 checkpoints/best/에 저장한다."""
    best_dir = self.cfg.checkpoint_dir / "best"
    self.model.save_pretrained(str(best_dir))
```

#### `__init__` 변경

```python
self.best_eval_loss: float = float("inf")   # 신규
```

### 5. `scripts/train.py` CLI 변경

```python
p.add_argument("--keep_last_n", type=int, default=5,
               help="보관할 최근 체크포인트 수 (0 = 무제한)")

# TrainerConfig 생성 시
tcfg = TrainerConfig(
    ...
    keep_last_n=args.keep_last_n,   # 신규
)
```

---

## 데이터 모델

### JSONL 항목 스키마

#### text 모드

```json
{
  "text":      "생성된 한국어 텍스트",
  "prompt":    "사용된 프롬프트",
  "mode":      "text",
  "teacher":   "gemini | deepseek | lmstudio",
  "score":     0.85,
  "timestamp": "2026-05-14T08:20:56"
}
```

#### qa 모드 (추가 필드)

```json
{
  "text":      "### 질문: ...\n### 답변: ...",
  "prompt":    "사용된 프롬프트 (질문)",
  "mode":      "qa",
  "teacher":   "gemini",
  "score":     0.90,
  "timestamp": "2026-05-14T08:20:56",
  "question":  "질문 텍스트",
  "answer":    "답변 텍스트"
}
```

> **설계 결정**: `reason` 필드는 현재 코드에 존재하지만 요구사항에 명시되지 않았습니다. 하위 호환성을 위해 유지하되 필수 필드로 취급하지 않습니다.

### TrainerConfig 파라미터 추가

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `keep_last_n` | `int` | `5` | 보관할 최근 체크포인트 수. `0`이면 자동 삭제 비활성화 |

### Trainer 상태 추가

| 속성 | 타입 | 초기값 | 설명 |
|---|---|---|---|
| `best_eval_loss` | `float` | `float('inf')` | 학습 세션 내 최저 eval loss |

---

## 정확성 속성 (Correctness Properties)

*속성(Property)이란 시스템의 모든 유효한 실행에서 참이어야 하는 특성 또는 동작입니다. 즉, 시스템이 무엇을 해야 하는지에 대한 형식적 명세입니다. 속성은 사람이 읽을 수 있는 명세와 기계가 검증할 수 있는 정확성 보장 사이의 다리 역할을 합니다.*

### Property 1: 출력 경로 형식 준수

*임의의* 모드 문자열(`text` 또는 `qa`)과 날짜에 대해, `--output-dir`을 지정하지 않았을 때 생성되는 출력 경로는 반드시 `data/generated/{mode}/run_{YYYYMMDD}` 형식을 따라야 한다.

**Validates: Requirements 2.1**

### Property 2: 명시적 출력 경로 우선

*임의의* 경로 문자열을 `--output-dir`로 제공했을 때, 실제 사용되는 출력 경로는 반드시 해당 경로와 동일해야 한다 (기본값 경로가 사용되어서는 안 된다).

**Validates: Requirements 2.2**

### Property 3: 동일 날짜 실행 시 고유 디렉토리 보장

*임의의* N번(N ≥ 2) 연속 실행에 대해, 생성된 N개의 출력 디렉토리는 모두 서로 다른 고유한 경로를 가져야 한다.

**Validates: Requirements 2.3**

### Property 4: JSONL 전용 저장 (txt 미생성)

*임의의* 텍스트 데이터를 저장했을 때, 출력 디렉토리에는 `.jsonl` 파일만 생성되어야 하며 `.txt` 파일은 생성되어서는 안 된다.

**Validates: Requirements 3.1**

### Property 5: JSONL 필수 필드 포함

*임의의* 생성 결과(텍스트, 프롬프트, 모드, 교사 모델)에 대해, 저장된 JSONL 항목은 반드시 `text`, `prompt`, `mode`, `teacher`, `score`, `timestamp` 필드를 모두 포함해야 한다.

**Validates: Requirements 3.2**

### Property 6: qa 모드 추가 필드 포함

*임의의* qa 모드 생성 결과에 대해, 저장된 JSONL 항목은 반드시 `question`과 `answer` 필드를 추가로 포함해야 한다.

**Validates: Requirements 3.3**

### Property 7: 병합 시 모든 text 필드 보존

*임의의* JSONL 파일 집합을 `--merge`로 병합했을 때, 각 파일의 모든 `text` 필드 값이 `data/processed/train.txt`에 포함되어야 한다.

**Validates: Requirements 4.1, 4.2**

### Property 8: 병합 시 기존 내용 보존 (append)

*임의의* 기존 내용이 있는 `train.txt`에 새 데이터를 병합했을 때, 기존 내용이 그대로 보존되고 새 내용이 뒤에 추가되어야 한다 (기존 내용이 삭제되거나 덮어써져서는 안 된다).

**Validates: Requirements 4.3**

### Property 9: 체크포인트 정렬 정확성

*임의의* 순서로 생성된 `step-N` 디렉토리 목록에 대해, 정렬 결과는 반드시 스텝 번호 N의 오름차순이어야 한다.

**Validates: Requirements 5.2**

### Property 10: keep_last_n 개수 유지

*임의의* keep_last_n 값(≥ 1)과 체크포인트 수 M에 대해, 새 체크포인트 저장 후 `step-*` 패턴 디렉토리의 수는 반드시 `min(M+1, keep_last_n)`이어야 한다. 단, `best/` 디렉토리는 이 계산에서 제외된다.

**Validates: Requirements 5.3, 5.4**

### Property 11: best_eval_loss 단조 감소 유지

*임의의* eval loss 시퀀스에 대해, `best_eval_loss`는 항상 지금까지 관찰된 eval loss의 최솟값과 동일해야 한다.

**Validates: Requirements 6.1, 6.2, 6.5**

---

## 오류 처리

### DataManager (마이그레이션)

| 상황 | 처리 방식 |
|---|---|
| 이동 대상 폴더가 존재하지 않음 | 건너뛰고 `skipped` 목록에 추가 |
| 목적지에 동일 이름 폴더가 이미 존재 | 오류 발생 후 사용자에게 수동 처리 안내 |
| 권한 오류 | 예외를 그대로 전파 |

### Generator

| 상황 | 처리 방식 |
|---|---|
| 출력 디렉토리 생성 실패 | `OSError` 전파 |
| JSONL 쓰기 실패 | 기존 재시도 로직 유지 |
| `--merge` 시 `data/processed/` 없음 | 자동 생성 |
| JSONL 항목에 `text` 필드 없음 | 해당 항목 건너뛰고 경고 출력 |

### Trainer

| 상황 | 처리 방식 |
|---|---|
| `_cleanup_checkpoints` 중 삭제 실패 | 경고 출력 후 계속 진행 (학습 중단 방지) |
| `_save_best` 실패 | 경고 출력 후 계속 진행 |
| `eval_loader`가 `None`인 경우 | best checkpoint 저장 로직 전체 건너뜀 |
| `keep_last_n = 0` | 정리 로직 전체 건너뜀 |

---

## 테스트 전략

### 단위 테스트 (예시 기반)

**DataManager:**
- `setup_dirs()` 호출 후 4개 표준 디렉토리(`raw/`, `generated/`, `processed/`, `archive/`) 존재 확인
- 존재하지 않는 폴더 마이그레이션 시 오류 없이 건너뜀 확인
- 마이그레이션 반환값에 `moved`, `skipped` 키 포함 확인

**Generator:**
- `--merge` 없이 실행 시 `train.txt` 미생성 확인
- `data/processed/` 없을 때 `--merge` 실행 시 디렉토리 자동 생성 확인

**TrainerConfig:**
- 기본 생성 시 `keep_last_n = 5` 확인
- `Trainer` 초기화 시 `best_eval_loss = float('inf')` 확인

**train.py:**
- `--keep_last_n` 기본값 `5` 확인
- 특정 값 지정 시 `TrainerConfig`에 올바르게 전달 확인

### 속성 기반 테스트 (Property-Based Testing)

**라이브러리**: [Hypothesis](https://hypothesis.readthedocs.io/) (Python)

**설정**: 각 속성 테스트는 최소 100회 반복 실행

각 테스트는 다음 태그 형식으로 주석을 답니다:
`# Feature: grinvi-data-refactor, Property {번호}: {속성 텍스트}`

**Property 1 — 출력 경로 형식 준수:**
- 생성기: `mode ∈ {"text", "qa"}`, `date` (임의의 날짜)
- 검증: 생성된 경로가 `data/generated/{mode}/run_{YYYYMMDD}` 패턴과 일치

**Property 2 — 명시적 출력 경로 우선:**
- 생성기: 임의의 경로 문자열
- 검증: `resolve_output_dir(mode, path)` 반환값 == 입력 경로

**Property 3 — 동일 날짜 고유 디렉토리:**
- 생성기: `n ∈ [2, 10]` (실행 횟수)
- 검증: N번 호출 결과 집합의 크기 == N

**Property 4 — JSONL 전용 저장:**
- 생성기: 임의의 텍스트 문자열
- 검증: 저장 후 디렉토리에 `.txt` 파일 없음, `.jsonl` 파일 존재

**Property 5 — JSONL 필수 필드:**
- 생성기: 임의의 `(text, prompt, mode, teacher)` 조합
- 검증: 저장된 JSON 객체에 6개 필수 필드 모두 존재

**Property 6 — qa 모드 추가 필드:**
- 생성기: 임의의 `(question, answer)` 쌍
- 검증: 저장된 JSON 객체에 `question`, `answer` 필드 존재

**Property 7 — 병합 시 text 필드 보존:**
- 생성기: 임의의 JSONL 항목 목록 (각 항목에 `text` 필드 포함)
- 검증: 병합 후 `train.txt`의 줄 수 == 입력 항목 수, 각 줄이 해당 `text` 값과 일치

**Property 8 — 병합 append 동작:**
- 생성기: 임의의 기존 내용 문자열, 임의의 새 JSONL 항목 목록
- 검증: 병합 후 `train.txt`가 기존 내용으로 시작하고 새 내용이 뒤에 추가됨

**Property 9 — 체크포인트 정렬:**
- 생성기: 임의의 순서로 섞인 `step-N` 디렉토리 이름 목록
- 검증: 정렬 결과가 N의 오름차순과 일치

**Property 10 — keep_last_n 개수 유지:**
- 생성기: `keep_last_n ∈ [1, 20]`, `existing_count ∈ [0, 30]`
- 검증: 정리 후 `step-*` 디렉토리 수 == `min(existing_count + 1, keep_last_n)`, `best/` 디렉토리는 영향 없음

**Property 11 — best_eval_loss 단조 감소:**
- 생성기: 임의의 float 시퀀스 (eval loss 값들)
- 검증: 각 값을 순서대로 처리한 후 `best_eval_loss` == `min(시퀀스)`
