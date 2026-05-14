# 요구사항 문서

## 소개

GrinVi는 한국어 LLM 트레이닝 프로젝트로, 현재 데이터 관리 구조와 체크포인트 저장 방식에 여러 문제가 있습니다.
이 리팩토링은 세 가지 핵심 영역을 개선합니다:

1. **데이터 폴더 구조 개편** — 임시 폴더 난립 해소, 모드별/실행별 명확한 분리
2. **체크포인트 자동 정리** — 무한 증가하는 체크포인트 디렉토리 관리
3. **`generate_training_data.py` 출력 경로 개선** — 하드코딩된 경로 제거, `.txt` 중복 저장 제거, 병합 자동화

## 용어 정의

- **DataManager**: 데이터 폴더 구조를 관리하는 모듈 또는 책임 영역
- **Trainer**: `grinvi/trainer.py`에 정의된 학습 루프 클래스
- **TrainerConfig**: `Trainer`의 설정을 담는 데이터 클래스
- **Generator**: `scripts/generate_training_data.py` 스크립트
- **Checkpoint**: 특정 학습 스텝에서 저장된 모델 가중치 및 설정 파일 (`model.safetensors`, `config.json`)
- **Best Checkpoint**: 평가 손실(eval loss)이 가장 낮은 체크포인트
- **run_디렉토리**: `run_{YYYYMMDD}` 형식의 실행별 출력 폴더
- **JSONL**: JSON Lines 형식 (`.jsonl`), 한 줄에 하나의 JSON 객체
- **text 모드**: 일반 한국어 텍스트 생성 모드
- **qa 모드**: 질문-답변 쌍 생성 모드
- **keep_last_n**: 보관할 최근 체크포인트 개수

---

## 요구사항

### 요구사항 1: 데이터 폴더 구조 표준화

**사용자 스토리:** 개발자로서, 생성된 데이터와 원본 데이터가 명확히 분리된 폴더 구조를 원합니다. 그래야 어떤 데이터가 어디서 왔는지 즉시 파악할 수 있습니다.

#### 인수 기준

1. THE DataManager SHALL `data/raw/` 디렉토리를 원본 소스 데이터(예: `ko_wikipedia`)의 저장 위치로 사용한다.
2. THE DataManager SHALL `data/generated/text/` 디렉토리를 text 모드 생성 결과의 저장 위치로 사용한다.
3. THE DataManager SHALL `data/generated/qa/` 디렉토리를 qa 모드 생성 결과의 저장 위치로 사용한다.
4. THE DataManager SHALL `data/processed/` 디렉토리를 학습용으로 병합된 최종 파일(`train.txt`, `val.txt`)의 저장 위치로 사용한다.
5. THE DataManager SHALL `data/archive/` 디렉토리를 기존 임시 폴더(`test_degs*`, `korean_training`, `qa_production_run` 등)를 임시 보관하는 위치로 사용한다.

---

### 요구사항 2: 실행별 출력 폴더 생성

**사용자 스토리:** 개발자로서, Generator를 실행할 때마다 독립된 폴더에 결과가 저장되기를 원합니다. 그래야 여러 실행 결과가 섞이지 않고 각 실행을 독립적으로 추적할 수 있습니다.

#### 인수 기준

1. WHEN Generator가 실행될 때, THE Generator SHALL `data/generated/{mode}/run_{YYYYMMDD}/` 형식의 디렉토리를 출력 경로 기본값으로 사용한다.
2. WHEN `--output-dir` 인수가 명시적으로 제공될 때, THE Generator SHALL 해당 경로를 기본값 대신 출력 경로로 사용한다.
3. WHEN 동일한 날짜에 Generator가 두 번 이상 실행될 때, THE Generator SHALL `run_{YYYYMMDD}_2`, `run_{YYYYMMDD}_3` 형식으로 충돌 없이 새 디렉토리를 생성한다.
4. WHEN 출력 디렉토리가 존재하지 않을 때, THE Generator SHALL 해당 디렉토리를 자동으로 생성한다.

---

### 요구사항 3: JSONL 단일 포맷 저장

**사용자 스토리:** 개발자로서, 생성된 데이터가 `.jsonl` 형식으로만 저장되기를 원합니다. 그래야 `.txt`와 `.jsonl`이 중복 저장되는 혼란을 없앨 수 있습니다.

#### 인수 기준

1. WHEN Generator가 데이터를 저장할 때, THE Generator SHALL `.jsonl` 형식으로만 저장하고 `.txt` 파일을 별도로 생성하지 않는다.
2. THE Generator SHALL 각 JSONL 항목에 `text`, `prompt`, `mode`, `teacher`, `score`, `timestamp` 필드를 포함한다.
3. WHEN qa 모드로 생성할 때, THE Generator SHALL JSONL 항목에 `question`과 `answer` 필드를 추가로 포함한다.
4. IF 기존 `.txt` 저장 코드가 `run_worker` 함수 내에 존재할 때, THEN THE Generator SHALL 해당 코드를 제거한다.

---

### 요구사항 4: 데이터 병합 옵션 (`--merge`)

**사용자 스토리:** 개발자로서, 생성 완료 후 자동으로 `data/processed/`에 병합하는 옵션을 원합니다. 그래야 수동으로 파일을 합치는 번거로움을 없앨 수 있습니다.

#### 인수 기준

1. WHERE `--merge` 옵션이 활성화된 경우, THE Generator SHALL 실행 완료 후 해당 run 디렉토리의 모든 `.jsonl` 파일을 `data/processed/train.txt`에 병합한다.
2. WHERE `--merge` 옵션이 활성화된 경우, THE Generator SHALL 병합 시 각 JSONL 항목의 `text` 필드 값만 추출하여 `train.txt`에 한 항목씩 줄바꿈으로 구분하여 저장한다.
3. WHERE `--merge` 옵션이 활성화된 경우, THE Generator SHALL 기존 `data/processed/train.txt`가 존재하면 덮어쓰지 않고 내용을 추가(append)한다.
4. IF `data/processed/` 디렉토리가 존재하지 않을 때, THEN THE Generator SHALL 해당 디렉토리를 자동으로 생성한다.
5. WHERE `--merge` 옵션이 비활성화된 경우, THE Generator SHALL 병합 동작을 수행하지 않는다.

---

### 요구사항 5: 체크포인트 보관 개수 제한 (`keep_last_n`)

**사용자 스토리:** 개발자로서, 최근 N개의 체크포인트만 보관하고 오래된 것은 자동 삭제되기를 원합니다. 그래야 디스크 공간이 무한히 증가하는 문제를 해결할 수 있습니다.

#### 인수 기준

1. THE TrainerConfig SHALL `keep_last_n` 파라미터를 지원하며, 기본값은 `5`이다.
2. WHEN Trainer가 새 체크포인트를 저장할 때, THE Trainer SHALL `checkpoints/step-*` 패턴의 디렉토리 목록을 스텝 번호 기준 오름차순으로 정렬한다.
3. WHEN 저장된 체크포인트 수가 `keep_last_n`을 초과할 때, THE Trainer SHALL 가장 오래된 체크포인트 디렉토리부터 삭제하여 총 개수를 `keep_last_n`으로 유지한다.
4. THE Trainer SHALL `checkpoints/best/` 디렉토리에 저장된 Best Checkpoint를 자동 삭제 대상에서 제외한다.
5. IF `keep_last_n`이 `0`으로 설정된 경우, THEN THE Trainer SHALL 체크포인트를 자동으로 삭제하지 않는다.

---

### 요구사항 6: Best Checkpoint 별도 보관

**사용자 스토리:** 개발자로서, 평가 손실이 가장 낮은 모델이 `checkpoints/best/`에 항상 보존되기를 원합니다. 그래야 자동 삭제로 인해 최고 성능 모델을 잃지 않을 수 있습니다.

#### 인수 기준

1. WHEN Trainer가 평가(eval)를 수행할 때, THE Trainer SHALL 현재 평가 손실을 이전 최저 평가 손실과 비교한다.
2. WHEN 현재 평가 손실이 이전 최저 평가 손실보다 낮을 때, THE Trainer SHALL 해당 체크포인트를 `checkpoints/best/`에 저장한다.
3. THE Trainer SHALL `checkpoints/best/`에 저장할 때 기존 내용을 덮어쓴다.
4. IF 평가 데이터 로더(`eval_loader`)가 없을 때, THEN THE Trainer SHALL Best Checkpoint 저장을 수행하지 않는다.
5. THE Trainer SHALL `best_eval_loss` 상태를 학습 세션 내에서 유지하며, 초기값은 양의 무한대(`float('inf')`)로 설정한다.

---

### 요구사항 7: `scripts/train.py` 체크포인트 옵션 노출

**사용자 스토리:** 개발자로서, `train.py` 실행 시 `--keep_last_n` 인수를 통해 체크포인트 보관 개수를 제어하기를 원합니다. 그래야 실행 시점에 유연하게 디스크 사용량을 조절할 수 있습니다.

#### 인수 기준

1. THE `train.py` 스크립트 SHALL `--keep_last_n` CLI 인수를 지원하며, 기본값은 `5`이다.
2. WHEN `train.py`가 `TrainerConfig`를 생성할 때, THE `train.py` SHALL `--keep_last_n` 값을 `TrainerConfig`의 `keep_last_n` 파라미터로 전달한다.
3. WHERE `--keep_last_n 0`이 지정된 경우, THE `train.py` SHALL 체크포인트 자동 삭제 없이 모든 체크포인트를 보관한다.

---

### 요구사항 8: 기존 데이터 아카이브 마이그레이션

**사용자 스토리:** 개발자로서, 기존의 임시 폴더들(`test_degs*`, `korean_training`, `qa_production_run`)이 `data/archive/`로 이동되기를 원합니다. 그래야 데이터 루트가 깔끔하게 정리됩니다.

#### 인수 기준

1. THE DataManager SHALL `data/test_degs`, `data/test_degs_2` ~ `data/test_degs_6`, `data/test_qa_final_v2`, `data/test_qa_fixed`, `data/korean_training`, `data/qa_production_run` 폴더를 `data/archive/` 하위로 이동한다.
2. THE DataManager SHALL `data/ko_wikipedia/` 폴더를 `data/raw/ko_wikipedia/`로 이동한다.
3. IF 이동 대상 폴더가 존재하지 않을 때, THEN THE DataManager SHALL 해당 폴더에 대한 이동 작업을 건너뛴다.
4. THE DataManager SHALL 마이그레이션 완료 후 이동된 폴더 목록과 건너뛴 폴더 목록을 출력한다.
