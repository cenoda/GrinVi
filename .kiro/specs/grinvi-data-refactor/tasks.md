# 구현 계획: GrinVi 데이터 리팩토링

## 개요

설계 문서에 정의된 네 가지 컴포넌트를 순서대로 구현합니다.
각 태스크는 이전 태스크 위에 쌓이며, 마지막에 모든 컴포넌트를 연결합니다.
테스트는 Hypothesis 기반 속성 테스트(PBT)와 단위 테스트로 구성됩니다.

## 태스크

- [x] 1. DataManager 구현 (`scripts/migrate_data.py`)
  - [x] 1.1 `DataManager` 클래스 및 표준 경로 상수 작성
    - `BASE`, `RAW_DIR`, `GENERATED_DIR`, `PROCESSED_DIR`, `ARCHIVE_DIR` 상수 정의
    - `setup_dirs()` 메서드 구현: 4개 표준 디렉토리 생성
    - `migrate()` 메서드 구현: 마이그레이션 대상 폴더 이동, `{"moved": [...], "skipped": [...]}` 반환
    - 이동 대상 폴더 부재 시 건너뜀, 목적지 충돌 시 오류 발생 처리
    - `__main__` 블록: `setup_dirs()` → `migrate()` → 결과 출력
    - _요구사항: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4_

  - [ ]* 1.2 `DataManager` 단위 테스트 작성
    - `setup_dirs()` 호출 후 4개 표준 디렉토리 존재 확인
    - 존재하지 않는 폴더 마이그레이션 시 오류 없이 건너뜀 확인
    - 반환값에 `moved`, `skipped` 키 포함 확인
    - _요구사항: 8.3, 8.4_

- [x] 2. Generator 출력 경로 로직 개선 (`scripts/generate_training_data.py`)
  - [x] 2.1 `resolve_output_dir(mode, output_dir_arg)` 함수 구현
    - `output_dir_arg`가 있으면 해당 경로 반환
    - 없으면 `data/generated/{mode}/run_{YYYYMMDD}/` 생성; 동일 날짜 충돌 시 `_2`, `_3` 증가
    - _요구사항: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 2.2 Property 1 속성 테스트 작성 — 출력 경로 형식 준수
    - **Property 1: 출력 경로 형식 준수**
    - **Validates: 요구사항 2.1**
    - `# Feature: grinvi-data-refactor, Property 1: 출력 경로 형식 준수`

  - [ ]* 2.3 Property 2 속성 테스트 작성 — 명시적 출력 경로 우선
    - **Property 2: 명시적 출력 경로 우선**
    - **Validates: 요구사항 2.2**
    - `# Feature: grinvi-data-refactor, Property 2: 명시적 출력 경로 우선`

  - [ ]* 2.4 Property 3 속성 테스트 작성 — 동일 날짜 고유 디렉토리 보장
    - **Property 3: 동일 날짜 실행 시 고유 디렉토리 보장**
    - **Validates: 요구사항 2.3**
    - `# Feature: grinvi-data-refactor, Property 3: 동일 날짜 고유 디렉토리 보장`

- [x] 3. Generator JSONL 전용 저장 및 `--merge` 옵션 구현 (`scripts/generate_training_data.py`)
  - [x] 3.1 `run_worker` 함수에서 `.txt` 저장 코드 제거
    - `open(output_path, "a", ...)` 블록 삭제
    - `.jsonl` 파일만 저장하도록 수정
    - qa 모드 시 JSONL 항목에 `question`, `answer` 필드 추가
    - _요구사항: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 3.2 Property 4 속성 테스트 작성 — JSONL 전용 저장
    - **Property 4: JSONL 전용 저장 (txt 미생성)**
    - **Validates: 요구사항 3.1**
    - `# Feature: grinvi-data-refactor, Property 4: JSONL 전용 저장`

  - [ ]* 3.3 Property 5 속성 테스트 작성 — JSONL 필수 필드 포함
    - **Property 5: JSONL 필수 필드 포함**
    - **Validates: 요구사항 3.2**
    - `# Feature: grinvi-data-refactor, Property 5: JSONL 필수 필드 포함`

  - [ ]* 3.4 Property 6 속성 테스트 작성 — qa 모드 추가 필드 포함
    - **Property 6: qa 모드 추가 필드 포함**
    - **Validates: 요구사항 3.3**
    - `# Feature: grinvi-data-refactor, Property 6: qa 모드 추가 필드 포함`

  - [x] 3.5 `merge_to_processed(run_dir, processed_dir)` 함수 구현
    - `run_dir`의 모든 `.jsonl` 파일에서 `text` 필드 추출
    - `processed_dir/train.txt`에 append 저장 (덮어쓰기 금지)
    - `text` 필드 없는 항목 건너뜀 및 경고 출력
    - `processed_dir` 없으면 자동 생성
    - 병합된 항목 수 반환
    - _요구사항: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 3.6 Property 7 속성 테스트 작성 — 병합 시 text 필드 보존
    - **Property 7: 병합 시 모든 text 필드 보존**
    - **Validates: 요구사항 4.1, 4.2**
    - `# Feature: grinvi-data-refactor, Property 7: 병합 시 text 필드 보존`

  - [ ]* 3.7 Property 8 속성 테스트 작성 — 병합 append 동작
    - **Property 8: 병합 시 기존 내용 보존 (append)**
    - **Validates: 요구사항 4.3**
    - `# Feature: grinvi-data-refactor, Property 8: 병합 append 동작`

  - [x] 3.8 `main()` 함수에 `resolve_output_dir` 및 `--merge` 옵션 연결
    - `--output-dir` 기본값 제거, `resolve_output_dir` 호출로 대체
    - `--merge` CLI 인수 추가 (`action="store_true"`)
    - 생성 완료 후 `--merge` 활성화 시 `merge_to_processed` 호출
    - _요구사항: 2.1, 2.2, 4.1, 4.5_

  - [ ]* 3.9 Generator 단위 테스트 작성
    - `--merge` 없이 실행 시 `train.txt` 미생성 확인
    - `data/processed/` 없을 때 `--merge` 실행 시 디렉토리 자동 생성 확인
    - _요구사항: 4.4, 4.5_

- [x] 4. 체크포인트 정리 및 Best Checkpoint 저장 (`grinvi/trainer.py`)
  - [x] 4.1 `TrainerConfig`에 `keep_last_n` 파라미터 추가
    - `__init__` 시그니처에 `keep_last_n: int = 5` 추가
    - _요구사항: 5.1_

  - [x] 4.2 `Trainer.__init__`에 `best_eval_loss` 상태 추가
    - `self.best_eval_loss: float = float("inf")` 초기화
    - _요구사항: 6.5_

  - [x] 4.3 `_cleanup_checkpoints()` 메서드 구현
    - `step-숫자` 패턴 디렉토리만 수집 (`best/` 제외)
    - 스텝 번호 기준 오름차순 정렬
    - `keep_last_n` 초과분 삭제 (삭제 실패 시 경고 후 계속)
    - `keep_last_n == 0`이면 즉시 반환
    - _요구사항: 5.2, 5.3, 5.4, 5.5_

  - [ ]* 4.4 Property 9 속성 테스트 작성 — 체크포인트 정렬 정확성
    - **Property 9: 체크포인트 정렬 정확성**
    - **Validates: 요구사항 5.2**
    - `# Feature: grinvi-data-refactor, Property 9: 체크포인트 정렬 정확성`

  - [ ]* 4.5 Property 10 속성 테스트 작성 — keep_last_n 개수 유지
    - **Property 10: keep_last_n 개수 유지**
    - **Validates: 요구사항 5.3, 5.4**
    - `# Feature: grinvi-data-refactor, Property 10: keep_last_n 개수 유지`

  - [x] 4.6 `_save_best()` 메서드 구현
    - `checkpoints/best/`에 현재 모델 저장 (덮어쓰기)
    - 저장 실패 시 경고 출력 후 계속 진행
    - _요구사항: 6.2, 6.3_

  - [x] 4.7 `_eval()` 메서드에 best checkpoint 저장 로직 추가
    - `avg < self.best_eval_loss` 조건 시 `best_eval_loss` 갱신 및 `_save_best()` 호출
    - `eval_loader`가 `None`이면 전체 건너뜀 (기존 동작 유지)
    - _요구사항: 6.1, 6.2, 6.4_

  - [x] 4.8 `_save()` 메서드에 `_cleanup_checkpoints()` 호출 추가
    - 기존 체크포인트 저장 후 `_cleanup_checkpoints()` 호출
    - `"final"` 태그 저장 시에는 정리 건너뜀
    - _요구사항: 5.2, 5.3_

  - [ ]* 4.9 Property 11 속성 테스트 작성 — best_eval_loss 단조 감소 유지
    - **Property 11: best_eval_loss 단조 감소 유지**
    - **Validates: 요구사항 6.1, 6.2, 6.5**
    - `# Feature: grinvi-data-refactor, Property 11: best_eval_loss 단조 감소 유지`

  - [ ]* 4.10 `TrainerConfig` 및 `Trainer` 단위 테스트 작성
    - 기본 생성 시 `keep_last_n == 5` 확인
    - `Trainer` 초기화 시 `best_eval_loss == float('inf')` 확인
    - _요구사항: 5.1, 6.5_

- [x] 5. 체크포인트 정리 중간 점검
  - 모든 테스트 통과 확인, 궁금한 점이 있으면 사용자에게 질문하세요.

- [x] 6. `scripts/train.py` CLI 인수 추가
  - [x] 6.1 `--keep_last_n` 인수 추가 및 `TrainerConfig` 연결
    - `parse_args()`에 `--keep_last_n` 인수 추가 (기본값 `5`)
    - `TrainerConfig` 생성 시 `keep_last_n=args.keep_last_n` 전달
    - _요구사항: 7.1, 7.2, 7.3_

  - [ ]* 6.2 `train.py` CLI 단위 테스트 작성
    - `--keep_last_n` 기본값 `5` 확인
    - 특정 값 지정 시 `TrainerConfig`에 올바르게 전달 확인
    - `--keep_last_n 0` 지정 시 자동 삭제 비활성화 확인
    - _요구사항: 7.1, 7.2, 7.3_

- [x] 7. 최종 점검
  - 모든 테스트 통과 확인, 궁금한 점이 있으면 사용자에게 질문하세요.

## 참고

- `*` 표시 태스크는 선택 사항으로, MVP 구현 시 건너뛸 수 있습니다.
- 각 태스크는 특정 요구사항을 참조하여 추적 가능성을 보장합니다.
- 속성 테스트는 Hypothesis 라이브러리를 사용하며, 각 속성은 최소 100회 반복 실행합니다.
- 단위 테스트와 속성 테스트는 `tests/test_grinvi_data_refactor.py`에 작성합니다.
