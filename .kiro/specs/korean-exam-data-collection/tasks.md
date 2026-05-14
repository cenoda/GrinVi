# 구현 계획: 국어 기출문제 데이터 수집 파이프라인

## 개요

역대 국어 기출문제를 수집하여 GrinVi 학습 데이터로 변환하는 파이프라인을 구현합니다.
구현 순서는 데이터 모델 → 검증기 → 포매터 → HF 수집 → 저장소 → 파이프라인/CLI → PDF 수집 순으로 진행합니다.

## Tasks

- [x] 1. 프로젝트 구조 및 데이터 모델 설정
  - [x] 1.1 패키지 디렉토리 구조 생성 및 의존성 설정
    - `grinvi/korean_exam/` 패키지 디렉토리 및 `__init__.py` 생성
    - `grinvi/korean_exam/collectors/`, `grinvi/korean_exam/parsers/` 하위 패키지 생성
    - `pyproject.toml`에 `korean_exam` optional dependency 그룹 추가 (`datasets`, `huggingface-hub`, `pdfplumber`, `requests`, `beautifulsoup4`)
    - _요구사항: 8.1_

  - [x] 1.2 ExamMetadata 및 ExamItem 데이터 모델 구현
    - `grinvi/korean_exam/models.py` 생성
    - `ExamMetadata` dataclass 구현 (`exam_type`, `year`, `month`, `question_number`, `subject`, `source_dataset`, `source_file`)
    - `ExamItem` dataclass 구현 (`passage`, `question`, `choices`, `answer`, `explanation`, `metadata`)
    - `ExamItem.is_valid()` 메서드 구현 (question 비어있지 않고 한국어 비율 50% 이상 검사)
    - _요구사항: 5.1, 5.2, 5.3_

- [x] 2. Validator 구현
  - [x] 2.1 Validator 클래스 구현
    - `grinvi/korean_exam/validator.py` 생성
    - `korean_ratio(text)` 메서드 구현: 한국어 문자(가-힣) 비율 계산
    - `validate(item)` 메서드 구현: question 비어있지 않고 한국어 비율 ≥ 0.5 검사
    - `deduplicate(items)` 메서드 구현: question 텍스트 기준 중복 제거 (set 기반)
    - _요구사항: 9.2, 9.4_

  - [ ]* 2.2 Property 테스트: 한국어 비율 정확성
    - **Property 5: 한국어 비율 정확성**
    - `korean_ratio(text) >= 0.5`인 텍스트만 `validate()`를 통과하는지 검증
    - Hypothesis로 한국어/비한국어 혼합 문자열 생성하여 테스트
    - **검증 대상: 요구사항 9.2**

  - [ ]* 2.3 Property 테스트: 중복 제거 멱등성
    - **Property 4: 중복 제거 멱등성**
    - `deduplicate(deduplicate(items)) == deduplicate(items)` 검증
    - Hypothesis로 ExamItem 리스트 생성하여 테스트
    - **검증 대상: 요구사항 9.4**

- [x] 3. Formatter 구현
  - [x] 3.1 Formatter 클래스 구현
    - `grinvi/korean_exam/formatter.py` 생성
    - `_build_text(item)` 메서드 구현: `[지문]`, `[문제]`, 선지(①②③④⑤), `[해설]` 구조로 텍스트 조합
    - `format(item)` 메서드 구현: ExamItem → TrainingRecord dict 변환, `MIN_TEXT_LENGTH=50` 미만 필터링
    - _요구사항: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 3.2 Property 테스트: 라운드트립 일관성
    - **Property 2: 라운드트립**
    - `Formatter.format(item)`의 `text` 필드에서 원본 `question`과 `passage` 핵심 내용 복원 가능 검증
    - Hypothesis로 ExamItem 생성하여 포맷 후 `[문제]`/`[지문]` 섹션 파싱으로 원본 복원 테스트
    - **검증 대상: 요구사항 6.6**

  - [ ]* 3.3 Property 테스트: 필터링 단조성
    - **Property 3: 필터링 단조성**
    - `min_length`를 높일수록 출력 레코드 수가 단조 감소하는지 검증
    - Hypothesis로 ExamItem 리스트와 두 개의 min_length 값(a ≤ b) 생성하여 `count(a) >= count(b)` 테스트
    - **검증 대상: 요구사항 6.4**

- [x] 4. HFCollector 및 HF 파서 구현
  - [x] 4.1 BaseCollector 추상 클래스 구현
    - `grinvi/korean_exam/collectors/base.py` 생성
    - `BaseCollector` ABC 정의: `collect() -> Iterator[Path]`, `_is_cached(path)` 메서드
    - _요구사항: 1.9_

  - [x] 4.2 HFCollector 구현
    - `grinvi/korean_exam/collectors/hf_collector.py` 생성
    - `HF_DATASETS` 설정 리스트 정의 (5개 데이터셋 메타정보)
    - `csatqa`는 URL 직접 다운로드, 나머지는 `load_dataset` API 사용
    - `kikikara` 데이터셋은 `task='국어'` 필터링 적용
    - 캐시 확인 로직 (`_is_cached`) 구현
    - 수집된 데이터를 `data/raw/korean_exam/huggingface/{id}/raw.jsonl`로 저장
    - 네트워크 오류 시 에러 메시지 출력 및 비정상 종료 코드 반환
    - _요구사항: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 4.3 BaseParser 추상 클래스 구현
    - `grinvi/korean_exam/parsers/base.py` 생성
    - `BaseParser` ABC 정의: `parse(raw: dict) -> ExamItem`, `parse_file(path: Path) -> List[ExamItem]`
    - _요구사항: 5.1_

  - [x] 4.4 HF 포맷별 파서 구현
    - `grinvi/korean_exam/parsers/hf_parsers.py` 생성
    - `CSATQAParser` 구현: `test_name` 정규식 파싱으로 연도/시험종류 추출, `option#1~5` → choices 변환
    - `CSATKKACHIParser` 구현: `paragraph`, `question`, `question_plus`, `A~E` 필드 매핑
    - `SFTParser` 구현: `messages` 필드에서 user/assistant 메시지 분리, 지문/문제/선지 추출
    - `CivilParser` 구현: `문제 내용`, `선택지 1~4`, `정답`, `year`, `source` 필드 매핑
    - 각 파서에서 `question` 비어있는 레코드 건너뛰기 및 경고 로그
    - _요구사항: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 4.5 Property 테스트: 결정론적 파싱
    - **Property 1: 결정론적 파싱**
    - 동일한 입력 dict를 두 번 파싱하면 동일한 ExamItem 반환 검증
    - Hypothesis로 각 파서 포맷에 맞는 입력 dict 생성하여 `parse(x) == parse(x)` 테스트
    - **검증 대상: 요구사항 5.5**

- [x] 5. 체크포인트 - HF 수집 파이프라인 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의.
  - HFCollector + HF 파서 + Validator + Formatter 통합 동작 확인.

- [x] 6. Storage 구현
  - [x] 6.1 Storage 클래스 구현
    - `grinvi/korean_exam/storage.py` 생성
    - `save(records, out_dir, merge)` 메서드 구현
    - JSONL 저장: `data/raw/korean_exam/korean_exam_{YYYYMMDD}.jsonl`
    - `--merge` 시 `data/processed/train.txt`에 `text` 필드 append
    - 디스크 쓰기 오류 시 부분 저장 파일 삭제 및 예외 전파
    - 저장 완료 후 레코드 수, 파일 크기, 경로 출력
    - _요구사항: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 7. Pipeline 오케스트레이터 및 CLI 구현
  - [x] 7.1 Pipeline 클래스 구현
    - `grinvi/korean_exam/pipeline.py` 생성
    - `run(args) -> PipelineResult` 메서드 구현
    - Collector → Parser → Validator → Formatter → Storage 순서 실행
    - `--validate` 옵션 시 한국어 비율 검증 적용
    - 중복 제거 항상 적용
    - `--max_items` 옵션 처리
    - 처리 통계 출력: 총 수집 수, 유효 수, 필터링 수, 시험 종류별/연도별 분포
    - 유효 항목 50% 미만 시 경고 메시지 출력
    - _요구사항: 9.1, 9.2, 9.3, 9.4_

  - [x] 7.2 CLI 스크립트 구현
    - `scripts/collect_korean_exam.py` 생성
    - `argparse` 기반 CLI 인수 파싱: `--source`, `--exam`, `--input`, `--out`, `--merge`, `--format`, `--max_items`, `--validate`
    - 필수 인수 누락 시 사용법 출력 및 종료 코드 1 반환
    - `--help` 옵션 시 각 인수 설명 및 사용 예시 출력
    - 정상 완료 시 다음 단계 명령어 예시 출력 (토크나이저 훈련, 모델 훈련)
    - _요구사항: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 8. 체크포인트 - 전체 HF 파이프라인 E2E 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의.
  - `python scripts/collect_korean_exam.py --source huggingface --out data --validate` 실행 가능 확인.

- [x] 9. PDFCollector 및 PDF 파서 구현
  - [x] 9.1 PDFCollector 및 크롤러 구현
    - `grinvi/korean_exam/collectors/pdf_collector.py` 생성
    - `PDFCollector` 클래스 구현: 크롤러 조합 및 PDF 다운로드 오케스트레이션
    - `CSATCrawler` 구현: `suneung.re.kr` 게시판 크롤링, PDF 링크 추출 및 다운로드
    - `MockCrawler` 구현: `kice.re.kr` 모의고사 게시판 크롤링
    - `TeacherCrawler` 구현: `kice.re.kr` 임용고시 게시판 크롤링
    - `LEETCrawler` 구현: `moja.uwayapply.com` LEET 기출 크롤링
    - `DistrictCrawler` 구현: 서울·경기·인천 교육청 모의고사 크롤링
    - 공통 다운로드 로직: 캐시 확인, 타임아웃 처리, 에러 로깅 후 계속 진행
    - 크롤링 완료 후 성공/실패/건너뛴 수 및 총 파일 크기 출력
    - _요구사항: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x] 9.2 PDF 파서 구현
    - `grinvi/korean_exam/parsers/pdf_parser.py` 생성
    - `PDFParser` 기본 클래스: `pdfplumber` 기반 텍스트 추출, 노이즈 패턴 제거
    - `CSATPDFParser` 구현: 문항 번호 패턴 탐지, 지문/선지/정답 추출 (5지선다)
    - `TeacherPDFParser` 구현: 4지선다 패턴, 서술형 문항 처리
    - `LEETPDFParser` 구현: 지문 세트 구조 파싱, 문항 그룹핑
    - 스캔 이미지 페이지 건너뛰기 및 경고 로그
    - 파싱 통계 출력: 추출 문항 수, 실패 페이지 수, 소요 시간
    - _요구사항: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 9.3 LocalCollector 및 LocalParser 구현
    - `grinvi/korean_exam/collectors/local_collector.py` 생성: 로컬 파일/디렉토리 읽기
    - `grinvi/korean_exam/parsers/local_parser.py` 생성: JSON, CSV, TXT 형식 파싱
    - 존재하지 않는 경로 또는 미지원 형식 시 에러 메시지 출력
    - 디렉토리 입력 시 재귀적 파일 탐색
    - _요구사항: 2.1, 2.2, 2.3, 2.4_

- [x] 10. 최종 체크포인트 - 전체 파이프라인 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의.
  - PDF 소스 포함 전체 CLI 옵션 동작 확인.

## 참고 사항

- `*` 표시된 태스크는 선택적이며 빠른 MVP를 위해 건너뛸 수 있습니다
- 각 태스크는 추적 가능성을 위해 구체적 요구사항을 참조합니다
- 체크포인트에서 점진적 검증을 수행합니다
- Property 테스트는 설계 문서의 정확성 속성을 검증합니다
- 단위 테스트는 구체적 예시와 엣지 케이스를 검증합니다
