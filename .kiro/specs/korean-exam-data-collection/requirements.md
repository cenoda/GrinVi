# 요구사항 문서

## 소개

역대 국어 기출문제(수능, 모의고사, 공무원 시험, LEET 등)를 수집하여 GrinVi 한국어 AI 모델 학습 데이터로 변환하는 파이프라인 기능입니다.

기출문제는 고품질 한국어 지문, 문학·비문학 텍스트, 문제-선지-해설 구조를 포함하므로, 기존 위키백과·챗봇 데이터와 차별화된 교육적·논리적 한국어 표현을 학습 데이터로 제공합니다. 수집된 데이터는 기존 파이프라인(`data/processed/train.txt`, JSONL 포맷)과 호환되는 형태로 저장됩니다.

### 수집 대상 데이터 소스

#### HuggingFace 공개 데이터셋 (즉시 사용 가능)

| 데이터셋 | 내용 | 문항 수 | 수집 방법 |
|---|---|---|---|
| `HAERAE-HUB/csatqa` | 수능 국어/언어 기출 (2006~2022) | 1,123개 | URL 직접 다운로드 |
| `KKACHI-HUB/CSAT-KOREAN-2025` | 2025 수능 국어 | 45개 | `load_dataset` |
| `LLMin/final_csat_sft` | 수능 기반 SFT (CoT 해설 포함) | 2,443개 | `load_dataset` |
| `kikikara/Korean-Civil-Service-Examination-Train` | 공무원 시험 국어 (2014~2024) | 340개 | `load_dataset` + task 필터 |
| `kikikara/Korean-Civil-Service-Examination-National` | 국가직 공무원 국어 (2014~2024) | 220개 | `load_dataset` |

#### PDF 크롤링 필요 소스 (공식 사이트)

| 데이터 소스 | 출처 | 연도 범위 | 예상 문항 수 |
|---|---|---|---|
| 수능 기출 PDF | KICE (suneung.re.kr) | 1994~2026 | ~1,350개 (30년 × 45문항) |
| 평가원 6월/9월 모의고사 PDF | KICE (kice.re.kr) | 최근 수년 | ~450개 |
| 임용고시 중등 국어 PDF | KICE (kice.re.kr) | 2019~2026 | ~160개 (8년 × 20문항) |
| LEET 언어이해 PDF | 법학전문대학원협의회 | 2009~현재 | ~510개 (17회 × 30문항) |
| 교육청 모의고사 PDF | 각 시도 교육청 | 최근 수년 | ~수백 개 |

**예상 총 수집 문항 수**: HuggingFace 4,171개 + PDF 크롤링 2,500개 이상 = **6,600개 이상**

---

## 용어 정의

- **Collector**: 기출문제 데이터를 외부 소스(공개 데이터셋, 파일, 웹 크롤링)로부터 수집하는 컴포넌트
- **HFCollector**: HuggingFace Hub에서 데이터셋을 다운로드하는 Collector 구현체
- **PDFCollector**: 공식 사이트에서 PDF를 크롤링하고 다운로드하는 Collector 구현체
- **PDFParser**: PDF 파일에서 기출문제 텍스트를 추출하고 구조화하는 컴포넌트
- **Parser**: 수집된 원시 데이터를 구조화된 `ExamItem` 객체로 변환하는 컴포넌트
- **Formatter**: `ExamItem` 객체를 학습용 텍스트 또는 JSONL 레코드로 변환하는 컴포넌트
- **ExamItem**: 단일 기출문제를 표현하는 데이터 구조 (지문, 문제, 선지, 정답, 해설, 메타데이터 포함)
- **학습 레코드**: GrinVi 학습 파이프라인이 소비하는 JSONL 한 줄 (`{"text": "...", ...}`)
- **지문(Passage)**: 문제에 딸린 읽기 자료 (문학 작품, 비문학 글 등)
- **선지(Choice)**: 객관식 문제의 보기 항목 (①②③④⑤)
- **메타데이터**: 시험 종류, 연도, 월, 문제 번호, 과목 영역 등 문제 식별 정보
- **Pipeline**: Collector → Parser → Formatter → Storage 순서로 실행되는 전체 처리 흐름
- **Storage**: 처리된 데이터를 `data/raw/` 및 `data/processed/`에 저장하는 컴포넌트
- **수능**: 대학수학능력시험 (KICE 주관, 매년 11월)
- **LEET**: 법학적성시험 (법학전문대학원 입학 시험)
- **임용고시**: 교원임용후보자선정경쟁시험 (중등 국어 교과)

---

## 요구사항

### 요구사항 1: HuggingFace 공개 데이터셋으로부터 기출문제 수집

**User Story:** 데이터 엔지니어로서, 공개된 한국어 기출문제 HuggingFace 데이터셋을 자동으로 다운로드하고 싶다. 그래야 수작업 없이 대량의 고품질 학습 데이터를 즉시 확보할 수 있다.

#### 수용 기준

1. WHEN 사용자가 `--source huggingface` 옵션으로 HFCollector를 실행하면, THE HFCollector SHALL 다음 5개 데이터셋을 모두 수집한다: `HAERAE-HUB/csatqa`, `KKACHI-HUB/CSAT-KOREAN-2025`, `LLMin/final_csat_sft`, `kikikara/Korean-Civil-Service-Examination-Train`, `kikikara/Korean-Civil-Service-Examination-National`.
2. WHEN `HAERAE-HUB/csatqa` 데이터셋을 수집하면, THE HFCollector SHALL `https://huggingface.co/datasets/HAERAE-HUB/csatqa/resolve/main/data/csatqa.json` URL에서 직접 다운로드한다 (해당 데이터셋은 `load_dataset` API를 지원하지 않음).
3. WHEN `KKACHI-HUB/CSAT-KOREAN-2025` 데이터셋을 수집하면, THE HFCollector SHALL `load_dataset('KKACHI-HUB/CSAT-KOREAN-2025')` API를 사용하여 `idx`, `paragraph`, `question`, `question_plus`, `A`~`E`, `answer`, `point` 필드를 읽는다.
4. WHEN `LLMin/final_csat_sft` 데이터셋을 수집하면, THE HFCollector SHALL `load_dataset('LLMin/final_csat_sft')` API를 사용하여 `messages` 필드(system/user/assistant 형식, CoT 해설 포함)를 읽는다.
5. WHEN `kikikara/Korean-Civil-Service-Examination-Train` 데이터셋을 수집하면, THE HFCollector SHALL `task='국어'` 조건으로 필터링하여 국어 과목 문항만 수집한다.
6. WHEN `kikikara/Korean-Civil-Service-Examination-National` 데이터셋을 수집하면, THE HFCollector SHALL `split='공무원_국가직'`을 사용하여 데이터를 로드한다.
7. WHEN 다운로드가 완료되면, THE HFCollector SHALL 원시 데이터를 `data/raw/korean_exam/huggingface/` 디렉토리에 데이터셋별 하위 폴더로 저장한다.
8. IF 네트워크 오류 또는 데이터셋 접근 실패가 발생하면, THEN THE HFCollector SHALL 오류 메시지와 재시도 방법을 출력하고 0이 아닌 종료 코드를 반환한다.
9. WHEN 동일한 데이터셋이 이미 `data/raw/korean_exam/huggingface/`에 존재하면, THE HFCollector SHALL 재다운로드를 건너뛰고 캐시된 파일을 사용한다.

---

### 요구사항 2: 로컬 파일로부터 기출문제 수집

**User Story:** 데이터 엔지니어로서, 직접 구한 기출문제 파일(JSON, CSV, TXT)을 파이프라인에 투입하고 싶다. 그래야 다양한 출처의 데이터를 유연하게 활용할 수 있다.

#### 수용 기준

1. WHEN 사용자가 `--source local --input <경로>` 옵션으로 Collector를 실행하면, THE Collector SHALL 지정된 경로의 파일 또는 디렉토리를 입력으로 읽는다.
2. THE Collector SHALL JSON, CSV, TXT 형식의 파일을 지원한다.
3. IF 지정된 경로가 존재하지 않거나 지원하지 않는 형식이면, THEN THE Collector SHALL 구체적인 오류 메시지를 출력하고 처리를 중단한다.
4. WHEN 디렉토리가 입력으로 지정되면, THE Collector SHALL 해당 디렉토리 내 지원 형식의 모든 파일을 재귀적으로 처리한다.

---

### 요구사항 3: PDF 크롤링 및 다운로드

**User Story:** 데이터 엔지니어로서, KICE 공식 사이트와 법학전문대학원협의회 사이트에서 기출문제 PDF를 자동으로 크롤링하고 다운로드하고 싶다. 그래야 수능·모의고사·임용고시·LEET 등 30년치 기출문제를 확보할 수 있다.

#### 수용 기준

1. WHEN 사용자가 `--source pdf --exam csat` 옵션으로 PDFCollector를 실행하면, THE PDFCollector SHALL `https://www.suneung.re.kr/boardCnts/list.do?boardID=1500234&m=0403&s=suneung`에서 수능 국어/언어 영역 PDF 목록을 크롤링하고 1994년부터 현재까지의 PDF를 다운로드한다.
2. WHEN 사용자가 `--source pdf --exam mock` 옵션으로 PDFCollector를 실행하면, THE PDFCollector SHALL `https://www.kice.re.kr/boardCnts/list.do?boardID=1500212&m=030306&s=kice`에서 평가원 6월·9월 모의고사 국어 PDF를 크롤링하고 다운로드한다.
3. WHEN 사용자가 `--source pdf --exam teacher` 옵션으로 PDFCollector를 실행하면, THE PDFCollector SHALL KICE 사이트에서 임용고시 중등 국어 1차 시험 PDF를 2019년부터 현재까지 크롤링하고 다운로드한다.
4. WHEN 사용자가 `--source pdf --exam leet` 옵션으로 PDFCollector를 실행하면, THE PDFCollector SHALL 법학전문대학원협의회 사이트에서 LEET 언어이해 영역 PDF를 2009년부터 현재까지 크롤링하고 다운로드한다.
5. WHEN 사용자가 `--source pdf --exam district` 옵션으로 PDFCollector를 실행하면, THE PDFCollector SHALL 서울·경기·인천 교육청 사이트에서 3월·4월·7월·10월 모의고사 국어 PDF를 크롤링하고 다운로드한다.
6. WHEN PDF 다운로드가 완료되면, THE PDFCollector SHALL 파일을 `data/raw/korean_exam/pdf/{exam_type}/{year}/` 디렉토리 구조로 저장한다.
7. IF 크롤링 중 HTTP 오류(4xx, 5xx) 또는 타임아웃이 발생하면, THEN THE PDFCollector SHALL 해당 URL을 건너뛰고 오류 내용을 로그에 기록한 뒤 나머지 PDF 다운로드를 계속한다.
8. WHEN 동일한 PDF 파일이 이미 존재하면, THE PDFCollector SHALL 재다운로드를 건너뛰고 기존 파일을 사용한다.
9. THE PDFCollector SHALL 크롤링 완료 후 다운로드 성공 수, 실패 수, 건너뛴 수, 총 파일 크기를 출력한다.

---

### 요구사항 4: PDF 파싱 및 기출문제 구조화

**User Story:** 데이터 엔지니어로서, 다운로드된 기출문제 PDF에서 지문·문제·선지·정답을 자동으로 추출하고 싶다. 그래야 PDF 형태의 기출문제를 학습 데이터로 활용할 수 있다.

#### 수용 기준

1. WHEN PDFParser가 수능/모의고사 PDF를 처리하면, THE PDFParser SHALL 각 문항의 지문(passage), 문제 텍스트(question), 5지선다 선지(choices), 정답 번호(answer)를 추출한다.
2. WHEN PDFParser가 임용고시 PDF를 처리하면, THE PDFParser SHALL 각 문항의 지문, 문제 텍스트, 4지선다 선지, 정답 번호를 추출한다.
3. WHEN PDFParser가 LEET 언어이해 PDF를 처리하면, THE PDFParser SHALL 각 지문 세트의 지문과 해당 문항들(5지선다)을 추출한다.
4. IF PDF에서 텍스트 추출이 불가능한 페이지(스캔 이미지 등)가 포함되면, THEN THE PDFParser SHALL 해당 페이지를 건너뛰고 경고 로그에 파일명과 페이지 번호를 기록한다.
5. THE PDFParser SHALL 추출된 텍스트에서 머리글·바닥글·페이지 번호·저작권 표시 등 비문제 텍스트를 제거한다.
6. THE PDFParser SHALL 파싱 완료 후 추출된 문항 수, 실패한 페이지 수, 소요 시간을 출력한다.

---

### 요구사항 5: 기출문제 파싱 및 구조화

**User Story:** 데이터 엔지니어로서, 다양한 형식의 원시 기출문제 데이터를 일관된 구조로 변환하고 싶다. 그래야 후속 처리 단계가 단일 인터페이스를 사용할 수 있다.

#### 수용 기준

1. THE Parser SHALL 원시 데이터를 `ExamItem` 구조로 변환한다. `ExamItem`은 다음 필드를 포함한다: `passage` (지문, 선택적), `question` (문제 텍스트), `choices` (선지 목록, 선택적), `answer` (정답, 선택적), `explanation` (해설, 선택적), `metadata` (시험 종류·연도·월·번호·영역).
2. WHEN 입력 데이터에 지문이 없는 독립 문제가 포함되면, THE Parser SHALL `passage` 필드를 빈 문자열로 설정하고 파싱을 계속한다.
3. IF 필수 필드인 `question`이 비어 있거나 누락된 레코드가 있으면, THEN THE Parser SHALL 해당 레코드를 건너뛰고 경고 로그에 레코드 식별 정보를 기록한다.
4. THE Parser SHALL 파싱 완료 후 처리된 항목 수, 건너뛴 항목 수, 소요 시간을 요약 출력한다.
5. FOR ALL 유효한 `ExamItem` 객체에 대해, Parser가 동일한 입력을 두 번 파싱하면 THE Parser SHALL 동일한 `ExamItem` 객체를 반환한다 (결정론적 파싱).

---

### 요구사항 6: 학습 데이터 포맷 변환

**User Story:** 데이터 엔지니어로서, 파싱된 기출문제를 GrinVi 학습 파이프라인이 바로 사용할 수 있는 형식으로 변환하고 싶다. 그래야 기존 훈련 스크립트를 수정하지 않아도 된다.

#### 수용 기준

1. THE Formatter SHALL `ExamItem`을 `{"text": "...", "source": "korean_exam", "exam_type": "...", "year": ..., "subject": "..."}` 형식의 JSONL 레코드로 변환한다.
2. WHEN `passage`가 존재하면, THE Formatter SHALL `text` 필드에 지문을 먼저 포함하고 이어서 문제와 선지를 포함한다.
3. WHEN `explanation`이 존재하면, THE Formatter SHALL `text` 필드 끝에 해설을 포함한다.
4. THE Formatter SHALL `text` 필드의 길이가 50자 미만인 레코드를 필터링하고 필터링된 수를 로그에 기록한다.
5. WHERE `--format plain_text` 옵션이 활성화되면, THE Formatter SHALL JSONL 대신 순수 텍스트 형식으로 출력한다.
6. FOR ALL 유효한 `ExamItem` 객체에 대해, `Formatter`가 변환한 레코드를 다시 파싱하면 THE Formatter SHALL 원본 `ExamItem`의 핵심 텍스트 내용을 복원할 수 있어야 한다 (라운드트립 속성).

---

### 요구사항 7: 처리된 데이터 저장 및 기존 파이프라인 통합

**User Story:** 데이터 엔지니어로서, 변환된 기출문제 데이터를 기존 학습 데이터와 합쳐서 바로 훈련에 사용하고 싶다. 그래야 별도의 수작업 병합 없이 전체 파이프라인이 자동으로 동작한다.

#### 수용 기준

1. THE Storage SHALL 변환된 JSONL 레코드를 `data/raw/korean_exam/korean_exam_{YYYYMMDD}.jsonl` 파일에 저장한다.
2. WHEN `--merge` 옵션이 활성화되면, THE Storage SHALL `data/processed/train.txt`에 `text` 필드 값을 한 줄씩 추가(append)한다.
3. THE Storage SHALL 저장 완료 후 저장된 레코드 수, 파일 크기, 저장 경로를 출력한다.
4. IF 저장 중 디스크 쓰기 오류가 발생하면, THEN THE Storage SHALL 오류 메시지를 출력하고 부분 저장된 파일을 삭제한 뒤 0이 아닌 종료 코드를 반환한다.
5. WHEN `--merge` 옵션 없이 실행되면, THE Storage SHALL `data/processed/train.txt`를 수정하지 않는다.

---

### 요구사항 8: 커맨드라인 인터페이스

**User Story:** 데이터 엔지니어로서, 기존 스크립트(`download_korean_data.py`, `prepare_data.py`)와 일관된 방식으로 기출문제 수집 파이프라인을 실행하고 싶다. 그래야 학습 곡선 없이 바로 사용할 수 있다.

#### 수용 기준

1. THE Pipeline SHALL `scripts/collect_korean_exam.py` 스크립트로 실행 가능하다.
2. THE Pipeline SHALL 다음 CLI 인수를 지원한다: `--source` (huggingface/local/pdf, 필수), `--exam` (csat/mock/teacher/leet/district, `--source pdf` 시 선택), `--input` (로컬 경로, `--source local` 시 필수), `--out` (출력 디렉토리, 기본값 `data`), `--merge` (train.txt 병합 플래그), `--format` (jsonl/plain_text, 기본값 jsonl), `--max_items` (최대 처리 항목 수).
3. WHEN `--help` 옵션으로 실행하면, THE Pipeline SHALL 각 인수의 설명과 사용 예시를 출력한다.
4. IF 필수 인수가 누락된 채로 실행되면, THEN THE Pipeline SHALL 누락된 인수 이름과 올바른 사용법을 출력하고 종료 코드 1을 반환한다.
5. WHEN 파이프라인이 정상 완료되면, THE Pipeline SHALL 다음 단계(토크나이저 훈련, 모델 훈련) 명령어 예시를 출력한다.

---

### 요구사항 9: 데이터 품질 검증

**User Story:** 데이터 엔지니어로서, 수집된 기출문제 데이터의 품질을 자동으로 검증하고 싶다. 그래야 저품질 데이터가 학습에 투입되는 것을 방지할 수 있다.

#### 수용 기준

1. THE Pipeline SHALL 처리 완료 후 다음 통계를 출력한다: 총 수집 항목 수, 유효 항목 수, 필터링된 항목 수, 시험 종류별 분포, 연도별 분포.
2. WHEN `--validate` 옵션이 활성화되면, THE Pipeline SHALL 각 `ExamItem`에 대해 한국어 문자 비율이 50% 이상인지 검사하고, 미달 항목을 필터링한다.
3. IF 유효 항목 수가 전체 수집 항목 수의 50% 미만이면, THEN THE Pipeline SHALL 경고 메시지를 출력하고 사용자에게 입력 데이터 품질을 확인하도록 안내한다.
4. THE Pipeline SHALL 중복 문제(동일한 `question` 텍스트)를 탐지하고 중복 제거 후 중복 수를 로그에 기록한다.
