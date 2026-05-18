# 2026-05-17 학습 세션 — 무엇을 잘못했고, 무엇을 배웠나

이번 세션은 한 모델을 살리고 다음 사이클에서 시간을 아끼기 위한 후일담입니다.
**다음번엔 이 문서만 그대로 따라가도 같은 함정에 안 빠집니다.**

---

## 0. 한 줄 요약

> **모델 자체보다 토크나이저 + 데이터 백업이 더 무서운 단일 장애점**이다.

3시간 학습한 137M 모델을 잘 만들고, **토크나이저를 백업 안 해서 서버 터지자 추론 불가**가 됐다.
나중에 로컬에서 결정적 재학습으로 복구는 됐지만 추가로 ~2시간 + 25GB 다운로드 + 32코어 풀가동 필요했음.

---

## 1. 시간순 실수 + 교훈

### ❌ 실수 1 — 옛 토크나이저로 4000 스텝 학습한 뒤 결과가 이상하다고 의심함

**증상**: 생성 시 `<unk>`, `**`, 의미 없는 조사 반복.

**원인**: 위키 데이터에 한자(예: `KBS木浦放送局`)가 잔뜩 있는데 vocab 64K 컷오프에 잘려서 학습 데이터의 **7.14% 토큰이 `<|unk|>`** 였음. 모델이 `<|unk|>`를 "흔한 토큰"으로 학습.

**교훈**:
- 학습 시작 전에 **반드시 `audit_unk.py`로 UNK 비율 측정**
- 5% 넘으면 토크나이저부터 손봐야 함
- "감"이나 "AI의 안심시키는 말"에 의존하지 말고 **숫자로 검증**

### ❌ 실수 2 — checkpoint 폴더를 git에 푸시함 (3.5GB)

**증상**: `git push`가 49분 돌고도 안 끝남. 결국 GitHub 100MB 제한으로 거부됐을 것.

**원인**: `checkpoints_medium/step-500/model.safetensors`, `step-2500/model.safetensors` 각 1.77GB가 커밋에 섞임. `.gitignore`에 `checkpoints/`는 있었지만 `checkpoints_medium/`은 없었음.

**해결**:
- `git filter-branch --index-filter` 로 히스토리에서 안전 제거
- `.gitignore`에 `checkpoints_*/`와 `*.safetensors` 추가

**교훈**: 학습 시작 전에 `.gitignore`가 모든 모델/체크포인트 패턴을 잡는지 확인.

### ❌ 실수 3 — 진짜 진단을 안 하고 "괜찮다"만 반복

**증상**: 이전 AI들이 loss 그래프만 보고 "학습은 정상"이라며 시간/돈만 날림.

**해결책**: `scripts/diagnose.py` 만들어서 **6가지 항목을 직접 측정**:
1. 토크나이저 vocab 크기/특수토큰
2. 프롬프트 인코드 시 UNK 비율
3. 인코드→디코드 라운드트립 보존 여부
4. **학습 데이터 한 줄 인코드 시 UNK 비율** ← 진짜 원인 찾은 곳
5. **모델 vocab_size vs 토크나이저 vocab_size 일치** ← 가장 무서운 사일런트 버그
6. 모델 top-10 다음 토큰 확률 분포

**교훈**: 새 세션 시작하면 무조건 `diagnose.py`부터 한 번 돌려라.

### ❌ 실수 4 — `scripts/train.py:117`에 토크나이저 경로 하드코딩

**문제 코드**:
```python
def __setstate__(self, state):
    ...
    self.tokenizer = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
```

**증상 가능성**: 토크나이저 파일 이름을 바꾸면 DataLoader 워커가 다시 fork될 때 로드 실패.

**교훈**:
- 토크나이저는 **반드시 `data/raw/ko_wikipedia/ko_tokenizer.json`** 이 정식 이름
- 새 버전 만들 땐 `_new.json` 같은 임시 이름으로 만들고 검증 후 mv로 교체
- 또는 `__setstate__`에서 경로를 self에 저장하도록 코드 수정 (다음 사이클 TODO)

### 🔥 실수 5 — 토크나이저를 gdrive에 백업 안 함

**문제**: `backup_checkpoints.sh`는 체크포인트만 백업. **토크나이저 자체는 서버에만 존재**했음. 서버 terminate 후 토크나이저 손실.

**복구 과정 (=> 2시간+ 추가 작업)**:
1. 로컬에서 같은 train.txt + 같은 retrain 스크립트로 결정적 재학습
2. Counter 기반 알고리즘은 입력 동일하면 출력도 bit-identical
3. 80K vocab 토크나이저 복원

**교훈**:
- 토크나이저는 1.5MB짜리 작은 파일. **학습 시작 직후 gdrive에 즉시 백업**
- 다음 명령을 학습 시작 직후 1회 실행:
  ```bash
  rclone copy data/raw/ko_wikipedia/ko_tokenizer.json gdrive:GrinVi/<run_name>/
  rclone copy data/raw/ko_wikipedia/ko_tokenizer.vocab gdrive:GrinVi/<run_name>/
  ```

### ⚠️ 실수 6 — 토크나이저 학습이 단일 스레드라 너무 느림

**증상**: 로컬 9950X에서 81분 동안 25GB 중 4.75%만 처리. ETA 28시간.

**해결**: `tokenizer_morph.py` 의 `train()` 을 `multiprocessing.Pool`로 병렬화.
- 모듈 레벨 worker init이 각 프로세스에 자체 Kiwi 인스턴스 생성
- Counter 집계는 순서 무관 → bit-identical 결과 보장
- 32 코어로 ~27배 가속 (이미 적용된 패치)

**교훈**: 토크나이저 학습도 시간 소요. 미리 측정하고 병렬화.

### ⚠️ 실수 7 — nohup + python (no -u) 로 stdout 버퍼링 → 진행률 안 보임

**증상**: 81분 동안 retrain.log가 비어있어서 "프로세스 죽은 줄 알았다"는 혼란.

**해결**: 항상 `python -u` 사용 또는 `PYTHONUNBUFFERED=1`. retrain_tokenizer.py 호출 시 `-u` 추가 권장.

**교훈**: 장시간 background 작업은 반드시 unbuffered stdout.

---

## 2. 다음 사이클 정식 파이프라인 (Copy & Paste 가능)

### 2-1. 환경 준비 (1회)

```bash
# 로컬 또는 서버 어디서든
git clone https://github.com/cenoda/GrinVi.git
cd GrinVi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install kiwipiepy  # 한국어 토크나이저
```

### 2-2. 데이터 준비

```bash
# 옵션 A: gdrive에서 기존 데이터 받기
mkdir -p data/processed
rclone copy gdrive:GrinVi/data/processed/train.txt data/processed/ --progress

# 옵션 B: 한국어 위키 다운로드 + 전처리
python scripts/prepare_data.py --dataset ko_wikipedia --out data/raw/
```

### 2-3. 토크나이저 학습 (32 코어 병렬, ~2시간)

```bash
nohup python -u scripts/retrain_tokenizer.py \
  --data data/processed/train.txt \
  --output_prefix data/raw/ko_wikipedia/ko_tokenizer \
  --vocab_size 80000 \
  > tokenizer_train.log 2>&1 &

# 진행 모니터링
tail -f tokenizer_train.log
```

### 2-4. 🚨 토크나이저 즉시 백업 (5초)

```bash
rclone copy data/raw/ko_wikipedia/ko_tokenizer.json gdrive:GrinVi/v3/tokenizer/
rclone copy data/raw/ko_wikipedia/ko_tokenizer.vocab gdrive:GrinVi/v3/tokenizer/
```

### 2-5. UNK 비율 검증

```bash
python scripts/audit_unk.py \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/processed/train.txt --max_lines 20000
# 결과가 "✅ EXCELLENT (< 0.1%)"이면 통과
```

### 2-6. 모델 + 토크나이저 매칭 진단

```bash
python scripts/diagnose.py \
  --checkpoint <ANY_EXISTING_CKPT_OR_SKIP> \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --prompt $'질문: 어디인가요?\n답변:'
```

### 2-7. 10 스텝 smoke test

```bash
python scripts/train.py \
  --preset small \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/processed/train.txt \
  --checkpoint_dir checkpoints_smoke \
  --seq_len 1024 --batch_size 16 --grad_accum 2 \
  --max_steps 10 --eval_interval 100 --save_interval 100
# OOM 없이 정상 종료되어야 함
rm -rf checkpoints_smoke
```

### 2-8. 200 스텝 벤치마크 (실 tok/s 측정)

```bash
python scripts/train.py \
  --preset small \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/processed/train.txt \
  --checkpoint_dir checkpoints_bench \
  --seq_len 1024 --batch_size 16 --grad_accum 2 \
  --max_steps 200 --eval_interval 1000 --save_interval 1000
# 끝의 tok/s 값을 기록
rm -rf checkpoints_bench
```

### 2-9. max_steps 계산

```
target_seconds = wall_clock_budget × 0.92  # 마진 8%
tokens_per_step = batch_size × grad_accum × seq_len  # 16 × 2 × 1024 = 32768
max_steps = target_seconds × tok_s / tokens_per_step
```

B200 small 137M 기준: steady-state ~300K tok/s. 3시간이면 ≈ 100,000 스텝.

### 2-10. 본 학습 + 자동 백업

```bash
# 1) 백업 데몬 (다른 터미널)
nohup bash scripts/backup_checkpoints.sh checkpoints v3_small_80k \
  > backup.log 2>&1 &

# 2) 학습
nohup python -u scripts/train.py \
  --preset small \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/processed/train.txt \
  --checkpoint_dir checkpoints \
  --seq_len 1024 --batch_size 16 --grad_accum 2 \
  --max_steps 100000 \
  --eval_interval 999999 --save_interval 2000 \
  --keep_last_n 3 \
  > training.log 2>&1 &

echo "TRAIN PID: $!"
```

### 2-11. 학습 중 모니터링

```bash
tail -f training.log                           # loss 추이
watch -n 1 nvidia-smi                          # GPU
tail -f backup.log                             # gdrive 업로드
```

### 2-12. 학습 후 추론

```bash
python scripts/generate.py \
  --checkpoint checkpoints/step-final \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --prompt $'질문: 안녕하세요\n답변:' \
  --max_new_tokens 100
```

---

## 3. 핵심 불변식 (이거 어기면 모든 게 깨진다)

| 불변식 | 깨졌을 때 증상 | 검증 방법 |
|---|---|---|
| 모델의 `config.vocab_size` == 토크나이저의 `vocab_size` | 추론이 완전 헛소리 | `diagnose.py` [5] |
| 학습 데이터 UNK 비율 < 1% | 모델이 `<unk>` 자주 생성 | `audit_unk.py` |
| 토크나이저 파일이 정확히 `data/raw/ko_wikipedia/ko_tokenizer.json` | DataLoader 워커 fork 시 로드 실패 | `train.py:117` 하드코딩 확인 |
| 체크포인트 + 토크나이저 둘 다 gdrive 백업 | 서버 죽으면 추론 불가 | `rclone ls gdrive:GrinVi/<run>/` |
| `.gitignore`에 `checkpoints_*/`, `*.safetensors` | git push 영원히 안 끝남 | `git status`에 거대한 파일 안 나오면 OK |

---

## 4. 시간/리소스 가이드 (B200 기준)

| 작업 | 시간 | 비용 (vast.ai $1.5/h 가정) |
|---|---|---|
| 데이터 다운/업로드 (25GB) | 5~50분 (회선) | 회선 비용 |
| 토크나이저 학습 (병렬, 32 코어) | 30~120분 (CPU) | $0 (CPU 작업) |
| 토크나이저 audit | 1분 | ~$0 |
| 10-step smoke test | 1분 | ~$0 |
| 200-step 벤치마크 | 1분 | ~$0 |
| 본 학습 100K 스텝 small | ~3시간 | ~$4.5 |
| 본 학습 100K 스텝 medium | ~5시간 (예상) | ~$7.5 |
| 체크포인트 다운 (gdrive→로컬) | 5~30분 | $0 |

---

## 5. 만든 도구들 (이번 세션 산출물)

| 파일 | 역할 |
|---|---|
| `scripts/diagnose.py` | 6항목 정합성 검증 |
| `scripts/audit_unk.py` | 대규모 텍스트 UNK 비율 측정 |
| `scripts/retrain_tokenizer.py` | 기존 train.txt로 토크나이저만 재학습 (병렬화 포함) |
| `scripts/backup_checkpoints.sh` | 체크포인트 자동 gdrive 백업 (60초 주기, 인자 받음) |
| `grinvi/tokenizer_morph.py` | 캐릭터 폴백 + 멀티프로세싱 학습 |

---

## 6. 다음에 추가하면 좋을 것

- [ ] `train.py:__setstate__`의 토크나이저 경로 하드코딩 제거 (config.json 또는 env var로)
- [ ] 학습 시작 시 자동으로 토크나이저 1회 gdrive 백업하는 훅
- [ ] `generate.py`에 stop sequence + UNK logit masking + 채팅 모드
- [ ] `tokenizer_morph.train()`의 line-level 토크나이즈 모드 (옵션) — 5× 추가 가속 가능, 단 결과 다름
- [ ] perplexity 평가 스크립트 (held-out val.txt)
- [ ] medium preset에 grad_ckpt OFF + batch 키운 권장 설정

---

**한 줄로 요약하자면:** 다음번엔 학습 돌리기 전에 이 문서의 2를 그대로 따라하세요. 그러면 오늘 우리가 8시간 걸려 배운 걸 30분 만에 끝낼 수 있습니다.

---

## 7. 추가 사후분석 (2026-05-18 — 모델 폐기 결정)

### 7.1 결국 일어난 일

- 25GB Q&A `data/processed/train.txt`로 토크나이저 재학습 성공 (UNK 거의 0)
- `verify_tokenizer_match.py`로 모델 ↔ 토크나이저 4/4 통과
- 실제 생성 시 출력: **위키 인명/지명**(`사이판`, `나가노`, `마르틴`, `박제가`, `아틀라스`, `League`, `showing`) 가 잡소리 사이에 박혀 나옴
- → **모델이 instruction following을 학습하지 못함**. 폐기.

### 7.2 진짜 원인 후보

1. **데이터 구성 의심**: `data/processed/train.txt` 25GB가 진짜 Q&A 위주였는지 검증 안 했음. audit_unk.py는 첫 줄 몇 개만 봤는데 그것만 Q&A 포맷이고 본문은 위키 덤프였을 가능성 큼.
2. **모델 capacity vs 데이터 비율**: 137M에 3.28B 토큰 학습은 데이터 양은 좋았지만, Q&A 패턴이 전체 데이터의 1~5%였다면 모델이 그 패턴을 학습할 incentive가 부족했을 수 있음.
3. **LR 스케줄**: cosine이 3e-4 → 3e-5에서 끝남 (10배 감쇠). 보통 100배(1/100)까지 감쇠하는 게 정석. 최종 단계 fine-grained 학습이 부족했을 수 있음.

### 7.3 `verify_tokenizer_match.py`의 한계

- 측정한 것: "model logits이 한국어 형태소 토큰을 우선시하는가" (한국어 분포 sanity check)
- **측정 못한 것**: "ID 매핑이 학습 시점과 정확히 같은가" (실제 정답성)
- 결과: top-3가 `의/JKG`, `는/JX` 같은 흔한 조사면 거의 무조건 통과 → **false positive 잘 나옴**
- → 토크나이저 매칭 검증으로는 부적합. **차라리 학습 데이터 한 줄을 인코드→디코드 라운드트립으로 비교하는 게 직접적 증거**.

### 7.4 다음 사이클에 반드시 할 것

- [ ] **학습 시작 전에 `data/processed/train.txt`를 직접 head/tail/shuf 로 50줄 이상 육안 검증.** 진짜 Q&A인지 확인.
- [ ] 데이터 통계 자동화: `wc -l`, "질문:" 시작 줄 비율, 위키 패턴(`'''`, `括弧`, 영어 비율) 측정.
- [ ] 토크나이저는 **학습 시작 시점에 즉시 gdrive 백업**. 모델 끝난 뒤 백업하면 늦음.
- [ ] verify는 "top-1 한국어 분포" 대신 **"learned embedding에서 ID 100~200 같은 흔한 토큰의 cosine similarity가 wiki 토크나이저와 비교했을 때 의미 있게 다른가"** 같은 더 강한 측정으로 교체.
- [ ] 작은 데이터로 sanity 학습 (예: 1000개 Q&A 100스텝) → 출력이 Q&A 패턴이라도 흉내 내는지 확인 → 그 다음에야 대규모 학습.
- [ ] **돈 들이기 전에 데이터를 의심하기.** 모델/코드/토크나이저보다 데이터가 무너졌을 확률이 항상 더 높다.

### 7.5 한 줄 결론

> "vocab이 맞다 ≠ 토크나이저가 맞다" 그리고 "토크나이저가 맞다 ≠ 모델이 답할 수 있다". 다음번엔 데이터부터 의심하자.
