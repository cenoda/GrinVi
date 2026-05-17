# Training Runbook / Gotchas

실수로 학습 시간을 날리지 않기 위한 체크리스트와, 이번 세션에서 실제로 터졌던 문제들을 정리한 문서입니다.

## 1. 큰 학습 돌리기 전에 꼭 할 일

### 1-1. 사전 점검 실행

```bash
python scripts/preflight_train.py \
  --data data/processed/train.txt \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json
```

재개(run resume)라면 체크포인트도 같이 넣습니다.

```bash
python scripts/preflight_train.py \
  --data data/processed/train.txt \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --resume checkpoints_medium/step-18000
```

### 1-2. 30분 smoke test 먼저

```bash
timeout --signal=INT 30m python -u scripts/train.py \
  --preset small \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/processed/train.txt \
  --seq_len 512 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_steps 100000 \
  --eval_interval 1000 \
  --save_interval 200 \
  --lr 3e-4 \
  --dtype bfloat16 \
  --grad_ckpt \
  --checkpoint_dir checkpoints_smoke \
  --device cuda
```

### 1-3. 실험마다 체크포인트 디렉토리 분리

절대 기존 실험과 새 실험을 같은 디렉토리에 섞지 마세요.

- 좋은 예: `checkpoints_medium_v2`, `checkpoints_local_small_30m_fixed`
- 나쁜 예: 예전 실험 위에 그대로 덮어쓰기

---

## 2. 실제로 터졌던 문제들

| 문제 | 증상 | 원인 | 대응 |
|---|---|---|---|
| DDP에서 데이터 중복 | loss가 이상하게 빨리 떨어지거나 흔들림 | `IterableDataset`가 rank별 shard를 안 나눔 | `scripts/train.py`에서 rank/worker별 byte chunk 분할 |
| 파일 뒷부분 과대표집 | 일부 데이터만 보는 것 같은 느낌, tail bias | worker마다 `start_offset`만 있고 `end_offset`이 없어서 EOF까지 읽음 | 각 worker가 자기 chunk 구간만 읽도록 수정 |
| resume 후 loss 폭등 | 재개 직후 loss가 8~9대로 튐 | 모델 가중치만 로드하고 optimizer/scheduler 상태는 안 로드함 | `trainer_state.pt` 저장/복원 추가 |
| 챗 포맷 붕괴 | 앵무새처럼 따라함, `<unk>` 과다 | 학습 데이터가 `<usr>`, `<bot>`인데 형태소 토크나이저가 이를 안정적으로 표현 못함 | `질문:` / `답변:`으로 치환 |
| 출력이 실제보다 멍청해 보임 | `사랑하어`, `누구이야` 같은 결과 | 형태소 디코드가 단순 문자열 이어붙이기 | `kiwi.join()` 기반 디코드로 수정 |
| 과적합인지 판단 불가 | loss가 빨리 떨어져서 불안함 | `eval_data` 없이 train loss만 봄 | holdout eval 파일 분리 권장 |

---

## 3. 재개(resume) 관련 규칙

### 안전한 resume 조건

아래 세 파일이 모두 있으면 가장 안전합니다.

- `config.json`
- `model.safetensors`
- `trainer_state.pt`

`trainer_state.pt`가 없으면, 그건 **진짜 resume가 아니라 가중치 warm-start**입니다.

### 확인 예시

```bash
ls -la checkpoints_medium/step-18000
```

`trainer_state.pt`가 없으면:
- optimizer momentum 없음
- scheduler step 없음
- warmup 위치 복구 안 됨
- loss가 다시 튈 수 있음

---

## 4. 데이터 포맷 규칙

### 권장 포맷

```text
질문: 안녕?
답변: 안녕하세요.
```

### 피해야 할 포맷

```text
<usr> 안녕?
<bot> 안녕하세요.
```

형태소 토크나이저 기반 한국어 실험에서는 `질문:` / `답변:`처럼 토크나이저가 안정적으로 표현 가능한 문자열을 쓰는 편이 안전합니다.

---

## 5. 대형 학습 전 최종 체크리스트

- [ ] `scripts/preflight_train.py` 통과
- [ ] 체크포인트 디렉토리 새로 분리
- [ ] `resume` 시 `trainer_state.pt` 존재 확인
- [ ] 30분 smoke test 먼저 성공
- [ ] 로그/백업 경로 확인
- [ ] 가능하면 `eval_data` 준비

---

## 6. 지금 프로젝트에서 특히 기억할 것

1. **초기 loss 급락 자체는 곧바로 과적합이 아니다.**
   - vocab 64k면 랜덤 CE가 약 `11.07`이라 초반에는 `11 -> 7 -> 5`가 꽤 빨리 나올 수 있음.

2. **데이터 전체를 공정하게 보는지 먼저 의심하자.**
   - `IterableDataset`는 `DistributedSampler`가 자동으로 해결해 주지 않음.
   - rank/worker별로 직접 shard를 나눠야 함.

3. **로그가 멀쩡해 보여도 잘못 학습될 수 있다.**
   - 그래서 본런 전에 smoke test + generation check가 필요함.

4. **결과가 이상하면 모델만 의심하지 말고 디코드/포맷도 의심하자.**
   - 출력이 이상해 보여도 실제 학습은 더 잘되어 있을 수 있음.

---

## 7. 권장 운영 순서

```text
preflight -> 30분 smoke test -> checkpoint 생성 확인 -> generation 확인 -> 본런
```

이 순서를 지키면 "몇 시간 태웠는데 설정이 틀렸네"를 크게 줄일 수 있습니다.

