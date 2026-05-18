# Vast.ai Training Guide & Checklist

Vast.ai와 같은 원격 서버 환경에서 GrinVi 모델을 학습할 때, 실수를 줄이고 안정적으로 완료하기 위한 가이드라인입니다. 이 문서는 에이전트(AI)와 유저(사람)가 각각 확인해야 할 사항을 단계별로 정의합니다.

---

## 1. 초기 환경 설정 (JupyterLab 진입 직후)

주피터랩을 열자마자 가장 먼저 수행해야 할 작업들입니다.

### 👤 유저 (User)
- [ ] **GPU 할당량 확인**: `nvidia-smi`를 실행하여 의도한 GPU 모델과 개수가 맞는지 확인합니다.
- [ ] **RCLONE 토큰 준비**: 데이터 및 체크포인트 백업을 위해 Google Drive rclone 토큰을 에이전트에게 전달합니다.
  - 로컬에서 `rclone config show gdrive` (혹은 설정한 이름) 명령어로 토큰 부분을 복사해둡니다.

### 🤖 에이전트 (Agent)
- [ ] **의존성 설치**: `pip install -r requirements.txt`를 실행합니다.
- [ ] **rclone 설정**: 유저가 제공한 토큰을 사용하여 `~/.config/rclone/rclone.conf`를 생성합니다.
- [ ] **데이터 동기화**: `scripts/infra/pull_gdrive_data.sh` 또는 직접 `rclone copy`를 사용하여 학습 데이터를 서버로 가져옵니다.
- [ ] **디스크 공간 확인**: `df -h`를 실행하여 최소 20GB 이상의 여유 공간이 있는지 확인합니다.

---

## 2. 학습 파이프라인 시작 (Step-by-Step)

통합 파이프라인(`scripts/training/train_pipeline.py`)을 사용하는 것을 강력히 권장합니다.

### 🤖 에이전트 (Agent)
- [ ] **Preflight 실행**: 데이터셋 무결성, 토크나이저 로드 여부를 먼저 검증합니다.
- [ ] **Smoke Test (50 steps)**: 짧은 학습을 수행하여 GPU 메모리(OOM) 여부와 체크포인트 저장 기능을 확인합니다.
- [ ] **Inference Check**: Smoke test로 만들어진 체크포인트를 로드하여 문장 생성이 시작되는지(예: 횡설수설하더라도 형식이 맞는지) 확인합니다.
- [ ] **유저 승인 대기**: Smoke test 결과와 샘플 출력을 유저에게 보고하고, 본 학습 시작 여부를 묻습니다.

### 👤 유저 (User)
- [ ] **샘플 출력 확인**: 에이전트가 보고한 Inference 결과를 보고, 토크나이저나 포맷팅에 문제가 없는지 최종 확인합니다.
- [ ] **본 학습 시작 승인**: 문제가 없다면 에이전트에게 학습 시작을 지시합니다.

---

## 3. 본 학습 모니터링 및 백업

### 🤖 에이전트 (Agent)
- [ ] **백업 프로세스 확인**: `backup_checkpoints.sh`가 배경에서 잘 돌고 있는지, `backup.log`에 에러가 없는지 주기적으로 확인합니다.
- [ ] **Overstacking 감지**: `pgrep -f train.py`를 통해 중복 프로세스가 실행 중인지 감지합니다.
- [ ] **Loss 모니터링**: Loss가 `nan`이 되거나 비정상적으로 튀지 않는지 확인합니다.

### 👤 유저 (User)
- [ ] **인스턴스 상태 확인**: Vast.ai 대시보드에서 인스턴스가 'Running' 상태인지, 혹은 'Overbid'의 위험이 없는지 확인합니다.

---

## 4. 학습 완료 및 사후 처리

### 🤖 에이전트 (Agent)
- [ ] **최종 백업**: 학습 종료 후 `step-final`이 Google Drive에 완전히 업로드되었는지 확인합니다.
- [ ] **프로세스 정리**: 모든 하위 프로세스를 종료하고 GPU 메모리를 해제합니다.
- [ ] **학습 요약 보고**: 최종 step, 최종 loss, 총 학습 시간, 저장된 경로를 유저에게 보고합니다.

### 👤 유저 (User)
- [ ] **인스턴스 삭제**: 백업이 완료된 것을 확인했다면 Vast.ai 인스턴스를 삭제하여 비용 발생을 중단합니다.

---

## 5. 예상되는 오류 및 대응 (Troubleshooting)

| 현상 | 원인 | 대응 방법 |
| :--- | :--- | :--- |
| **Out of Memory (OOM)** | 배치 사이즈가 너무 크거나 중복 프로세스 실행 | `batch_size` 축소 또는 `pkill -9 -f train.py`로 정리 |
| **Connection Timeout** | rclone 백업 중 네트워크 불안정 | `train_pipeline.py`가 자동으로 재시도하므로 대기 |
| **Disk Full** | 체크포인트가 너무 많이 쌓임 | `keep_last_n` 옵션을 사용하여 오래된 체크포인트 자동 삭제 |
| **Tokenizer Discord** | 학습 시와 다른 토크나이저 사용 | 체크포인트 폴더 내의 `tokenizer.json`을 사용하도록 설정 확인 |
| **Loss is NaN** | 학습률이 너무 높거나 데이터 오염 | `lr`을 낮추거나 데이터셋 사전 점검 재실행 |

---

## 6. 편리한 시작을 위한 자동화 스크립트 (start_vast.sh)

인스턴스 생성 시 On-start script로 입력하거나, 주피터랩 터미널에서 즉시 실행하여 환경 구축부터 학습 시작까지 자동화할 수 있습니다.

```bash
# 환경변수 설정 (rclone 토큰)
export RCLONE_GDRIVE_TOKEN='YOUR_TOKEN_HERE'

# 스크립트 실행
bash scripts/infra/start_vast.sh
```

`start_vast.sh`는 내부적으로 `train_pipeline.py`를 호출하여 안전 점검, Smoke test, 본 학습, 자동 백업을 모두 수행합니다.

---

## 에이전트를 위한 명령어 템플릿 (Vast.ai 전용)

```bash
# 1. 의존성 및 설정
pip install -r requirements.txt
mkdir -p ~/.config/rclone

# 2. 파이프라인 실행 (백업 포함)
python scripts/training/train_pipeline.py \
  --data data/processed/train.txt \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --checkpoint_dir checkpoints/v1_run \
  --backup_name v1_run_backup \
  --preset medium \
  --gpus 1
```
