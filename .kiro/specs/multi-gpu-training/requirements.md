# Requirements Document

## Introduction

GrinVi는 홈메이드 한국어 LLM 프로젝트로, 현재 `grinvi/trainer.py`가 단일 GPU 학습만 지원한다.
이 기능은 `torch.distributed` + `DistributedDataParallel(DDP)` 방식으로 멀티 GPU 학습을 지원하여,
vast.ai 등 클라우드 환경에서 2개 이상의 GPU를 활용해 학습 처리량(throughput)을 선형적으로 향상시키는 것을 목표로 한다.
진입점은 `torchrun`이며, 기존 단일 GPU 학습 경로(`python scripts/train.py`)는 그대로 유지된다.

## Glossary

- **DDP**: `torch.nn.parallel.DistributedDataParallel` — 각 GPU가 모델 전체를 보유하고 gradient를 동기화하는 데이터 병렬 학습 방식
- **Trainer**: `grinvi/trainer.py`의 `Trainer` 클래스 — 학습 루프를 담당
- **TrainerConfig**: `grinvi/trainer.py`의 `TrainerConfig` 클래스 — 학습 하이퍼파라미터 설정
- **LaunchScript**: `scripts/train.py` — 학습 실행 진입점 스크립트
- **Rank**: DDP 프로세스 식별자. `rank=0`이 마스터 프로세스(로깅·체크포인트 담당)
- **World Size**: 전체 DDP 프로세스(GPU) 수
- **Local Rank**: 단일 노드 내 GPU 인덱스 (`CUDA_VISIBLE_DEVICES` 기준)
- **DistributedSampler**: 각 GPU가 겹치지 않는 데이터 샤드를 처리하도록 보장하는 PyTorch 샘플러
- **Global Batch Size**: `batch_size × world_size × gradient_accumulation_steps` — 실제 유효 배치 크기
- **Checkpoint**: 모델 가중치와 학습 상태를 저장한 파일 (`model.safetensors`, `config.json`)
- **torchrun**: PyTorch 공식 DDP 런처 (`python -m torch.distributed.run`)

## Requirements

### Requirement 1: DDP 프로세스 그룹 초기화

**User Story:** 개발자로서, `torchrun`으로 학습을 실행했을 때 DDP 프로세스 그룹이 자동으로 초기화되기를 원한다. 그래야 멀티 GPU 학습이 올바르게 시작된다.

#### Acceptance Criteria

1. WHEN `torchrun`이 환경 변수 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`를 설정한 상태로 프로세스를 시작할 때, THE Trainer SHALL `torch.distributed.init_process_group(backend="nccl")`을 호출하여 프로세스 그룹을 초기화한다.
2. WHEN `RANK` 환경 변수가 설정되지 않은 상태로 실행될 때, THE Trainer SHALL 단일 GPU 모드로 동작하며 `torch.distributed` 초기화를 건너뛴다.
3. WHEN DDP 초기화가 완료된 후, THE Trainer SHALL 각 프로세스를 `LOCAL_RANK`에 해당하는 CUDA 디바이스에 바인딩한다.
4. WHEN 학습이 종료될 때, THE Trainer SHALL `torch.distributed.destroy_process_group()`을 호출하여 프로세스 그룹을 정리한다.

---

### Requirement 2: 모델 DDP 래핑

**User Story:** 개발자로서, 모델이 DDP로 래핑되어 gradient가 GPU 간에 자동으로 동기화되기를 원한다. 그래야 모든 GPU가 동일한 가중치로 수렴한다.

#### Acceptance Criteria

1. WHEN DDP 모드가 활성화된 상태에서 Trainer가 초기화될 때, THE Trainer SHALL `GrinViModel`을 `torch.nn.parallel.DistributedDataParallel`로 래핑한다.
2. WHEN DDP 래핑이 적용된 후, THE Trainer SHALL `model.module`을 통해 원본 `GrinViModel`에 접근하여 체크포인트 저장 및 파라미터 수 계산에 사용한다.
3. WHEN `torch.compile`이 활성화된 경우, THE Trainer SHALL DDP 래핑 이후에 `torch.compile`을 적용한다.
4. WHEN gradient checkpointing이 활성화된 경우, THE Trainer SHALL DDP 래핑 이전에 `enable_gradient_checkpointing()`을 호출한다.

---

### Requirement 3: 데이터 분산 처리

**User Story:** 개발자로서, 각 GPU가 겹치지 않는 데이터 샤드를 처리하기를 원한다. 그래야 동일한 데이터를 중복 학습하지 않고 전체 데이터셋을 효율적으로 활용한다.

#### Acceptance Criteria

1. WHEN DDP 모드가 활성화된 상태에서 DataLoader가 생성될 때, THE LaunchScript SHALL `torch.utils.data.distributed.DistributedSampler`를 사용하여 각 rank에 고유한 데이터 샤드를 할당한다.
2. WHEN `DistributedSampler`가 사용될 때, THE LaunchScript SHALL DataLoader의 `shuffle=True` 옵션을 제거하고 `DistributedSampler(shuffle=True)`로 대체한다.
3. WHEN eval interval에 도달할 때마다, THE Trainer SHALL `DistributedSampler.set_epoch(global_step // steps_per_epoch)`을 호출하여 셔플 순서를 갱신한다. (현재 Trainer가 epoch 단위가 아닌 step 기반으로 동작하므로 `set_epoch`의 인자는 `global_step // steps_per_epoch`으로 근사한다.)
4. WHEN 단일 GPU 모드로 실행될 때, THE LaunchScript SHALL 기존 `shuffle=True` DataLoader 동작을 변경하지 않는다.

---

### Requirement 4: 유효 배치 크기 및 학습률 스케일링

**User Story:** 개발자로서, 멀티 GPU 사용 시 유효 배치 크기가 올바르게 계산되고 학습률 스케일링 방식을 선택할 수 있기를 원한다. 그래야 실험 목적에 맞게 학습 안정성을 조정할 수 있다.

#### Acceptance Criteria

1. THE TrainerConfig SHALL `world_size` 필드를 포함하며, DDP 모드에서는 실제 `WORLD_SIZE` 값으로, 단일 GPU 모드에서는 `1`로 설정된다.
2. THE TrainerConfig SHALL `scale_lr` 필드를 포함하며, 허용 값은 `"linear"`, `"sqrt"`, `"none"`이고 기본값은 `"none"`이다.
3. WHEN `scale_lr`이 `"linear"`로 설정된 경우, THE Trainer SHALL 학습률을 `learning_rate × world_size`로 스케일링한다.
4. WHEN `scale_lr`이 `"sqrt"`로 설정된 경우, THE Trainer SHALL 학습률을 `learning_rate × sqrt(world_size)`로 스케일링한다.
5. WHEN `scale_lr`이 `"none"`으로 설정된 경우, THE Trainer SHALL 학습률을 변경하지 않고 `learning_rate` 값을 그대로 사용한다.
6. THE Trainer SHALL 로그에 `global_batch_size = batch_size × world_size × gradient_accumulation_steps` 값과 적용된 `scale_lr` 모드를 출력한다.
7. WHEN `world_size`가 변경될 때, THE Trainer SHALL `warmup_steps`를 조정하지 않는다 (warmup은 global step 기준으로 유지).

---

### Requirement 5: 체크포인트 저장 (Rank 0 전용)

**User Story:** 개발자로서, 체크포인트가 rank 0 프로세스에서만 저장되기를 원한다. 그래야 여러 GPU가 동시에 같은 파일에 쓰는 충돌이 발생하지 않는다.

#### Acceptance Criteria

1. WHEN 체크포인트 저장 조건이 충족될 때, THE Trainer SHALL `rank == 0`인 프로세스에서만 `_save()` 및 `_save_best()`를 실행한다.
2. WHEN rank 0이 체크포인트를 저장할 때, THE Trainer SHALL `model.module.save_pretrained()`를 호출하여 DDP 래퍼를 제거한 원본 모델 가중치를 저장한다.
3. WHEN 단일 GPU 모드로 실행될 때, THE Trainer SHALL 기존 `model.save_pretrained()` 호출 방식을 유지한다.
4. WHEN 체크포인트 저장이 완료된 후, THE Trainer SHALL 모든 프로세스가 `torch.distributed.barrier()`를 통해 동기화된 후 학습을 재개한다.

---

### Requirement 6: 로깅 및 진행 상황 출력 (Rank 0 전용)

**User Story:** 개발자로서, 학습 로그와 진행 상황이 rank 0에서만 출력되기를 원한다. 그래야 여러 GPU의 중복 출력으로 터미널이 혼잡해지지 않는다.

#### Acceptance Criteria

1. WHEN 학습 중 로그 메시지가 생성될 때, THE Trainer SHALL `rank == 0`인 프로세스에서만 `console.print()` 및 `print()` 출력을 실행한다.
2. WHEN `rank != 0`인 프로세스에서 실행될 때, THE Trainer SHALL Rich Progress 바를 생성하지 않는다.
3. WHEN eval 루프가 실행될 때, THE Trainer SHALL 모든 rank에서 eval loss를 계산하고, `torch.distributed.all_reduce`로 평균을 집계한 후 rank 0에서만 결과를 출력한다.
4. WHEN 학습 시작 시, THE Trainer SHALL rank 0에서 `world_size`, 각 GPU 모델명, 총 파라미터 수를 포함한 DDP 설정 요약을 출력한다.

---

### Requirement 7: torchrun 실행 인터페이스

**User Story:** 개발자로서, 기존 `python scripts/train.py` 명령어를 최소한으로 변경하여 멀티 GPU 학습을 시작할 수 있기를 원한다. 그래야 학습 실행 방법을 새로 익히는 부담이 적다.

#### Acceptance Criteria

1. THE LaunchScript SHALL `torchrun --nproc_per_node=<N> scripts/train.py [기존 인수]` 형태로 실행될 때 DDP 모드로 동작한다.
2. THE LaunchScript SHALL `python scripts/train.py [기존 인수]` 형태로 실행될 때 단일 GPU 모드로 동작하며 기존 동작을 완전히 유지한다.
3. THE LaunchScript SHALL `--nproc_per_node` 인수를 별도로 추가하지 않으며, `torchrun`이 설정하는 환경 변수(`RANK`, `WORLD_SIZE`, `LOCAL_RANK`)를 통해 DDP 여부를 자동 감지한다.
4. WHEN `--device` 인수가 명시적으로 지정된 경우, THE LaunchScript SHALL DDP 모드에서 해당 인수를 무시하고 `LOCAL_RANK` 기반 디바이스를 사용한다.

---

### Requirement 8: 체크포인트 재개(Resume) DDP 호환성

**User Story:** 개발자로서, 기존 단일 GPU 체크포인트에서 멀티 GPU 학습을 재개할 수 있기를 원한다. 그래야 이미 학습된 모델을 멀티 GPU로 계속 학습할 수 있다.

#### Acceptance Criteria

1. WHEN `--resume` 인수로 체크포인트 경로가 지정된 경우, THE LaunchScript SHALL 모든 rank에서 동일한 체크포인트를 로드한다.
2. WHEN 체크포인트를 로드한 후 DDP 래핑이 적용될 때, THE Trainer SHALL 모델 가중치가 모든 GPU에 동일하게 복제되어 있음을 보장한다.
3. WHEN 단일 GPU로 저장된 체크포인트를 멀티 GPU 학습에서 로드할 때, THE LaunchScript SHALL 추가 변환 없이 `GrinViModel.from_pretrained()`로 로드할 수 있어야 한다.

---

### Requirement 9: 에러 처리 및 안정성

**User Story:** 개발자로서, DDP 학습 중 한 GPU에서 에러가 발생했을 때 전체 학습이 안전하게 종료되기를 원한다. 그래야 불완전한 상태로 학습이 계속되는 것을 방지한다.

#### Acceptance Criteria

1. WHEN 임의의 rank에서 `RuntimeError`(예: CUDA OOM)가 발생할 때, THE Trainer SHALL 해당 에러를 로그에 기록하고 `torch.distributed.destroy_process_group(timeout=30초)`을 호출한 후 프로세스를 종료한다. (timeout을 명시하여 `destroy_process_group()` 블로킹 가능성을 방지한다.)
2. WHEN 에러가 발생할 때, THE Trainer SHALL 즉시 학습을 중단하고 안전하게 종료한다. (연속 에러 횟수 카운팅 없이 단순 에러 발생 시 안전 종료)
3. IF DDP 초기화가 실패할 때, THEN THE Trainer SHALL 명확한 에러 메시지와 함께 `sys.exit(1)`을 호출한다.
4. WHEN 학습이 정상 종료되거나 에러로 종료될 때, THE Trainer SHALL `finally` 블록에서 `destroy_process_group()`이 항상 호출되도록 보장한다.

---

### Requirement 10: Gradient Accumulation과 DDP no_sync() 최적화

**User Story:** 개발자로서, gradient accumulation 중간 스텝에서 불필요한 gradient 동기화가 발생하지 않기를 원한다. 그래야 DDP 통신 오버헤드를 줄이고 학습 처리량을 높일 수 있다.

#### Acceptance Criteria

1. WHEN DDP 모드가 활성화된 상태에서 gradient accumulation이 진행될 때, THE Trainer SHALL accumulation의 마지막 스텝을 제외한 중간 스텝에서 `model.no_sync()` 컨텍스트 매니저를 사용하여 gradient 동기화를 억제한다.
2. WHEN accumulation의 마지막 스텝(즉, `(step + 1) % gradient_accumulation_steps == 0`)에 도달할 때, THE Trainer SHALL `no_sync()` 없이 정상적으로 `backward()`를 호출하여 모든 rank 간 gradient를 동기화한다.
3. WHEN 단일 GPU 모드로 실행될 때, THE Trainer SHALL `no_sync()` 로직을 적용하지 않고 기존 gradient accumulation 동작을 유지한다.

---

### Requirement 11: Mixed Precision(AMP)과 DDP 호환성

**User Story:** 개발자로서, mixed precision 설정이 DDP 환경에서 올바르게 동작하기를 원한다. 그래야 각 rank의 수치 안정성이 보장된다.

#### Acceptance Criteria

1. WHEN `dtype`이 `"bfloat16"`으로 설정된 경우, THE Trainer SHALL `GradScaler`를 사용하지 않는다. (`bfloat16`은 동적 loss scaling이 불필요하므로 `GradScaler`는 해당 없음)
2. WHEN `dtype`이 `"float16"`으로 설정된 경우, THE Trainer SHALL 각 rank에서 독립적인 `GradScaler` 인스턴스를 생성하여 사용한다.
3. THE Trainer SHALL `GradScaler` 상태를 체크포인트에 저장하지 않는다. (재개 시 scaler는 초기 상태에서 재시작하며, 이는 의도된 동작이다.)
4. WHEN DDP 모드에서 `float16` + `GradScaler`를 사용할 때, THE Trainer SHALL `scaler.unscale_()` 및 `scaler.step()` 호출이 각 rank에서 독립적으로 수행됨을 보장한다.
