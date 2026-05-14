# Implementation Plan: Multi-GPU Training (DDP)

## Overview

`grinvi/trainer.py`와 `scripts/train.py`에 `torch.nn.parallel.DistributedDataParallel(DDP)` 기반 멀티 GPU 학습을 추가한다.
`torchrun`으로 실행 시 DDP 모드가 자동 활성화되며, 기존 `python scripts/train.py` 단일 GPU 경로는 변경 없이 유지된다.

## Tasks

- [ ] 1. TrainerConfig에 DDP 관련 필드 추가
  - `grinvi/trainer.py`의 `TrainerConfig`에 `world_size: int = 1`, `scale_lr: str = "none"` 필드를 추가한다.
  - `scale_lr` 허용 값: `"linear"`, `"sqrt"`, `"none"`
  - _Requirements: 4.1, 4.2_

- [ ] 2. `_compute_effective_lr` 함수 구현 및 검증
  - [ ] 2.1 `_compute_effective_lr(base_lr, world_size, scale_lr)` 순수 함수를 `grinvi/trainer.py`에 구현한다.
    - `"linear"` → `base_lr * world_size`
    - `"sqrt"` → `base_lr * math.sqrt(world_size)`
    - `"none"` → `base_lr` 그대로 반환
    - _Requirements: 4.3, 4.4, 4.5_

  - [ ]* 2.2 Property 1: LR 스케일링 수식 정확성 및 불변성 속성 테스트 작성
    - **Property 1: LR scaling formula correctness and world_size=1 invariance**
    - `hypothesis`로 `lr`, `world_size`, `mode`를 무작위 생성하여 수식 정확성 검증
    - `world_size=1`일 때 모든 모드에서 `lr` 불변 검증
    - **Validates: Requirements 4.3, 4.4, 4.5**

  - [ ]* 2.3 `_compute_effective_lr` 단위 테스트 작성
    - `test_lr_scaling_linear`, `test_lr_scaling_sqrt`, `test_lr_scaling_none`, `test_lr_scaling_world_size_1` 케이스
    - `test_trainer_config_defaults`로 기본값 확인
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 3. `_get_raw_model` 헬퍼 함수 구현 및 검증
  - [ ] 3.1 `_get_raw_model(model, is_ddp)` 함수를 `grinvi/trainer.py`에 구현한다.
    - `is_ddp=True`이면 `model.module` 반환, `False`이면 `model` 반환
    - 반환 타입은 항상 `GrinViModel`
    - _Requirements: 2.2, 5.2, 5.3_

  - [ ]* 3.2 Property 3: `_get_raw_model` 정확성 속성 테스트 작성
    - **Property 3: _get_raw_model returns correct object**
    - `is_ddp` 불리언을 무작위로 생성하여 반환 객체가 항상 원본 `GrinViModel`임을 검증
    - **Validates: Requirements 2.2, 5.2, 5.3**

  - [ ]* 3.3 `_get_raw_model` 단위 테스트 작성
    - `test_get_raw_model_ddp`, `test_get_raw_model_single` 케이스
    - _Requirements: 2.2, 5.2, 5.3_

- [ ] 4. `Trainer.__init__`에 DDP 초기화 로직 추가
  - [ ] 4.1 `_setup_distributed()` 메서드를 구현한다.
    - `RANK` 환경변수 존재 여부로 DDP 모드 자동 감지
    - DDP 모드: `dist.init_process_group(backend="nccl")` 호출, `rank`, `local_rank` 반환
    - 단일 GPU 모드: 초기화 건너뜀, `(False, 0, 0)` 반환
    - DDP 초기화 실패 시 `sys.exit(1)` 호출
    - _Requirements: 1.1, 1.2, 1.3, 9.3_

  - [ ] 4.2 `Trainer.__init__`에서 모델 래핑 순서를 구현한다.
    - 순서: `model.to(device)` → `enable_gradient_checkpointing()` (선택) → `DDP(model, device_ids=[local_rank])` → `torch.compile()` (선택)
    - `is_ddp`, `rank`, `local_rank` 인스턴스 변수 설정
    - _Requirements: 1.3, 2.1, 2.3, 2.4_

  - [ ] 4.3 `_compute_effective_lr`를 `Trainer.__init__`에 연결하여 optimizer 생성 시 스케일링된 LR을 적용한다.
    - 시작 시 rank 0에서 `world_size`, GPU 모델명, 총 파라미터 수, `scale_lr` 모드, `global_batch_size`를 출력한다.
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 6.4_

- [ ] 5. 학습 루프에 `no_sync()` 최적화 및 로깅 제어 적용
  - [ ] 5.1 gradient accumulation 중간 스텝에 `model.no_sync()` 컨텍스트 매니저를 적용한다.
    - 마지막 accumulation 스텝(`(step + 1) % grad_accum == 0`)에서만 정상 backward 수행
    - 단일 GPU 모드에서는 `contextlib.nullcontext()` 사용
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 5.2 로그 출력을 rank 0 전용으로 제한한다.
    - `console.print()`, `print()` 호출을 `if self.rank == 0:` 조건으로 감싼다.
    - `rank != 0`에서는 Rich Progress 바를 생성하지 않는다.
    - _Requirements: 6.1, 6.2_

- [ ] 6. eval 루프에 `all_reduce` 집계 추가
  - `_eval()` 메서드에서 모든 rank의 eval loss를 `dist.all_reduce(op=ReduceOp.AVG)`로 집계한다.
  - 결과 출력은 rank 0에서만 수행한다.
  - 단일 GPU 모드에서는 기존 동작 유지.
  - _Requirements: 6.3_

- [ ] 7. 체크포인트 저장을 rank 0 전용으로 수정
  - `_save()` 및 `_save_best()` 메서드에 `if self.rank != 0: return` 가드를 추가한다.
  - DDP 모드에서 `_get_raw_model().save_pretrained()`를 호출하여 DDP 래퍼 없이 저장한다.
  - 저장 완료 후 `dist.barrier()`로 모든 rank를 동기화한다.
  - 단일 GPU 모드에서는 기존 `model.save_pretrained()` 호출 유지.
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 8. Checkpoint round-trip 검증
  - [ ]* 8.1 Property 4: 체크포인트 save/load round-trip 속성 테스트 작성
    - **Property 4: checkpoint save/load round-trip**
    - `tie_word_embeddings` 불리언을 무작위로 생성하여 `save_pretrained` → `from_pretrained` 후 모든 파라미터 텐서가 수치적으로 동일함을 검증
    - **Validates: Requirements 5.2, 5.3, 8.3**

  - [ ]* 8.2 체크포인트 단위 테스트 작성
    - `test_checkpoint_roundtrip`: 단일 GPU 저장 후 로드 파라미터 동등성 확인
    - _Requirements: 5.2, 5.3, 8.3_

- [ ] 9. 에러 처리 및 `destroy_process_group` 정리 로직 구현
  - `Trainer.train()`에 `try/except RuntimeError/finally` 블록을 추가한다.
  - `finally` 블록에서 `is_ddp`일 때 `dist.destroy_process_group()` 항상 호출.
  - `RuntimeError` 발생 시 rank 0에서 에러 로그 출력 후 예외를 상위로 전파.
  - _Requirements: 1.4, 9.1, 9.2, 9.4_

- [ ] 10. Checkpoint
  - 여기까지 구현된 `grinvi/trainer.py` 변경사항에 대해 모든 테스트가 통과하는지 확인한다.
  - `pytest tests/test_multi_gpu_training.py -v`를 실행하여 단위 테스트 및 속성 기반 테스트 통과 확인.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. `scripts/train.py` DDP 지원 추가
  - [ ] 11.1 `RANK` 환경변수로 DDP 여부를 감지하고 `device`, `world_size`를 결정하는 로직을 추가한다.
    - DDP 모드: `device = f"cuda:{local_rank}"`, `--device` 인수 무시
    - 단일 GPU 모드: 기존 `args.device` 또는 자동 감지 유지
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 11.2 DDP 모드에서 `DistributedSampler`를 사용하는 DataLoader를 생성한다.
    - `DistributedSampler(train_ds, shuffle=True)` 사용, DataLoader의 `shuffle=False`로 설정
    - `drop_last=True`, `pin_memory=True`, `persistent_workers=True` 적용
    - 단일 GPU 모드에서는 기존 `shuffle=True` DataLoader 유지
    - _Requirements: 3.1, 3.2, 3.4_

  - [ ] 11.3 eval interval마다 `DistributedSampler.set_epoch(global_step // steps_per_epoch)`을 호출하는 로직을 추가한다.
    - _Requirements: 3.3_

  - [ ] 11.4 `--scale_lr {linear,sqrt,none}` CLI 인수를 추가하고 `TrainerConfig`에 전달한다.
    - `TrainerConfig(world_size=world_size, scale_lr=args.scale_lr)` 생성
    - _Requirements: 4.1, 4.2, 7.1, 7.2_

- [ ] 12. Mixed Precision(AMP)과 DDP 호환성 확인
  - `dtype="bfloat16"` 시 `GradScaler`를 사용하지 않는 기존 로직이 DDP 환경에서도 유지되는지 확인한다.
  - `dtype="float16"` 시 각 rank에서 독립적인 `GradScaler` 인스턴스가 생성되는지 확인한다.
  - `GradScaler` 상태를 체크포인트에 저장하지 않는 기존 동작을 유지한다.
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 13. 체크포인트 재개(Resume) DDP 호환성 확인
  - `--resume` 인수로 지정된 체크포인트를 모든 rank에서 동일하게 로드하는 로직을 확인한다.
  - `GrinViModel.from_pretrained()`로 단일 GPU 체크포인트를 추가 변환 없이 로드할 수 있는지 확인한다.
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 14. 최종 Checkpoint
  - `pytest tests/ -v`로 전체 테스트 스위트를 실행하여 기존 테스트 회귀가 없는지 확인한다.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- `*` 표시 서브태스크는 선택 사항으로, MVP 구현 시 건너뛸 수 있다.
- 속성 기반 테스트는 `hypothesis` 라이브러리 필요: `pip install hypothesis`
- 각 태스크는 이전 태스크 결과를 기반으로 하며, 단일 GPU 경로 회귀를 방지하기 위해 단계별로 검증한다.
- DDP 통합 테스트(`torchrun --nproc_per_node=2`)는 GPU 2개 이상 환경에서만 실행 가능하다.
- Property 2(global_batch_size 계산)는 단순 산술 공식으로 별도 테스트 파일 없이 태스크 4.3의 로그 출력 검증으로 커버한다.
