"""
GrinVi Trainer — a clean training loop with:
  ✦ Gradient accumulation
  ✦ Learning-rate warm-up + cosine decay
  ✦ Gradient clipping
  ✦ Periodic checkpointing
  ✦ Rich progress reporting
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from grinvi.model import GrinViModel

console = Console()


# ---------------------------------------------------------------------------
# LR Schedule: linear warm-up + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

class TrainerConfig:
    def __init__(
        self,
        max_steps: int = 100_000,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 2000,
        eval_interval: int = 500,
        save_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",   # "float32" | "float16" | "bfloat16"
        compile_model: bool = False,
        gradient_checkpointing: bool = False,
        log_interval: int = 10,
        keep_last_n: int = 5,      # 보관할 최근 체크포인트 수 (0 = 무제한)
    ):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.dtype = dtype
        self.compile_model = compile_model
        self.gradient_checkpointing = gradient_checkpointing
        self.log_interval = log_interval
        self.keep_last_n = keep_last_n


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: GrinViModel,
        trainer_cfg: TrainerConfig,
        train_loader: Iterable,   # yields (input_ids, labels, attention_mask)
        eval_loader: Optional[Iterable] = None,
    ):
        self.cfg = trainer_cfg
        self.device = trainer_cfg.device

        # Move model to device
        self.model = model.to(self.device)

        # Mixed precision
        self.ptdtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[trainer_cfg.dtype]
        self.scaler = torch.amp.GradScaler('cuda', enabled=(trainer_cfg.dtype == "float16"))

        # TF32 — Ampere+ (A100, RTX 30xx+) 에서 matmul 속도 향상
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Optional torch.compile
        if trainer_cfg.compile_model:
            console.print("[bold cyan]Compiling model with torch.compile …[/bold cyan]")
            self.model = torch.compile(self.model)

        # Gradient checkpointing
        if trainer_cfg.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # Optimiser (AdamW with weight-decay exclusion for biases / norms)
        decay_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and p.dim() < 2]
        # fused=True: CUDA 커널 퓨전으로 옵티마이저 속도 향상 (RTX 5080 최적화)
        _fused_available = torch.cuda.is_available()
        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": trainer_cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=trainer_cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=_fused_available,
        )

        self.scheduler = get_cosine_schedule(
            self.optimizer,
            num_warmup_steps=trainer_cfg.warmup_steps,
            num_training_steps=trainer_cfg.max_steps,
        )

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.step = 0
        self.train_losses: list[float] = []
        self.best_eval_loss: float = float("inf")

    # ------------------------------------------------------------------
    def _forward(self, batch) -> torch.Tensor:
        input_ids, labels, *rest = batch
        attention_mask = rest[0] if rest else None
        input_ids = input_ids.to(self.device)
        labels    = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # Validate batch
        if input_ids.isnan().any() or labels.isnan().any():
            raise ValueError("[ERROR] NaN detected in batch data")
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.ptdtype):
            loss = self.model(input_ids, labels=labels, attention_mask=attention_mask)
        return loss

    # ------------------------------------------------------------------
    def train(self):
        console.rule("[bold green]GrinVi Training")
        console.print(f"  Device      : [bold]{self.device}[/bold]")
        console.print(f"  Params      : [bold]{self.model.num_parameters():,}[/bold]")
        console.print(f"  Max steps   : [bold]{self.cfg.max_steps:,}[/bold]")
        console.print(f"  Batch size  : [bold]{self.cfg.batch_size}[/bold]  × grad_accum {self.cfg.gradient_accumulation_steps}")
        console.print(f"  Grad ckpt   : [bold]{self.cfg.gradient_checkpointing}[/bold]")
        console.print(f"  dtype       : [bold]{self.cfg.dtype}[/bold]")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        t0 = time.time()
        tokens_seen = 0
        consecutive_errors = 0

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training …", total=self.cfg.max_steps)

            global_step = 0
            train_iter = iter(self.train_loader)
            while global_step < self.cfg.max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Start a new epoch transparently; keep training until max_steps.
                    train_iter = iter(self.train_loader)
                    continue
                try:
                    loss = self._forward(batch) / self.cfg.gradient_accumulation_steps

                    # Check for NaN/Inf
                    if not torch.isfinite(loss):
                        console.print(f"  [red]⚠ NaN/Inf detected at step {self.step}! Skipping batch…[/red]")
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()  # Reset scaler
                        consecutive_errors += 1
                        if consecutive_errors > 10:
                            console.print(f"  [red]✗ Too many errors ({consecutive_errors}). Stopping.[/red]")
                            break
                        self.step += 1
                        continue

                    consecutive_errors = 0  # Reset error counter
                    self.scaler.scale(loss).backward()
                    accum_loss += loss.item()

                    # Count tokens
                    input_ids = batch[0]
                    tokens_seen += input_ids.numel()

                    if (self.step + 1) % self.cfg.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step = (self.step + 1) // self.cfg.gradient_accumulation_steps
                        self.train_losses.append(accum_loss)
                        lr = self.optimizer.param_groups[0]["lr"]

                        if global_step % self.cfg.log_interval == 0:
                            dt = time.time() - t0
                            tok_per_sec = tokens_seen / dt if dt > 0 else 0
                            log_line = (
                                f"step {global_step:>6} | loss {accum_loss:.4f} | "
                                f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
                            )
                            progress.update(task, advance=self.cfg.gradient_accumulation_steps,
                                            description=log_line)
                            # 파일에도 plain 텍스트로 기록
                            print(log_line, flush=True)
                        else:
                            progress.update(task, advance=self.cfg.gradient_accumulation_steps)

                        accum_loss = 0.0

                        if global_step % self.cfg.eval_interval == 0 and self.eval_loader is not None:
                            try:
                                self._eval(global_step)
                            except Exception as e:
                                console.print(f"  [yellow]⚠ Eval failed: {e}[/yellow]")
                            self.model.train()

                        if global_step % self.cfg.save_interval == 0:
                            try:
                                self._save(global_step)
                            except Exception as e:
                                console.print(f"  [yellow]⚠ Save failed: {e}[/yellow]")


                    self.step += 1

                except RuntimeError as e:
                    # Handle CUDA OOM, data loading errors, etc.
                    console.print(f"  [red]✗ RuntimeError at step {self.step}: {e}[/red]")
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        console.print(f"  [red]✗ Too many errors. Stopping.[/red]")
                        break
                    self.step += 1
                    continue
                except Exception as e:
                    console.print(f"  [red]✗ Unexpected error at step {self.step}: {type(e).__name__}: {e}[/red]")
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        break
                    self.step += 1
                    continue

        self._save("final")
        total_time = time.time() - t0
        console.rule("[bold green]Training complete")
        console.print(f"  Total time  : {total_time/3600:.2f} h")
        console.print(f"  Tokens seen : {tokens_seen:,}")
        if consecutive_errors > 0:
            console.print(f"  [yellow]⚠ Training ended with {consecutive_errors} error(s)[/yellow]")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, step: int):
        if self.eval_loader is None:
            return
        self.model.eval()
        losses = []
        eval_errors = 0
        for batch in self.eval_loader:
            try:
                loss = self._forward(batch)
                if torch.isfinite(loss):
                    losses.append(loss.item())
                else:
                    eval_errors += 1
            except Exception as e:
                eval_errors += 1
                if eval_errors > 10:
                    console.print(f"    [yellow]⚠ Too many eval errors, stopping eval[/yellow]")
                    break

        if len(losses) == 0:
            console.print(f"  [cyan]eval[/cyan]  step={step}  [yellow]no valid eval data[/yellow]")
            return

        avg = sum(losses) / len(losses)
        ppl = math.exp(min(avg, 20))
        status = f"  [cyan]eval[/cyan]  step={step}  loss={avg:.4f}  ppl={ppl:.2f}"
        if eval_errors > 0:
            status += f"  [yellow]({eval_errors} errors)[/yellow]"
        console.print(status)

        # Best checkpoint 저장
        if avg < self.best_eval_loss:
            self.best_eval_loss = avg
            self._save_best()

    def _save_best(self):
        """현재 모델을 checkpoints/best/에 저장한다 (덮어쓰기)."""
        best_dir = self.cfg.checkpoint_dir / "best"
        try:
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(best_dir))
            console.print(f"  [green]★ Best checkpoint saved → {best_dir}  (loss={self.best_eval_loss:.4f})[/green]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Failed to save best checkpoint: {e}[/yellow]")

    def _cleanup_checkpoints(self):
        """keep_last_n에 따라 오래된 step-* 체크포인트를 삭제한다."""
        if self.cfg.keep_last_n == 0:
            return

        import re
        import shutil

        checkpoint_dir = self.cfg.checkpoint_dir
        if not checkpoint_dir.exists():
            return

        # step-숫자 패턴 디렉토리만 수집 (best/ 제외)
        step_pattern = re.compile(r"^step-(\d+)$")
        step_dirs = []
        for entry in checkpoint_dir.iterdir():
            if entry.is_dir():
                m = step_pattern.match(entry.name)
                if m:
                    step_dirs.append((int(m.group(1)), entry))

        # 스텝 번호 기준 오름차순 정렬
        step_dirs.sort(key=lambda x: x[0])

        # keep_last_n 초과분(가장 오래된 것부터) 삭제
        excess = len(step_dirs) - self.cfg.keep_last_n
        if excess <= 0:
            return

        for _, dir_path in step_dirs[:excess]:
            try:
                shutil.rmtree(dir_path)
                console.print(f"  [dim]Removed old checkpoint: {dir_path}[/dim]")
            except Exception as e:
                console.print(f"  [yellow]⚠ Failed to remove checkpoint {dir_path}: {e}[/yellow]")

    def _save(self, tag):
        try:
            out = self.cfg.checkpoint_dir / f"step-{tag}"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(out))
            console.print(f"  [yellow]Checkpoint saved → {out}[/yellow]")
        except Exception as e:
            console.print(f"  [red]✗ Failed to save checkpoint: {e}[/red]")

        # final 태그일 때는 정리 건너뜀
        if tag != "final":
            self._cleanup_checkpoints()

