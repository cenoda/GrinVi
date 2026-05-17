#!/usr/bin/env python3
"""
scripts/train_pipeline.py — Unified training pipeline for GrinVi.
Combines preflight, smoke test, and main training run into one command.

Example:
python scripts/train_pipeline.py \
  --data data/processed/train.txt \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --checkpoint_dir checkpoints/v1_run \
  --gpus 1
"""
import argparse
import subprocess
import sys
import os
import signal
import atexit
import time
from pathlib import Path

# Track processes for cleanup
processes = []

def cleanup():
    """Ensure all child processes are killed on exit."""
    if not processes:
        return
    print("\n[Pipeline] Cleaning up child processes...")
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    
    # Give them a moment to exit gracefully
    time.sleep(2)
    for p in processes:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

atexit.register(cleanup)

def run_command(cmd, env=None, background=False):
    print(f"\n[Pipeline] Running: {' '.join(cmd)}")
    if background:
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        return p
    else:
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        try:
            p.wait()
        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")
            sys.exit(1)
        return p

def check_for_overstacking():
    """Check if there are already training processes running."""
    try:
        # Simple check using pgrep
        res = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        pids = res.stdout.strip().split()
        # Exclude our own PID
        pids = [p for p in pids if int(p) != os.getpid()]
        if pids:
            print(f"\n[WARNING] Detected {len(pids)} existing 'train.py' processes (PIDs: {', '.join(pids)}).")
            print("This might lead to 'overstacking' and OOM on Vast.ai.")
            ans = input("Continue anyway? [y/N]: ")
            if ans.lower() != 'y':
                sys.exit(0)
    except Exception:
        pass # pgrep might not be available

def main():
    parser = argparse.ArgumentParser(description="GrinVi Unified Training Pipeline")
    
    # Core arguments that the pipeline needs to know about
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument("--eval_data", help="Path to eval text file")
    parser.add_argument("--tokenizer", choices=["cl100k_base", "sentencepiece", "morph"], default="cl100k_base")
    parser.add_argument("--tokenizer_model", help="Path to tokenizer model (required for morph/sp)")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory to save main checkpoints")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (uses torchrun if > 1)")
    
    # Training arguments to be passed through (explicitly handled to avoid duplication in smoke test)
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", help="Path to checkpoint to resume from")

    # Pipeline control
    parser.add_argument("--skip_preflight", action="store_true", help="Skip safety checks")
    parser.add_argument("--skip_smoke", action="store_true", help="Skip 50-step smoke test")
    parser.add_argument("--smoke_steps", type=int, default=50, help="Number of steps for smoke test")
    parser.add_argument("--no_interactive", action="store_true", help="Do not ask for confirmation before main run")
    parser.add_argument("--prompt", default="질문: 인공지능에 대해 설명해줘.\n답변:", help="Prompt for generation check")
    parser.add_argument("--backup_name", help="Remote folder name for backup (e.g. my_experiment)")
    parser.add_argument("--upload_interval", type=int, default=300, help="Upload interval in seconds")

    # The rest are passed to train.py
    args, unknown = parser.parse_known_args()

    if not args.no_interactive:
        check_for_overstacking()

    # Determine command prefix (python or torchrun)
    def get_run_prefix(num_gpus):
        if num_gpus > 1:
            return ["torchrun", f"--nproc_per_node={num_gpus}"]
        else:
            return [sys.executable]

    # Shared args for both smoke and main
    base_args = [
        "--data", args.data,
        "--tokenizer", args.tokenizer,
        "--preset", args.preset,
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--lr", str(args.lr),
    ]
    if args.eval_data: base_args += ["--eval_data", args.eval_data]
    if args.tokenizer_model: base_args += ["--tokenizer_model", args.tokenizer_model]
    if args.resume: base_args += ["--resume", args.resume]
    base_args += unknown

    # 1. Preflight
    if not args.skip_preflight:
        print("\n" + "="*60)
        print("STEP 1: Preflight Safety Checks")
        print("="*60)
        preflight_cmd = [sys.executable, "scripts/training/preflight_train.py", "--data", args.data, "--tokenizer", args.tokenizer]
        if args.tokenizer_model: preflight_cmd += ["--tokenizer_model", args.tokenizer_model]
        if args.eval_data: preflight_cmd += ["--eval_data", args.eval_data]
        if args.resume: preflight_cmd += ["--resume", args.resume]

        res = run_command(preflight_cmd)
        if res.returncode != 0:
            print("\n[Pipeline] Preflight FAILED. Please fix the issues above.")
            sys.exit(1)
        print("[Pipeline] Preflight PASSED.")

    # 2. Smoke Test
    main_checkpoint_path = Path(args.checkpoint_dir)
    smoke_dir = main_checkpoint_path.parent / f"{main_checkpoint_path.name}_smoke"
    
    if not args.skip_smoke:
        print("\n" + "="*60)
        print(f"STEP 2: Smoke Test ({args.smoke_steps} steps)")
        print("="*60)
        
        smoke_cmd = get_run_prefix(args.gpus) + ["scripts/training/train.py"] + base_args
        smoke_cmd += ["--checkpoint_dir", str(smoke_dir), "--max_steps", str(args.smoke_steps)]
        # Ensure we save/eval at the end of smoke test
        smoke_cmd += ["--save_interval", str(args.smoke_steps), "--eval_interval", str(args.smoke_steps)]

        res = run_command(smoke_cmd)
        if res.returncode != 0:
            print("\n[Pipeline] Smoke test FAILED.")
            sys.exit(1)
        print("[Pipeline] Smoke test COMPLETED.")

        # 3. Generation Check
        print("\n" + "="*60)
        print("STEP 3: Generation Check")
        print("="*60)
        
        # Find latest checkpoint in smoke_dir
        last_ckpt = smoke_dir / "final"
        if not last_ckpt.exists():
            ckpts = sorted(smoke_dir.glob("step-*"))
            if ckpts: last_ckpt = ckpts[-1]
            
        if last_ckpt.exists() and last_ckpt.is_dir():
            gen_cmd = [sys.executable, "scripts/tools/inference.py", "--checkpoint", str(last_ckpt), "--tokenizer", args.tokenizer]
            if args.tokenizer_model: gen_cmd += ["--tokenizer_model", args.tokenizer_model]
            gen_cmd += ["--prompt", args.prompt, "--max_new_tokens", "100"]
            run_command(gen_cmd)
        else:
            print(f"[Pipeline] WARNING: Could not find checkpoint in {smoke_dir} for generation check.")

    # 4. Confirmation
    if not args.no_interactive:
        print("\n" + "="*60)
        try:
            ans = input(f"[Pipeline] Ready for MAIN training run in '{args.checkpoint_dir}'.\nProceed? [y/N]: ")
        except EOFError:
            ans = "n"
        if ans.lower() != 'y':
            print("[Pipeline] Aborting main training.")
            sys.exit(0)

    # 5. Main Run
    print("\n" + "="*60)
    print("STEP 4: Main Training Run")
    print("="*60)
    
    # Check if checkpoint_dir already exists
    if main_checkpoint_path.exists():
        print(f"[Pipeline] WARNING: Checkpoint directory '{args.checkpoint_dir}' already exists.")
        if not args.no_interactive:
            try:
                ans = input("Overwrite or continue in the same directory? [y/N]: ")
            except EOFError:
                ans = "n"
            if ans.lower() != 'y':
                print("[Pipeline] Aborting to prevent accidental overwrite.")
                sys.exit(0)

    main_cmd = get_run_prefix(args.gpus) + ["scripts/training/train.py"] + base_args
    main_cmd += ["--checkpoint_dir", args.checkpoint_dir, "--max_steps", str(args.max_steps)]
    
    # 6. Start Backup in background if requested
    if args.backup_name:
        backup_cmd = ["bash", "scripts/infra/backup_checkpoints.sh", args.checkpoint_dir, args.backup_name]
        env = os.environ.copy()
        env["INTERVAL"] = str(args.upload_interval)
        run_command(backup_cmd, env=env, background=True)

    res = run_command(main_cmd)
    if res.returncode == 0:
        print("\n[Pipeline] ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n[Pipeline] Main training failed with exit code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
