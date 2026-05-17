#!/bin/bash
# GrinVi Training with Live Monitor
# Usage: ./train_with_monitor.sh [--preset tiny|small|medium|large] [--max-steps 4000]

set -e

PRESET="tiny"
MAX_STEPS="4000"
LOG_FILE="training.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preset) PRESET="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting GrinVi Training with Live Monitor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Preset:       $PRESET"
echo "Max Steps:    $MAX_STEPS"
echo "Log File:     $LOG_FILE"
echo ""
echo "📌 Monitor will start in 3 seconds..."
echo "   Press Ctrl+C to stop training"
echo ""
sleep 3

# Clear old log
> "$LOG_FILE"

# Start training in background
python scripts/training/train.py \
  --preset "$PRESET" \
  --tokenizer morph \
  --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
  --data data/raw/ko_wikipedia/train.txt \
  --eval_data data/raw/ko_wikipedia/val.txt \
  --max_steps "$MAX_STEPS" \
  --grad_ckpt 2>&1 | tee "$LOG_FILE" &

TRAIN_PID=$!

# Give training time to start
sleep 2

# Start advanced monitor
python scripts/monitoring/monitor_advanced.py \
  --log "$LOG_FILE" \
  --max-steps "$MAX_STEPS"

# If monitor quits, kill training
kill $TRAIN_PID 2>/dev/null || true

echo ""
echo "✓ Training session ended"

