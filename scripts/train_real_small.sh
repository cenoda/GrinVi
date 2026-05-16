#!/bin/bash
# Train GrinVi Small Model with 25GB Processed Data
set -e
cd /home/cenoda/GrinVi
echo "════════════════════════════════════════════════════════════"
echo "   🇰🇷 GrinVi REAL DATA + SMALL MODEL TRAINING (117M params)"
echo "════════════════════════════════════════════════════════════"
echo ""
DATA_FILE="data/processed/train.txt"
TOKENIZER_MODEL="data/raw/ko_wikipedia/ko_tokenizer.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found."
    exit 1
fi
echo "Model: small (12 layers, 768 hidden)"
echo "Data: 25GB Combined Korean Data"
echo "Batch: 4 sequences, Grad Accum: 8 (effective 32)"
python scripts/train.py \
    --preset small \
    --tokenizer morph \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --data "$DATA_FILE" \
    --checkpoint_dir "checkpoints_small" \
    --seq_len 1024 \
    --batch_size 4 \
    --grad_accum 8 \
    --max_steps 50000 \
    --eval_interval 1000 \
    --save_interval 1000 \
    --grad_ckpt
