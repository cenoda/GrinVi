#!/bin/bash
# Train GrinVi Tiny Model with Real Korean Data
set -e
cd /home/cenoda/GrinVi
MAX_ITEMS=${MAX_ITEMS:-50000}
MAX_STEPS=${MAX_STEPS:-2000}
echo "════════════════════════════════════════════════════════════"
echo "   🇰🇷 GrinVi REAL DATA + TINY MODEL TRAINING"
echo "════════════════════════════════════════════════════════════"
echo ""
# Step 1: Prepare real Korean data
echo "════════════════════════════════════════════════════════════"
echo "📚 STEP 1: Downloading Korean Wikipedia + building tokenizer"
echo "════════════════════════════════════════════════════════════"
python scripts/prepare_data.py --dataset ko_wikipedia --out data/raw/ --max_tokens "$MAX_ITEMS"

DATA_FILE="data/raw/ko_wikipedia/train.txt"
VAL_FILE="data/raw/ko_wikipedia/val.txt"
TOKENIZER_MODEL="data/raw/ko_wikipedia/ko_tokenizer.json"

echo "✓ Data ready!"
echo "  Train: $(du -h $DATA_FILE | cut -f1)"
echo "  Val:   $(du -h $VAL_FILE | cut -f1)"
echo "  Tok:   $TOKENIZER_MODEL"
echo ""
# Step 2: Train tiny model
echo "════════════════════════════════════════════════════════════"
echo "🤖 STEP 2: Training Tiny Model (15M params)"
echo "════════════════════════════════════════════════════════════"
echo "Model: tiny (4 layers, 256 hidden)"
echo "Data: Real Korean (Wikipedia + NSMC)"
echo "Batch: 4 sequences"
echo "Tokens/sec: ~30k (expected)"
echo "Time: ~30-60 minutes for 5000 steps"
echo ""
python scripts/training/train.py \
    --preset tiny \
    --tokenizer morph \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --data "$DATA_FILE" \
    --eval_data "$VAL_FILE" \
    --seq_len 256 \
    --batch_size 4 \
    --grad_accum 2 \
    --max_steps "$MAX_STEPS" \
    --eval_interval 200 \
    --save_interval 200 \
    --grad_ckpt
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Model saved at: checkpoints/step-final"
echo ""
echo "Next: Generate Korean text!"
echo ""
echo "Interactive mode:"
echo "  python scripts/tools/inference.py --checkpoint checkpoints/step-final --tokenizer morph --tokenizer_model \"$TOKENIZER_MODEL\""
echo ""
echo "Example:"
echo "  python scripts/tools/inference.py --checkpoint checkpoints/step-final --prompt '한국은' --tokenizer morph --tokenizer_model \"$TOKENIZER_MODEL\""
echo ""
