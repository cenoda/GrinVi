#!/bin/bash
# Quick start script for Korean GrinVi training
# Usage: bash scripts/quickstart_korean.sh

set -e

echo "🚀 GrinVi Korean Training Quick Start"
echo "======================================"
echo ""

# Step 1: Create sample Korean data if it doesn't exist
if [ ! -f "data/korean_sample.txt" ]; then
    echo "📝 Creating sample Korean corpus..."
    mkdir -p data
    cat > data/korean_sample.txt << 'EOF'
안녕하세요! GrinVi 한글 모델 훈련에 오신 것을 환영합니다.
한국어는 매우 아름다운 언어입니다.
세종대왕께서 만드신 한글은 과학적이고 체계적입니다.
대한민국은 디지털 강국으로 알려져 있습니다.
한국 음식은 건강하고 맛있습니다.
서울은 동아시아에서 가장 발전한 도시 중 하나입니다.
한국 문화는 전 세계에 영향을 미치고 있습니다.
한국 사람들은 매우 친절합니다.
한국의 자연은 아름답고 다양합니다.
한국의 역사는 길고 영광스럽습니다.
공부는 열심히 하면 반드시 좋은 결과를 만듭니다.
기술 발전은 인류의 미래를 밝게 합니다.
우리는 함께 더 나은 세상을 만들 수 있습니다.
EOF
    echo "✓ Sample data created at data/korean_sample.txt"
fi

# Step 2: Train tokenizer
echo ""
echo "🔤 Step 1: Training Korean tokenizer..."
python scripts/train_tokenizer.py \
    --data data/korean_sample.txt \
    --output data/korean_tok \
    --vocab_size 8000 \
    --character_coverage 0.9995 \
    --test "안녕하세요"

# Step 3: Train model
echo ""
echo "🤖 Step 2: Training GrinVi model with Korean data..."
python scripts/train.py \
    --preset tiny \
    --tokenizer sentencepiece \
    --tokenizer_model data/korean_tok.model \
    --data data/korean_sample.txt \
    --seq_len 256 \
    --batch_size 4 \
    --max_steps 50 \
    --grad_ckpt

echo ""
echo "✅ Korean training complete!"
echo ""
echo "Next steps:"
echo "  1. Generate Korean text:"
echo "     python scripts/generate.py --checkpoint checkpoints/step-final --prompt '안녕하세요'"
echo ""
echo "  2. For larger training, use Korean Wikipedia:"
echo "     python scripts/prepare_data.py --dataset ko_wikipedia --out data/"
echo ""
echo "  3. For real training, use more Korean data:"
echo "     See KOREAN_GUIDE.md for detailed instructions"

