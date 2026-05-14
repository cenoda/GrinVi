#!/bin/bash
# GrinVi — vast.ai startup script
# Tested on: PyTorch NGC image, RTX Pro 6000 / A100
set -e

echo "════════════════════════════════════════"
echo "   GrinVi vast.ai Setup"
echo "════════════════════════════════════════"

# 1. rclone 설치
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | bash
fi

# 2. gdrive 설정 (토큰 붙여넣기)
mkdir -p ~/.config/rclone
# rclone.conf를 직접 붙여넣거나 아래 환경변수로 설정
# 방법 1: 환경변수 (권장)
#   export RCLONE_GDRIVE_TOKEN='{"access_token":"...","refresh_token":"YOUR_REFRESH_TOKEN",...}'
# 방법 2: 파일 직접 복사
#   scp ~/.config/rclone/rclone.conf root@<vast-ip>:~/.config/rclone/rclone.conf

if [ -z "$RCLONE_GDRIVE_TOKEN" ]; then
    echo "ERROR: RCLONE_GDRIVE_TOKEN env var not set."
    echo "Set it before running this script:"
    echo "  export RCLONE_GDRIVE_TOKEN='\$(cat ~/.config/rclone/rclone.conf | grep token | cut -d= -f2-)'"
    exit 1
fi

cat > ~/.config/rclone/rclone.conf << RCLONE_EOF
[gdrive]
type = drive
scope = drive
token = ${RCLONE_GDRIVE_TOKEN}
team_drive =
RCLONE_EOF

# 3. 레포 클론
if [ ! -d "/workspace/GrinVi" ]; then
    git clone https://github.com/cenoda/GrinVi /workspace/GrinVi
fi
cd /workspace/GrinVi

# 4. 의존성 설치
pip install -r requirements.txt -q

# 5. 데이터 다운로드
echo "Downloading training data from Google Drive..."
rclone copy gdrive:GrinVi/data/ /workspace/GrinVi/data/ --progress

echo "Data ready!"
du -sh data/processed/train.txt

# 6. GPU 확인
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 7. 학습 시작
echo "Starting training..."
nohup python scripts/train.py \
    --preset medium \
    --tokenizer sentencepiece \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.model \
    --data data/processed/train.txt \
    --seq_len 512 \
    --batch_size 64 \
    --grad_accum 2 \
    --max_steps 100000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --lr 3e-4 \
    --dtype bfloat16 \
    --grad_ckpt \
    --compile \
    --keep_last_n 3 \
    > training.log 2>&1 &

echo "Training PID: $!"
echo "Monitor: tail -f /workspace/GrinVi/training.log"
