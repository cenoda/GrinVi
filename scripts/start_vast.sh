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
rclone copy gdrive:GrinVi/data/raw/ /workspace/GrinVi/data/raw/ --progress
rclone copy gdrive:GrinVi/data/processed/ /workspace/GrinVi/data/processed/ --progress

echo "Data ready!"
du -sh data/processed/train.txt

# 6. 체크포인트 복원 (medium 모델만 받아오기)
echo "Checking for medium checkpoints on Google Drive..."
mkdir -p /workspace/GrinVi/checkpoints

# 드라이브에서 체크포인트 목록 가져와서 medium(hidden_size=1024)만 필터링
for ckpt in $(rclone lsd gdrive:GrinVi/checkpoints/ 2>/dev/null | awk '{print $NF}' | grep "^step-" | grep -v "step-final"); do
    config=$(rclone cat "gdrive:GrinVi/checkpoints/$ckpt/config.json" 2>/dev/null)
    hidden=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('hidden_size',0))" 2>/dev/null)
    if [ "$hidden" = "1024" ]; then
        echo "Found medium checkpoint: $ckpt (hidden_size=1024)"
        rclone copy "gdrive:GrinVi/checkpoints/$ckpt" "/workspace/GrinVi/checkpoints/$ckpt" --progress
    fi
done

# 가장 최근 medium 체크포인트 찾기
RESUME_FLAG=""
LATEST_CKPT=$(ls -td /workspace/GrinVi/checkpoints/step-* 2>/dev/null | grep -v step-final | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "No medium checkpoint found, starting from scratch."
fi

# 7. GPU 확인 및 DDP 설정
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multiple GPUs detected ($NUM_GPUS). Using torchrun for DDP."
    LAUNCH_CMD="torchrun --nproc_per_node=$NUM_GPUS"
else
    echo "Single GPU detected. Using python."
    LAUNCH_CMD="python"
fi

# 8. 백업 프로세스 시작
echo "Starting checkpoint backup daemon..."
nohup bash scripts/backup_checkpoints.sh /workspace/GrinVi > /workspace/GrinVi/backup.log 2>&1 &
echo "Backup PID: $!"

# 9. 학습 시작
echo "Starting training..."
nohup $LAUNCH_CMD scripts/train.py \
    --preset medium \
    --tokenizer morph \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
    --data data/processed/train.txt \
    --seq_len 512 \
    --batch_size 128 \
    --grad_accum 1 \
    --max_steps 150000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --lr 3e-4 \
    --dtype bfloat16 \
    --grad_ckpt \
    --compile \
    --keep_last_n 3 \
    $RESUME_FLAG \
    > training.log 2>&1 &

echo "Training PID: $!"
echo "Monitor: tail -f /workspace/GrinVi/training.log"
echo "Backup:  tail -f /workspace/GrinVi/backup.log"
