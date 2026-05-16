#!/bin/bash
# 체크포인트 저장될 때마다 구글 드라이브에 자동 백업
# 사용법: bash scripts/backup_checkpoints.sh &

WORKSPACE="${1:-/workspace/GrinVi}"
CKPT_DIR="$WORKSPACE/checkpoints_medium"
GDRIVE_DEST="gdrive:GrinVi/checkpoints_medium"
INTERVAL=60  # 체크 주기 (초)

echo "[backup] 시작 — $CKPT_DIR → $GDRIVE_DEST"
echo "[backup] ${INTERVAL}초마다 새 체크포인트 확인"

LAST_SYNCED=""

while true; do
    # 가장 최근 체크포인트 폴더 찾기
    LATEST=$(ls -td "$CKPT_DIR"/step-* 2>/dev/null | head -1)

    if [ -z "$LATEST" ]; then
        sleep $INTERVAL
        continue
    fi

    if [ "$LATEST" != "$LAST_SYNCED" ]; then
        STEP=$(basename "$LATEST")
        echo "[backup] 새 체크포인트 감지: $STEP — 업로드 중..."

        rclone copy "$LATEST" "$GDRIVE_DEST/$STEP" \
            --transfers 4 \
            --checkers 8 \
            --fast-list \
            2>&1 | tail -1

        if [ $? -eq 0 ]; then
            echo "[backup] ✓ $STEP 업로드 완료 ($(date '+%H:%M:%S'))"
            LAST_SYNCED="$LATEST"
        else
            echo "[backup] ✗ $STEP 업로드 실패, 다음 주기에 재시도"
        fi
    fi

    sleep $INTERVAL
done
