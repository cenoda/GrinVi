#!/bin/bash
# 체크포인트가 새로 저장될 때마다 구글 드라이브에 자동 백업
# 사용법:
#   bash scripts/backup_checkpoints.sh &                    # 기본 (checkpoints/)
#   bash scripts/backup_checkpoints.sh checkpoints my_run & # 커스텀 dir + 원격 이름

CKPT_DIR="${1:-checkpoints}"
REMOTE_NAME="${2:-$(basename "$CKPT_DIR")}"
GDRIVE_DEST="gdrive:GrinVi/$REMOTE_NAME"
INTERVAL="${INTERVAL:-60}"  # 체크 주기 (초)

# 상대 경로면 현재 작업 디렉토리 기준으로 절대 경로화
CKPT_DIR="$(cd "$(dirname "$CKPT_DIR")" 2>/dev/null && pwd)/$(basename "$CKPT_DIR")"

echo "[backup] 시작 — $CKPT_DIR → $GDRIVE_DEST"
echo "[backup] ${INTERVAL}초마다 새 체크포인트 확인 (Ctrl+C로 중지)"

LAST_SYNCED=""

while true; do
    # 가장 최근 체크포인트 폴더 찾기 (mtime 기준)
    LATEST=$(ls -td "$CKPT_DIR"/step-* 2>/dev/null | head -1)

    if [ -z "$LATEST" ]; then
        sleep "$INTERVAL"
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

    sleep "$INTERVAL"
done
