#!/bin/bash
# Verify Google Drive access and required GrinVi training files.
# Usage:
#   bash scripts/check_gdrive.sh
#   bash scripts/check_gdrive.sh --remote gdrive:GrinVi

set -euo pipefail

REMOTE="gdrive:GrinVi"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            REMOTE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash scripts/check_gdrive.sh [--remote gdrive:GrinVi]" >&2
            exit 1
            ;;
    esac
done

require_path() {
    local path="$1"
    if rclone lsf "$path" >/dev/null 2>&1; then
        echo "[OK] $path"
    else
        echo "[MISS] $path"
        return 1
    fi
}

echo "== GrinVi Google Drive Check =="
echo "Remote: $REMOTE"

if ! command -v rclone >/dev/null 2>&1; then
    echo "[FAIL] rclone CLI is not installed"
    echo "       Note: 'pip install rclone' installs a Python package, not the rclone command-line tool."
    echo "       Install the CLI with one of the following:"
    echo "         curl https://rclone.org/install.sh | sudo bash"
    echo "         sudo apt-get update && sudo apt-get install -y rclone"
    exit 1
fi
echo "[OK] rclone installed: $(command -v rclone)"

if [[ ! -f "$HOME/.config/rclone/rclone.conf" && -z "${RCLONE_GDRIVE_TOKEN:-}" ]]; then
    echo "[FAIL] rclone is not configured"
    echo "       Need ~/.config/rclone/rclone.conf or RCLONE_GDRIVE_TOKEN"
    exit 1
fi

if ! rclone lsf "$REMOTE" >/dev/null 2>&1; then
    echo "[FAIL] cannot access remote: $REMOTE"
    echo "       Check token, remote name, or drive permissions"
    exit 1
fi
echo "[OK] remote accessible"

echo
echo "-- Required directories --"
require_path "$REMOTE/data/raw/"
require_path "$REMOTE/data/processed/"

echo
echo "-- Required training files --"
missing=0
require_path "$REMOTE/data/processed/train.txt" || missing=1
require_path "$REMOTE/data/raw/ko_wikipedia/train.txt" || missing=1
require_path "$REMOTE/data/raw/ko_wikipedia/val.txt" || missing=1
require_path "$REMOTE/data/raw/ko_wikipedia/ko_tokenizer.json" || missing=1
require_path "$REMOTE/data/raw/ko_wikipedia/ko_tokenizer.vocab" || missing=1

echo
echo "-- Optional checkpoint area --"
if rclone lsf "$REMOTE/checkpoints/" >/dev/null 2>&1; then
    echo "[OK] $REMOTE/checkpoints/"
else
    echo "[INFO] $REMOTE/checkpoints/ not found"
fi

echo
if [[ "$missing" -eq 0 ]]; then
    echo "Drive check passed. Required training inputs are present."
else
    echo "Drive check incomplete. One or more required files are missing."
    exit 2
fi