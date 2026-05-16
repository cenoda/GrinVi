#!/bin/bash
# Pull GrinVi datasets from Google Drive into the current workspace.
# Usage:
#   bash scripts/pull_gdrive_data.sh
#   bash scripts/pull_gdrive_data.sh --with-checkpoints
#   bash scripts/pull_gdrive_data.sh --workspace /workspaces/GrinVi --dry-run

set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
WITH_CHECKPOINTS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --with-checkpoints)
            WITH_CHECKPOINTS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash scripts/pull_gdrive_data.sh [--workspace PATH] [--with-checkpoints] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is not installed." >&2
    echo "Install it first, or use scripts/start_vast.sh on a GPU machine." >&2
    exit 1
fi

mkdir -p "$HOME/.config/rclone"
if [[ ! -f "$HOME/.config/rclone/rclone.conf" && -z "${RCLONE_GDRIVE_TOKEN:-}" ]]; then
    echo "ERROR: rclone is not configured." >&2
    echo "Provide ~/.config/rclone/rclone.conf or export RCLONE_GDRIVE_TOKEN first." >&2
    exit 1
fi

if [[ ! -f "$HOME/.config/rclone/rclone.conf" && -n "${RCLONE_GDRIVE_TOKEN:-}" ]]; then
    cat > "$HOME/.config/rclone/rclone.conf" <<RCLONE_EOF
[gdrive]
type = drive
scope = drive
token = ${RCLONE_GDRIVE_TOKEN}
team_drive =
RCLONE_EOF
fi

SYNC_FLAGS=(--progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
    SYNC_FLAGS+=(--dry-run)
fi

mkdir -p "$WORKSPACE/data/raw" "$WORKSPACE/data/processed"

echo "Pulling Google Drive data into $WORKSPACE"
rclone copy gdrive:GrinVi/data/raw/ "$WORKSPACE/data/raw/" "${SYNC_FLAGS[@]}"
rclone copy gdrive:GrinVi/data/processed/ "$WORKSPACE/data/processed/" "${SYNC_FLAGS[@]}"

if [[ "$WITH_CHECKPOINTS" -eq 1 ]]; then
    mkdir -p "$WORKSPACE/checkpoints"
    rclone copy gdrive:GrinVi/checkpoints/ "$WORKSPACE/checkpoints/" "${SYNC_FLAGS[@]}"
fi

echo "Done."
echo "  raw:       $WORKSPACE/data/raw"
echo "  processed: $WORKSPACE/data/processed"
if [[ "$WITH_CHECKPOINTS" -eq 1 ]]; then
    echo "  checkpoints: $WORKSPACE/checkpoints"
fi