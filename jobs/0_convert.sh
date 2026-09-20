#!/usr/bin/env bash
# convert.sh — Chuyển đổi audio crawl (.webm/.m4a/.tar.gz) sang chuẩn PCM 16kHz mono WAV
#
# Cách dùng:
#   bash convert.sh --youtube         # Chỉ chuyển đổi folder YouTube
#   bash convert.sh --podcast-index   # Chỉ chuyển đổi folder Podcast Index
#   bash convert.sh                   # Chuyển đổi cả YouTube và Podcast Index (mặc định)
#   bash convert.sh --all             # Chuyển đổi cả YouTube và Podcast Index

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -d "/storage-voice/voice/vdt/baottn/duplex-data" ]; then
    DEFAULT_BASE_DATA="/storage-voice/voice/vdt/baottn/duplex-data"
else
    DEFAULT_BASE_DATA="$PROJECT_ROOT/data"
fi
BASE_DATA="${BASE_DATA:-$DEFAULT_BASE_DATA}"
CRAWL_DIR="${CRAWL_DIR:-$BASE_DATA/crawl}"
RAW_DIR="${RAW_DIR:-$BASE_DATA/raw}"

FLAGS=()
WORKERS="${WORKERS:-8}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --youtube|-y)
            FLAGS+=("--youtube")
            shift
            ;;
        --podcast-index|-p)
            FLAGS+=("--podcast-index")
            shift
            ;;
        --all|-a)
            # Default is both
            shift
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        *)
            FLAGS+=("$1")
            shift
            ;;
    esac
done
FLAGS+=("--workers" "$WORKERS")

# Resolve Python interpreter (prioritize explicit PYTHON, duplex-pipelines conda env or local .venv over base)
if [ -n "${PYTHON:-}" ]; then
    : # already set by caller
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" >/dev/null 2>&1; then
    CONDA_ENV_PATH=$(conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" | awk '{print $NF}')
    PYTHON="${CONDA_ENV_PATH}/bin/python"
elif [ -x "/opt/conda/bin/python" ]; then
    PYTHON="/opt/conda/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

"$PYTHON" "$PROJECT_ROOT/scripts/convert_crawl_to_raw.py" \
    --crawl-dir "$CRAWL_DIR" \
    --raw-dir "$RAW_DIR" \
    ${FLAGS[@]+"${FLAGS[@]}"}

