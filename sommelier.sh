#!/usr/bin/env bash
# sommelier.sh — Bước 2: Tách âm và tái tạo stereo bằng Sommelier (SepReformer)
#
# Cách dùng:
#   bash sommelier.sh --youtube                   # Chạy cho folder YouTube (mặc định)
#   bash sommelier.sh --podcast-index             # Chạy cho folder Podcast Index
#   bash sommelier.sh --all                       # Chạy tuần tự cho cả hai nguồn
#   bash sommelier.sh --youtube --gpu 0           # Chỉ định GPU 0
#   bash sommelier.sh --youtube --dev             # Chế độ Dev online
#   bash sommelier.sh --youtube --dry-run         # Chạy thử

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-sever}"
GPU_ID="${GPU_ID:-2}"
SOURCES=()
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --youtube|-y)
            SOURCES+=("youtube")
            shift
            ;;
        --podcast-index|-p)
            SOURCES+=("podcast_index")
            shift
            ;;
        --all|-a)
            SOURCES=("youtube" "podcast_index")
            shift
            ;;
        --gpu|-g)
            GPU_ID="$2"
            shift 2
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        --sever|--server)
            MODE="sever"
            shift
            ;;
        --workers|-w)
            HYDRA_ARGS+=("optimization.workers=$2")
            shift 2
            ;;
        --dry-run)
            HYDRA_ARGS+=("dry_run=true")
            shift
            ;;
        *)
            HYDRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Resolve Python interpreter (prioritize pipeline .venv, conda env, or local .venv over base)
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -x "$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" >/dev/null 2>&1; then
    CONDA_ENV_PATH=$(conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" | awk '{print $NF}')
    PYTHON="${CONDA_ENV_PATH}/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

for SRC in "${SOURCES[@]}"; do
    echo "======================================================================"
    echo " [Phase 2 - Sommelier] Separate & Reconstruct | Source: [${SRC}] | GPU: [${GPU_ID}]"
    echo "======================================================================"
    "$PYTHON" "$PROJECT_ROOT/run_pipeline.py" \
        step=sommelier \
        pipeline=sommelier \
        env="$MODE" \
        gpu="$GPU_ID" \
        data.source="$SRC" \
        ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
done
