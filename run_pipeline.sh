#!/usr/bin/env bash
# run_pipeline.sh — Thực thi toàn bộ Pipeline Duplex-Pipelines (End-to-End)
#
# Cách dùng:
#   bash run_pipeline.sh --youtube                # Chạy toàn bộ cho folder YouTube
#   bash run_pipeline.sh --podcast-index          # Chạy toàn bộ cho folder Podcast Index
#   bash run_pipeline.sh --all                    # Chạy toàn bộ cho cả hai nguồn
#   bash run_pipeline.sh --sommelier --youtube    # Chạy pipeline Sommelier
#   bash run_pipeline.sh --youtube --gpu 0        # Chỉ định GPU 0
#   bash run_pipeline.sh --youtube --dry-run      # Chạy thử toàn bộ các bước

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIPELINE="${PIPELINE:-duplexchat}"
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
        --duplexchat)
            PIPELINE="duplexchat"
            shift
            ;;
        --sommelier)
            PIPELINE="sommelier"
            shift
            ;;
        --pipeline)
            PIPELINE="$2"
            shift 2
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

# Resolve Python interpreter (prioritize active conda env, pipeline .venv, or local .venv over base)
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [ "$PIPELINE" = "sommelier" ] && [ -x "$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python"
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
    echo " [End-to-End] Full Pipeline | [${PIPELINE}] | Source: [${SRC}] | GPU: [${GPU_ID}]"
    echo "======================================================================"
    "$PYTHON" "$PROJECT_ROOT/run_pipeline.py" \
        step=all \
        pipeline="$PIPELINE" \
        env="$MODE" \
        gpu="$GPU_ID" \
        data.source="$SRC" \
        ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
done
