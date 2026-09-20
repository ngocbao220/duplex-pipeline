#!/usr/bin/env bash
# benchmark.sh — Đánh giá chất lượng âm thanh stereo + Báo cáo Data Retention
#
# Cách dùng:
#   bash benchmark.sh --youtube                   # Đánh giá kết quả YouTube (mặc định duplexchat)
#   bash benchmark.sh --podcast-index             # Đánh giá kết quả Podcast Index
#   bash benchmark.sh --all                       # Đánh giá tuần tự cả hai nguồn
#   bash benchmark.sh --sommelier --youtube       # Đánh giá kết quả Sommelier
#   bash benchmark.sh --youtube --gpu 0           # Chỉ định GPU 0
#   bash benchmark.sh --youtube --dry-run         # Chạy thử

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/pipeline/duplexchat/src:$PROJECT_ROOT/pipeline/sommelier/src:$PROJECT_ROOT/pipeline/cholimex/src:$PROJECT_ROOT/pipeline/sommelier/vendor/podcast_pipeline:$PROJECT_ROOT/pipeline/sommelier/vendor/SepReformer:${PYTHONPATH:-}"

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

# Mặc định là youtube nếu không chỉ định nguồn nào
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    SOURCES=("youtube")
fi

# Resolve Python interpreter (prioritize explicit PYTHON, pipeline .venv, conda env, or local .venv over base)
if [ -n "${PYTHON:-}" ]; then
    : # already set by caller
elif [ "$PIPELINE" = "sommelier" ] && [ -x "$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python"
elif [ "$PIPELINE" = "duplexchat" ] && [ -x "$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/pipeline/duplexchat/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v conda >/dev/null 2>&1 && [ "$PIPELINE" = "duplexchat" ] && conda env list 2>/dev/null | grep -E "^duplexchat\s" >/dev/null 2>&1; then
    CONDA_ENV_PATH=$(conda env list 2>/dev/null | grep -E "^duplexchat\s" | awk '{print $NF}')
    PYTHON="${CONDA_ENV_PATH}/bin/python"
elif command -v conda >/dev/null 2>&1 && [ "$PIPELINE" = "sommelier" ] && conda env list 2>/dev/null | grep -E "^sommelier\s" >/dev/null 2>&1; then
    CONDA_ENV_PATH=$(conda env list 2>/dev/null | grep -E "^sommelier\s" | awk '{print $NF}')
    PYTHON="${CONDA_ENV_PATH}/bin/python"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" >/dev/null 2>&1; then
    CONDA_ENV_PATH=$(conda env list 2>/dev/null | grep -E "^duplex-pipelines\s" | awk '{print $NF}')
    PYTHON="${CONDA_ENV_PATH}/bin/python"
elif [ -x "/opt/conda/bin/python" ]; then
    PYTHON="/opt/conda/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

for SRC in "${SOURCES[@]}"; do
    echo "======================================================================"
    echo " [Phase 3] Stereo Benchmark | Pipeline: [${PIPELINE}] | Source: [${SRC}] | GPU: [${GPU_ID}]"
    echo "======================================================================"
    "$PYTHON" "$PROJECT_ROOT/run_pipeline.py" \
        step=benchmark \
        pipeline="$PIPELINE" \
        env="$MODE" \
        gpu="$GPU_ID" \
        data.source="$SRC" \
        ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
done
