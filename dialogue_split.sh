#!/usr/bin/env bash
# dialogue_split.sh — Bước 1: Tiền xử lý & Lọc cuộc thoại hai người (Dialogue Filtering)
#
# Cách dùng:
#   bash dialogue_split.sh --youtube                   # Chạy cho folder YouTube (mặc định)
#   bash dialogue_split.sh --podcast-index             # Chạy cho folder Podcast Index
#   bash dialogue_split.sh --all                       # Chạy tuần tự cho cả hai nguồn
#   bash dialogue_split.sh --youtube --gpu 0           # Chỉ định GPU 0
#   bash dialogue_split.sh --youtube --dev             # Chạy chế độ Dev (online)
#   bash dialogue_split.sh --youtube --dry-run         # Chạy thử (không ghi dữ liệu)

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

# Mặc định là youtube nếu không chỉ định nguồn nào
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    SOURCES=("youtube")
fi

# Resolve Python interpreter (prioritize duplex-pipelines conda env or local .venv over base)
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
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
    echo " [Phase 1] Split Dialogue | Source: [${SRC}] | GPU: [${GPU_ID}]"
    echo "======================================================================"
    "$PYTHON" "$PROJECT_ROOT/run_pipeline.py" \
        step=split_dialogue \
        env="$MODE" \
        gpu="$GPU_ID" \
        data.source="$SRC" \
        ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
done
