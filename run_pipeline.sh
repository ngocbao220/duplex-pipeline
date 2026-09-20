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
        gpu=*)
            GPU_ID="${1#gpu=}"
            shift
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        env=*)
            MODE="${1#env=}"
            shift
            ;;
        --sever|--server)
            MODE="sever"
            shift
            ;;
        --max-chunk-seconds|--diarize-chunk)
            HYDRA_ARGS+=("diarization.max_chunk_seconds=$2")
            shift 2
            ;;
        max_chunk_seconds=*|+max_chunk_seconds=*)
            val="${1#*=}"
            HYDRA_ARGS+=("diarization.max_chunk_seconds=$val")
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

# Resolve Python interpreter (prioritize pipeline .venv, conda env, active env, or system python with hydra)
if [ "$PIPELINE" = "sommelier" ] && [ -x "$PROJECT_ROOT/pipeline/sommelier/.venv/bin/python" ]; then
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
elif command -v python >/dev/null 2>&1 && python -c "import hydra" >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import hydra" >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif [ -x "/opt/conda/bin/python" ]; then
    PYTHON="/opt/conda/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

# Safeguard: Ensure hydra is available in the selected python interpreter
if ! "$PYTHON" -c "import hydra" >/dev/null 2>&1; then
    echo "[WARN] Interpreter '$PYTHON' is missing 'hydra'. Auto-installing hydra-core & omegaconf..."
    "$PYTHON" -m pip install "hydra-core>=1.3.2" "omegaconf>=2.3.0" || true
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
