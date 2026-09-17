#!/usr/bin/env bash
# ==============================================================================
# scripts/env.sh — Cấu hình môi trường dùng chung cho tất cả pipeline scripts
# Source file này thay vì chạy trực tiếp:
#   source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
# ==============================================================================

# ANSI colors
export C_RESET='\033[0m'
export C_INFO='\033[0;36m'      # Cyan
export C_SUCCESS='\033[0;32m'   # Green
export C_WARN='\033[0;33m'      # Yellow
export C_ERROR='\033[0;31m'     # Red
export C_BOLD='\033[1m'

log_info()    { echo -e "${C_INFO}[INFO]${C_RESET}  $*"; }
log_warn()    { echo -e "${C_WARN}[WARNING]${C_RESET} $*"; }
log_error()   { echo -e "${C_ERROR}[ERROR]${C_RESET} $*"; }
log_section() { echo -e "\n${C_BOLD}$*${C_RESET}"; }

# --------------------------------------------------------------------------
# Pipeline run ID (dùng để gom timing của cùng 1 run)
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Project root (được set bởi script gọi, mặc định fallback)
if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/pipeline/duplexchat/src:${PYTHONPATH:-}"

# Thread limiting
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --------------------------------------------------------------------------
# MODE: dev (HuggingFace Hub) | sever (offline local only)
export MODE="${MODE:-sever}"
export GPU_ID="${GPU_ID:-2}"

if [ "$MODE" = "sever" ] || [ "$MODE" = "server" ] || [ "$MODE" = "offline" ]; then
    log_info "[MODE: SEVER] Offline — chỉ dùng model local disk, cấm tải mạng"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    log_info "[MODE: DEV] Dev — cho phép tải model tự động từ HuggingFace Hub"
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    unset HF_DATASETS_OFFLINE
fi

# --------------------------------------------------------------------------
# Base paths (overridable)
BASE_MODELS="${BASE_MODELS:-/storage-voice/voice/vdt/baottn/duplex-model-dir}"
BASE_DATA="${BASE_DATA:-/storage-voice/voice/vdt/baottn/duplex-data}"

# Models (overridable)
export DUPLEX_MODEL_DIR="${DUPLEX_MODEL_DIR:-$BASE_MODELS}"
export DIALOGUESIDON_MODEL_PATH="${DIALOGUESIDON_MODEL_PATH:-$BASE_MODELS/DialogueSidon}"
export DNSMOS_DIR="${DNSMOS_DIR:-$BASE_MODELS/dnsmos}"
export SPEECHBRAIN_MODEL_PATH="${SPEECHBRAIN_MODEL_PATH:-$BASE_MODELS/spkrec-ecapa-voxceleb}"
export NISQA_MODEL_PATH="${NISQA_MODEL_PATH:-$BASE_MODELS/nisqa.tar}"
export SQUIM_MODEL_PATH="${SQUIM_MODEL_PATH:-$BASE_MODELS/squim_objective_dns2020.pth}"
export WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-/raid/voice/chauhn3/whisper/whisper-large-v3}"

# Data directories (overridable)
SOURCE="${SOURCE:-youtube}"
CRAWL_DIR="${CRAWL_DIR:-$BASE_DATA/crawl/}"
RAW_DIR="${RAW_DIR:-$BASE_DATA/raw/$SOURCE}"
DIALOGUE_DIR="${DIALOGUE_DIR:-$BASE_DATA/processed/dialogue/$SOURCE}"
DUPLEX_OUT_DIR="${DUPLEX_OUT_DIR:-$BASE_DATA/processed/duplexchat/$SOURCE}"
SOMMELIER_OUT_DIR="${SOMMELIER_OUT_DIR:-$BASE_DATA/processed/sommelier/$SOURCE}"
