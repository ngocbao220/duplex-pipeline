#!/usr/bin/env bash

# ==============================================================================
# Script thực thi Pipeline Duplex-Pipelines (DuplexChat & Sommelier)
# ==============================================================================

set -e

# --- Cấu hình Môi trường ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 1. Cấu hình Môi trường thực thi: dev (tải online từ HuggingFace) hoặc sever (offline, bắt buộc local)
export MODE="${MODE:-sever}"   # "dev" (huggingface) hoặc "sever" (local)
export GPU_ID="${GPU_ID:-2}"

if [ "$MODE" = "sever" ] || [ "$MODE" = "server" ] || [ "$MODE" = "offline" ]; then
    echo "[MODE: SEVER] Chạy chế độ Offline (chỉ dùng model local disk, cấm tải mạng)"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    echo "[MODE: DEV] Chạy chế độ Dev (cho phép tải model tự động từ HuggingFace Hub)"
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    unset HF_DATASETS_OFFLINE
fi

# 2. Base paths dùng chung
BASE_MODELS="/storage-voice/voice/vdt/baottn/duplex-model-dir"
BASE_DATA="/storage-voice/voice/vdt/baottn/duplex-data"

# 3. Models
export DUPLEX_MODEL_DIR="$BASE_MODELS"
export DIALOGUESIDON_MODEL_PATH="$BASE_MODELS/DialogueSidon"
export DNSMOS_DIR="$BASE_MODELS/dnsmos"
export WHISPER_MODEL_PATH="/raid/voice/chauhn3/whisper/whisper-large-v3"
export SPEECHBRAIN_MODEL_PATH="/storage-voice/voice/vdt/baottn/duplex-model-dir/spkrec-ecapa-voxceleb"

# 4. Data directories
CRAWL_DIR="$BASE_DATA/crawl/"
RAW_DIR="$BASE_DATA/raw/"

SOURCE="youtube"
RAW_DIR="$BASE_DATA/raw/$SOURCE"
DIALOGUE_DIR="$BASE_DATA/processed/dialogue/$SOURCE"
DUPLEX_OUT_DIR="$BASE_DATA/processed/duplexchat/$SOURCE"
SOMMELIER_OUT_DIR="$BASE_DATA/processed/sommelier/$SOURCE"

# ==========================================
# 3. DUPLEXCHAT: Benchmark Stereo
# ==========================================
echo "======================================================================"
echo " 3. DUPLEXCHAT: BENCHMARK STEREO (Đánh giá chất lượng âm thanh)"
echo "======================================================================"

python scripts/benchmark_stereo.py \
    --corpus "$DUPLEX_OUT_DIR" \
    --check-models \
    --dnsmos-model-dir "$DNSMOS_DIR" \
    --output-dir "outputs/benchmark/duplexchat_$SOURCE"

# ==========================================
# 4. Report Data Retention
# ==========================================
echo "======================================================================"
echo " 4. REPORT DATA RETENTION (Báo cáo tỷ lệ dữ liệu giữ lại)"
echo "======================================================================"

python scripts/report_data_retention.py \
    --raw-dir "$RAW_DIR" \
    --processed-dir "$BASE_DATA/processed" \
    --output-md "outputs/data_retention_report.md"