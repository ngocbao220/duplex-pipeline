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

# 1. Cấu hình Offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GPU_ID="2"

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

python -m scripts.benchmark_stereo \
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