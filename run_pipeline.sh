#!/usr/bin/env bash

# ==============================================================================
# Script thực thi Pipeline Duplex-Pipelines (DuplexChat & Sommelier)
# ==============================================================================

set -e

# A shared ID makes split/separation timing accumulate in one inspectable report.
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

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

# 2. Chọn pipeline: duplexchat / sommelier
PIPELINE="duplexchat"

# 3. Base paths dùng chung
BASE_MODELS="/storage-voice/voice/vdt/baottn/duplex-model-dir"
BASE_DATA="/storage-voice/voice/vdt/baottn/duplex-data"

# 4. Models
export DUPLEX_MODEL_DIR="$BASE_MODELS"
export DIALOGUESIDON_MODEL_PATH="$BASE_MODELS/DialogueSidon"
export DNSMOS_DIR="$BASE_MODELS/dnsmos"
export SPEECHBRAIN_MODEL_PATH="$BASE_MODELS/spkrec-ecapa-voxceleb"
export NISQA_MODEL_PATH="$BASE_MODELS/nisqa.tar"
export SQUIM_MODEL_PATH="$BASE_MODELS/squim_objective_dns2020.pth"
export WHISPER_MODEL_PATH="/raid/voice/chauhn3/whisper/whisper-large-v3"

# 5. Data directories
CRAWL_DIR="$BASE_DATA/crawl/"

SOURCE="youtube"
RAW_DIR="$BASE_DATA/raw/$SOURCE"
DIALOGUE_DIR="$BASE_DATA/processed/dialogue/$SOURCE"
DUPLEX_OUT_DIR="$BASE_DATA/processed/duplexchat/$SOURCE"
SOMMELIER_OUT_DIR="$BASE_DATA/processed/sommelier/$SOURCE"

echo "======================================================================"
echo " 0. CONVERT TO WAV"
echo "======================================================================"

python scripts/convert_crawl_to_raw.py \
    --crawl-dir "$CRAWL_DIR" \
    --raw-dir "$BASE_DATA/raw"

echo "======================================================================"
echo " 1. SPLIT DIALOGUE (Tách cuộc thoại & Lọc nhạc & Lọc Tiếng Việt Whisper LID)"
echo "======================================================================"

python scripts/run_pipeline_batch.py \
    --step split_dialogue \
    --input-dir "$RAW_DIR" \
    --output-dir "$DIALOGUE_DIR" \
    --diarization-backend sortformer \
    --no-filter-music \
    --diarization-model nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --lid vi \
    --gpu "$GPU_ID"

# ==============================================================================
# PIPELINE
# ==============================================================================

if [ "$PIPELINE" = "duplexchat" ]; then

    echo "======================================================================"
    echo " 2. DUPLEXCHAT: SEPARATE DIALOGUE (Tách kênh thoại 2 người)"
    echo "======================================================================"

    python scripts/run_pipeline_batch.py \
        --step separate_dialogue \
        --input-dir "$DIALOGUE_DIR" \
        --output-dir "$DUPLEX_OUT_DIR" \
        --separation-chunk 60 \
        --gpu "$GPU_ID"

    echo "======================================================================"
    echo " 3. DUPLEXCHAT: BENCHMARK STEREO"
    echo "======================================================================"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/benchmark_stereo.py \
        --corpus "$DUPLEX_OUT_DIR" \
        --check-models \
        --dnsmos-model-dir "$DNSMOS_DIR" \
        --output-dir "outputs/benchmark/duplexchat_$SOURCE"

elif [ "$PIPELINE" = "sommelier" ]; then

    echo "======================================================================"
    echo " 2. SOMMELIER PIPELINE"
    echo "======================================================================"

    python scripts/run_pipeline_batch.py \
        --step sommelier \
        --input-dir "$RAW_DIR" \
        --output-dir "$SOMMELIER_OUT_DIR" \
        --gpu "$GPU_ID"

    echo "======================================================================"
    echo " 3. SOMMELIER: BENCHMARK STEREO"
    echo "======================================================================"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/benchmark_stereo.py \
        --corpus "$SOMMELIER_OUT_DIR" \
        --check-models \
        --dnsmos-model-dir "$DNSMOS_DIR" \
        --output-dir "outputs/benchmark/sommelier_$SOURCE"

else
    echo "Unknown PIPELINE: $PIPELINE"
    echo "Use: PIPELINE=duplexchat or PIPELINE=sommelier"
    exit 1
fi
