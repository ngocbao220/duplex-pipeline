#!/usr/bin/env bash
# ==============================================================================
# dialogue_filter.sh — Bước 0+1: Convert WAV + Dialogue Filtering
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     RAW_DIR=/path/to/raw \
#     DIALOGUE_DIR=/path/to/dialogue \
#     bash dialogue_filter.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

# --------------------------------------------------------------------------
# ANSI colors đã được export từ env.sh
# --------------------------------------------------------------------------

echo ""
echo -e "${C_BOLD}=============== Dialogue Filtering ===============${C_RESET}"
log_info "Dataset  : $RAW_DIR"
log_info "Output   : $DIALOGUE_DIR"
log_info "Device   : GPU $GPU_ID"
echo ""

# --------------------------------------------------------------------------
# 0. Convert Crawl → WAV (16kHz mono)
# --------------------------------------------------------------------------
echo -e "${C_INFO}[INFO]${C_RESET} ${C_BOLD}0. Preprocessing — Convert crawl to WAV${C_RESET}"
echo -e "${C_INFO}[INFO]${C_RESET}    Crawl dir : $CRAWL_DIR"
echo -e "${C_INFO}[INFO]${C_RESET}    Raw dir   : $BASE_DATA/raw"

_T0=$(date +%s%N)
python "$PROJECT_ROOT/scripts/convert_crawl_to_raw.py" \
    --crawl-dir "$CRAWL_DIR" \
    --raw-dir "$BASE_DATA/raw"
_T1=$(date +%s%N)
_DT=$(( (_T1 - _T0) / 1000000 ))
log_info "Preprocessing done. Time: ${_DT}ms"

# --------------------------------------------------------------------------
# 1. Split Dialogue — Diarization + LID filter
# --------------------------------------------------------------------------
echo ""
echo -e "${C_INFO}[INFO]${C_RESET} ${C_BOLD}1. Split Dialogue${C_RESET}"
echo -e "${C_INFO}[INFO]${C_RESET}    Diarization  : nvidia-sortformer-4spk-streaming-v2.1"
echo -e "${C_INFO}[INFO]${C_RESET}    Mode         : $([ "$MODE" = "sever" ] && echo 'local' || echo 'HuggingFace Hub')"
echo -e "${C_INFO}[INFO]${C_RESET}    LID filter   : vi (Whisper LID)"
echo -e "${C_INFO}[INFO]${C_RESET}    Music filter : DISABLED (--no-filter-music)"
echo ""

_T0=$(date +%s%N)
python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step split_dialogue \
    --input-dir "$RAW_DIR" \
    --output-dir "$DIALOGUE_DIR" \
    --diarization-backend sortformer \
    --no-filter-music \
    --diarization-model nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --lid vi \
    --gpu "$GPU_ID"
_T1=$(date +%s%N)
_DT=$(( (_T1 - _T0) / 1000000 ))

echo ""
log_info "====> Dialogue filtering complete."
log_info "1. Preprocessing            : done"
log_info "2. Split dialogue           : ${_DT}ms"
log_info "3. Remove music background  : SKIPPED (--no-filter-music)"
log_info "Result saved to: $DIALOGUE_DIR"
