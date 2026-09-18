#!/usr/bin/env bash
# dialogue_filter.sh — Bước 0+1: Convert WAV + Dialogue Filtering
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     RAW_DIR=/path/to/raw \
#     DIALOGUE_DIR=/path/to/dialogue \
#     bash dialogue_filter.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

if [ -d "${CRAWL_DIR:-}" ] && [ -n "$(ls -A "$CRAWL_DIR" 2>/dev/null)" ]; then
    python "$PROJECT_ROOT/scripts/convert_crawl_to_raw.py" \
        --crawl-dir "$CRAWL_DIR" \
        --raw-dir "$BASE_DATA/raw"
fi

python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step split_dialogue \
    --input-dir "$RAW_DIR" \
    --output-dir "$DIALOGUE_DIR" \
    --diarization-backend sortformer \
    --no-filter-music \
    --diarization-model nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --lid vi \
    --gpu "$GPU_ID"
