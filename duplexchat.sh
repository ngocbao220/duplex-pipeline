#!/usr/bin/env bash
# duplexchat.sh — Bước 2: Tách kênh thoại 2 người bằng DialogueSidon
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     DIALOGUE_DIR=/path/to/dialogue \
#     DUPLEX_OUT_DIR=/path/to/duplexchat \
#     bash duplexchat.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step separate_dialogue \
    --input-dir "$DIALOGUE_DIR" \
    --output-dir "$DUPLEX_OUT_DIR" \
    --separation-chunk 60 \
    --gpu "$GPU_ID"
