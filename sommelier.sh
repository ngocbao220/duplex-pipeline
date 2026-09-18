#!/usr/bin/env bash
# sommelier.sh — Bước 2 (Sommelier): Overlap detection + Separation + Reconstruction
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     DIALOGUE_DIR=/path/to/dialogue \
#     SOMMELIER_OUT_DIR=/path/to/sommelier \
#     bash sommelier.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step sommelier \
    --input-dir "$DIALOGUE_DIR" \
    --output-dir "$SOMMELIER_OUT_DIR" \
    --gpu "$GPU_ID"

