#!/usr/bin/env bash
# ==============================================================================
# duplexchat.sh — Bước 2: Tách kênh thoại 2 người bằng DialogueSidon
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     DIALOGUE_DIR=/path/to/dialogue \
#     DUPLEX_OUT_DIR=/path/to/duplexchat \
#     bash duplexchat.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

# --------------------------------------------------------------------------
echo ""
echo -e "${C_BOLD}=============== DuplexChat ===============${C_RESET}"
echo -e "${C_INFO}[INFO]${C_RESET}  Separation by \"DialogueSidon\" (load from ${MODE} — ${DIALOGUESIDON_MODEL_PATH})"
echo -e "${C_INFO}[INFO]${C_RESET}  + Output sample rate: 24kHz"
echo -e "${C_INFO}[INFO]${C_RESET}  Input  : $DIALOGUE_DIR"
echo -e "${C_INFO}[INFO]${C_RESET}  Output : $DUPLEX_OUT_DIR"
echo -e "${C_INFO}[INFO]${C_RESET}  Device : GPU $GPU_ID"
echo ""

_T0=$(date +%s%N)
python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step separate_dialogue \
    --input-dir "$DIALOGUE_DIR" \
    --output-dir "$DUPLEX_OUT_DIR" \
    --separation-chunk 60 \
    --gpu "$GPU_ID"
_T1=$(date +%s%N)
_DT=$(( (_T1 - _T0) / 1000000 ))
_SEC=$(echo "scale=2; ${_DT}/1000" | bc)

echo ""
log_info "====> Done. Result saved to: $DUPLEX_OUT_DIR"
log_info "Time: ${_SEC}s"
