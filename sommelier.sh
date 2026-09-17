#!/usr/bin/env bash
# ==============================================================================
# sommelier.sh — Bước 2 (Sommelier): Overlap detection + Separation + Reconstruction
#
# Cách dùng:
#   MODE=dev GPU_ID=0 \
#     RAW_DIR=/path/to/raw \
#     SOMMELIER_OUT_DIR=/path/to/sommelier \
#     bash sommelier.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

# --------------------------------------------------------------------------
echo ""
echo -e "${C_BOLD}=============== Sommelier ===============${C_RESET}"
echo -e "${C_INFO}[INFO]${C_RESET}  Input  : $RAW_DIR"
echo -e "${C_INFO}[INFO]${C_RESET}  Output : $SOMMELIER_OUT_DIR"
echo -e "${C_INFO}[INFO]${C_RESET}  Device : GPU $GPU_ID"
echo ""

# Overlap detection
echo -e "${C_INFO}[INFO]${C_RESET}  Overlap detection : Diarization by \"sortformer\" (load from ${MODE} — ${DUPLEX_MODEL_DIR})"

# Overlap separation
echo -e "${C_INFO}[INFO]${C_RESET}  Overlap separation: SepReformer (load from ${MODE} — ${DUPLEX_MODEL_DIR})"
echo ""

_T0=$(date +%s%N)
python "$PROJECT_ROOT/scripts/run_pipeline_batch.py" \
    --step sommelier \
    --input-dir "$RAW_DIR" \
    --output-dir "$SOMMELIER_OUT_DIR" \
    --gpu "$GPU_ID"
_T1=$(date +%s%N)
_DT=$(( (_T1 - _T0) / 1000000 ))
_SEC=$(echo "scale=2; ${_DT}/1000" | bc)

echo ""
log_info "====> Done. Result saved to: $SOMMELIER_OUT_DIR"
log_info "Time: ${_SEC}s"
