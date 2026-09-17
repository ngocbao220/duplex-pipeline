#!/usr/bin/env bash
# ==============================================================================
# benchmark.sh — Benchmark stereo quality + Data retention report
#
# Cách dùng:
#   MODE=dev GPU_ID=0 PIPELINE=duplexchat \
#     DUPLEX_OUT_DIR=/path/to/duplexchat \
#     SOMMELIER_OUT_DIR=/path/to/sommelier \
#     bash benchmark.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

PIPELINE="${PIPELINE:-duplexchat}"

# --------------------------------------------------------------------------
if [ "$PIPELINE" = "duplexchat" ]; then
    CORPUS_DIR="$DUPLEX_OUT_DIR"
    BENCH_TAG="duplexchat_${SOURCE}"
elif [ "$PIPELINE" = "sommelier" ]; then
    CORPUS_DIR="$SOMMELIER_OUT_DIR"
    BENCH_TAG="sommelier_${SOURCE}"
else
    log_error "Unknown PIPELINE: $PIPELINE. Use: duplexchat | sommelier"
    exit 1
fi

echo ""
echo -e "${C_BOLD}=============== Benchmark Stereo [${PIPELINE}] ===============${C_RESET}"
log_info "Corpus  : $CORPUS_DIR"
log_info "Output  : outputs/benchmark/${BENCH_TAG}"
log_info "Device  : GPU $GPU_ID"
echo ""

_T0=$(date +%s%N)
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$PROJECT_ROOT/scripts/benchmark_stereo.py" \
    --corpus "$CORPUS_DIR" \
    --check-models \
    --dnsmos-model-dir "$DNSMOS_DIR" \
    --output-dir "outputs/benchmark/${BENCH_TAG}"
_T1=$(date +%s%N)
_DT=$(( (_T1 - _T0) / 1000000 ))
_SEC=$(echo "scale=2; ${_DT}/1000" | bc)
log_info "Benchmark done. Time: ${_SEC}s"

# --------------------------------------------------------------------------
echo ""
echo -e "${C_BOLD}=============== Data Retention Report ===============${C_RESET}"
log_info "Raw dir       : $RAW_DIR"
log_info "Processed dir : $BASE_DATA/processed"
log_info "Report output : outputs/data_retention_report.md"
echo ""

python "$PROJECT_ROOT/scripts/report_data_retention.py" \
    --raw-dir "$RAW_DIR" \
    --processed-dir "$BASE_DATA/processed" \
    --output-md "outputs/data_retention_report.md"

log_info "Report saved to: outputs/data_retention_report.md"
