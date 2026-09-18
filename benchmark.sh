#!/usr/bin/env bash
# benchmark.sh — Benchmark stereo quality + Data retention report
#
# Cách dùng:
#   MODE=dev GPU_ID=0 PIPELINE=duplexchat \
#     DUPLEX_OUT_DIR=/path/to/duplexchat \
#     bash benchmark.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

PIPELINE="${PIPELINE:-duplexchat}"

if [ "$PIPELINE" = "duplexchat" ]; then
    CORPUS_DIR="$DUPLEX_OUT_DIR"
    BENCH_TAG="duplexchat_${SOURCE}"
elif [ "$PIPELINE" = "sommelier" ]; then
    CORPUS_DIR="$SOMMELIER_OUT_DIR"
    BENCH_TAG="sommelier_${SOURCE}"
else
    echo "[ERROR] Unknown PIPELINE: $PIPELINE. Use: duplexchat | sommelier" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$PROJECT_ROOT/scripts/benchmark_stereo.py" \
    --corpus "$CORPUS_DIR" \
    --check-models \
    --dnsmos-model-dir "$DNSMOS_DIR" \
    --output-dir "outputs/benchmark/${BENCH_TAG}"

python "$PROJECT_ROOT/scripts/report_data_retention.py" \
    --raw-dir "$RAW_DIR" \
    --processed-dir "$BASE_DATA/processed" \
    --output-md "outputs/data_retention_report.md"
