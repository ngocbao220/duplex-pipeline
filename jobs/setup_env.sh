#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh — Tự động thiết lập môi trường chuẩn, cách ly chống xung đột
#
# Cách dùng:
#   bash setup_env.sh --duplexchat    # Tạo Conda env 'duplexchat' (Phase 1, 2-DuplexChat, 3-Benchmark)
#   bash setup_env.sh --sommelier     # Tạo Conda env 'sommelier' (Phase 2-Sommelier / SepReformer)
#   bash setup_env.sh --all           # Tạo cả 2 Conda envs cách ly
#   bash setup_env.sh --uv            # Tạo môi trường cô lập bằng uv cho từng pipeline
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/conda/bin:/opt/anaconda3/bin:/opt/miniconda3/bin:$PATH"

# ANSI Colors
C_RESET='\033[0m'
C_CYAN='\033[0;36m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_RED='\033[1;31m'
C_BOLD='\033[1m'

log_info()  { echo -e "${C_CYAN}[INFO]${C_RESET}  $*"; }
log_ok()    { echo -e "${C_GREEN}[OK]${C_RESET}    $*"; }
log_warn()  { echo -e "${C_YELLOW}[WARN]${C_RESET}  $*"; }
log_error() { echo -e "${C_RED}[ERROR]${C_RESET} $*"; }

install_system_deps() {
    log_info "Kiểm tra thư viện hệ thống (ffmpeg, libsndfile)..."
    if command -v ffmpeg &>/dev/null; then
        log_ok "ffmpeg đã được cài đặt: $(which ffmpeg)"
    else
        log_warn "Chưa tìm thấy ffmpeg. Vui lòng cài đặt: 'apt-get install -y ffmpeg' hoặc 'brew install ffmpeg'"
    fi
}

setup_conda_duplexchat() {
    log_info "=== Thiết lập Conda Environment [duplexchat] (Python 3.12) ==="
    if ! command -v conda &>/dev/null; then
        log_error "Không tìm thấy conda trong PATH. Vui lòng cài Miniconda hoặc Anaconda."
        return 1
    fi

    if conda info --envs | grep -q "duplexchat"; then
        log_info "Conda env 'duplexchat' đã tồn tại. Đang cập nhật gói..."
    else
        log_info "Đang tạo conda env 'duplexchat' với Python 3.12..."
        conda create -n duplexchat python=3.12 -y
    fi

    log_info "Cài đặt dependencies cho DuplexChat từ requirements-duplexchat.txt..."
    conda run -n duplexchat pip install --upgrade pip
    conda run -n duplexchat pip install -r "$PROJECT_ROOT/requirements-duplexchat.txt"

    log_info "Cài đặt NVIDIA NeMo Toolkit (Sortformer Diarization)..."
    conda run -n duplexchat pip install Cython packaging
    conda run -n duplexchat pip install "nemo_toolkit[asr]==2.3.0"

    log_ok "Môi trường [duplexchat] đã sẵn sàng!"
}

setup_conda_sommelier() {
    log_info "=== Thiết lập Conda Environment [sommelier] (Python 3.11/3.12) ==="
    if ! command -v conda &>/dev/null; then
        log_error "Không tìm thấy conda trong PATH."
        return 1
    fi

    if conda info --envs | grep -q "sommelier"; then
        log_info "Conda env 'sommelier' đã tồn tại. Đang cập nhật gói..."
    else
        log_info "Đang tạo conda env 'sommelier' với Python 3.11..."
        conda create -n sommelier python=3.11 -y
    fi

    log_info "Cài đặt dependencies cho Sommelier từ requirements-sommelier.txt..."
    conda run -n sommelier pip install --upgrade pip
    conda run -n sommelier pip install -r "$PROJECT_ROOT/requirements-sommelier.txt"

    log_ok "Môi trường [sommelier] đã sẵn sàng!"
}

setup_uv_environments() {
    log_info "=== Thiết lập môi trường cô lập độc lập qua 'uv' ==="
    if ! command -v uv &>/dev/null; then
        log_error "Chưa tìm thấy 'uv'. Hãy cài đặt uv: 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        return 1
    fi

    log_info "Syncing pipeline/duplexchat..."
    UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache" uv sync --project "$PROJECT_ROOT/pipeline/duplexchat"

    log_info "Syncing pipeline/sommelier..."
    UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache" uv sync --project "$PROJECT_ROOT/pipeline/sommelier"

    log_ok "Môi trường uv cho các pipeline đã hoàn tất cách ly!"
}

# --- Main CLI ---
install_system_deps

TARGET="${1:-}"
if [[ "$TARGET" == "--duplexchat" ]]; then
    setup_conda_duplexchat
elif [[ "$TARGET" == "--sommelier" ]]; then
    setup_conda_sommelier
elif [[ "$TARGET" == "--uv" ]]; then
    setup_uv_environments
elif [[ "$TARGET" == "--all" ]]; then
    setup_conda_duplexchat
    setup_conda_sommelier
else
    echo -e "${C_BOLD}Sử dụng:${C_RESET}"
    echo "  bash setup_env.sh --duplexchat   # Tạo conda env duplexchat (Phase 1, 2-DuplexChat, 3)"
    echo "  bash setup_env.sh --sommelier    # Tạo conda env sommelier (Phase 2-Sommelier)"
    echo "  bash setup_env.sh --all          # Tạo cả hai conda envs"
    echo "  bash setup_env.sh --uv           # Cài đặt qua uv projects"
fi
