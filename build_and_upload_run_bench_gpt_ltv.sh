#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------
# Helper to load a .env file (KEY=value, with optional quotes)
# -----------------------------------------------------------------
load_env_file() {
    local env_file="$1"
    if [ ! -f "$env_file" ]; then
        return 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # Match KEY=value, allowing spaces around = and optional quotes
        if [[ "$line" =~ ^[[:space:]]*([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Remove surrounding single or double quotes
            if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi
            export "$key"="$value"
        fi
    done < "$env_file"
    return 0
}

# -----------------------------------------------------------------
# 1. Load token from .env (script dir first, then $HOME)
# -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! load_env_file "$SCRIPT_DIR/.env"; then
    load_env_file "$HOME/.env" || true
fi

if [ -z "$LIGHTNING_AI_REMOTE_DEST" ]; then
    echo "❌ Error: LIGHTNING_AI_REMOTE_DEST not found in $SCRIPT_DIR/.env or $HOME/.env"
    exit 1
fi
# -----------------------------------------------------------------


# ----------------------------
# Defaults
# ----------------------------
BUILD_RUST=true
BUILD_CUDA=true
BUILD_PYTHON=true

# ----------------------------
# Parse Arguments
# ----------------------------
usage() {
    echo "Usage: $0 [--no-rust] [--no-cuda] [--no-python]"
    exit 1
}

while (( "$#" )); do
    case "$1" in
        --no-rust)
            BUILD_RUST=false
            shift
            ;;
        --no-cuda)
            BUILD_CUDA=false
            shift
            ;;
        --no-python)
            BUILD_PYTHON=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# ----------------------------
# Configuration
# ----------------------------
REMOTE_DEST="$LIGHTNING_AI_REMOTE_DEST"
SSH_HOST="${REMOTE_DEST%:*}"

PYTHON_VER="3.10"
RUST_WHEEL_DIR="rust_ewma/target/wheels"
DIST_DIR="dist"

REMOTE_CPP_DIR="/teamspace/studios/this_studio/cpp_ewma"

# ----------------------------
# Build Rust extension
# ----------------------------
if [ "$BUILD_RUST" = true ]; then
    echo "🚀 Building Rust extension for Linux..."
    (
        cd rust_ewma/
        uv run maturin build \
            --features python \
            --release \
            --target x86_64-unknown-linux-gnu \
            --zig \
            --interpreter "python${PYTHON_VER}"
    )
else
    echo "⏭️  Skipping Rust extension build..."
fi

# ----------------------------
# Sync & Build CPP / CUDA
# ----------------------------
if [ "$BUILD_CUDA" = true ]; then
    echo "🚀 Syncing CPP source..."
    rsync -avP \
        cpp_ewma/.python-version \
        cpp_ewma/pyproject.toml \
        cpp_ewma/setup.py \
        cpp_ewma/remote_build.sh \
        "$REMOTE_DEST/cpp_ewma/"

    rsync -avP \
        cpp_ewma/src/bindings.cpp \
        cpp_ewma/src/ltv_scan_common.cuh \
        cpp_ewma/src/fused_scan_kernel.cu \
        cpp_ewma/src/fused_scan.cpp \
        cpp_ewma/src/fused_concat_register_cast_scan_kernel.cu \
        cpp_ewma/src/fused_concat_head_major_register_cast_scan_kernel.cu \
        cpp_ewma/src/fused_concat_scan.cpp \
        cpp_ewma/src/plane_cuda_kernel.cu \
        cpp_ewma/src/plane_cuda.cpp \
        cpp_ewma/src/ltv_fused_concat_register_cast_blelloch_scan_cuda.cu \
        cpp_ewma/src/ltv_fused_concat_register_cast_blelloch_scan_cuda_bindings.cpp \
        cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda.cu \
        cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_bindings.cu \
        cpp_ewma/src/fused_linear_transpose_forward.cu \
        cpp_ewma/src/fused_linear_transpose_forward_bindings.cpp \
        cpp_ewma/src/fused_linear_transpose_backward.cu \
        cpp_ewma/src/fused_linear_transpose_backward_bindings.cpp \
        "$REMOTE_DEST/cpp_ewma/src/"

    echo "🏗️  Starting Remote CUDA Build..."
    ssh -tt "$SSH_HOST" \
    "tmux new-session -A -s cuda_build 'bash /teamspace/studios/this_studio/cpp_ewma/remote_build.sh |& tee /teamspace/studios/this_studio/cpp_ewma/remote_build.log'"
else
    echo "⏭️  Skipping Remote CUDA Build..."
fi

# ----------------------------
# Build Python wheel
# ----------------------------
if [ "$BUILD_PYTHON" = true ]; then
    echo "📦 Building nanochat Python package..."
    uv build --wheel
else
    echo "⏭️  Skipping nanochat Python package build..."
fi

# ----------------------------
# Upload wheels + benchmark script
# ----------------------------
echo "📡 Syncing Artifacts to Lightning AI..."

RSYNC_CMD="rsync -avP"
if [ "$BUILD_PYTHON" = true ]; then
    RSYNC_CMD="$RSYNC_CMD $DIST_DIR/nanochat-0.1.0-py3-none-any.whl"
fi

# Always sync the Rust wheel (built or cached) and the benchmark script
RSYNC_CMD="$RSYNC_CMD $RUST_WHEEL_DIR/rust_ewma-*.whl run_bench_gpt_ltv.sh $REMOTE_DEST"

eval $RSYNC_CMD

echo "✅ Done!"
[ "$BUILD_CUDA" = true ] && echo "Remote CUDA build running in tmux session: cuda_build"
