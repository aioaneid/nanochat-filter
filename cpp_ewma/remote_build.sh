#!/usr/bin/env bash
set -e

# cd /teamspace/studios/this_studio
# uv pip install -e ".[gpu]"

cd /teamspace/studios/this_studio/cpp_ewma

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr ' ' _)
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

export TORCH_CUDA_ARCH_LIST=$CUDA_ARCH

OUT_DIR=target/wheels/$GPU_NAME
mkdir -p $OUT_DIR

echo "GPU: $GPU_NAME"
echo "CUDA arch: $CUDA_ARCH"
echo "Wheel dir: $OUT_DIR"

export UV_CACHE_DIR="/teamspace/studios/this_studio/.uv_cache/${GPU_NAME}"

uv build --wheel --out-dir $OUT_DIR -v
