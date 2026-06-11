#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Argument Parsing
# ----------------------------
RUN_BENCH=false
RUN_TRAIN=false
RUN_TEST_STANDARD=false
RUN_TEST_HEAD_MAJOR=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --bench)
      RUN_BENCH=true
      shift
      ;;
    --base_train)
      RUN_TRAIN=true
      shift
      ;;
    --test-standard)
      RUN_TEST_STANDARD=true
      shift
      ;;
    --test-head-major)
      RUN_TEST_HEAD_MAJOR=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--bench] [--base_train] [--test-standard] [--test-head-major]"
      exit 1
      ;;
  esac
done

# ----------------------------
# Configuration
# ----------------------------
PYTHON_VERSION="3.10"
VENV=".venv-py310"
LOG_FILE="bench-gpt-ltv.log"
PY="$VENV/bin/python"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Benchmark parameters
START_T="${START_T:-1024}"
END_T="${END_T:-0}"
STEP_T="${STEP_T:--1024}"
SEQUENCE_LEN="${SEQUENCE_LEN:-1024}"
H="${H:-8}"
# L4 accepts B=16 but not B=32
B="${B:-4}"
R="${R:-128}"
N_LAYER="${N_LAYER:-12}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-1}"

PYTORCH_MAX_T=$(($START_T / 10))

# ----------------------------
# Detect GPU
# ----------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr ' ' _)"

# ----------------------------
# Detect local wheels
# ----------------------------
NANOCHAT_WHEEL="nanochat-0.1.0-py3-none-any.whl"
RUST_EWMA_WHEEL="rust_ewma-0.1.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
CPP_EWMA_WHEEL="cpp_ewma-0.1.0-cp310-cp310-linux_x86_64.whl"

export UV_CACHE_DIR="/teamspace/studios/this_studio/.uv_cache/${GPU_NAME}"

# ----------------------------
# Environment Setup
# ----------------------------
if ! uv python list | grep -q "cpython-${PYTHON_VERSION}"; then
    echo "Installing Python ${PYTHON_VERSION} via uv"
    uv pip install --python "$PY" "$PYTHON_VERSION"
fi

if [ ! -d "$VENV" ]; then
    uv venv --python "$PYTHON_VERSION" "$PWD/$VENV"
fi

echo "Installing dependencies..."
uv pip install --python "$PY" --upgrade torch --index-url "$TORCH_INDEX_URL"
uv pip install --python "$PY" "$RUST_EWMA_WHEEL"
uv pip install --python "$PY" "cpp_ewma/target/wheels/$GPU_NAME/$CPP_EWMA_WHEEL"
uv pip install --python "$PY" "$NANOCHAT_WHEEL"

export TORCH_USE_CUDA_DSA=0

# Enable all kernels by default for standard builds
export BUILD_LTV_FUSED_SCAN=1
export BUILD_LTV_FUSED_CONCAT_SCAN=1
export BUILD_LTV_FUSED_CONCAT_HM_SCAN=1
export BUILD_LTV_FUSED_CONCAT_BLELLOCH=1
export BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH=1
export BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL=1
export BUILD_LTV_PLANE=1

# ----------------------------
# Run tests
# ----------------------------
if [ "$RUN_TEST_STANDARD" = true ]; then
    echo "Running standard LTV CUDA tests..."
    uv run --python "$PY" pytest tests/test_ltv_cuda_fused_concat_register_cast_blelloch_scan.py --benchmark=false --sigmoid=false --check_against_expected=true
fi

if [ "$RUN_TEST_HEAD_MAJOR" = true ]; then
    echo "Running head-major LTV CUDA tests..."
    uv run --python "$PY" pytest tests/test_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan.py --benchmark=false --sigmoid=false --check_against_expected=true
fi

# ----------------------------
# Run benchmark
# ----------------------------
if [ "$RUN_BENCH" = true ]; then
    echo "Running benchmark..."
    PYTHONUNBUFFERED=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=".torch_cache/torchinductor-$GPU_NAME" \
    TRITON_CACHE_DIR=".torch_cache/triton-$GPU_NAME" \
    uv run \
        --python "$PY" -m nanochat.bench.bench_gpt_ltv \
        --start_t "$START_T" \
        --end_t "$END_T" \
        --step_t "$STEP_T" \
        --sequence_len "$SEQUENCE_LEN" \
        --h "$H" \
        --b "$((B / 4))" \
        --r "$R" \
        --n_layer "$N_LAYER" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --pytorch_max_t "$PYTORCH_MAX_T" \
        --no-original \
        --no-numpy \
        --no-dlpack \
        --no-original \
        --no-rust_cpu \
        --verify \
        --verify_history \
        --bfloat16 \
        --no-ltv_fused_py_torch \
        --no-concat_original \
        --ltv_concat_fused_scan_triton \
        --ltv_concat_fused_scan_cuda \
        --no-ltv_concat_register_cast_scan_cuda \
        --ltv_fused_concat_register_cast_scan_cuda \
        --plane_max_cuda \
        --no-associative_scan_triton \
        |& tee -a "$LOG_FILE"
fi

case "$B" in
    1) NUM_ITERATIONS=2 ;;
    *) NUM_ITERATIONS=4 ;;
esac

# ----------------------------
# Base Train
# ----------------------------
if [ "$RUN_TRAIN" = true ]; then
    echo "Starting base training..."
    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv-${GPU_NAME}"
    mkdir -p "$NANOCHAT_BASE_DIR"
    
    # Check if source exists before symlinking
    if [ -d "$HOME/.cache/nanochat.base.240" ]; then
        ln -sfn "$HOME/.cache/nanochat.base.240"/* "$NANOCHAT_BASE_DIR"/
    fi

    WANDB_RUN=tensorboard
    COMBINED_OPTIMIZERS="true"
    DEVICE_BATCH_SIZE=$B

    # Model specifications
    options=()

    options+=("depth=11 model_type=ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    options+=("depth=12 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false")

    # options+=("depth=11 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    # options+=("depth=11 model_type=ltv_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_fused_concat_head_major_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false")

    # options+=("depth=11 model_type=ltv_cuda_fused_concat_register_cast_blelloch_scan ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    # options+=("depth=11 model_type=ltv_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    # options+=("depth=11 model_type=ltv_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    # options+=("depth=11 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=10 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=10 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=9 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=9 model_type=ltv_fused_concat_register_cast_scan_cuda ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    # options+=("depth=10 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false")
    # options+=("depth=9 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false")

    # options+=("depth=11 model_type=ltv_concat_fused_scan_triton ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_concat_register_cast_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=9 model_type=ltv_concat_fused_scan_triton ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=9 model_type=ltv_concat_register_cast_scan_cuda ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_concat_fused_scan_cuda ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=9 model_type=ltv_concat_fused_scan_cuda ltv_r=1 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")
    # options+=("depth=11 model_type=ltv_plane_cuda_max ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true")

    mkdir -p "$HOME/tmp"

    # Save original stdout/stderr descriptors
    exec 3>&1 4>&2

    for opt in "${options[@]}"; do
        eval "$opt"
        suffix="d_${depth}-co_${COMBINED_OPTIMIZERS}-mt_${model_type}-r_${ltv_r}-q_${ltv_query}-k_${ltv_key}-v_${ltv_value}-ambi_${alpha_mlp_bias_init}-rfb_${rand_filter_bias}-rfi_${rand_filter_init}"

        if [ -f "/tmp/speedrun-${suffix}.skip" ]; then
            echo "Skipping ${suffix} because skip file exists."
            continue
        fi

        # Redirect this loop's output to log
        exec > >(tee -a "$LOG_FILE") 2>&1

        echo "Running with: $opt"

        PYTHONUNBUFFERED=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        TORCHINDUCTOR_CACHE_DIR=".torch_cache/torchinductor-$GPU_NAME" \
        TRITON_CACHE_DIR=".torch_cache/triton-$GPU_NAME" \
        uv run \
            --python "$PY" -m scripts.base_train \
            --model_tag="${suffix}" \
            --report_name=report/${suffix} \
            --depth=$depth \
            --max_seq_len=$SEQUENCE_LEN \
            --run=${WANDB_RUN}-${suffix} \
            --device_batch_size=$DEVICE_BATCH_SIZE \
            --eval_every=-1 \
            --core_metric_every=-1 \
            --sample_every=-1 \
            --save_every=-1 \
            --save_checkpoint_at_last_step=false \
            --exit_after_save=false \
            --model_dim="$((H * R))" \
            --combined_optimizers=${COMBINED_OPTIMIZERS} \
            --model_type=$model_type \
            --ltv_r=$ltv_r \
            --ltv_query=$ltv_query \
            --ltv_key=$ltv_key \
            --ltv_value=$ltv_value \
            --alpha_mlp_bias_init=$alpha_mlp_bias_init \
            --rand_filter_bias=$rand_filter_bias \
            --rand_filter_init=$rand_filter_init \
            --resume_from_step=-2 \
            --num_iterations=${NUM_ITERATIONS} \
            --sync_on_mps=true

        # Restore original stdout/stderr for next iteration
        exec 1>&3 2>&4
        sleep 1
    done
fi
