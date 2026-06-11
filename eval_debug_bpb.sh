#!/usr/bin/env bash

# Exit on error, treat unset variables as error
set -euo

DRYRUN=${DRYRUN:-"false"}

# 1) Environment Setup 
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv"

cd rust_ewma
uv run maturin build --features python,metal --release
cd ..
uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv"

# -----------------------------------------------------------------------------
# Checkpoints to evaluate

BASE_TRAIN_NUM_ITERATIONS=10880

checkpoints=()
checkpoints+=("${BASE_TRAIN_NUM_ITERATIONS}")

# -----------------------------------------------------------------------------
# Model Specs
REGISTER_CAST_DUPLICATE_DEPTH_11_R_128="depth=11 model_type=ltv_fused_concat_duplicate_register_cast_scan_metal ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true ltv_layer_count=0"
REGISTER_CAST_DEPTH_11_LAYER_0_R_128="depth=11 model_type=ltv_fused_concat_register_cast_scan_metal ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true ltv_layer_count=1"

# Configuration Array
options=()

# Add the specs you want to run evaluate for:
options+=("time_step=1023 $REGISTER_CAST_DUPLICATE_DEPTH_11_R_128")
options+=("time_step=1023 $REGISTER_CAST_DEPTH_11_LAYER_0_R_128")
# options+=("time_step=511 $REGISTER_CAST_DUPLICATE_DEPTH_11_R_128")
# options+=("time_step=511 $REGISTER_CAST_DEPTH_11_LAYER_0_R_128")

# -----------------------------------------------------------------------------
COMBINED_OPTIMIZERS="true"

# -----------------------------------------------------------------------------
# Main Evaluation Loop
for opt in "${options[@]}"; do
    echo "----------------------------------------------------------"
    echo "Preparing Eval Debug for: $opt"
    
    # eval sets model_type, ltv_r, depth, etc. as env vars in this subshell
    eval "$opt"

    # Construct the exact same suffix used in base_train
    # suffix="d_${depth}-co_${COMBINED_OPTIMIZERS}-mt_${model_type}-r_${ltv_r}-q_${ltv_query}-k_${ltv_key}-v_${ltv_value}-ambi_${alpha_mlp_bias_init}-rfb_${rand_filter_bias}-rfi_${rand_filter_init}"

    suffix="d_${depth}-co_${COMBINED_OPTIMIZERS}-mt_${model_type}-r_${ltv_r}-q_${ltv_query}-k_${ltv_key}-v_${ltv_value}-ambi_${alpha_mlp_bias_init}-rfb_${rand_filter_bias}-rfi_${rand_filter_init}"
    (( ltv_layer_count )) && suffix+="-ltv_layer_count_${ltv_layer_count}"

    remap_attention_keys="false"
    
    echo "Model Tag: $suffix"
    echo "----------------------------------------------------------"

    # TODO: 10485760
    MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/metal TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/inductor \
    uv run --no-sync paper/analysis/eval_debug_bpb.py \
        --model_tag="${suffix}" \
        --device_batch_size=1 \
        --device_type=mps \
        --eval_tokens=10485760 \
        --remap_attention_keys="${remap_attention_keys}" \
        --dryrun="${DRYRUN}" \
        --time_step="${time_step}" \
        --checkpoints "${checkpoints[@]}"
done
