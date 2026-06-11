#!/usr/bin/env bash

# Exit on error, treat unset variables as error
set -euo

# 1) Environment Setup 
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv"

BASE_TRAIN_NUM_ITERATIONS=${BASE_TRAIN_NUM_ITERATIONS:-10880}

REGISTER_CAST_DUPLICATE_DEPTH_11_R_128="model_tag=d_11-co_true-mt_ltv_fused_concat_duplicate_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true checkpoint_step=10880"
REGISTER_CAST_DEPTH_11_LAYER_0_R_128="model_tag=d_11-co_true-mt_ltv_fused_concat_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true-ltv_layer_count_1 checkpoint_step=10880"

options=()
options+=("time_step=1023 $REGISTER_CAST_DUPLICATE_DEPTH_11_R_128")
# options+=("time_step=1023 $REGISTER_CAST_DEPTH_11_LAYER_0_R_128")
# options+=("time_step=511 $REGISTER_CAST_DUPLICATE_DEPTH_11_R_128")
# options+=("time_step=511 $REGISTER_CAST_DEPTH_11_LAYER_0_R_128")

for opt in "${options[@]}"; do
    eval "$opt"
    echo "----------------------------------------------------------"
    echo "Analyzing Debug Stats for: ${model_tag} at checkpoint ${checkpoint_step}"
    echo "----------------------------------------------------------"
    uv run --no-sync -m paper.analysis.analyze_debug_bpb \
        --model_tag="${model_tag}" \
        --checkpoints "${checkpoint_step}" \
        --time_step="${time_step}" \
        --save_md .analysis/debug_bpb_${model_tag}_time_step_${time_step}.md \
        --save_tex .analysis/debug_bpb_${model_tag}_time_step_${time_step}.tex \
        --save_k99_md .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99.md \
        --save_k99_tex .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99.tex \
        --save_k99_p99_md .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99_p99.md \
        --save_k99_p99_tex .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99_p99.tex \
        --save_k99_quantiles_md .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99_quantiles.md \
        --save_k99_quantiles_tex .analysis/debug_bpb_${model_tag}_time_step_${time_step}_k99_quantiles.tex \
        --filter_lower_bound 4000,10240,10240,971,1400,56 \
        --filter_upper_bound 4100,10250,10250,1025,102500,57 \
        --filter_layer 10,0,0,0,0,0 \
        --filter_metric k99,k99,k99,expectation,expectation,k99 \
        --filter_projection V,K,V,V,Q,Q
done
