#!/usr/bin/env bash

cd rust_ewma
uv run maturin build --features python,metal --release
cd ..
uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv"

options=()
options+=("model_tag=d_11-co_true-mt_ltv_fused_concat_duplicate_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true checkpoint_step=10880")
options+=("model_tag=d_11-co_true-mt_ltv_fused_concat_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true-ltv_layer_count_1 checkpoint_step=10880")
# options+=("model_tag=d_11-co_true-mt_ltv_concat_duplicate_fused_scan-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true-ltv_layer_count_1 checkpoint_step=-1")
# options+=("model_tag=d_11-co_true-mt_ltv_concat_fused_scan-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true checkpoint_step=10880")
options+=("model_tag=d_11-co_true-mt_ltv_fused_concat_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true checkpoint_step=6180")

for opt in "${options[@]}"; do
    eval "$opt"
    uv run --no-sync paper/analysis/analyze_checkpoint_ltv.py \
        --model_tag ${model_tag} --checkpoint_step ${checkpoint_step} \
        --device_type mps \
        --save_md .analysis/checkpoint_ltv_${model_tag}_${checkpoint_step}.md \
        --save_tex .analysis/checkpoint_ltv_${model_tag}_${checkpoint_step}.tex
done
