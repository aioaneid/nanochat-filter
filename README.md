# nanochat

This repo is a fork of [NanoChat](https://github.com/karpathy/nanochat) which implements a time-variable low-pass filter of the Q, K and V attention projections.

The pre-trained checkpoints can be found at:
* [Baseline (d=11)](https://huggingface.co/aioaneid/nanochat_n_layer_12_seq_len_1024_n_embd_1024/tree/main/optimizers/combined/base_checkpoints/d_11-co_true-mt_concat_duplicate_original-r_0-q_none-k_none-v_none-ambi_0-rfb_false-rfi_false)
* [TVLP (layer-0, d=11)](https://huggingface.co/aioaneid/nanochat_n_layer_12_seq_len_1024_n_embd_1024/tree/main/optimizers/combined/base_checkpoints/d_11-co_true-mt_ltv_fused_concat_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true-ltv_layer_count_1)
* [TVLP (all layers, d=11)](https://huggingface.co/aioaneid/nanochat_n_layer_12_seq_len_1024_n_embd_1024/tree/main/optimizers/combined/base_checkpoints/d_11-co_true-mt_ltv_fused_concat_duplicate_register_cast_scan_metal-r_128-q_active-k_active-v_active-ambi_0-rfb_false-rfi_true)
* [Baseline (d=12)](https://huggingface.co/aioaneid/nanochat_n_layer_12_seq_len_1024_n_embd_1024/tree/main/optimizers/combined/base_checkpoints/d_12-co_true-mt_concat_original-r_0-q_none-k_none-v_none-ambi_0-rfb_false-rfi_false)

The [training data set and tokenizer](https://huggingface.co/datasets/aioaneid/nanochat.base.240) have been obtained by running [this part](https://github.com/karpathy/nanochat/blob/ae0bf525299633d973d39ecf996edcb48e1fa6f5/speedrun.sh#L1-L76) of the original `speedrun.sh` script with [a bug fix](https://github.com/karpathy/nanochat/pull/429).

In order to train the model in all 4 configuration on Apple MacBook Pro M4 or later, one can download the training data set to `$HOME/.cache/nanochat.base.240`, and then run [train-speedrun.sh](train-speedrun.sh). The training can be monitored via tensorboard like this: `uv run tensorboard --logdir $HOME/.cache/nanochat/nanochat.ltv/logs/nanochat`.

Other commands:

* Run all unit tests on Mac: `uv sync --group dev && MATURIN_FEATURES="python,metal" uv pip install --no-build-isolation -e rust_ewma && METAL_DEVICE_CHECK_ERRORS=1 MT_DEVICE_CHECK_ERRORS=1 MTL_DEBUG_LAYER=1 MTL_LOG_ERRORS=1 METAL_DEVICE_WRAPPER_TYPE=0 uv run pytest --log-cli-level=INFO`
* Run the CUDA unit tests on Nvidia L40S: `modal run scripts/modal_app.py::run_benchmark --mode test-head-major-blelloch --no-sigmoid`
* Run 4 training steps on Nvidia L40S: `modal run scripts/modal_app.py::run_perf_base_train --max-forward-specs 1000 --max-backward-specs 1000`

There is also some limited support for running benchmarks and training on lightning.ai in [build_and_upload_run_bench_gpt_ltv.sh](build_and_upload_run_bench_gpt_ltv.sh) and [run_bench_gpt_ltv.sh](run_bench_gpt_ltv.sh).

Code links:
* The baseline models use [nanochat/concat_qkv_computer.py](nanochat/concat_qkv_computer.py) without a filter.
* The low-pass filtered models use [nanochat/ltv_combined_qkv_computer_module.py](nanochat/ltv_combined_qkv_computer_module.py) with a filter backed by custom kernels:
  * On MPS:
    * [nanochat/ltv_fused_concat_register_cast_scan_metal_qkv_filter.py](nanochat/ltv_fused_concat_register_cast_scan_metal_qkv_filter.py)
    * [rust_ewma/src/ltv_fused_concat_register_cast_scan_forward.msl](rust_ewma/src/ltv_fused_concat_register_cast_scan_forward.msl)
    * [rust_ewma/src/ltv_fused_concat_register_cast_scan_backward_atomic.msl](rust_ewma/src/ltv_fused_concat_register_cast_scan_backward_atomic.msl)
  * On CUDA:
    * [nanochat/ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter.py](nanochat/ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter.py)
    * [cpp_ewma/src/fused_linear_transpose_forward.cu](cpp_ewma/src/fused_linear_transpose_forward.cu)
    * [cpp_ewma/src/fused_linear_transpose_backward.cu](cpp_ewma/src/fused_linear_transpose_backward.cu)
    * [cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu](cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu)
    * [cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu](cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu)
