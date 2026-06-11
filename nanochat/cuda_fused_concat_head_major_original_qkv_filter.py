import torch

from nanochat.ltv_fused_concat_register_cast_scan_util import (
    CUDA_SUPPORTS_NATIVE_BF16,
    should_run_cuda_ltv_in_fp32,
    _SCAN_THREADS_FORWARD,
    _R_THREADS_FORWARD,
    _R_ITEMS_FORWARD,
    _SCAN_THREADS_BACKWARD,
    _R_THREADS_BACKWARD,
    _R_ITEMS_BACKWARD,
)
from nanochat.ops.ltv_cuda_ops import (
    fused_linear_transpose,
    cuda_fused_concat_head_major_original,
)

import nanochat.ops.cuda_fused_concat_head_major_original as _cuda_fused_concat_head_major_original  # noqa: F401


def cuda_fused_concat_head_major_original_qkv_computer(
    c: torch.nn.Linear,
    x: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    split_sections: list[int],
    da: int,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV computation using cuda_fused_concat_head_major_original kernel."""
    autocast_dtype = torch.get_autocast_dtype("cuda")
    run_ltv_in_fp32 = (
        autocast_dtype == torch.float32 if CUDA_SUPPORTS_NATIVE_BF16 else True
    )

    out_buf = fused_linear_transpose(
        x.float() if run_ltv_in_fp32 else x,
        c.weight.float() if run_ltv_in_fp32 else c.weight,
    )

    kernel_combined = out_buf.transpose(1, 2)
    if should_run_cuda_ltv_in_fp32(kernel_combined):
        kernel_combined = kernel_combined.float()

    hq, hk, hv = (
        cuda_fused_concat_head_major_original(
            kernel_combined,
            n_head,
            n_kv_head,
            da,
            r,
            _SCAN_THREADS_FORWARD,
            _R_THREADS_FORWARD,
            _R_ITEMS_FORWARD,
            _SCAN_THREADS_BACKWARD,
            _R_THREADS_BACKWARD,
            _R_ITEMS_BACKWARD,
        )
    )
    if run_ltv_in_fp32:
        hq = hq.to(kernel_combined.dtype)
        hk = hk.to(kernel_combined.dtype)
        hv = hv.to(kernel_combined.dtype)

    q, k, v = hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
    return q, k, v

