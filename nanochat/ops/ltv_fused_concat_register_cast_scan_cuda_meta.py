import torch


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_scan_cuda_op", "Meta"
)
def ltv_fused_concat_register_cast_scan_cuda_meta(combined, inits, NH, NKVH, da, r):
    B, T, _ = combined.shape
    D = da * r
    return (
        torch.empty(
            (B, NH, T, D),
            device=combined.device,
            dtype=combined.dtype,
            memory_format=torch.contiguous_format,
        ),
        torch.empty(
            (B, NKVH, T, D),
            device=combined.device,
            dtype=combined.dtype,
            memory_format=torch.contiguous_format,
        ),
        torch.empty(
            (B, NKVH, T, D),
            device=combined.device,
            dtype=combined.dtype,
            memory_format=torch.contiguous_format,
        ),
    )


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_scan_cuda_backward_kernel", "Meta"
)
def ltv_fused_concat_register_cast_scan_cuda_backward_kernel_meta(
    combined,
    inits,
    hq,
    hk,
    hv,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    grad_inits,
    NH,
    NKVH,
    da,
    r,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_scan_cuda_forward_kernel", "Meta"
)
def ltv_fused_concat_register_cast_scan_cuda_forward_kernel_meta(
    combined, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
):
    return None
