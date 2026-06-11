import torch

from nanochat.ops.ltv_metal_meta import ltv_metal_meta_spatial_major


@torch.library.impl("nanochat::ltv_associative_scan_triton_op", "Meta")
def ltv_associative_scan_triton_meta(h0, alpha, x):
    return ltv_metal_meta_spatial_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat::ltv_associative_scan_triton_kernel", "Meta")
def ltv_associative_scan_triton_kernel_meta(
    alpha, x, out_alpha, out_x, t_seq, parallel_a, r
):
    return None


@torch.library.impl("nanochat::ltv_fused_scan_triton_op", "Meta")
def ltv_fused_scan_triton_meta(q, k, v, logits, inits, NH, NKVH, da, r):
    B, T, _, D = q.shape
    return (
        torch.empty(
            (B, NH, T, D),
            device=q.device,
            dtype=q.dtype,
            memory_format=torch.contiguous_format,
        ),
        torch.empty(
            (B, NKVH, T, D),
            device=k.device,
            dtype=k.dtype,
            memory_format=torch.contiguous_format,
        ),
        torch.empty(
            (B, NKVH, T, D),
            device=v.device,
            dtype=v.dtype,
            memory_format=torch.contiguous_format,
        ),
    )


@torch.library.impl("nanochat::ltv_fused_scan_triton_backward_kernel", "Meta")
def ltv_fused_scan_triton_backward_kernel_meta(
    q,
    k,
    v,
    logits,
    inits,
    hq,
    hk,
    hv,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_q,
    grad_k,
    grad_v,
    grad_logits,
    grad_inits,
    NH,
    NKVH,
    da,
    r,
):
    return None


@torch.library.impl("nanochat::ltv_fused_concat_scan_triton_op", "Meta")
def ltv_fused_concat_scan_triton_meta(combined, logit_bias, inits, NH, NKVH, da, r):
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


@torch.library.impl("nanochat::ltv_fused_concat_scan_triton_backward_kernel", "Meta")
def ltv_fused_concat_scan_triton_backward_kernel_meta(
    combined,
    logit_bias,
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
