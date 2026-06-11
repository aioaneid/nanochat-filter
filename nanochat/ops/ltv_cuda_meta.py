import torch

from nanochat.ops.ltv_metal_meta import ltv_metal_meta_spatial_major


@torch.library.impl("nanochat_cuda::ltv_plane_cuda_op", "Meta")
def ltv_plane_cuda_meta(h0, alpha, x, schedule: str):
    return ltv_metal_meta_spatial_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat_cuda::ltv_plane_cuda_kernel", "Meta")
def ltv_plane_cuda_kernel_meta(
    alpha, x, out_alpha, out_x, t_seq, parallel_a, r, schedule: str
):
    return None


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_op", "Meta")
def ltv_fused_scan_cuda_meta(q, k, v, logits, inits, NH, NKVH, da, r):
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


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_backward_kernel", "Meta")
def ltv_fused_scan_cuda_backward_kernel_meta(
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


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_forward_kernel", "Meta")
def ltv_fused_scan_cuda_forward_kernel_meta(
    q, k, v, logits, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_op", "Meta"
)
def ltv_fused_concat_register_cast_blelloch_scan_cuda_meta(
    combined, inits, NH, NKVH, da, r, block_threads, items_per_thread, use_sigmoid
):
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
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_kernel",
    "Meta",
)
def ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_kernel_meta(
    combined,
    inits,
    out_q,
    out_k,
    out_v,
    T,
    NH,
    NKVH,
    da,
    r,
    block_threads,
    items_per_thread,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_kernel",
    "Meta",
)
def ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_kernel_meta(
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
    grad_logits,
    NH,
    NKVH,
    da,
    r,
    block_threads,
    items_per_thread,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_op",
    "Meta",
)
def ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_meta(
    combined,
    inits,
    logit_bias,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
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
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel",
    "Meta",
)
def ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel_meta(
    combined,
    inits,
    logit_bias,
    out_q,
    out_k,
    out_v,
    T,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel",
    "Meta",
)
def ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel_meta(
    combined,
    inits,
    logit_bias,
    hq,
    hk,
    hv,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    grad_inits,
    grad_logit_bias,
    NH,
    NKVH,
    da,
    r,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_op",
    "Meta",
)
def cuda_fused_concat_head_major_original_cuda_meta(
    combined,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
):
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
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_forward_kernel",
    "Meta",
)
def cuda_fused_concat_head_major_original_cuda_forward_kernel_meta(
    combined,
    out_q,
    out_k,
    out_v,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
):
    return None


@torch.library.impl(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_backward_kernel",
    "Meta",
)
def cuda_fused_concat_head_major_original_cuda_backward_kernel_meta(
    combined,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
):
    return None


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose",
    "Meta",
)
def fused_linear_transpose_meta(x, weight, out):
    # Meta kernel used for fake-tensor tracing / torch.compile: validate
    # shapes and dtypes but do not access any data pointers.
    if x.dim() < 2 or weight.dim() != 2:
        raise RuntimeError("fused_linear_transpose: invalid input shapes")
    return None


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose_backward_kernel",
    "Meta",
)
def fused_linear_transpose_backward_kernel_meta(
    x, weight, grad_out, grad_x, grad_weight
):
    # No-op meta implementation for backward kernel; shapes are validated
    # by the forward meta, and this prevents fake-tensor tracing from
    # attempting to access data pointers.
    return None


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose_cuda_op",
    "Meta",
)
def fused_linear_transpose_cuda_op_meta(x, weight):
    if x.dim() < 2 or weight.dim() != 2:
        raise RuntimeError("fused_linear_transpose_cuda_op: invalid input shapes")
    B, T, _ = x.shape
    F = weight.size(0)
    return torch.empty(
        (B, F, T), device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format
    )


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel_meta(
    combined,
    inits,
    logit_bias,
    out_q,
    out_k,
    out_v,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel_meta(
    combined,
    inits,
    logit_bias,
    hq,
    hk,
    hv,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    grad_inits,
    grad_logit_bias,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_meta(
    combined,
    inits,
    logit_bias,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
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
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward_kernel",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward_kernel_meta(
    combined,
    inits,
    logit_bias,
    out_q,
    out_k,
    out_v,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    r_threads_forward,
    r_items_forward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward_kernel",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward_kernel_meta(
    combined,
    inits,
    logit_bias,
    hq,
    hk,
    hv,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    grad_inits,
    grad_logit_bias,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_op",
    "Meta",
)
def ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_meta(
    combined,
    inits,
    logit_bias,
    NH,
    NKVH,
    da,
    r,
    P,
    Q,
    K,
    r_threads_forward,
    r_items_forward,
    r_threads_backward,
    r_items_backward,
    use_sigmoid,
):
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
