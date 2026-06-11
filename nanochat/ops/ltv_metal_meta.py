import logging
import torch

logger = logging.getLogger(__name__)


def ltv_metal_meta_time_major(shape, dtype, device):
    # Inputs/Outputs (B, T, D)
    # Memory Layout: (T, B, D)
    b, _, d = shape
    stride_b = d
    stride_t = b * d
    stride_d = 1
    out_stride = (stride_b, stride_t, stride_d)
    return torch.empty_strided(shape, out_stride, dtype=dtype, device=device)


def ltv_metal_meta_spatial_major(shape, dtype, device):
    B, T, D = shape
    return torch.empty_strided(shape, (D * T, 1, T), dtype=dtype, device=device)


@torch.library.impl("nanochat::ltv_noop_metal_op", "Meta")
def ltv_noop_metal_meta(h0, alpha, x):
    return ltv_metal_meta_time_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat::ltv_plane_metal_op", "Meta")
def ltv_plane_metal_meta(h0, alpha, x, schedule: str):
    return ltv_metal_meta_spatial_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat::ltv_shared_metal_op", "Meta")
def ltv_shared_metal_meta(h0, alpha, x, schedule: str):
    return ltv_metal_meta_spatial_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat::ltv_scan_metal_op", "Meta")
def ltv_scan_metal_meta(h0, alpha, x):
    return ltv_metal_meta_time_major(x.shape, x.dtype, x.device)


@torch.library.impl("nanochat::ltv_noop_metal_kernel", "Meta")
def ltv_noop_metal_kernel_meta(h_init, alpha, x, out, t_dim, parallel_a, r):
    return None


@torch.library.impl("nanochat::ltv_plane_metal_kernel", "Meta")
def ltv_plane_metal_kernel_meta(
    alpha, x, out_alpha, out_x, t_seq, parallel_a, r, schedule
):
    return None


@torch.library.impl("nanochat::ltv_shared_metal_kernel", "Meta")
def ltv_shared_metal_kernel_meta(
    alpha, x, out_alpha, out_x, t_seq, parallel_a, r, schedule
):
    return None


@torch.library.impl("nanochat::ltv_scan_metal_kernel", "Meta")
def ltv_scan_metal_kernel_meta(alpha, x, out_alpha, out_x, t_seq, parallel_a, r):
    return None


@torch.library.impl("nanochat::ltv_fused_scan_metal_op", "Meta")
def ltv_fused_scan_metal_meta(q, k, v, logits, inits, NH, NKVH, da, r):
    B, T, _, D = q.shape
    # Returns (B, NH, T, D), (B, NKVH, T, D), (B, NKVH, T, D)
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


@torch.library.impl("nanochat::ltv_fused_linear_scan_metal_op", "Meta")
def ltv_fused_linear_scan_metal_meta(q, k, v, x_gate, w_gate, b_gate, inits, da, r):
    B, T, NH, D = q.shape
    _, _, NKVH, _ = k.shape

    # We use q.new_empty to ensure the meta tensors inherit the correct
    # dtype (e.g., float32 or float16) and are placed on the 'meta' device.
    # Layout matches the forward implementation: (B, H, T, D)
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


@torch.library.impl("nanochat::ltv_fused_concat_scan_metal_op", "Meta")
def ltv_fused_concat_scan_metal_meta(combined, logit_bias, inits, NH, NKVH, da, r):
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


@torch.library.impl("nanochat::ltv_fused_concat_register_cast_scan_metal_op", "Meta")
def ltv_fused_concat_register_cast_scan_metal_meta(combined, inits, NH, NKVH, da, r):
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
    "nanochat::ltv_fused_concat_register_cast_scan_metal_forward_kernel", "Meta"
)
def ltv_fused_concat_register_cast_scan_metal_forward_kernel_meta(
    combined, inits, out_q, out_k, out_v, sc_b, sc_t, sc_d, T, NH, NKVH, da, r
):
    return None


@torch.library.impl(
    "nanochat::ltv_fused_concat_register_cast_scan_metal_backward_kernel", "Meta"
)
def ltv_fused_concat_register_cast_scan_metal_backward_kernel_meta(
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
    sc_b, sc_t, sc_d,
    sgc_b, sgc_t, sgc_d,
    sbq, shq, stq, sdq,
    sbk, shk, stk, sdk,
    sbv, shv, stv, sdv,
    NH,
    NKVH,
    da,
    r,
):
    return None


@torch.library.impl(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_op", "Meta"
)
def ltv_metal_fused_concat_register_cast_blelloch_scan_meta(
    combined, inits, NH, NKVH, da, r, use_sigmoid
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
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_forward_kernel",
    "Meta",
)
def ltv_metal_fused_concat_register_cast_blelloch_scan_forward_kernel_meta(
    combined,
    inits,
    out_q,
    out_k,
    out_v,
    sc_b,
    sc_t,
    sc_d,
    T,
    NH,
    NKVH,
    da,
    r,
    use_sigmoid,
):
    return None


@torch.library.impl(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_backward_kernel",
    "Meta",
)
def ltv_metal_fused_concat_register_cast_blelloch_scan_backward_kernel_meta(
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
    sc_in_b,
    sc_in_t,
    sc_in_d,
    sc_out_b,
    sc_out_t,
    sc_out_d,
    NH,
    NKVH,
    da,
    r,
    use_sigmoid,
):
    return None
