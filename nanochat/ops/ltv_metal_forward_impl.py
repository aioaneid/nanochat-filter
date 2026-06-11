import torch
from nanochat.ltv.ltv_metal_core import (
    core_metal_forward_time,
    core_metal_forward_spatial,
)


def metal_forward_time(h0, alpha, x, kernel_op):
    result, _ = core_metal_forward_time(h0, alpha, x, kernel_op)
    return result


def metal_forward_spatial(h0, alpha, x, kernel_op):
    result, _ = core_metal_forward_spatial(h0, alpha, x, kernel_op)
    return result


def schedule_forward_spatial(h0, alpha, x, schedule: str, kernel_op):
    out, _ = core_metal_forward_spatial(
        h0,
        alpha,
        x,
        lambda alpha_mps, x_mps, h_out_alpha_mps, h_out_x_mps, T, parallel_dim_a, r: (
            kernel_op(
                alpha_mps,
                x_mps,
                h_out_alpha_mps,
                h_out_x_mps,
                T,
                parallel_dim_a,
                r,
                schedule,
            )
        ),
    )
    return out


@torch.library.impl("nanochat::ltv_noop_metal_op", "MPS")
def noop_metal_impl(h0, alpha, x):
    return metal_forward_time(h0, alpha, x, torch.ops.nanochat.ltv_noop_metal_kernel)


@torch.library.impl("nanochat::ltv_plane_metal_op", "MPS")
def ltv_plane_metal_impl(h0, alpha, x, schedule: str):
    return schedule_forward_spatial(
        h0, alpha, x, schedule, torch.ops.nanochat.ltv_plane_metal_kernel
    )


@torch.library.impl("nanochat::ltv_shared_metal_op", "MPS")
def ltv_shared_metal_impl(h0, alpha, x, schedule: str):
    return schedule_forward_spatial(
        h0, alpha, x, schedule, torch.ops.nanochat.ltv_shared_metal_kernel
    )


@torch.library.impl("nanochat::ltv_scan_metal_op", "MPS")
def ltv_scan_metal_impl(h0, alpha, x):
    return metal_forward_time(h0, alpha, x, torch.ops.nanochat.ltv_scan_metal_kernel)


@torch.library.impl("nanochat::ltv_fused_scan_metal_op", "MPS")
def ltv_fused_scan_metal_impl(q, k, v, logits, inits, NH, NKVH, da, r):
    B, T, _, D = q.shape

    # Pre-allocate output buffers
    out_q = torch.empty((B, NH, T, D), device=q.device, dtype=q.dtype)
    out_k = torch.empty((B, NKVH, T, D), device=k.device, dtype=k.dtype)
    out_v = torch.empty((B, NKVH, T, D), device=v.device, dtype=v.dtype)

    torch.ops.nanochat.ltv_fused_scan_metal_forward_kernel(
        q, k, v, logits, inits, out_q, out_k, out_v, NH, NKVH, da, r
    )
    return out_q, out_k, out_v


@torch.library.impl("nanochat::ltv_fused_linear_scan_metal_op", "MPS")
def ltv_fused_linear_scan_metal_impl(
    q, k, v, x_gate, w_gate, b_gate, inits, num_alpha_groups, r
):
    B, T, NH, D = q.shape
    _, _, NKVH, _ = k.shape

    # Pre-allocate output buffers.
    # Note: These are in (B, H, T, D) layout to optimize the backward scan re-read.
    out_q = torch.empty((B, NH, T, D), device=q.device, dtype=q.dtype)
    out_k = torch.empty((B, NKVH, T, D), device=k.device, dtype=k.dtype)
    out_v = torch.empty((B, NKVH, T, D), device=v.device, dtype=v.dtype)

    torch.ops.nanochat.ltv_fused_linear_scan_metal_forward_kernel(
        q,
        k,
        v,
        x_gate,
        w_gate,
        b_gate,
        inits,
        out_q,
        out_k,
        out_v,
        da=num_alpha_groups,
        r=r,
    )

    return out_q, out_k, out_v


@torch.library.impl("nanochat::ltv_fused_concat_scan_metal_op", "MPS")
def ltv_fused_concat_scan_metal_impl(combined, logit_bias, inits, NH, NKVH, da, r):
    B, T, _ = combined.shape
    D = da * r

    out_q = torch.empty((B, NH, T, D), device=combined.device, dtype=combined.dtype)
    out_k = torch.empty((B, NKVH, T, D), device=combined.device, dtype=combined.dtype)
    out_v = torch.empty((B, NKVH, T, D), device=combined.device, dtype=combined.dtype)

    # Get strides of the input combined tensor
    sc_b, sc_t, sc_d = combined.stride()

    torch.ops.nanochat.ltv_fused_concat_scan_metal_forward_kernel(
        combined,
        logit_bias,
        inits,
        out_q,
        out_k,
        out_v,
        sc_b,
        sc_t,
        sc_d,
        NH,
        NKVH,
        da,
        r,
    )
    return out_q, out_k, out_v
