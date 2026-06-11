"""
Things are very slow even before the code reaches here.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    # Associative scan: (a, x) ⊗ (a_p, x_p) = (a * a_p, a * x_p + x)
    @triton.jit
    def _combine(a1, x1, a2, x2):
        return a1 * a2, a2 * x1 + x2

    @triton.jit
    def ltv_associative_scan_kernel(
        alpha_ptr,
        x_ptr,
        out_alpha_ptr,
        out_x_ptr,
        stride_a_b: tl.constexpr,
        stride_a_da: tl.constexpr,
        stride_a_t: tl.constexpr,
        stride_x_b: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_x_t: tl.constexpr,
        B: tl.constexpr,
        D: tl.constexpr,
        Da: tl.constexpr,
        T: tl.constexpr,
        R: tl.constexpr,
    ):
        pid = tl.program_id(0)  # Grid size is (B * D)
        b_idx = pid // D
        d_idx = pid % D

        da_idx = d_idx // R

        # 3. Calculate the starting pointers for this specific row
        a_row_ptr = alpha_ptr + (b_idx * stride_a_b) + (da_idx * stride_a_da)
        x_row_ptr = x_ptr + (b_idx * stride_x_b) + (d_idx * stride_x_d)

        # 4. Load the data across the T dimension
        t_offsets = tl.arange(0, T)

        a = tl.load(a_row_ptr + t_offsets * stride_a_t)
        x = tl.load(x_row_ptr + t_offsets * stride_x_t)

        out_a, out_h = tl.associative_scan((a, x), 0, _combine)

        # Store results
        tl.store(
            out_x_ptr
            + (b_idx * stride_x_b)
            + (d_idx * stride_x_d)
            + t_offsets * stride_x_t,
            out_h,
        )

        if (d_idx % R) == 0:
            tl.store(
                out_alpha_ptr
                + (b_idx * stride_a_b)
                + (da_idx * stride_a_da)
                + t_offsets * stride_a_t,
                out_a,
            )


@torch.library.impl("nanochat::ltv_associative_scan_triton_kernel", "CUDA")
def ltv_associative_scan_triton_kernel_impl(
    alpha, x, out_alpha, out_x, t_seq, parallel_a, r
):
    if triton is None:
        raise RuntimeError("Triton not available")

    # alpha: (B, Da, T)
    # x: (B, D, T)
    B = alpha.shape[0]
    Da = alpha.shape[1]
    D = x.shape[1]
    T = alpha.shape[2]

    # We launch one program per line of x
    grid = (B * D,)

    ltv_associative_scan_kernel[grid](
        alpha,
        x,
        out_alpha,
        out_x,
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B,
        D,
        Da,
        T=t_seq,
        R=r,
    )


@torch.library.impl("nanochat::ltv_associative_scan_triton_op", "CUDA")
def ltv_associative_scan_triton_op_impl(h0, alpha, x):
    # This is the autograd-visible op
    from nanochat.ltv.ltv_metal_core import core_metal_forward_spatial

    # We reuse core_metal_forward_spatial but pass our triton kernel wrapper
    def kernel_wrapper(a_mps, x_mps, oa_mps, ox_mps, T, pa, r):
        torch.ops.nanochat.ltv_associative_scan_triton_kernel(
            a_mps, x_mps, oa_mps, ox_mps, T, pa, r
        )

    return core_metal_forward_spatial(h0, alpha, x, kernel_wrapper)[0]
