import torch
try:
    import rust_ewma
except ImportError:
    rust_ewma = None


@torch.library.impl(
    "nanochat::ltv_fused_concat_register_cast_scan_metal_forward_kernel", "MPS"
)
def ltv_fused_concat_register_cast_scan_metal_forward_kernel_impl(
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
):
    B, _, _ = combined.shape
    rust_ewma.ltv_fused_concat_register_cast_scan_metal_forward(
        combined, inits, out_q, out_k, out_v, B, T, sc_b, sc_t, sc_d, NH, NKVH, da, r
    )


@torch.library.impl(
    "nanochat::ltv_fused_concat_register_cast_scan_metal_op", "MPS"
)
def ltv_fused_concat_register_cast_scan_metal_op_impl(
    combined, inits, NH, NKVH, da, r
):
    B, T, _ = combined.shape
    D = da * r
    sc_b, sc_t, sc_d = combined.stride()

    out_q = torch.empty(
        (B, NH, T, D),
        device=combined.device,
        dtype=combined.dtype,
        memory_format=torch.contiguous_format,
    )
    out_k = torch.empty(
        (B, NKVH, T, D),
        device=combined.device,
        dtype=combined.dtype,
        memory_format=torch.contiguous_format,
    )
    out_v = torch.empty(
        (B, NKVH, T, D),
        device=combined.device,
        dtype=combined.dtype,
        memory_format=torch.contiguous_format,
    )

    torch.ops.nanochat.ltv_fused_concat_register_cast_scan_metal_forward_kernel(
        combined, inits, out_q, out_k, out_v,
        sc_b, sc_t, sc_d,
        T, NH, NKVH, da, r
    )
    return out_q, out_k, out_v


@torch.library.impl(
    "nanochat::ltv_fused_concat_register_cast_scan_metal_backward_kernel", "MPS"
)
def ltv_fused_concat_register_cast_scan_metal_backward_kernel_impl(
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
    B, T, _ = combined.shape

    rust_ewma.ltv_fused_concat_register_cast_scan_metal_backward(
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
        B,
        T,
        sc_b, sc_t, sc_d,
        sgc_b, sgc_t, sgc_d,
        sbq, shq, stq, sdq,
        sbk, shk, stk, sdk,
        sbv, shv, stv, sdv,
        NH,
        NKVH,
        da,
        r,
    )


def _ltv_fused_concat_register_cast_scan_metal_setup(ctx, inputs, output):
    combined, inits, NH, NKVH, da, r = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_concat_register_cast_scan_metal_backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r

    grad_combined = torch.empty_like(combined, memory_format=torch.contiguous_format)
    grad_inits = torch.empty_like(inits, memory_format=torch.contiguous_format)

    sc_b, sc_t, sc_d = combined.stride()
    sgc_b, sgc_t, sgc_d = grad_combined.stride()
    sbq, shq, stq, sdq = grad_hq.stride()
    sbk, shk, stk, sdk = grad_hk.stride()
    sbv, shv, stv, sdv = grad_hv.stride()

    torch.ops.nanochat.ltv_fused_concat_register_cast_scan_metal_backward_kernel(
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
    )

    return grad_combined, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_fused_concat_register_cast_scan_metal_op",
    ltv_fused_concat_register_cast_scan_metal_backward,
    setup_context=_ltv_fused_concat_register_cast_scan_metal_setup,
)
