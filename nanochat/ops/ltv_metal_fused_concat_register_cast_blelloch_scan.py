import torch
try:
    import rust_ewma
except ImportError:
    rust_ewma = None


# ---------------- FORWARD ----------------

@torch.library.impl(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_forward_kernel",
    "MPS",
)
def forward_kernel(
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
    use_sigmoid: bool,
):
    B, _, _ = combined.shape
    rust_ewma.ltv_metal_fused_concat_register_cast_blelloch_scan_forward(
        combined,
        inits,
        out_q,
        out_k,
        out_v,
        (sc_b, sc_t, sc_d),
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        use_sigmoid,
    )


@torch.library.impl(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_op",
    "MPS",
)
def forward_op(combined, inits, NH, NKVH, da, r, use_sigmoid: bool):
    B, T, _ = combined.shape
    D = da * r
    sc_b, sc_t, sc_d = combined.stride()

    assert combined.dtype == inits.dtype

    out_q = torch.empty((B, NH, T, D), device=combined.device, dtype=combined.dtype)
    out_k = torch.empty((B, NKVH, T, D), device=combined.device, dtype=combined.dtype)
    out_v = torch.empty((B, NKVH, T, D), device=combined.device, dtype=combined.dtype)

    torch.ops.nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_forward_kernel(
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
    )
    return out_q, out_k, out_v


# ---------------- BACKWARD ----------------

@torch.library.impl(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_backward_kernel",
    "MPS",
)
def backward_kernel(
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
    use_sigmoid: bool,
):
    B, T, _ = combined.shape

    rust_ewma.ltv_metal_fused_concat_register_cast_blelloch_scan_backward(
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
        (sc_in_b, sc_in_t, sc_in_d),
        (sc_out_b, sc_out_t, sc_out_d),
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        use_sigmoid,
    )


def _setup(ctx, inputs, output):
    combined, inits, NH, NKVH, da, r, use_sigmoid = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r
    ctx.use_sigmoid = use_sigmoid


def backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    use_sigmoid = ctx.use_sigmoid

    B, T, _ = combined.shape
    D = da * r
    total_h = NH + 2 * NKVH

    # Optimization: grad_combined is allocated as contiguous by default.
    grad_combined = torch.empty(
        combined.shape,
        dtype=combined.dtype, device=combined.device,
    )
    grad_inits = torch.empty_like(inits)

    sc_in_b, sc_in_t, sc_in_d = combined.stride()
    sc_out_b, sc_out_t, sc_out_d = grad_combined.stride()

    # direct_atomic optimization: if dtype is float32, kernel accumulates logit grads directly.
    # Otherwise (bfloat16), we use the intermediate float32 grad_logits buffer.
    if r > 1:
        grad_logits = torch.zeros(
            B, T, total_h, da,
            dtype=torch.float32,
            device=combined.device,
        )
    else:
        grad_logits = torch.empty(1, dtype=torch.float32, device=combined.device)

    torch.ops.nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_backward_kernel(
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
    )

    # Post-process: scatter float logit grads into grad_combined for r > 1 case
    if r > 1:
        offset_l = total_h * D
        # grad_logits is now (B, T, total_h, da), which aligns with the T-major output order.
        gl = grad_logits.reshape(B, T, total_h * da)
        # Automatic conversion to combined.dtype.
        grad_combined[:, :, offset_l:offset_l + total_h * da] = gl

    return grad_combined, grad_inits, None, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_metal_fused_concat_register_cast_blelloch_scan_op",
    backward,
    setup_context=_setup,
)
