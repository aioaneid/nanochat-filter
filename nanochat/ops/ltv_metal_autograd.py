import torch
import logging

from nanochat.ltv.ltv_metal_core import (
    core_scan_backward_time,
    core_scan_backward_spatial,
)

logger = logging.getLogger(__name__)


def ltv_metal_setup(ctx, inputs, output):
    h0, alpha, x = inputs
    # Save as they are; the core_backward functions handle the specific transpositions
    ctx.save_for_backward(h0, alpha, output)
    ctx.dtype_x = x.dtype


def ltv_schedule_setup(ctx, inputs, output):
    ltv_metal_setup(ctx, inputs[:-1], output)
    ctx.schedule = inputs[-1]


def ltv_metal_backward(ctx, grad_h_x, kernel_fn):
    h0, alpha, h_out_user = ctx.saved_tensors

    # Use the Time-Major core backward
    grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_time(
        h0,
        alpha.transpose(0, 1),  # Prepare for internal (T, B, Da) logic
        h_out_user.transpose(0, 1),
        grad_h_x,
        kernel_fn,
    )

    return (
        grad_init_mps.to(dtype=h0.dtype),
        grad_alpha_mps_t.to(dtype=alpha.dtype),
        grad_x_mps_t.to(dtype=ctx.dtype_x),
    )


def ltv_schedule_backward(ctx, grad_h_x, kernel_fn):
    h0, alpha, h_out_user = ctx.saved_tensors

    # Define the kernel wrapper for the blelloch schedule
    wrapped_kernel = lambda a, x, oa, ox, ts, pa, r: kernel_fn(
        a, x, oa, ox, ts, pa, r, ctx.schedule
    )

    # Use the Spatial-Major core backward
    # Inputs: alpha (B, T, Da), h_out_user (B, T, D)
    # The spatial core handles transposing to (B, Da, T) and (B, D, T) internally
    grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_spatial(
        h0,
        alpha.transpose(1, 2),  # Prepare for internal (B, Da, T) logic
        h_out_user.transpose(1, 2),
        grad_h_x,
        wrapped_kernel,
    )

    return (
        grad_init_mps.to(dtype=h0.dtype),
        grad_alpha_mps_t.to(dtype=alpha.dtype),
        grad_x_mps_t.to(dtype=ctx.dtype_x),
        None,  # For schedule string
    )




torch.library.register_autograd(
    "nanochat::ltv_noop_metal_op",
    lambda ctx, grad: ltv_metal_backward(
        ctx, grad, torch.ops.nanochat.ltv_noop_metal_kernel
    ),
    setup_context=ltv_metal_setup,
)


torch.library.register_autograd(
    "nanochat::ltv_plane_metal_op",
    lambda ctx, grad: ltv_schedule_backward(
        ctx,
        grad,
        torch.ops.nanochat.ltv_plane_metal_kernel,
    ),
    setup_context=ltv_schedule_setup,
)




torch.library.register_autograd(
    "nanochat::ltv_shared_metal_op",
    lambda ctx, grad: ltv_schedule_backward(
        ctx,
        grad,
        torch.ops.nanochat.ltv_shared_metal_kernel,
    ),
    setup_context=ltv_schedule_setup,
)


torch.library.register_autograd(
    "nanochat::ltv_scan_metal_op",
    lambda ctx, grad: ltv_metal_backward(
        ctx, grad, torch.ops.nanochat.ltv_scan_metal_kernel
    ),
    setup_context=ltv_metal_setup,
)


def _ltv_fused_scan_setup(ctx, inputs, output):
    # Unpack inputs explicitly
    q, k, v, logits, inits, NH, NKVH, da, r = inputs
    # output is (out_q, out_k, out_v)

    # Save tensors for the backward pass
    ctx.save_for_backward(q, k, v, logits, inits, *output)

    # Explicitly store scalar parameters
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_scan_backward(ctx, grad_out_q, grad_out_k, grad_out_v):
    # 1. Unpack saved tensors
    q, k, v, logits, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    B, T, _, D = q.shape

    # logger.info(
    #     f"Backward pass: B={B}, T={T}, NH={NH}, NKVH={NKVH}, da={da}, r={r}, q.shape: {q.shape}, grad_out_q.shape: {grad_out_q.shape}"
    # )
    # logger.info(
    #     f"grad_out_q.shape: {grad_out_q.shape} grad_out_q.stride(): {grad_out_q.stride()}"
    # )

    grad_out_q_stride = grad_out_q.stride()
    grad_out_k_stride = grad_out_k.stride()
    grad_out_v_stride = grad_out_v.stride()

    # 4. PREPARE OUTPUTS
    # We allocate these in standard Time-Major (B, T, H, D)
    dx_q = torch.empty_like(q)
    dx_k = torch.empty_like(k)
    dx_v = torch.empty_like(v)
    dl = torch.zeros_like(logits)  # Atomic destination
    di = torch.empty_like(inits)  # Unified gradient for combined inits

    # logger.info("logits.shape: %s logits.stride(): %s", logits.shape, logits.stride())

    # 5. KERNEL EXECUTION
    # The kernel must use:
    #   - std_idx for: q, k, v, grad_out, dx, dk, dv
    #   - tr_idx for:  hq, hk, hv (because they were written Head-Major in Forward)
    torch.ops.nanochat.ltv_fused_scan_metal_backward_kernel(
        q,
        k,
        v,
        logits,
        inits,
        hq,
        hk,
        hv,
        grad_out_q,
        grad_out_k,
        grad_out_v,
        dx_q,
        dx_k,
        dx_v,
        dl,
        di,
        grad_out_q_stride[0],
        grad_out_q_stride[1],
        grad_out_q_stride[2],
        grad_out_q_stride[3],
        grad_out_k_stride[0],
        grad_out_k_stride[1],
        grad_out_k_stride[2],
        grad_out_k_stride[3],
        grad_out_v_stride[0],
        grad_out_v_stride[1],
        grad_out_v_stride[2],
        grad_out_v_stride[3],
        NH,
        NKVH,
        da,
        r,
    )

    # Return gradients corresponding to:
    # (q, k, v, logits, inits, NH, NKVH, da, r)
    return dx_q, dx_k, dx_v, dl, di, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_fused_scan_metal_op",
    ltv_fused_scan_backward,
    setup_context=_ltv_fused_scan_setup,
)


def _ltv_fused_linear_scan_setup(ctx, inputs, output):
    # Unpack inputs explicitly
    # inputs: (q, k, v, x_gate, w_gate, b_gate, inits, da, r)
    q, k, v, x_gate, w_gate, b_gate, inits, da, r = inputs
    # output: (out_q, out_k, out_v)
    hq, hk, hv = output

    # Save tensors for the backward pass
    # We save the forward outputs (hq, hk, hv) to use as 'h_prev' during backward scan
    ctx.save_for_backward(q, k, v, x_gate, w_gate, b_gate, inits, hq, hk, hv)

    # Extract B, T, NH, NKVH from input tensors
    B, T, NH, _ = q.shape
    _, _, NKVH, _ = k.shape
    ctx.B = B
    ctx.T = T
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_linear_scan_backward(ctx, grad_out_q, grad_out_k, grad_out_v):
    # Unpack saved tensors
    q, k, v, x_gate, w_gate, b_gate, inits, hq, hk, hv = ctx.saved_tensors

    # Access explicit params
    _, _, _, _, da, r = (
        ctx.B,
        ctx.T,
        ctx.NH,
        ctx.NKVH,
        ctx.da,
        ctx.r,
    )

    # logger.info(
    #     f"Backward pass: B={B}, T={T}, NH={NH}, NKVH={NKVH}, da={da}, r={r}, q.shape: {q.shape}, grad_out_q.shape: {grad_out_q.shape}"
    # )
    # logger.info(
    #     f"grad_out_q.shape: {grad_out_q.shape} grad_out_q.stride(): {grad_out_q.stride()}"
    # )

    grad_out_q_stride = grad_out_q.stride()
    grad_out_k_stride = grad_out_k.stride()
    grad_out_v_stride = grad_out_v.stride()

    # Prepare gradient buffers
    # dx_q/k/v and di can be empty_like as they are overwritten
    dx_q = torch.empty_like(q)
    dx_k = torch.empty_like(k)
    dx_v = torch.empty_like(v)
    di = torch.empty_like(inits)

    # Must be zeros because the kernel uses atomic_fetch_add.
    dx_gate = torch.zeros_like(x_gate)
    dw_gate = torch.zeros_like(w_gate)
    db_gate = torch.zeros_like(b_gate)

    # Call the kernel
    torch.ops.nanochat.ltv_fused_linear_scan_metal_backward_kernel(
        q,
        k,
        v,
        x_gate,
        w_gate,
        b_gate,
        inits,
        hq,
        hk,
        hv,
        grad_out_q,
        grad_out_k,
        grad_out_v,
        dx_q,
        dx_k,
        dx_v,
        dx_gate,
        dw_gate,
        db_gate,
        di,
        grad_out_q_stride[0],
        grad_out_q_stride[1],
        grad_out_q_stride[2],
        grad_out_q_stride[3],
        grad_out_k_stride[0],
        grad_out_k_stride[1],
        grad_out_k_stride[2],
        grad_out_k_stride[3],
        grad_out_v_stride[0],
        grad_out_v_stride[1],
        grad_out_v_stride[2],
        grad_out_v_stride[3],
        da,
        r,
    )

    # (q, k, v, x_gate, w_gate, b_gate, inits, da, r, n_embd)
    return (
        dx_q,
        dx_k,
        dx_v,
        dx_gate,
        dw_gate,
        db_gate,
        di,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nanochat::ltv_fused_linear_scan_metal_op",
    ltv_fused_linear_scan_backward,
    setup_context=_ltv_fused_linear_scan_setup,
)


def _ltv_fused_concat_scan_setup(ctx, inputs, output):
    combined, logit_bias, inits, NH, NKVH, da, r = inputs
    ctx.save_for_backward(combined, logit_bias, inits, *output)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_concat_scan_backward(ctx, grad_out_q, grad_out_k, grad_out_v):
    combined, logit_bias, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    B, T, _ = combined.shape
    D = da * r

    # Strides for input combined tensor
    sc_b, sc_t, sc_d = combined.stride()

    # Strides for incoming gradients
    gq_s = grad_out_q.stride()
    gk_s = grad_out_k.stride()
    gv_s = grad_out_v.stride()

    # Destination gradients
    # We use preserve_format to try to match strides, but we pass explicit strides
    # to the kernel anyway to be absolutely safe.

    # TODO: As of 37156a49d221cf0b538f395171d529254fcf8f5a, no longer needed
    # to zero it out.
    grad_combined = torch.zeros_like(combined, memory_format=torch.preserve_format)
    grad_inits = torch.empty_like(inits)

    # Strides for grad_combined (might differ from combined if preserve_format fails or isn't supported for complex views)
    sgc_b, sgc_t, sgc_d = grad_combined.stride()

    torch.ops.nanochat.ltv_fused_concat_scan_metal_backward_kernel(
        combined,
        logit_bias,
        inits,
        hq,
        hk,
        hv,
        grad_out_q,
        grad_out_k,
        grad_out_v,
        grad_combined,
        grad_inits,
        sc_b,
        sc_t,
        sc_d,  # Input strides
        sgc_b,
        sgc_t,
        sgc_d,  # Output gradient strides
        gq_s[0],
        gq_s[1],
        gq_s[2],
        gq_s[3],
        gk_s[0],
        gk_s[1],
        gk_s[2],
        gk_s[3],
        gv_s[0],
        gv_s[1],
        gv_s[2],
        gv_s[3],
        NH,
        NKVH,
        da,
        r,
    )

    # Reduction for logit_bias (handled by Torch, faster than kernel reduction)
    # The 'logits' part of combined starts after Q, K, V
    total_heads = NH + 2 * NKVH
    logit_offset = total_heads * D

    # We slice grad_combined to get the gradients corresponding to the logit part
    # shape [B, T, TotalHeads * Da]
    grad_logits_part = grad_combined[..., logit_offset:]

    # Sum over B and T to get bias gradient [TotalHeads * Da]
    grad_logit_bias = grad_logits_part.sum(dim=(0, 1))

    return grad_combined, grad_logit_bias, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_fused_concat_scan_metal_op",
    ltv_fused_concat_scan_backward,
    setup_context=_ltv_fused_concat_scan_setup,
)
