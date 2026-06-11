import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    @triton.jit
    def ltv_fused_concat_scan_forward_vectorized_kernel(
        combined_ptr,
        bias_ptr,
        inits_ptr,
        out_q_ptr,
        out_k_ptr,
        out_v_ptr,
        sc_b,
        sc_t,
        sc_d,
        B,
        T,
        NH,
        NKVH,
        Da,
        R: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_glob = tl.program_id(1)
        da_idx = tl.program_id(2)  # Now indexing Da directly

        D = Da * R
        total_h = NH + 2 * NKVH
        # Create the vector of offsets for the R dimension
        d_offs = tl.arange(0, R)

        # Determine which output pointer and local head index to use
        if h_glob < NH:
            x_out_ptr = out_q_ptr
            h_loc = h_glob
            h_comp_total = NH
            offset_in_row = 0
        elif h_glob < NH + NKVH:
            x_out_ptr = out_k_ptr
            h_loc = h_glob - NH
            h_comp_total = NKVH
            offset_in_row = NH * D
        else:
            x_out_ptr = out_v_ptr
            h_loc = h_glob - (NH + NKVH)
            h_comp_total = NKVH
            offset_in_row = (NH + NKVH) * D

        b_offset_combined = b * sc_b
        offset_l = total_h * D

        # logit_idx is scalar relative to this program, val/out are vectors
        val_base = b_offset_combined + (offset_in_row + h_loc * D + da_idx * R) * sc_d
        logit_idx = b_offset_combined + (offset_l + h_glob * Da + da_idx) * sc_d
        out_base = b * (h_comp_total * T * D) + h_loc * (T * D) + (da_idx * R)

        # Initial states for the R vector
        acc = tl.load(inits_ptr + b * (total_h * D) + h_glob * D + da_idx * R + d_offs)
        b_val = tl.load(bias_ptr + h_glob * Da + da_idx)

        for _ in range(T):
            x_val = tl.load(combined_ptr + val_base + d_offs * sc_d)
            l_raw = tl.load(combined_ptr + logit_idx)

            alpha = 1.0 / (1.0 + tl.exp(-(l_raw + b_val)))
            acc = (1.0 - alpha) * acc + alpha * x_val

            tl.store(x_out_ptr + out_base + d_offs, acc)

            val_base += sc_t
            logit_idx += sc_t
            out_base += D

    # ------------------------------------------------------------
    # BACKWARD (DIRECT, r == 1)
    # ------------------------------------------------------------

    @triton.jit
    def ltv_fused_concat_scan_backward_vectorized_kernel(
        combined_ptr,
        bias_ptr,
        inits_ptr,
        hq_ptr,
        hk_ptr,
        hv_ptr,
        go_q_ptr,
        go_k_ptr,
        go_v_ptr,
        grad_comb_ptr,
        grad_init_ptr,
        sc_b,
        sc_t,
        sc_d,
        sgc_b,
        sgc_t,
        sgc_d,
        sbq,
        shq,
        stq,
        sdq,
        sbk,
        shk,
        stk,
        sdk,
        sbv,
        shv,
        stv,
        sdv,
        B,
        T,
        NH,
        NKVH,
        Da,
        R: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_glob = tl.program_id(1)
        da_idx = tl.program_id(2)

        D = Da * R
        total_h = NH + 2 * NKVH
        offset_l = total_h * D
        last_t = T - 1
        d_offs = tl.arange(0, R)

        # 1. Pointer Branching Logic
        if h_glob < NH:
            h_ptr = hq_ptr
            go_ptr = go_q_ptr
            h_loc = h_glob
            s_b, s_h, s_t, s_d = sbq, shq, stq, sdq
            offset_in_row = 0
        elif h_glob < NH + NKVH:
            h_ptr = hk_ptr
            go_ptr = go_k_ptr
            h_loc = h_glob - NH
            s_b, s_h, s_t, s_d = sbk, shk, stk, sdk
            offset_in_row = NH * D
        else:
            h_ptr = hv_ptr
            go_ptr = go_v_ptr
            h_loc = h_glob - (NH + NKVH)
            s_b, s_h, s_t, s_d = sbv, shv, stv, sdv
            offset_in_row = (NH + NKVH) * D

        # 2. Setup Base Pointers for the last timestep (T-1)
        # We move backwards through time inside the loop
        x_base = (
            combined_ptr
            + b * sc_b
            + last_t * sc_t
            + (offset_in_row + h_loc * D + da_idx * R) * sc_d
        )
        l_ptr = (
            combined_ptr
            + b * sc_b
            + last_t * sc_t
            + (offset_l + h_glob * Da + da_idx) * sc_d
        )
        tr_base = go_ptr + b * s_b + h_loc * s_h + last_t * s_t + (da_idx * R) * s_d

        g_x_base = (
            grad_comb_ptr
            + b * sgc_b
            + last_t * sgc_t
            + (offset_in_row + h_loc * D + da_idx * R) * sgc_d
        )
        g_l_ptr = (
            grad_comb_ptr
            + b * sgc_b
            + last_t * sgc_t
            + (offset_l + h_glob * Da + da_idx) * sgc_d
        )

        h_fwd_off = (
            b * ((NH if h_glob < NH else NKVH) * T * D)
            + h_loc * (T * D)
            + (last_t - 1) * D
            + (da_idx * R)
        )

        b_val = tl.load(bias_ptr + h_glob * Da + da_idx)
        gh_rec = tl.full((R,), 0.0, dtype=tl.float32)

        # 3. Main Backward Scan Loop (T-1 down to 1)
        for _ in range(T - 1):
            x_val = tl.load(x_base + d_offs * sc_d)
            l_val = tl.load(l_ptr)
            tr = tl.load(tr_base + d_offs * s_d)

            alpha = 1.0 / (1.0 + tl.exp(-(l_val + b_val)))
            gh_t = tr + gh_rec

            # Grad x: Vectorized store
            tl.store(g_x_base + d_offs * sgc_d, gh_t * alpha)

            # Grad logit: Program-local sum over the R dimension
            h_prev = tl.load(h_ptr + h_fwd_off + d_offs)
            d_logit = tl.sum(gh_t * (x_val - h_prev) * (alpha * (1.0 - alpha)), axis=0)
            tl.store(g_l_ptr, d_logit)

            # Update recurrent gradient for next timestep
            gh_rec = gh_t * (1.0 - alpha)

            # Move pointers backward one step in T
            x_base -= sc_t
            l_ptr -= sc_t
            tr_base -= s_t
            g_x_base -= sgc_t
            g_l_ptr -= sgc_t
            h_fwd_off -= D

        # 4. Final Timestep (t=0) using inits
        x_val = tl.load(x_base + d_offs * sc_d)
        l_val = tl.load(l_ptr)
        tr = tl.load(tr_base + d_offs * s_d)

        alpha = 1.0 / (1.0 + tl.exp(-(l_val + b_val)))
        gh_t = tr + gh_rec

        tl.store(g_x_base + d_offs * sgc_d, gh_t * alpha)

        # Initial states are the "h_prev" for t=0
        init_v = tl.load(inits_ptr + b * total_h * D + h_glob * D + da_idx * R + d_offs)
        d_logit = tl.sum(gh_t * (x_val - init_v) * (alpha * (1.0 - alpha)), axis=0)
        tl.store(g_l_ptr, d_logit)

        # Store the final gradient for the initial states
        tl.store(
            grad_init_ptr + b * total_h * D + h_glob * D + da_idx * R + d_offs,
            gh_t * (1.0 - alpha),
        )

    # ------------------------------------------------------------
    # PYTHON WRAPPERS
    # ------------------------------------------------------------

    def ltv_fused_concat_scan_forward(
        combined,
        bias,
        inits,
        out_q,
        out_k,
        out_v,
        B,
        T,
        sc_b,
        sc_t,
        sc_d,
        NH,
        NKVH,
        Da,
        R,
    ):
        # The grid dimension 2 is now just Da
        grid = (B, NH + 2 * NKVH, Da)

        # Calculate warps: 1 warp per 32 elements in R, minimum 1
        warps = max(1, R // 32)

        ltv_fused_concat_scan_forward_vectorized_kernel[grid](
            combined,
            bias,
            inits,
            out_q,
            out_k,
            out_v,
            combined.stride(0),
            combined.stride(1),
            combined.stride(2),
            B,
            T,
            NH,
            NKVH,
            Da,
            R,
            num_warps=warps,
        )


@torch.library.impl("nanochat::ltv_fused_concat_scan_triton_op", "default")
def ltv_fused_concat_scan_triton_op(combined, logit_bias, inits, NH, NKVH, da, r):
    B, T, _ = combined.shape
    D = da * r
    TOTAL_H = NH + 2 * NKVH

    # Layout: [B, H, T, D] to match Metal's internal expectations
    hq = combined.new_empty((B, NH, T, D), dtype=torch.float32)
    hk = combined.new_empty((B, NKVH, T, D), dtype=torch.float32)
    hv = combined.new_empty((B, NKVH, T, D), dtype=torch.float32)

    # FIX: Call the helper function instead of the kernel directly
    ltv_fused_concat_scan_forward(
        combined,
        logit_bias,
        inits,
        hq,
        hk,
        hv,
        B,
        T,
        combined.stride(0),
        combined.stride(1),
        combined.stride(2),
        NH,
        NKVH,
        da,
        r,
    )
    return hq, hk, hv


@torch.library.impl("nanochat::ltv_fused_concat_scan_triton_backward_kernel", "default")
def ltv_fused_concat_scan_triton_backward_kernel_impl(
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
    if triton is None:
        raise RuntimeError("Triton not available")

    B, T, _ = combined.shape
    # Grid: [Batch, Total Heads, Sub-head head dimension]
    grid = (B, NH + 2 * NKVH, da)

    # Calculate num_warps based on vector size r
    # If r is small (e.g., 4 or 8), we use 1 warp.
    # If r is large (e.g., 128), we use 4 warps.
    warps = max(1, min(8, r // 32))

    sc_b, sc_t, sc_d = combined.stride()
    sgc_b, sgc_t, sgc_d = grad_combined.stride()
    sbq, shq, stq, sdq = grad_hq.stride()
    sbk, shk, stk, sdk = grad_hk.stride()
    sbv, shv, stv, sdv = grad_hv.stride()

    # We now only have one kernel (vectorized) which replaces both direct/atomic
    ltv_fused_concat_scan_backward_vectorized_kernel[grid](
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
        sc_b,
        sc_t,
        sc_d,
        sgc_b,
        sgc_t,
        sgc_d,
        sbq,
        shq,
        stq,
        sdq,
        sbk,
        shk,
        stk,
        sdk,
        sbv,
        shv,
        stv,
        sdv,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        num_warps=warps,
    )


def _ltv_fused_concat_scan_triton_setup(ctx, inputs, output):
    combined, logit_bias, inits, NH, NKVH, da, r = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined, logit_bias, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_concat_scan_triton_backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, logit_bias, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r

    grad_hq = grad_hq.contiguous()
    grad_hk = grad_hk.contiguous()
    grad_hv = grad_hv.contiguous()

    grad_combined = torch.zeros_like(combined)
    grad_inits = torch.empty_like(inits)

    torch.ops.nanochat.ltv_fused_concat_scan_triton_backward_kernel(
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
    )

    # compute logit bias gradient as before
    TOTAL_H = NH + 2 * NKVH
    logit_offset = TOTAL_H * da * r
    grad_logit_bias = grad_combined[..., logit_offset:].sum(dim=(0, 1))

    return grad_combined, grad_logit_bias, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_fused_concat_scan_triton_op",
    ltv_fused_concat_scan_triton_backward,
    setup_context=_ltv_fused_concat_scan_triton_setup,
)
