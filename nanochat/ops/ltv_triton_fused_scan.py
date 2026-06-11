import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def ltv_fused_scan_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        logits_ptr,
        inits_ptr,
        hq_ptr,
        hk_ptr,
        hv_ptr,
        stride_q_b: tl.constexpr,
        stride_q_t: tl.constexpr,
        stride_q_h: tl.constexpr,
        stride_q_d: tl.constexpr,
        stride_k_b: tl.constexpr,
        stride_k_t: tl.constexpr,
        stride_k_h: tl.constexpr,
        stride_k_d: tl.constexpr,
        stride_v_b: tl.constexpr,
        stride_v_t: tl.constexpr,
        stride_v_h: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_l_b: tl.constexpr,
        stride_l_t: tl.constexpr,
        stride_l_h: tl.constexpr,
        stride_l_g: tl.constexpr,
        stride_i_b: tl.constexpr,
        stride_i_h: tl.constexpr,
        stride_i_d: tl.constexpr,
        stride_hq_b: tl.constexpr,
        stride_hq_t: tl.constexpr,
        stride_hq_h: tl.constexpr,
        stride_hq_d: tl.constexpr,
        stride_hk_b: tl.constexpr,
        stride_hk_t: tl.constexpr,
        stride_hk_h: tl.constexpr,
        stride_hk_d: tl.constexpr,
        stride_hv_b: tl.constexpr,
        stride_hv_t: tl.constexpr,
        stride_hv_h: tl.constexpr,
        stride_hv_d: tl.constexpr,
        B: tl.constexpr,
        T: tl.constexpr,
        NH: tl.constexpr,
        NKVH: tl.constexpr,
        D: tl.constexpr,
        G: tl.constexpr,
        R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        TOTAL_H = NH + 2 * NKVH

        b_idx = pid // TOTAL_H
        h_idx = pid % TOTAL_H

        is_q = h_idx < NH
        is_k = (h_idx >= NH) and (h_idx < NH + NKVH)

        if is_q:
            h_local = h_idx
            in_ptr = q_ptr + b_idx * stride_q_b + h_local * stride_q_h
            in_stride_t = stride_q_t
            in_stride_d = stride_q_d
            out_ptr = hq_ptr + b_idx * stride_hq_b + h_local * stride_hq_h
            out_stride_t = stride_hq_t
            out_stride_d = stride_hq_d
        elif is_k:
            h_local = h_idx - NH
            in_ptr = k_ptr + b_idx * stride_k_b + h_local * stride_k_h
            in_stride_t = stride_k_t
            in_stride_d = stride_k_d
            out_ptr = hk_ptr + b_idx * stride_hk_b + h_local * stride_hk_h
            out_stride_t = stride_hk_t
            out_stride_d = stride_hk_d
        else:
            h_local = h_idx - NH - NKVH
            in_ptr = v_ptr + b_idx * stride_v_b + h_local * stride_v_h
            in_stride_t = stride_v_t
            in_stride_d = stride_v_d
            out_ptr = hv_ptr + b_idx * stride_hv_b + h_local * stride_hv_h
            out_stride_t = stride_hv_t
            out_stride_d = stride_hv_d

        logits_seq_ptr = logits_ptr + b_idx * stride_l_b + h_idx * stride_l_h
        init_head_ptr = inits_ptr + b_idx * stride_i_b + h_idx * stride_i_h

        d_offsets = tl.arange(0, D)
        g_offsets = tl.arange(0, G)

        # keep state flat
        h_prev = tl.load(init_head_ptr + d_offsets * stride_i_d)

        for t in tl.range(T):
            u_t = tl.load(
                in_ptr + t * in_stride_t + d_offsets * in_stride_d,
            )

            l_t = tl.load(
                logits_seq_ptr + t * stride_l_t + g_offsets * stride_l_g,
            )

            # create 2D views only for math
            h_prev_deep = h_prev.reshape((G, R))
            u_t_deep = u_t.reshape((G, R))

            alpha_t = tl.sigmoid(l_t)
            alpha_t_deep = alpha_t.reshape((G, 1))

            h_t_deep = (1.0 - alpha_t_deep) * h_prev_deep + alpha_t_deep * u_t_deep

            # flatten again for storage and next iteration
            h_prev = h_t_deep.reshape((D,))

            tl.store(
                out_ptr + t * out_stride_t + d_offsets * out_stride_d,
                h_prev,
            )

    @triton.jit
    def ltv_fused_scan_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        logits_ptr,
        inits_ptr,
        hq_ptr,
        hk_ptr,
        hv_ptr,
        grad_hq_ptr,
        grad_hk_ptr,
        grad_hv_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        grad_logits_ptr,
        grad_inits_ptr,
        stride_q_b: tl.constexpr,
        stride_q_t: tl.constexpr,
        stride_q_h: tl.constexpr,
        stride_q_d: tl.constexpr,
        stride_k_b: tl.constexpr,
        stride_k_t: tl.constexpr,
        stride_k_h: tl.constexpr,
        stride_k_d: tl.constexpr,
        stride_v_b: tl.constexpr,
        stride_v_t: tl.constexpr,
        stride_v_h: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_l_b: tl.constexpr,
        stride_l_t: tl.constexpr,
        stride_l_h: tl.constexpr,
        stride_l_g: tl.constexpr,
        stride_i_b: tl.constexpr,
        stride_i_h: tl.constexpr,
        stride_i_d: tl.constexpr,
        stride_hq_b: tl.constexpr,
        stride_hq_t: tl.constexpr,
        stride_hq_h: tl.constexpr,
        stride_hq_d: tl.constexpr,
        stride_hk_b: tl.constexpr,
        stride_hk_t: tl.constexpr,
        stride_hk_h: tl.constexpr,
        stride_hk_d: tl.constexpr,
        stride_hv_b: tl.constexpr,
        stride_hv_t: tl.constexpr,
        stride_hv_h: tl.constexpr,
        stride_hv_d: tl.constexpr,
        stride_ghq_b: tl.constexpr,
        stride_ghq_t: tl.constexpr,
        stride_ghq_h: tl.constexpr,
        stride_ghq_d: tl.constexpr,
        stride_ghk_b: tl.constexpr,
        stride_ghk_t: tl.constexpr,
        stride_ghk_h: tl.constexpr,
        stride_ghk_d: tl.constexpr,
        stride_ghv_b: tl.constexpr,
        stride_ghv_t: tl.constexpr,
        stride_ghv_h: tl.constexpr,
        stride_ghv_d: tl.constexpr,
        stride_gq_b: tl.constexpr,
        stride_gq_t: tl.constexpr,
        stride_gq_h: tl.constexpr,
        stride_gq_d: tl.constexpr,
        stride_gk_b: tl.constexpr,
        stride_gk_t: tl.constexpr,
        stride_gk_h: tl.constexpr,
        stride_gk_d: tl.constexpr,
        stride_gv_b: tl.constexpr,
        stride_gv_t: tl.constexpr,
        stride_gv_h: tl.constexpr,
        stride_gv_d: tl.constexpr,
        stride_gl_b: tl.constexpr,
        stride_gl_t: tl.constexpr,
        stride_gl_h: tl.constexpr,
        stride_gl_g: tl.constexpr,
        stride_gi_b: tl.constexpr,
        stride_gi_h: tl.constexpr,
        stride_gi_d: tl.constexpr,
        B: tl.constexpr,
        T: tl.constexpr,
        NH: tl.constexpr,
        NKVH: tl.constexpr,
        D: tl.constexpr,
        G: tl.constexpr,
        R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        TOTAL_H = NH + 2 * NKVH
        b_idx = pid // TOTAL_H
        h_idx = pid % TOTAL_H

        is_q = h_idx < NH
        is_k = (h_idx >= NH) and (h_idx < NH + NKVH)

        if is_q:
            h_local = h_idx
            in_ptr = q_ptr + b_idx * stride_q_b + h_local * stride_q_h
            in_stride_t = stride_q_t
            in_stride_d = stride_q_d
            out_ptr = hq_ptr + b_idx * stride_hq_b + h_local * stride_hq_h
            out_stride_t = stride_hq_t
            out_stride_d = stride_hq_d
            grad_out_ptr = grad_hq_ptr + b_idx * stride_ghq_b + h_local * stride_ghq_h
            grad_out_stride_t = stride_ghq_t
            grad_out_stride_d = stride_ghq_d
            grad_in_ptr = grad_q_ptr + b_idx * stride_gq_b + h_local * stride_gq_h
            grad_in_stride_t = stride_gq_t
            grad_in_stride_d = stride_gq_d
        elif is_k:
            h_local = h_idx - NH
            in_ptr = k_ptr + b_idx * stride_k_b + h_local * stride_k_h
            in_stride_t = stride_k_t
            in_stride_d = stride_k_d
            out_ptr = hk_ptr + b_idx * stride_hk_b + h_local * stride_hk_h
            out_stride_t = stride_hk_t
            out_stride_d = stride_hk_d
            grad_out_ptr = grad_hk_ptr + b_idx * stride_ghk_b + h_local * stride_ghk_h
            grad_out_stride_t = stride_ghk_t
            grad_out_stride_d = stride_ghk_d
            grad_in_ptr = grad_k_ptr + b_idx * stride_gk_b + h_local * stride_gk_h
            grad_in_stride_t = stride_gk_t
            grad_in_stride_d = stride_gk_d
        else:
            h_local = h_idx - NH - NKVH
            in_ptr = v_ptr + b_idx * stride_v_b + h_local * stride_v_h
            in_stride_t = stride_v_t
            in_stride_d = stride_v_d
            out_ptr = hv_ptr + b_idx * stride_hv_b + h_local * stride_hv_h
            out_stride_t = stride_hv_t
            out_stride_d = stride_hv_d
            grad_out_ptr = grad_hv_ptr + b_idx * stride_ghv_b + h_local * stride_ghv_h
            grad_out_stride_t = stride_ghv_t
            grad_out_stride_d = stride_ghv_d
            grad_in_ptr = grad_v_ptr + b_idx * stride_gv_b + h_local * stride_gv_h
            grad_in_stride_t = stride_gv_t
            grad_in_stride_d = stride_gv_d

        logits_seq_ptr = logits_ptr + b_idx * stride_l_b + h_idx * stride_l_h
        init_head_ptr = inits_ptr + b_idx * stride_i_b + h_idx * stride_i_h

        grad_logits_seq_ptr = (
            grad_logits_ptr + b_idx * stride_gl_b + h_idx * stride_gl_h
        )
        grad_init_head_ptr = grad_inits_ptr + b_idx * stride_gi_b + h_idx * stride_gi_h

        d_offsets = tl.arange(0, D)
        g_offsets = tl.arange(0, G)

        dh_next = tl.zeros([D], dtype=tl.float32)

        for t_rev in tl.range(T - 1, -1, -1):
            dh_ext = tl.load(
                grad_out_ptr
                + t_rev * grad_out_stride_t
                + d_offsets * grad_out_stride_d,
            )

            dh_t = dh_ext + dh_next

            x_t = tl.load(in_ptr + t_rev * in_stride_t + d_offsets * in_stride_d)

            l_t = tl.load(logits_seq_ptr + t_rev * stride_l_t + g_offsets * stride_l_g)

            alpha_t = tl.sigmoid(l_t)

            if t_rev > 0:
                h_prev = tl.load(
                    out_ptr + (t_rev - 1) * out_stride_t + d_offsets * out_stride_d
                )
            else:
                h_prev = tl.load(init_head_ptr + d_offsets * stride_i_d)

            # ---- reshape to (G,R) for broadcast math ----
            dh_t_deep = dh_t.reshape((G, R))
            x_t_deep = x_t.reshape((G, R))
            h_prev_deep = h_prev.reshape((G, R))
            alpha_deep = alpha_t.reshape((G, 1))

            # gradient w.r.t x_t
            grad_u_deep = dh_t_deep * alpha_deep
            grad_u_t = grad_u_deep.reshape((D,))

            tl.store(
                grad_in_ptr + t_rev * grad_in_stride_t + d_offsets * grad_in_stride_d,
                grad_u_t,
            )

            # gradient w.r.t logits
            dl_deep = (
                dh_t_deep * (x_t_deep - h_prev_deep) * alpha_deep * (1.0 - alpha_deep)
            )

            grad_l_t = tl.sum(dl_deep, axis=1)

            tl.store(
                grad_logits_seq_ptr + t_rev * stride_gl_t + g_offsets * stride_gl_g,
                grad_l_t,
            )

            # dh for next iteration
            dh_next_deep = dh_t_deep * (1.0 - alpha_deep)
            dh_next = dh_next_deep.reshape((D,))

        # gradient for initial state
        tl.store(
            grad_init_head_ptr + d_offsets * stride_gi_d,
            dh_next,
        )


@torch.library.impl("nanochat::ltv_fused_scan_triton_op", "default")
def ltv_fused_scan_triton_op(q, k, v, logits, inits, NH, NKVH, da, r):
    # inputs are float32
    q = q.float().contiguous()
    k = k.float().contiguous()
    v = v.float().contiguous()
    logits = logits.float().contiguous()
    inits = inits.float().contiguous()

    B, T, _, D = q.shape
    hq = q.new_empty((B, NH, T, D))
    hk = k.new_empty((B, NKVH, T, D))
    hv = v.new_empty((B, NKVH, T, D))

    if triton is None:
        raise RuntimeError(
            "Triton is not available, cannot run ltv_fused_scan_triton_op"
        )

    grid = (B * (NH + 2 * NKVH),)

    ltv_fused_scan_fwd_kernel[grid](
        q,
        k,
        v,
        logits,
        inits,
        hq,
        hk,
        hv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        logits.stride(3),
        inits.stride(0),
        inits.stride(1),
        inits.stride(2),
        hq.stride(0),
        hq.stride(2),
        hq.stride(1),
        hq.stride(3),
        hk.stride(0),
        hk.stride(2),
        hk.stride(1),
        hk.stride(3),
        hv.stride(0),
        hv.stride(2),
        hv.stride(1),
        hv.stride(3),
        B,
        T,
        NH,
        NKVH,
        D,
        da,
        r,
    )
    return hq, hk, hv


@torch.library.impl("nanochat::ltv_fused_scan_triton_backward_kernel", "default")
def ltv_fused_scan_triton_backward_kernel_impl(
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
    G,
    r,
):
    if triton is None:
        raise RuntimeError(
            "Triton is not available, cannot run ltv_fused_scan_triton_backward_kernel"
        )

    B, T, _, D = q.shape
    grid = (B * (NH + 2 * NKVH),)
    ltv_fused_scan_bwd_kernel[grid](
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
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        logits.stride(3),
        inits.stride(0),
        inits.stride(1),
        inits.stride(2),
        hq.stride(0),
        hq.stride(2),
        hq.stride(1),
        hq.stride(3),
        hk.stride(0),
        hk.stride(2),
        hk.stride(1),
        hk.stride(3),
        hv.stride(0),
        hv.stride(2),
        hv.stride(1),
        hv.stride(3),
        grad_hq.stride(0),
        grad_hq.stride(2),
        grad_hq.stride(1),
        grad_hq.stride(3),
        grad_hk.stride(0),
        grad_hk.stride(2),
        grad_hk.stride(1),
        grad_hk.stride(3),
        grad_hv.stride(0),
        grad_hv.stride(2),
        grad_hv.stride(1),
        grad_hv.stride(3),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_q.stride(3),
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        grad_k.stride(3),
        grad_v.stride(0),
        grad_v.stride(1),
        grad_v.stride(2),
        grad_v.stride(3),
        grad_logits.stride(0),
        grad_logits.stride(1),
        grad_logits.stride(2),
        grad_logits.stride(3),
        grad_inits.stride(0),
        grad_inits.stride(1),
        grad_inits.stride(2),
        B,
        T,
        NH,
        NKVH,
        D,
        G,
        r,
    )


def _ltv_fused_scan_triton_setup(ctx, inputs, output):
    q, k, v, logits, inits, NH, NKVH, da, r = inputs
    hq, hk, hv = output
    ctx.save_for_backward(q, k, v, logits, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_scan_triton_backward(ctx, grad_hq, grad_hk, grad_hv):
    q, k, v, logits, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, G, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r

    grad_hq = grad_hq.contiguous()
    grad_hk = grad_hk.contiguous()
    grad_hv = grad_hv.contiguous()

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_logits = torch.empty_like(logits)
    grad_inits = torch.empty_like(inits)

    torch.ops.nanochat.ltv_fused_scan_triton_backward_kernel(
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
        G,
        r,
    )

    return grad_q, grad_k, grad_v, grad_logits, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat::ltv_fused_scan_triton_op",
    ltv_fused_scan_triton_backward,
    setup_context=_ltv_fused_scan_triton_setup,
)
