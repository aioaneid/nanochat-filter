import torch

try:
    import rust_ewma
except ImportError:
    rust_ewma = None


@torch.library.impl("nanochat::ltv_scan_metal_kernel", "MPS")
def ltv_scan_metal_kernel_impl(
    alpha, x, out_alpha, out_x, t_seq: int, parallel_a: int, r: int
):
    rust_ewma.ltv_scan_metal(
        alpha,
        x,
        out_alpha,
        out_x,
        t_seq,
        parallel_a,
        r,
    )


@torch.library.impl("nanochat::ltv_noop_metal_kernel", "MPS")
def ltv_noop_metal_kernel_impl(
    h_init, alpha, x, out, t_dim: int, parallel_a: int, r: int
):
    rust_ewma.ltv_noop_metal(
        h_init,
        alpha,
        x,
        out,
        t_dim,
        parallel_a,
        r,
    )


@torch.library.impl("nanochat::ltv_plane_metal_kernel", "MPS")
def ltv_plane_metal_kernel_impl(
    alpha, x, out_alpha, out_x, t_seq: int, parallel_a: int, r: int, schedule: str
):
    rust_ewma.ltv_plane_metal(
        alpha,
        x,
        out_alpha,
        out_x,
        t_seq,
        parallel_a,
        r,
        schedule,
    )


@torch.library.impl("nanochat::ltv_shared_metal_kernel", "MPS")
def ltv_shared_metal_kernel_impl(
    alpha, x, out_alpha, out_x, t_seq: int, parallel_a: int, r: int, schedule: str
):
    rust_ewma.ltv_shared_metal(
        alpha,
        x,
        out_alpha,
        out_x,
        t_seq,
        parallel_a,
        r,
        schedule,
    )


@torch.library.impl("nanochat::ltv_fused_scan_metal_forward_kernel", "MPS")
def ltv_fused_scan_metal_forward_kernel_impl(
    q, k, v, logits, inits, out_q, out_k, out_v, NH: int, NKVH: int, da: int, r: int
):
    B, T, _, D = q.shape
    rust_ewma.ltv_fused_scan_metal_forward(
        q, k, v, logits, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
    )


@torch.library.impl("nanochat::ltv_fused_scan_metal_backward_kernel", "MPS")
def ltv_fused_scan_metal_backward_kernel_impl(
    q,
    k,
    v,
    logits,
    inits,
    h_fwd_q,
    h_fwd_k,
    h_fwd_v,
    grad_out_q,
    grad_out_k,
    grad_out_v,
    dx_q,
    dx_k,
    dx_v,
    dl,
    di,
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
    NH: int,
    NKVH: int,
    da: int,
    r: int,
):
    B, T, _, D = q.shape
    rust_ewma.ltv_fused_scan_metal_backward(
        q,
        k,
        v,
        logits,
        inits,
        h_fwd_q,
        h_fwd_k,
        h_fwd_v,
        grad_out_q,
        grad_out_k,
        grad_out_v,
        dx_q,
        dx_k,
        dx_v,
        dl,
        di,
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
    )


@torch.library.impl("nanochat::ltv_fused_linear_scan_metal_forward_kernel", "MPS")
def ltv_fused_linear_scan_metal_forward_kernel_impl(
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
    da: int,
    r: int,
):
    B, T, NH, _ = q.shape
    _, _, NKVH, _ = k.shape
    rust_ewma.ltv_fused_linear_scan_metal_forward(
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
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        n_embd=da * NH * r,
    )


@torch.library.impl("nanochat::ltv_fused_linear_scan_metal_backward_kernel", "MPS")
def ltv_fused_linear_scan_metal_backward_kernel_impl(
    q,
    k,
    v,
    x_gate,
    w_gate,
    b_gate,
    inits,
    h_fwd_q,
    h_fwd_k,
    h_fwd_v,
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
    da: int,
    r: int,
):
    B, T, NH, _ = q.shape
    _, _, NKVH, _ = k.shape

    rust_ewma.ltv_fused_linear_scan_metal_backward(
        q,
        k,
        v,
        x_gate,
        w_gate,
        b_gate,
        inits,
        h_fwd_q,
        h_fwd_k,
        h_fwd_v,
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
    )


@torch.library.impl("nanochat::ltv_fused_concat_scan_metal_forward_kernel", "MPS")
def ltv_fused_concat_scan_metal_forward_kernel_impl(
    combined, logit_bias, inits, out_q, out_k, out_v, sc_b, sc_t, sc_d, NH, NKVH, da, r
):
    B, T, _ = combined.shape
    rust_ewma.ltv_fused_concat_scan_metal_forward(
        combined,
        logit_bias,
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
        da,
        r,
    )


@torch.library.impl("nanochat::ltv_fused_concat_scan_metal_backward_kernel", "MPS")
def ltv_fused_concat_scan_metal_backward_kernel_impl(
    combined,
    logit_bias,
    inits,
    h_fwd_q,
    h_fwd_k,
    h_fwd_v,
    grad_out_q,
    grad_out_k,
    grad_out_v,
    grad_combined,
    grad_inits,
    sc_b,
    sc_t,
    sc_d,
    sgc_b,
    sgc_t,
    sgc_d,  # New explicit strides
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
    NH,
    NKVH,
    da,
    r,
):
    B, T, _ = combined.shape
    rust_ewma.ltv_fused_concat_scan_metal_backward(
        combined,
        logit_bias,
        inits,
        h_fwd_q,
        h_fwd_k,
        h_fwd_v,
        grad_out_q,
        grad_out_k,
        grad_out_v,
        grad_combined,
        grad_inits,
        B,
        T,
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
        NH,
        NKVH,
        da,
        r,
    )
