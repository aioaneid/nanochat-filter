import torch
import nanochat.ops.ltv_cuda_meta  # noqa: F401

try:
    import cpp_ewma_cuda

    cpp_ewma_cuda_import_error = None

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        for f in [
            cpp_ewma_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward,
            cpp_ewma_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward,
        ]:
            torch.compiler.allow_in_graph(f)
except (ImportError, AttributeError) as err:
    cpp_ewma_cuda = None
    cpp_ewma_cuda_import_error = err


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise ImportError(
            "cpp_ewma_cuda is required to run "
            "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op"
        ) from cpp_ewma_cuda_import_error
    return cpp_ewma_cuda


# ---------------- FORWARD ----------------


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel",
    "CUDA",
)
def forward_kernel(
    combined,
    inits,
    logit_bias,
    out_q,
    out_k,
    out_v,
    B,  # <-- ADDED
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
    ext = _require_cpp_ewma_cuda()
    return ext.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
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
    )


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op",
    "CUDA",
)
def forward_op(
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
    use_sigmoid: bool,
):
    B, T, _ = combined.shape
    D = da * r

    assert combined.dtype == inits.dtype

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

    torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel(
        combined,
        inits,
        logit_bias,
        out_q,
        out_k,
        out_v,
        B,  # <-- pass B before T
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
    )
    return out_q, out_k, out_v


# ---------------- BACKWARD ----------------


@torch.library.impl(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel",
    "CUDA",
)
def backward_kernel(
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
    use_sigmoid: bool,
):
    B, T, _ = combined.shape

    ext = _require_cpp_ewma_cuda()
    return ext.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
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
        B,
        T,
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
    )


def _setup(ctx, inputs, output):
    (
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
    ) = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined, inits, logit_bias, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r
    ctx.K = K
    ctx.P = P
    ctx.Q = Q
    ctx.scan_threads_backward = scan_threads_backward
    ctx.r_threads_backward = r_threads_backward
    ctx.r_items_backward = r_items_backward
    ctx.use_sigmoid = use_sigmoid


def backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, inits, logit_bias, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    K, P, Q = ctx.K, ctx.P, ctx.Q
    scan_threads_backward = ctx.scan_threads_backward
    r_threads_backward = ctx.r_threads_backward
    r_items_backward = ctx.r_items_backward
    use_sigmoid = ctx.use_sigmoid

    grad_combined = torch.empty_like(combined)
    grad_inits = torch.empty_like(inits, memory_format=torch.contiguous_format)
    grad_logit_bias = torch.zeros_like(
        logit_bias, memory_format=torch.contiguous_format
    )

    torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel(
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
    )

    # Return grads for all 16 inputs (3 tensor grads + 13 Nones)
    return (
        grad_combined,
        grad_inits,
        grad_logit_bias,
        None,  # NH
        None,  # NKVH
        None,  # da
        None,  # r
        None,  # K
        None,  # P
        None,  # Q
        None,  # scan_threads_forward
        None,  # r_threads_forward
        None,  # r_items_forward
        None,  # scan_threads_backward
        None,  # r_threads_backward
        None,  # r_items_backward
        None,  # use_sigmoid
    )


torch.library.register_autograd(
    "nanochat_cuda::ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op",
    backward,
    setup_context=_setup,
)
