import torch


try:
    import cpp_ewma_cuda

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        for f in [cpp_ewma_cuda.ltv_fused_scan_forward, cpp_ewma_cuda.ltv_fused_scan_backward]:
            torch.compiler.allow_in_graph(f)
except (ImportError, AttributeError):
    cpp_ewma_cuda = None


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension with fused scan symbols is not available"
        )
    return cpp_ewma_cuda


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_forward_kernel", "CUDA")
def ltv_fused_scan_cuda_forward_kernel_impl(
    q,
    k,
    v,
    logits,
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
):
    # Pass all 14 to your C++ extension
    return _require_cpp_ewma_cuda().ltv_fused_scan_forward(
        q, k, v, logits, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
    )


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_op", "CUDA")
def ltv_fused_scan_cuda_op_impl(q, k, v, logits, inits, NH, NKVH, da, r):
    B, T, _, D = q.shape

    out_q = torch.empty((B, NH, T, D), device=q.device, dtype=q.dtype)
    out_k = torch.empty((B, NKVH, T, D), device=k.device, dtype=k.dtype)
    out_v = torch.empty((B, NKVH, T, D), device=v.device, dtype=v.dtype)

    # Ensure you are passing B and T here too!
    torch.ops.nanochat_cuda.ltv_fused_scan_cuda_forward_kernel(
        q, k, v, logits, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
    )
    return out_q, out_k, out_v


@torch.library.impl("nanochat_cuda::ltv_fused_scan_cuda_backward_kernel", "CUDA")
def ltv_fused_scan_cuda_backward_kernel_impl(
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
    da,
    r,
):
    B, T, _, _ = q.shape

    hq_cpp = hq.contiguous()
    hk_cpp = hk.contiguous()
    hv_cpp = hv.contiguous()
    grad_hq_cpp = grad_hq.contiguous()
    grad_hk_cpp = grad_hk.contiguous()
    grad_hv_cpp = grad_hv.contiguous()

    _require_cpp_ewma_cuda().ltv_fused_scan_backward(
        q,
        k,
        v,
        logits,
        inits,
        hq_cpp,
        hk_cpp,
        hv_cpp,
        grad_hq_cpp,
        grad_hk_cpp,
        grad_hv_cpp,
        grad_q,
        grad_k,
        grad_v,
        grad_logits,
        grad_inits,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
    )


def _ltv_fused_scan_cuda_setup(ctx, inputs, output):
    q, k, v, logits, inits, NH, NKVH, da, r = inputs
    hq, hk, hv = output
    ctx.save_for_backward(q, k, v, logits, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_scan_cuda_backward(ctx, grad_hq, grad_hk, grad_hv):
    q, k, v, logits, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r

    grad_hq = grad_hq.contiguous()
    grad_hk = grad_hk.contiguous()
    grad_hv = grad_hv.contiguous()

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_logits = torch.empty_like(logits)
    grad_inits = torch.empty_like(inits)

    torch.ops.nanochat_cuda.ltv_fused_scan_cuda_backward_kernel(
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
        da,
        r,
    )

    return grad_q, grad_k, grad_v, grad_logits, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat_cuda::ltv_fused_scan_cuda_op",
    ltv_fused_scan_cuda_backward,
    setup_context=_ltv_fused_scan_cuda_setup,
)
