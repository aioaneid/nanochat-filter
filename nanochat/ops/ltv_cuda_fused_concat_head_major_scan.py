import torch

try:
    import cpp_ewma_cuda

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        for f in [
            cpp_ewma_cuda.ltv_fused_concat_head_major_register_cast_scan_cuda_forward,
            cpp_ewma_cuda.ltv_fused_concat_head_major_register_cast_scan_cuda_backward,
        ]:
            torch.compiler.allow_in_graph(f)
except (ImportError, AttributeError):
    cpp_ewma_cuda = None


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension with head-major register-cast scan symbols "
            "is not available"
        )
    return cpp_ewma_cuda


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_scan_cuda_forward_kernel", "CUDA"
)
def ltv_fused_concat_head_major_register_cast_scan_cuda_forward_kernel_impl(
    combined,
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
    return _require_cpp_ewma_cuda().ltv_fused_concat_head_major_register_cast_scan_cuda_forward(
        combined, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
    )


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_scan_cuda_op", "CUDA"
)
def ltv_fused_concat_head_major_register_cast_scan_cuda_op_impl(
    combined, inits, NH, NKVH, da, r
):
    B, T, _ = combined.shape
    D = da * r

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

    torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_scan_cuda_forward_kernel(
        combined, inits, out_q, out_k, out_v, B, T, NH, NKVH, da, r
    )
    return out_q, out_k, out_v


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_scan_cuda_backward_kernel", "CUDA"
)
def ltv_fused_concat_head_major_register_cast_scan_cuda_backward_kernel_impl(
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
    NH,
    NKVH,
    da,
    r,
):
    B, T, _ = combined.shape

    _require_cpp_ewma_cuda().ltv_fused_concat_head_major_register_cast_scan_cuda_backward(
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
        NH,
        NKVH,
        da,
        r,
    )


def _ltv_fused_concat_head_major_register_cast_scan_cuda_setup(ctx, inputs, output):
    combined, inits, NH, NKVH, da, r = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r


def ltv_fused_concat_head_major_register_cast_scan_cuda_backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r

    grad_combined = torch.empty_like(combined, memory_format=torch.contiguous_format)
    # Inits is already contiguous
    grad_inits = torch.empty_like(inits, memory_format=torch.contiguous_format)

    torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_scan_cuda_backward_kernel(
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
        NH,
        NKVH,
        da,
        r,
    )

    return grad_combined, grad_inits, None, None, None, None


torch.library.register_autograd(
    "nanochat_cuda::ltv_fused_concat_head_major_register_cast_scan_cuda_op",
    ltv_fused_concat_head_major_register_cast_scan_cuda_backward,
    setup_context=_ltv_fused_concat_head_major_register_cast_scan_cuda_setup,
)
