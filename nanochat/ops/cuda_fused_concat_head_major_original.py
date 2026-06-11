import torch

try:
    import cpp_ewma_cuda

    cpp_ewma_cuda_import_error = None

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        for f in [
            cpp_ewma_cuda.cuda_fused_concat_head_major_original_cuda_forward,
            cpp_ewma_cuda.cuda_fused_concat_head_major_original_cuda_backward,
        ]:
            torch.compiler.allow_in_graph(f)
except (ImportError, AttributeError) as err:
    cpp_ewma_cuda = None
    cpp_ewma_cuda_import_error = err


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension with head-major original symbols "
            "is not available"
        ) from cpp_ewma_cuda_import_error
    return cpp_ewma_cuda


@torch.library.impl(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_forward_kernel",
    "CUDA",
)
def forward_kernel(
    combined,
    out_q,
    out_k,
    out_v,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
):
    ext = _require_cpp_ewma_cuda()
    return ext.cuda_fused_concat_head_major_original_cuda_forward(
        combined,
        out_q,
        out_k,
        out_v,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
    )


@torch.library.impl(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_op",
    "CUDA",
)
def forward_op(
    combined,
    NH,
    NKVH,
    da,
    r,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
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

    torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_forward_kernel(
        combined,
        out_q,
        out_k,
        out_v,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
    )
    return out_q, out_k, out_v


@torch.library.impl(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_backward_kernel",
    "CUDA",
)
def backward_kernel(
    combined,
    grad_hq,
    grad_hk,
    grad_hv,
    grad_combined,
    B,
    T,
    NH,
    NKVH,
    da,
    r,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
):
    ext = _require_cpp_ewma_cuda()
    return ext.cuda_fused_concat_head_major_original_cuda_backward(
        combined,
        grad_hq,
        grad_hk,
        grad_hv,
        grad_combined,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
    )


def _setup(ctx, inputs, output):
    (
        combined,
        NH,
        NKVH,
        da,
        r,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
    ) = inputs
    hq, hk, hv = output
    ctx.save_for_backward(combined)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r
    ctx.scan_threads_backward = scan_threads_backward
    ctx.r_threads_backward = r_threads_backward
    ctx.r_items_backward = r_items_backward


def backward(ctx, grad_hq, grad_hk, grad_hv):
    (combined,) = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    scan_threads_backward = ctx.scan_threads_backward
    r_threads_backward = ctx.r_threads_backward
    r_items_backward = ctx.r_items_backward
    B, T, _ = combined.shape

    grad_combined = torch.zeros_like(combined, memory_format=torch.preserve_format)

    torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_backward_kernel(
        combined,
        grad_hq,
        grad_hk,
        grad_hv,
        grad_combined,
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
    )

    return (
        grad_combined,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nanochat_cuda::cuda_fused_concat_head_major_original_cuda_op",
    backward,
    setup_context=_setup,
)
