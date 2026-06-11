import torch

try:
    import cpp_ewma_cuda

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        for f in [
            cpp_ewma_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_forward,
            cpp_ewma_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_backward,
        ]:
            torch.compiler.allow_in_graph(f)
except (ImportError, AttributeError):
    cpp_ewma_cuda = None


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension with fused concat register-cast Blelloch "
            "scan symbols is not available"
        )
    return cpp_ewma_cuda


# ---------------- FORWARD ----------------


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_kernel",
    "CUDA",
)
def forward_kernel(
    combined,
    inits,
    out_q,
    out_k,
    out_v,
    T,
    NH,
    NKVH,
    da,
    r,
    block_threads,
    items_per_thread,
    use_sigmoid,
):
    B, _, _ = combined.shape
    return _require_cpp_ewma_cuda().ltv_fused_concat_register_cast_blelloch_scan_cuda_forward(
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
        block_threads,
        items_per_thread,
        use_sigmoid,
    )


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_op",
    "CUDA",
)
def forward_op(
    combined,
    inits,
    NH,
    NKVH,
    da,
    r,
    block_threads=128,
    items_per_thread=8,
    use_sigmoid: bool = False,
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

    torch.ops.nanochat_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_kernel(
        combined,
        inits,
        out_q,
        out_k,
        out_v,
        T,
        NH,
        NKVH,
        da,
        r,
        block_threads,
        items_per_thread,
        use_sigmoid,
    )
    return out_q, out_k, out_v


# ---------------- BACKWARD ----------------


@torch.library.impl(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_kernel",
    "CUDA",
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
    NH,
    NKVH,
    da,
    r,
    block_threads,
    items_per_thread,
    use_sigmoid: bool,
):
    B, T, _ = combined.shape

    return _require_cpp_ewma_cuda().ltv_fused_concat_register_cast_blelloch_scan_cuda_backward(
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
        B,
        T,
        NH,
        NKVH,
        da,
        r,
        block_threads,
        items_per_thread,
        use_sigmoid,
    )


def _setup(ctx, inputs, output):
    # inputs: (combined, inits, NH, NKVH, da, r, block_threads, items_per_thread, use_sigmoid)
    combined, inits, NH, NKVH, da, r, block_threads, items_per_thread, use_sigmoid = (
        inputs
    )
    hq, hk, hv = output
    ctx.save_for_backward(combined, inits, hq, hk, hv)
    ctx.NH = NH
    ctx.NKVH = NKVH
    ctx.da = da
    ctx.r = r
    ctx.block_threads = block_threads
    ctx.items_per_thread = items_per_thread
    ctx.use_sigmoid = use_sigmoid


def backward(ctx, grad_hq, grad_hk, grad_hv):
    combined, inits, hq, hk, hv = ctx.saved_tensors
    NH, NKVH, da, r = ctx.NH, ctx.NKVH, ctx.da, ctx.r
    block_threads, items_per_thread = ctx.block_threads, ctx.items_per_thread
    use_sigmoid = ctx.use_sigmoid

    B, T, _ = combined.shape
    D = da * r
    total_h = NH + 2 * NKVH

    grad_combined = torch.empty(
        combined.shape,
        dtype=combined.dtype,
        device=combined.device,
        memory_format=torch.contiguous_format,
    )
    grad_inits = torch.empty_like(inits, memory_format=torch.contiguous_format)

    if r > 1:
        grad_logits = torch.zeros(
            B,
            T,
            total_h,
            da,
            dtype=torch.float32,
            device=combined.device,
        )
    else:
        grad_logits = torch.empty(1, dtype=torch.float32, device=combined.device)

    torch.ops.nanochat_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_kernel(
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
        NH,
        NKVH,
        da,
        r,
        block_threads,
        items_per_thread,
        use_sigmoid,
    )

    if r > 1:
        offset_l = total_h * D
        gl = grad_logits.reshape(B, T, total_h * da)
        grad_combined[:, :, offset_l : offset_l + total_h * da] = gl

    return grad_combined, grad_inits, None, None, None, None, None, None, None


torch.library.register_autograd(
    "nanochat_cuda::ltv_fused_concat_register_cast_blelloch_scan_cuda_op",
    backward,
    setup_context=_setup,
)
