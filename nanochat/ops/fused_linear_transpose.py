import torch

try:
    import cpp_ewma_cuda

    cpp_ewma_cuda_import_error = None
except (ImportError, AttributeError) as err:
    cpp_ewma_cuda = None
    cpp_ewma_cuda_import_error = err


def _require_cpp_ewma_cuda():
    if cpp_ewma_cuda is None:
        raise ImportError(
            "cpp_ewma_cuda is required to run fused_linear_transpose"
        ) from cpp_ewma_cuda_import_error
    return cpp_ewma_cuda


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose",
    "CUDA",
)
def fused_linear_transpose(x, weight, out):
    # Require native extension; do not perform Python-side computation here.
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension is required for fused_linear_transpose"
        )

    # Require contiguous inputs for native kernel
    if not x.is_contiguous() or not weight.is_contiguous() or not out.is_contiguous():
        raise RuntimeError(
            "fused_linear_transpose requires contiguous x, weight, and out tensors"
        )

    ext = _require_cpp_ewma_cuda()
    ext.fused_linear_transpose(x, weight, out)

    return None


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose_cuda_op",
    "CUDA",
)
def fused_linear_transpose_cuda_op(x, weight):
    # Functional operator: allocate output tensor here and call the kernel.
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension is required for fused_linear_transpose"
        )

    B, T, K = x.shape
    F = weight.size(0)

    out = torch.zeros((B, F, T), dtype=x.dtype, device=x.device)

    torch.ops.nanochat_cuda.fused_linear_transpose(x, weight, out)
    return out


@torch.library.impl(
    "nanochat_cuda::fused_linear_transpose_backward_kernel",
    "CUDA",
)
def fused_linear_transpose_backward_kernel(x, weight, grad_out, grad_x, grad_weight):
    # CUDA impl: MUST call into compiled extension. Do not compute gradients
    # in Python here; callers must have the native extension available.
    if cpp_ewma_cuda is None:
        raise RuntimeError(
            "cpp_ewma_cuda extension is not available for fused_linear_transpose backward"
        )

    # Ensure inputs are contiguous so the C++ kernel can safely access data_ptr
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or not (grad_out.is_contiguous() or _is_transposed_grad_out(grad_out))
    ):
        raise RuntimeError(
            "fused_linear_transpose_backward_kernel requires contiguous x and weight, "
            f"and either contiguous or transposed grad_out. Found strides: x={x.stride()}, weight={weight.stride()}, grad_out={grad_out.stride()}"
        )

    ext = _require_cpp_ewma_cuda()
    ext.fused_linear_transpose_backward(x, weight, grad_out, grad_x, grad_weight)
    return None


def _fused_linear_transpose_setup(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _is_transposed_grad_out(grad_out):
    return (
        grad_out.stride(1) == 1
        and grad_out.stride(2) == grad_out.size(1)
        and grad_out.stride(0) == grad_out.size(1) * grad_out.size(2)
    )


def _fused_linear_transpose_backward(ctx, grad_out):
    x, weight = ctx.saved_tensors

    # Fail early if inputs aren't contiguous; kernel requires contiguous buffers
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or not (grad_out.is_contiguous() or _is_transposed_grad_out(grad_out))
    ):
        raise RuntimeError(
            "fused_linear_transpose backward requires contiguous x and weight, "
            f"and either contiguous or transposed grad_out. Found strides: x={x.stride()}, weight={weight.stride()}, grad_out={grad_out.stride()}"
        )

    # Allocate output buffers in the same dtype/device/shape as the inputs
    grad_x = torch.zeros_like(x, memory_format=torch.contiguous_format)
    grad_weight = torch.zeros_like(weight, memory_format=torch.contiguous_format)

    # Call the native kernel which must handle dtype-specific computation
    torch.ops.nanochat_cuda.fused_linear_transpose_backward_kernel(
        x, weight, grad_out, grad_x, grad_weight
    )

    return grad_x, grad_weight, None


torch.library.register_autograd(
    "nanochat_cuda::fused_linear_transpose_cuda_op",
    _fused_linear_transpose_backward,
    setup_context=_fused_linear_transpose_setup,
)
