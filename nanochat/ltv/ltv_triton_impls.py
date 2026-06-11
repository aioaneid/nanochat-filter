import torch
import logging

try:
    from nanochat.ltv.ltv_metal_core import core_scan_backward_spatial
except ImportError:
    # On some systems, the folder structure might not be available or some deps missing, but the op wrappers should stay safe
    core_scan_backward_spatial = None

logger = logging.getLogger(__name__)

def ltv_associative_scan_triton(initial_value, alpha, x):
    # This op uses associative scan (log T) in Triton
    return torch.ops.nanochat.ltv_associative_scan_triton_op(initial_value, alpha, x)

def ltv_associative_scan_backward(ctx, grad_h_x, kernel_fn):
    h0, alpha, h_out_user = ctx.saved_tensors

    if core_scan_backward_spatial is None:
        raise ImportError("Could not import core_scan_backward_spatial for ltv_associative_scan_backward")

    # Use the Spatial-Major core backward
    grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_spatial(
        h0,
        alpha.transpose(1, 2),  # Prepare for internal (B, Da, T) logic
        h_out_user.transpose(1, 2),
        grad_h_x,
        kernel_fn,
    )

    return (
        grad_init_mps.to(dtype=h0.dtype),
        grad_alpha_mps_t.to(dtype=alpha.dtype),
        grad_x_mps_t.to(dtype=ctx.dtype_x),
    )

def _ltv_metal_setup(ctx, inputs, output):
    h0, alpha, x = inputs
    # Save as they are; the core_backward functions handle the specific transpositions
    ctx.save_for_backward(h0, alpha, output)
    ctx.dtype_x = x.dtype

# Register autograd for the Triton op if it exists
def register_triton_autograd():
    if hasattr(torch.ops.nanochat, "ltv_associative_scan_triton_op"):
        torch.library.register_autograd(
            "nanochat::ltv_associative_scan_triton_op",
            lambda ctx, grad: ltv_associative_scan_backward(
                ctx,
                grad,
                torch.ops.nanochat.ltv_associative_scan_triton_kernel,
            ),
            setup_context=_ltv_metal_setup,
        )

# Perform registration on import
try:
    register_triton_autograd()
except (AttributeError, RuntimeError):
    # Safe to ignore if op not defined yet or already registered
    pass
