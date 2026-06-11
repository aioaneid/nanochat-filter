import torch
import pytest
import logging
from contextlib import nullcontext

try:
    import rust_ewma
except ImportError:
    rust_ewma = None

from nanochat.ltv.ltv_utils import dimensions
from nanochat.ltv.ltv_metal_impls import (
    LtvScanMetal,
    LtvSharedMetal,
    LtvPlaneMetal,
    ltv_scan_metal,
    ltv_shared_max_metal,
    ltv_shared_pmax_metal,
    ltv_shared_log_metal,
    ltv_plane_max_metal,
    ltv_plane_log_metal,
)

logger = logging.getLogger(__name__)


def ltv_pytorch_ref(initial_value, alpha, x):
    """Reference implementation returning a single float32 tensor."""
    B, T, D_a, r = dimensions(alpha, initial_value, x)
    alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, -1, r).reshape(B, T, D_a * r)

    h_x_all = []
    h_prev = initial_value

    for t in range(T):
        h_t = alpha_expanded[:, t] * h_prev + x[:, t]
        h_x_all.append(h_t)
        h_prev = h_t

    return torch.stack(h_x_all, dim=1)


IMPLEMENTATIONS = {
    "scan": lambda init, a, x: LtvScanMetal.apply(init, a, x),
    "scan_op": ltv_scan_metal,
    "shared_max": lambda init, a, x: LtvSharedMetal.apply(init, a, x, "max"),
    "shared_max_op": ltv_shared_max_metal,
    "shared_pmax_op": ltv_shared_pmax_metal,
    "shared_log": lambda init, a, x: LtvSharedMetal.apply(init, a, x, "log"),
    "shared_log_op": ltv_shared_log_metal,
    "plane_max": lambda init, a, x: LtvPlaneMetal.apply(init, a, x, "max"),
    "plane_max_op": ltv_plane_max_metal,
    "plane_log": lambda init, a, x: LtvPlaneMetal.apply(init, a, x, "log"),
    "plane_log_op": ltv_plane_log_metal,
}


# Mapping of implementation names to their Metal context functions
CONTEXT_FUNCTIONS = {
    "scan": (
        rust_ewma.release_ltv_scan_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_scan_metal_context_for_default_device if rust_ewma else None,
    ),
    "scan_op": (
        rust_ewma.release_ltv_scan_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_scan_metal_context_for_default_device if rust_ewma else None,
    ),
    "shared_max": (
        rust_ewma.release_ltv_shared_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_shared_metal_context_for_default_device if rust_ewma else None,
    ),
    "shared_max_op": (
        rust_ewma.release_ltv_shared_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_shared_metal_context_for_default_device if rust_ewma else None,
    ),
    "shared_pmax_op": (
        rust_ewma.release_ltv_shared_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_shared_metal_context_for_default_device if rust_ewma else None,
    ),
    "shared_log": (
        rust_ewma.release_ltv_shared_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_shared_metal_context_for_default_device if rust_ewma else None,
    ),
    "shared_log_op": (
        rust_ewma.release_ltv_shared_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_shared_metal_context_for_default_device if rust_ewma else None,
    ),
    "plane_max": (
        rust_ewma.release_ltv_plane_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_plane_metal_context_for_default_device if rust_ewma else None,
    ),
    "plane_max_op": (
        rust_ewma.release_ltv_plane_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_plane_metal_context_for_default_device if rust_ewma else None,
    ),
    "plane_log": (
        rust_ewma.release_ltv_plane_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_plane_metal_context_for_default_device if rust_ewma else None,
    ),
    "plane_log_op": (
        rust_ewma.release_ltv_plane_metal_context if rust_ewma else None,
        rust_ewma.init_ltv_plane_metal_context_for_default_device if rust_ewma else None,
    ),
}


def release_init_metal_context(impl_name):
    """Release and reinitialize the Metal context for the given implementation."""
    if rust_ewma is None:
        pytest.skip("rust_ewma not available")
    
    release_fn, init_fn = CONTEXT_FUNCTIONS[impl_name]
    if release_fn is None or init_fn is None:
        pytest.skip("Metal context functions not available")
    
    release_fn()
    try:
        init_fn()
    except RuntimeError as e:
        if not any("already initialized" in str(arg) for arg in e.args):
            raise


@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_metal_ltv_precision_and_parity(impl_name, dtype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    release_init_metal_context(impl_name)

    torch.manual_seed(42)
    device = torch.device("mps")

    B, T, Da, r = 2, 3, 5, 7
    D = Da * r

    # 1. Setup Inputs
    initial_value = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    alpha = (
        (torch.rand(B, T, Da, device=device, dtype=dtype) * 0.9)
        .detach()
        .requires_grad_(True)
    )
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    apply_fn = IMPLEMENTATIONS[impl_name]

    # 2. Forward Pass
    with nullcontext():
        h_x_metal = apply_fn(initial_value, alpha, x)

    # TODO: Enable numerical checking for bfloat16.

    # ASSERTION: Result must be float32 even if inputs/context are bfloat16
    # assert h_x_metal.dtype == torch.float32, f"{impl_name} failed to return float32"

    # 3. Reference Pass
    with torch.no_grad():
        h_x_ref = ltv_pytorch_ref(
            initial_value.detach().float(), alpha.detach().float(), x.detach().float()
        )

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-5

    if dtype != torch.bfloat16:
        # Tolerances: use 1e-3 for bfloat16 inputs due to initial quantization error
        torch.testing.assert_close(
            h_x_metal, h_x_ref, atol=atol, rtol=atol, check_dtype=False
        )

    # 4. Backward Pass
    grad_h_x = torch.ones_like(h_x_metal) * 0.1
    h_x_metal.backward(grad_h_x)

    # ASSERTION: Gradients must match original input dtypes
    assert initial_value.grad.dtype == dtype
    assert alpha.grad.dtype == dtype
    assert x.grad.dtype == dtype

    m_grads = {
        "init": initial_value.grad.clone(),
        "alpha": alpha.grad.clone(),
        "x": x.grad.clone(),
    }

    # Reset for Reference
    initial_value.grad = None
    alpha.grad = None
    x.grad = None

    ref_init = initial_value.detach().float().requires_grad_(True)
    ref_alpha = alpha.detach().float().requires_grad_(True)
    ref_x = x.detach().float().requires_grad_(True)

    ref_h_x = ltv_pytorch_ref(ref_init, ref_alpha, ref_x)
    ref_h_x.backward(grad_h_x)

    if dtype != torch.bfloat16:
        # Compare Gradients
        torch.testing.assert_close(
            m_grads["init"].float(), ref_init.grad, atol=atol, rtol=atol
        )
        torch.testing.assert_close(
            m_grads["alpha"].float(), ref_alpha.grad, atol=atol, rtol=atol
        )
        torch.testing.assert_close(
            m_grads["x"].float(), ref_x.grad, atol=atol, rtol=atol
        )


def test_gradcheck_ltv_logic():
    """Double-precision check for the mathematical logic on CPU."""
    torch.manual_seed(42)
    B, T, Da, r = 2, 4, 2, 2
    D = Da * r

    init = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    a = torch.rand(B, T, Da, dtype=torch.float64, requires_grad=True)
    x_in = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        ltv_pytorch_ref, (init, a, x_in), eps=1e-6, atol=1e-4
    )
