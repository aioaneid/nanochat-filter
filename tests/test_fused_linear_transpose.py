import pytest
import torch
import itertools
import numpy as np

import nanochat.ops.fused_linear_transpose  # Ensure op is registered

from nanochat.ltv_fused_concat_register_cast_scan_util import CUDA_SUPPORTS_NATIVE_BF16

def assert_equal(actual, expected, label, case, atol=1e-3, rtol=1e-3):
    actual_np = actual.detach().cpu().contiguous().to(dtype=torch.float32).numpy()
    expected_np = expected.detach().cpu().contiguous().to(dtype=torch.float32).numpy()

    try:
        np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol)
    except AssertionError as e:
        diff = np.abs(actual_np - expected_np)
        close = np.isclose(actual_np, expected_np, rtol=rtol, atol=atol)
        mismatches = np.where(~close)
        num_mismatch = len(mismatches[0])

        print(f"\nAssertionError in {label} for {case}")
        print(f"Number of mismatched elements: {num_mismatch} out of {expected_np.size}")
        if num_mismatch > 0:
            max_diff_idx = np.argmax(diff)
            max_diff_val = diff.flat[max_diff_idx]
            idx_tuple = np.unravel_index(max_diff_idx, diff.shape)
            print(f"Max absolute difference: {max_diff_val} at index {idx_tuple}")
            print(f"  actual: {actual_np[idx_tuple]}, expected: {expected_np[idx_tuple]}")
        raise e

@pytest.mark.parametrize("dtype", [
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not CUDA_SUPPORTS_NATIVE_BF16, reason="No BF16 support"))
])
@pytest.mark.parametrize("B, T, K, F", [
    (1, 16, 32, 128),     # Small case
    (4, 128, 128, 256),   # Medium case
    (4, 1024, 1024, 3096),# Large case (resembling user bug)
    (2, 31, 63, 127),     # Odd dimensions
])
def test_fused_linear_transpose_exact(B, T, K, F, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Initialize random inputs
    x = torch.randn(B, T, K, dtype=dtype, device=device) / (K ** 0.5)
    weight = torch.randn(F, K, dtype=dtype, device=device) / (K ** 0.5)

    x.requires_grad_(True)
    weight.requires_grad_(True)

    # 1. Reference forward
    # out = weight @ x^T
    expected_out = torch.matmul(weight, x.transpose(-1, -2))
    
    # 2. Reference backward
    grad_out = torch.randn_like(expected_out)
    
    expected_out.backward(grad_out, retain_graph=True)
    expected_grad_x = x.grad.clone()
    expected_grad_weight = weight.grad.clone()

    x.grad.zero_()
    weight.grad.zero_()

    # 3. Custom operator forward
    actual_out = torch.ops.nanochat_cuda.fused_linear_transpose_cuda_op(x, weight)
    
    case_desc = f"B={B}, T={T}, K={K}, F={F}, dtype={dtype}"
    
    # Tolerances are higher for bfloat16
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-4

    assert_equal(actual_out, expected_out, "out", case_desc, atol=atol, rtol=rtol)

    # 4. Custom operator backward
    actual_out.backward(grad_out)
    actual_grad_x = x.grad.clone()
    actual_grad_weight = weight.grad.clone()

    assert_equal(actual_grad_x, expected_grad_x, "grad_x", case_desc, atol=atol, rtol=rtol)
    assert_equal(actual_grad_weight, expected_grad_weight, "grad_weight", case_desc, atol=atol, rtol=rtol)
