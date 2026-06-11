import pytest
import torch
import numpy as np

from nanochat.ltv_fused_concat_register_cast_scan_util import CUDA_SUPPORTS_NATIVE_BF16

def assert_equal(actual, expected, label, case, atol=1e-4, rtol=1e-4):
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

def reference_forward(combined, NH, NKVH, da, r):
    B, T, _ = combined.shape
    D = da * r
    features_per_head = D + da
    total_h = NH + 2 * NKVH

    # Reshape to (B, T, total_h, features_per_head)
    reshaped = combined.view(B, T, total_h, features_per_head)

    # Extract only the first D features (values), not the last da (logits)
    values = reshaped[..., :D]  # (B, T, total_h, D)

    # Split into Q, K, V heads
    q_values = values[:, :, :NH, :]  # (B, T, NH, D)
    k_values = values[:, :, NH:NH + NKVH, :]  # (B, T, NKVH, D)
    v_values = values[:, :, NH + NKVH:, :]  # (B, T, NKVH, D)

    # Transpose to (NH, B, T, D) then contiguous and reshape to (NH, B*T, D)
    # Actually we want (NH, T, D) per batch, so (B, NH, T, D) -> (NH, B, T, D)
    out_q = q_values.transpose(1, 2).contiguous()  # (B, NH, T, D)
    out_k = k_values.transpose(1, 2).contiguous()  # (B, NKVH, T, D)
    out_v = v_values.transpose(1, 2).contiguous()  # (B, NKVH, T, D)

    return out_q, out_k, out_v

@pytest.mark.parametrize("dtype", [
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not CUDA_SUPPORTS_NATIVE_BF16, reason="No BF16 support"))
])
@pytest.mark.parametrize("B, T, NH, NKVH, da, r", [
    (1, 16, 2, 2, 4, 8),     # Small case
    (4, 128, 8, 8, 4, 32),   # Medium case
    (2, 31, 3, 2, 3, 7),     # Odd dimensions
])
def test_forward_exact(B, T, NH, NKVH, da, r, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    D = da * r
    total_h = NH + 2 * NKVH
    total_features = total_h * (D + da)

    combined = torch.randn(B, T, total_features, dtype=dtype, device=device)

    # 1. Custom Operator Forward
    actual_q, actual_k, actual_v = torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_op(
        combined, NH, NKVH, da, r,
        4, 16, 2,  # scan_threads_forward, r_threads_forward, r_items_forward
        32, 2, 4,  # scan_threads_backward, r_threads_backward, r_items_backward
    )

    # 2. Reference Forward
    expected_q, expected_k, expected_v = reference_forward(combined, NH, NKVH, da, r)

    case_desc = f"B={B}, T={T}, NH={NH}, NKVH={NKVH}, da={da}, r={r}, dtype={dtype}"

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    assert_equal(actual_q, expected_q, "out_q", case_desc, atol=atol, rtol=rtol)
    assert_equal(actual_k, expected_k, "out_k", case_desc, atol=atol, rtol=rtol)
    assert_equal(actual_v, expected_v, "out_v", case_desc, atol=atol, rtol=rtol)

@pytest.mark.parametrize("dtype", [
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not CUDA_SUPPORTS_NATIVE_BF16, reason="No BF16 support"))
])
@pytest.mark.parametrize("B, T, NH, NKVH, da, r", [
    (1, 16, 2, 2, 4, 8),     # Small case
    (4, 128, 8, 8, 4, 32),   # Medium case
    (2, 31, 3, 2, 3, 7),     # Odd dimensions
])
def test_backward_exact(B, T, NH, NKVH, da, r, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    D = da * r
    total_h = NH + 2 * NKVH
    total_features = total_h * (D + da)

    combined_ref = torch.randn(B, T, total_features, dtype=dtype, device=device, requires_grad=True)
    combined_act = combined_ref.clone().detach().requires_grad_(True)

    # Reference pass
    expected_q, expected_k, expected_v = reference_forward(combined_ref, NH, NKVH, da, r)
    grad_q = torch.randn_like(expected_q)
    grad_k = torch.randn_like(expected_k)
    grad_v = torch.randn_like(expected_v)

    torch.autograd.backward([expected_q, expected_k, expected_v], [grad_q, grad_k, grad_v])
    expected_grad_combined = combined_ref.grad.clone()

    # Actual pass
    actual_q, actual_k, actual_v = torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_op(
        combined_act, NH, NKVH, da, r,
        4, 16, 2,  # scan_threads_forward, r_threads_forward, r_items_forward
        32, 2, 4,  # scan_threads_backward, r_threads_backward, r_items_backward
    )
    torch.autograd.backward([actual_q, actual_k, actual_v], [grad_q, grad_k, grad_v])
    actual_grad_combined = combined_act.grad.clone()

    case_desc = f"B={B}, T={T}, NH={NH}, NKVH={NKVH}, da={da}, r={r}, dtype={dtype}"

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    assert_equal(actual_grad_combined, expected_grad_combined, "grad_combined", case_desc, atol=atol, rtol=rtol)

def test_gradcheck():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    # Keep sizes very small for gradcheck numerical stability
    B, T, NH, NKVH, da, r = 1, 4, 1, 1, 2, 2
    D = da * r
    total_h = NH + 2 * NKVH
    total_features = total_h * (D + da)

    combined = torch.randn(B, T, total_features, dtype=torch.float64, device=device, requires_grad=True)

    def op_wrapper(x):
        return torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_op(
            x, NH, NKVH, da, r,
            4, 16, 2,  # scan_threads_forward, r_threads_forward, r_items_forward
            32, 2, 4,  # scan_threads_backward, r_threads_backward, r_items_backward
        )

    # Run gradcheck on double precision
    assert torch.autograd.gradcheck(op_wrapper, (combined,), eps=1e-6, atol=1e-4)
