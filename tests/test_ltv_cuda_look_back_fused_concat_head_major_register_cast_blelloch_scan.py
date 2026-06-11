# tests/test_ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan.py
import torch
import pytest
import time
import logging

from nanochat.ops.ltv_reference_ops import (
    ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_minimal,
)
from tests.ltv_scan_test_utils import (
    Settings,
    settings,
    get_standard_stride_layout,
    stride_layouts,
    make_combined_view,
    make_inits,
    make_logit_bias,
    assert_equal,
    add_common_parametrization,
)

logger = logging.getLogger(__name__)


def pytest_generate_tests(metafunc):
    add_common_parametrization(metafunc, include_scan_threads=True)


def reference_forward(
    settings, combined, inits, logit_bias, seq_len, da, r, P, Q, K, use_sigmoid
):
    """
    combined: (B, T, F) with arbitrary strides – the original tensor from make_combined_view.
    The Blelloch reference expects shape (B, F, T). We just transpose (view).
    """
    combined_for_ref = combined.transpose(1, 2)  # (B, T, F) -> (B, F, T) view
    return ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_minimal(
        combined_for_ref,
        inits,
        logit_bias,
        settings.nh,
        settings.nkvh,
        da,
        r,
        P,
        Q,
        K,
        use_sigmoid,
    )


def test_forward_exact(
    settings,
    pqk_config,
    seq_len,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    P, Q, K, use_sigmoid = pqk_config
    device = torch.device("cuda")

    for r in settings.r_values:
        da = settings.total_d // r
        total_h = settings.total_h
        total_features = total_h * (da * r + da)
        shape = (settings.batch, seq_len, total_features)

        for stride in (
            [get_standard_stride_layout(shape)]
            if settings.benchmark
            else stride_layouts(shape)
        ):
            combined_base_cpu, combined_cpu = make_combined_view(
                settings, seq_len, da, r, stride
            )
            inits_cpu = make_inits(settings, da, r)
            logit_bias_cpu = make_logit_bias(settings, da, r)

            expected_q = expected_k = expected_v = None
            if settings.check_against_expected:
                expected_q, expected_k, expected_v = reference_forward(
                    settings,
                    combined_cpu,
                    inits_cpu,
                    logit_bias_cpu,
                    seq_len,
                    da,
                    r,
                    P,
                    Q,
                    K,
                    use_sigmoid,
                )

            combined_cuda = torch.as_strided(
                combined_base_cpu.to(device), size=shape, stride=stride
            )
            inits_cuda = inits_cpu.to(device)
            logit_bias_cuda = logit_bias_cpu.to(device)

            repeats = 2 if settings.benchmark else 1
            for _ in range(repeats):
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()

                    actual_q, actual_k, actual_v = (
                        torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
                            combined_cuda,
                            inits_cuda,
                            logit_bias_cuda,
                            settings.nh,
                            settings.nkvh,
                            da,
                            r,
                            P,
                            Q,
                            K,
                            scan_threads_forward,
                            r_threads_forward,
                            r_items_forward,
                            0,
                            0,
                            0,
                            use_sigmoid,
                        )
                    )

                    torch.cuda.synchronize()
                    end = time.perf_counter_ns()

                M_dyn = (seq_len > P) and ((seq_len - P + Q - 1) // Q) or 0
                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"P={P}, M={M_dyn}, Q={Q}, K={K}, "
                    f"sf={scan_threads_forward}, rf={r_threads_forward}, rif={r_items_forward}"
                )
                if settings.check_against_expected:
                    assert_equal(actual_q, expected_q, "q", case, use_sigmoid)
                    assert_equal(actual_k, expected_k, "k", case, use_sigmoid)
                    assert_equal(actual_v, expected_v, "v", case, use_sigmoid)

            if settings.benchmark:
                logger.info(f"Case Forward: {case} time_ns: {end - start:_}")


def test_backward_exact(
    settings,
    pqk_config,
    seq_len,
    scan_threads_forward,
    r_threads_forward,
    r_items_forward,
    scan_threads_backward,
    r_threads_backward,
    r_items_backward,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    P, Q, K, use_sigmoid = pqk_config
    device = torch.device("cuda")

    for r in settings.r_values:
        da = settings.total_d // r
        total_h = settings.total_h
        total_features = total_h * (da * r + da)
        shape = (settings.batch, seq_len, total_features)

        for stride in (
            [get_standard_stride_layout(shape)]
            if settings.benchmark
            else stride_layouts(shape)
        ):
            combined_base_cpu, combined_cpu = make_combined_view(
                settings, seq_len, da, r, stride
            )
            inits_cpu = make_inits(settings, da, r)
            logit_bias_cpu = make_logit_bias(settings, da, r)

            # Prepare reference tensors with correct shape for gradient tracking
            if settings.check_against_expected:
                # Create a copy of the transposed tensor that requires grad
                combined_for_ref = (
                    combined_cpu.transpose(1, 2).clone().requires_grad_(True)
                )
                inits_ref = inits_cpu.clone().requires_grad_(True)
                logit_bias_ref = logit_bias_cpu.clone().requires_grad_(True)

                expected_q, expected_k, expected_v = (
                    ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_minimal(
                        combined_for_ref,
                        inits_ref,
                        logit_bias_ref,
                        settings.nh,
                        settings.nkvh,
                        da,
                        r,
                        P,
                        Q,
                        K,
                        use_sigmoid,
                    )
                )
                grad_q = torch.ones_like(expected_q)
                grad_k = torch.ones_like(expected_k)
                grad_v = torch.ones_like(expected_v)
                torch.autograd.backward(
                    [expected_q, expected_k, expected_v],
                    [grad_q, grad_k, grad_v],
                )
                # Gradients of the reference tensors
                # combined_for_ref.grad is shape (B, F, T). Transpose back to (B, T, F) for comparison.
                expected_grad_combined = combined_for_ref.grad.transpose(1, 2)
                expected_grad_inits = inits_ref.grad
                expected_grad_logit_bias = logit_bias_ref.grad
            else:
                D = da * r
                grad_q = torch.ones((settings.batch, settings.nh, seq_len, D))
                grad_k = torch.ones((settings.batch, settings.nkvh, seq_len, D))
                grad_v = torch.ones((settings.batch, settings.nkvh, seq_len, D))

            combined_cuda = torch.as_strided(
                combined_base_cpu.to(device), size=shape, stride=stride
            ).requires_grad_(True)
            inits_cuda = inits_cpu.detach().to(device).requires_grad_(True)
            logit_bias_cuda = logit_bias_cpu.detach().to(device).requires_grad_(True)

            repeats = 2 if settings.benchmark else 1
            for _ in range(repeats):
                if combined_cuda.grad is not None:
                    combined_cuda.grad.zero_()
                if inits_cuda.grad is not None:
                    inits_cuda.grad.zero_()
                if logit_bias_cuda.grad is not None:
                    logit_bias_cuda.grad.zero_()

                torch.cuda.synchronize()
                start = time.perf_counter_ns()

                q_act, k_act, v_act = (
                    torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
                        combined_cuda,
                        inits_cuda,
                        logit_bias_cuda,
                        settings.nh,
                        settings.nkvh,
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
                    )
                )
                torch.autograd.backward(
                    [q_act, k_act, v_act],
                    [grad_q.to(device), grad_k.to(device), grad_v.to(device)],
                )

                torch.cuda.synchronize()
                end = time.perf_counter_ns()

                M_dyn = (seq_len > P) and ((seq_len - P + Q - 1) // Q) or 0
                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"P={P}, M={M_dyn}, Q={Q}, K={K}, "
                    f"sf={scan_threads_forward}, rf={r_threads_forward}, rif={r_items_forward}, "
                    f"sb={scan_threads_backward}, rb={r_threads_backward}, rib={r_items_backward}"
                )

                if settings.check_against_expected:
                    assert_equal(
                        combined_cuda.grad,
                        expected_grad_combined,
                        "grad_combined",
                        case,
                        use_sigmoid,
                    )
                    assert_equal(
                        inits_cuda.grad,
                        expected_grad_inits,
                        "grad_inits",
                        case,
                        use_sigmoid,
                    )
                    assert_equal(
                        logit_bias_cuda.grad,
                        expected_grad_logit_bias,
                        "grad_logit_bias",
                        case,
                        use_sigmoid,
                    )

            if settings.benchmark:
                logger.info(f"Case Backward: {case} time_ns: {end - start:_}")
