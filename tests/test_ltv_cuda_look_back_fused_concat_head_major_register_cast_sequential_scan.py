import torch
import pytest
import time
import logging

from nanochat.ops.ltv_reference_ops import (
    ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal,
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
    add_common_parametrization(metafunc, include_scan_threads=False)


def reference_forward(settings, combined, inits, logit_bias, seq_len, da, r, P, Q, K):
    """
    combined: (B, T, F) - the native layout for the sequential scan.
    The reference expects exactly this shape, no transposition needed.
    """
    return ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal(
        combined,
        inits,
        logit_bias,
        settings.nh,
        settings.nkvh,
        da,
        r,
        P,
        Q,
        K,
        settings.sigmoid,
    )


def test_forward_exact(
    settings,
    pqk_config,
    seq_len,
    r_threads_forward,
    r_items_forward,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    P, Q, K = pqk_config
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
                        torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_op(
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
                            r_threads_forward,
                            r_items_forward,
                            0,
                            0,
                            settings.sigmoid,
                        )
                    )

                    torch.cuda.synchronize()
                    end = time.perf_counter_ns()

                M_dyn = (seq_len > P) and ((seq_len - P + Q - 1) // Q) or 0
                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"P={P}, M={M_dyn}, Q={Q}, K={K}, "
                    f"rf={r_threads_forward}, rif={r_items_forward}"
                )
                if settings.check_against_expected:
                    assert_equal(actual_q, expected_q, "q", case, settings.sigmoid)
                    assert_equal(actual_k, expected_k, "k", case, settings.sigmoid)
                    assert_equal(actual_v, expected_v, "v", case, settings.sigmoid)

            if settings.benchmark:
                logger.info(f"Case Forward: {case} time_ns: {end - start:_}")


def test_backward_exact(
    settings,
    pqk_config,
    seq_len,
    r_threads_forward,
    r_items_forward,
    r_threads_backward,
    r_items_backward,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    P, Q, K = pqk_config
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

            # Prepare reference tensors for gradient tracking
            if settings.check_against_expected:
                combined_ref = combined_cpu.clone().requires_grad_(True)
                inits_ref = inits_cpu.clone().requires_grad_(True)
                logit_bias_ref = logit_bias_cpu.clone().requires_grad_(True)

                expected_q, expected_k, expected_v = (
                    ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal(
                        combined_ref,
                        inits_ref,
                        logit_bias_ref,
                        settings.nh,
                        settings.nkvh,
                        da,
                        r,
                        P,
                        Q,
                        K,
                        settings.sigmoid,
                    )
                )
                grad_q = torch.ones_like(expected_q)
                grad_k = torch.ones_like(expected_k)
                grad_v = torch.ones_like(expected_v)
                torch.autograd.backward(
                    [expected_q, expected_k, expected_v],
                    [grad_q, grad_k, grad_v],
                )
                expected_grad_combined = combined_ref.grad
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
                    torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_op(
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
                        r_threads_forward,
                        r_items_forward,
                        r_threads_backward,
                        r_items_backward,
                        settings.sigmoid,
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
                    f"rf={r_threads_forward}, rif={r_items_forward}, "
                    f"rb={r_threads_backward}, rib={r_items_backward}"
                )

                if settings.check_against_expected:
                    assert_equal(
                        combined_cuda.grad,
                        expected_grad_combined,
                        "grad_combined",
                        case,
                        settings.sigmoid,
                    )
                    assert_equal(
                        inits_cuda.grad,
                        expected_grad_inits,
                        "grad_inits",
                        case,
                        settings.sigmoid,
                    )
                    assert_equal(
                        logit_bias_cuda.grad,
                        expected_grad_logit_bias,
                        "grad_logit_bias",
                        case,
                        settings.sigmoid,
                    )

            if settings.benchmark:
                logger.info(f"Case Backward: {case} time_ns: {end - start:_}")
