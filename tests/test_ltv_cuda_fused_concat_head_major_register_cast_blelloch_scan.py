import itertools
import torch
import pytest
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List

import nanochat.ops  # noqa: F401
from nanochat.ltv_fused_concat_register_cast_scan_util import CUDA_SUPPORTS_NATIVE_BF16

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    benchmark: bool
    sigmoid: bool
    check_against_expected: bool
    batch: int = 4
    nh: int = 8
    nkvh: int = 8
    total_d: int = 128
    r_values: List[int] = None

    def __post_init__(self):
        self.total_d = 128 if self.benchmark else 2
        if self.r_values is None:
            self.r_values = (
                [self.total_d]
                if self.benchmark
                else [r for r in range(1, self.total_d + 1) if self.total_d % r == 0]
            )
        self.total_h = self.nh + 2 * self.nkvh


@pytest.fixture(scope="session")
def settings(pytestconfig):
    benchmark = pytestconfig.getoption("--benchmark").lower() == "true"
    sigmoid = pytestconfig.getoption("--sigmoid").lower() == "true"
    check_against_expected = (
        pytestconfig.getoption("--check_against_expected").lower() == "true"
    )
    return Settings(
        benchmark=benchmark,
        sigmoid=sigmoid,
        check_against_expected=check_against_expected,
    )


def pytest_generate_tests(metafunc):
    # Dynamic parametrization based on CLI flags
    benchmark = metafunc.config.getoption("--benchmark").lower() == "true"

    if "seq_len" in metafunc.fixturenames:
        if benchmark:
            seq_lengths = [1024]
        else:
            seq_lengths = [
                1,
                2,
                3,
                7,
                8,
                7,
                9,
                15,
                16,
                17,
                31,
                32,
                33,
                127,
                128,
                129,
            ]
            # seq_lengths = list(range(1, 32))
        metafunc.parametrize("seq_len", seq_lengths)

    # Parameterize the tuning values
    if "scan_threads_forward" in metafunc.fixturenames:
        metafunc.parametrize("scan_threads_forward", [1, 8])
        # metafunc.parametrize("scan_threads_forward", [1])

    if "r_threads_forward" in metafunc.fixturenames:
        metafunc.parametrize("r_threads_forward", [2, 32])
        # metafunc.parametrize("r_threads_forward", [1])

    if "r_items_forward" in metafunc.fixturenames:
        metafunc.parametrize("r_items_forward", [1, 4])
        # metafunc.parametrize("r_items_forward", [1])

    # Parameterize the tuning values
    if "scan_threads_backward" in metafunc.fixturenames:
        metafunc.parametrize("scan_threads_backward", [1, 8])
        # metafunc.parametrize("scan_threads_backward", [1])

    if "r_threads_backward" in metafunc.fixturenames:
        metafunc.parametrize("r_threads_backward", [2, 32])
        # metafunc.parametrize("r_threads_backward", [1])

    if "r_items_backward" in metafunc.fixturenames:
        metafunc.parametrize("r_items_backward", [1, 4])
        # metafunc.parametrize("r_items_backward", [1])


def get_standard_stride_layout(shape):
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def stride_layouts(shape):
    layouts = set()
    ndim = len(shape)
    for order in itertools.permutations(range(ndim)):
        for gap_factors in itertools.product((1, 2), repeat=ndim):
            stride = [0] * ndim
            running = 1
            for axis, gap in zip(order, gap_factors):
                stride[axis] = running * gap
                running = stride[axis] * shape[axis]
            layouts.add(tuple(stride))
    return sorted(layouts)


def storage_size(shape, stride):
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def make_combined_view(settings, seq_len, da, r, stride):
    """
    Head-major layout: [B, T, total_h, (D + da)]
    Flattened: [B, T, total_h * (D + da)]
    """
    total_h = settings.total_h
    D = da * r
    features_per_head = D + da
    total_features = total_h * features_per_head
    shape = (settings.batch, seq_len, total_features)

    dtype = torch.bfloat16 if CUDA_SUPPORTS_NATIVE_BF16 else torch.float32
    base = torch.empty(storage_size(shape, stride), dtype=dtype)
    view = torch.as_strided(base, size=shape, stride=stride)

    for b in range(settings.batch):
        for t in range(seq_len):
            for h_glob in range(total_h):
                head_offset = h_glob * features_per_head

                # Values first (D elements)
                for d in range(D):
                    view[b, t, head_offset + d] = (
                        (9 * b + 7 * t + 3 * h_glob + d) % 5
                    ) - 2

                # Alphas second (da elements)
                for da_idx in range(da):
                    view[b, t, head_offset + D + da_idx] = (b + t + h_glob + da_idx) % 2

    return base, view


def make_inits(settings, da, r):
    total_h = settings.total_h
    D = da * r
    dtype = torch.bfloat16 if CUDA_SUPPORTS_NATIVE_BF16 else torch.float32
    inits = torch.empty((settings.batch, total_h, D), dtype=dtype)
    for b in range(settings.batch):
        for h_glob in range(total_h):
            for d in range(D):
                inits[b, h_glob, d] = ((13 * b + 3 * h_glob + d) % 7) - 3
    return inits


def make_logit_bias(settings, da, r):
    total_h = settings.total_h
    # Logit bias is always float32.
    logit_bias = torch.empty((total_h, da), dtype=torch.float32)
    for h_glob in range(total_h):
        for da_idx in range(da):
            logit_bias[h_glob, da_idx] = ((h_glob + 2 * da_idx) % 3) - 1
    return logit_bias


def reference_forward(settings, combined, inits, logit_bias, seq_len, da, r):
    total_h = settings.total_h
    D = da * r
    features_per_head = D + da

    q_batches = []
    k_batches = []
    v_batches = []

    for b in range(settings.batch):
        q_heads = []
        k_heads = []
        v_heads = []

        for h_glob in range(total_h):
            if h_glob < settings.nh:
                heads = q_heads
            elif h_glob < settings.nh + settings.nkvh:
                heads = k_heads
            else:
                heads = v_heads

            state = inits[b, h_glob].clone()
            head_offset = h_glob * features_per_head

            time_steps = []
            for t in range(seq_len):
                # In head-major, values and alphas for a head are contiguous
                x_val = combined[b, t, head_offset : head_offset + D]
                alpha = combined[b, t, head_offset + D : head_offset + D + da]
                alpha = alpha + logit_bias[h_glob, :]

                if settings.sigmoid:
                    alpha = torch.sigmoid(alpha)
                alpha = alpha.repeat_interleave(r)
                state = (1.0 - alpha) * state + alpha * x_val
                time_steps.append(state)

            heads.append(torch.stack(time_steps, dim=0))

        if q_heads:
            q_batches.append(torch.stack(q_heads, dim=0))
        if k_heads:
            k_batches.append(torch.stack(k_heads, dim=0))
        if v_heads:
            v_batches.append(torch.stack(v_heads, dim=0))

    out_q = (
        torch.stack(q_batches, dim=0)
        if q_batches
        else torch.empty(settings.batch, settings.nh, seq_len, D)
    )
    out_k = (
        torch.stack(k_batches, dim=0)
        if k_batches
        else torch.empty(settings.batch, settings.nkvh, seq_len, D)
    )
    out_v = (
        torch.stack(v_batches, dim=0)
        if v_batches
        else torch.empty(settings.batch, settings.nkvh, seq_len, D)
    )

    combined_dtype = combined.dtype
    return (
        out_q.to(dtype=combined_dtype),
        out_k.to(dtype=combined_dtype),
        out_v.to(dtype=combined_dtype),
    )


def assert_equal(settings, actual, expected, label, case, r=1):
    atol = 0.01 if settings.sigmoid else 0
    rtol = 0.01 if settings.sigmoid else 0

    actual_np = actual.cpu().contiguous().to(dtype=torch.float32).numpy()
    expected_np = expected.contiguous().to(dtype=torch.float32).numpy()

    try:
        np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol)
    except AssertionError as e:
        diff = np.abs(actual_np - expected_np)
        close = np.isclose(actual_np, expected_np, rtol=rtol, atol=atol)
        mismatches = np.where(~close)
        num_mismatch = len(mismatches[0])

        print(f"\nAssertionError in {label} for {case}")
        print(
            f"Number of mismatched elements: {num_mismatch} out of {expected_np.size}"
        )
        if num_mismatch > 0:
            max_diff_idx = np.argmax(diff)
            max_diff_val = diff.flat[max_diff_idx]
            idx_tuple = np.unravel_index(max_diff_idx, diff.shape)
            print(f"Max absolute difference: {max_diff_val} at index {idx_tuple}")
            print(
                f"  actual: {actual_np[idx_tuple]}, expected: {expected_np[idx_tuple]}"
            )
            print("First 10 mismatches (index, actual, expected):")
            for i in range(min(num_mismatch, 10)):
                idx = tuple(m[i] for m in mismatches)
                print(f"  {idx}: actual={actual_np[idx]}, expected={expected_np[idx]}")
        raise e


def test_forward_exact(
    settings, seq_len, scan_threads_forward, r_threads_forward, r_items_forward
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

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

            expected_q, expected_k, expected_v = None, None, None
            if settings.check_against_expected:
                expected_q, expected_k, expected_v = reference_forward(
                    settings, combined_cpu, inits_cpu, logit_bias_cpu, seq_len, da, r
                )

            combined_cuda = torch.as_strided(
                combined_base_cpu.to(device),
                size=shape,
                stride=stride,
            )
            inits_cuda = inits_cpu.to(device)
            logit_bias_cuda = logit_bias_cpu.to(device)

            repeats = 2 if settings.benchmark else 1

            for _ in range(repeats):
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time_ns = time.perf_counter_ns()

                    actual_q, actual_k, actual_v = (
                        torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
                            combined_cuda,
                            inits_cuda,
                            logit_bias_cuda,
                            settings.nh,
                            settings.nkvh,
                            da,
                            r,
                            scan_threads_forward,
                            r_threads_forward,
                            r_items_forward,
                            0,
                            0,
                            0,
                            settings.sigmoid,
                        )
                    )

                    torch.cuda.synchronize()
                    end_time_ns = time.perf_counter_ns()

                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"btf={scan_threads_forward}, rbf={r_threads_forward}, rif={r_items_forward}"
                )
                if settings.check_against_expected:
                    assert_equal(settings, actual_q, expected_q, "q", case)
                    assert_equal(settings, actual_k, expected_k, "k", case)
                    assert_equal(settings, actual_v, expected_v, "v", case)

            if settings.benchmark:
                logger.info(
                    f"Case Forward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )


def test_backward_exact(
    settings,
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
            combined_cpu.requires_grad_(True)
            inits_cpu.requires_grad_(True)
            logit_bias_cpu.requires_grad_(True)

            expected_grad_combined, expected_grad_inits, expected_grad_logit_bias = (
                None,
                None,
                None,
            )
            if settings.check_against_expected:
                expected_q, expected_k, expected_v = reference_forward(
                    settings, combined_cpu, inits_cpu, logit_bias_cpu, seq_len, da, r
                )

                grad_q = torch.ones_like(expected_q)
                grad_k = torch.ones_like(expected_k)
                grad_v = torch.ones_like(expected_v)

                torch.autograd.backward(
                    [expected_q, expected_k, expected_v], [grad_q, grad_k, grad_v]
                )
                expected_grad_combined = combined_cpu.grad
                expected_grad_inits = inits_cpu.grad
                expected_grad_logit_bias = logit_bias_cpu.grad
            else:
                D = da * r
                grad_q = torch.ones((settings.batch, settings.nh, seq_len, D))
                grad_k = torch.ones((settings.batch, settings.nkvh, seq_len, D))
                grad_v = torch.ones((settings.batch, settings.nkvh, seq_len, D))

            combined_cuda = torch.as_strided(
                combined_base_cpu.to(device),
                size=shape,
                stride=stride,
            )
            inits_cuda = inits_cpu.detach().to(device)
            logit_bias_cuda = logit_bias_cpu.detach().to(device)

            combined_cuda.requires_grad_(True)
            inits_cuda.requires_grad_(True)
            logit_bias_cuda.requires_grad_(True)

            repeats = 2 if settings.benchmark else 1
            for _ in range(repeats):
                if combined_cuda.grad is not None:
                    combined_cuda.grad.zero_()
                if inits_cuda.grad is not None:
                    inits_cuda.grad.zero_()
                if logit_bias_cuda.grad is not None:
                    logit_bias_cuda.grad.zero_()

                torch.cuda.synchronize()
                start_time_ns = time.perf_counter_ns()

                q_act, k_act, v_act = (
                    torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
                        combined_cuda,
                        inits_cuda,
                        logit_bias_cuda,
                        settings.nh,
                        settings.nkvh,
                        da,
                        r,
                        scan_threads_forward,
                        r_threads_forward,
                        r_items_forward,
                        scan_threads_backward,
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
                end_time_ns = time.perf_counter_ns()

                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"btf={scan_threads_forward}, rbf={r_threads_forward}, rif={r_items_forward}, "
                    f"btb={scan_threads_backward}, rbb={r_threads_backward}, rib={r_items_backward}"
                )

                if settings.check_against_expected:
                    assert_equal(
                        settings,
                        combined_cuda.grad,
                        expected_grad_combined,
                        "grad_combined",
                        case,
                        r=r,
                    )
                    assert_equal(
                        settings,
                        inits_cuda.grad,
                        expected_grad_inits,
                        "grad_inits",
                        case,
                        r=r,
                    )
                    assert_equal(
                        settings,
                        logit_bias_cuda.grad,
                        expected_grad_logit_bias,
                        "grad_logit_bias",
                        case,
                        r=r,
                    )

            if settings.benchmark:
                logger.info(
                    f"Case Backward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )
