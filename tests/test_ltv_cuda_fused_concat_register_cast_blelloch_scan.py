import itertools
import torch
import pytest
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List

import nanochat.ops  # noqa: F401

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
        self.total_d = 128 if self.benchmark else 1
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
            seq_lengths = [1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129]
        metafunc.parametrize("seq_len", seq_lengths)

    # Parameterize the CUB-specific hardware tuning values
    if "block_threads" in metafunc.fixturenames:
        metafunc.parametrize("block_threads", [128, 256])

    if "items_per_thread" in metafunc.fixturenames:
        metafunc.parametrize("items_per_thread", [1, 2, 4, 8, 16])


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
    total_h = settings.total_h
    D = da * r
    features = total_h * D + total_h * da
    shape = (settings.batch, seq_len, features)

    base = torch.empty(storage_size(shape, stride), dtype=torch.bfloat16)
    view = torch.as_strided(base, size=shape, stride=stride)

    offset_l = total_h * D

    for b in range(settings.batch):
        for t in range(seq_len):
            for h_glob in range(total_h):
                if h_glob < settings.nh:
                    h_loc = h_glob
                    offset_in_row = 0
                elif h_glob < settings.nh + settings.nkvh:
                    h_loc = h_glob - settings.nh
                    offset_in_row = settings.nh * D
                else:
                    h_loc = h_glob - (settings.nh + settings.nkvh)
                    offset_in_row = (settings.nh + settings.nkvh) * D

                value_base = offset_in_row + h_loc * D
                for d in range(D):
                    view[b, t, value_base + d] = (
                        (11 * b + 7 * t + 5 * h_glob + d) % 7
                    ) - 3

                logit_base = offset_l + h_glob * da
                for da_idx in range(da):
                    view[b, t, logit_base + da_idx] = (b + t + h_glob + da_idx) % 2

    return base, view


def make_inits(settings, da, r):
    total_h = settings.total_h
    D = da * r
    inits = torch.empty((settings.batch, total_h, D), dtype=torch.bfloat16)
    for b in range(settings.batch):
        for h_glob in range(total_h):
            for d in range(D):
                inits[b, h_glob, d] = ((13 * b + 3 * h_glob + d) % 7) - 3
    return inits


def reference_forward(settings, combined, inits, seq_len, da, r):
    total_h = settings.total_h
    D = da * r
    offset_l = total_h * D

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
                h_loc = h_glob
                offset_in_row = 0
            elif h_glob < settings.nh + settings.nkvh:
                heads = k_heads
                h_loc = h_glob - settings.nh
                offset_in_row = settings.nh * D
            else:
                heads = v_heads
                h_loc = h_glob - (settings.nh + settings.nkvh)
                offset_in_row = (settings.nh + settings.nkvh) * D

            state = inits[b, h_glob].clone()

            time_steps = []
            for t in range(seq_len):
                x_val = combined[
                    b, t, offset_in_row + h_loc * D : offset_in_row + (h_loc + 1) * D
                ]
                alpha = combined[
                    b,
                    t,
                    offset_l + h_glob * da : offset_l + (h_glob + 1) * da,
                ]
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

    return out_q, out_k, out_v


def assert_equal(settings, actual, expected, label, case, r=1):
    atol = 0.01 if settings.sigmoid else 0
    rtol = 0.01 if settings.sigmoid else 0

    np.testing.assert_allclose(
        actual.cpu().contiguous().to(dtype=torch.float32),
        expected.contiguous().to(dtype=torch.float32),
        rtol=rtol,
        atol=atol,
    )


def test_forward_exact(settings, seq_len, block_threads, items_per_thread):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    for r in settings.r_values:
        da = settings.total_d // r
        total_h = settings.total_h
        features = total_h * settings.total_d + total_h * da
        shape = (settings.batch, seq_len, features)

        for stride in (
            [get_standard_stride_layout(shape)]
            if settings.benchmark
            else stride_layouts(shape)
        ):
            combined_base_cpu, combined_cpu = make_combined_view(
                settings, seq_len, da, r, stride
            )
            inits_cpu = make_inits(settings, da, r)

            expected_q, expected_k, expected_v = None, None, None
            if settings.check_against_expected:
                expected_q, expected_k, expected_v = reference_forward(
                    settings, combined_cpu, inits_cpu, seq_len, da, r
                )

            combined_cuda = torch.as_strided(
                combined_base_cpu.to(device),
                size=shape,
                stride=stride,
            )
            inits_cuda = inits_cpu.to(device)

            repeats = 2 if settings.benchmark else 1

            for _ in range(repeats):
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time_ns = time.perf_counter_ns()

                    actual_q, actual_k, actual_v = (
                        torch.ops.nanochat_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_op(
                            combined_cuda,
                            inits_cuda,
                            settings.nh,
                            settings.nkvh,
                            da,
                            r,
                            block_threads,
                            items_per_thread,
                            settings.sigmoid,
                        )
                    )

                    torch.cuda.synchronize()
                    end_time_ns = time.perf_counter_ns()

                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    f"bt={block_threads}, ipt={items_per_thread}"
                )
                if settings.check_against_expected:
                    assert_equal(settings, actual_q, expected_q, "q", case)
                    assert_equal(settings, actual_k, expected_k, "k", case)
                    assert_equal(settings, actual_v, expected_v, "v", case)

            if settings.benchmark:
                logger.info(
                    f"Case Forward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )


def test_backward_exact(settings, seq_len, block_threads, items_per_thread):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    for r in settings.r_values:
        da = settings.total_d // r
        total_h = settings.total_h
        features = total_h * settings.total_d + total_h * da
        shape = (settings.batch, seq_len, features)

        for stride in (
            [get_standard_stride_layout(shape)]
            if settings.benchmark
            else stride_layouts(shape)
        ):
            combined_base_cpu, combined_cpu = make_combined_view(
                settings, seq_len, da, r, stride
            )
            inits_cpu = make_inits(settings, da, r)

            combined_cpu.requires_grad_(True)
            inits_cpu.requires_grad_(True)

            expected_grad_combined, expected_grad_inits = None, None
            if settings.check_against_expected:
                expected_q, expected_k, expected_v = reference_forward(
                    settings, combined_cpu, inits_cpu, seq_len, da, r
                )

                grad_q = torch.ones_like(expected_q)
                grad_k = torch.ones_like(expected_k)
                grad_v = torch.ones_like(expected_v)

                torch.autograd.backward(
                    [expected_q, expected_k, expected_v], [grad_q, grad_k, grad_v]
                )
                expected_grad_combined = combined_cpu.grad
                expected_grad_inits = inits_cpu.grad
            else:
                total_h = settings.total_h
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

            combined_cuda.requires_grad_(True)
            inits_cuda.requires_grad_(True)

            repeats = 2 if settings.benchmark else 1
            for _ in range(repeats):
                if combined_cuda.grad is not None:
                    combined_cuda.grad.zero_()
                if inits_cuda.grad is not None:
                    inits_cuda.grad.zero_()

                torch.cuda.synchronize()
                start_time_ns = time.perf_counter_ns()

                q_act, k_act, v_act = (
                    torch.ops.nanochat_cuda.ltv_fused_concat_register_cast_blelloch_scan_cuda_op(
                        combined_cuda,
                        inits_cuda,
                        settings.nh,
                        settings.nkvh,
                        da,
                        r,
                        block_threads,
                        items_per_thread,
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
                    f"bt={block_threads}, ipt={items_per_thread}"
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

            if settings.benchmark:
                logger.info(
                    f"Case Backward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )
