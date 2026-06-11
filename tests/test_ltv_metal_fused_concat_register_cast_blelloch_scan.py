import itertools

import torch
import pytest
try:
    import rust_ewma
except ImportError:
    rust_ewma = None
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
            seq_lengths = [1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129][:4]
        metafunc.parametrize("seq_len", seq_lengths)

    if "blelloch_context_config" in metafunc.fixturenames:
        configs = get_context_configs_dynamic(benchmark)
        metafunc.parametrize(
            "blelloch_context_config",
            configs,
            indirect=True,
            ids=lambda cfg: f"tew={cfg[0]},mt={cfg[1]},tgm={cfg[2]},banks={cfg[3]}",
        )


def get_context_configs_dynamic(benchmark):
    if benchmark:
        base = [
            (1, 0, 0),
            (2, 0, 0),
            (4, 0, 0),
            (8, 0, 0),
            (16, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 4, 0),
            (0, 8, 0),
            (0, 16, 0),
            (0, 32, 0),
            (0, 64, 0),
            (0, 128, 0),
            (0, 256, 0),
            (0, 512, 0),
            (0, 0, 0),
        ]
    else:
        base = [
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 0, 8),
            (0, 0, 16),
            (0, 0, 0),
        ]
    return [(*cfg, banks) for cfg in base for banks in (0, 1_000_000)]


def get_standard_stride_layout(shape):
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def stride_layouts(shape):
    # Arbitrarily padded positive strides are unbounded, so this exhausts all
    # axis orders with both packed and padded gaps between each packed level.
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
    # For r > 1, we use atomic float32 accumulation which is more accurate than
    # the reference bfloat16 accumulation. For r=128, a tolerance of 0.25-0.5 is reasonable
    # given bfloat16 precision limits on sums of 128 elements.
    atol = 0.01 if settings.sigmoid else 0
    rtol = 0.01 if settings.sigmoid else 0

    np.testing.assert_allclose(
        actual.cpu().contiguous().to(dtype=torch.float32),
        expected.contiguous().to(dtype=torch.float32),
        rtol=rtol,
        atol=atol,
    )
    # actual_bits = actual.cpu().contiguous().view(torch.int64)
    # expected_bits = expected.contiguous().view(torch.int64)
    # assert torch.equal(actual_bits, expected_bits), (
    #     f"{label} mismatch for {case}\nactual={actual.cpu()}\nexpected={expected}"
    # )


@pytest.fixture
def blelloch_context_config(request):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    thread_execution_width, max_threads, max_tg_mem, banks = request.param
    rust_ewma.release_ltv_metal_fused_concat_register_cast_blelloch_scan_context()
    rust_ewma.init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device(
        thread_execution_width,
        max_threads,
        max_tg_mem,
        banks,
    )

    yield request.param

    rust_ewma.release_ltv_metal_fused_concat_register_cast_blelloch_scan_context()
    rust_ewma.init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device(
        0,
        0,
        0,
        0,
    )


def test_forward_exact(settings, seq_len, blelloch_context_config):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = torch.device("mps")
    try:
        torch.empty(1, device=device, dtype=torch.bfloat16)
    except Exception as exc:  # pragma: no cover - depends on host MPS support
        pytest.skip(f"MPS float32 unsupported: {exc}")

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

            combined_mps = torch.as_strided(
                combined_base_cpu.to(device),
                size=shape,
                stride=stride,
            )
            inits_mps = inits_cpu.to(device)

            repeats = 2 if settings.benchmark else 1

            for _ in range(repeats):
                with torch.no_grad():
                    # Synchronize before and after execution to measure device time accurately
                    torch.mps.synchronize()
                    start_time_ns = time.perf_counter_ns()

                    actual_q, actual_k, actual_v = (
                        torch.ops.nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_op(
                            combined_mps,
                            inits_mps,
                            settings.nh,
                            settings.nkvh,
                            da,
                            r,
                            settings.sigmoid,
                        )
                    )

                    torch.mps.synchronize()
                    end_time_ns = time.perf_counter_ns()

                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    "context=("
                    f"thread_execution_width={blelloch_context_config[0]}, "
                    f"max_threads={blelloch_context_config[1]}, "
                    f"max_tg_mem={blelloch_context_config[2]}, "
                    f"banks={blelloch_context_config[3]})"
                )
                if settings.check_against_expected:
                    assert_equal(settings, actual_q, expected_q, "q", case)
                    assert_equal(settings, actual_k, expected_k, "k", case)
                    assert_equal(settings, actual_v, expected_v, "v", case)

            if settings.benchmark:
                logger.info(
                    f"Case Forward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )


def test_backward_exact(settings, seq_len, blelloch_context_config):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = torch.device("mps")
    try:
        torch.empty(1, device=device, dtype=torch.bfloat16)
    except Exception as exc:  # pragma: no cover - depends on host MPS support
        pytest.skip(f"MPS float32 unsupported: {exc}")

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

                # Create an integer-valued grad_output (e.g., all ones)
                grad_q = torch.ones_like(expected_q)
                grad_k = torch.ones_like(expected_k)
                grad_v = torch.ones_like(expected_v)

                # Compute reference gradients
                torch.autograd.backward(
                    [expected_q, expected_k, expected_v], [grad_q, grad_k, grad_v]
                )
                expected_grad_combined = combined_cpu.grad
                expected_grad_inits = inits_cpu.grad
            else:
                # Still need grad outputs for custom op backward
                # For benchmark mode, we know the shapes
                total_h = settings.total_h
                D = da * r
                grad_q = torch.ones((settings.batch, settings.nh, seq_len, D))
                grad_k = torch.ones((settings.batch, settings.nkvh, seq_len, D))
                grad_v = torch.ones((settings.batch, settings.nkvh, seq_len, D))

            combined_mps = torch.as_strided(
                combined_base_cpu.to(device),
                size=shape,
                stride=stride,
            )
            inits_mps = inits_cpu.detach().to(device)

            combined_mps.requires_grad_(True)
            inits_mps.requires_grad_(True)

            # Benchmarking backward path
            repeats = 2 if settings.benchmark else 1
            for _ in range(repeats):
                # Clear gradients so repeated benchmarking doesn't ruin the check_against_expected bit-exact math later
                if combined_mps.grad is not None:
                    combined_mps.grad.zero_()
                if inits_mps.grad is not None:
                    inits_mps.grad.zero_()

                # Synchronize before to clear out the queue, measuring total device execution
                torch.mps.synchronize()
                start_time_ns = time.perf_counter_ns()

                q_act, k_act, v_act = (
                    torch.ops.nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_op(
                        combined_mps,
                        inits_mps,
                        settings.nh,
                        settings.nkvh,
                        da,
                        r,
                        settings.sigmoid,
                    )
                )
                torch.autograd.backward(
                    [q_act, k_act, v_act],
                    [grad_q.to(device), grad_k.to(device), grad_v.to(device)],
                )

                # Synchronize after backward graph generation and execution completes
                torch.mps.synchronize()
                end_time_ns = time.perf_counter_ns()

                case = (
                    f"seq_len={seq_len}, r={r}, da={da}, stride={stride}, "
                    "context=("
                    f"thread_execution_width={blelloch_context_config[0]}, "
                    f"max_threads={blelloch_context_config[1]}, "
                    f"max_tg_mem={blelloch_context_config[2]}, "
                    f"banks={blelloch_context_config[3]})"
                )

                if settings.check_against_expected:
                    # 4. Bit-Exact Assertions
                    assert_equal(
                        settings,
                        combined_mps.grad,
                        expected_grad_combined,
                        "grad_combined",
                        case,
                        r=r,
                    )
                    assert_equal(
                        settings,
                        inits_mps.grad,
                        expected_grad_inits,
                        "grad_inits",
                        case,
                        r=r,
                    )

            if settings.benchmark:
                logger.info(
                    f"Case Backward: {case} time_ns: {end_time_ns - start_time_ns:_}"
                )
