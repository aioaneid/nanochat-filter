# Shared utilities for LTV scan kernel tests
import itertools
import torch
import pytest
import numpy as np
from typing import List, Optional

from nanochat.ltv_fused_concat_register_cast_scan_util import CUDA_SUPPORTS_NATIVE_BF16

# All (P, Q, K) configurations supported by the dispatchers
SUPPORTED_PQK_CONFIGS = [
    (2147483647, 1, 1),
    (1024, 1024, 1),
    (64, 60, 5),
    (128, 112, 17),
    (160, 108, 53),
    (192, 104, 89),
    (224, 100, 125),
]


class Settings:
    def __init__(
        self,
        benchmark: bool,
        check_against_expected: bool,
        batch: int = 4,
        nh: int = 8,
        nkvh: int = 8,
        total_d: Optional[int] = None,
        r_values: Optional[List[int]] = None,
    ):
        self.benchmark = benchmark
        self.check_against_expected = check_against_expected
        self.batch = batch
        self.nh = nh
        self.nkvh = nkvh
        self.total_d = total_d or (128 if benchmark else 2)
        self.r_values = r_values or (
            [self.total_d]
            if benchmark
            else [r for r in range(1, self.total_d + 1) if self.total_d % r == 0]
        )
        self.total_h = self.nh + 2 * self.nkvh


@pytest.fixture(scope="session")
def settings(pytestconfig):
    benchmark = pytestconfig.getoption("--benchmark", "false").lower() == "true"
    check = (
        pytestconfig.getoption("--check_against_expected", "false").lower() == "true"
    )
    return Settings(benchmark=benchmark, check_against_expected=check)


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
                for d in range(D):
                    view[b, t, head_offset + d] = (
                        (9 * b + 7 * t + 3 * h_glob + d) % 5
                    ) - 2
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
    logit_bias = torch.empty((total_h, da), dtype=torch.float32)
    for h_glob in range(total_h):
        for da_idx in range(da):
            logit_bias[h_glob, da_idx] = ((h_glob + 2 * da_idx) % 3) - 1
    return logit_bias


def assert_equal(actual, expected, label, case, use_sigmoid: bool):
    atol = 0.025 if use_sigmoid else 0
    rtol = 0.025 if use_sigmoid else 0
    actual_np = actual.cpu().contiguous().to(torch.float32).numpy()
    expected_np = expected.contiguous().to(torch.float32).numpy()
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
        if num_mismatch:
            max_diff_idx = np.argmax(diff)
            max_diff_val = diff.flat[max_diff_idx]
            idx_tuple = np.unravel_index(max_diff_idx, diff.shape)
            print(f"Max absolute difference: {max_diff_val} at {idx_tuple}")
            print(
                f"  actual: {actual_np[idx_tuple]}, expected: {expected_np[idx_tuple]}"
            )
            for i in range(min(num_mismatch, 10)):
                idx = tuple(m[i] for m in mismatches)
                print(f"  {idx}: actual={actual_np[idx]}, expected={expected_np[idx]}")
        raise e


# Utilities for dynamic test parametrization (to be called from each test file)
def add_common_parametrization(metafunc, include_scan_threads=True):
    benchmark = metafunc.config.getoption("--benchmark", "false").lower() == "true"

    if "seq_len" in metafunc.fixturenames:
        seq_lengths = [1024] if benchmark else [1, 2, 3, 31, 32, 33]
        metafunc.parametrize("seq_len", seq_lengths)

    if include_scan_threads and "scan_threads_forward" in metafunc.fixturenames:
        metafunc.parametrize("scan_threads_forward", [1, 8])

    if "r_threads_forward" in metafunc.fixturenames:
        metafunc.parametrize("r_threads_forward", [2, 32])

    if "r_items_forward" in metafunc.fixturenames:
        metafunc.parametrize("r_items_forward", [1, 4])

    if include_scan_threads and "scan_threads_backward" in metafunc.fixturenames:
        metafunc.parametrize("scan_threads_backward", [1, 8])

    if "r_threads_backward" in metafunc.fixturenames:
        metafunc.parametrize("r_threads_backward", [2, 32])

    if "r_items_backward" in metafunc.fixturenames:
        metafunc.parametrize("r_items_backward", [1, 4])

    if "pqk_config" in metafunc.fixturenames:
        use_sigmoid = metafunc.config.getoption("--sigmoid", "false").lower() == "true"
        configs = [cfg + (use_sigmoid,) for cfg in SUPPORTED_PQK_CONFIGS]
        metafunc.parametrize("pqk_config", configs)
