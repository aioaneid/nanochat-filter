import itertools
import torch
import pytest

from nanochat.ltv_look_back_qkv_computer_module import (
    LtvLookBackQkvComputerModule,
    advance_kv_cache,
)
from nanochat.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter import (
    LtvLookBackFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_qkv_filter import (
    LtvLookBackFusedConcatHeadMajorRegisterCastSequentialScanQkvFilter,
)
from nanochat.ops.ltv_reference_ops import (
    fused_linear_transpose_reference,
    ltv_look_back_blelloch_scan_reference,
    ltv_sequential_scan_reference,
)
from nanochat.kv_cache import KVCache
from nanochat.gpt_config import ChunksAndHistory


# -------------------------------------------------------------------------
# Helper to build a minimal filter + computer module for testing
# -------------------------------------------------------------------------
def make_blelloch_filter(n_head, n_kv_head, head_dim, r, layer_spec):
    return LtvLookBackFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter(
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        r=r,
        layer_spec=layer_spec,
        transpose_op=fused_linear_transpose_reference,
        look_back_op=ltv_look_back_blelloch_scan_reference,
    )


def make_sequential_filter(n_head, n_kv_head, head_dim, r, layer_spec):
    return LtvLookBackFusedConcatHeadMajorRegisterCastSequentialScanQkvFilter(
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        r=r,
        layer_spec=layer_spec,
        sequential_look_back_op=ltv_sequential_scan_reference,
    )


FILTER_VARIANTS = [
    pytest.param(make_blelloch_filter, id="blelloch_scan"),
    pytest.param(make_sequential_filter, id="sequential_scan"),
]


def make_test_module(make_ltv_filter, p, q, k, n_head=1, n_kv_head=1, head_dim=2, r=1):
    """Create an LtvLookBackQkvComputerModule with deterministic weights."""
    ltv_filter = make_ltv_filter(
        n_head,
        n_kv_head,
        head_dim,
        r,
        ChunksAndHistory(p=p, q=q, k=k),
    )
    # Set weights to fixed values (e.g., ones) for reproducibility
    with torch.no_grad():
        ltv_filter.c.weight.fill_(1.0)
        ltv_filter.logit_bias.fill_(0.0)
        ltv_filter.init_param.fill_(0.0)

    computer = LtvLookBackQkvComputerModule(
        layer_idx=0,
        ltv_filter=ltv_filter,
    )
    return computer, ltv_filter


# -------------------------------------------------------------------------
# Generate all token count sequences with product(range(A), repeat=N)
# -------------------------------------------------------------------------
def generate_token_count_sequences(N, A):
    """Return list of sequences (tuples of length N) where each element in [0..A-1]."""
    return list(itertools.product(range(A), repeat=N))


# -------------------------------------------------------------------------
# Test parameters
# -------------------------------------------------------------------------
N_VALUES = [1, 2, 3]  # number of steps
A_VALUES = [2, 3]  # max tokens per step (0..A-1)
# (p, q, k) configurations to test
CHUNK_PARAMS = [
    (3, 2, 2),  # tiny chunks
    (5, 3, 2),
    (4, 2, 3),
    (1, 1, 1),
    (10, 2, 5),
]


@pytest.mark.parametrize("N", N_VALUES)
@pytest.mark.parametrize("A", A_VALUES)
@pytest.mark.parametrize("p,q,k", CHUNK_PARAMS)
@pytest.mark.parametrize("make_ltv_filter", FILTER_VARIANTS)
def test_advance_kv_cache_exhaustive(make_ltv_filter, N, A, p, q, k):
    token_count_sequences = generate_token_count_sequences(N, A)

    n_head = 1
    n_kv_head = 1
    head_dim = 2
    r = 1
    n_embd = n_head * head_dim

    computer, ltv_filter = make_test_module(
        make_ltv_filter, p, q, k, n_head, n_kv_head, head_dim, r
    )
    computer.eval()

    for counts in token_count_sequences:
        total_tokens = sum(counts)
        # Generate random data – even if total_tokens == 0, we create an empty tensor
        torch.manual_seed(42)
        x_full = (
            torch.randn(1, total_tokens, n_embd)
            if total_tokens > 0
            else torch.empty(1, 0, n_embd)
        )

        B = 1
        resolved_inits = ltv_filter.init_param.unsqueeze(0).expand(B, -1, -1)

        layer_spec = ChunksAndHistory(p=p, q=q, k=k)
        cache_full = KVCache(
            batch_size=B,
            num_q_heads=n_head,
            num_kv_heads=n_kv_head,
            seq_len=max(1, total_tokens) + 10,
            head_dim=head_dim,
            num_layers=1,
            layer_specs=[layer_spec],
        )

        # Method 1: all at once
        with torch.no_grad():
            q1, k1, v1 = advance_kv_cache(
                x_full, 0, ltv_filter, resolved_inits, cache_full
            )

        # Method 2: step by step with a single cache
        cache_step = KVCache(
            batch_size=B,
            num_q_heads=n_head,
            num_kv_heads=n_kv_head,
            seq_len=max(1, total_tokens) + 10,
            head_dim=head_dim,
            num_layers=1,
            layer_specs=[layer_spec],
        )
        q_parts, k_parts, v_parts = [], [], []
        start = 0
        for t in counts:
            x_step = (
                x_full[:, start : start + t, :] if t > 0 else torch.empty(1, 0, n_embd)
            )
            start += t
            with torch.no_grad():
                qs, ks, vs = advance_kv_cache(
                    x_step, 0, ltv_filter, resolved_inits, cache_step
                )
            q_parts.append(qs)
            k_parts.append(ks)
            v_parts.append(vs)

        q2 = (
            torch.cat(q_parts, dim=1)
            if q_parts
            else torch.empty(1, 0, n_head, head_dim)
        )
        k2 = (
            torch.cat(k_parts, dim=1)
            if k_parts
            else torch.empty(1, 0, n_kv_head, head_dim)
        )
        v2 = (
            torch.cat(v_parts, dim=1)
            if v_parts
            else torch.empty(1, 0, n_kv_head, head_dim)
        )

        # Method 3: step by step, prefill after each step
        cache_prev = KVCache(
            batch_size=B,
            num_q_heads=n_head,
            num_kv_heads=n_kv_head,
            seq_len=max(1, total_tokens) + 10,
            head_dim=head_dim,
            num_layers=1,
            layer_specs=[layer_spec],
        )
        q_parts3, k_parts3, v_parts3 = [], [], []
        start = 0
        for t in counts:
            x_step = (
                x_full[:, start : start + t, :] if t > 0 else torch.empty(1, 0, n_embd)
            )
            start += t
            with torch.no_grad():
                qs, ks, vs = advance_kv_cache(
                    x_step, 0, ltv_filter, resolved_inits, cache_prev
                )
            q_parts3.append(qs)
            k_parts3.append(ks)
            v_parts3.append(vs)

            # Prefill new cache
            cache_new = KVCache(
                batch_size=B,
                num_q_heads=n_head,
                num_kv_heads=n_kv_head,
                seq_len=max(1, total_tokens) + 10,
                head_dim=head_dim,
                num_layers=1,
                layer_specs=[layer_spec],
            )
            cache_new.prefill(cache_prev)
            cache_prev = cache_new

        q3 = (
            torch.cat(q_parts3, dim=1)
            if q_parts3
            else torch.empty(1, 0, n_head, head_dim)
        )
        k3 = (
            torch.cat(k_parts3, dim=1)
            if k_parts3
            else torch.empty(1, 0, n_kv_head, head_dim)
        )
        v3 = (
            torch.cat(v_parts3, dim=1)
            if v_parts3
            else torch.empty(1, 0, n_kv_head, head_dim)
        )

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)
        torch.testing.assert_close(v1, v2)

        torch.testing.assert_close(q1, q3)
        torch.testing.assert_close(k1, k3)
        torch.testing.assert_close(v1, v3)
