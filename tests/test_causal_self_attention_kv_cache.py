import itertools
import logging

import pytest

try:
    import rust_ewma
except ImportError:
    rust_ewma = None

import torch

from nanochat.gpt import (
    CausalSelfAttention,
    make_ltv_filter,
    precompute_rotary_embeddings,
)
from nanochat.gpt_config import UNBOUNDED_LAYER_SPEC, GPTConfig
from nanochat.kv_cache import KVCache
from nanochat.ltv.ltv_impls import ltv_pytorch_loop
from nanochat.ltv.ltv_mode import LtvMode

from nanochat.ltv.ltv_metal_impls import (
    ltv_plane_max_metal,
    ltv_scan_metal,
    ltv_shared_max_metal,
    ltv_shared_pmax_metal,
)
from nanochat.split_qkv_computer import SplitQkvComputerModule

logger = logging.getLogger(__name__)


QKV_LTV_CONFIGS = list(itertools.product([LtvMode.NONE, LtvMode.ACTIVE], repeat=3))

LTV_FNS = [
    (ltv_pytorch_loop, lambda: None, lambda: None),
] + (
    [
        (
            ltv_scan_metal,
            rust_ewma.release_ltv_scan_metal_context,
            rust_ewma.init_ltv_scan_metal_context_for_default_device,
        ),
        (
            ltv_plane_max_metal,
            rust_ewma.release_ltv_plane_metal_context,
            rust_ewma.init_ltv_plane_metal_context_for_default_device,
        ),
        (
            ltv_shared_max_metal,
            rust_ewma.release_ltv_shared_metal_context,
            rust_ewma.init_ltv_shared_metal_context_for_default_device,
        ),
        (
            ltv_shared_pmax_metal,
            rust_ewma.release_ltv_shared_metal_context,
            rust_ewma.init_ltv_shared_metal_context_for_default_device,
        ),
    ]
    if rust_ewma
    else []
)


def _make_kv_cache(
    *, batch_size: int, num_q_heads: int, num_kv_heads: int, seq_len: int, head_dim: int
):
    return KVCache(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=1,
        layer_specs=[UNBOUNDED_LAYER_SPEC],
    )


def _run_attention_in_chunks(
    *,
    attn: CausalSelfAttention,
    x_full: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: KVCache,
    chunks: list[int],
) -> torch.Tensor:
    outs = []
    t = 0
    for chunk_len in chunks:
        x = x_full[:, t : t + chunk_len, :]
        t0 = kv_cache.get_pos()
        cos_sin = (cos[:, t0 : t0 + chunk_len], sin[:, t0 : t0 + chunk_len])
        outs.append(attn(x, cos_sin, kv_cache))
        t += chunk_len
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("ltv_query,ltv_key,ltv_value", QKV_LTV_CONFIGS)
@pytest.mark.parametrize("ltv_triple", LTV_FNS)
def test_causal_self_attention_kv_cache_chunking_equivalence(
    ltv_query, ltv_key, ltv_value, ltv_triple
):
    ltv_fn, ltv_release, ltv_init = ltv_triple
    if ltv_fn != ltv_pytorch_loop and not torch.backends.mps.is_available():
        pytest.skip(f"{ltv_fn} not available")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    ltv_release()
    ltv_init()

    torch.manual_seed(123)

    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=6,
        n_kv_head=2,
        n_embd=12,
        ltv_r=2,
        ltv_query=ltv_query,
        ltv_key=ltv_key,
        ltv_value=ltv_value,
        layer_specs=[UNBOUNDED_LAYER_SPEC],
    )

    layer_idx = 0
    attn = CausalSelfAttention(
        config,
        layer_idx,
        SplitQkvComputerModule(
            config,
            layer_idx,
            make_ltv_filter(config, ltv_fn),
        ),
    ).to(device=device)
    attn.eval()

    # Initialize all weights to deterministic non-zero values for testing
    def init_weights_non_zero(m):
        if isinstance(m, torch.nn.Linear):
            # Use 1/k pattern for weights where k increases with parameter count
            # PyTorch Linear expects weight shape (out_features, in_features)
            out_features, in_features = m.weight.shape
            k = torch.arange(
                1,
                in_features * out_features + 1,
                dtype=torch.float32,
                device=m.weight.device,
            ).reshape(out_features, in_features)
            m.weight.data = (
                1.0 / k
            )  # Already in correct shape (out_features, in_features)
            if m.bias is not None:
                m.bias.data = torch.ones_like(m.bias.data) * 0.1
        elif hasattr(m, "q_init") and m.q_init.numel() > 0:
            # Use 1/k pattern for q_init
            k = torch.arange(
                1, m.q_init.numel() + 1, dtype=torch.float32, device=m.q_init.device
            ).reshape(m.q_init.shape)
            m.q_init.data = 1.0 / k
        elif hasattr(m, "k_init") and m.k_init.numel() > 0:
            # Use 1/k pattern for k_init
            k = torch.arange(
                1, m.k_init.numel() + 1, dtype=torch.float32, device=m.k_init.device
            ).reshape(m.k_init.shape)
            m.k_init.data = 1.0 / k
        elif hasattr(m, "v_init") and m.v_init.numel() > 0:
            # Use 1/k pattern for v_init
            k = torch.arange(
                1, m.v_init.numel() + 1, dtype=torch.float32, device=m.v_init.device
            ).reshape(m.v_init.shape)
            m.v_init.data = 1.0 / k

    attn.apply(init_weights_non_zero)

    B, T, C = 2, 3, config.n_embd
    x_full = torch.randn(B, T, C, device=device, dtype=torch.float32)

    seq_len = 8
    cos, sin = precompute_rotary_embeddings(
        seq_len=seq_len, head_dim=config.head_dim(), base=100, device=device
    )

    def run(chunks: list[int]) -> torch.Tensor:
        kv_cache = _make_kv_cache(
            batch_size=B,
            num_q_heads=config.n_head,
            num_kv_heads=config.n_kv_head,
            seq_len=seq_len,
            head_dim=config.head_dim(),
        )
        return _run_attention_in_chunks(
            attn=attn, x_full=x_full, cos=cos, sin=sin, kv_cache=kv_cache, chunks=chunks
        )

    y_3 = run([3])
    logger.debug("y_3: %s", y_3)
    y_1_2 = run([1, 2])
    y_2_1 = run([2, 1])
    y_1_1_1 = run([1, 1, 1])

    torch.testing.assert_close(y_3, y_1_2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(y_3, y_2_1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(y_3, y_1_1_1, rtol=1e-6, atol=1e-6)
