"""
Test the Engine class using a GPT model that employs the pure-Python
reference look-back LTV op. Verifies that generation with KV cache
matches naive (no-cache) generation, for various chunk parameters.

Usage:
    python -m pytest tests/test_engine_with_full_lower_bound_reference_model.py -v
"""

import logging
import pytest

from nanochat.gpt_config import ChunksAndHistory, GPTConfig
from nanochat.ltv.ltv_mode import LtvMode
from nanochat.engine import Engine
from nanochat.gpt import GPT, CausalSelfAttention
from nanochat.ltv_look_back_qkv_computer_module import LtvLookBackQkvComputerModule
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

logger = logging.getLogger(__name__)

UNBOUNDED_P = 2_147_483_647


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


# ---------------------------------------------------------------------------
# ByteTokenizer – copied from the existing engine test for convenience
# ---------------------------------------------------------------------------
class ByteTokenizer:
    def __init__(self):
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Model builder for the reference look-back LTV
# ---------------------------------------------------------------------------
def make_gpt_with_lower_bound_reference(config: GPTConfig, make_ltv_filter):
    """
    Create a GPT model where every LTV layer uses the pure-Python reference
    look-back op. The per-layer chunking parameters are taken from the config.
    """
    assert config.ltv_r
    def make_attn(layer_idx):
        ltv_filter = make_ltv_filter(
            config.n_head,
            config.n_kv_head,
            config.head_dim(),
            config.ltv_r,
            config.layer_specs[layer_idx],
        )
        return CausalSelfAttention(
            config,
            layer_idx,
            LtvLookBackQkvComputerModule(
                layer_idx=layer_idx,
                ltv_filter=ltv_filter,
            ),
        )

    return GPT(
        config,
        attns=[make_attn(layer_idx) for layer_idx in range(config.n_layer)],
    )


# ---------------------------------------------------------------------------
# Parameterised generation test
# ---------------------------------------------------------------------------
CHUNK_PARAMS = [
    (3, 2, 2),  # tiny chunks
    (5, 3, 2),
    (UNBOUNDED_P, 1, 1),  # full recurrence (unbounded)
    (10, 4, 6),
    (8, 2, 3),
]


@pytest.mark.parametrize("P,K,Q", CHUNK_PARAMS)
@pytest.mark.parametrize("make_ltv_filter", FILTER_VARIANTS)
def test_reference_lower_bound_generation(make_ltv_filter, P, K, Q):
    """
    Generate a small sequence using Engine (with KV cache) and compare
    to naive model.generate (without cache). Must match token-for-token.
    """
    # Minimal model config – keep n_layer=2 to stress the multi-layer cache.
    config = GPTConfig(
        vocab_size=262,
        sequence_len=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        ltv_r=2,  # head_dim=16 → r=2, D=16, Da=8
        ltv_query=LtvMode.ACTIVE,
        ltv_key=LtvMode.ACTIVE,
        ltv_value=LtvMode.ACTIVE,
        layer_specs=[ChunksAndHistory(p=P, k=K, q=Q)] * 2,
    )

    # Build the reference model and initialise weights.
    model = make_gpt_with_lower_bound_reference(config, make_ltv_filter)
    model.init_weights(
        alpha_mlp_bias_init=0.0, rand_filter_bias=False, rand_filter_init=False
    )
    # The reference op runs on CPU; keep everything on CPU.
    model.to("cpu")
    model.eval()

    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    max_tokens = 10

    # 1) Generate via Engine (KV cache path)
    engine_results, _ = engine.generate_batch(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        num_samples=1,
        seed=42,
        stop_tokens=[],
    )
    engine_tokens = engine_results[0]

    # 2) Generate via naive model.generate (no cache)
    naive_tokens = list(prompt)
    for token in model.generate(prompt, max_tokens=max_tokens, temperature=0.0):
        naive_tokens.append(token)

    # Assert equality
    assert engine_tokens == naive_tokens, (
        f"Reference look-back LTV (P={P}, K={K}, Q={Q}): "
        f"Engine tokens differ from naive generation.\n"
        f"Engine: {engine_tokens}\n"
        f"Naive:  {naive_tokens}"
    )
