"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

from contextlib import nullcontext
import logging
import pytest
from nanochat.bench.bench_gpt_ltv import (
    create_concat_original_model,
    create_model_with_fused_ltv,
)
from nanochat.engine import Engine
from nanochat.gpt import make_gpt_with_concat_fused_scan_ltv
from nanochat.gpt_config import UNBOUNDED_LAYER_SPEC, GPTConfig
from nanochat.ltv.ltv_mode import LtvMode

try:
    import rust_ewma
except ImportError:
    rust_ewma = None

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

# Parameterized test configurations for MockModel
LTV_CONFIGURATIONS = [
    (0, LtvMode.NONE, LtvMode.NONE, LtvMode.NONE),
    (1, LtvMode.ACTIVE, LtvMode.ACTIVE, LtvMode.ACTIVE),
]


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """

    def __init__(self):
        # Special tokens start at 256
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
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


@pytest.mark.parametrize("ltv_r,ltv_query,ltv_key,ltv_value", LTV_CONFIGURATIONS)
def test_kv_cache_matches_naive_generate(ltv_r, ltv_query, ltv_key, ltv_value):
    """
    Verify that generation using Engine (which uses KVCache)
    produces identical tokens to naive GPT.generate (no KVCache).

    We use temperature=0 to remove stochasticity.
    """

    config = GPTConfig(
        vocab_size=262,
        sequence_len=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        ltv_r=ltv_r,
        ltv_query=ltv_query,  # Assuming we want to test active LTV
        ltv_key=ltv_key,
        ltv_value=ltv_value,
        layer_specs=[UNBOUNDED_LAYER_SPEC] * (2 if ltv_r else 0),
    )

    if rust_ewma:
        rust_ewma.release_ltv_fused_scan_metal_context()
        try:
            rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
        except RuntimeError as e:
            if not any("already initialized" in str(arg) for arg in e.args):
                raise
    else:
        pytest.skip("rust_ewma not available")

    autocast_ctx = nullcontext()
    model = (
        create_model_with_fused_ltv(
            config,
            seed=42,
            autocast_ctx=autocast_ctx,
            model_builder_fn=make_gpt_with_concat_fused_scan_ltv,
        )
        if ltv_r
        else create_concat_original_model(config, seed=42, autocast_ctx=autocast_ctx)
    )

    # --- Setup model ---

    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    max_tokens = 10

    # --- 1) Generate using Engine (KV cache path) ---
    engine_results, _ = engine.generate_batch(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,  # deterministic
        num_samples=1,
        seed=42,
    )
    engine_tokens = engine_results[0]
    logger.info("engine_tokens: %s", engine_tokens)

    # --- 2) Generate using naive model.generate (no KV cache) ---
    naive_tokens = list(prompt)
    for token in model.generate(prompt, max_tokens=max_tokens, temperature=0.0):
        naive_tokens.append(token)
    logger.info("naive_tokens: %s", naive_tokens)

    # --- Compare full sequences ---
    assert engine_tokens == naive_tokens, (
        "Generation with KVCache does not match naive generation.\n"
        f"Engine: {engine_tokens}\n"
        f"Naive:  {naive_tokens}"
    )
