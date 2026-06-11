"""
Test GPT model with LTV integration.

Tests:
a) ltv_r=0 generates same sequence as ltv_r=1 with all modes NONE
b) For ltv_r=1 and all 8 mode possibilities, token-by-token generation
   equals all-at-once generation (KV cache correctness)
c) For all LTV implementations, compare against pytorch_loop baseline
"""

from typing import Callable
from nanochat.gpt_factory import make_split_original_model, make_concat_original_model
import pytest
import torch
import itertools
import logging

from dataclasses import replace
from nanochat.gpt import (
    GPT,
    make_gpt_with_ltv_fn,
    uniform_normal_std_equalizer,
)
from nanochat.gpt_config import UNBOUNDED_LAYER_SPEC, GPTConfig
from nanochat.ltv.ltv_mode import LtvMode
from nanochat.ltv.ltv_impls import (
    ltv_pytorch_loop,
    LtvRustCpu,
    py_identity,
)

logger = logging.getLogger(__name__)


# Determine device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# LTV implementations to test against baseline
IMPLEMENTATIONS = [
    ("pytorch_loop", ltv_pytorch_loop),
    ("rust_cpu", LtvRustCpu.apply),
]
IMPL_IDS = [name for name, _ in IMPLEMENTATIONS]

# Small config for testing (similar to test_ltv_filter.py params)
SMALL_CONFIG = GPTConfig(
    sequence_len=32,
    vocab_size=256,
    n_layer=2,
    n_head=4,
    n_kv_head=2,
    n_embd=32,
    ltv_r=0,
    ltv_query=LtvMode.NONE,
    ltv_key=LtvMode.NONE,
    ltv_value=LtvMode.NONE,
)


def init_projections(model, std):
    for block in model.transformer.h:
        # Change from zeros to identity or small random values
        # This ensures the LTV output actually reaches the Loss
        torch.nn.init.normal_(block.attn.c_proj.weight, std=std)
        torch.nn.init.normal_(block.mlp.c_proj.weight, std=std)


def create_model_with_ltv(
    config: GPTConfig, model_factory: Callable[[GPTConfig], GPT], ltv_fn, seed: int = 42
) -> GPT:
    """Create and initialize a GPT model with LTV function applied to all layers."""
    torch.manual_seed(seed)
    model = (
        model_factory(config)
        if not config.ltv_r
        else make_gpt_with_ltv_fn(config, ltv_fn)
    ).to(DEVICE)
    model.init_weights(0, rand_filter_bias=False, rand_filter_init=False)
    projection_std = uniform_normal_std_equalizer(config.n_embd)
    init_projections(model, projection_std)
    if config.ltv_r:
        for block in model.transformer.h:
            filt = block.attn.qkv_computer.ltv_filter
            # Use Kaiming/Xavier for the MLP to get healthy random variance
            torch.nn.init.kaiming_normal_(filt.alpha_mlp.weight)
            torch.nn.init.normal_(filt.alpha_mlp.bias, std=projection_std)
            # Randomize the learnable initial states
            torch.nn.init.normal_(filt.q_init, std=projection_std)
            torch.nn.init.normal_(filt.k_init, std=projection_std)
            torch.nn.init.normal_(filt.v_init, std=projection_std)
    return model


def create_model(
    config: GPTConfig, model_factory: Callable[[GPTConfig], GPT], seed: int = 42
) -> GPT:
    """Create and initialize a GPT model with deterministic weights."""
    return create_model_with_ltv(
        config, model_factory=model_factory, ltv_fn=py_identity, seed=seed
    )


def run_forward_backward(model: GPT, seq_len: int = 8):
    """Run one forward/backward pass to warm up the model."""
    torch.manual_seed(123)  # deterministic input
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=DEVICE)
    targets = torch.randint(0, model.config.vocab_size, (1, seq_len), device=DEVICE)

    loss = model.forward(input_ids, targets=targets)
    loss.backward()

    # Zero gradients after backward
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def generate_sequence(
    model: GPT, prompt: list, num_tokens: int = 3, seed: int = 42
) -> list:
    """Generate tokens using the model's generate function."""
    model.eval()
    tokens = []
    for token in model.generate(
        prompt, max_tokens=num_tokens, temperature=0.0, seed=seed
    ):
        tokens.append(token)
    return tokens


def test_ltv_disabled_equals_all_none():
    """
    Test that ltv_r=0 produces the same sequence as
    ltv_r=1 with all modes set to NONE.
    """
    # Create config with ltv disabled
    config_disabled = replace(
        SMALL_CONFIG,
        ltv_r=0,
        ltv_query=LtvMode.NONE,
        ltv_key=LtvMode.NONE,
        ltv_value=LtvMode.NONE,
    )
    split_model_disabled = create_model(
        config_disabled, model_factory=make_split_original_model, seed=42
    )
    run_forward_backward(split_model_disabled)

    concat_model_disabled = create_model(
        config_disabled, model_factory=make_concat_original_model, seed=42
    )
    run_forward_backward(concat_model_disabled)

    # Create config with ltv enabled but all modes NONE
    config_enabled_none = replace(
        SMALL_CONFIG,
        ltv_r=1,
        ltv_query=LtvMode.NONE,
        ltv_key=LtvMode.NONE,
        ltv_value=LtvMode.NONE,
        layer_specs=[UNBOUNDED_LAYER_SPEC] * SMALL_CONFIG.n_layer,
    )
    split_model_enabled_none = create_model(
        config_enabled_none, model_factory=make_split_original_model, seed=42
    )
    run_forward_backward(split_model_enabled_none)

    concat_model_enabled_none = create_model(
        config_enabled_none, model_factory=make_concat_original_model, seed=42
    )
    run_forward_backward(concat_model_enabled_none)

    # Generate sequences
    prompt = [1, 2, 3, 4]  # Small prompt
    split_tokens_disabled = generate_sequence(
        split_model_disabled, prompt, num_tokens=3
    )
    concat_tokens_disabled = generate_sequence(
        concat_model_disabled, prompt, num_tokens=3
    )
    split_tokens_enabled_none = generate_sequence(
        split_model_enabled_none, prompt, num_tokens=3
    )
    concat_tokens_enabled_none = generate_sequence(
        concat_model_enabled_none, prompt, num_tokens=3
    )

    assert split_tokens_disabled == split_tokens_enabled_none, (
        f"ltv_r=0 should produce same tokens as ltv_r=1 with all NONE modes.\n"
        f"Disabled: {split_tokens_disabled}\n"
        f"Enabled (all NONE): {split_tokens_enabled_none}"
    )
    assert split_tokens_disabled == concat_tokens_disabled
    assert concat_tokens_disabled == concat_tokens_enabled_none


# Generate all 8 combinations of LtvMode for q, k, v
LTV_MODES = list(itertools.product([LtvMode.NONE, LtvMode.ACTIVE], repeat=3))
LTV_MODE_IDS = [f"q={m[0].name},k={m[1].name},v={m[2].name}" for m in LTV_MODES]

MODEL_FACTORY_IDS = ["split", "concat"]
MODEL_FACTORIES = [make_split_original_model, make_concat_original_model]


@pytest.mark.parametrize("modes", LTV_MODES, ids=LTV_MODE_IDS)
@pytest.mark.parametrize("model_factory", MODEL_FACTORIES, ids=MODEL_FACTORY_IDS)
def test_kv_cache_token_by_token_equals_batch(modes, model_factory):
    """
    Test that generating tokens one-by-one (using KV cache) produces
    the same sequence as generating all at once (no KV cache).

    This verifies the KV cache and LTV state persistence work correctly.
    """
    q_mode, k_mode, v_mode = modes

    config = replace(
        SMALL_CONFIG,
        ltv_r=1,
        ltv_query=q_mode,
        ltv_key=k_mode,
        ltv_value=v_mode,
        layer_specs=[UNBOUNDED_LAYER_SPEC] * SMALL_CONFIG.n_layer,
    )

    # Create model
    model = create_model(config, model_factory=model_factory, seed=42)
    run_forward_backward(model)
    model.eval()

    prompt = [1, 2, 3, 4]
    num_tokens = 3

    # Method 1: Generate all at once (the model.generate uses incremental KV cache)
    tokens_generated = generate_sequence(model, prompt, num_tokens=num_tokens, seed=42)

    # Method 2: Generate by running full forward each time (no KV cache)
    torch.manual_seed(42)
    with torch.inference_mode():
        ids = torch.tensor([prompt], dtype=torch.long, device=DEVICE)
        tokens_batch = []

        for _ in range(num_tokens):
            logits = model.forward(ids)  # Full forward, no KV cache
            logits = logits[:, -1, :]  # Last position
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            tokens_batch.append(next_id.item())

    assert tokens_generated == tokens_batch, (
        f"Token-by-token generation should match batch generation.\n"
        f"Modes: q={q_mode.name}, k={k_mode.name}, v={v_mode.name}\n"
        f"Generated (KV cache): {tokens_generated}\n"
        f"Batch (no KV cache): {tokens_batch}"
    )


@pytest.mark.parametrize("ltv_fn_name, ltv_fn", IMPLEMENTATIONS, ids=IMPL_IDS)
@pytest.mark.parametrize("modes", LTV_MODES, ids=LTV_MODE_IDS)
def test_ltv_implementations_against_baseline(ltv_fn_name, ltv_fn, modes):
    """
    Test that all LTV implementations produce the same results as the pytorch_loop baseline.
    This tests the integration of LTV functions into the GPT model.
    """
    q_mode, k_mode, v_mode = modes

    config = replace(
        SMALL_CONFIG,
        ltv_r=1,
        ltv_query=q_mode,
        ltv_key=k_mode,
        ltv_value=v_mode,
    )

    # Skip baseline comparison for baseline itself
    if ltv_fn_name == "pytorch_loop":
        logger.debug("Comparing %s against itself.", ltv_fn_name)

    # Create baseline model (pytorch_loop)
    baseline_model = create_model_with_ltv(
        config,
        model_factory=make_split_original_model,
        ltv_fn=ltv_pytorch_loop,
        seed=42,
    )
    run_forward_backward(baseline_model)
    baseline_model.eval()

    # Create target model with current implementation
    target_model = create_model_with_ltv(
        config, model_factory=make_concat_original_model, ltv_fn=ltv_fn, seed=42
    )
    run_forward_backward(target_model)
    target_model.eval()

    prompt = [1, 2, 3, 4]
    num_tokens = 3

    # Generate sequences from both models
    torch.manual_seed(42)
    baseline_tokens = generate_sequence(
        baseline_model, prompt, num_tokens=num_tokens, seed=42
    )

    torch.manual_seed(42)
    target_tokens = generate_sequence(
        target_model, prompt, num_tokens=num_tokens, seed=42
    )

    assert baseline_tokens == target_tokens, (
        f"LTV implementation {ltv_fn_name} should match baseline pytorch_loop.\n"
        f"Modes: q={q_mode.name}, k={k_mode.name}, v={v_mode.name}\n"
        f"Baseline (pytorch_loop): {baseline_tokens}\n"
        f"Target ({ltv_fn_name}): {target_tokens}"
    )
