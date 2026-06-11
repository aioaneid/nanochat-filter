from nanochat.gpt import make_gpt_with_ltv_metal_fused_concat_register_cast_blelloch_scan_ltv, make_gpt_with_fused_concat_register_cast_scan_metal_ltv
from nanochat.gpt import make_gpt_with_concat_register_cast_scan_cuda_ltv, make_gpt_with_fused_concat_register_cast_scan_cuda_ltv, make_gpt_with_fused_concat_head_major_register_cast_scan_cuda_ltv, make_gpt_with_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_ltv
from collections.abc import Iterable
from typing import Callable
from nanochat.gpt_factory import make_concat_original_model, make_split_original_model
import torch

import gc
import time
import argparse
import math
import sys
import logging
try:
    import rust_ewma
except ImportError:
    rust_ewma = None
from datetime import timedelta
import numpy as np
from contextlib import nullcontext
from dataclasses import replace
import torch._dynamo as dynamo
import torch.amp

# Ensure we can import from the package
from nanochat.gpt import (
    GPT,
    GPTConfig,
    make_gpt_with_concat_fused_scan_ltv,
    make_gpt_with_concat_fused_scan_triton_ltv,
    make_gpt_with_concat_fused_scan_cuda_ltv,
    make_gpt_with_fused_concat_scan_ltv,
    make_gpt_with_fused_concat_scan_triton_ltv,
    make_gpt_with_fused_scan_ltv,
    make_gpt_with_fused_linear_scan_ltv,
    make_gpt_with_fused_py_torch_ltv,
    make_gpt_with_ltv_fn,
    uniform_normal_std_equalizer,
)
from nanochat.ltv.ltv_mode import LtvMode
from nanochat.ltv.ltv_impls import (
    AutogradIdentity,
    Noop,
    ltv_pytorch_loop,
    LtvRustCpu,
)
from nanochat.common import setup_default_logging
from nanochat.ltv.ltv_impls import py_identity
from nanochat.ltv.ltv_metal_impls import (
    ltv_scan_metal,
    LtvScanMetal,
    ltv_noop_metal,
    ltv_shared_log_metal,
    ltv_shared_max_metal,
    ltv_shared_pmax_metal,
    ltv_plane_log_metal,
    ltv_plane_max_metal,
)
from nanochat.ltv.ltv_triton_impls import (
    ltv_associative_scan_triton,
)
from nanochat.ltv.ltv_cuda_impls import (
    ltv_plane_log_cuda,
    ltv_plane_max_cuda,
)

import nanochat.ops.ltv_triton_associative_scan

setup_default_logging()

logger = logging.getLogger(__name__)


# Device Configuration (PyTorch needs 'cuda' or 'mps' or 'cpu')
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_torch_device()
logger.info(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def init_projections(model, std):
    for block in model.transformer.h:
        # Change from zeros to identity or small random values
        # This ensures the LTV output actually reaches the Loss
        torch.nn.init.normal_(block.attn.c_proj.weight, std=std)
        torch.nn.init.normal_(block.mlp.c_proj.weight, std=std)


def create_original_model(
    config: GPTConfig, seed: int, autocast_ctx: torch.amp.autocast
):
    """Create and initialize a GPT model with LTV function applied to all layers."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    # Ensure ltv_r is True so the custom function is actually used
    config = replace(
        config,
        ltv_r=0,
        ltv_query=LtvMode.NONE,
        ltv_key=LtvMode.NONE,
        ltv_value=LtvMode.NONE,
    )
    model = make_split_original_model(config).to(DEVICE)
    model.init_weights(
        alpha_mlp_bias_init=0, rand_filter_bias=False, rand_filter_init=False
    )
    projection_std = uniform_normal_std_equalizer(config.n_embd)
    init_projections(model, projection_std)
    # Even the original modal recompiles for every sequence size if dynamic is False.
    # But with True it seems to recompile only once, so it probably marks that dimension
    # as being dynamic. Who knows.
    for i, (name, p) in enumerate(model.named_parameters()):
        logger.debug("LTV    Model param[%d: %s]: %s", i, name, p)
    return model


def create_concat_original_model(
    config: GPTConfig, seed: int, autocast_ctx: torch.amp.autocast
):
    """Create and initialize a GPT model with LTV function applied to all layers."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    # Ensure ltv_r is True so the custom function is actually used
    config = replace(
        config,
        ltv_r=0,
        ltv_query=LtvMode.NONE,
        ltv_key=LtvMode.NONE,
        ltv_value=LtvMode.NONE,
    )
    model = make_concat_original_model(config).to(DEVICE)
    model.init_weights(
        alpha_mlp_bias_init=0, rand_filter_bias=False, rand_filter_init=False
    )
    projection_std = uniform_normal_std_equalizer(config.n_embd)
    init_projections(model, projection_std)
    # Even the original modal recompiles for every sequence size if dynamic is False.
    # But with True it seems to recompile only once, so it probably marks that dimension
    # as being dynamic. Who knows.
    for i, (name, p) in enumerate(model.named_parameters()):
        logger.debug("LTV    Model param[%d: %s]: %s", i, name, p)
    return model


def create_model_with_ltv(
    config: GPTConfig, ltv_fn, seed: int, autocast_ctx: torch.amp.autocast
) -> GPT:
    """Create and initialize a GPT model with LTV function applied to all layers."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    # Ensure ltv_r is True so the custom function is actually used
    config = replace(config, ltv_r=config.ltv_r)
    model = make_gpt_with_ltv_fn(config, ltv_fn).to(DEVICE)
    model.init_weights(
        alpha_mlp_bias_init=0, rand_filter_bias=True, rand_filter_init=True
    )
    projection_std = uniform_normal_std_equalizer(config.n_embd)
    init_projections(model, projection_std)
    for i, block in enumerate(model.transformer.h):
        for j, p in enumerate(block.attn.qkv_computer.ltv_filter.get_init_parameters()):
            logger.debug("LTV   init_parameters[%d, %d]: %s", i, j, p)
    for i, (name, p) in enumerate(model.named_parameters()):
        logger.debug("LTV    Model param[%d: %s]: %s", i, name, p)
    return model


def create_model_with_fused_ltv(
    config: GPTConfig,
    seed: int,
    autocast_ctx: torch.amp.autocast,
    model_builder_fn: Callable[[GPTConfig], GPT],
) -> GPT:
    """Create and initialize a GPT model with LTV function applied to all layers."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    # Ensure ltv_r is True so the custom function is actually used
    config = replace(config, ltv_r=config.ltv_r)
    model = model_builder_fn(config).to(DEVICE)
    model.init_weights(
        alpha_mlp_bias_init=0, rand_filter_bias=True, rand_filter_init=True
    )
    projection_std = uniform_normal_std_equalizer(config.n_embd)
    init_projections(model, projection_std)
    for i, block in enumerate(model.transformer.h):
        for j, p in enumerate(block.attn.qkv_computer.ltv_filter.get_init_parameters()):
            logger.debug("Fused init_parameters[%d, %d]: %s", i, j, p)

    for i, (name, p) in enumerate(model.named_parameters()):
        logger.debug("LTV    Model param[%d: %s]: %s", i, name, p)
    return model


def run_pass(model, input_ids, targets, autocast_ctx: torch.amp.autocast):
    """Runs a single forward + backward pass."""
    # Zero grads
    for p in model.parameters():
        p.grad = None

    # Forward
    with autocast_ctx:
        loss = model(input_ids, targets=targets, probability_distribution=True)

    # torch.mps.synchronize()

    # Backward
    loss.backward()

    return loss


def get_grad_diagnostics(model):
    """
    Returns a dictionary of {layer_name: sum_of_grads}.
    This acts as a 'fingerprint' for the backward pass.
    """
    synchronize()

    stats = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            stats[name] = 0.0
        else:
            stats[name] = p.grad.detach().float().sum().item()
    return stats


def get_full_grads(model):
    """Collects a flat vector of gradients for verification."""
    synchronize()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
        else:
            # Use a zero-filler only if necessary
            grads.append(torch.zeros(p.numel(), device=p.device))

    # if not grads:
    #     return torch.tensor([0.0], device=DEVICE)

    return torch.cat(grads)


def get_sum_grads(model):
    """Collects a flat vector of gradients for verification."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten().sum((0,), keepdim=True))

    # if not grads:
    #     return torch.tensor([0.0], device=DEVICE)

    return torch.cat(grads).sum().view(1)


# -----------------------------------------------------------------------------
# Benchmark Logic
# -----------------------------------------------------------------------------


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif DEVICE.type == "mps":
        torch.mps.manual_seed(seed)


def get_detailed_grads(model):
    """Returns a dict of parameter names to their gradient sums."""
    # Ensure all ops are finished before reading memory
    synchronize()

    return {
        name: p.grad.detach().cpu().numpy().copy()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def check_history(history, keep_full_grads: bool):
    ref_loss, ref_flat_grads, ref_grad_stats = history[0]

    for i in range(1, len(history)):
        logger.debug("i: %d", i)
        curr_loss, curr_flat_grads, curr_grad_stats = history[i]

        np.testing.assert_equal(curr_loss, ref_loss)

        # Orig needs at iteration 37 with command:
        # (cd rust_ewma && uv run maturin build --features python,metal --release) && uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl && PYTHONUNBUFFERED=1 METAL_DEVICE_CHECK_ERRORS=1 MT_DEVICE_CHECK_ERRORS=1 uv run --no-sync -m nanochat.bench.bench_gpt_ltv --start_t 2231 --end_t -1 --step_t -1207 --h 8 --b 1 --r 128 --n_layer 12 --warmup 4 --repeat 128 --no-verify --sequence_len 1024 --pytorch_max_t 128 --pytorch_warmup 1 --pytorch_repeat 1 --original --no-numpy --no-keep_full_grads --scan_metal --bfloat16 && echo "Success"
        # atol 0.002, rtol=1.6, with bfloat16.
        # Original wants large values.
        for name in ref_grad_stats:
            np.testing.assert_allclose(
                curr_grad_stats[name],
                ref_grad_stats[name],
                atol=2e-3,  # 3e-4,
                rtol=1.4,  # 0.2,
            )

        if keep_full_grads:
            total_ref = ref_flat_grads.sum().item()
            total_curr = curr_flat_grads.sum().item()

            torch.testing.assert_close(total_ref, total_curr, atol=1.0, rtol=0.1)

            # Optionally check if tensors are bitwise identical
            # np.testing.assert_array_equal(curr_grads.numpy(), ref_grads.numpy())

            # Original wants this at iteration 68 with command:
            # (cd rust_ewma && uv run maturin build --features python,metal --release) && uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl && PYTHONUNBUFFERED=1 METAL_DEVICE_CHECK_ERRORS=1 MT_DEVICE_CHECK_ERRORS=1 uv run --no-sync -m nanochat.bench.bench_gpt_ltv --start_t 2231 --end_t -1 --step_t -1207 --h 8 --b 2 --r 128 --n_layer 12 --warmup 4 --repeat 128 --no-verify --sequence_len 1024 --pytorch_max_t 128 --pytorch_warmup 1 --pytorch_repeat 1 --original --no-numpy --no-keep_full_grads --scan_metal --no-bfloat16 && echo "Success"
            torch.testing.assert_close(
                curr_flat_grads, ref_flat_grads, atol=0.5, rtol=0.1
            )


def empty_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def profile_training(
    model,
    input_ids,
    targets,
    optimizers,
    autocast_ctx,
    warmup: int,
    repeat: int,
    grad_accum_steps: int = 1,  # Defaulting to 1 if not provided
    grad_clip: float = 1.0,  # Matches base_train.py default
):
    model.train()

    # Helper for precision timing on MPS

    # --- Reproduce base_train.py Schedulers (CPU Overhead) ---
    def get_lr_multiplier(it):
        # Simplified version of the production scheduler
        return 1.0

    def get_muon_momentum(it):
        # Matches the logic: transition from 0.85 to 0.95
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    # --- The Core Training Step Logic ---
    def run_production_step(step_idx):
        # 1. Gradient Accumulation
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(input_ids, targets=targets, probability_distribution=True)
            # Normalizing loss as per base_train.py
            (loss / grad_accum_steps).backward()

        # 2. Gradient Clipping (The major MPS Sync Point)
        if grad_clip > 0.0:
            # This call + .item() forces a CPU-GPU synchronization
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            _ = grad_norm_tensor.item()

        # 3. Scheduler Updates
        lrm = get_lr_multiplier(step_idx)
        muon_momentum = get_muon_momentum(step_idx)
        if isinstance(optimizers, Iterable):
            adamw_optimizer, muon_optimizer = optimizers
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lrm
            for group in muon_optimizer.param_groups:
                group["momentum"] = muon_momentum
            muon_optimizer.step()
            adamw_optimizer.step()
        else:
            optimizer = optimizers
            muon_weight_decay = 0.1
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()

        # 5. Clear Gradients
        model.zero_grad(set_to_none=True)
        return loss

    # --- Warmup ---
    # Trigger torch.compile kernels and Metal command buffer caching
    for i in range(warmup):
        loss = run_production_step(i)

    synchronize()
    start = time.perf_counter()

    # --- Measured Loop ---
    for i in range(repeat):
        loss = run_production_step(warmup + i)

    synchronize()
    end = time.perf_counter()

    return loss, (end - start) / repeat


def run_model(
    model,
    optimizers,
    input_ids,
    targets,
    autocast_ctx,
    warmup: int,
    repeat: int,
    keep_full_grads: bool,
    verify_history: bool,
    name: str,
):
    # Store (scalar_loss, flat_grad_vector, layer_stats_dict)
    history = []

    total_iters = warmup + repeat
    repeat_timedelta = timedelta(seconds=0)

    # empty_cache()

    for i in range(total_iters):
        set_seed(87)

        # gc.collect()
        start_time = time.perf_counter()
        # Run the actual pass
        loss = run_pass(model, input_ids, targets, autocast_ctx)
        synchronize()
        td = timedelta(seconds=time.perf_counter() - start_time)
        logger.info(
            "Model: %s Iteration: %d [td=%s, memory=%s]",
            name,
            i,
            td,
            torch.mps.current_allocated_memory()
            if DEVICE.type == "mps"
            else torch.cuda.memory_allocated()
            if DEVICE.type == "cuda"
            else 0,
        )
        if i >= warmup:
            repeat_timedelta += td

        current_loss = loss.detach().cpu().item()
        current_flat_grads = (
            get_full_grads(model) if keep_full_grads else get_sum_grads(model)
        )
        current_grad_stats = get_grad_diagnostics(model)
        if len(history) < 2:
            history.append((current_loss, current_flat_grads, current_grad_stats))
        else:
            history[1] = (current_loss, current_flat_grads, current_grad_stats)
        if verify_history:
            check_history(history, keep_full_grads)

        # Clean up for next run
        model.zero_grad(set_to_none=True)

    duration_s = repeat_timedelta / repeat

    optimized_loss, delta_t_s = profile_training(
        model,
        input_ids,
        targets,
        optimizers,
        autocast_ctx,
        warmup=warmup,
        repeat=repeat,
    )

    logger.info("optimized_loss: %s delta_t_s: %s", optimized_loss, delta_t_s)

    return torch.tensor(history[0][0]), history[0][1], duration_s


def main():
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # Should not be needed, but whatever
    # "log" and "max"
    # dynamic=True freezes at graph compilation time.

    # dynamo.config.recompile_limit = 0

    parser = argparse.ArgumentParser(
        description="GPT LTV Benchmark"
    )
    parser.add_argument("--start_t", type=int, default=1, help="Start sequence length")
    parser.add_argument("--end_t", type=int, default=2232, help="End sequence length")
    parser.add_argument("--step_t", type=int, default=1, help="Step size for T")
    parser.add_argument(
        "--sequence_len", type=int, default=1024, help="Max sequence length for model"
    )
    parser.add_argument("--h", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--b", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--r", type=int, default=128, help="Feature dimension (per head)"
    )
    parser.add_argument("--n_layer", type=int, default=2, help="Number of layers")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip correctness checks",
    )
    parser.add_argument(
        "--verify_history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--original",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Original nanochat impl",
    )
    parser.add_argument(
        "--concat_original",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Original nanochat impl",
    )
    parser.add_argument(
        "--ltv_concat_fused_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_concat_fused_scan_triton",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_concat_fused_scan_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_concat_register_cast_scan_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_concat_register_cast_scan_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_concat_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_concat_register_cast_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_metal_fused_concat_register_cast_blelloch_scan",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_cuda_fused_concat_register_cast_blelloch_scan",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_concat_scan_triton",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pytorch_max_t",
        type=int,
        default=2232,
        help="Maximum sequence length for PyTorch implementation",
    )
    parser.add_argument(
        "--pytorch_warmup",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pytorch_repeat",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--keep_full_grads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--ltv_noop_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--ltv_fused_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_linear_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltv_fused_py_torch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deprecated_scan_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--shared_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plane_log_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plane_max_metal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plane_log_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plane_max_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--associative_scan_triton",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--noop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Autocast to bfloat16",
    )
    parser.add_argument(
        "--rust_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--filtered_identity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--autograd_identity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--combined_optimizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if args.scan_metal:
        print(f"🔧 Initializing LtvScan Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_scan_metal_context_for_default_device()
    if args.ltv_fused_scan_metal or args.ltv_concat_fused_scan_metal:
        print(f"🔧 Initializing LtvFusedScan Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
    if args.ltv_fused_concat_scan_metal:
        print(f"🔧 Initializing LtvFusedConcatScan Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_fused_concat_scan_metal_context_for_default_device()
    if args.ltv_fused_concat_register_cast_scan_metal:
        print(f"🔧 Initializing LtvFused Register Cast Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device()
    if args.ltv_metal_fused_concat_register_cast_blelloch_scan:
        print(f"🔧 Initializing LtvMetalFusedConcatRegisterCastBlellochScan Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device(
            0, 0, 0, 1_000_000
        )
    if args.ltv_fused_linear_scan_metal:
        print(f"🔧 Initializing LtvFusedLinearScan Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_fused_linear_scan_metal_context_for_default_device()
    if args.plane_log_metal or args.plane_max_metal:
        print(f"🔧 Initializing Metal Plane Context on {sys.platform}...")
        rust_ewma.init_ltv_plane_metal_context_for_default_device()
    if args.shared_metal:
        print(f"🔧 Initializing Metal Shared Context on {sys.platform}...")
        rust_ewma.init_ltv_shared_metal_context_for_default_device()
    if args.ltv_noop_metal:
        print(f"🔧 Initializing noop Metal Context on {sys.platform}...")
        rust_ewma.init_ltv_noop_metal_context_for_default_device()

    # Define Implementations
    # format: (name, function_handle)
    impls = (
        dict(
            rust_cpu=LtvRustCpu.apply,
        )
        if args.rust_cpu
        else dict()
    ) | dict(
        sorted(
            (
                (
                    dict(
                        noop=Noop.apply,
                    )
                    if args.noop
                    else dict()
                )
                | (
                    dict(
                        filtered_identity=py_identity,
                    )
                    if args.filtered_identity
                    else dict()
                )
                | (
                    dict(
                        autograd_identity=AutogradIdentity.apply,
                    )
                    if args.autograd_identity
                    else dict()
                )
                | (
                    dict(
                        ltv_noop_metal=ltv_noop_metal,
                    )
                    if args.ltv_noop_metal
                    else dict()
                )
                | (
                    dict(
                        scan_metal=ltv_scan_metal,
                    )
                    if args.scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_scan_metal=None,
                    )
                    if args.ltv_fused_scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_concat_fused_scan_metal=None,
                    )
                    if args.ltv_concat_fused_scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_concat_fused_scan_triton=None,
                    )
                    if args.ltv_concat_fused_scan_triton
                    else dict()
                )
                | (
                    dict(
                        ltv_concat_fused_scan_cuda=None,
                    )
                    if args.ltv_concat_fused_scan_cuda
                    else dict()
                )
                | (
                    dict(
                        ltv_concat_register_cast_scan_cuda=None,
                    )
                    if args.ltv_concat_register_cast_scan_cuda
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_concat_register_cast_scan_cuda=None,
                    )
                    if args.ltv_fused_concat_register_cast_scan_cuda
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_concat_register_cast_scan_metal=None,
                    )
                    if args.ltv_fused_concat_register_cast_scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_metal_fused_concat_register_cast_blelloch_scan=None,
                    )
                    if args.ltv_metal_fused_concat_register_cast_blelloch_scan
                    else dict()
                )
                | (
                    dict(
                        ltv_cuda_fused_concat_register_cast_blelloch_scan=None,
                    )
                    if args.ltv_cuda_fused_concat_register_cast_blelloch_scan
                    else dict()
                )
                | (
                    dict(
                        ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan=None,
                    )
                    if args.ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_concat_scan_metal=None,
                    )
                    if args.ltv_fused_concat_scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_concat_scan_triton=None,
                    )
                    if args.ltv_fused_concat_scan_triton
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_linear_scan_metal=None,
                    )
                    if args.ltv_fused_linear_scan_metal
                    else dict()
                )
                | (
                    dict(
                        ltv_fused_py_torch=None,
                    )
                    if args.ltv_fused_py_torch
                    else dict()
                )
                | (
                    dict(
                        deprecated_scan_metal=LtvScanMetal.apply,
                    )
                    if args.deprecated_scan_metal
                    else dict()
                )
                | (
                    dict(
                        plane_log_metal=ltv_plane_log_metal,
                    )
                    if args.plane_log_metal
                    else dict()
                )
                | (
                    dict(
                        plane_max_metal=ltv_plane_max_metal,
                    )
                    if args.plane_max_metal
                    else dict()
                )
                | (
                    dict(
                        plane_log_cuda=ltv_plane_log_cuda,
                    )
                    if args.plane_log_cuda
                    else dict()
                )
                | (
                    dict(
                        plane_max_cuda=ltv_plane_max_cuda,
                    )
                    if args.plane_max_cuda
                    else dict()
                )
                | (
                    dict(
                        associative_scan_triton=ltv_associative_scan_triton,
                    )
                    if args.associative_scan_triton
                    else dict()
                )
                | (
                    dict(
                        shared_log_metal=ltv_shared_log_metal,
                        shared_max_metal=ltv_shared_max_metal,
                        shared_pmax_metal=ltv_shared_pmax_metal,
                    )
                    if args.shared_metal
                    else dict()
                )
            ).items()
        )
    )
    logger.info("Implementations to benchmark: %s", impls.keys())

    # Filter out implementations if needed (e.g. if specific hardware is missing)

    # 2. Configure Model
    # We create a config that can handle the maximum sequence length
    vocab_size = 65536  # GPT-2 standard
    config = GPTConfig(
        vocab_size=vocab_size,
        sequence_len=args.sequence_len,  # Buffer
        n_layer=args.n_layer,
        n_head=args.h,
        n_kv_head=args.h,
        n_embd=args.h
        * args.head_dim,  # Standard convention: n_embd = n_head * head_dim
        ltv_r=args.r,
        ltv_query=LtvMode.ACTIVE,  # Assuming we want to test active LTV
        ltv_key=LtvMode.ACTIVE,
        ltv_value=LtvMode.ACTIVE,
    )

    print(
        f"🚀 Starting Benchmark: T=[{args.start_t}..{args.end_t}], B={args.b}, H={args.h}, R={args.r}, HeadDim={args.head_dim}, Layers={args.n_layer}"
    )
    print(f"{'Algo':<17} | {'T':<4} | {'Time (s)':<10} | {'Check':<6}")
    print("-" * 50)

    autocast_ctx = (
        torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16)
        if args.bfloat16
        else nullcontext()
    )

    try:
        for t in range(args.start_t, args.end_t, args.step_t):
            # At training time, the sequence length T is fixed apparently, for speedrun/runcpu to 1024.
            # So we reset the cache so that dynamo re-traces the model, with dynamic=False.
            dynamo.reset()
            # Generate Data for this T
            set_seed(1337)
            input_ids = torch.randint(0, vocab_size, (args.b, t), device=DEVICE)

            original_model = torch.compile(
                create_original_model(config, seed=42, autocast_ctx=autocast_ctx)
            )

            with autocast_ctx:
                with torch.no_grad():
                    logits = original_model(input_ids, targets=None)
                    # 1. Apply a very low temperature to "sharpen" the distribution
                    # 2. This makes the target much more certain than the model
                    # 3. Results in a probability loss of 0.9799675345420837.
                    temperature = 0.25 / math.log(max(t, 2), 2)
                    targets = torch.softmax(logits / temperature, dim=-1)

            if args.original:
                # --- Reference Run (Original) ---
                optimizers = (
                    original_model.setup_optimizer()
                    if args.combined_optimizers
                    else original_model.setup_optimizers()
                )
                orig_loss, orig_grads, orig_time = run_model(
                    original_model,
                    optimizers,
                    input_ids,
                    targets,
                    autocast_ctx,
                    args.warmup,
                    args.repeat,
                    args.keep_full_grads,
                    args.verify_history,
                    "SplitOriginal",
                )
                print(
                    f"{'SplitOriginal':<17} | {t:<4} | {orig_time.total_seconds():8.6f}   | Orig(Loss: {orig_loss.sum()}; Grad: {orig_grads.sum()})"
                )

            if args.concat_original:
                concat_original_model = torch.compile(
                    create_concat_original_model(
                        config, seed=42, autocast_ctx=autocast_ctx
                    )
                )

                optimizers = (
                    concat_original_model.setup_optimizer()
                    if args.combined_optimizers
                    else concat_original_model.setup_optimizers()
                )
                orig_loss, orig_grads, orig_time = run_model(
                    concat_original_model,
                    optimizers,
                    input_ids,
                    targets,
                    autocast_ctx,
                    args.warmup,
                    args.repeat,
                    args.keep_full_grads,
                    args.verify_history,
                    "ConcatOriginal",
                )
                print(
                    f"{'ConcatOriginal':<17} | {t:<4} | {orig_time.total_seconds():8.6f}   | Orig(Loss: {orig_loss.sum()}; Grad: {orig_grads.sum()})"
                )

            if t <= args.pytorch_max_t:
                # --- Reference Run (PyTorch) ---
                ref_model = torch.compile(
                    create_model_with_ltv(
                        config, ltv_pytorch_loop, seed=42, autocast_ctx=autocast_ctx
                    )
                )
                optimizers = (
                    ref_model.setup_optimizer()
                    if args.combined_optimizers
                    else ref_model.setup_optimizers()
                )

                ref_loss, ref_grads, ref_time = run_model(
                    ref_model,
                    optimizers,
                    input_ids,
                    targets,
                    autocast_ctx,
                    args.pytorch_warmup,
                    args.pytorch_repeat,
                    args.keep_full_grads,
                    args.verify_history,
                    "PyTorch",
                )

                print(
                    f"{'pytorch':<17} | {t:<4} | {ref_time.total_seconds():8.6f}   | LtvRef(Loss: {ref_loss.sum()}; Grad: {ref_grads.sum()})"
                )

            rust_cpu_loss, rust_cpu_grads, rust_cpu_time = None, None, None

            # --- Candidates ---
            for name, fn in impls.items():
                logger.debug("name: %s fn: %s", name, fn)
                try:
                    # Create Fresh Model
                    match name:
                        case "ltv_fused_scan_metal":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_scan_ltv,
                                )
                            )
                        case "ltv_concat_fused_scan_metal":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_concat_fused_scan_ltv,
                                )
                            )
                        case "ltv_concat_fused_scan_triton":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_concat_fused_scan_triton_ltv,
                                )
                            )
                        case "ltv_concat_fused_scan_cuda":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_concat_fused_scan_cuda_ltv,
                                )
                            )
                        case "ltv_concat_register_cast_scan_cuda":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_concat_register_cast_scan_cuda_ltv,
                                )
                            )
                        case "ltv_fused_concat_register_cast_scan_cuda":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_concat_register_cast_scan_cuda_ltv,
                                )
                            )
                        case "ltv_fused_concat_head_major_register_cast_scan_cuda":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_concat_head_major_register_cast_scan_cuda_ltv,
                                )
                            )
                        case "ltv_fused_concat_register_cast_scan_metal":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_concat_register_cast_scan_metal_ltv,
                                )
                            )
                        case "ltv_metal_fused_concat_register_cast_blelloch_scan":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_ltv_metal_fused_concat_register_cast_blelloch_scan_ltv,
                                )
                            )
                        case "ltv_cuda_fused_concat_register_cast_blelloch_scan":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_ltv_cuda_fused_concat_register_cast_blelloch_scan_ltv,
                                )
                            )
                        case "ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_ltv,
                                )
                            )
                        case "ltv_fused_concat_scan_metal":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_concat_scan_ltv,
                                )
                            )
                        case "ltv_fused_concat_scan_triton":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_concat_scan_triton_ltv,
                                )
                            )
                        case "ltv_fused_linear_scan_metal":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_linear_scan_ltv,
                                )
                            )
                        case "ltv_fused_py_torch":
                            model = torch.compile(
                                create_model_with_fused_ltv(
                                    config,
                                    seed=42,
                                    autocast_ctx=autocast_ctx,
                                    model_builder_fn=make_gpt_with_fused_py_torch_ltv,
                                )
                            )
                        case _:
                            model = torch.compile(
                                create_model_with_ltv(
                                    config, fn, seed=42, autocast_ctx=autocast_ctx
                                )
                            )
                    optimizers = (
                        model.setup_optimizer()
                        if args.combined_optimizers
                        else model.setup_optimizers()
                    )

                    is_identity = name in [
                        "identity",
                        "filtered_identity",
                        "autograd_identity",
                    ]
                    if is_identity:
                        for block in model.transformer.h:
                            # block.attn.ltv_filter.requires_grad_(False)
                            torch.nn.init.constant_(
                                block.attn.qkv_computer.ltv_filter.alpha_mlp.bias,
                                math.inf,
                            )

                    logger.debug("Run model: %s", name)
                    loss, grads, dur = run_model(
                        model,
                        optimizers,
                        input_ids,
                        targets,
                        autocast_ctx,
                        args.warmup,
                        args.repeat,
                        args.keep_full_grads,
                        args.verify_history,
                        name,
                    )
                    logger.debug("Finished running model")

                    if name == "rust_cpu":
                        rust_cpu_loss, rust_cpu_grads, rust_cpu_time = loss, grads, dur

                    # Verification
                    status = []
                    if args.verify and (args.original or not is_identity):
                        # Check 1: Loss/Logits
                        expected_loss = (
                            orig_loss
                            if is_identity
                            else (
                                ref_loss if t <= args.pytorch_max_t else rust_cpu_loss
                            )
                        )
                        if expected_loss is None or torch.allclose(
                            expected_loss,
                            loss,
                            atol=1e-3,
                            rtol=1e-4,
                        ):
                            status.append(f"✅ Loss ({expected_loss} {loss})")
                        else:
                            status.append(f"❌ Loss ({expected_loss} {loss})")

                        # Check 2: Gradients
                        expected_grads = (
                            None  # orig_grads
                            if is_identity
                            else (
                                ref_grads if t <= args.pytorch_max_t else rust_cpu_grads
                            )
                        )
                        # np.testing.assert_allclose(expected_grads.detach().cpu().numpy(), grads.detach().cpu().numpy())
                        # shared_max_metal wants such later errors for the command:
                        # (cd rust_ewma && uv run maturin build --features python,metal --release) && uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl && PYTHONUNBUFFERED=1 METAL_DEVICE_CHECK_ERRORS=1 MT_DEVICE_CHECK_ERRORS=1 uv run --no-sync -m nanochat.bench.bench_gpt_ltv --start_t 2231 --end_t -1 --step_t -1207 --h 8 --b 1 --r 128 --n_layer 12 --warmup 0 --repeat 1 --verify --sequence_len 1024 --pytorch_max_t 3072 --pytorch_warmup 0 --pytorch_repeat 1 --original --no-numpy --no-dlpack --no-keep_full_grads --no-rust_cpu --scan_metal --shared_metal --no-noop --no-ltv_noop_metal --bfloat16 --verify_history
                        if expected_grads is None or torch.allclose(
                            expected_grads,
                            grads,
                            atol=1,
                            rtol=0.01,
                        ):
                            status.append(
                                f"✅ Grad ({None if expected_grads is None else expected_grads.sum()} {grads.sum()})"
                            )
                        else:
                            status.append(
                                f"❌ Grad ({None if expected_grads is None else expected_grads.sum()} {grads.sum()})"
                            )
                            logger.debug("expected_grads: %s", expected_grads)
                            logger.debug("grads: %s", grads)

                    print(
                        f"{name:<17} | {t:<4} | {dur.total_seconds():8.6f}   | {status}"
                    )

                except Exception as e:
                    print(f"{name:<17} | {t:<4} | {'ERR':>8}   | {str(e)}")
                    raise

            print("-" * 50)
            print()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    finally:
        if args.scan_metal:
            rust_ewma.release_ltv_scan_metal_context()
        if args.ltv_fused_scan_metal or args.ltv_concat_fused_scan_metal:
            rust_ewma.release_ltv_fused_scan_metal_context()
        if args.ltv_fused_concat_scan_metal:
            rust_ewma.release_ltv_fused_concat_scan_metal_context()
        if args.ltv_fused_linear_scan_metal:
            rust_ewma.release_ltv_fused_linear_scan_metal_context()
        if args.ltv_metal_fused_concat_register_cast_blelloch_scan:
            rust_ewma.release_ltv_metal_fused_concat_register_cast_blelloch_scan_context()
        if args.ltv_noop_metal:
            rust_ewma.release_noop_metal_context()
        if args.plane_log_metal or args.plane_max_metal:
            rust_ewma.release_ltv_plane_metal_context()
        if args.shared_metal:
            rust_ewma.release_ltv_shared_metal_context()


if __name__ == "__main__":
    main()
