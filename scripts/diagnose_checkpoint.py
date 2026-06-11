"""
Diagnose a base_train checkpoint with detailed logit statistics.

Usage:

python -m scripts.diagnose_checkpoint \
    --checkpoint_step 8000 \
    --model_tag d20 \
    --device_type cuda \
    --max_tokens 16
"""

from nanochat.utils.concat_fused_formatter import print_effective_memory_table
from nanochat.utils.concat_fused_formatter import compute_effective_memory
from nanochat.utils.concat_fused_formatter import print_alpha_logit_tables
from nanochat.utils.concat_fused_formatter import split_alpha_logit
import os
import argparse
from enum_actions import enum_action
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from distutils.util import strtobool

from nanochat.common import (
    compute_init,
    compute_cleanup,
    get_base_dir,
    autodetect_device_type,
)
from nanochat.ltv_debug_concat_fused_scan_qkv_filter import (
    LtvDebugConcatFusedScanQkvFilter,
)
from nanochat.gpt_config import GPTConfig
from nanochat.checkpoint_manager import existing_checkpoints, load_checkpoint
from nanochat.kv_cache import KVCache
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine
from nanochat.gpt_factory import (
    make_ltv_combined_model_with_filter_factory,
    one_time_model_factory,
)
from nanochat.gpt_factory import LtvSplitMode, ModelType
from nanochat.utils.load_model import remap_attention_keys

try:
    import rust_ewma
except ImportError:
    rust_ewma = None


def diagnose_ltv_usage(model):
    print("\n" + "#" * 80)
    print("LTV USAGE DIAGNOSTICS")
    print("#" * 80)

    for name, module in model.named_modules():
        if not hasattr(module, "logit_bias"):
            continue

        print("\n" + "-" * 80)
        print(f"LTV module: {name}")
        print("-" * 80)

        with torch.no_grad():
            bias = module.logit_bias.detach().float()
            init = module.init_param.detach().float()

            total_h = module.total_active_heads
            num_alpha_groups = module.num_alpha_groups
            r = module.r
            head_dim = module.head_dim
            n_head = module.n_head
            n_kv_head = module.n_kv_head

            # reshape bias: (total_h, num_alpha_groups)
            bias = bias.view(total_h, num_alpha_groups)
            alpha = torch.sigmoid(bias)

            print(f"n_head={n_head} n_kv_head={n_kv_head} head_dim={head_dim} r={r}")
            print(f"total_active_heads={total_h}")
            print(f"num_alpha_groups={num_alpha_groups}")

            # Global stats
            print("\n[ALPHA GLOBAL]")
            print(f"mean(alpha): {alpha.mean().item():.6f}")
            print(f"std(alpha) : {alpha.std(correction=0).item():.6f}")
            print(f"min(alpha) : {alpha.min().item():.6f}")
            print(f"max(alpha) : {alpha.max().item():.6f}")

            # Saturation
            frac_small = (alpha < 0.1).float().mean().item()
            frac_large = (alpha > 0.9).float().mean().item()

            print(f"alpha < 0.1 : {frac_small:.4f}")
            print(f"alpha > 0.9 : {frac_large:.4f}")

            # Effective memory length approximation
            # Roughly E[length] ≈ 1/alpha for small alpha
            mem_length = 1.0 / alpha.clamp(min=1e-6)
            print("\n[MEMORY LENGTH ~ 1/alpha]")
            print(f"mean: {mem_length.mean().item():.2f}")
            print(f"median: {mem_length.median().item():.2f}")
            print(f"max: {mem_length.max().item():.2f}")

            # Q / K / V separation
            q_alpha = alpha[:n_head]
            k_alpha = alpha[n_head : n_head + n_kv_head]
            v_alpha = alpha[n_head + n_kv_head :]

            def block_stats(name, a):
                print(f"\n[{name}]")
                print(f"mean(alpha): {a.mean().item():.6f}")
                print(f"std(alpha) : {a.std(correction=0).item():.6f}")
                print(f"min(alpha) : {a.min().item():.6f}")
                print(f"max(alpha) : {a.max().item():.6f}")
                print(f"alpha<0.1  : {(a < 0.1).float().mean().item():.4f}")
                print(f"alpha>0.9  : {(a > 0.9).float().mean().item():.4f}")

            block_stats("Q", q_alpha)
            block_stats("K", k_alpha)
            block_stats("V", v_alpha)

            # Per-alpha-group variation
            print("\n[PER-ALPHA-GROUP VARIATION]")
            group_std = alpha.std(dim=0, correction=0)
            print(f"group std mean: {group_std.mean().item():.6f}")
            print(f"group std max : {group_std.max().item():.6f}")

            # Init magnitude
            print("\n[INIT PARAM STATS]")
            print(f"mean(init): {init.mean().item():.6f}")
            print(f"std(init) : {init.std(correction=0).item():.6f}")
            print(f"L2(init)  : {init.norm().item():.6f}")

            # Compare init scale vs bias magnitude
            print("\n[BIAS LOGIT STATS]")
            print(f"mean(bias): {bias.mean().item():.6f}")
            print(f"std(bias) : {bias.std(correction=0).item():.6f}")
            print(f"min(bias) : {bias.min().item():.6f}")
            print(f"max(bias) : {bias.max().item():.6f}")


class PrintAlphaLogitCallback:
    def __init__(self, alpha_tables: bool, effective_memory: bool):
        self.alpha_tables = alpha_tables
        self.effective_memory = effective_memory

    def __call__(self, alpha_logit: torch.Tensor):
        if self.alpha_tables or self.effective_memory:
            alpha_q, alpha_k, alpha_v = split_alpha_logit(
                alpha_logit, n_head=self.n_head, n_kv_head=self.n_kv_head
            )
            if self.alpha_tables:
                print_alpha_logit_tables(alpha_q, alpha_k, alpha_v)
            if self.effective_memory:
                effective_memory = compute_effective_memory(alpha_q, alpha_k, alpha_v)
                print_effective_memory_table(effective_memory)


def make_debug_ltv_concat_fused_scan_model(
    config: GPTConfig, alpha_tables: bool, effective_memory: bool
):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvDebugConcatFusedScanQkvFilter(
            config.n_head,
            config.n_kv_head,
            config.head_dim(),
            config.ltv_r,
            PrintAlphaLogitCallback(
                alpha_tables=alpha_tables, effective_memory=effective_memory
            ),
        ),
    )


def main():
    # -----------------------------------------------------------------------------
    # CLI

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_step", type=int, required=False)
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--device_type", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument(
        "--remap_attention_keys", type=lambda v: bool(strtobool(v)), default=False
    )
    parser.add_argument(
        "--debug_model", type=lambda v: bool(strtobool(v)), default=False
    )
    parser.add_argument(
        "--alpha_tables", type=lambda v: bool(strtobool(v)), default=True
    )
    parser.add_argument(
        "--effective_memory", type=lambda v: bool(strtobool(v)), default=True
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Device + autocast (identical to base_train)

    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    _, ddp_rank, _, _, device = compute_init(device_type)

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type in ["cuda", "mps"]
        else nullcontext()
    )

    # -----------------------------------------------------------------------------
    # Tokenizer (IDENTICAL to base_train)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # -----------------------------------------------------------------------------
    # Load checkpoint metadata

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)

    if args.checkpoint_step is None:
        steps = existing_checkpoints(checkpoint_dir)
        args.checkpoint_step = max(steps)

    print(args)

    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir,
        args.checkpoint_step,
        device,
        load_optimizer=False,
        rank=ddp_rank,
    )

    model_config_kwargs = meta_data["model_config"]
    user_config = meta_data["user_config"]

    print("Loaded checkpoint metadata:")
    print(model_config_kwargs)

    # -----------------------------------------------------------------------------
    # Recreate model EXACTLY like base_train

    model_type = (
        ModelType[s] if (s := user_config.get("model_type")) else ModelType.ORIGINAL
    )
    match args.debug_model, model_type:
        case True, ModelType.LTV_CONCAT_FUSED_SCAN:
            rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
            model_factory = lambda config: make_debug_ltv_concat_fused_scan_model(
                config, args.alpha_tables, args.effective_memory
            )
        case _, _:
            model_factory = one_time_model_factory(
                model_type,
                LtvSplitMode[s]
                if (s := user_config.get("ltv_query"))
                else LtvSplitMode.NONE,
                LtvSplitMode[s]
                if (s := user_config.get("ltv_key"))
                else LtvSplitMode.NONE,
                LtvSplitMode[s]
                if (s := user_config.get("ltv_value"))
                else LtvSplitMode.NONE,
            )

    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = model_factory(model_config)

    model.to_empty(device=device)

    model.init_weights(
        alpha_mlp_bias_init=user_config.get("alpha_mlp_bias_init", 0),
        rand_filter_bias=user_config.get("rand_filter_bias", 0),
        rand_filter_init=user_config.get("rand_filter_init", 0),
    )

    if args.remap_attention_keys:
        new_state_dict = remap_attention_keys(model_data)
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(model_data, strict=True, assign=True)

    orig_model = model
    orig_model.eval()

    print(f"Number of parameters: {sum(p.numel() for p in orig_model.parameters()):,}")

    diagnose_ltv_usage(model)

    # -----------------------------------------------------------------------------
    # Diagnostics helpers

    def entropy_from_logits(logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum().item()

    def print_logit_stats(step, logits):
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, 10)

        print(f"\n--- Step {step} ---")
        print(f"logits mean: {logits.mean().item():.6f}")
        print(f"logits std : {logits.std(correction=0).item():.6f}")
        print(f"logits min : {logits.min().item():.6f}")
        print(f"logits max : {logits.max().item():.6f}")
        print(f"logits L2  : {logits.norm().item():.6f}")
        print(f"entropy    : {entropy_from_logits(logits):.6f}")

        print("Top 10 tokens:")
        for i in range(10):
            tok = topk_idx[i].item()
            p = topk_vals[i].item()
            decoded = tokenizer.decode([tok])
            print(f"  id={tok:4d}  p={p:.6f}  repr={repr(decoded)}")

    # -----------------------------------------------------------------------------
    # Prompts (identical to base_train sampling)

    prompts = [
        # "If John is taller than Mark and Mark is taller than Steve, and Steve is taller than Alex, then John is taller than",
        "The capital of France is",
        # "The chemical symbol of gold is",
        # "If yesterday was Friday, then tomorrow will be",
        # "The opposite of hot is",
        # "The planets of the solar system are:",
        # "My favorite color is",
        # "If 5*x + 3 = 13, then x is",
        # "Alice gave a book to Maria. "
        # "Several unrelated events happened in between. "
        # "After a long discussion about astronomy and cooking recipes, "
        # "the person who received the book was"
    ]

    engine = Engine(orig_model, tokenizer)

    # -----------------------------------------------------------------------------
    # Global weight statistics

    with torch.no_grad():
        weights = torch.cat(
            [p.detach().flatten().float() for p in orig_model.parameters()]
        )
        print("\nGlobal parameter statistics:")
        print(f"mean : {weights.mean().item():.6f}")
        print(f"std  : {weights.std(correction=0).item():.6f}")
        print(f"min  : {weights.min().item():.6f}")
        print(f"max  : {weights.max().item():.6f}")
        print(f"L2   : {weights.norm().item():.6f}")

    for name, p in orig_model.named_parameters():
        print(name, p.std(correction=0).item())

    # -----------------------------------------------------------------------------
    # Per-prompt deep diagnostics

    for prompt in prompts:
        print("\n" + "=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        tokens = tokenizer(prompt, prepend="<|bos|>")

        with autocast_ctx:
            samples, masks = engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
        print("Samples:", samples)
        print("Masks:", masks)
        decoded = tokenizer.decode(samples[0])
        print("\nGenerated (argmax):")
        print(decoded)
        continue

        # Now do manual forward step-by-step for logit inspection
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        kv_model_kwargs = {
            "num_q_heads": model_config.n_head,
            "num_kv_heads": model_config.n_kv_head,
            "head_dim": model_config.head_dim(),
            "num_layers": model_config.n_layer,
            "layer_specs": model_config.layer_specs,
        }
        print("kv_model_kwargs:", kv_model_kwargs)
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        assert kv_cache_prefill.last_qkv is None
        if not args.max_tokens:
            return

        with torch.no_grad():
            with autocast_ctx:
                logits = orig_model(input_ids, kv_cache=kv_cache_prefill)

        last_logits = logits[0, -1].float()
        print_logit_stats(0, last_logits)

        kv_length_hint = len(tokens) + args.max_tokens
        kv_cache_decode = KVCache(
            batch_size=1,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        for step in range(1, args.max_tokens):
            next_token = torch.argmax(F.softmax(last_logits, dim=-1)).item()
            input_ids = torch.tensor([[next_token]], device=device)

            with torch.no_grad():
                with autocast_ctx:
                    logits = orig_model(input_ids, kv_cache=kv_cache_decode)

            last_logits = logits[0, -1].float()
            print_logit_stats(step, last_logits)

    # -----------------------------------------------------------------------------
    compute_cleanup()


if __name__ == "__main__":
    main()
