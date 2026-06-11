"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

from nanochat.ltv_combined_qkv_computer_module import LtvCombinedQkvComputerModule
from nanochat.ltv_fused_concat_register_cast_scan_metal_qkv_filter import (
    LtvFusedConcatRegisterCastScanMetalQkvFilter,
)
from nanochat.ltv_cuda_fused_concat_register_cast_blelloch_scan_qkv_filter import (
    LtvCudaFusedConcatRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter import (
    LtvCudaFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_qkv_filter import (
    LtvMetalFusedConcatRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_concat_register_cast_scan_cuda_qkv_filter import (
    LtvConcatRegisterCastScanCudaQkvFilter,
)
from nanochat.gpt_config import GPTConfig
from nanochat.kv_cache import KVCache
from nanochat.ltv.ltv_fused_linear_scan_filter import LtvFusedLinearScanFilter
from nanochat.ltv.ltv_fused_scan_filter import LtvFusedScanFilter, LtvFusedPyTorchFilter
from nanochat.ltv.ltv_mode import LtvMode
from nanochat.ltv.ltv_filter import (
    LtvFilter,
    LtvSplitFilter,
    QkvLtv,
    QkvSplitLtv,
)
from nanochat.ltv_concat_fused_scan_qkv_filter import LtvConcatFusedScanQkvFilter
from nanochat.ltv_concat_fused_scan_cuda_qkv_filter import (
    LtvConcatFusedScanCudaQkvFilter,
)
from nanochat.ltv_concat_fused_scan_triton_qkv_filter import (
    LtvConcatFusedScanTritonQkvFilter,
)
import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from arrgh import arrgh as orig_arrgh

from nanochat.common import get_dist_info, print0
from nanochat.ltv_fused_concat_register_cast_scan_cuda_qkv_filter import (
    LtvFusedConcatRegisterCastScanCudaQkvFilter,
)
from nanochat.ltv_fused_concat_head_major_register_cast_scan_cuda_qkv_filter import (
    LtvFusedConcatHeadMajorRegisterCastScanCudaQkvFilter,
)
from nanochat.ltv_fused_concat_scan_qkv_filter import LtvFusedConcatScanQkvFilter
from nanochat.ltv_fused_concat_scan_triton_qkv_filter import (
    LtvFusedConcatScanTritonQkvFilter,
)
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.split_qkv_computer import SplitQkvComputerModule

logger = logging.getLogger(__name__)


def arrgh(*args, **kwargs):
    if False:
        orig_arrgh(*args, **kwargs)


def count_trainable_parameters(module: nn.Module) -> int:
    """
    Recursively count all trainable parameters in the module.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ==============================================================================
# CONFIG & UTILS
# ==============================================================================


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    if (head_dim := x.shape[3]) & 1 != 0:
        return x
    d = head_dim // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@torch._dynamo.disable
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, is_causal=False, enable_gqa=False
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


def make_ltv_filter(
    config: GPTConfig,
    ltv_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> LtvFilter:
    assert config.ltv_r
    return LtvFilter(
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim(),
        r=config.ltv_r,
        qkv_ltv=QkvLtv(
            query=config.ltv_query, key=config.ltv_key, value=config.ltv_value
        ),
        ltv_fn=ltv_fn,
    )


def make_ltv_fused_scan_filter(config: GPTConfig) -> LtvFusedScanFilter:
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE
    assert config.ltv_key == LtvMode.ACTIVE
    assert config.ltv_value == LtvMode.ACTIVE
    return LtvFusedScanFilter(
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim(),
        r=config.ltv_r,
    )


def make_ltv_fused_linear_scan_filter(config: GPTConfig) -> LtvFusedLinearScanFilter:
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE
    assert config.ltv_key == LtvMode.ACTIVE
    assert config.ltv_value == LtvMode.ACTIVE
    return LtvFusedLinearScanFilter(
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim(),
        r=config.ltv_r,
    )


def make_ltv_fused_py_torch_filter(config: GPTConfig) -> LtvFusedPyTorchFilter:
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE
    assert config.ltv_key == LtvMode.ACTIVE
    assert config.ltv_value == LtvMode.ACTIVE
    return LtvFusedPyTorchFilter(
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim(),
        r=config.ltv_r,
    )


def make_split_ltv_filter(config: GPTConfig, qkv_split_ltv: QkvSplitLtv):
    assert config.ltv_r
    return LtvSplitFilter(
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim(),
        r=config.ltv_r,
        qkv_split_ltv=qkv_split_ltv,
    )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        layer_idx: int,
        qkv_computer: Callable[
            [torch.Tensor, Optional[KVCache]],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.kv_insert = lambda kv_cache, k, v: kv_cache.insert_kv(layer_idx, k, v)

        # Necessary to make it work also with torch.compile(dynamic=True) on Metal.
        self.attention_scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_computer = qkv_computer

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q, k, v = self.qkv_computer(x, kv_cache)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = (
            apply_rotary_emb(q, cos, sin),
            apply_rotary_emb(k, cos, sin),
        )  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = self.kv_insert(kv_cache, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(
            2
        )  # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        assert k.size(-1) == self.head_dim
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.attention_scale,
                enable_gqa=enable_gqa,
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.attention_scale,
                enable_gqa=enable_gqa,
            )
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=self.attention_scale,
                enable_gqa=enable_gqa,
            )

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, attn: CausalSelfAttention):
        super().__init__()
        self.attn = attn
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # logger.info("forward a %s", x)
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


def uniform_normal_std_equalizer(n_embd):
    return (
        3**0.5 * n_embd**-0.5
    )  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal


DEFAULT_ROTARY_BASE = 10_000


def precompute_rotary_embeddings(seq_len, head_dim, base, device):
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
    cos, sin = (
        cos[None, :, None, :],
        sin[None, :, None, :],
    )  # add batch and head dims for later broadcasting
    return cos, sin


class GPT(nn.Module):
    def __init__(
        self,
        config,
        attns: list[CausalSelfAttention],
        pad_vocab_size_to=64,
    ):
        super().__init__()
        self.config = config
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}"
            )

        # Handle LTV functions - ensure we have one per layer or None for all
        if len(attns) != config.n_layer:
            raise ValueError(
                f"attns must have length {config.n_layer} (one per layer), got {len(attns)}"
            )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, attn) for attn in attns]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.max_sequence_len()
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(
        self, alpha_mlp_bias_init: float, rand_filter_bias: bool, rand_filter_init: bool
    ):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = uniform_normal_std_equalizer(n_embd)
        num_ltv_layers = self.config.num_ltv_layers()
        for layer_idx, block in enumerate(self.transformer.h):
            block.attn.qkv_computer.init_weights(
                alpha_mlp_bias_init=alpha_mlp_bias_init if layer_idx < num_ltv_layers else 0,
                weight_s=s,
                bias_s=s * rand_filter_bias,
                init_s=s * rand_filter_init if layer_idx < num_ltv_layers else 0,
            )
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=DEFAULT_ROTARY_BASE, device=None
    ):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        return precompute_rotary_embeddings(
            seq_len=seq_len,
            head_dim=head_dim,
            base=base,
            device=device,
        )

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        nparams_ltv = sum(
            (
                block.attn.qkv_computer.ltv_parameter_count()
                for block in self.transformer.h
                if hasattr(block.attn.qkv_computer, "ltv_parameter_count")
            ),
            start=0,
        )
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        flops_ltv_per_token = sum(
            (
                block.attn.qkv_computer.estimate_flops(t)
                for block in self.transformer.h
                if hasattr(block.attn.qkv_computer, "estimate_flops")
            ),
            start=0,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding - nparams_ltv) + 12 * l * h * q * t + flops_ltv_per_token
        return num_flops_per_token

    def _trainable_parameters(self):
        matrix_params = []
        adam_params_other = []  # LM head and other scalar/vector params
        # Main Embedding
        adam_params_embedding = list(self.transformer.wte.parameters())
        # LM Head
        adam_params_other.extend(list(self.lm_head.parameters()))
        # Iterate over blocks
        for block in self.transformer.h:
            # LTV Parameters
            qkv_computer_matrix_params, qkv_computer_linear_params = (
                block.attn.qkv_computer.get_matrix_linear_params()
            )
            matrix_params.extend(qkv_computer_matrix_params)
            adam_params_other.extend(qkv_computer_linear_params)
            # Standard Linear weights -> Muon
            matrix_params.extend(
                [
                    block.attn.c_proj.weight,
                    block.mlp.c_fc.weight,
                    block.mlp.c_proj.weight,
                ]
            )
        # Verify all parameters are covered
        # Note: We aren't doing the strict assert check here because we are building lists manually,
        # effectively excluding any non-grad params implicitly.
        # But for correctness in this snippet context:
        all_trainable = [
            (name, param)
            for name, param in self.named_parameters()
            if param.requires_grad
        ]
        all_collected = matrix_params + adam_params_embedding + adam_params_other
        for p in all_collected:
            assert p.requires_grad, f"Parameter {p} is not trainable but was collected"
        assert len(all_trainable) == len(all_collected), (
            f"Parameter count mismatch: found {len(all_collected)}, expected {len(all_trainable)}; all_trainable: {[name for name, _ in all_trainable]}"
        )
        return matrix_params, adam_params_embedding, adam_params_other

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params, adam_params_embedding, adam_params_other = (
            self._trainable_parameters()
        )

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind="adamw",
                params=adam_params_other,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=adam_params_embedding,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Identify parameters by type for proper optimizer grouping
        # Muon: 2D matrix weights (Linear layers)
        # AdamW: Embeddings, Biases, 1D vectors (RMSNorm if learnable, or LTV inits)

        matrix_params, adam_params_embedding, adam_params_other = (
            self._trainable_parameters()
        )

        # Create the AdamW optimizer for the embedding and lm_head/biases
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        adam_groups = [
            dict(params=adam_params_other, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=adam_params_embedding, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        probability_distribution=False,
    ):
        B, T = idx.size()
        # logger.debug("Forwarding %d tokens", T)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(
            x
        )  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            if probability_distribution:
                # 1. Prepare the distributions
                flat_targets = targets.view(
                    -1, targets.size(-1)
                )  # These are probabilities
                log_probs = F.log_softmax(
                    flat_logits, dim=-1
                )  # These are log-probabilities

                # 2. Compute KL Divergence
                # reduction='batchmean' is the standard for KL Div to get a proper mean
                # We use 'none' first to handle your custom loss_reduction logic
                loss = F.kl_div(log_probs, flat_targets, reduction="none").sum(dim=-1)

                if loss_reduction == "mean":
                    loss = loss.mean()
                elif loss_reduction == "sum":
                    loss = loss.sum()
            else:
                # training: given the targets, compute and return the loss
                # TODO experiment with chunked cross-entropy?
                loss = F.cross_entropy(
                    flat_logits,
                    targets.view(-1),
                    ignore_index=-1,
                    reduction=loss_reduction,
                )
            return loss
        else:
            assert not probability_distribution
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


def make_gpt_with_ltv_fn(
    config: GPTConfig,
    ltv_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                SplitQkvComputerModule(
                    config, layer_idx, make_ltv_filter(config, ltv_fn)
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_gpt_with_fused_ltv(config: GPTConfig, ltv_fused_filter_fn):
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_key == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_value == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                SplitQkvComputerModule(
                    config,
                    layer_idx,
                    ltv_fused_filter_fn(config),
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_gpt_with_fused_scan_ltv(config: GPTConfig):
    return make_gpt_with_fused_ltv(
        config,
        make_ltv_fused_scan_filter,
    )


def make_gpt_with_fused_py_torch_ltv(config: GPTConfig):
    return make_gpt_with_fused_ltv(
        config,
        make_ltv_fused_py_torch_filter,
    )


def make_gpt_with_fused_linear_scan_ltv(config: GPTConfig):
    return make_gpt_with_fused_ltv(
        config,
        make_ltv_fused_linear_scan_filter,
    )


def make_gpt_with_concat_fused_scan_ltv(config: GPTConfig):
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_key == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_value == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvCombinedQkvComputerModule(
                    layer_idx=layer_idx,
                    ltv_filter=LtvConcatFusedScanQkvFilter(
                        config.n_head,
                        config.n_kv_head,
                        head_dim=config.head_dim(),
                        r=config.ltv_r,
                    ),
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def _make_gpt_with_ltv_filter_factory(config: GPTConfig, ltv_filter_factory):
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_key == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_value == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvCombinedQkvComputerModule(
                    layer_idx=layer_idx,
                    ltv_filter=ltv_filter_factory(
                        config.n_head,
                        config.n_kv_head,
                        head_dim=config.head_dim(),
                        r=config.ltv_r,
                    ),
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_gpt_with_concat_fused_scan_triton_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(config, LtvConcatFusedScanTritonQkvFilter)


def make_gpt_with_concat_fused_scan_cuda_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(config, LtvConcatFusedScanCudaQkvFilter)


def make_gpt_with_concat_register_cast_scan_cuda_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvConcatRegisterCastScanCudaQkvFilter
    )


def make_gpt_with_fused_concat_register_cast_scan_cuda_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvFusedConcatRegisterCastScanCudaQkvFilter
    )


def make_gpt_with_fused_concat_head_major_register_cast_scan_cuda_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvFusedConcatHeadMajorRegisterCastScanCudaQkvFilter
    )


def make_gpt_with_fused_concat_register_cast_scan_metal_ltv(config: GPTConfig):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvFusedConcatRegisterCastScanMetalQkvFilter
    )


def make_gpt_with_ltv_metal_fused_concat_register_cast_blelloch_scan_ltv(
    config: GPTConfig,
):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvMetalFusedConcatRegisterCastBlellochScanQkvFilter
    )


def make_gpt_with_ltv_cuda_fused_concat_register_cast_blelloch_scan_ltv(
    config: GPTConfig,
):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvCudaFusedConcatRegisterCastBlellochScanQkvFilter
    )


def make_gpt_with_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_ltv(
    config: GPTConfig,
):
    return _make_gpt_with_ltv_filter_factory(
        config, LtvCudaFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter
    )


def make_gpt_with_fused_concat_scan_ltv(config: GPTConfig):
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_key == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_value == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvCombinedQkvComputerModule(
                    layer_idx=layer_idx,
                    ltv_filter=LtvFusedConcatScanQkvFilter(
                        config.n_head,
                        config.n_kv_head,
                        head_dim=config.head_dim(),
                        r=config.ltv_r,
                    ),
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_gpt_with_fused_concat_scan_triton_ltv(config: GPTConfig):
    assert config.ltv_r
    assert config.ltv_query == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_key == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    assert config.ltv_value == LtvMode.ACTIVE, (
        "Fused LTV currently only supports all active"
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvCombinedQkvComputerModule(
                    layer_idx=layer_idx,
                    ltv_filter=LtvFusedConcatScanTritonQkvFilter(
                        config.n_head,
                        config.n_kv_head,
                        head_dim=config.head_dim(),
                        r=config.ltv_r,
                    ),
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )
