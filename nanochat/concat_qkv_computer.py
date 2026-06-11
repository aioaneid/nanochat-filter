import logging
from typing import Optional, Callable
import torch
import torch.nn as nn

from nanochat.gpt_config import GPTConfig
from nanochat.ltv.ltv_filter import LtvFilterRoot, LtvInitValues
from nanochat.ltv.ltv_mode import LtvMode


logger = logging.getLogger(__name__)


def default_qkv_computer(
    c: nn.Linear,
    x: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    split_sections: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Default QKV computation using standard split and view operations."""
    B, T, _ = x.size()
    combined = c(x)
    q_raw, k_raw, v_raw = torch.split(combined, split_sections, dim=-1)
    q = q_raw.view(B, T, n_head, head_dim)
    k = k_raw.view(B, T, n_kv_head, head_dim)
    v = v_raw.view(B, T, n_kv_head, head_dim)
    return q, k, v


class ConcatQkvComputerModule(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        layer_idx: int,
        ltv_filter: Optional[LtvFilterRoot],
        qkv_computer_fn: Callable[
            [
                nn.Linear,
                torch.Tensor,
                int,
                int,
                int,
                list[int],
            ],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ):
        super().__init__()
        self.head_dim = config.head_dim()
        self.c = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_kv_head) * self.head_dim,
            bias=False,
        )

        self.ltv_query = config.ltv_query
        self.ltv_key = config.ltv_key
        self.ltv_value = config.ltv_value

        self.ltv_filter = ltv_filter

        self.ltv_init = lambda last_qkv: LtvInitValues(
            q=last_qkv.q.special[layer_idx] if config.ltv_query == LtvMode.ACTIVE else None,
            k=last_qkv.k.special[layer_idx] if config.ltv_key == LtvMode.ACTIVE else None,
            v=last_qkv.v.special[layer_idx] if config.ltv_value == LtvMode.ACTIVE else None,
        )
        self.has_last_qkv = lambda kv_cache: kv_cache.has_last_qkv(layer_idx)
        self.update_last_qkv = lambda kv_cache, q, k, v: kv_cache.update_last_qkv(
            layer_idx,
            q=q if self.ltv_query else None,
            k=k if self.ltv_key else None,
            v=v if self.ltv_value else None,
        )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.q_size = config.n_head * self.head_dim
        self.kv_size = config.n_kv_head * self.head_dim
        self.split_sections = [self.q_size, self.kv_size, self.kv_size]
        self.qkv_computer_fn = qkv_computer_fn

    def forward(self, x, kv_cache):
        q, k, v = self.qkv_computer_fn(
            self.c, x, self.n_head, self.n_kv_head, self.head_dim, self.split_sections
        )

        if self.ltv_filter is not None:
            if kv_cache and self.has_last_qkv(kv_cache):
                previous_values = self.ltv_init(kv_cache.last_qkv)
            else:
                previous_values = LtvInitValues(None, None, None)
            # The fused scan filters require contiguous inputs.
            q, k, v = self.ltv_filter(
                x,
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                previous_values=previous_values,
            )
            # Save new state to cache (the last timestep)
            if kv_cache:
                self.update_last_qkv(
                    kv_cache, q[:, -1, :, :], k[:, -1, :, :], v[:, -1, :, :]
                )
        return q, k, v

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        assert not alpha_mlp_bias_init
        assert not bias_s
        assert not init_s

        q_val = torch.empty_like(self.c.weight[: self.n_head * self.head_dim, :])
        k_val = torch.empty_like(
            self.c.weight[
                self.n_head * self.head_dim : self.n_head * self.head_dim
                + self.n_kv_head * self.head_dim,
                :,
            ]
        )
        v_val = torch.empty_like(
            self.c.weight[
                self.n_head * self.head_dim
                + self.n_kv_head * self.head_dim : self.n_head * self.head_dim
                + 2 * self.n_kv_head * self.head_dim,
                :,
            ]
        )
        assert (
            self.c.weight.shape[0] == (self.n_head + 2 * self.n_kv_head) * self.head_dim
        )

        torch.nn.init.uniform_(q_val, -weight_s, weight_s)
        torch.nn.init.uniform_(k_val, -weight_s, weight_s)
        torch.nn.init.uniform_(v_val, -weight_s, weight_s)

        assert self.c.weight.requires_grad
        c_source = torch.cat([q_val, k_val, v_val], dim=0)
        with torch.no_grad():
            self.c.weight.copy_(c_source)
        assert self.c.weight.requires_grad

    def get_matrix_linear_params(self):
        # Alpha MLP weight is 2D -> Muon
        # Alpha MLP bias is 1D -> AdamW
        # Learnable Init states (vectors) -> AdamW
        matrix_params = [self.c.weight] + (
            [self.ltv_filter.alpha_mlp.weight] if self.ltv_filter else []
        )
        linear_params = (
            ([self.ltv_filter.alpha_mlp.bias] + self.ltv_filter.get_init_parameters())
            if self.ltv_filter
            else []
        )
        return matrix_params, linear_params

    def estimate_flops(self, sequence_len: int) -> int:
        if self.ltv_filter is None:
            return 0
        return self.ltv_filter.estimate_flops(sequence_len)

    def ltv_parameter_count(self) -> int:
        if self.ltv_filter is None:
            return 0
        return self.ltv_filter.ltv_parameter_count()
