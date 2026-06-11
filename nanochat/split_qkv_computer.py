import logging
from typing import Optional
import torch
import torch.nn as nn

from nanochat.gpt_config import GPTConfig
from nanochat.ltv.ltv_filter import LtvFilterRoot, LtvInitValues
from nanochat.ltv.ltv_mode import LtvMode


logger = logging.getLogger(__name__)


class SplitQkvComputerModule(nn.Module):
    def __init__(
        self, config: GPTConfig, layer_idx: int, ltv_filter: Optional[LtvFilterRoot]
    ):
        super().__init__()
        self.head_dim = config.head_dim()
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(
            config.n_embd, config.n_kv_head * self.head_dim, bias=False
        )
        self.c_v = nn.Linear(
            config.n_embd, config.n_kv_head * self.head_dim, bias=False
        )

        self.ltv_query = config.ltv_query
        self.ltv_key = config.ltv_key
        self.ltv_value = config.ltv_value

        assert (not config.ltv_r) == (not ltv_filter)
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

    def forward(self, x, kv_cache):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if self.ltv_filter is not None:
            if kv_cache and self.has_last_qkv(kv_cache):
                previous_values = self.ltv_init(kv_cache.last_qkv)
            else:
                previous_values = LtvInitValues(None, None, None)
            q, k, v = self.ltv_filter(x, q, k, v, previous_values=previous_values)
            # Save new state to cache (the last timestep)
            if kv_cache:
                self.update_last_qkv(
                    kv_cache, q[:, -1, :, :], k[:, -1, :, :], v[:, -1, :, :]
                )
        return q, k, v

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        torch.nn.init.uniform_(self.c_q.weight, -weight_s, weight_s)
        torch.nn.init.uniform_(self.c_k.weight, -weight_s, weight_s)
        torch.nn.init.uniform_(self.c_v.weight, -weight_s, weight_s)
        if self.ltv_filter is not None:
            self.ltv_filter.init_weights(
                alpha_mlp_bias_init=alpha_mlp_bias_init,
                weight_s=weight_s,
                bias_s=bias_s,
                init_s=init_s,
            )

    def get_matrix_linear_params(self):
        # Alpha MLP weight is 2D -> Muon
        # Alpha MLP bias is 1D -> AdamW
        # Learnable Init states (vectors) -> AdamW
        matrix_params = [self.c_q.weight, self.c_k.weight, self.c_v.weight] + (
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
