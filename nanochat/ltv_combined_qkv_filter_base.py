from abc import abstractmethod
import logging
import torch
import torch.nn as nn

from nanochat.ltv.ltv_filter import LtvInitValues

logger = logging.getLogger(__name__)


class LtvCombinedQkvFilterBase(nn.Module):
    """Base class containing common definitions for combined QKV filters."""

    def __init__(self, n_head: int, n_kv_head: int, head_dim: int, r: int):
        super().__init__()
        assert head_dim % r == 0, "Head dimension must be divisible by r"
        num_alpha_groups = head_dim // r
        total_active_heads = n_head + 2 * n_kv_head

        self.c = nn.Linear(
            n_head * head_dim,
            total_active_heads * (head_dim + num_alpha_groups),
            bias=False,
        )

        self.init_param = nn.Parameter(torch.empty(total_active_heads, head_dim))

        self.head_dim = head_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.num_alpha_groups = num_alpha_groups
        self.total_active_heads = total_active_heads
        self.r = r

        self.q_size = n_head * head_dim
        self.kv_size = n_kv_head * head_dim
        self.logit_size = total_active_heads * self.num_alpha_groups
        self.split_sections = [self.q_size, self.kv_size, self.kv_size, self.logit_size]

    @abstractmethod
    def forward(
        self, x, previous_values: LtvInitValues
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        pass

    def get_matrix_linear_params(self):
        matrix_params = [self.c.weight]
        linear_params = [self.logit_bias, self.init_param]
        return matrix_params, linear_params

    def get_init_parameters(self):
        return [self.init_param]

    def estimate_flops(self, sequence_len: int) -> int:
        alpha_proj_params = (
            self.total_active_heads * self.num_alpha_groups * self.c.in_features
        )
        flops = 6 * alpha_proj_params
        flops += 7 * self.total_active_heads * self.head_dim
        return flops

    def ltv_parameter_count(self) -> int:
        qkv_proj_params = self.total_active_heads * self.head_dim * self.c.in_features
        total_params = sum(p.numel() for p in self.parameters())
        return total_params - qkv_proj_params
