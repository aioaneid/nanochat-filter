import logging
import torch
import torch.nn as nn

from nanochat.ltv.ltv_fused_scan_filter import init_weights_of_init
from nanochat.ltv_combined_qkv_filter_base import LtvCombinedQkvFilterBase

logger = logging.getLogger(__name__)


class LtvCombinedHeadMajorQkvFilter(LtvCombinedQkvFilterBase):
    """Optimized layout: [Head 0 Values | Head 0 Logits | Head 1 Values | Head 1 Logits | ...]"""

    def __init__(self, n_head: int, n_kv_head: int, head_dim: int, r: int):
        super().__init__(n_head, n_kv_head, head_dim, r)
        # 2D bias matching the interleaved layout: [total_heads, num_alpha_groups]
        self.logit_bias = nn.Parameter(
            torch.empty(self.total_active_heads, self.num_alpha_groups)
        )

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        # Determine the sizes of Q, K, V blocks
        q_size = self.n_head * self.head_dim
        k_size = self.n_kv_head * self.head_dim
        v_size = self.n_kv_head * self.head_dim

        # USE new_empty TO INHERIT DTYPE AND DEVICE
        q_val = self.c.weight.new_empty(q_size, self.c.in_features)
        k_val = self.c.weight.new_empty(k_size, self.c.in_features)
        v_val = self.c.weight.new_empty(v_size, self.c.in_features)

        torch.nn.init.uniform_(q_val, -weight_s, weight_s)
        torch.nn.init.uniform_(k_val, -weight_s, weight_s)
        torch.nn.init.uniform_(v_val, -weight_s, weight_s)

        alpha_mlp_weight = self.c.weight.new_empty(
            self.total_active_heads * self.num_alpha_groups, self.c.in_features
        )
        torch.nn.init.uniform_(alpha_mlp_weight, -weight_s, weight_s)

        # Reshape to group by head
        val_weights = torch.cat([q_val, k_val, v_val], dim=0).view(
            self.total_active_heads, self.head_dim, -1
        )
        alpha_weights = alpha_mlp_weight.view(
            self.total_active_heads, self.num_alpha_groups, -1
        )

        # Interleave along the inner dimension and flatten back for the linear layer
        c_source = torch.cat([val_weights, alpha_weights], dim=1).view(
            -1, self.c.in_features
        )

        assert self.c.weight.requires_grad
        with torch.no_grad():
            self.c.weight.copy_(c_source)
        assert self.c.weight.requires_grad

        torch.nn.init.uniform_(
            self.logit_bias,
            alpha_mlp_bias_init - bias_s,
            alpha_mlp_bias_init + bias_s,
        )
        init_weights_of_init(self.init_param, self.n_head, self.n_kv_head, init_s)
