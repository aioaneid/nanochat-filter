import logging
import torch
import torch.nn as nn

from nanochat.ltv.ltv_fused_scan_filter import init_weights_of_init
from nanochat.ltv_combined_qkv_filter_base import LtvCombinedQkvFilterBase

logger = logging.getLogger(__name__)


class LtvCombinedQkvFilter(LtvCombinedQkvFilterBase):
    """Legacy layout: [All Qs | All Ks | All Vs | All Logits]"""

    def __init__(self, n_head: int, n_kv_head: int, head_dim: int, r: int):
        super().__init__(n_head, n_kv_head, head_dim, r)
        # 1D bias matching the concatenated logits at the end
        self.logit_bias = nn.Parameter(
            torch.empty(self.total_active_heads * self.num_alpha_groups)
        )

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
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
                + self.n_kv_head * self.head_dim : self.total_active_heads
                * self.head_dim,
                :,
            ]
        )

        torch.nn.init.uniform_(q_val, -weight_s, weight_s)
        torch.nn.init.uniform_(k_val, -weight_s, weight_s)
        torch.nn.init.uniform_(v_val, -weight_s, weight_s)

        alpha_mlp_weight = torch.empty_like(
            self.c.weight[self.total_active_heads * self.head_dim :, :]
        )

        torch.nn.init.uniform_(alpha_mlp_weight, -weight_s, weight_s)

        assert self.c.weight.requires_grad
        c_source = torch.cat([q_val, k_val, v_val, alpha_mlp_weight], dim=0)
        with torch.no_grad():
            self.c.weight.copy_(c_source)
        assert self.c.weight.requires_grad

        torch.nn.init.uniform_(
            self.logit_bias,
            alpha_mlp_bias_init - bias_s,
            alpha_mlp_bias_init + bias_s,
        )
        init_weights_of_init(self.init_param, self.n_head, self.n_kv_head, init_s)
