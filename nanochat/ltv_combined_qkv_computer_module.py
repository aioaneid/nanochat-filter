import logging
from typing import Optional
import torch.nn as nn

from nanochat.kv_cache import KVCache
from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv_combined_qkv_filter_base import LtvCombinedQkvFilterBase

logger = logging.getLogger(__name__)


class LtvCombinedQkvComputerModule(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        ltv_filter: LtvCombinedQkvFilterBase,  # Type hinted to Base
    ):
        super().__init__()
        self.ltv_filter = ltv_filter
        self.has_last_qkv = lambda kv_cache: kv_cache.has_last_qkv(layer_idx)
        self.ltv_init = lambda last_qkv: LtvInitValues(
            q=last_qkv.q.special[layer_idx], k=last_qkv.k.special[layer_idx], v=last_qkv.v.special[layer_idx]
        )
        self.update_last_qkv = lambda kv_cache, q, k, v: kv_cache.update_last_qkv(
            layer_idx, q=q, k=k, v=v
        )

    def forward(self, x, kv_cache: Optional[KVCache]):
        if kv_cache and self.has_last_qkv(kv_cache):
            previous_values = self.ltv_init(kv_cache.last_qkv)
        else:
            previous_values = LtvInitValues(None, None, None)

        q, k, v = self.ltv_filter(x, previous_values)

        if kv_cache:
            self.update_last_qkv(
                kv_cache, q[:, -1, :, :], k[:, -1, :, :], v[:, -1, :, :]
            )

        return q, k, v

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        self.ltv_filter.init_weights(
            alpha_mlp_bias_init=alpha_mlp_bias_init,
            weight_s=weight_s,
            bias_s=bias_s,
            init_s=init_s,
        )

    def get_matrix_linear_params(self):
        return self.ltv_filter.get_matrix_linear_params()

    def estimate_flops(self, sequence_len: int) -> int:
        return self.ltv_filter.estimate_flops(sequence_len)

    def ltv_parameter_count(self) -> int:
        return self.ltv_filter.ltv_parameter_count()
