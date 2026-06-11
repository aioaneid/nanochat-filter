from typing import Callable
import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv.ltv_fused_linear_scan_filter import resolve_inits
from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter


logger = logging.getLogger(__name__)


class LtvDebugConcatFusedScanQkvFilter(LtvCombinedQkvFilter):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        alpha_logit_callback: Callable[[torch.Tensor], None],
    ):
        super().__init__(n_head, n_kv_head, head_dim, r)
        self.alpha_logit_callback = alpha_logit_callback

    def forward(self, x, previous_values: LtvInitValues):
        B, T, _ = x.size()
        head_dim = self.head_dim
        combined = self.c(x)
        combined_dtype = combined.dtype

        q_raw, k_raw, v_raw, a_raw = torch.split(
            combined.float(), self.split_sections, dim=-1
        )
        combined = None
        # Now view them into their multi-head shapes
        # These views remain contiguous in their last dimension
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            alpha_logit = (
                a_raw.view(
                    B, T, self.total_active_heads, self.num_alpha_groups
                ).contiguous()
                + self.logit_bias.view(
                    1, 1, self.total_active_heads, self.num_alpha_groups
                )
            ).contiguous()
            self.alpha_logit_callback(alpha_logit)

            hq, hk, hv = torch.ops.nanochat.ltv_fused_scan_metal_op(
                q_raw.view(B, T, self.n_head, head_dim).contiguous(),
                k_raw.view(B, T, self.n_kv_head, head_dim).contiguous(),
                v_raw.view(B, T, self.n_kv_head, head_dim).contiguous(),
                alpha_logit,
                resolve_inits(B, self.init_param, previous_values),
                self.n_head,
                self.n_kv_head,
                self.num_alpha_groups,
                self.r,
            )
        q, k, v = (
            hq.transpose(1, 2).to(dtype=combined_dtype),
            hk.transpose(1, 2).to(dtype=combined_dtype),
            hv.transpose(1, 2).to(dtype=combined_dtype),
        )

        return q, k, v
