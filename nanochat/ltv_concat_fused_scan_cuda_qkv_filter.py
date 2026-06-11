import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv.ltv_fused_linear_scan_filter import resolve_inits
from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter
import nanochat.ops.ltv_cuda_fused_scan as _ltv_cuda_fused_scan  # Force execution of library.impl decorator


logger = logging.getLogger(__name__)

_ = _ltv_cuda_fused_scan


class LtvConcatFusedScanCudaQkvFilter(LtvCombinedQkvFilter):
    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        super().init_weights(alpha_mlp_bias_init, weight_s, bias_s, init_s)
        
    def forward(self, x, previous_values: LtvInitValues):
        B, T, _ = x.size()
        head_dim = self.head_dim
        combined = self.c(x)
        combined_dtype = combined.dtype

        q_raw, k_raw, v_raw, a_raw = torch.split(
            combined.float(), self.split_sections, dim=-1
        )
        combined = None

        with torch.amp.autocast(device_type="cuda", enabled=False):
            inits = resolve_inits(B, self.init_param, previous_values).contiguous()
            logits_bt_hd = (
                a_raw.view(B, T, self.total_active_heads, self.num_alpha_groups)
                + self.logit_bias.view(1, 1, self.total_active_heads, self.num_alpha_groups)
            ).contiguous()

            q_inp = q_raw.view(B, T, self.n_head, head_dim).contiguous()
            k_inp = k_raw.view(B, T, self.n_kv_head, head_dim).contiguous()
            v_inp = v_raw.view(B, T, self.n_kv_head, head_dim).contiguous()

            hq, hk, hv = torch.ops.nanochat_cuda.ltv_fused_scan_cuda_op(
                q_inp,
                k_inp,
                v_inp,
                logits_bt_hd,
                inits,
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
