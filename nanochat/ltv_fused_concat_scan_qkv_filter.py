import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv.ltv_fused_linear_scan_filter import resolve_inits
from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter


logger = logging.getLogger(__name__)


class LtvFusedConcatScanQkvFilter(LtvCombinedQkvFilter):
    def forward(self, x, previous_values: LtvInitValues):
        B, _, _ = x.size()
        # combined is [B, T, OutDim]
        combined = self.c(x)
        combined_dtype = combined.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # No split(), no contiguous() on parts.
            # Cast combined to float once.
            combined_f32 = combined.float()
            combined = None
            logit_bias_f32 = self.logit_bias.float()
            inits_f32 = resolve_inits(B, self.init_param, previous_values)

            hq, hk, hv = torch.ops.nanochat.ltv_fused_concat_scan_metal_op(
                combined_f32,
                logit_bias_f32,
                inits_f32,
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
