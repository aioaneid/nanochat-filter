import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    resolve_inits_original_dtype,
    should_run_cuda_ltv_in_fp32,
)
import nanochat.ops.ltv_cuda_fused_concat_scan as _ltv_cuda_fused_concat_scan  # Force execution of library.impl decorator

logger = logging.getLogger(__name__)

_ = _ltv_cuda_fused_concat_scan


class LtvFusedConcatRegisterCastScanCudaQkvFilter(LtvCombinedQkvFilter):
    def forward(self, x, previous_values: LtvInitValues):
        B, _, _ = x.size()

        combined = self.c(x)  # [B, T, total_active_heads*(head_dim + num_alpha_groups)]

        # Split sections: q, k, v, logits
        q_size, kv_size, kv_size2, logit_size = self.split_sections
        logit_start = q_size + kv_size + kv_size2
        logits_slice = combined[..., logit_start : logit_start + logit_size]
    
        # In-place add of bias (broadcast automatically)
        logits_slice.add_(self.logit_bias)

        # Prepare inits
        inits = resolve_inits_original_dtype(
            B, self.init_param, previous_values
        ).to(dtype=combined.dtype).contiguous()

        # T4 and other pre-Ampere CUDA GPUs do not support native bf16 kernels.
        run_ltv_in_fp32 = should_run_cuda_ltv_in_fp32(combined)
        kernel_combined = combined.float() if run_ltv_in_fp32 else combined
        kernel_inits = inits.float() if run_ltv_in_fp32 else inits

        # Call fused kernel — pass combined, no separate logits
        hq, hk, hv = (
            torch.ops.nanochat_cuda.ltv_fused_concat_register_cast_scan_cuda_op(
                kernel_combined,
                kernel_inits,
                self.n_head,
                self.n_kv_head,
                self.num_alpha_groups,
                self.r,
            )
        )
        if run_ltv_in_fp32:
            hq = hq.to(combined.dtype)
            hk = hk.to(combined.dtype)
            hv = hv.to(combined.dtype)

        q, k, v = hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
        return q, k, v
