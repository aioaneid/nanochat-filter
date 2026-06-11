import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv_combined_head_major_qkv_filter import LtvCombinedHeadMajorQkvFilter
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    resolve_inits_original_dtype,
    should_run_cuda_ltv_in_fp32,
)
import nanochat.ops.ltv_cuda_fused_concat_head_major_scan as _ltv_cuda_fused_concat_head_major_scan  # Force execution of library.impl decorator

logger = logging.getLogger(__name__)

_ = _ltv_cuda_fused_concat_head_major_scan


class LtvFusedConcatHeadMajorRegisterCastScanCudaQkvFilter(LtvCombinedHeadMajorQkvFilter):
    """Executes the optimized interleaved/head-major layout."""

    def forward(self, x, previous_values: LtvInitValues):
        B, _, _ = x.size()

        # Extract raw linear projection
        combined_raw = self.c(x)

        # View as 4D to expose the head dimension explicitly:
        # [B, T, total_active_heads, head_dim + num_alpha_groups]
        combined = combined_raw.view(
            B, -1, self.total_active_heads, self.head_dim + self.num_alpha_groups
        )

        # Logits are the trailing elements in the last dimension
        logits_slice = combined[:, :, :, self.head_dim :]

        # Bias is shape [total_active_heads, num_alpha_groups], broadcasts perfectly
        logits_slice.add_(self.logit_bias)

        # Flatten the head dimension back out for the CUDA kernel
        # Shape: [B, T, total_active_heads * (head_dim + num_alpha_groups)]
        kernel_combined_flattened = combined.view(
            B, -1, self.total_active_heads * (self.head_dim + self.num_alpha_groups)
        )

        inits = (
            resolve_inits_original_dtype(B, self.init_param, previous_values)
            .to(dtype=kernel_combined_flattened.dtype)
            .contiguous()
        )

        run_ltv_in_fp32 = should_run_cuda_ltv_in_fp32(kernel_combined_flattened)
        kernel_combined = (
            kernel_combined_flattened.float()
            if run_ltv_in_fp32
            else kernel_combined_flattened
        )
        kernel_inits = inits.float() if run_ltv_in_fp32 else inits

        hq, hk, hv = (
            torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_scan_cuda_op(
                kernel_combined,
                kernel_inits,
                self.n_head,
                self.n_kv_head,
                self.num_alpha_groups,
                self.r,
            )
        )
        if run_ltv_in_fp32:
            hq = hq.to(kernel_combined_flattened.dtype)
            hk = hk.to(kernel_combined_flattened.dtype)
            hv = hv.to(kernel_combined_flattened.dtype)

        q, k, v = hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
        return q, k, v
