import torch

from nanochat.ltv_combined_head_major_qkv_filter import LtvCombinedHeadMajorQkvFilter
from nanochat.ops.ltv_cuda_ops import (
    ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan,
    fused_linear_transpose,
)
from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    CUDA_SUPPORTS_NATIVE_BF16,
    resolve_inits_original_dtype,
    _SCAN_THREADS_FORWARD,
    _R_THREADS_FORWARD,
    _R_ITEMS_FORWARD,
    _SCAN_THREADS_BACKWARD,
    _R_THREADS_BACKWARD,
    _R_ITEMS_BACKWARD,
)


class LtvCudaFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter(
    LtvCombinedHeadMajorQkvFilter
):
    def forward(self, x, previous_values: LtvInitValues):
        B, _, _ = x.size()

        autocast_dtype = torch.get_autocast_dtype("cuda")
        run_ltv_in_fp32 = (
            autocast_dtype == torch.float32 if CUDA_SUPPORTS_NATIVE_BF16 else True
        )

        # Call the functional op which allocates the physical buffer in the
        # forward operator implementation (ensures consistent allocation
        # strategy and centralizes stride/contiguity policy).
        out_buf = fused_linear_transpose(
            x.float() if run_ltv_in_fp32 else x,
            self.c.weight.float() if run_ltv_in_fp32 else self.c.weight,
        )

        # 3. Create the (B, T, F) view with target strides (F*T, 1, T)
        kernel_combined = out_buf.transpose(1, 2)

        inits = (
            resolve_inits_original_dtype(B, self.init_param, previous_values)
            .to(dtype=kernel_combined.dtype)
            .contiguous()
        )

        kernel_inits = inits.float() if run_ltv_in_fp32 else inits

        hq, hk, hv = ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan(
            kernel_combined,
            kernel_inits,
            self.logit_bias.contiguous(),
            self.n_head,
            self.n_kv_head,
            self.num_alpha_groups,
            self.r,
            _SCAN_THREADS_FORWARD,
            _R_THREADS_FORWARD,
            _R_ITEMS_FORWARD,
            _SCAN_THREADS_BACKWARD,
            _R_THREADS_BACKWARD,
            _R_ITEMS_BACKWARD,
            True,  # use_sigmoid
        )
        if run_ltv_in_fp32:
            hq = hq.to(kernel_combined.dtype)
            hk = hk.to(kernel_combined.dtype)
            hv = hv.to(kernel_combined.dtype)

        q, k, v = hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
        return q, k, v
