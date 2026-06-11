import logging
import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    should_run_cuda_ltv_in_fp32,
)
import nanochat.ops.ltv_cuda_fused_scan as _ltv_cuda_fused_scan  # Force execution of library.impl decorator

logger = logging.getLogger(__name__)

_ = _ltv_cuda_fused_scan


def resolve_inits_original_dtype(B, init_param, previous_values: LtvInitValues):
    if previous_values.q is not None:
        return torch.cat(
            [previous_values.q, previous_values.k, previous_values.v], dim=1
        )
    return init_param.unsqueeze(0).expand(B, -1, -1)


class LtvConcatRegisterCastScanCudaQkvFilter(LtvCombinedQkvFilter):
    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        super().init_weights(alpha_mlp_bias_init, weight_s, bias_s, init_s)

    def forward(self, x, previous_values: LtvInitValues):
        B, T, _ = x.size()
        head_dim = self.head_dim

        combined = self.c(x)  # keep autocast dtype (bf16/fp32)
        combined_dtype = combined.dtype

        # IMPORTANT: no .float() here
        q_raw, k_raw, v_raw, a_raw = torch.split(combined, self.split_sections, dim=-1)
        combined = None

        # Will correctly propagate gradients back from the full batch to the shared init.
        inits = resolve_inits_original_dtype(
            B, self.init_param, previous_values
        ).to(dtype=combined_dtype).contiguous()

        logits_bt_hd = a_raw.view(
            B, T, self.total_active_heads, self.num_alpha_groups
        ).contiguous()
        logits_bt_hd.add_(
            self.logit_bias.view(1, 1, self.total_active_heads, self.num_alpha_groups)
        )

        q_inp = q_raw.view(B, T, self.n_head, head_dim).contiguous()
        k_inp = k_raw.view(B, T, self.n_kv_head, head_dim).contiguous()
        v_inp = v_raw.view(B, T, self.n_kv_head, head_dim).contiguous()

        # T4 and other pre-Ampere CUDA GPUs do not support native bf16 kernels.
        run_ltv_in_fp32 = should_run_cuda_ltv_in_fp32(q_inp)
        kernel_q = q_inp.float() if run_ltv_in_fp32 else q_inp
        kernel_k = k_inp.float() if run_ltv_in_fp32 else k_inp
        kernel_v = v_inp.float() if run_ltv_in_fp32 else v_inp
        kernel_logits = logits_bt_hd.float() if run_ltv_in_fp32 else logits_bt_hd
        kernel_inits = inits.float() if run_ltv_in_fp32 else inits

        hq, hk, hv = torch.ops.nanochat_cuda.ltv_fused_scan_cuda_op(
            kernel_q,
            kernel_k,
            kernel_v,
            kernel_logits,
            kernel_inits,
            self.n_head,
            self.n_kv_head,
            self.num_alpha_groups,
            self.r,
        )
        if run_ltv_in_fp32:
            hq = hq.to(combined_dtype)
            hk = hk.to(combined_dtype)
            hv = hv.to(combined_dtype)

        q, k, v = hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)

        return q, k, v
    