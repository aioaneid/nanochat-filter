from typing import Tuple
import torch
import logging

from nanochat.ltv.ltv_fused_scan_filter import LtvFusedFilterRoot
from nanochat.ltv.ltv_filter import LtvInitValues

# Needed to avoid AttributeError
import nanochat.ops

logger = logging.getLogger(__name__)


def resolve_inits(B, init_param, previous_values: LtvInitValues):
    if previous_values.q is not None:
        return torch.cat(
            [previous_values.q, previous_values.k, previous_values.v], dim=1
        ).contiguous().float()
    return init_param.unsqueeze(0).expand(B, -1, -1).contiguous().float()


class LtvFusedLinearScanFilter(LtvFusedFilterRoot):
    """
    Highly optimized Metal version.
    Fuses MLP (x * W + b) directly into the recursive scan kernel.
    """

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.n_active_q and self.n_active_k and self.n_active_v

        # Input is already (B, T, H, D) - verify strides are contiguous
        B, T, NH, D = q.shape
        _, _, NKVH, _ = k.shape  # Get actual NKVH from the tensor

        # Calculate expected contiguous strides for each
        q_stride = (T * NH * D, NH * D, D, 1)
        kv_stride = (T * NKVH * D, NKVH * D, D, 1)

        assert q.stride() == q_stride, f"Q stride mismatch: {q.stride()} vs {q_stride}"
        assert k.stride() == kv_stride, (
            f"K stride mismatch: {k.stride()} vs {kv_stride}"
        )
        assert v.stride() == kv_stride, (
            f"V stride mismatch: {v.stride()} vs {kv_stride}"
        )

        # Metal kernels require contiguous memory for reliable indexing
        inits = resolve_inits(B, self.init_param, previous_values)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # Pass RAW gate input and weights to the kernel
            hq, hk, hv = torch.ops.nanochat.ltv_fused_linear_scan_metal_op(
                q.float().contiguous(),
                k.float().contiguous(),
                v.float().contiguous(),
                x.float().contiguous(),  # x_gate
                self.alpha_mlp.weight,  # w_gate
                self.alpha_mlp.bias,  # b_gate
                inits,
                self.num_alpha_groups,
                self.r,
            )

        # Transpose from (B, H, T, D) back to (B, T, H, D)
        return (
            hq.transpose(1, 2).to(dtype=q.dtype),
            hk.transpose(1, 2).to(dtype=k.dtype),
            hv.transpose(1, 2).to(dtype=v.dtype),
        )
