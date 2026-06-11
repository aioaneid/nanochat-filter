from typing import Tuple
import torch
import torch.nn as nn
import logging

from nanochat.ltv.ltv_mode import LtvMode
from nanochat.ltv.ltv_filter import LtvFilterRoot, QkvLtv, LtvInitValues

# Needed to avoid AttributeError
import nanochat.ops

logger = logging.getLogger(__name__)


# Identical initialization like LtvFilter, instead of
# simply `torch.nn.init.uniform_(self.init_param, -init_s, init_s)`.
def init_weights_of_init(init_param, n_head, n_kv_head, init_s):
    q_val = torch.empty_like(init_param[: n_head, :])
    k_val = torch.empty_like(
        init_param[n_head : n_head + n_kv_head, :]
    )
    v_val = torch.empty_like(
        init_param[
            n_head + n_kv_head : n_head + 2 * n_kv_head, :
        ]
    )
    assert init_param.shape[0] == n_head + 2 * n_kv_head, (
        "init_param size mismatch"
    )

    torch.nn.init.uniform_(q_val, -init_s, init_s)
    torch.nn.init.uniform_(k_val, -init_s, init_s)
    torch.nn.init.uniform_(v_val, -init_s, init_s)

    assert init_param.requires_grad
    init_param_source = torch.cat([q_val, k_val, v_val], dim=0)
    with torch.no_grad():
        init_param.copy_(init_param_source)
    assert init_param.requires_grad



class LtvFusedFilterRoot(LtvFilterRoot):
    """
    Common logic for filters that use a unified init parameter and a single
    Alpha MLP to process Q, K, and V as a single block.
    """

    def __init__(
        self,
        n_head,
        n_kv_head,
        head_dim,
        r,
    ):
        super().__init__(
            n_head,
            n_kv_head,
            head_dim,
            r,
            QkvLtv(query=LtvMode.ACTIVE, key=LtvMode.ACTIVE, value=LtvMode.ACTIVE),
        )

        # Unified learnable state for all active heads
        self.init_param = nn.Parameter(torch.empty(self.total_active_heads, head_dim))

    def init_weights(self, alpha_mlp_bias_init, weight_s, bias_s, init_s):
        super()._init_some_weights(alpha_mlp_bias_init, weight_s, bias_s)
        init_weights_of_init(self.init_param, self.n_head, self.n_kv_head, init_s)

    def get_init_parameters(self) -> list[torch.nn.Parameter]:
        return [self.init_param]


class LtvFusedFilterBase(LtvFusedFilterRoot):
    def _resolve_inits_and_logits(self, x, previous_values: LtvInitValues):
        B, T, _ = x.shape

        # 1. Resolve Initial States
        if previous_values.q is not None:
            # User provided state (B, H, D)
            inits = torch.cat(
                [previous_values.q, previous_values.k, previous_values.v], dim=1
            ).float()
        else:
            # Use learnable parameter (TotalH, D) expanded to (B, TotalH, D)
            inits = self.init_param.unsqueeze(0).expand(B, -1, -1).float().contiguous()

        # 2. Compute Gating Logits (B, T, TotalH, G)
        # We perform this here so it's consistent across both implementations
        logits = self.alpha_mlp(x.float()).view(
            B, T, self.total_active_heads, self.num_alpha_groups
        )

        return inits, logits


class LtvFusedScanFilter(LtvFusedFilterBase):
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Enforce the fused constraint
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

        inits, logits = self._resolve_inits_and_logits(x, previous_values)

        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert inits.is_contiguous()
        assert logits.is_contiguous()

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # Pass to Metal/Rust Op
            hq, hk, hv = torch.ops.nanochat.ltv_fused_scan_metal_op(
                q.float(),
                k.float(),
                v.float(),
                logits.float(),
                inits,
                self.n_head,
                self.n_kv_head,
                self.num_alpha_groups,
                self.r,
            )

        return (
            hq.transpose(1, 2).to(dtype=q.dtype),
            hk.transpose(1, 2).to(dtype=k.dtype),
            hv.transpose(1, 2).to(dtype=v.dtype),
        )


class LtvFusedPyTorchFilter(LtvFusedFilterBase):
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        D = self.head_dim

        inits, logits = self._resolve_inits_and_logits(x, previous_values)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # reference gating logic
            alpha = logits.sigmoid()
            # Use repeat_interleave for semantic and stride safety
            alpha_expanded = alpha.repeat_interleave(self.r, dim=3)

            # Prepare flattened inputs for the loop
            u = torch.cat([q, k, v], dim=2).float()

            h_prev = inits.reshape(B, -1)
            alpha_flat = alpha_expanded.reshape(B, T, -1)
            x_flat = u.reshape(B, T, -1)

            h_all = []
            for t in range(T):
                # h_t = (1 - alpha) * h_prev + alpha * u
                h_t = (1.0 - alpha_flat[:, t]) * h_prev + alpha_flat[:, t] * x_flat[
                    :, t
                ]
                h_all.append(h_t)
                h_prev = h_t

            result = torch.stack(h_all, dim=1).view(B, T, self.total_active_heads, D)

            q_out, k_out, v_out = torch.split(
                result, [self.n_active_q, self.n_active_k, self.n_active_v], dim=2
            )

        return (
            q_out.to(dtype=q.dtype),
            k_out.to(dtype=k.dtype),
            v_out.to(dtype=v.dtype),
        )
