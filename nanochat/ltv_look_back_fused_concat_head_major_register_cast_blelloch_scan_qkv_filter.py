from collections.abc import Callable

import torch

from nanochat.gpt_config import ChunksAndHistory
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    _SCAN_THREADS_FORWARD,
    _R_THREADS_FORWARD,
    _R_ITEMS_FORWARD,
    _SCAN_THREADS_BACKWARD,
    _R_THREADS_BACKWARD,
    _R_ITEMS_BACKWARD,
)
from nanochat.ltv_look_back_head_major_qkv_filter_base import (
    LtvLookBackHeadMajorQkvFilter,
    UNBOUNDED_P,
)


TransposeOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

LookBackOp = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        bool,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class LtvLookBackFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter(
    LtvLookBackHeadMajorQkvFilter
):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        layer_spec: ChunksAndHistory,
        transpose_op: TransposeOp,
        look_back_op: LookBackOp,
    ):
        super().__init__(n_head, n_kv_head, head_dim, r, layer_spec)
        self.transpose_op = transpose_op
        self.look_back_op = look_back_op

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # The original fused linear + transpose
        out_buf = self.transpose_op(
            x.float() if x.dtype == torch.float32 else x,
            self.c.weight.float() if x.dtype == torch.float32 else self.c.weight,
        )
        # Create (B, F, T) view with strides (F*T, T, 1)
        return out_buf.transpose(1, 2)

    def _look_back_op(
        self,
        combined: torch.Tensor,
        inits: torch.Tensor,
        logit_bias: torch.Tensor,
        full_recurrence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hq, hk, hv = self.look_back_op(
            combined,
            inits,
            logit_bias,
            self.n_head,
            self.n_kv_head,
            self.num_alpha_groups,
            self.r,
            UNBOUNDED_P if full_recurrence else self.layer_spec.p,
            1 if full_recurrence else self.layer_spec.q,
            1 if full_recurrence else self.layer_spec.k,
            _SCAN_THREADS_FORWARD,
            _R_THREADS_FORWARD,
            _R_ITEMS_FORWARD,
            _SCAN_THREADS_BACKWARD,
            _R_THREADS_BACKWARD,
            _R_ITEMS_BACKWARD,
            True,  # use_sigmoid
        )
        # The Blelloch op returns (B,H,T,D); transpose to (B,T,H,D)
        return hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
