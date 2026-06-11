from typing import Callable

import torch

from nanochat.gpt_config import ChunksAndHistory
from nanochat.ltv_fused_concat_register_cast_scan_util import (
    _R_THREADS_FORWARD,
    _R_ITEMS_FORWARD,
    _R_THREADS_BACKWARD,
    _R_ITEMS_BACKWARD,
)
from nanochat.ltv_look_back_head_major_qkv_filter_base import (
    LtvLookBackHeadMajorQkvFilter,
    UNBOUNDED_P,
)

# Type for the external sequential op. There is no scan-thread tuning for this
# path; only r-thread/r-item tuning is meaningful.
SequentialLookBackOp = Callable[
    [
        torch.Tensor,  # combined (B,T,F)
        torch.Tensor,  # inits (B,total_h,D)
        torch.Tensor,  # logit_bias (total_h,Da)
        int,
        int,
        int,
        int,  # NH, NKVH, Da, r
        int,
        int,
        int,  # P, Q, K
        int,
        int,  # r_threads_forward, r_items_forward
        int,
        int,  # r_threads_backward, r_items_backward
        bool,  # use_sigmoid
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class LtvLookBackFusedConcatHeadMajorRegisterCastSequentialScanQkvFilter(
    LtvLookBackHeadMajorQkvFilter
):
    """
    Sequential-scan variant using a plain nn.Linear projection.
    The combined tensor has the natural (B,T,F) layout with strides (T*F, F, 1).
    """

    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        layer_spec: ChunksAndHistory,
        sequential_look_back_op: SequentialLookBackOp,
    ):
        super().__init__(n_head, n_kv_head, head_dim, r, layer_spec)
        self.sequential_look_back_op = sequential_look_back_op

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # Natural (B,T,F) contiguous output
        return self.c(x).contiguous()

    def _look_back_op(
        self,
        combined: torch.Tensor,
        inits: torch.Tensor,
        logit_bias: torch.Tensor,
        full_recurrence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hq, hk, hv = self.sequential_look_back_op(
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
            _R_THREADS_FORWARD,
            _R_ITEMS_FORWARD,
            _R_THREADS_BACKWARD,
            _R_ITEMS_BACKWARD,
            True,  # use_sigmoid
        )
        # The sequential op returns (B,H,T,D); transpose to (B,T,H,D).
        return hq.transpose(1, 2), hk.transpose(1, 2), hv.transpose(1, 2)
