from abc import ABC, abstractmethod
import torch

from nanochat.gpt_config import ChunksAndHistory
from nanochat.ltv_combined_head_major_qkv_filter import LtvCombinedHeadMajorQkvFilter
from nanochat.ltv_fused_concat_register_cast_scan_util import should_run_ltv_in_fp32


UNBOUNDED_P = 2_147_483_647


class LtvLookBackHeadMajorQkvFilter(LtvCombinedHeadMajorQkvFilter, ABC):
    """
    Shared base for look-back QKV filters that use a combined (q,k,v) projection
    followed by a recurrent scan.
    """

    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        layer_spec: ChunksAndHistory,
    ):
        super().__init__(n_head, n_kv_head, head_dim, r)
        self.layer_spec = layer_spec

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input x (B,T,d_model) to the combined tensor.
        Must return a tensor of shape (B,T,total_features) or the layout
        required by _look_back_op.
        """
        ...

    @abstractmethod
    def _look_back_op(
        self,
        combined: torch.Tensor,
        inits: torch.Tensor,
        logit_bias: torch.Tensor,
        full_recurrence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute the recurrent scan and return hq, hk, hv of shape (B,H,T,D).
        """
        ...

    # ------------------------------------------------------------------
    # Shared forward logic
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        resolved_inits: torch.Tensor,
        full_recurrence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        run_ltv_in_fp32 = should_run_ltv_in_fp32(x.device.type)

        # Project with optional fp32 upcast
        x_proj = x.float() if run_ltv_in_fp32 else x
        combined = self._project(x_proj)

        # Prepare bias and inits
        logit_bias = self.logit_bias.contiguous()
        inits = resolved_inits.to(dtype=combined.dtype).contiguous()

        # Run the scan
        hq, hk, hv = self._look_back_op(
            combined,
            inits.float() if run_ltv_in_fp32 else inits,
            logit_bias.float() if run_ltv_in_fp32 else logit_bias,
            full_recurrence,
        )

        if run_ltv_in_fp32:
            hq = hq.to(combined.dtype)
            hk = hk.to(combined.dtype)
            hv = hv.to(combined.dtype)

        # hq, hk, hv are already (B,T,H,D) after the op – no transpose needed
        return hq, hk, hv
