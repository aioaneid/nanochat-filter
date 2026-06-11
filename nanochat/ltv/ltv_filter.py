from abc import abstractmethod
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
import torch
import logging
import torch.nn as nn

from nanochat.ltv.ltv_mode import LtvMode

logger = logging.getLogger(__name__)


@dataclass
class QkvLtv:
    query: LtvMode
    key: LtvMode
    value: LtvMode


@dataclass
class QkvSplitLtv:
    query: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    key: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    value: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]


def qkv_ltv_from_qkv_split_ltv(qkv_split_ltv: QkvSplitLtv):
    return QkvLtv(
        query=LtvMode.NONE if qkv_split_ltv.query is None else LtvMode.ACTIVE,
        key=LtvMode.NONE if qkv_split_ltv.key is None else LtvMode.ACTIVE,
        value=LtvMode.NONE if qkv_split_ltv.value is None else LtvMode.ACTIVE,
    )


@dataclass
class LtvInitValues:
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None


# Helper to select between passed state or learnable parameter
def _get_init(passed_val, B, param, n_active):
    if passed_val is None:
        # logger.info("param: %s %s %s", param.shape, param.dtype, param.device)
        # param is (n_active, D). expand to (B, n_active, D)
        # Since we are inside the ACTIVE check, n_active == n_h
        return param[:n_active, :].unsqueeze(0).expand(B, -1, -1)
    else:
        # passed_val should be (B, H, D)
        return passed_val[:, :n_active, :]


class LtvFilterRoot(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        qkv_ltv: QkvLtv,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        assert head_dim % r == 0, (
            "head_dim must be divisible by r, which represent the alpha multiplicity"
        )
        self.head_dim = head_dim
        assert self.n_head % self.n_kv_head == 0

        self.n_embd = self.n_head * self.head_dim

        # Calculate total active heads
        # These are used to size the output splits and the init parameters
        self.n_active_q = self.n_head * int(qkv_ltv.query == LtvMode.ACTIVE)
        self.n_active_k = self.n_kv_head * int(qkv_ltv.key == LtvMode.ACTIVE)
        self.n_active_v = self.n_kv_head * int(qkv_ltv.value == LtvMode.ACTIVE)
        self.total_active_heads = self.n_active_q + self.n_active_k + self.n_active_v

        self.r = r
        self.num_alpha_groups = head_dim // r

        # Single linear layer to produce alpha per head (B, T, TotalHeads).
        self.alpha_mlp = nn.Linear(
            self.n_embd,
            self.total_active_heads * self.num_alpha_groups,
            bias=True,
            # Not sure if this is useful, ever.
            dtype=torch.float32,
        )

    def _init_some_weights(
        self, alpha_mlp_bias_init, weight_s: float, bias_s: float):
        torch.nn.init.uniform_(
            self.alpha_mlp.weight, -weight_s, weight_s
        )
        # logger.info("self.alpha_mlp.weight.data: %s", self.alpha_mlp.weight.data)
        torch.nn.init.uniform_(
            self.alpha_mlp.bias,
            alpha_mlp_bias_init - bias_s,
            alpha_mlp_bias_init + bias_s,
        )

    @abstractmethod
    def init_weights(
        self, alpha_mlp_bias_init, weight_s: float, bias_s: float, init_s: float
    ):
        pass

    @abstractmethod
    def get_init_parameters(self) -> list[torch.nn.Parameter]:
        pass

    def estimate_flops(self, sequence_len: int) -> int:
        alpha_proj_params = self.alpha_mlp.weight.numel()
        flops = 6 * alpha_proj_params
        flops += 7 * self.total_active_heads * self.head_dim
        return flops

    def ltv_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_embd)
            q: (B, T, n_head, head_dim)
            k: (B, T, n_kv_head, head_dim)
            v: (B, T, n_kv_head, head_dim)
            previous_values: Struct containing (B, H, D) tensors for q/k/v.
        Returns:
            q_out, k_out, v_out
        """
        pass


class LtvFilterBase(LtvFilterRoot):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        qkv_ltv: QkvLtv,
    ):
        super().__init__(n_head, n_kv_head, head_dim, r, qkv_ltv)

        # Initial values for q, k, v as learnable parameters.
        # Sized by n_active_* so if a mode is NONE, we get a 0-size parameter (unused).
        self.q_init = nn.Parameter(torch.empty(self.n_active_q, head_dim))
        self.k_init = nn.Parameter(torch.empty(self.n_active_k, head_dim))
        self.v_init = nn.Parameter(torch.empty(self.n_active_v, head_dim))

    def init_weights(
        self, alpha_mlp_bias_init, weight_s: float, bias_s: float, init_s: float
    ):
        super()._init_some_weights(alpha_mlp_bias_init, weight_s, bias_s)

        torch.nn.init.uniform_(self.q_init, -init_s, init_s)
        torch.nn.init.uniform_(self.k_init, -init_s, init_s)
        torch.nn.init.uniform_(self.v_init, -init_s, init_s)

    def get_init_parameters(self) -> list[torch.nn.Parameter]:
        return [self.q_init, self.k_init, self.v_init]

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_embd)
            q: (B, T, n_head, head_dim)
            k: (B, T, n_kv_head, head_dim)
            v: (B, T, n_kv_head, head_dim)
            previous_values: Struct containing (B, H, D) tensors for q/k/v.
        Returns:
            q_out, k_out, v_out
        """
        pass


class LtvFilter(LtvFilterBase):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        qkv_ltv: QkvLtv,
        ltv_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__(n_head, n_kv_head, head_dim, r, qkv_ltv)
        self.ltv_fn = ltv_fn

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Capture input dtype for final output cast
        (head_dim,) = set([q.shape[3], k.shape[3], v.shape[3]])
        assert head_dim == self.head_dim

        ((B, T),) = set([x.shape[:2], q.shape[:2], k.shape[:2], v.shape[:2]])

        # assert self.q_init.dtype == self.k_init.dtype == self.v_init.dtype

        init_tensors = [
            _get_init(previous_values.q, B, self.q_init, self.n_active_q),
            _get_init(previous_values.k, B, self.k_init, self.n_active_k),
            _get_init(previous_values.v, B, self.v_init, self.n_active_v),
        ]

        # Concatenate and Flatten
        previous_state = torch.cat(init_tensors, dim=1)
        previous_state_flat = previous_state.view(B, -1)

        # --- Prepare Inputs for Scan ---
        # We perform concatenation in the original dtype (BF16) to save memory bandwidth
        concatenated_qkv = torch.cat(
            [
                q[:, :, : self.n_active_q, :],
                k[:, :, : self.n_active_k, :],
                v[:, :, : self.n_active_v, :],
            ],
            dim=2,
        )  # (B, T, H_total, D)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            alpha_logits = self.alpha_mlp(x.float())  # (B, T, H_total * G)
            alpha_val = alpha_logits.sigmoid()
            beta_val = 1.0 - alpha_val
            alpha_val = alpha_val.view(
                B, T, self.total_active_heads, self.num_alpha_groups
            )
            alpha_val = alpha_val.repeat_interleave(self.r, dim=3)  # (B, T, H_total, D)

            concatenated_alpha_times_qkv = alpha_val * concatenated_qkv

            concatenated_alpha_times_qkv_flat = concatenated_alpha_times_qkv.flatten(
                start_dim=2
            )

            # No need to pass r because it will be inferred from the shapes.
            ltv_result_flat = self.ltv_fn(
                previous_state_flat.float(),
                beta_val.float(),
                concatenated_alpha_times_qkv_flat.float(),
            )

            # --- Reshape and Split Output ---
            ltv_result = ltv_result_flat.unflatten(
                dim=2, sizes=(self.total_active_heads, head_dim)
            )

            q_ltv, k_ltv, v_ltv = torch.split(
                ltv_result,
                [
                    self.n_active_q,
                    self.n_active_k,
                    self.n_active_v,
                ],
                dim=2,
            )

            return (
                q_ltv.to(dtype=q.dtype) if self.n_active_q else q,
                k_ltv.to(dtype=k.dtype) if self.n_active_k else k,
                v_ltv.to(dtype=v.dtype) if self.n_active_v else v,
            )


class LtvSplitFilter(LtvFilterBase):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        r: int,
        qkv_split_ltv: QkvSplitLtv,
    ):
        super().__init__(
            n_head,
            n_kv_head,
            head_dim,
            r,
            qkv_ltv_from_qkv_split_ltv(qkv_split_ltv),
        )
        self.qkv_split_ltv = qkv_split_ltv

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        previous_values: LtvInitValues,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Capture input dtype for final output cast
        (head_dim,) = set([q.shape[3], k.shape[3], v.shape[3]])
        assert head_dim == self.head_dim

        ((B, T),) = set([x.shape[:2], q.shape[:2], k.shape[:2], v.shape[:2]])

        # assert self.q_init.dtype == self.k_init.dtype == self.v_init.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            alpha_logits = self.alpha_mlp(x.float())
            alpha_val = alpha_logits.sigmoid()
            alpha_q, alpha_k, alpha_v = torch.split(
                alpha_val,
                [
                    self.n_active_q * self.num_alpha_groups,
                    self.n_active_k * self.num_alpha_groups,
                    self.n_active_v * self.num_alpha_groups,
                ],
                dim=2,
            )
            qkv_split_ltv = self.qkv_split_ltv
            if fn := qkv_split_ltv.query:
                q = (
                    fn(
                        _get_init(previous_values.q, B, self.q_init, self.n_active_q)
                        .view(B, -1)
                        .float(),
                        (1.0 - alpha_q).float(),
                        (
                            alpha_q.view(
                                B, T, self.n_active_q, self.num_alpha_groups
                            ).repeat_interleave(self.r, dim=3)
                            * q
                        )
                        .flatten(start_dim=2)
                        .float(),
                    )
                    .unflatten(dim=2, sizes=q.shape[2:])
                    .to(dtype=q.dtype)
                )
            if fn := qkv_split_ltv.key:
                k = (
                    fn(
                        _get_init(previous_values.k, B, self.k_init, self.n_active_k)
                        .view(B, -1)
                        .float(),
                        (1.0 - alpha_k).float(),
                        (
                            alpha_k.view(
                                B, T, self.n_active_k, self.num_alpha_groups
                            ).repeat_interleave(self.r, dim=3)
                            * k
                        )
                        .flatten(start_dim=2)
                        .float(),
                    )
                    .unflatten(dim=2, sizes=k.shape[2:])
                    .to(dtype=k.dtype)
                )
            if fn := qkv_split_ltv.value:
                v = (
                    fn(
                        _get_init(previous_values.v, B, self.v_init, self.n_active_v)
                        .view(B, -1)
                        .float(),
                        (1.0 - alpha_v).float(),
                        (
                            alpha_v.view(
                                B, T, self.n_active_v, self.num_alpha_groups
                            ).repeat_interleave(self.r, dim=3)
                            * v
                        )
                        .flatten(start_dim=2)
                        .float(),
                    )
                    .unflatten(dim=2, sizes=v.shape[2:])
                    .to(dtype=v.dtype)
                )

        return q, k, v
