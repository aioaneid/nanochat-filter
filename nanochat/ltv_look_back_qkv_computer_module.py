import collections
import logging
from typing import Optional
import torch
import torch.nn as nn

from nanochat.kv_cache import KVCache, TypeDevice, make_empty_last_qkv_like_specials
from nanochat.ltv_look_back_head_major_qkv_filter_base import (
    LtvLookBackHeadMajorQkvFilter,
)

logger = logging.getLogger(__name__)


def advance_kv_cache(
    x: torch.Tensor,
    layer_idx: int,
    ltv_filter: LtvLookBackHeadMajorQkvFilter,
    resolved_inits: torch.Tensor,
    kv_cache: KVCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (self_last_qkv := kv_cache.last_qkv) is None:
        assert not layer_idx
        type_device = TypeDevice(x.dtype, x.device)
        self_last_qkv = kv_cache.last_qkv = make_empty_last_qkv_like_specials(
            kv_cache.last_q_shape,
            kv_cache.last_k_v_shape,
            type_device,
            type_device,
            type_device,
            mls=[
                chunks_and_history.deque_size()
                for chunks_and_history in kv_cache.layer_specs
            ],
        )
    else:
        assert self_last_qkv.q
        assert self_last_qkv.k
        assert self_last_qkv.v

    layer_spec = kv_cache.layer_specs[layer_idx]

    # Split inits along the HEAD dimension (dim=1), NOT the feature dimension.
    # resolved_inits has shape (B, total_h, D) where total_h = NH + 2*NKVH.
    q0, k0, v0 = torch.split(
        resolved_inits,
        [ltv_filter.n_head, ltv_filter.n_kv_head, ltv_filter.n_kv_head],
        dim=1,
    )

    # countdown/x0 initialization for this layer, if not already initialized
    for projection, x0 in zip(
        [self_last_qkv.q, self_last_qkv.k, self_last_qkv.v],
        [q0, k0, v0],
    ):
        layer = projection.layers[layer_idx]
        if not layer.is_initialized():
            projection.special[layer_idx, :, :, :] = x0
            layer.countdown = layer_spec.p or layer_spec.q
            layer.d = (
                collections.deque(maxlen=0)
                if layer_spec.is_unbounded()
                else collections.deque(
                    [x0] * (ml := layer_spec.deque_size()), maxlen=ml
                )
            )
            assert layer.is_initialized()

    n = x.shape[1]
    start = 0
    result = []
    q_layer = self_last_qkv.q.layers[layer_idx]
    k_layer = self_last_qkv.k.layers[layer_idx]
    v_layer = self_last_qkv.v.layers[layer_idx]
    while start < n:
        # Q, K, V have the same spec, so we just mirror K and V after Q
        al = min(q_layer.countdown, n - start)

        # Build resolved_inits by concatenating the current specials along the HEAD dimension.
        resolved_inits_cat = torch.cat(
            [
                self_last_qkv.q.special[layer_idx, :, :, :],
                self_last_qkv.k.special[layer_idx, :, :, :],
                self_last_qkv.v.special[layer_idx, :, :, :],
            ],
            dim=1,
        )

        a_q, a_k, a_v = ltv_filter.forward(
            x=x[:, start : start + al, :],
            resolved_inits=resolved_inits_cat,
            full_recurrence=True,
        )
        result.append((a_q, a_k, a_v))

        (countdown,) = {q_layer.countdown, k_layer.countdown, v_layer.countdown}
        for i, (x_dq, y_dk, z_dv) in enumerate(
            zip(q_layer.d, k_layer.d, v_layer.d, strict=True)
        ):
            # Offset adjusts for the fact that d[i] resolves i boundaries in the future
            offset_i = q_layer.countdown + i * layer_spec.q - layer_spec.k + 1
            bl = max(offset_i, 0)
            kt = x[:, start + bl : start + al, :]

            # Only run the forward accumulation if there are tokens overlapping the window
            if kt.shape[1]:
                # Build resolved_inits from the deque states (each has shape (B, H, D)).
                resolved_inits_i = torch.cat([x_dq, y_dk, z_dv], dim=1)
                q, k, v = ltv_filter.forward(
                    x=kt,
                    resolved_inits=resolved_inits_i,
                    full_recurrence=True,
                )
                q_layer.d[i], k_layer.d[i], v_layer.d[i] = (
                    q[:, -1, :, :],
                    k[:, -1, :, :],
                    v[:, -1, :, :],
                )

        if al == q_layer.countdown:
            new_q = q_layer.d[0] if q_layer.d else q0
            new_k = k_layer.d[0] if k_layer.d else k0
            new_v = v_layer.d[0] if v_layer.d else v0
            q_layer.d.append(q0)
            k_layer.d.append(k0)
            v_layer.d.append(v0)
            self_last_qkv.q.special[layer_idx, :, :, :] = new_q
            self_last_qkv.k.special[layer_idx, :, :, :] = new_k
            self_last_qkv.v.special[layer_idx, :, :, :] = new_v
            q_layer.countdown = k_layer.countdown = v_layer.countdown = layer_spec.q
        else:
            q_layer.countdown = k_layer.countdown = v_layer.countdown = countdown - al
            self_last_qkv.q.special[layer_idx, :, :, :] = a_q[:, -1, :, :]
            self_last_qkv.k.special[layer_idx, :, :, :] = a_k[:, -1, :, :]
            self_last_qkv.v.special[layer_idx, :, :, :] = a_v[:, -1, :, :]

        start += al

    assert self_last_qkv.m == layer_idx or self_last_qkv.m == kv_cache.num_ltv_layers()
    if self_last_qkv.m != kv_cache.num_ltv_layers():
        self_last_qkv.m += 1

    # ---- Safe concatenation: always include an empty placeholder to handle zero tokens ----
    B = x.shape[0]
    D = ltv_filter.head_dim

    q_parts = [torch.empty(B, 0, ltv_filter.n_head, D, dtype=x.dtype, device=x.device)]
    k_parts = [
        torch.empty(B, 0, ltv_filter.n_kv_head, D, dtype=x.dtype, device=x.device)
    ]
    v_parts = [
        torch.empty(B, 0, ltv_filter.n_kv_head, D, dtype=x.dtype, device=x.device)
    ]

    for q, k, v in result:
        q_parts.append(q)
        k_parts.append(k)
        v_parts.append(v)

    return (
        torch.cat(q_parts, dim=1),
        torch.cat(k_parts, dim=1),
        torch.cat(v_parts, dim=1),
    )


class LtvLookBackQkvComputerModule(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        ltv_filter: LtvLookBackHeadMajorQkvFilter,
    ):
        super().__init__()
        self.ltv_filter = ltv_filter
        self.advance_last_qkv = lambda kv_cache, resolved_inits, x: advance_kv_cache(
            x, layer_idx, self.ltv_filter, resolved_inits, kv_cache
        )

    def forward(
        self, x, kv_cache: Optional[KVCache]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, _ = x.size()
        resolved_inits = self.ltv_filter.init_param.unsqueeze(0).expand(B, -1, -1)

        if kv_cache is not None:
            q, k, v = self.advance_last_qkv(kv_cache, resolved_inits, x)
        else:
            q, k, v = self.ltv_filter(x, resolved_inits, full_recurrence=False)

        return q, k, v

    def init_weights(
        self, alpha_mlp_bias_init: float, weight_s: float, bias_s: float, init_s: float
    ):
        self.ltv_filter.init_weights(
            alpha_mlp_bias_init=alpha_mlp_bias_init,
            weight_s=weight_s,
            bias_s=bias_s,
            init_s=init_s,
        )

    def get_matrix_linear_params(self):
        return self.ltv_filter.get_matrix_linear_params()

    def estimate_flops(self, sequence_len: int) -> int:
        return self.ltv_filter.estimate_flops(sequence_len)

    def ltv_parameter_count(self) -> int:
        return self.ltv_filter.ltv_parameter_count()
