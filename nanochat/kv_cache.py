import collections
from typing import Optional
import torch

from nanochat.gpt_config import ChunksAndHistory

_INVALID_COUNTDOWN = -1


class Layer:
    countdown: int
    # Each layer may have a different mt, hence a different deque size
    d: collections.deque[torch.Tensor]

    def __init__(self, countdown: int, d: collections.deque[torch.Tensor]):
        self.countdown = countdown
        self.d = d

    def merge_from(self, other):
        self.countdown = other.countdown
        for x, y in zip(self.d, other.d, strict=True):
            # assert x.shape == y.shape
            x[:, :, :] = y

    def is_initialized(self):
        return self.countdown != _INVALID_COUNTDOWN


class LookBack:
    # Special has all layers combined.
    special: torch.Tensor
    layers: list[Layer]

    def __init__(self, special: torch.Tensor, layers: list[Layer]):
        self.special = special
        self.layers = layers
        assert len(special) == len(self.layers)
        for layer in layers:
            for x in layer.d:
                assert x.shape == special.shape[1:]

    def merge_from(self, other):
        self.special[:, :, :, :] = other.special
        for layer, other_layer in zip(self.layers, other.layers):
            layer.merge_from(other_layer)


def _empty_with_shape_and_mt(shape, dtype, device, mls: list[int]):
    assert shape[0] == len(mls)
    return LookBack(
        special=torch.empty(
            shape,
            dtype=dtype,
            device=device,
            memory_format=torch.contiguous_format,
        ),
        layers=[
            Layer(
                countdown=_INVALID_COUNTDOWN,
                d=collections.deque(
                    [
                        torch.empty(
                            shape[1:],
                            dtype=dtype,
                            device=device,
                            memory_format=torch.contiguous_format,
                        )
                        for _ in range(ml)
                    ],
                    maxlen=ml,
                ),
            )
            for ml in mls
        ],
    )


def _empty_like_with_shape(shape, other: LookBack):
    return _empty_with_shape_and_mt(
        shape,
        other.special.dtype,
        other.special.device,
        mls=[len(layer.d) for layer in other.layers],
    )


class LastQkv:
    q: Optional[LookBack]
    k: Optional[LookBack]
    v: Optional[LookBack]
    m: int

    def __init__(self, q, k, v, m: int):
        self.q = q
        self.k = k
        self.v = v
        self.m = m

    def merge_from(self, other_last_qkv):
        if (t := other_last_qkv.q) is not None:
            self.q.merge_from(t)
            assert other_last_qkv.m == len(t.special)
        else:
            assert self.q is None
        if (t := other_last_qkv.k) is not None:
            self.k.merge_from(t)
            assert other_last_qkv.m == len(t.special)
        else:
            assert self.k is None
        if (t := other_last_qkv.v) is not None:
            self.v.merge_from(t)
            assert other_last_qkv.m == len(t.special)
        else:
            assert self.v is None
        self.m = other_last_qkv.m


def _make_empty_last_qkv_like(
    last_q_shape: tuple, last_k_v_shape: tuple, other: LastQkv
) -> LastQkv:
    return LastQkv(
        q=None if (t := other.q) is None else _empty_like_with_shape(last_q_shape, t),
        k=None if (t := other.k) is None else _empty_like_with_shape(last_k_v_shape, t),
        v=None if (t := other.v) is None else _empty_like_with_shape(last_k_v_shape, t),
        m=0,
    )


class TypeDevice:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device


def make_empty_last_qkv_like_specials(
    last_q_shape: tuple,
    last_k_v_shape: tuple,
    q: Optional[TypeDevice],
    k: Optional[TypeDevice],
    v: Optional[TypeDevice],
    mls: list[int],
) -> LastQkv:
    return LastQkv(
        q=None
        if q is None
        else _empty_with_shape_and_mt(
            last_q_shape,
            dtype=q.dtype,
            device=q.device,
            mls=mls,
        ),
        k=None
        if k is None
        else _empty_with_shape_and_mt(
            last_k_v_shape,
            dtype=k.dtype,
            device=k.device,
            mls=mls,
        ),
        v=None
        if v is None
        else _empty_with_shape_and_mt(
            last_k_v_shape,
            dtype=v.dtype,
            device=v.device,
            mls=mls,
        ),
        m=0,
    )


def _create_last_qkv_from(
    last_q_shape, last_k_v_shape, other_last_qkv: LastQkv
) -> LastQkv:
    last_qkv = _make_empty_last_qkv_like(last_q_shape, last_k_v_shape, other_last_qkv)
    last_qkv.merge_from(other_last_qkv)
    return last_qkv


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(
        self,
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        num_layers,
        layer_specs: list[ChunksAndHistory],
    ):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        assert num_q_heads % num_kv_heads == 0
        self.kv_shape = (num_layers, 2, batch_size, num_kv_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in time in the cache
        self.last_qkv = None
        self.last_q_shape = (len(layer_specs), batch_size, num_q_heads, head_dim)
        self.last_k_v_shape = (len(layer_specs), batch_size, num_kv_heads, head_dim)
        self.layer_specs = layer_specs

    def num_ltv_layers(self):
        return len(self.layer_specs)

    def get_pos(self):
        return self.pos

    def update_last_qkv(
        self,
        layer_idx: int,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
    ):
        """Update extra state by key."""
        if (self_last_qkv := self.last_qkv) is None:
            assert layer_idx == 0
            self_last_qkv = self.last_qkv = make_empty_last_qkv_like_specials(
                self.last_q_shape,
                self.last_k_v_shape,
                None if q is None else TypeDevice(q.dtype, q.device),
                None if k is None else TypeDevice(k.dtype, k.device),
                None if v is None else TypeDevice(v.dtype, v.device),
                [0] * self.num_ltv_layers(),
            )
        else:
            assert (q is None) == (self_last_qkv.q is None)
            assert (k is None) == (self_last_qkv.k is None)
            assert (v is None) == (self_last_qkv.v is None)
        assert self_last_qkv.m == layer_idx or self_last_qkv.m == self.num_ltv_layers()
        if q is None:
            assert self_last_qkv.q is None
        else:
            assert self_last_qkv.q is not None
            assert q.shape == self.last_q_shape[1:]
            self_last_qkv.q.special[layer_idx, :, :, :] = q
        if k is None:
            assert self_last_qkv.k is None
        else:
            assert self_last_qkv.k is not None
            assert k.shape == self.last_k_v_shape[1:]
            self_last_qkv.k.special[layer_idx, :, :, :] = k
        if v is None:
            assert self_last_qkv.v is None
        else:
            assert self_last_qkv.v is not None
            assert v.shape == self.last_k_v_shape[1:]
            self_last_qkv.v.special[layer_idx, :, :, :] = v
        if self_last_qkv.m != self.num_ltv_layers():
            self_last_qkv.m += 1

    def has_last_qkv(self, layer_idx: int):
        return (self_last_qkv := self.last_qkv) and layer_idx < self_last_qkv.m

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        # other.kv_cache may be None if the source cache has never been used (pos == 0)
        if other.kv_cache is not None:
            # Normal case: source has KV tensor
            self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
            other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape

            assert self_layers == other_layers, f"Layer count mismatch: {self_layers} != {other_layers}"
            assert self_kv == other_kv, f"K/V dimension mismatch: {self_kv} != {other_kv}"
            assert self_heads == other_heads, f"Head count mismatch: {self_heads} != {other_heads}"
            assert self_head_dim == other_head_dim, f"Head dim mismatch: {self_head_dim} != {other_head_dim}"
            assert self_batch == other_batch or other_batch == 1, f"Batch size mismatch: {self_batch} vs {other_batch}"
            assert self_seq >= other.pos, f"Sequence length insufficient: {self_seq} < {other.pos}"

            dtype, device = other.kv_cache.dtype, other.kv_cache.device
            self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
            self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache
        else:
            # Source is completely empty – nothing to copy from KV side
            assert other.pos == 0, "Source has no KV cache but pos != 0"

        # 2) Update the look-back state (last_qkv) regardless
        self.last_qkv = (
            _create_last_qkv_from(self.last_q_shape, self.last_k_v_shape, other_last_qkv)
            if (other_last_qkv := other.last_qkv) is not None
            else None
        )
        # 3) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            t_needed = (
                t_needed + 1023
            ) & ~1023  # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(
                additional_shape, dtype=k.dtype, device=k.device
            )
            self.kv_cache = torch.cat(
                [self.kv_cache, additional_cache], dim=4
            ).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view
