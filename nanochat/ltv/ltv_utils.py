import torch


def dimensions(alpha, initial_value, x):
    B, T, D = x.shape

    assert initial_value.shape == (B, D), (
        f"initial_value shape {initial_value.shape} must match (B, D)"
    )

    # Check shape compatibility
    assert alpha.shape[:2] == (B, T), (
        f"alpha shape {alpha.shape} must match (B, T, ...)"
    )
    D_a = alpha.shape[2]

    if D_a == 0:
        if D != 0:
            raise ValueError(f"x dim {D} > 0 but alpha dim is 0")
        r = 1
    else:
        if D % D_a != 0:
            raise ValueError(f"x dim {D} must be multiple of alpha dim {D_a}")
        r = D // D_a
    return B, T, D_a, r


def sdpa_deterministic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool,
    enable_gqa: bool = False,
):
    """
    Deterministic scaled dot-product attention with optional GQA.

    Shapes:
      q: (B, Hq,  Tq, D)
      k: (B, Hkv, Tk, D)
      v: (B, Hkv, Tk, D)

    attn_mask (optional):
      shape (Tq, Tk), dtype=bool
      True = keep, False = mask
    """

    B, Hq, Tq, D = q.shape
    _, Hkv, Tk, _ = k.shape

    if enable_gqa:
        if Hq % Hkv != 0:
            raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({Hkv})")
        group_size = Hq // Hkv

        # Expand K/V to match query heads deterministically
        # (B, Hkv, Tk, D) -> (B, Hq, Tk, D)
        k = k[:, :, None, :, :].expand(B, Hkv, group_size, Tk, D).reshape(B, Hq, Tk, D)
        v = v[:, :, None, :, :].expand(B, Hkv, group_size, Tk, D).reshape(B, Hq, Tk, D)
    else:
        if Hq != Hkv:
            raise ValueError("enable_gqa=False requires Hq == Hkv")

    # Promote to fp32 for deterministic accumulation
    qf = q.float()
    kf = k.float()
    vf = v.float()

    scale = 1.0 / (D**0.5)

    # (B, Hq, Tq, Tk)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    if attn_mask is not None:
        # attn_mask: (Tq, Tk), True = keep
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    elif is_causal:
        causal = torch.tril(
            torch.ones((Tq, Tk), dtype=torch.bool, device=scores.device)
        )
        scores = scores.masked_fill(~causal, float("-inf"))

    # Explicit softmax
    attn = torch.softmax(scores, dim=-1)

    # (B, Hq, Tq, D)
    out = torch.matmul(attn, vf)

    return out.to(dtype=q.dtype)


def make_aligned_copy(tensor, alignment=16384):
    # 1. Check if already aligned and contiguous
    if tensor.data_ptr() % alignment == 0 and tensor.is_contiguous():
        return tensor

    size_bytes = tensor.numel() * tensor.element_size()

    # 2. Minimum padding required is alignment - 1
    padding = alignment - 1

    # 3. Allocate storage with strict minimum size
    storage = torch.empty(size_bytes + padding, dtype=torch.uint8, device=tensor.device)

    # 4. Calculate offset
    ptr = storage.data_ptr()
    ptr_mod_alignment = ptr % alignment
    offset = alignment - ptr_mod_alignment if ptr_mod_alignment else 0

    # 5. Create aligned view
    # Because we allocated size + (alignment - 1),
    # even if offset is (alignment - 1), the slice will fit exactly.
    aligned_tensor = (
        storage[offset : offset + size_bytes]
        .view(tensor.dtype)
        .to(tensor.dtype)
        .reshape(tensor.shape)
    )

    aligned_tensor.copy_(tensor)

    assert aligned_tensor.data_ptr() % alignment == 0, aligned_tensor.data_ptr()
    assert aligned_tensor.is_contiguous()

    return aligned_tensor
