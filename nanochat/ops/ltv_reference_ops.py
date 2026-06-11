import torch


def ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal(
    combined: torch.Tensor,  # (B, T, F) with F = total_h * (D + da)
    inits: torch.Tensor,  # (B, total_h, D)
    logit_bias: torch.Tensor,  # (total_h, da)
    NH: int,
    NKVH: int,
    Da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    use_sigmoid: bool,
):
    """
    Pure-Python reference for the chunked sequential scan.
    combined has shape (B, T, F) - exactly the layout used by the sequential
    CUDA kernel and also the layout produced by the Blelloch filter's _project.
    Returns:
        out_q : (B, NH, T, D)
        out_k : (B, NKVH, T, D)
        out_v : (B, NKVH, T, D)
    """
    B, T_seq, F = combined.shape
    total_h = NH + 2 * NKVH
    D = Da * r
    features_per_head = D + Da
    assert F == total_h * features_per_head

    # Chunk boundaries exactly as in the CUDA kernel
    if T_seq > P:
        M_dyn = (T_seq - P + Q - 1) // Q
        chunks = [(0, min(P, T_seq))]
        for i in range(1, M_dyn + 1):
            start = P + (i - 1) * Q
            end = min(P + i * Q, T_seq)
            if start < T_seq:
                chunks.append((start, end))
    else:
        chunks = [(0, T_seq)]

    out_q = torch.empty(B, NH, T_seq, D, device=combined.device, dtype=combined.dtype)
    out_k = torch.empty(B, NKVH, T_seq, D, device=combined.device, dtype=combined.dtype)
    out_v = torch.empty(B, NKVH, T_seq, D, device=combined.device, dtype=combined.dtype)

    compute_dtype = torch.float32  # matches kernel internal compute

    for b in range(B):
        for h_glob in range(total_h):
            if h_glob < NH:
                out_buf = out_q[b]
                h_loc = h_glob
            elif h_glob < NH + NKVH:
                out_buf = out_k[b]
                h_loc = h_glob - NH
            else:
                out_buf = out_v[b]
                h_loc = h_glob - (NH + NKVH)

            head_offset = h_glob * features_per_head
            bias = logit_bias[h_glob].to(dtype=compute_dtype)  # (da,)
            init_state = inits[b, h_glob].to(dtype=compute_dtype)  # (D,)

            for t_start, t_end in chunks:
                t_pref_start = max(0, t_start - (K - 1))
                state = init_state.clone()

                for t in range(t_pref_start, t_end):
                    x_val = combined[b, t, head_offset : head_offset + D].to(
                        compute_dtype
                    )
                    alpha_raw = (
                        combined[b, t, head_offset + D : head_offset + D + Da].to(
                            compute_dtype
                        )
                        + bias
                    )
                    alpha = torch.sigmoid(alpha_raw) if use_sigmoid else alpha_raw
                    alpha_bc = alpha.repeat_interleave(r)  # broadcast to D
                    state = (1.0 - alpha_bc) * state + alpha_bc * x_val

                    if t >= t_start:
                        out_buf[h_loc, t] = state.to(dtype=combined.dtype)

    return out_q, out_k, out_v


def ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_minimal(
    combined: torch.Tensor,  # (B, T, F) - same shape as sequential scan
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    Da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    use_sigmoid: bool,
):
    """
    Pure-Python reference for the Blelloch warp-scan kernel.
    The input combined tensor has the same (B, T, F) shape as the sequential
    scan reference because the filter's _project already transposes the output
    of the fused linear op.  The recurrence is identical, so we simply delegate
    to the sequential reference.
    """
    return ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        Da,
        r,
        P,
        Q,
        K,
        use_sigmoid,
    )


# ---------------------------------------------------------------------------
# Public references with full op signatures (ignored tuning parameters)
# ---------------------------------------------------------------------------


def ltv_sequential_scan_reference(
    combined: torch.Tensor,
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    Da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    r_threads_forward: int,  # ignored
    r_items_forward: int,  # ignored
    r_threads_backward: int,  # ignored
    r_items_backward: int,  # ignored
    use_sigmoid: bool,
):
    """Wrapper for the sequential scan op; drops tuning parameters."""
    return ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_minimal(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        Da,
        r,
        P,
        Q,
        K,
        use_sigmoid,
    )


def ltv_look_back_blelloch_scan_reference(
    combined: torch.Tensor,  # (B, T, F) - same layout as sequential
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    Da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    scan_threads_forward: int,  # ignored
    r_threads_forward: int,  # ignored
    r_items_forward: int,  # ignored
    scan_threads_backward: int,  # ignored
    r_threads_backward: int,  # ignored
    r_items_backward: int,  # ignored
    use_sigmoid: bool,
):
    """Wrapper for the Blelloch scan op; drops tuning parameters."""
    return ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_minimal(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        Da,
        r,
        P,
        Q,
        K,
        use_sigmoid,
    )


def fused_linear_transpose_reference(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Pure-Python equivalent of the CUDA fused_linear_transpose.
    x:   (B, T, K)
    weight: (F, K)
    returns: (B, F, T)
    """
    return torch.matmul(x, weight.T).transpose(1, 2).contiguous()
