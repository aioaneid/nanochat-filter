import torch

lib = torch.library.Library("nanochat_cuda", "DEF")

# Autograd-visible op
lib.define(
    "ltv_plane_cuda_op(Tensor h0, Tensor alpha, Tensor x, str schedule) -> Tensor"
)

# Opaque kernel op (NO autograd, NO meta)
lib.define(
    "ltv_plane_cuda_kernel("
    "Tensor alpha, Tensor x, "
    "Tensor(a!) out_alpha, Tensor(b!) out_x, "
    "int t_seq, int parallel_a, int r, str schedule"
    ") -> ()"
)

# CUDA fused scan (q/k/v/logits/inits)
lib.define(
    "ltv_fused_scan_cuda_op(Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, int NH, int NKVH, int da, int r) -> (Tensor, Tensor, Tensor)"
)
lib.define(
    "ltv_fused_scan_cuda_backward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_q, Tensor(b!) grad_k, Tensor(c!) grad_v, "
    "Tensor(d!) grad_logits, Tensor(e!) grad_inits, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_scan_cuda_forward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r"
    ") -> ()"
)

# CUDA fused concat scan with register-level casting
lib.define(
    "ltv_fused_concat_register_cast_scan_cuda_op(Tensor combined, Tensor inits, int NH, int NKVH, int da, int r) -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_register_cast_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_register_cast_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_scan_cuda_op("
    "Tensor combined, Tensor inits, int NH, int NKVH, int da, int r"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_register_cast_blelloch_scan_cuda_op("
    "Tensor combined, Tensor inits, "
    "int NH, int NKVH, int da, int r, int chunk_size, int bank_count, bool use_sigmoid"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int T, int NH, int NKVH, int da, int r, int chunk_size, int bank_count, bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor hq, Tensor hk, Tensor hv, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "Tensor(c!) grad_logits, "
    "int NH, int NKVH, int da, int r, int chunk_size, int bank_count, bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_op("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int T, int NH, int NKVH, int da, int r, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor hq, Tensor hk, Tensor hv, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, Tensor(c!) grad_logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "cuda_fused_concat_head_major_original_cuda_op("
    "Tensor combined, int NH, int NKVH, int da, int r, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "cuda_fused_concat_head_major_original_cuda_forward_kernel("
    "Tensor combined, Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward"
    ") -> ()"
)

lib.define(
    "cuda_fused_concat_head_major_original_cuda_backward_kernel("
    "Tensor combined, Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, "
    "int B, int T, int NH, int NKVH, int da, int r, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward"
    ") -> ()"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int scan_threads_forward, int r_threads_forward, int r_items_forward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor hq, Tensor hk, Tensor hv, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, Tensor(c!) grad_logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int scan_threads_backward, int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_op("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int r_threads_forward, int r_items_forward, "
    "int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int B, int T, int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int r_threads_forward, int r_items_forward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward_kernel("
    "Tensor combined, Tensor inits, Tensor logit_bias, "
    "Tensor hq, Tensor hk, Tensor hv, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, Tensor(c!) grad_logit_bias, "
    "int NH, int NKVH, int da, int r, "
    "int P, int Q, int K, "
    "int r_threads_backward, int r_items_backward, "
    "bool use_sigmoid"
    ") -> ()"
)

lib.define("fused_linear_transpose(Tensor x, Tensor weight, Tensor(a!) out) -> ()")

lib.define("fused_linear_transpose_cuda_op(Tensor x, Tensor weight) -> Tensor")

lib.define(
    "fused_linear_transpose_backward_kernel("
    "Tensor x, Tensor weight, Tensor grad_out, Tensor(a!) grad_x, Tensor(b!) grad_weight"
    ") -> ()"
)


import nanochat.ops.fused_linear_transpose  # noqa: F401


def fused_linear_transpose(x: torch.Tensor, weight: torch.Tensor):
    """
    Functional wrapper that creates the output tensor and calls the native kernel.
    Returns a new tensor shaped (B, F, T) with contiguous memory format.
    """
    import nanochat.ops.fused_linear_transpose  # ensure impls are registered

    return torch.ops.nanochat_cuda.fused_linear_transpose_cuda_op(x, weight)


def ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan(
    combined: torch.Tensor,
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    da: int,
    r: int,
    scan_threads_forward: int,
    r_threads_forward: int,
    r_items_forward: int,
    scan_threads_backward: int,
    r_threads_backward: int,
    r_items_backward: int,
    use_sigmoid: bool,
):
    import nanochat.ops.ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan

    return torch.ops.nanochat_cuda.ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        da,
        r,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
        use_sigmoid,
    )


def cuda_fused_concat_head_major_original(
    combined: torch.Tensor,
    NH: int,
    NKVH: int,
    da: int,
    r: int,
    scan_threads_forward: int,
    r_threads_forward: int,
    r_items_forward: int,
    scan_threads_backward: int,
    r_threads_backward: int,
    r_items_backward: int,
):
    import nanochat.ops.cuda_fused_concat_head_major_original

    return torch.ops.nanochat_cuda.cuda_fused_concat_head_major_original_cuda_op(
        combined,
        NH,
        NKVH,
        da,
        r,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
    )


def ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan(
    combined: torch.Tensor,
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    scan_threads_forward: int,
    r_threads_forward: int,
    r_items_forward: int,
    scan_threads_backward: int,
    r_threads_backward: int,
    r_items_backward: int,
    use_sigmoid: bool,
):
    import nanochat.ops.ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan

    return torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_op(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        da,
        r,
        P,
        Q,
        K,
        scan_threads_forward,
        r_threads_forward,
        r_items_forward,
        scan_threads_backward,
        r_threads_backward,
        r_items_backward,
        use_sigmoid,
    )


def ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan(
    combined: torch.Tensor,
    inits: torch.Tensor,
    logit_bias: torch.Tensor,
    NH: int,
    NKVH: int,
    da: int,
    r: int,
    P: int,
    Q: int,
    K: int,
    r_threads_forward: int,
    r_items_forward: int,
    r_threads_backward: int,
    r_items_backward: int,
    use_sigmoid: bool,
):
    import nanochat.ops.ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan

    return torch.ops.nanochat_cuda.ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_op(
        combined,
        inits,
        logit_bias,
        NH,
        NKVH,
        da,
        r,
        P,
        Q,
        K,
        r_threads_forward,
        r_items_forward,
        r_threads_backward,
        r_items_backward,
        use_sigmoid,
    )
