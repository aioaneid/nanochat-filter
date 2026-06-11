import torch

lib = torch.library.Library("nanochat", "FRAGMENT")

# Autograd-visible op
lib.define("ltv_associative_scan_triton_op(Tensor h0, Tensor alpha, Tensor x) -> Tensor")

# Opaque kernel op (NO autograd, NO meta)
lib.define(
    "ltv_associative_scan_triton_kernel("
    "Tensor alpha, Tensor x, "
    "Tensor(a!) out_alpha, Tensor(b!) out_x, "
    "int t_seq, int parallel_a, int r"
    ") -> ()"
)

# Triton fused scan
lib.define(
    "ltv_fused_scan_triton_op(Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, int NH, int NKVH, int da, int r) -> (Tensor, Tensor, Tensor)"
)
lib.define(
    "ltv_fused_scan_triton_backward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_q, Tensor(b!) grad_k, Tensor(c!) grad_v, "
    "Tensor(d!) grad_logits, Tensor(e!) grad_inits, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

# Triton fused concat scan
lib.define(
    "ltv_fused_concat_scan_triton_op("
    "Tensor combined, Tensor logit_bias, Tensor inits, "
    "int NH, int NKVH, int da, int r"
    ") -> (Tensor, Tensor, Tensor)"
)
lib.define(
    "ltv_fused_concat_scan_triton_backward_kernel("
    "Tensor combined, Tensor logit_bias, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)
