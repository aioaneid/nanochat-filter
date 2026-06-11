import torch

lib = torch.library.Library("nanochat", "DEF")

# Autograd-visible op
lib.define("ltv_noop_metal_op(Tensor h0, Tensor alpha, Tensor x) -> Tensor")
lib.define(
    "ltv_shared_metal_op(Tensor h0, Tensor alpha, Tensor x, str schedule) -> Tensor"
)
lib.define(
    "ltv_plane_metal_op(Tensor h0, Tensor alpha, Tensor x, str schedule) -> Tensor"
)
lib.define("ltv_scan_metal_op(Tensor h0, Tensor alpha, Tensor x) -> Tensor")

# Opaque kernel op (NO autograd, NO meta)
lib.define(
    "ltv_noop_metal_kernel("
    "Tensor h_init, Tensor alpha, Tensor x, "
    "Tensor(a!) out, "
    "int t_dim, int parallel_a, int r"
    ") -> ()"
)

lib.define(
    "ltv_plane_metal_kernel("
    "Tensor alpha, Tensor x, "
    "Tensor(a!) out_alpha, Tensor(b!) out_x, "
    "int t_seq, int parallel_a, int r, str schedule"
    ") -> ()"
)


lib.define(
    "ltv_shared_metal_kernel("
    "Tensor alpha, Tensor x, "
    "Tensor(a!) out_alpha, Tensor(b!) out_x, "
    "int t_seq, int parallel_a, int r, str schedule"
    ") -> ()"
)

lib.define(
    schema="ltv_scan_metal_kernel("
    "Tensor alpha, Tensor x, Tensor(a!) out_alpha, Tensor(b!) out_x, "
    "int t_seq, int parallel_a, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_scan_metal_op(Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, int NH, int NKVH, int da, int r) -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_scan_metal_forward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_scan_metal_backward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor logits, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_out_q, Tensor grad_out_k, Tensor grad_out_v, "
    "Tensor(a!) dx_q, Tensor(b!) dx_k, Tensor(c!) dx_v, "
    "Tensor(d!) dl, Tensor(e!) di, "
    "int sbq, int shq, int stq, int sdq, "  # Strides for grad_out_q
    "int sbk, int shk, int stk, int sdk, "  # Strides for grad_out_k
    "int sbv, int shv, int stv, int sdv, "  # Strides for grad_out_v
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

# Main autograd-registered operation
lib.define(
    "ltv_fused_linear_scan_metal_op(Tensor q, Tensor k, Tensor v, Tensor x_gate, Tensor w_gate, Tensor b_gate, Tensor inits, int da, int r) -> (Tensor, Tensor, Tensor)"
)

# Forward Kernel (Mutates out_q, out_k, out_v)
lib.define(
    "ltv_fused_linear_scan_metal_forward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor x_gate, Tensor w_gate, Tensor b_gate, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int da, int r"
    ") -> ()"
)

# Backward Kernel (Mutates dx_q, dx_k, dx_v, dw_gate, db_gate, di)
lib.define(
    "ltv_fused_linear_scan_metal_backward_kernel("
    "Tensor q, Tensor k, Tensor v, Tensor x_gate, Tensor w_gate, Tensor b_gate, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_out_q, Tensor grad_out_k, Tensor grad_out_v, "
    "Tensor(a!) dx_q, Tensor(b!) dx_k, Tensor(c!) dx_v, "
    "Tensor(d!) dx_gate, Tensor(d!) dw_gate, Tensor(e!) db_gate, "
    "Tensor(f!) di, "
    "int sbq, int shq, int stq, int sdq, "  # Strides for grad_out_q
    "int sbk, int shk, int stk, int sdk, "  # Strides for grad_out_k
    "int sbv, int shv, int stv, int sdv, "  # Strides for grad_out_v
    "int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_scan_metal_op("
    "Tensor combined, Tensor logit_bias, Tensor inits, "
    "int NH, int NKVH, int da, int r"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_scan_metal_forward_kernel("
    "Tensor combined, Tensor logit_bias, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int sc_b, int sc_t, int sc_d, "  # Strides for combined
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_scan_metal_backward_kernel("
    "Tensor combined, Tensor logit_bias, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_out_q, Tensor grad_out_k, Tensor grad_out_v, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "int sc_b, int sc_t, int sc_d, "  # Strides for input combined
    "int sgc_b, int sgc_t, int sgc_d, "  # Strides for grad_combined
    "int sbq, int shq, int stq, int sdq, "
    "int sbk, int shk, int stk, int sdk, "
    "int sbv, int shv, int stv, int sdv, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

# Metal fused concat scan with register-level casting
lib.define(
    "ltv_fused_concat_register_cast_scan_metal_op(Tensor combined, Tensor inits, int NH, int NKVH, int da, int r) -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_fused_concat_register_cast_scan_metal_backward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor h_fwd_q, Tensor h_fwd_k, Tensor h_fwd_v, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "int sc_b, int sc_t, int sc_d, "
    "int sgc_b, int sgc_t, int sgc_d, "
    "int sbq, int shq, int stq, int sdq, "
    "int sbk, int shk, int stk, int sdk, "
    "int sbv, int shv, int stv, int sdv, "
    "int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_fused_concat_register_cast_scan_metal_forward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int sc_b, int sc_t, int sc_d, "
    "int T, int NH, int NKVH, int da, int r"
    ") -> ()"
)

lib.define(
    "ltv_metal_fused_concat_register_cast_blelloch_scan_op("
    "Tensor combined, Tensor inits, "
    "int NH, int NKVH, int da, int r, bool use_sigmoid"
    ") -> (Tensor, Tensor, Tensor)"
)

lib.define(
    "ltv_metal_fused_concat_register_cast_blelloch_scan_forward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor(a!) out_q, Tensor(b!) out_k, Tensor(c!) out_v, "
    "int sc_b, int sc_t, int sc_d, "
    "int T, int NH, int NKVH, int da, int r, bool use_sigmoid"
    ") -> ()"
)

lib.define(
    "ltv_metal_fused_concat_register_cast_blelloch_scan_backward_kernel("
    "Tensor combined, Tensor inits, "
    "Tensor hq, Tensor hk, Tensor hv, "
    "Tensor grad_hq, Tensor grad_hk, Tensor grad_hv, "
    "Tensor(a!) grad_combined, Tensor(b!) grad_inits, "
    "Tensor(c!) grad_logits, "
    "int sc_in_b, int sc_in_t, int sc_in_d, "
    "int sc_out_b, int sc_out_t, int sc_out_d, "
    "int NH, int NKVH, int da, int r, bool use_sigmoid"
    ") -> ()"
)
