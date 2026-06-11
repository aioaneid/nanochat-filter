#include <torch/extension.h>

void ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_combined,
    torch::Tensor grad_inits, torch::Tensor grad_logit_bias, int B, int T,
    int NH, int NKVH, int Da, int r, int P, int Q, int K, int scan_threads,
    int r_threads, int r_items, bool use_sigmoid);

void register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
    pybind11::module_ &m) {
  m.def(
      "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_"
      "backward",
      &ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_impl,
      "LTV Lower Bound Fused Concat Head-Major Register-Cast Blelloch Scan "
      "Backward "
      "(CUDA)");
}
