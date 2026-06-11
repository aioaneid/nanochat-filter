#include <torch/extension.h>

void ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    int B, int T, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int scan_threads, int r_threads, int r_items, bool use_sigmoid);

void register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
    pybind11::module_ &m) {
  m.def(
      "ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward",
      &ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_impl,
      "LTV Lower Bound Fused Concat Head-Major Register-Cast Blelloch Scan Forward (CUDA)");
}
