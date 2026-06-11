#include <torch/extension.h>
#include <vector>

void ltv_fused_concat_register_cast_scan_cuda_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r);

void ltv_fused_concat_register_cast_scan_cuda_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits, int B, int T, int NH,
    int NKVH, int Da, int r);

void ltv_fused_concat_register_cast_scan_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r) {
  ltv_fused_concat_register_cast_scan_cuda_forward(
      combined, inits, out_q, out_k, out_v, B, T, NH, NKVH, Da, r);
}

void ltv_fused_concat_register_cast_scan_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits, int B, int T, int NH,
    int NKVH, int Da, int r) {
  ltv_fused_concat_register_cast_scan_cuda_backward(
      combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k, grad_out_v,
      grad_combined, grad_inits, B, T, NH, NKVH, Da, r);
}
