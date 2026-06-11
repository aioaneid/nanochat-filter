#include <torch/extension.h>

void ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r, int block_threads, int items_per_thread, bool use_sigmoid);

void ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits,
    torch::Tensor grad_logits, int B, int T, int NH, int NKVH, int Da, int r,
    int block_threads, int items_per_thread, bool use_sigmoid);

void ltv_fused_concat_register_cast_blelloch_scan_cuda_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r, int block_threads, int items_per_thread, bool use_sigmoid) {
  ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_impl(
      combined, inits, out_q, out_k, out_v, B, T, NH, NKVH, Da, r,
      block_threads, items_per_thread, use_sigmoid);
}

void ltv_fused_concat_register_cast_blelloch_scan_cuda_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits,
    torch::Tensor grad_logits, int B, int T, int NH, int NKVH, int Da, int r,
    int block_threads, int items_per_thread, bool use_sigmoid) {
  ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_impl(
      combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k, grad_out_v,
      grad_combined, grad_inits, grad_logits, B, T, NH, NKVH, Da, r,
      block_threads, items_per_thread, use_sigmoid);
}
