#include <torch/extension.h>
#include <vector>

// ============================================================================
// CUDA Forward Declarations
// These tell the C++ compiler that these functions exist elsewhere (in .cu)
// ============================================================================

void ltv_fused_scan_cuda_forward(torch::Tensor q, torch::Tensor k,
                                 torch::Tensor v, torch::Tensor logits,
                                 torch::Tensor inits, torch::Tensor out_q,
                                 torch::Tensor out_k, torch::Tensor out_v,
                                 int B, int T, int NH, int NKVH, int Da, int r);

void ltv_fused_scan_cuda_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor logits,
    torch::Tensor inits, torch::Tensor out_q, torch::Tensor out_k,
    torch::Tensor out_v, torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_q, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_logits, torch::Tensor grad_inits,
    int B, int T, int NH, int NKVH, int Da, int r);

void ltv_plane(torch::Tensor alpha, torch::Tensor x, torch::Tensor out_alpha,
               torch::Tensor out_x, int t_seq, int parallel_a, int r,
               int schedule_kind);

// ============================================================================
// Python Wrappers (with Error Checking)
// ============================================================================

void ltv_fused_scan_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            torch::Tensor logits, torch::Tensor inits,
                            torch::Tensor out_q, torch::Tensor out_k,
                            torch::Tensor out_v, int B, int T, int NH, int NKVH,
                            int Da, int r) {
  ltv_fused_scan_cuda_forward(q, k, v, logits, inits, out_q, out_k, out_v, B, T,
                              NH, NKVH, Da, r);
}

void ltv_fused_scan_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             torch::Tensor logits, torch::Tensor inits,
                             torch::Tensor out_q, torch::Tensor out_k,
                             torch::Tensor out_v, torch::Tensor grad_out_q,
                             torch::Tensor grad_out_k, torch::Tensor grad_out_v,
                             torch::Tensor grad_q, torch::Tensor grad_k,
                             torch::Tensor grad_v, torch::Tensor grad_logits,
                             torch::Tensor grad_inits, int B, int T, int NH,
                             int NKVH, int Da, int r) {
  ltv_fused_scan_cuda_backward(q, k, v, logits, inits, out_q, out_k, out_v,
                               grad_out_q, grad_out_k, grad_out_v, grad_q,
                               grad_k, grad_v, grad_logits, grad_inits, B, T,
                               NH, NKVH, Da, r);
}
