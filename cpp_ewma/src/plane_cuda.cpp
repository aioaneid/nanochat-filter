#include <torch/extension.h>

// Implemented in plane_cuda_kernel.cu
void ltv_plane_cuda(torch::Tensor alpha, torch::Tensor x, torch::Tensor out_alpha,
                    torch::Tensor out_x, int t_seq, int parallel_a, int r,
                    int schedule_kind);

void ltv_plane(torch::Tensor alpha, torch::Tensor x, torch::Tensor out_alpha,
               torch::Tensor out_x, int t_seq, int parallel_a, int r,
               int schedule_kind) {
  ltv_plane_cuda(alpha, x, out_alpha, out_x, t_seq, parallel_a, r, schedule_kind);
}
