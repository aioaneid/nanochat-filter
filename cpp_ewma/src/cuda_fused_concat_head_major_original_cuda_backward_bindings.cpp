#include <torch/extension.h>

void cuda_fused_concat_head_major_original_cuda_backward(
    torch::Tensor combined, torch::Tensor grad_hq, torch::Tensor grad_hk,
    torch::Tensor grad_hv, torch::Tensor grad_combined, int B, int T, int NH,
    int NKVH, int Da, int r, int scan_threads, int r_threads, int r_items);

void register_cuda_fused_concat_head_major_original_cuda_backward(
    pybind11::module_ &m) {
  m.def("cuda_fused_concat_head_major_original_cuda_backward",
        &cuda_fused_concat_head_major_original_cuda_backward,
        "LTV Fused Concat Head-Major Original Backward (CUDA)");
}
