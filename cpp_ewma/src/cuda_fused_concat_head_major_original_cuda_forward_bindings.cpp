#include <torch/extension.h>

void cuda_fused_concat_head_major_original_cuda_forward(
    torch::Tensor combined, torch::Tensor out_q, torch::Tensor out_k,
    torch::Tensor out_v, int B, int T, int NH, int NKVH, int Da, int r,
    int scan_threads, int r_threads, int r_items);

void register_cuda_fused_concat_head_major_original_cuda_forward(
    pybind11::module_ &m) {
  m.def("cuda_fused_concat_head_major_original_cuda_forward",
        &cuda_fused_concat_head_major_original_cuda_forward,
        "LTV Fused Concat Head-Major Original Forward (CUDA)");
}
