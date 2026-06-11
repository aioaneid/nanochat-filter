#include <torch/extension.h>

void fused_linear_transpose_impl(torch::Tensor x, torch::Tensor weight,
             torch::Tensor out);

void register_fused_linear_transpose_forward(pybind11::module_ &m) {
  m.def("fused_linear_transpose", &fused_linear_transpose_impl,
    "Fused Linear and Transpose via cuBLAS Strided Batched GEMM");
}
