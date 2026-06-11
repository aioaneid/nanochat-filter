#include <torch/extension.h>

void fused_linear_transpose_backward(torch::Tensor x, torch::Tensor weight, torch::Tensor grad_out, torch::Tensor grad_x, torch::Tensor grad_weight);

void register_fused_linear_transpose_backward(pybind11::module_ &m) {
  m.def("fused_linear_transpose_backward", &fused_linear_transpose_backward,
    "Fused Linear Transpose backward (CUDA)");
}
