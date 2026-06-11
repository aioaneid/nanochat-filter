#include <torch/extension.h>

#ifdef BUILD_LTV_FUSED_SCAN
void ltv_fused_scan_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            torch::Tensor logits, torch::Tensor inits,
                            torch::Tensor out_q, torch::Tensor out_k,
                            torch::Tensor out_v, int B, int T, int NH, int NKVH,
                            int Da, int r);

void ltv_fused_scan_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             torch::Tensor logits, torch::Tensor inits,
                             torch::Tensor out_q, torch::Tensor out_k,
                             torch::Tensor out_v, torch::Tensor grad_out_q,
                             torch::Tensor grad_out_k, torch::Tensor grad_out_v,
                             torch::Tensor grad_q, torch::Tensor grad_k,
                             torch::Tensor grad_v, torch::Tensor grad_logits,
                             torch::Tensor grad_inits, int B, int T, int NH,
                             int NKVH, int Da, int r);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_SCAN
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
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_SCAN
void ltv_fused_concat_head_major_register_cast_scan_cuda_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r);

void ltv_fused_concat_head_major_register_cast_scan_cuda_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits, int B, int T, int NH,
    int NKVH, int Da, int r);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_BLELLOCH
void ltv_fused_concat_register_cast_blelloch_scan_cuda_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r, int chunk_size, int bank_count, bool use_sigmoid);

void ltv_fused_concat_register_cast_blelloch_scan_cuda_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits,
    torch::Tensor grad_logits, int B, int T, int NH, int NKVH, int Da, int r,
    int chunk_size, int bank_count, bool use_sigmoid);
#endif

#if defined(BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH) ||                             \
    defined(BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL) ||                             \
    defined(BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH)
void register_fused_linear_transpose_forward(pybind11::module_ &m);
void register_fused_linear_transpose_backward(pybind11::module_ &m);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH
void register_ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
    pybind11::module_ &m);
void register_ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
    pybind11::module_ &m);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL
void register_cuda_fused_concat_head_major_original_cuda_forward(
    pybind11::module_ &m);
void register_cuda_fused_concat_head_major_original_cuda_backward(
    pybind11::module_ &m);
#endif

#ifdef BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH
void register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
    pybind11::module_ &m);
void register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
    pybind11::module_ &m);
#endif

#ifdef BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_SEQUENTIAL
void register_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward(
    pybind11::module_ &m);
void register_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward(
    pybind11::module_ &m);
#endif

#ifdef BUILD_LTV_PLANE
void ltv_plane(torch::Tensor alpha, torch::Tensor x, torch::Tensor out_alpha,
               torch::Tensor out_x, int t_seq, int parallel_a, int r,
               int schedule_kind);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef BUILD_LTV_FUSED_SCAN
  m.def("ltv_fused_scan_forward", &ltv_fused_scan_forward,
        "LTV Fused Scan Forward (CUDA)");
  m.def("ltv_fused_scan_backward", &ltv_fused_scan_backward,
        "LTV Fused Scan Backward (CUDA)");
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_SCAN
  m.def("ltv_fused_concat_register_cast_scan_cuda_forward",
        &ltv_fused_concat_register_cast_scan_cuda_forward,
        "LTV Fused Concat Register-Cast Scan Forward (CUDA)");
  m.def("ltv_fused_concat_register_cast_scan_cuda_backward",
        &ltv_fused_concat_register_cast_scan_cuda_backward,
        "LTV Fused Concat Register-Cast Scan Backward (CUDA)");
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_SCAN
  m.def("ltv_fused_concat_head_major_register_cast_scan_cuda_forward",
        &ltv_fused_concat_head_major_register_cast_scan_cuda_forward,
        "LTV Fused Concat Head-Major Register-Cast Scan Forward (CUDA)");
  m.def("ltv_fused_concat_head_major_register_cast_scan_cuda_backward",
        &ltv_fused_concat_head_major_register_cast_scan_cuda_backward,
        "LTV Fused Concat Head-Major Register-Cast Scan Backward (CUDA)");
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_BLELLOCH
  m.def("ltv_fused_concat_register_cast_blelloch_scan_cuda_forward",
        &ltv_fused_concat_register_cast_blelloch_scan_cuda_forward,
        "LTV Fused Concat Register-Cast Blelloch Scan Forward (CUDA)");
  m.def("ltv_fused_concat_register_cast_blelloch_scan_cuda_backward",
        &ltv_fused_concat_register_cast_blelloch_scan_cuda_backward,
        "LTV Fused Concat Register-Cast Blelloch Scan Backward (CUDA)");
#endif

#if defined(BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH) ||                             \
    defined(BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL) ||                             \
    defined(BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH)
  register_fused_linear_transpose_forward(m);
  register_fused_linear_transpose_backward(m);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH
  register_ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
      m);
  register_ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
      m);
#endif

#ifdef BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL
  register_cuda_fused_concat_head_major_original_cuda_forward(m);
  register_cuda_fused_concat_head_major_original_cuda_backward(m);
#endif

#ifdef BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH
  register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward(
      m);
  register_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward(
      m);
#endif

#ifdef BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_SEQUENTIAL
  register_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward(
      m);
  register_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward(
      m);
#endif

#ifdef BUILD_LTV_PLANE
  m.def("ltv_plane", &ltv_plane, "LTV Plane Blelloch Scan (CUDA)");
#endif
}
