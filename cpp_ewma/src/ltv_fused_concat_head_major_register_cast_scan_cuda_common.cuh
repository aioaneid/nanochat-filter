#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// -------------------------------------------------------------------------
// Generic constants (shared by both implementations)
// -------------------------------------------------------------------------
constexpr int kT = 1024;
constexpr int kNH = 8;
constexpr int kNKVH = 8;
constexpr int kDa = 1;
constexpr int kR = 128;

constexpr int64_t kTotalH = static_cast<int64_t>(kNH + 2 * kNKVH);
constexpr int64_t kD = static_cast<int64_t>(kDa) * kR;

// -------------------------------------------------------------------------
// Output gradient strides (shared by both Blelloch and sequential scans)
//   The output tensors (hq, hk, hv) are always allocated as contiguous
//   (B, H, T, D) with strides (H*T*D, T*D, D, 1), so these constants
//   are valid for both implementations.
// -------------------------------------------------------------------------
constexpr int64_t kStrideGradOutQB = static_cast<int64_t>(kNH) * kT * kD;
constexpr int64_t kStrideGradOutQH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutQT = kD;
constexpr int64_t kStrideGradOutQD = 1;

constexpr int64_t kStrideGradOutKB = static_cast<int64_t>(kNKVH) * kT * kD;
constexpr int64_t kStrideGradOutKH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutKT = kD;
constexpr int64_t kStrideGradOutKD = 1;

constexpr int64_t kStrideGradOutVB = static_cast<int64_t>(kNKVH) * kT * kD;
constexpr int64_t kStrideGradOutVH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutVT = kD;
constexpr int64_t kStrideGradOutVD = 1;

// Transposed variants (used when gradients arrive with H and T strides swapped)
constexpr int64_t kStrideGradOutQHTransposed = kD;
constexpr int64_t kStrideGradOutQTTransposed = static_cast<int64_t>(kNH) * kD;
constexpr int64_t kStrideGradOutKHTransposed = kD;
constexpr int64_t kStrideGradOutKTTransposed = static_cast<int64_t>(kNKVH) * kD;
constexpr int64_t kStrideGradOutVHTransposed = kD;
constexpr int64_t kStrideGradOutVTTransposed = static_cast<int64_t>(kNKVH) * kD;

// Total features in the combined tensor (with logit bias)
constexpr int64_t kTotalE_f(bool has_logit_bias)
{
  return kTotalH * kD + (has_logit_bias ? kTotalH * kDa : 0);
}

// Stride of the batch dimension for the combined tensor, given logit bias flag
constexpr int64_t kStrideCombinedB_f(bool has_logit_bias)
{
  return static_cast<int64_t>(kT) * kTotalE_f(has_logit_bias);
}

// -------------------------------------------------------------------------
// Dispatch macros (R_THREADS / R_ITEMS)
// -------------------------------------------------------------------------
#define DISPATCH_R_THREADS(R_THREADS_VAL, ...)                    \
  if constexpr (false)                                            \
  {                                                               \
  }                                                               \
  else if (R_THREADS_VAL == 2)                                    \
  {                                                               \
    constexpr int kRThreads = 2;                                  \
    __VA_ARGS__;                                                  \
  }                                                               \
  else if (R_THREADS_VAL == 8)                                    \
  {                                                               \
    constexpr int kRThreads = 8;                                  \
    __VA_ARGS__;                                                  \
  }                                                               \
  else if (R_THREADS_VAL == 32)                                   \
  {                                                               \
    constexpr int kRThreads = 32;                                 \
    __VA_ARGS__;                                                  \
  }                                                               \
  else                                                            \
  {                                                               \
    TORCH_CHECK(false, "Unsupported r_threads: ", R_THREADS_VAL); \
  }

#define DISPATCH_R_ITEMS(R_ITEMS_VAL, ...)                    \
  if constexpr (false)                                        \
  {                                                           \
  }                                                           \
  else if (R_ITEMS_VAL == 1)                                  \
  {                                                           \
    constexpr int kRItems = 1;                                \
    __VA_ARGS__;                                              \
  }                                                           \
  else if (R_ITEMS_VAL == 4)                                  \
  {                                                           \
    constexpr int kRItems = 4;                                \
    __VA_ARGS__;                                              \
  }                                                           \
  else if (R_ITEMS_VAL == 8)                                  \
  {                                                           \
    constexpr int kRItems = 8;                                \
    __VA_ARGS__;                                              \
  }                                                           \
  else                                                        \
  {                                                           \
    TORCH_CHECK(false, "Unsupported r_items: ", R_ITEMS_VAL); \
  }

// -------------------------------------------------------------------------
// Common helper functions
// -------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_f32(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

// -------------------------------------------------------------------------
// Shared memory size caps
// -------------------------------------------------------------------------
#ifndef LTV_FUSED_CONCAT_HM_SEQUENTIAL_MAX_SMEM_BYTES
#define LTV_FUSED_CONCAT_HM_SEQUENTIAL_MAX_SMEM_BYTES (48 * 1024)
#endif

#ifndef LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES
#define LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES (48 * 1024)
#endif

// -------------------------------------------------------------------------
// Generic stride‑mismatch warning (all expected strides passed as arguments)
// -------------------------------------------------------------------------
static inline void warn_specialized_backward_stride_mismatch(
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t s_hq_b, int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d,
    int64_t s_hk_b, int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d,
    int64_t s_hv_b, int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d,
    int64_t exp_sc_b, int64_t exp_sc_t, int64_t exp_sc_e,
    int64_t exp_out_qb, int64_t exp_out_qh, int64_t exp_out_qt, int64_t exp_out_qd,
    int64_t exp_out_kb, int64_t exp_out_kh, int64_t exp_out_kt, int64_t exp_out_kd,
    int64_t exp_out_vb, int64_t exp_out_vh, int64_t exp_out_vt, int64_t exp_out_vd,
    int64_t exp_gq_b, int64_t exp_gq_h, int64_t exp_gq_t, int64_t exp_gq_d,
    int64_t exp_gk_b, int64_t exp_gk_h, int64_t exp_gk_t, int64_t exp_gk_d,
    int64_t exp_gv_b, int64_t exp_gv_h, int64_t exp_gv_t, int64_t exp_gv_d)
{
  TORCH_WARN(
      "ltv_fused_concat_head_major_register_cast_scan_cuda_backward "
      "matched the specialized shape but not the specialized strides. "
      "Falling back to the generic kernel. ",
      "combined.stride=(", sc_in_b, ", ", sc_in_t, ", ", sc_in_e, "), ",
      "expected=(", exp_sc_b, ", ", exp_sc_t, ", ", exp_sc_e, "); ",
      "grad_combined.stride=(", sc_out_b, ", ", sc_out_t, ", ", sc_out_e, "), ",
      "expected=(", exp_sc_b, ", ", exp_sc_t, ", ", exp_sc_e, "); ",
      "out_q.stride=(", s_hq_b, ", ", s_hq_h, ", ", s_hq_t, ", ", s_hq_d, "), ",
      "expected=(", exp_out_qb, ", ", exp_out_qh, ", ", exp_out_qt, ", ", exp_out_qd, "); ",
      "out_k.stride=(", s_hk_b, ", ", s_hk_h, ", ", s_hk_t, ", ", s_hk_d, "), ",
      "expected=(", exp_out_kb, ", ", exp_out_kh, ", ", exp_out_kt, ", ", exp_out_kd, "); ",
      "out_v.stride=(", s_hv_b, ", ", s_hv_h, ", ", s_hv_t, ", ", s_hv_d, "), ",
      "expected=(", exp_out_vb, ", ", exp_out_vh, ", ", exp_out_vt, ", ", exp_out_vd, "); ",
      "grad_out_q.stride=(", s_gq_b, ", ", s_gq_h, ", ", s_gq_t, ", ", s_gq_d, "), ",
      "expected=(", exp_gq_b, ", ", exp_gq_h, ", ", exp_gq_t, ", ", exp_gq_d, "); ",
      "grad_out_k.stride=(", s_gk_b, ", ", s_gk_h, ", ", s_gk_t, ", ", s_gk_d, "), ",
      "expected=(", exp_gk_b, ", ", exp_gk_h, ", ", exp_gk_t, ", ", exp_gk_d, "); ",
      "grad_out_v.stride=(", s_gv_b, ", ", s_gv_h, ", ", s_gv_t, ", ", s_gv_d, "), ",
      "expected=(", exp_gv_b, ", ", exp_gv_h, ", ", exp_gv_t, ", ", exp_gv_d, ")");
}
