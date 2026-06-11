#pragma once

#include "ltv_fused_concat_head_major_register_cast_scan_cuda_common.cuh"
#include <cub/cub.cuh>

// -------------------------------------------------------------------------
// Dispatch macro for SCAN_THREADS
// -------------------------------------------------------------------------
#define DISPATCH_SCAN_THREADS(SCAN_THREADS_VAL, ...)                    \
  if constexpr (false)                                                  \
  {                                                                     \
  }                                                                     \
  else if (SCAN_THREADS_VAL == 1)                                       \
  {                                                                     \
    constexpr int kScanThreads = 1;                                     \
    __VA_ARGS__;                                                        \
  }                                                                     \
  else if (SCAN_THREADS_VAL == 8)                                       \
  {                                                                     \
    constexpr int kScanThreads = 8;                                     \
    __VA_ARGS__;                                                        \
  }                                                                     \
  else if (SCAN_THREADS_VAL == 32)                                      \
  {                                                                     \
    constexpr int kScanThreads = 32;                                    \
    __VA_ARGS__;                                                        \
  }                                                                     \
  else                                                                  \
  {                                                                     \
    TORCH_CHECK(false, "Unsupported scan_threads: ", SCAN_THREADS_VAL); \
  }

// -------------------------------------------------------------------------
// Layout constants for the Blelloch‑scan transposed combined tensor
//   combined shape: (B, F, T)   strides: (F*T, T, 1)
// -------------------------------------------------------------------------
constexpr int64_t kStrideCombinedB = kStrideCombinedB_f(true); // uses has_logit_bias=true
constexpr int64_t kStrideCombinedT = 1;
constexpr int64_t kStrideCombinedE = kT;

// -------------------------------------------------------------------------
// Warp‑scan helpers (Blelloch only)
// -------------------------------------------------------------------------
template <typename T, int RItems>
struct AffineState
{
  T a;
  T x[RItems];
};

template <typename T, int RItems>
struct AffineScanOp
{
  __device__ __forceinline__ AffineState<T, RItems>
  operator()(const AffineState<T, RItems> &left,
             const AffineState<T, RItems> &right) const
  {
    AffineState<T, RItems> res;
    res.a = right.a * left.a;
#pragma unroll
    for (int i = 0; i < RItems; ++i)
      res.x[i] = right.a * left.x[i] + right.x[i];
    return res;
  }
};

template <typename T, int RItems>
__device__ __forceinline__ AffineState<T, RItems> affine_identity()
{
  AffineState<T, RItems> res;
  res.a = T(1);
#pragma unroll
  for (int i = 0; i < RItems; ++i)
    res.x[i] = T(0);
  return res;
}
