#pragma once

#include "ltv_fused_concat_head_major_register_cast_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Layout constants for the sequential‑scan combined tensor
//   combined shape: (B, T, F)  contiguous  strides: (T*F, F, 1)
// -------------------------------------------------------------------------
constexpr int64_t kSeqStrideCombinedB = kT * kTotalE_f(true); // B stride
constexpr int64_t kSeqStrideCombinedT = kTotalE_f(true);      // T stride
constexpr int64_t kSeqStrideCombinedE = 1;                    // F stride
