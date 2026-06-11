#include "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Backward device function (simple copy, no scan)
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems>
__device__ __forceinline__ void
cuda_fused_concat_head_major_original_backward_device(
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    int64_t sc_b, int64_t sc_t, int64_t sc_e, int T_seq, int NH, int NKVH,
    int Da, int r, int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t,
    int64_t s_gq_d, int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t,
    int64_t s_gk_d, int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t,
    int64_t s_gv_d)
{
  const int tid = threadIdx.x;
  const int lane_x = tid % kScanThreads;
  const int lane_y = tid / kScanThreads;

  const unsigned int b_idx = blockIdx.x;
  const unsigned int h_glob = blockIdx.y;

  const unsigned int da_idx = blockIdx.z % Da;
  const int r_chunk = blockIdx.z / Da;

  const unsigned int D = Da * r;
  // Original layout: NO logit channels → head stride = D
  const unsigned int head_stride = D;

  // Select the correct gradient tensor and its strides
  const T_out *grad_x_out;
  unsigned int h_loc;
  int64_t stride_b, stride_h, stride_t, stride_d;

  if (h_glob < NH)
  {
    grad_x_out = grad_hq;
    h_loc = h_glob;
    stride_b = s_gq_b;
    stride_h = s_gq_h;
    stride_t = s_gq_t;
    stride_d = s_gq_d;
  }
  else if (h_glob < NH + NKVH)
  {
    grad_x_out = grad_hk;
    h_loc = h_glob - NH;
    stride_b = s_gk_b;
    stride_h = s_gk_h;
    stride_t = s_gk_t;
    stride_d = s_gk_d;
  }
  else
  {
    grad_x_out = grad_hv;
    h_loc = h_glob - (NH + NKVH);
    stride_b = s_gv_b;
    stride_h = s_gv_h;
    stride_t = s_gv_t;
    stride_d = s_gv_d;
  }

  const int num_r_chunks =
      (r + kRThreads * kRItems - 1) / (kRThreads * kRItems);
  if (r_chunk >= num_r_chunks)
    return;

  const int64_t b_offset_combined = (int64_t)b_idx * sc_b;
  const unsigned int val_base = h_glob * head_stride + da_idx * r;

  const int full_t_chunks = T_seq / kScanThreads;
  const bool has_tail_chunk = (T_seq % kScanThreads) != 0;

  const int r_stride = kRThreads * kRItems;

  // Compute r start for this block's chunk
  const int r_start = lane_y * kRItems + r_chunk * r_stride;

  // Base indices for the combined tensor (destination)
  const int64_t val_r_stride = (int64_t)r_stride * sc_e;
  const int64_t val_base_idx = b_offset_combined + (int64_t)val_base * sc_e +
                               (int64_t)lane_x * sc_t +
                               (int64_t)(lane_y * kRItems) * sc_e;
  int64_t val_idx_t = val_base_idx + r_chunk * val_r_stride;
  const int64_t val_t_stride = (int64_t)kScanThreads * sc_t;

  // Base index for the gradient tensor (source) using the selected strides
  const int64_t grad_base = (int64_t)b_idx * stride_b +
                            (int64_t)h_loc * stride_h +
                            (int64_t)(da_idx * r) * stride_d;
  int64_t out_idx_t = grad_base + (int64_t)lane_x * stride_t +
                      (int64_t)(lane_y * kRItems) * stride_d +
                      (int64_t)r_chunk * r_stride * stride_d;
  const int64_t out_t_stride = (int64_t)kScanThreads * stride_t;

  // Validity mask for this chunk's r items
  bool valid_r[kRItems];
#pragma unroll
  for (int item = 0; item < kRItems; ++item)
  {
    valid_r[item] = (r_start + item < r);
  }

  int t_global = lane_x;

  // ==================== FAST PATH: Full T-Chunks ====================
  for (int t_chunk = 0; t_chunk < full_t_chunks; ++t_chunk)
  {
    int64_t val_idx_item = val_idx_t;
#pragma unroll
    for (int item = 0; item < kRItems; ++item)
    {
      if (valid_r[item])
      {
        int64_t out_idx_item = out_idx_t + (int64_t)item * stride_d;
        T_out g = grad_x_out[out_idx_item];
        grad_combined[val_idx_item] = g;
      }
      val_idx_item += sc_e;
    }

    t_global += kScanThreads;
    val_idx_t += val_t_stride;
    out_idx_t += out_t_stride;
  }

  // ==================== SLOW PATH: Tail T-Chunk ====================
  if (has_tail_chunk)
  {
    bool valid_t = (t_global < T_seq);
    if (valid_t)
    {
      int64_t val_idx_item = val_idx_t;
#pragma unroll
      for (int item = 0; item < kRItems; ++item)
      {
        if (valid_r[item])
        {
          int64_t out_idx_item = out_idx_t + (int64_t)item * stride_d;
          T_out g = grad_x_out[out_idx_item];
          grad_combined[val_idx_item] = g;
        }
        val_idx_item += sc_e;
      }
    }
  }
}

// -------------------------------------------------------------------------
// Generic backward kernel
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems>
__global__ void cuda_fused_concat_head_major_original_backward_kernel(
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    int64_t sc_b, int64_t sc_t, int64_t sc_e, int T_seq, int NH, int NKVH,
    int Da, int r, int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t,
    int64_t s_gq_d, int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t,
    int64_t s_gk_d, int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t,
    int64_t s_gv_d)
{
  cuda_fused_concat_head_major_original_backward_device<
      T_in, T_out, kScanThreads, kRThreads, kRItems>(
      grad_hq, grad_hk, grad_hv, grad_combined, sc_b, sc_t, sc_e, T_seq, NH,
      NKVH, Da, r, s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t,
      s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d);
}

// -------------------------------------------------------------------------
// Constant‑strides backward kernel – supports both standard and transposed
// gradient layouts (the same layouts the blelloch kernel supports).
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems, int T_const, int NH_const, int NKVH_const, int Da_const,
          int r_const, int64_t sc_b_const, int64_t sc_t_const,
          int64_t sc_e_const, int64_t s_gq_b_const, int64_t s_gq_h_const,
          int64_t s_gq_t_const, int64_t s_gq_d_const, int64_t s_gk_b_const,
          int64_t s_gk_h_const, int64_t s_gk_t_const, int64_t s_gk_d_const,
          int64_t s_gv_b_const, int64_t s_gv_h_const, int64_t s_gv_t_const,
          int64_t s_gv_d_const>
__global__ void cuda_fused_concat_head_major_original_backward_kernel_const(
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined)
{
  cuda_fused_concat_head_major_original_backward_device<
      T_in, T_out, kScanThreads, kRThreads, kRItems>(
      grad_hq, grad_hk, grad_hv, grad_combined, sc_b_const, sc_t_const,
      sc_e_const, T_const, NH_const, NKVH_const, Da_const, r_const,
      s_gq_b_const, s_gq_h_const, s_gq_t_const, s_gq_d_const, s_gk_b_const,
      s_gk_h_const, s_gk_t_const, s_gk_d_const, s_gv_b_const, s_gv_h_const,
      s_gv_t_const, s_gv_d_const);
}

// -------------------------------------------------------------------------
// Dispatch inner – NO T_compute
// -------------------------------------------------------------------------
template <int kScanThreads, int kRThreads, int kRItems>
void backward_dispatch_inner(
    torch::Tensor combined, torch::Tensor grad_hq,
    torch::Tensor grad_hk, torch::Tensor grad_hv,
    torch::Tensor grad_combined,
    int T, int NH, int NKVH, int Da, int r,
    dim3 grid, cudaStream_t stream, at::ScalarType dtype,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  dim3 block(kScanThreads * kRThreads);

  // --------------------------------------------------------------------
  // Constant‑stride fast path: same layout as blelloch backward,
  // minus logit channels.
  // Requires:
  //   - T‑contiguous combined (stride_t = 1, stride_e = kT)
  //   - Gradient tensors in canonical [B,H,T,D] layout *or*
  //     transposed Q/K layout (as supported by blelloch).
  // --------------------------------------------------------------------
  if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
  {
    constexpr int64_t kExpectedStrideB = kStrideCombinedB_f(false);
    constexpr int64_t kExpectedStrideT = 1;
    constexpr int64_t kExpectedStrideE = kT;

    bool combined_strides_ok = (sc_b == kExpectedStrideB) &&
                               (sc_t == kExpectedStrideT) &&
                               (sc_e == kExpectedStrideE);

    // Canonical [B,H,T,D] layout
    bool grad_strides_standard =
        (s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQH &&
         s_gq_t == kStrideGradOutQT && s_gq_d == kStrideGradOutQD) &&
        (s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKH &&
         s_gk_t == kStrideGradOutKT && s_gk_d == kStrideGradOutKD) &&
        (s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH &&
         s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD);

    // Transposed Q/K gradient layout (H and T strides swapped)
    bool grad_strides_transposed =
        (s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQHTransposed &&
         s_gq_t == kStrideGradOutQTTransposed && s_gq_d == kStrideGradOutQD) &&
        (s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKHTransposed &&
         s_gk_t == kStrideGradOutKTTransposed && s_gk_d == kStrideGradOutKD) &&
        (s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH &&
         s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD);

    if (combined_strides_ok)
    {
      if (grad_strides_standard)
      {
        AT_DISPATCH_SWITCH(
            dtype, "cuda_fused_concat_head_major_backward_const_standard",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  float, float, kScanThreads, kRThreads, kRItems, kT, kNH,
                  kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT,
                  kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH,
                  kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB,
                  kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, 0, stream>>>(grad_hq.data_ptr<float>(),
                                               grad_hk.data_ptr<float>(),
                                               grad_hv.data_ptr<float>(),
                                               grad_combined.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                   {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  at::BFloat16, at::BFloat16, kScanThreads, kRThreads, kRItems,
                  kT, kNH, kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT,
                  kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH,
                  kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB,
                  kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, 0, stream>>>(
                      grad_hq.data_ptr<at::BFloat16>(),
                      grad_hk.data_ptr<at::BFloat16>(),
                      grad_hv.data_ptr<at::BFloat16>(),
                      grad_combined.data_ptr<at::BFloat16>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::Double, [&]
                                                                         {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  double, double, kScanThreads, kRThreads, kRItems, kT, kNH,
                  kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT,
                  kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH,
                  kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB,
                  kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, 0, stream>>>(
                      grad_hq.data_ptr<double>(), grad_hk.data_ptr<double>(),
                      grad_hv.data_ptr<double>(),
                      grad_combined.data_ptr<double>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
        return;
      }

      if (grad_strides_transposed)
      {
        AT_DISPATCH_SWITCH(
            dtype, "cuda_fused_concat_head_major_backward_const_transposed",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  float, float, kScanThreads, kRThreads, kRItems, kT, kNH,
                  kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQHTransposed,
                  kStrideGradOutQTTransposed, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKHTransposed,
                  kStrideGradOutKTTransposed, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD><<<grid, block, 0, stream>>>(
                  grad_hq.data_ptr<float>(), grad_hk.data_ptr<float>(),
                  grad_hv.data_ptr<float>(), grad_combined.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                   {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  at::BFloat16, at::BFloat16, kScanThreads, kRThreads, kRItems,
                  kT, kNH, kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQHTransposed,
                  kStrideGradOutQTTransposed, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKHTransposed,
                  kStrideGradOutKTTransposed, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD><<<grid, block, 0, stream>>>(
                  grad_hq.data_ptr<at::BFloat16>(),
                  grad_hk.data_ptr<at::BFloat16>(),
                  grad_hv.data_ptr<at::BFloat16>(),
                  grad_combined.data_ptr<at::BFloat16>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::Double, [&]
                                                                         {
              cuda_fused_concat_head_major_original_backward_kernel_const<
                  double, double, kScanThreads, kRThreads, kRItems, kT, kNH,
                  kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT,
                  kStrideGradOutQB, kStrideGradOutQHTransposed,
                  kStrideGradOutQTTransposed, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKHTransposed,
                  kStrideGradOutKTTransposed, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD><<<grid, block, 0, stream>>>(
                  grad_hq.data_ptr<double>(), grad_hk.data_ptr<double>(),
                  grad_hv.data_ptr<double>(), grad_combined.data_ptr<double>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
        return;
      }
    }

    // Provide detailed warning if shapes match but strides don't.
    TORCH_WARN_ONCE(
        "cuda_fused_concat_head_major_original_backward: "
        "matched specialized shape but not the specialized strides. "
        "Falling back to generic kernel.\n"
        "  combined.stride=(",
        sc_b, ", ", sc_t, ", ", sc_e,
        "), expected=(", kExpectedStrideB, ", ", kExpectedStrideT, ", ",
        kExpectedStrideE, ")\n"
                          "  grad_hq.stride=(",
        s_gq_b, ", ", s_gq_h, ", ", s_gq_t, ", ",
        s_gq_d, "), expected standard=(", kStrideGradOutQB, ", ",
        kStrideGradOutQH, ", ", kStrideGradOutQT, ", ", kStrideGradOutQD,
        ") or transposed=(", kStrideGradOutQB, ", ",
        kStrideGradOutQHTransposed, ", ", kStrideGradOutQTTransposed, ", ",
        kStrideGradOutQD, ")\n"
                          "  grad_hk.stride=(",
        s_gk_b, ", ", s_gk_h, ", ", s_gk_t, ", ",
        s_gk_d, "), expected standard=(", kStrideGradOutKB, ", ",
        kStrideGradOutKH, ", ", kStrideGradOutKT, ", ", kStrideGradOutKD,
        ") or transposed=(", kStrideGradOutKB, ", ",
        kStrideGradOutKHTransposed, ", ", kStrideGradOutKTTransposed, ", ",
        kStrideGradOutKD, ")\n"
                          "  grad_hv.stride=(",
        s_gv_b, ", ", s_gv_h, ", ", s_gv_t, ", ",
        s_gv_d, "), expected=(", kStrideGradOutVB, ", ", kStrideGradOutVH,
        ", ", kStrideGradOutVT, ", ", kStrideGradOutVD, ")");
  }

  // --------------------------------------------------------------------
  // Generic fallback
  // --------------------------------------------------------------------
  AT_DISPATCH_SWITCH(
      dtype, "cuda_fused_concat_head_major_backward",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                       {
        cuda_fused_concat_head_major_original_backward_kernel<
            float, float, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(
                grad_hq.data_ptr<float>(), grad_hk.data_ptr<float>(),
                grad_hv.data_ptr<float>(), grad_combined.data_ptr<float>(),
                sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, s_gq_b, s_gq_h, s_gq_t,
                s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t,
                s_gv_d);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                             {
        cuda_fused_concat_head_major_original_backward_kernel<
            at::BFloat16, at::BFloat16, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(
                grad_hq.data_ptr<at::BFloat16>(),
                grad_hk.data_ptr<at::BFloat16>(),
                grad_hv.data_ptr<at::BFloat16>(),
                grad_combined.data_ptr<at::BFloat16>(), sc_b, sc_t, sc_e, T, NH,
                NKVH, Da, r, s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h,
                s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::Double, [&]
                                                                   {
        cuda_fused_concat_head_major_original_backward_kernel<
            double, double, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(
                grad_hq.data_ptr<double>(), grad_hk.data_ptr<double>(),
                grad_hv.data_ptr<double>(), grad_combined.data_ptr<double>(),
                sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, s_gq_b, s_gq_h, s_gq_t,
                s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t,
                s_gv_d);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
}

// -------------------------------------------------------------------------
// Outer dispatch – NO T_compute
// -------------------------------------------------------------------------
template <int kScanThreads>
void backward_dispatch_outer(
    torch::Tensor combined, torch::Tensor grad_hq, torch::Tensor grad_hk,
    torch::Tensor grad_hv, torch::Tensor grad_combined, int T, int NH, int NKVH,
    int Da, int r, int r_threads, int r_items, dim3 grid, cudaStream_t stream,
    at::ScalarType dtype, int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  DISPATCH_R_THREADS(r_threads, {
    DISPATCH_R_ITEMS(r_items, {
      backward_dispatch_inner<kScanThreads, kRThreads, kRItems>(
          combined, grad_hq, grad_hk, grad_hv, grad_combined, T, NH, NKVH, Da,
          r, grid, stream, dtype, sc_b, sc_t, sc_e, s_gq_b, s_gq_h, s_gq_t,
          s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t,
          s_gv_d);
    });
  });
}

// -------------------------------------------------------------------------
// Launcher – with dtype safety checks
// -------------------------------------------------------------------------
void cuda_fused_concat_head_major_original_cuda_backward(
    torch::Tensor combined, torch::Tensor grad_hq, torch::Tensor grad_hk,
    torch::Tensor grad_hv, torch::Tensor grad_combined, int B, int T, int NH,
    int NKVH, int Da, int r, int scan_threads, int r_threads, int r_items)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t sc_b = combined.stride(0);
  int64_t sc_t = combined.stride(1);
  int64_t sc_e = combined.stride(2);

  int64_t s_gq_b = grad_hq.stride(0);
  int64_t s_gq_h = grad_hq.stride(1);
  int64_t s_gq_t = grad_hq.stride(2);
  int64_t s_gq_d = grad_hq.stride(3);
  int64_t s_gk_b = grad_hk.stride(0);
  int64_t s_gk_h = grad_hk.stride(1);
  int64_t s_gk_t = grad_hk.stride(2);
  int64_t s_gk_d = grad_hk.stride(3);
  int64_t s_gv_b = grad_hv.stride(0);
  int64_t s_gv_h = grad_hv.stride(1);
  int64_t s_gv_t = grad_hv.stride(2);
  int64_t s_gv_d = grad_hv.stride(3);

  int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
  dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks);

  auto dtype = combined.scalar_type();

  // Sanity: gradient dtypes must match combined
  TORCH_CHECK(grad_hq.scalar_type() == dtype,
              "grad_hq dtype mismatch");
  TORCH_CHECK(grad_hk.scalar_type() == dtype,
              "grad_hk dtype mismatch");
  TORCH_CHECK(grad_hv.scalar_type() == dtype,
              "grad_hv dtype mismatch");

  DISPATCH_SCAN_THREADS(scan_threads, {
    backward_dispatch_outer<kScanThreads>(
        combined, grad_hq, grad_hk, grad_hv, grad_combined, T, NH, NKVH, Da, r,
        r_threads, r_items, grid, stream, dtype, sc_b, sc_t, sc_e, s_gq_b,
        s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h,
        s_gv_t, s_gv_d);
  });
}
