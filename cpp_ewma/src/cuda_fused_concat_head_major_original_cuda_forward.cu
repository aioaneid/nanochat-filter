#include "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Forward device function (simple copy, no scan)
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems>
__device__ __forceinline__ void
cuda_fused_concat_head_major_original_forward_device(
    const T_in *__restrict__ combined, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v, int64_t sc_b,
    int64_t sc_t, int64_t sc_e, int T_seq, int NH, int NKVH, int Da, int r)
{
  const int tid = threadIdx.x;
  const int lane_x = tid % kScanThreads;
  const int lane_y = tid / kScanThreads;

  const unsigned int b_idx = blockIdx.x;
  const unsigned int h_glob = blockIdx.y;

  // Grid z-dimension handles da_idx and parallelizes r_chunks
  const unsigned int da_idx = blockIdx.z % Da;
  const int r_chunk = blockIdx.z / Da;

  const unsigned int D = Da * r;
  // Original layout: NO logit channels → head stride = D
  const unsigned int head_stride = D;

  T_out *x_out;
  unsigned int h_loc, h_comp_total;

  if (h_glob < NH)
  {
    x_out = out_q;
    h_loc = h_glob;
    h_comp_total = NH;
  }
  else if (h_glob < NH + NKVH)
  {
    x_out = out_k;
    h_loc = h_glob - NH;
    h_comp_total = NKVH;
  }
  else
  {
    x_out = out_v;
    h_loc = h_glob - (NH + NKVH);
    h_comp_total = NKVH;
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

  // Base indices
  const int64_t val_r_stride = (int64_t)r_stride * sc_e;
  const int64_t val_base_idx = b_offset_combined + (int64_t)val_base * sc_e +
                               (int64_t)lane_x * sc_t +
                               (int64_t)(lane_y * kRItems) * sc_e;
  int64_t val_idx_t = val_base_idx + r_chunk * val_r_stride;
  const int64_t val_t_stride = (int64_t)kScanThreads * sc_t;

  const int64_t out_base_idx = (int64_t)b_idx * (h_comp_total * T_seq * D) +
                               (int64_t)h_loc * (T_seq * D) + (da_idx * r) +
                               (int64_t)lane_x * D + (lane_y * kRItems);
  int64_t out_idx_t = out_base_idx + r_chunk * r_stride;
  const int64_t out_t_stride = (int64_t)kScanThreads * D;

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
        T_out val = (T_out)combined[val_idx_item];
        int64_t out_idx_item = out_idx_t + item;
        x_out[out_idx_item] = val;
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
          T_out val = (T_out)combined[val_idx_item];
          int64_t out_idx_item = out_idx_t + item;
          x_out[out_idx_item] = val;
        }
        val_idx_item += sc_e;
      }
    }
  }
}

// -------------------------------------------------------------------------
// Generic forward kernel
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems>
__global__ void cuda_fused_concat_head_major_original_forward_kernel(
    const T_in *__restrict__ combined, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v, int64_t sc_b,
    int64_t sc_t, int64_t sc_e, int T_seq, int NH, int NKVH, int Da, int r)
{
  cuda_fused_concat_head_major_original_forward_device<
      T_in, T_out, kScanThreads, kRThreads, kRItems>(
      combined, out_q, out_k, out_v, sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
}

// -------------------------------------------------------------------------
// Constant‑strides forward kernel
// -------------------------------------------------------------------------
template <typename T_in, typename T_out, int kScanThreads, int kRThreads,
          int kRItems, int T_const, int NH_const, int NKVH_const, int Da_const,
          int r_const, int64_t sc_b_const, int64_t sc_t_const,
          int64_t sc_e_const>
__global__ void cuda_fused_concat_head_major_original_forward_kernel_const(
    const T_in *__restrict__ combined, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
  cuda_fused_concat_head_major_original_forward_device<
      T_in, T_out, kScanThreads, kRThreads, kRItems>(
      combined, out_q, out_k, out_v, sc_b_const, sc_t_const, sc_e_const,
      T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Dispatch inner – no T_compute
// -------------------------------------------------------------------------
template <int kScanThreads, int kRThreads, int kRItems>
void forward_dispatch_inner(torch::Tensor combined, torch::Tensor out_q,
                            torch::Tensor out_k, torch::Tensor out_v, int T,
                            int NH, int NKVH, int Da, int r, dim3 grid,
                            cudaStream_t stream, at::ScalarType dtype,
                            int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
  dim3 block(kScanThreads * kRThreads);

  if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
  {
    // Use the same T‑contiguous strides as the blelloch kernel
    constexpr int64_t kExpectedStrideT = 1;
    constexpr int64_t kExpectedStrideE = kT;
    if (sc_b == kStrideCombinedB_f(false) && sc_t == kExpectedStrideT &&
        sc_e == kExpectedStrideE)
    {
      AT_DISPATCH_SWITCH(
          dtype, "cuda_fused_concat_head_major_original_forward_const",
          AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                           {
            cuda_fused_concat_head_major_original_forward_kernel_const<
                float, float, kScanThreads, kRThreads, kRItems, kT, kNH, kNKVH,
                kDa, kR, kStrideCombinedB_f(false), 1, kT>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<float>(), out_q.data_ptr<float>(),
                    out_k.data_ptr<float>(), out_v.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                 {
            cuda_fused_concat_head_major_original_forward_kernel_const<
                at::BFloat16, at::BFloat16, kScanThreads, kRThreads, kRItems,
                kT, kNH, kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT>
                <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(),
                                             out_q.data_ptr<at::BFloat16>(),
                                             out_k.data_ptr<at::BFloat16>(),
                                             out_v.data_ptr<at::BFloat16>());
            C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::Double, [&]
                                                                       {
            cuda_fused_concat_head_major_original_forward_kernel_const<
                double, double, kScanThreads, kRThreads, kRItems, kT, kNH,
                kNKVH, kDa, kR, kStrideCombinedB_f(false), 1, kT>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<double>(), out_q.data_ptr<double>(),
                    out_k.data_ptr<double>(), out_v.data_ptr<double>());
            C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
      return;
    }
    TORCH_WARN_ONCE(
        "cuda_fused_concat_head_major_original_forward: "
        "matched specialized shape but not the specialized strides. "
        "Falling back to generic kernel. "
        "combined.stride=(",
        sc_b, ", ", sc_t, ", ", sc_e,
        "), "
        "expected=(",
        kStrideCombinedB_f(false), ", ", kExpectedStrideT, ", ",
        kExpectedStrideE, ")");
  }

  AT_DISPATCH_SWITCH(
      dtype, "cuda_fused_concat_head_major_original_forward",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                       {
        cuda_fused_concat_head_major_original_forward_kernel<
            float, float, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<float>(), out_q.data_ptr<float>(),
                out_k.data_ptr<float>(), out_v.data_ptr<float>(), sc_b, sc_t,
                sc_e, T, NH, NKVH, Da, r);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                             {
        cuda_fused_concat_head_major_original_forward_kernel<
            at::BFloat16, at::BFloat16, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(),
                                         out_q.data_ptr<at::BFloat16>(),
                                         out_k.data_ptr<at::BFloat16>(),
                                         out_v.data_ptr<at::BFloat16>(), sc_b,
                                         sc_t, sc_e, T, NH, NKVH, Da, r);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::Double, [&]
                                                                   {
        cuda_fused_concat_head_major_original_forward_kernel<
            double, double, kScanThreads, kRThreads, kRItems>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<double>(), out_q.data_ptr<double>(),
                out_k.data_ptr<double>(), out_v.data_ptr<double>(), sc_b, sc_t,
                sc_e, T, NH, NKVH, Da, r);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
}

// -------------------------------------------------------------------------
// Outer dispatch – no T_compute
// -------------------------------------------------------------------------
template <int kScanThreads>
void forward_dispatch_outer(torch::Tensor combined, torch::Tensor out_q,
                            torch::Tensor out_k, torch::Tensor out_v, int T,
                            int NH, int NKVH, int Da, int r, int r_threads,
                            int r_items, dim3 grid, cudaStream_t stream,
                            at::ScalarType dtype, int64_t sc_b, int64_t sc_t,
                            int64_t sc_e)
{
  DISPATCH_R_THREADS(r_threads, {
    DISPATCH_R_ITEMS(r_items, {
      forward_dispatch_inner<kScanThreads, kRThreads, kRItems>(
          combined, out_q, out_k, out_v, T, NH, NKVH, Da, r, grid, stream,
          dtype, sc_b, sc_t, sc_e);
    });
  });
}

// -------------------------------------------------------------------------
// Launcher
// -------------------------------------------------------------------------
void cuda_fused_concat_head_major_original_cuda_forward(
    torch::Tensor combined, torch::Tensor out_q, torch::Tensor out_k,
    torch::Tensor out_v, int B, int T, int NH, int NKVH, int Da, int r,
    int scan_threads, int r_threads, int r_items)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t sc_b = combined.stride(0);
  int64_t sc_t = combined.stride(1);
  int64_t sc_e = combined.stride(2);

  // Compute number of r_chunks for grid z dimension
  int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
  dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks);
  auto dtype = combined.scalar_type();

  DISPATCH_SCAN_THREADS(scan_threads, {
    forward_dispatch_outer<kScanThreads>(
        combined, out_q, out_k, out_v, T, NH, NKVH, Da, r, r_threads, r_items,
        grid, stream, dtype, sc_b, sc_t, sc_e);
  });
}
