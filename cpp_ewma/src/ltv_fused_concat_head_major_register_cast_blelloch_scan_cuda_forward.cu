#include "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Loop macro – expands to a #pragma unroll for over kRItems
// -------------------------------------------------------------------------
#define LTV_FORWARD_UNROLL_OVER_R_ITEMS(BODY_MACRO)            \
  _Pragma("unroll") for (int item = 0; item < kRItems; ++item) \
  {                                                            \
    BODY_MACRO                                                 \
  }

// -------------------------------------------------------------------------
// Strength‑reduced pointer setup macros (unchanged)
// -------------------------------------------------------------------------
#define LTV_FORWARD_SETUP_COMBINED_PTR(start_idx) \
  const T_in *ptr_combined = &combined[start_idx];

#define LTV_FORWARD_SETUP_OUT_PTR(start_idx) \
  T_out *ptr_out = &x_out[start_idx];

#define LTV_FORWARD_SETUP_INIT_PTR(start_idx) \
  const T_in *ptr_init = &inits[start_idx];

// -------------------------------------------------------------------------
// Strength‑reduced body macros (use pointers, advance by element stride)
// -------------------------------------------------------------------------
#define LTV_FORWARD_LOAD_COMBINED_PTR(sc_e)                  \
  thread_input.x[item] = alpha * (T_compute)(*ptr_combined); \
  ptr_combined += sc_e;

#define LTV_FORWARD_UPDATE_PTR                                         \
  *ptr_out = (T_out)(inc_a * running_x[item] + inclusive_out.x[item]); \
  running_x[item] = blk_a * running_x[item] + block_aggregate.x[item]; \
  ptr_out += 1; // <-- stride is 1

#define LTV_FORWARD_LOAD_INIT_PTR           \
  running_x[item] = (T_compute)(*ptr_init); \
  ptr_init += 1; // <-- stride is 1

// -------------------------------------------------------------------------
// Main forward device function macro – now with shared alpha cache
// -------------------------------------------------------------------------
#define LTV_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(T_in, T_compute, T_out,             \
                                                kScanThreads, kRThreads, kRItems,   \
                                                UseSigmoid,                         \
                                                combined, inits, logit_bias,        \
                                                out_q, out_k, out_v,                \
                                                sc_b, sc_t, sc_e,                   \
                                                T_seq, NH, NKVH, Da, r)             \
  do                                                                                \
  {                                                                                 \
    const int tid = threadIdx.x;                                                    \
    const int lane_x = tid % kScanThreads;                                          \
    const int lane_y = tid / kScanThreads;                                          \
                                                                                    \
    using ScanTemp = typename cub::WarpScan<AffineState<T_compute, kRItems>,        \
                                            kScanThreads>::TempStorage;             \
    __shared__ alignas(16) ScanTemp scan_temp[kRThreads];                           \
    /* Cache for sigmoid/alpha values – one per time step in the chunk */           \
    __shared__ T_compute alpha_cache[kScanThreads];                                 \
                                                                                    \
    const unsigned int b_idx = blockIdx.x;                                          \
    const unsigned int h_glob = blockIdx.y;                                         \
                                                                                    \
    const unsigned int da_idx = blockIdx.z % Da;                                    \
    const int r_chunk = blockIdx.z / Da;                                            \
                                                                                    \
    const unsigned int D = Da * r;                                                  \
    const unsigned int total_h = NH + 2 * NKVH;                                     \
    const unsigned int head_stride = D + Da;                                        \
                                                                                    \
    T_out *x_out;                                                                   \
    unsigned int h_loc, h_comp_total;                                               \
                                                                                    \
    if (h_glob < NH)                                                                \
    {                                                                               \
      x_out = out_q;                                                                \
      h_loc = h_glob;                                                               \
      h_comp_total = NH;                                                            \
    }                                                                               \
    else if (h_glob < NH + NKVH)                                                    \
    {                                                                               \
      x_out = out_k;                                                                \
      h_loc = h_glob - NH;                                                          \
      h_comp_total = NKVH;                                                          \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      x_out = out_v;                                                                \
      h_loc = h_glob - (NH + NKVH);                                                 \
      h_comp_total = NKVH;                                                          \
    }                                                                               \
                                                                                    \
    const int num_r_chunks =                                                        \
        (r + kRThreads * kRItems - 1) / (kRThreads * kRItems);                      \
    if (r_chunk >= num_r_chunks)                                                    \
      return;                                                                       \
                                                                                    \
    const int64_t b_offset_combined = (int64_t)b_idx * sc_b;                        \
    const unsigned int val_base = h_glob * head_stride + da_idx * r;                \
    const unsigned int logit_base = h_glob * head_stride + D + da_idx;              \
                                                                                    \
    const int full_t_chunks = T_seq / kScanThreads;                                 \
    const bool has_tail_chunk = (T_seq % kScanThreads) != 0;                        \
                                                                                    \
    const int r_stride = kRThreads * kRItems;                                       \
    const T_compute bias_val = logit_bias[h_glob * Da + da_idx];                    \
                                                                                    \
    const int r_start = lane_y * kRItems + r_chunk * r_stride;                      \
    const bool all_r_valid = (r_start + kRItems <= r);                              \
                                                                                    \
    const int64_t init_base =                                                       \
        (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + (da_idx * r);        \
    int64_t init_idx_item = init_base + r_start;                                    \
                                                                                    \
    T_compute running_x[kRItems];                                                   \
    LTV_FORWARD_SETUP_INIT_PTR(init_idx_item)                                       \
    if (all_r_valid)                                                                \
    {                                                                               \
      LTV_FORWARD_UNROLL_OVER_R_ITEMS(LTV_FORWARD_LOAD_INIT_PTR)                    \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                              \
          if (r_start + item < r) {                                                 \
            LTV_FORWARD_LOAD_INIT_PTR                                               \
          } else {                                                                  \
            running_x[item] = T_compute(0);                                         \
            ptr_init += 1;                                                          \
          })                                                                        \
    }                                                                               \
                                                                                    \
    const int64_t logit_base_idx =                                                  \
        b_offset_combined + (int64_t)logit_base * sc_e + (int64_t)lane_x * sc_t;    \
    const int64_t logit_t_stride = (int64_t)kScanThreads * sc_t;                    \
                                                                                    \
    const int64_t val_r_stride = (int64_t)r_stride * sc_e;                          \
    const int64_t val_base_idx = b_offset_combined + (int64_t)val_base * sc_e +     \
                                 (int64_t)lane_x * sc_t +                           \
                                 (int64_t)(lane_y * kRItems) * sc_e;                \
    int64_t val_idx_t = val_base_idx + r_chunk * val_r_stride;                      \
    const int64_t val_t_stride = (int64_t)kScanThreads * sc_t;                      \
                                                                                    \
    const int64_t out_base_idx = (int64_t)b_idx * (h_comp_total * T_seq * D) +      \
                                 (int64_t)h_loc * (T_seq * D) + (da_idx * r) +      \
                                 (int64_t)lane_x * D + (lane_y * kRItems);          \
    int64_t out_idx_t = out_base_idx + r_chunk * r_stride;                          \
    const int64_t out_t_stride = (int64_t)kScanThreads * D;                         \
                                                                                    \
    int t_global = lane_x;                                                          \
    int64_t logit_idx_t = logit_base_idx;                                           \
                                                                                    \
    cub::WarpScan<AffineState<T_compute, kRItems>, kScanThreads> warp_scan(         \
        scan_temp[lane_y]);                                                         \
                                                                                    \
    /* ==================== FAST PATH: Full T-Chunks ==================== */        \
    for (int t_chunk = 0; t_chunk < full_t_chunks; ++t_chunk)                       \
    {                                                                               \
      /* ---- One warp computes alpha for the whole chunk ---- */                   \
      T_compute alpha;                                                              \
      if (lane_y == 0)                                                              \
      {                                                                             \
        T_compute l_raw = (T_compute)combined[logit_idx_t];                         \
        l_raw += bias_val;                                                          \
        alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                            \
        alpha_cache[lane_x] = alpha;                                                \
      }                                                                             \
      __syncthreads();                                                              \
      if (lane_y != 0)                                                              \
      {                                                                             \
        alpha = alpha_cache[lane_x];                                                \
      }                                                                             \
                                                                                    \
      AffineState<T_compute, kRItems> thread_input;                                 \
      thread_input.a = T_compute(1) - alpha;                                        \
                                                                                    \
      LTV_FORWARD_SETUP_COMBINED_PTR(val_idx_t)                                     \
      if (all_r_valid)                                                              \
      {                                                                             \
        LTV_FORWARD_UNROLL_OVER_R_ITEMS(LTV_FORWARD_LOAD_COMBINED_PTR(sc_e))        \
      }                                                                             \
      else                                                                          \
      {                                                                             \
        LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                            \
            if (r_start + item < r) {                                               \
              LTV_FORWARD_LOAD_COMBINED_PTR(sc_e)                                   \
            } else {                                                                \
              ptr_combined += sc_e;                                                 \
              thread_input.x[item] = T_compute(0);                                  \
            })                                                                      \
      }                                                                             \
                                                                                    \
      AffineState<T_compute, kRItems> inclusive_out;                                \
      AffineState<T_compute, kRItems> block_aggregate;                              \
      warp_scan.InclusiveScan(thread_input, inclusive_out,                          \
                              AffineScanOp<T_compute, kRItems>(), block_aggregate); \
                                                                                    \
      T_compute inc_a = inclusive_out.a;                                            \
      T_compute blk_a = block_aggregate.a;                                          \
                                                                                    \
      LTV_FORWARD_SETUP_OUT_PTR(out_idx_t)                                          \
      if (all_r_valid)                                                              \
      {                                                                             \
        LTV_FORWARD_UNROLL_OVER_R_ITEMS(LTV_FORWARD_UPDATE_PTR)                     \
      }                                                                             \
      else                                                                          \
      {                                                                             \
        LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                            \
            if (r_start + item < r) {                                               \
              LTV_FORWARD_UPDATE_PTR                                                \
            } else {                                                                \
              ptr_out += 1;                                                         \
            })                                                                      \
      }                                                                             \
                                                                                    \
      t_global += kScanThreads;                                                     \
      logit_idx_t += logit_t_stride;                                                \
      val_idx_t += val_t_stride;                                                    \
      out_idx_t += out_t_stride;                                                    \
    }                                                                               \
                                                                                    \
    /* ==================== SLOW PATH: Tail T-Chunk ==================== */         \
    if (has_tail_chunk)                                                             \
    {                                                                               \
      bool valid_t = (t_global < T_seq);                                            \
                                                                                    \
      /* ---- One warp computes alpha (or writes 0 for invalid lanes) ---- */       \
      T_compute alpha;                                                              \
      if (lane_y == 0)                                                              \
      {                                                                             \
        if (valid_t)                                                                \
        {                                                                           \
          T_compute l_raw = (T_compute)combined[logit_idx_t];                       \
          l_raw += bias_val;                                                        \
          alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                          \
          alpha_cache[lane_x] = alpha;                                              \
        }                                                                           \
        else                                                                        \
        {                                                                           \
          alpha = T_compute(0);                                                     \
          alpha_cache[lane_x] = alpha;                                              \
        }                                                                           \
      }                                                                             \
      __syncthreads();                                                              \
      if (lane_y != 0)                                                              \
      {                                                                             \
        alpha = alpha_cache[lane_x];                                                \
      }                                                                             \
                                                                                    \
      AffineState<T_compute, kRItems> thread_input;                                 \
                                                                                    \
      if (valid_t)                                                                  \
      {                                                                             \
        thread_input.a = T_compute(1) - alpha;                                      \
                                                                                    \
        LTV_FORWARD_SETUP_COMBINED_PTR(val_idx_t)                                   \
        if (all_r_valid)                                                            \
        {                                                                           \
          LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                          \
              thread_input.x[item] = alpha * (T_compute)(*ptr_combined);            \
              ptr_combined += sc_e;)                                                \
        }                                                                           \
        else                                                                        \
        {                                                                           \
          LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                          \
              if (r_start + item < r) {                                             \
                thread_input.x[item] = alpha * (T_compute)(*ptr_combined);          \
                ptr_combined += sc_e;                                               \
              } else {                                                              \
                thread_input.x[item] = T_compute(0);                                \
                ptr_combined += sc_e;                                               \
              })                                                                    \
        }                                                                           \
      }                                                                             \
      else                                                                          \
      {                                                                             \
        thread_input.a = T_compute(1);                                              \
        LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                            \
            thread_input.x[item] = T_compute(0);)                                   \
      }                                                                             \
                                                                                    \
      AffineState<T_compute, kRItems> inclusive_out;                                \
      AffineState<T_compute, kRItems> block_aggregate;                              \
      warp_scan.InclusiveScan(thread_input, inclusive_out,                          \
                              AffineScanOp<T_compute, kRItems>(), block_aggregate); \
                                                                                    \
      if (valid_t)                                                                  \
      {                                                                             \
        T_compute inc_a = inclusive_out.a;                                          \
        T_compute blk_a = block_aggregate.a;                                        \
                                                                                    \
        LTV_FORWARD_SETUP_OUT_PTR(out_idx_t)                                        \
        if (all_r_valid)                                                            \
        {                                                                           \
          LTV_FORWARD_UNROLL_OVER_R_ITEMS(LTV_FORWARD_UPDATE_PTR)                   \
        }                                                                           \
        else                                                                        \
        {                                                                           \
          LTV_FORWARD_UNROLL_OVER_R_ITEMS(                                          \
              if (r_start + item < r) {                                             \
                LTV_FORWARD_UPDATE_PTR                                              \
              } else {                                                              \
                ptr_out += 1;                                                       \
              })                                                                    \
        }                                                                           \
      }                                                                             \
    }                                                                               \
  } while (0)

// -------------------------------------------------------------------------
// Generic forward device function (unchanged signature)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems, bool UseSigmoid>
__device__ __forceinline__ void
ltv_fused_concat_head_major_register_cast_cub_forward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
  LTV_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(
      T_in, T_compute, T_out,
      kScanThreads, kRThreads, kRItems, UseSigmoid,
      combined, inits, logit_bias, out_q, out_k, out_v,
      sc_b, sc_t, sc_e,
      T_seq, NH, NKVH, Da, r);
}

// -------------------------------------------------------------------------
// Const forward device function (unchanged)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems, bool UseSigmoid,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const>
__device__ __forceinline__ void
ltv_fused_concat_head_major_register_cast_cub_forward_device_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
  LTV_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(
      T_in, T_compute, T_out,
      kScanThreads, kRThreads, kRItems, UseSigmoid,
      combined, inits, logit_bias, out_q, out_k, out_v,
      sc_b_const, sc_t_const, sc_e_const,
      T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Kernels (unchanged)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems>
__global__ void ltv_fused_concat_head_major_register_cast_cub_forward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r, bool use_sigmoid)
{
  if (use_sigmoid)
  {
    ltv_fused_concat_head_major_register_cast_cub_forward_device<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, true>(
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
  }
  else
  {
    ltv_fused_concat_head_major_register_cast_cub_forward_device<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, false>(
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
  }
}

template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const>
__global__ void
ltv_fused_concat_head_major_register_cast_cub_forward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
  ltv_fused_concat_head_major_register_cast_cub_forward_device_const<
      T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, true,
      sc_b_const, sc_t_const, sc_e_const, T_const, NH_const, NKVH_const,
      Da_const, r_const>(
      combined, inits, logit_bias, out_q, out_k, out_v);
}

// -------------------------------------------------------------------------
// Dispatch validation – updated to include alpha_cache
// -------------------------------------------------------------------------
template <typename T_compute, int kScanThreads, int kRThreads, int kRItems>
constexpr bool is_dispatch_valid_forward()
{
  constexpr int block_size = kScanThreads * kRThreads;
  if (block_size > 1024)
    return false;
  using ScanTemp = typename cub::WarpScan<AffineState<T_compute, kRItems>,
                                          kScanThreads>::TempStorage;
  constexpr size_t scan_temp_size = sizeof(ScanTemp) * kRThreads;
  constexpr size_t alpha_cache_size = sizeof(T_compute) * kScanThreads;
  constexpr size_t static_smem = scan_temp_size + alpha_cache_size;
  constexpr size_t max_total_smem = LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES;
  return static_smem <= max_total_smem;
}

// -------------------------------------------------------------------------
// Dispatch inner (unchanged logic, but uses updated smem_bytes=0)
// -------------------------------------------------------------------------
template <typename T_compute, int kScanThreads, int kRThreads, int kRItems>
void forward_dispatch_inner(torch::Tensor combined, torch::Tensor inits,
                            torch::Tensor logit_bias, torch::Tensor out_q,
                            torch::Tensor out_k, torch::Tensor out_v, int T,
                            int NH, int NKVH, int Da, int r, bool use_sigmoid,
                            dim3 grid, cudaStream_t stream,
                            at::ScalarType dtype, int64_t sc_b, int64_t sc_t,
                            int64_t sc_e)
{
  constexpr bool kValid =
      is_dispatch_valid_forward<T_compute, kScanThreads, kRThreads, kRItems>();
  if constexpr (kValid)
  {
    int smem_bytes = 0; // all shared memory is statically known
    dim3 block(kScanThreads * kRThreads);

    if (use_sigmoid && T == kT && NH == kNH && NKVH == kNKVH && Da == kDa &&
        r == kR)
    {
      constexpr int64_t kStrideCombinedB = kStrideCombinedB_f(true);
      if (sc_b == kStrideCombinedB && sc_t == kStrideCombinedT &&
          sc_e == kStrideCombinedE)
      {
        AT_DISPATCH_SWITCH(
            dtype, "ltv_fused_concat_cub_forward_const",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
              ltv_fused_concat_head_major_register_cast_cub_forward_kernel_const<
                  float, float, float, kScanThreads, kRThreads, kRItems, kT,
                  kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT,
                  kStrideCombinedE><<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<float>(), inits.data_ptr<float>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<float>(),
                  out_k.data_ptr<float>(), out_v.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                   {
              ltv_fused_concat_head_major_register_cast_cub_forward_kernel_const<
                  at::BFloat16, float, at::BFloat16, kScanThreads, kRThreads,
                  kRItems, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                  kStrideCombinedT, kStrideCombinedE>
                  <<<grid, block, smem_bytes, stream>>>(
                      combined.data_ptr<at::BFloat16>(),
                      inits.data_ptr<at::BFloat16>(),
                      logit_bias.data_ptr<float>(),
                      out_q.data_ptr<at::BFloat16>(),
                      out_k.data_ptr<at::BFloat16>(),
                      out_v.data_ptr<at::BFloat16>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
        return;
      }
      TORCH_WARN_ONCE(
          "cuda_fused_concat_head_major_original_forward: "
          "matched specialized shape but not the specialized strides. "
          "Falling back to generic kernel. "
          "combined.stride=(",
          sc_b, ", ", sc_t, ", ", sc_e, "), ",
          "expected=(", kStrideCombinedB, ", ", kStrideCombinedT, ", ",
          kStrideCombinedE, ")");
    }

    TORCH_WARN_ONCE("ltv_fused_concat_cub_forward: using generic kernel "
                    "(shape/strides not specialised). "
                    "T=",
                    T, " NH=", NH, " NKVH=", NKVH, " Da=", Da, " r=", r,
                    " stride=(", sc_b, ",", sc_t, ",", sc_e, ")");

    AT_DISPATCH_SWITCH(
        dtype, "ltv_fused_concat_cub_forward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
          ltv_fused_concat_head_major_register_cast_cub_forward_kernel<
              float, float, float, kScanThreads, kRThreads, kRItems>
              <<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<float>(), inits.data_ptr<float>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<float>(),
                  out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                  sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, use_sigmoid);
          C10_CUDA_KERNEL_LAUNCH_CHECK(); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                               {
          ltv_fused_concat_head_major_register_cast_cub_forward_kernel<
              at::BFloat16, float, at::BFloat16, kScanThreads, kRThreads,
              kRItems><<<grid, block, smem_bytes, stream>>>(
              combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(),
              logit_bias.data_ptr<float>(), out_q.data_ptr<at::BFloat16>(),
              out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
              sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, use_sigmoid);
          C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
  }
  else
  {
    TORCH_CHECK(false, "Invalid forward dispatch parameters (shared memory overflow). ",
                "scan_threads=", kScanThreads, ", r_threads=", kRThreads,
                ", r_items=", kRItems);
  }
}

template <typename T_compute, int kScanThreads>
void forward_dispatch_outer(torch::Tensor combined, torch::Tensor inits,
                            torch::Tensor logit_bias, torch::Tensor out_q,
                            torch::Tensor out_k, torch::Tensor out_v, int T,
                            int NH, int NKVH, int Da, int r, int r_threads,
                            int r_items, bool use_sigmoid, dim3 grid,
                            cudaStream_t stream, at::ScalarType dtype,
                            int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
  DISPATCH_R_THREADS(r_threads, {
    DISPATCH_R_ITEMS(r_items, {
      forward_dispatch_inner<T_compute, kScanThreads, kRThreads, kRItems>(
          combined, inits, logit_bias, out_q, out_k, out_v, T, NH, NKVH, Da, r,
          use_sigmoid, grid, stream, dtype, sc_b, sc_t, sc_e);
    });
  });
}

void ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v, int B, int T,
    int NH, int NKVH, int Da, int r, int scan_threads, int r_threads,
    int r_items, bool use_sigmoid)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t sc_b = combined.stride(0);
  int64_t sc_t = combined.stride(1);
  int64_t sc_e = combined.stride(2);

  int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
  dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks);
  auto dtype = combined.scalar_type();

  DISPATCH_SCAN_THREADS(scan_threads, {
    forward_dispatch_outer<float, kScanThreads>(
        combined, inits, logit_bias, out_q, out_k, out_v, T, NH, NKVH, Da, r,
        r_threads, r_items, use_sigmoid, grid, stream, dtype, sc_b, sc_t, sc_e);
  });
}
