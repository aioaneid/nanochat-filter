#include "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Loop macro – expands to a #pragma unroll for over kRItems
// -------------------------------------------------------------------------
#define LTV_BACKWARD_UNROLL_OVER_R_ITEMS(BODY_MACRO)           \
  _Pragma("unroll") for (int item = 0; item < kRItems; ++item) \
  {                                                            \
    BODY_MACRO                                                 \
  }

// -------------------------------------------------------------------------
// Strength‑reduced pointer setup macros
// -------------------------------------------------------------------------
#define LTV_BACKWARD_SETUP_GRAD_H_PTR(start_idx) \
  const T_out *ptr_grad_h = &grad_h_ptr[start_idx];

#define LTV_BACKWARD_SETUP_H_PTR(start_idx) \
  const T_out *ptr_h = &h_ptr[start_idx];

#define LTV_BACKWARD_SETUP_VAL_IN_PTR(start_idx) \
  const T_in *ptr_val_in = &combined[start_idx];

#define LTV_BACKWARD_SETUP_VAL_OUT_PTR(start_idx) \
  T_out *ptr_val_out = &grad_combined[start_idx];

#define LTV_BACKWARD_SETUP_INIT_PTR(start_idx) \
  const T_in *ptr_init = &inits[start_idx];

#define LTV_BACKWARD_SETUP_GRAD_INIT_PTR(start_idx) \
  T_out *ptr_grad_init = &grad_inits[start_idx];

// -------------------------------------------------------------------------
// Strength‑reduced body macros for the hot loops
// -------------------------------------------------------------------------

// Load one grad_h element (no validity check)
#define LTV_BACKWARD_LOAD_GRAD_H_PTR       \
  elem.x[item] = (T_compute)(*ptr_grad_h); \
  ptr_grad_h += str_gh_d;

// Load one h element (no validity check)
#define LTV_BACKWARD_LOAD_H_PTR \
  T_compute h_prev = (T_compute)(*ptr_h);

// Load one combined (x_val) element
#define LTV_BACKWARD_LOAD_VAL_IN_PTR \
  T_compute x_val = (T_compute)(*ptr_val_in);

// The big computation & store block (fast path, no bound check)
#define LTV_BACKWARD_COMPUTE_AND_STORE_PTR(sc_in_e, sc_out_e) \
  T_compute X_future = running_x[lane_y * kRItems + item];    \
  T_compute g = scan_out.a * X_future + scan_out.x[item];     \
  T_compute h_prev_val;                                       \
  if (is_t0)                                                  \
  {                                                           \
    h_prev_val = (T_compute)(*ptr_init);                      \
  }                                                           \
  else                                                        \
  {                                                           \
    h_prev_val = (T_compute)(*ptr_h);                         \
  }                                                           \
  T_compute x_val = (T_compute)(*ptr_val_in);                 \
  T_compute d_alpha = (x_val - h_prev_val) * g;               \
  T_compute d_x = alpha_gate * g;                             \
  T_compute d_l = d_alpha * d_l_multiplier;                   \
  *ptr_val_out = (T_out)d_x;                                  \
  d_l_reg += d_l;                                             \
  if (is_t0)                                                  \
  {                                                           \
    *ptr_grad_init = (T_out)(g * a_t_gate);                   \
  }                                                           \
  if (update_running)                                         \
  {                                                           \
    running_x[lane_y * kRItems + item] = g;                   \
  }                                                           \
  ptr_init += 1;                                              \
  ptr_h += str_h_d;                                           \
  ptr_val_in += sc_in_e;                                      \
  ptr_val_out += sc_out_e;                                    \
  ptr_grad_init += 1;

#define LTV_BACKWARD_COMPUTE_AND_STORE_PTR_WITH_BREAK(sc_in_e, sc_out_e, r) \
  if (r_start + item >= r)                                                  \
    break;                                                                  \
  LTV_BACKWARD_COMPUTE_AND_STORE_PTR(sc_in_e, sc_out_e)

// -------------------------------------------------------------------------
// Main backward device function macro
// -------------------------------------------------------------------------
#define LTV_FUSED_CONCAT_HM_BACKWARD_DEVICE_BODY(                                                                                                    \
    T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, UseSigmoid,                                                                            \
    combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,                                                                              \
    grad_combined, grad_inits, grad_logit_bias, grad_logit_accum,                                                                                    \
    sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,                                                                                         \
    str_hq_b, str_hq_h, str_hq_t, str_hq_d, str_hk_b, str_hk_h, str_hk_t, str_hk_d,                                                                  \
    str_hv_b, str_hv_h, str_hv_t, str_hv_d, str_gq_b, str_gq_h, str_gq_t, str_gq_d,                                                                  \
    str_gk_b, str_gk_h, str_gk_t, str_gk_d, str_gv_b, str_gv_h, str_gv_t, str_gv_d,                                                                  \
    T_seq, NH, NKVH, Da, r)                                                                                                                          \
  do                                                                                                                                                 \
  {                                                                                                                                                  \
    const int tid = threadIdx.x;                                                                                                                     \
    const int lane_x = tid % kScanThreads;                                                                                                           \
    const int lane_y = tid / kScanThreads;                                                                                                           \
                                                                                                                                                     \
    using ScanTemp = typename cub::WarpScan<AffineState<T_compute, kRItems>,                                                                         \
                                            kScanThreads>::TempStorage;                                                                              \
    __shared__ union                                                                                                                                 \
    {                                                                                                                                                \
      ScanTemp scan_temp[kRThreads];                                                                                                                 \
      T_compute partial_flat[kRThreads * kScanThreads];                                                                                              \
    } shared_mem;                                                                                                                                    \
    __shared__ T_compute running_x[kRThreads * kRItems];                                                                                             \
    __shared__ T_compute shm_alpha_gate[kScanThreads];                                                                                               \
    __shared__ T_compute carry_alpha_cache;                                                                                                          \
                                                                                                                                                     \
    const unsigned int shuffle_mask = __activemask();                                                                                                \
                                                                                                                                                     \
    const unsigned int b_idx = blockIdx.x;                                                                                                           \
    const unsigned int h_glob = blockIdx.y;                                                                                                          \
    const unsigned int da_idx = blockIdx.z % Da;                                                                                                     \
    const int r_chunk = blockIdx.z / Da;                                                                                                             \
                                                                                                                                                     \
    const unsigned int D = Da * r;                                                                                                                   \
    const unsigned int total_h = NH + 2 * NKVH;                                                                                                      \
    const unsigned int head_stride = D + Da;                                                                                                         \
                                                                                                                                                     \
    const T_out *h_ptr;                                                                                                                              \
    const T_out *grad_h_ptr;                                                                                                                         \
    unsigned int h_loc;                                                                                                                              \
    int64_t str_h_b, str_h_h, str_h_t, str_h_d;                                                                                                      \
    int64_t str_gh_b, str_gh_h, str_gh_t, str_gh_d;                                                                                                  \
                                                                                                                                                     \
    if (h_glob < NH)                                                                                                                                 \
    {                                                                                                                                                \
      h_ptr = hq;                                                                                                                                    \
      grad_h_ptr = grad_hq;                                                                                                                          \
      h_loc = h_glob;                                                                                                                                \
      str_h_b = str_hq_b;                                                                                                                            \
      str_h_h = str_hq_h;                                                                                                                            \
      str_h_t = str_hq_t;                                                                                                                            \
      str_h_d = str_hq_d;                                                                                                                            \
      str_gh_b = str_gq_b;                                                                                                                           \
      str_gh_h = str_gq_h;                                                                                                                           \
      str_gh_t = str_gq_t;                                                                                                                           \
      str_gh_d = str_gq_d;                                                                                                                           \
    }                                                                                                                                                \
    else if (h_glob < NH + NKVH)                                                                                                                     \
    {                                                                                                                                                \
      h_ptr = hk;                                                                                                                                    \
      grad_h_ptr = grad_hk;                                                                                                                          \
      h_loc = h_glob - NH;                                                                                                                           \
      str_h_b = str_hk_b;                                                                                                                            \
      str_h_h = str_hk_h;                                                                                                                            \
      str_h_t = str_hk_t;                                                                                                                            \
      str_h_d = str_hk_d;                                                                                                                            \
      str_gh_b = str_gk_b;                                                                                                                           \
      str_gh_h = str_gk_h;                                                                                                                           \
      str_gh_t = str_gk_t;                                                                                                                           \
      str_gh_d = str_gk_d;                                                                                                                           \
    }                                                                                                                                                \
    else                                                                                                                                             \
    {                                                                                                                                                \
      h_ptr = hv;                                                                                                                                    \
      grad_h_ptr = grad_hv;                                                                                                                          \
      h_loc = h_glob - (NH + NKVH);                                                                                                                  \
      str_h_b = str_hv_b;                                                                                                                            \
      str_h_h = str_hv_h;                                                                                                                            \
      str_h_t = str_hv_t;                                                                                                                            \
      str_h_d = str_hv_d;                                                                                                                            \
      str_gh_b = str_gv_b;                                                                                                                           \
      str_gh_h = str_gv_h;                                                                                                                           \
      str_gh_t = str_gv_t;                                                                                                                           \
      str_gh_d = str_gv_d;                                                                                                                           \
    }                                                                                                                                                \
                                                                                                                                                     \
    const T_compute bias = logit_bias[h_glob * Da + da_idx];                                                                                         \
                                                                                                                                                     \
    const int num_r_chunks = (r + kRThreads * kRItems - 1) / (kRThreads * kRItems);                                                                  \
    if (r_chunk >= num_r_chunks)                                                                                                                     \
    {                                                                                                                                                \
      /* Safe because r_chunk is derived from blockIdx.z and is therefore uniform across all threads in the block. */                                \
      return;                                                                                                                                        \
    }                                                                                                                                                \
                                                                                                                                                     \
    const unsigned int val_base = h_glob * head_stride + da_idx * r;                                                                                 \
    const unsigned int logit_base = h_glob * head_stride + D + da_idx;                                                                               \
    const int64_t b_offset_in = (int64_t)b_idx * sc_in_b;                                                                                            \
    const int64_t b_offset_out = (int64_t)b_idx * sc_out_b;                                                                                          \
                                                                                                                                                     \
    int num_t_chunks = (T_seq + kScanThreads - 1) / kScanThreads;                                                                                    \
    int t_start = (num_t_chunks - 1) * kScanThreads;                                                                                                 \
    int t_end = min(t_start + kScanThreads, T_seq);                                                                                                  \
    int chunk_len = t_end - t_start;                                                                                                                 \
                                                                                                                                                     \
    const int r_stride = kRThreads * kRItems;                                                                                                        \
    const int r_start = lane_y * kRItems + r_chunk * r_stride;                                                                                       \
    const bool all_r_valid = (r_start + kRItems <= r);                                                                                               \
                                                                                                                                                     \
    if (threadIdx.x == 0)                                                                                                                            \
      carry_alpha_cache = T_compute(0);                                                                                                              \
    if (lane_x == 0)                                                                                                                                 \
    {                                                                                                                                                \
      LTV_BACKWARD_UNROLL_OVER_R_ITEMS(                                                                                                              \
          running_x[lane_y * kRItems + item] = T_compute(0);)                                                                                        \
    }                                                                                                                                                \
    __syncthreads();                                                                                                                                 \
                                                                                                                                                     \
    const int64_t da_r = da_idx * r;                                                                                                                 \
    const int64_t gh_base_invariant =                                                                                                                \
        (int64_t)b_idx * str_gh_b + (int64_t)h_loc * str_gh_h + da_r * str_gh_d;                                                                     \
    const int64_t h_prev_base_invariant =                                                                                                            \
        (int64_t)b_idx * str_h_b + (int64_t)h_loc * str_h_h + da_r * str_h_d;                                                                        \
    const int64_t init_base_invariant =                                                                                                              \
        (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + da_r;                                                                                 \
    const int64_t val_in_base_invariant =                                                                                                            \
        b_offset_in + (int64_t)val_base * sc_in_e;                                                                                                   \
    const int64_t val_out_base_invariant =                                                                                                           \
        b_offset_out + (int64_t)val_base * sc_out_e;                                                                                                 \
    const int64_t logit_in_base_invariant =                                                                                                          \
        b_offset_in + (int64_t)logit_base * sc_in_e;                                                                                                 \
                                                                                                                                                     \
    const int64_t r_chunk_offset = r_chunk * r_stride;                                                                                               \
    const int partial_idx = lane_y * kScanThreads + lane_x;                                                                                          \
                                                                                                                                                     \
    AffineState<T_compute, kRItems> scan_out;                                                                                                        \
    cub::WarpScan<AffineState<T_compute, kRItems>, kScanThreads> warp_scan(                                                                          \
        shared_mem.scan_temp[lane_y]);                                                                                                               \
                                                                                                                                                     \
    T_compute block_bias_sum = T_compute(0);                                                                                                         \
                                                                                                                                                     \
    for (int t_chunk = num_t_chunks - 1; t_chunk >= 0; --t_chunk)                                                                                    \
    {                                                                                                                                                \
      const int offset = kScanThreads - chunk_len;                                                                                                   \
      const bool valid_t = lane_x >= offset;                                                                                                         \
      const int t_global = valid_t ? (t_end - 1 - (lane_x - offset)) : 0;                                                                            \
                                                                                                                                                     \
      int64_t t_logit_in_idx = logit_in_base_invariant + (int64_t)t_global * sc_in_t;                                                                \
                                                                                                                                                     \
      int64_t t_gh_base = 0;                                                                                                                         \
      int64_t t_h_prev_base = 0;                                                                                                                     \
      int64_t t_val_in_base = 0;                                                                                                                     \
      int64_t t_val_out_base = 0;                                                                                                                    \
      if (valid_t)                                                                                                                                   \
      {                                                                                                                                              \
        t_gh_base = gh_base_invariant + (int64_t)t_global * str_gh_t + (int64_t)(lane_y * kRItems) * str_gh_d;                                       \
        t_h_prev_base = h_prev_base_invariant + (int64_t)((t_global > 0 ? t_global - 1 : 0)) * str_h_t + (int64_t)(lane_y * kRItems) * str_h_d;      \
        t_val_in_base = val_in_base_invariant + (int64_t)t_global * sc_in_t + (int64_t)(lane_y * kRItems) * sc_in_e + r_chunk_offset * sc_in_e;      \
        t_val_out_base = val_out_base_invariant + (int64_t)t_global * sc_out_t + (int64_t)(lane_y * kRItems) * sc_out_e + r_chunk_offset * sc_out_e; \
      }                                                                                                                                              \
                                                                                                                                                     \
      T_compute d_l_reg = T_compute(0);                                                                                                              \
      AffineState<T_compute, kRItems> elem;                                                                                                          \
                                                                                                                                                     \
      T_compute alpha_gate = T_compute(0);                                                                                                           \
      if (lane_y == 0 && valid_t)                                                                                                                    \
      {                                                                                                                                              \
        T_compute l_raw = (T_compute)combined[t_logit_in_idx];                                                                                       \
        T_compute logit = l_raw + bias;                                                                                                              \
        alpha_gate = UseSigmoid ? sigmoid_f32(logit) : logit;                                                                                        \
        shm_alpha_gate[lane_x] = alpha_gate;                                                                                                         \
      }                                                                                                                                              \
      __syncthreads();                                                                                                                               \
                                                                                                                                                     \
      if (valid_t && lane_y != 0)                                                                                                                    \
      {                                                                                                                                              \
        alpha_gate = shm_alpha_gate[lane_x];                                                                                                         \
      }                                                                                                                                              \
                                                                                                                                                     \
      /* Unconditional shuffle – all threads in the logical warp must participate */                                                                 \
      T_compute alpha_next_shfl = __shfl_up_sync(shuffle_mask, alpha_gate, 1, kScanThreads);                                                         \
                                                                                                                                                     \
      if (valid_t)                                                                                                                                   \
      {                                                                                                                                              \
        T_compute alpha_next;                                                                                                                        \
        if (lane_x == offset)                                                                                                                        \
          alpha_next = carry_alpha_cache;                                                                                                            \
        else                                                                                                                                         \
          alpha_next = alpha_next_shfl;                                                                                                              \
        T_compute a_next = T_compute(1) - alpha_next;                                                                                                \
        T_compute a_t_gate = T_compute(1) - alpha_gate;                                                                                              \
        T_compute d_l_multiplier = UseSigmoid ? (alpha_gate * a_t_gate) : T_compute(1);                                                              \
                                                                                                                                                     \
        elem.a = a_next;                                                                                                                             \
                                                                                                                                                     \
        int64_t item_gh_idx = t_gh_base;                                                                                                             \
        LTV_BACKWARD_SETUP_GRAD_H_PTR(item_gh_idx)                                                                                                   \
        if (all_r_valid)                                                                                                                             \
        {                                                                                                                                            \
          LTV_BACKWARD_UNROLL_OVER_R_ITEMS(LTV_BACKWARD_LOAD_GRAD_H_PTR)                                                                             \
        }                                                                                                                                            \
        else                                                                                                                                         \
        {                                                                                                                                            \
          LTV_BACKWARD_UNROLL_OVER_R_ITEMS(                                                                                                          \
              bool valid_r_item = (r_start + item < r);                                                                                              \
              elem.x[item] = valid_r_item ? (T_compute)(*ptr_grad_h) : T_compute(0);                                                                 \
              ptr_grad_h += str_gh_d;)                                                                                                               \
        }                                                                                                                                            \
                                                                                                                                                     \
        warp_scan.InclusiveScan(elem, scan_out, AffineScanOp<T_compute, kRItems>());                                                                 \
                                                                                                                                                     \
        const bool is_t0 = (t_global == 0);                                                                                                          \
        const bool update_running = (lane_x == kScanThreads - 1);                                                                                    \
                                                                                                                                                     \
        int64_t item_init_idx = init_base_invariant + r_start;                                                                                       \
        int64_t item_h_prev_idx = t_h_prev_base;                                                                                                     \
        int64_t item_val_in_idx = t_val_in_base;                                                                                                     \
        int64_t item_val_out_idx = t_val_out_base;                                                                                                   \
                                                                                                                                                     \
        LTV_BACKWARD_SETUP_INIT_PTR(item_init_idx)                                                                                                   \
        LTV_BACKWARD_SETUP_H_PTR(item_h_prev_idx)                                                                                                    \
        LTV_BACKWARD_SETUP_VAL_IN_PTR(item_val_in_idx)                                                                                               \
        LTV_BACKWARD_SETUP_VAL_OUT_PTR(item_val_out_idx)                                                                                             \
        LTV_BACKWARD_SETUP_GRAD_INIT_PTR(item_init_idx)                                                                                              \
                                                                                                                                                     \
        if (all_r_valid)                                                                                                                             \
        {                                                                                                                                            \
          LTV_BACKWARD_UNROLL_OVER_R_ITEMS(LTV_BACKWARD_COMPUTE_AND_STORE_PTR(                                                                       \
              sc_in_e, sc_out_e))                                                                                                                    \
        }                                                                                                                                            \
        else                                                                                                                                         \
        {                                                                                                                                            \
          LTV_BACKWARD_UNROLL_OVER_R_ITEMS(                                                                                                          \
              LTV_BACKWARD_COMPUTE_AND_STORE_PTR_WITH_BREAK(sc_in_e, sc_out_e, r))                                                                   \
        }                                                                                                                                            \
                                                                                                                                                     \
        if (lane_y == 0 && update_running)                                                                                                           \
          carry_alpha_cache = alpha_gate;                                                                                                            \
      }                                                                                                                                              \
      else                                                                                                                                           \
      {                                                                                                                                              \
        elem.a = T_compute(1);                                                                                                                       \
        LTV_BACKWARD_UNROLL_OVER_R_ITEMS(                                                                                                            \
            elem.x[item] = T_compute(0);)                                                                                                            \
                                                                                                                                                     \
        AffineState<T_compute, kRItems> scan_out;                                                                                                    \
        warp_scan.InclusiveScan(elem, scan_out, AffineScanOp<T_compute, kRItems>());                                                                 \
                                                                                                                                                     \
        const bool update_running = (lane_x == kScanThreads - 1);                                                                                    \
        LTV_BACKWARD_UNROLL_OVER_R_ITEMS(                                                                                                            \
            if (r_start + item >= r) break;                                                                                                          \
            T_compute X_future = running_x[lane_y * kRItems + item];                                                                                 \
            T_compute g = scan_out.a * X_future + scan_out.x[item];                                                                                  \
            if (update_running) {                                                                                                                    \
              running_x[lane_y * kRItems + item] = g;                                                                                                \
            })                                                                                                                                       \
      }                                                                                                                                              \
                                                                                                                                                     \
      __syncthreads();                                                                                                                               \
      shared_mem.partial_flat[partial_idx] = d_l_reg;                                                                                                \
      __syncthreads();                                                                                                                               \
                                                                                                                                                     \
      constexpr bool is_pow2 = (kRThreads > 0) && ((kRThreads & (kRThreads - 1)) == 0);                                                              \
      for (int stride = 1; stride < kRThreads; stride <<= 1)                                                                                         \
      {                                                                                                                                              \
        const int offset_red = stride * kScanThreads;                                                                                                \
        if ((lane_y & (2 * stride - 1)) == 0)                                                                                                        \
        {                                                                                                                                            \
          int other = lane_y + stride;                                                                                                               \
          if (is_pow2 || other < kRThreads)                                                                                                          \
          {                                                                                                                                          \
            shared_mem.partial_flat[partial_idx] +=                                                                                                  \
                shared_mem.partial_flat[partial_idx + offset_red];                                                                                   \
          }                                                                                                                                          \
        }                                                                                                                                            \
        __syncthreads();                                                                                                                             \
      }                                                                                                                                              \
                                                                                                                                                     \
      if (lane_y == 0 && valid_t)                                                                                                                    \
      {                                                                                                                                              \
        int64_t accum_idx = (((int64_t)b_idx * total_h + (int64_t)h_glob) * Da +                                                                     \
                             (int64_t)da_idx) *                                                                                                      \
                                T_seq +                                                                                                              \
                            (int64_t)t_global;                                                                                                       \
        atomicAdd(&grad_logit_accum[accum_idx], shared_mem.partial_flat[lane_x]);                                                                    \
      }                                                                                                                                              \
                                                                                                                                                     \
      if (threadIdx.x == 0)                                                                                                                          \
      {                                                                                                                                              \
        for (int i = 0; i < chunk_len; ++i)                                                                                                          \
          block_bias_sum += shared_mem.partial_flat[offset + i];                                                                                     \
      }                                                                                                                                              \
      __syncthreads();                                                                                                                               \
                                                                                                                                                     \
      if (t_chunk > 0)                                                                                                                               \
      {                                                                                                                                              \
        bool is_first = t_chunk == num_t_chunks - 1;                                                                                                 \
        int step_t_global = is_first ? (T_seq - t_start) : kScanThreads;                                                                             \
        t_start -= step_t_global;                                                                                                                    \
        t_end -= step_t_global;                                                                                                                      \
        chunk_len = kScanThreads;                                                                                                                    \
      }                                                                                                                                              \
    }                                                                                                                                                \
                                                                                                                                                     \
    if (threadIdx.x == 0)                                                                                                                            \
    {                                                                                                                                                \
      const int bias_idx = h_glob * Da + da_idx;                                                                                                     \
      atomicAdd(&grad_logit_bias[bias_idx], block_bias_sum);                                                                                         \
    }                                                                                                                                                \
  } while (0)

// -------------------------------------------------------------------------
// Generic backward device function (runtime strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int kScanThreads,
          int kRThreads, int kRItems, bool UseSigmoid>
__device__ __forceinline__ void
ltv_fused_concat_head_major_register_cast_cub_backward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t str_hq_b, int64_t str_hq_h, int64_t str_hq_t, int64_t str_hq_d,
    int64_t str_hk_b, int64_t str_hk_h, int64_t str_hk_t, int64_t str_hk_d,
    int64_t str_hv_b, int64_t str_hv_h, int64_t str_hv_t, int64_t str_hv_d,
    int64_t str_gq_b, int64_t str_gq_h, int64_t str_gq_t, int64_t str_gq_d,
    int64_t str_gk_b, int64_t str_gk_h, int64_t str_gk_t, int64_t str_gk_d,
    int64_t str_gv_b, int64_t str_gv_h, int64_t str_gv_t, int64_t str_gv_d,
    int T_seq, int NH, int NKVH, int Da, int r)
{
  LTV_FUSED_CONCAT_HM_BACKWARD_DEVICE_BODY(
      T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, UseSigmoid,
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits, grad_logit_bias, grad_logit_accum,
      sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
      str_hq_b, str_hq_h, str_hq_t, str_hq_d,
      str_hk_b, str_hk_h, str_hk_t, str_hk_d,
      str_hv_b, str_hv_h, str_hv_t, str_hv_d,
      str_gq_b, str_gq_h, str_gq_t, str_gq_d,
      str_gk_b, str_gk_h, str_gk_t, str_gk_d,
      str_gv_b, str_gv_h, str_gv_t, str_gv_d,
      T_seq, NH, NKVH, Da, r);
}

// -------------------------------------------------------------------------
// Const backward device function (compile‑time strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int kScanThreads,
          int kRThreads, int kRItems, bool UseSigmoid,
          int64_t sc_in_b_const, int64_t sc_in_t_const, int64_t sc_in_e_const,
          int64_t sc_out_b_const, int64_t sc_out_t_const, int64_t sc_out_e_const,
          int64_t str_hq_b_const, int64_t str_hq_h_const, int64_t str_hq_t_const,
          int64_t str_hq_d_const,
          int64_t str_hk_b_const, int64_t str_hk_h_const, int64_t str_hk_t_const,
          int64_t str_hk_d_const,
          int64_t str_hv_b_const, int64_t str_hv_h_const, int64_t str_hv_t_const,
          int64_t str_hv_d_const,
          int64_t str_gq_b_const, int64_t str_gq_h_const, int64_t str_gq_t_const,
          int64_t str_gq_d_const,
          int64_t str_gk_b_const, int64_t str_gk_h_const, int64_t str_gk_t_const,
          int64_t str_gk_d_const,
          int64_t str_gv_b_const, int64_t str_gv_h_const, int64_t str_gv_t_const,
          int64_t str_gv_d_const,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const>
__device__ __forceinline__ void
ltv_fused_concat_head_major_register_cast_cub_backward_device_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum)
{
  LTV_FUSED_CONCAT_HM_BACKWARD_DEVICE_BODY(
      T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, UseSigmoid,
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits, grad_logit_bias, grad_logit_accum,
      sc_in_b_const, sc_in_t_const, sc_in_e_const,
      sc_out_b_const, sc_out_t_const, sc_out_e_const,
      str_hq_b_const, str_hq_h_const, str_hq_t_const, str_hq_d_const,
      str_hk_b_const, str_hk_h_const, str_hk_t_const, str_hk_d_const,
      str_hv_b_const, str_hv_h_const, str_hv_t_const, str_hv_d_const,
      str_gq_b_const, str_gq_h_const, str_gq_t_const, str_gq_d_const,
      str_gk_b_const, str_gk_h_const, str_gk_t_const, str_gk_d_const,
      str_gv_b_const, str_gv_h_const, str_gv_t_const, str_gv_d_const,
      T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Generic backward kernel (calls generic device function)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int kScanThreads,
          int kRThreads, int kRItems>
__global__ void ltv_fused_concat_head_major_register_cast_cub_backward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t s_hq_b, int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d,
    int64_t s_hk_b, int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d,
    int64_t s_hv_b, int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d,
    int T_seq, int NH, int NKVH, int Da, int r, bool use_sigmoid)
{
  if (use_sigmoid)
  {
    ltv_fused_concat_head_major_register_cast_cub_backward_device<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, true>(
        combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
        grad_combined, grad_inits, grad_logit_bias, grad_logit_accum,
        sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
        s_hq_b, s_hq_h, s_hq_t, s_hq_d,
        s_hk_b, s_hk_h, s_hk_t, s_hk_d,
        s_hv_b, s_hv_h, s_hv_t, s_hv_d,
        s_gq_b, s_gq_h, s_gq_t, s_gq_d,
        s_gk_b, s_gk_h, s_gk_t, s_gk_d,
        s_gv_b, s_gv_h, s_gv_t, s_gv_d,
        T_seq, NH, NKVH, Da, r);
  }
  else
  {
    ltv_fused_concat_head_major_register_cast_cub_backward_device<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, false>(
        combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
        grad_combined, grad_inits, grad_logit_bias, grad_logit_accum,
        sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
        s_hq_b, s_hq_h, s_hq_t, s_hq_d,
        s_hk_b, s_hk_h, s_hk_t, s_hk_d,
        s_hv_b, s_hv_h, s_hv_t, s_hv_d,
        s_gq_b, s_gq_h, s_gq_t, s_gq_d,
        s_gk_b, s_gk_h, s_gk_t, s_gk_d,
        s_gv_b, s_gv_h, s_gv_t, s_gv_d,
        T_seq, NH, NKVH, Da, r);
  }
}

// -------------------------------------------------------------------------
// Constant‑strides backward kernel (calls const device function)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int kScanThreads,
          int kRThreads, int kRItems, int T_const, int NH_const, int NKVH_const,
          int Da_const, int r_const, int64_t sc_in_b_const,
          int64_t sc_in_t_const, int64_t sc_in_e_const, int64_t sc_out_b_const,
          int64_t sc_out_t_const, int64_t sc_out_e_const, int64_t s_hq_b_const,
          int64_t s_hq_h_const, int64_t s_hq_t_const, int64_t s_hq_d_const,
          int64_t s_hk_b_const, int64_t s_hk_h_const, int64_t s_hk_t_const,
          int64_t s_hk_d_const, int64_t s_hv_b_const, int64_t s_hv_h_const,
          int64_t s_hv_t_const, int64_t s_hv_d_const, int64_t s_gq_b_const,
          int64_t s_gq_h_const, int64_t s_gq_t_const, int64_t s_gq_d_const,
          int64_t s_gk_b_const, int64_t s_gk_h_const, int64_t s_gk_t_const,
          int64_t s_gk_d_const, int64_t s_gv_b_const, int64_t s_gv_h_const,
          int64_t s_gv_t_const, int64_t s_gv_d_const>
__global__ void
ltv_fused_concat_head_major_register_cast_cub_backward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum)
{
  ltv_fused_concat_head_major_register_cast_cub_backward_device_const<
      T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems, true,
      sc_in_b_const, sc_in_t_const, sc_in_e_const,
      sc_out_b_const, sc_out_t_const, sc_out_e_const,
      s_hq_b_const, s_hq_h_const, s_hq_t_const, s_hq_d_const,
      s_hk_b_const, s_hk_h_const, s_hk_t_const, s_hk_d_const,
      s_hv_b_const, s_hv_h_const, s_hv_t_const, s_hv_d_const,
      s_gq_b_const, s_gq_h_const, s_gq_t_const, s_gq_d_const,
      s_gk_b_const, s_gk_h_const, s_gk_t_const, s_gk_d_const,
      s_gv_b_const, s_gv_h_const, s_gv_t_const, s_gv_d_const,
      T_const, NH_const, NKVH_const, Da_const, r_const>(
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits, grad_logit_bias, grad_logit_accum);
}

// -------------------------------------------------------------------------
// Shared memory size helper (updated for simplified gate buffer)
// -------------------------------------------------------------------------
template <typename T_compute, int kScanThreads, int kRThreads, int kRItems>
constexpr size_t backward_shared_memory_bytes()
{
  using ScanTemp = typename cub::WarpScan<AffineState<T_compute, kRItems>,
                                          kScanThreads>::TempStorage;
  constexpr size_t size_scan = sizeof(ScanTemp) * kRThreads;
  constexpr size_t size_partial = kRThreads * kScanThreads * sizeof(T_compute);
  constexpr size_t union_size =
      (size_scan > size_partial) ? size_scan : size_partial;
  constexpr size_t size_running = kRThreads * kRItems * sizeof(T_compute);
  constexpr size_t size_gates = kScanThreads * sizeof(T_compute); // shm_alpha_gate
  constexpr size_t size_carry = sizeof(T_compute);                // carry_alpha_cache
  return union_size + size_running + size_gates + size_carry;
}

template <typename T_compute, int kScanThreads, int kRThreads, int kRItems>
constexpr bool is_dispatch_valid_backward()
{
  constexpr int block_size = kScanThreads * kRThreads;
  if (block_size > 1024)
    return false;
  constexpr size_t static_smem =
      backward_shared_memory_bytes<T_compute, kScanThreads, kRThreads, kRItems>();
  return static_smem <= LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES;
}

// -------------------------------------------------------------------------
// Dispatch inner (unchanged – calls the const kernel with constants)
// -------------------------------------------------------------------------
template <typename T_compute, int kScanThreads, int kRThreads, int kRItems>
void backward_dispatch_inner(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_combined,
    torch::Tensor grad_inits, torch::Tensor grad_logit_bias, int B, int T,
    int NH, int NKVH, int Da, int r, bool use_sigmoid,
    bool backward_has_expected_grad_output_strides,
    bool backward_has_transposed_qk_grad_output_strides,
    bool backward_has_specialized_strides, dim3 grid, cudaStream_t stream,
    at::ScalarType dtype, int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e, int64_t s_hq_b,
    int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d, int64_t s_hk_b,
    int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d, int64_t s_hv_b,
    int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d, int64_t s_gq_b,
    int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d, int64_t s_gk_b,
    int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d, int64_t s_gv_b,
    int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  constexpr bool kValid =
      is_dispatch_valid_backward<T_compute, kScanThreads, kRThreads, kRItems>();
  if constexpr (kValid)
  {
    int smem_bytes = static_cast<int>(
        backward_shared_memory_bytes<T_compute, kScanThreads, kRThreads, kRItems>());
    dim3 block(kScanThreads * kRThreads);
    int D = Da * r;
    int total_h = NH + 2 * NKVH;

    auto options = torch::TensorOptions()
                       .dtype(c10::CppTypeToScalarType<T_compute>::value)
                       .device(combined.device());
    torch::Tensor grad_logit_accum = torch::zeros({B, total_h, Da, T}, options);
    auto copy_grad_logits = [&]()
    {
      auto grad_logit_accum_t = grad_logit_accum.permute({0, 3, 1, 2});
      auto grad_logits_view =
          grad_combined.view({B, T, total_h, D + Da}).slice(3, D, D + Da);
      grad_logits_view.copy_(grad_logit_accum_t);
    };

    if (use_sigmoid && T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
      constexpr int64_t kStrideCombinedB = kStrideCombinedB_f(true);
      if (backward_has_expected_grad_output_strides && backward_has_specialized_strides)
      {
        AT_DISPATCH_SWITCH(
            dtype, "ltv_fused_concat_cub_backward_const",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
              ltv_fused_concat_head_major_register_cast_cub_backward_kernel_const<
                  float, float, float, kScanThreads, kRThreads, kRItems, kT,
                  kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT,
                  kStrideCombinedE, kStrideCombinedB, kStrideCombinedT,
                  kStrideCombinedE, kStrideGradOutQB, kStrideGradOutQH,
                  kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB,
                  kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD, kStrideGradOutQB, kStrideGradOutQH,
                  kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB,
                  kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD><<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<float>(), inits.data_ptr<float>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<float>(),
                  out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                  grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(),
                  grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(),
                  grad_inits.data_ptr<float>(), grad_logit_bias.data_ptr<float>(),
                  grad_logit_accum.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); })
                AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                 {
              ltv_fused_concat_head_major_register_cast_cub_backward_kernel_const<
                  at::BFloat16, float, at::BFloat16, kScanThreads, kRThreads,
                  kRItems, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                  kStrideCombinedT, kStrideCombinedE, kStrideCombinedB,
                  kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                  kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT,
                  kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                  kStrideGradOutVT, kStrideGradOutVD, kStrideGradOutQB,
                  kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT,
                  kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                  kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<at::BFloat16>(),
                  out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                  grad_out_q.data_ptr<at::BFloat16>(),
                  grad_out_k.data_ptr<at::BFloat16>(),
                  grad_out_v.data_ptr<at::BFloat16>(),
                  grad_combined.data_ptr<at::BFloat16>(),
                  grad_inits.data_ptr<at::BFloat16>(),
                  grad_logit_bias.data_ptr<float>(),
                  grad_logit_accum.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
        copy_grad_logits();
        return;
      }
      if (backward_has_transposed_qk_grad_output_strides && backward_has_specialized_strides)
      {
        AT_DISPATCH_SWITCH(
            dtype, "ltv_fused_concat_cub_backward_const_transposed_qk_grad",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
              ltv_fused_concat_head_major_register_cast_cub_backward_kernel_const<
                  float, float, float, kScanThreads, kRThreads, kRItems, kT,
                  kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT,
                  kStrideCombinedE, kStrideCombinedB, kStrideCombinedT,
                  kStrideCombinedE, kStrideGradOutQB, kStrideGradOutQH,
                  kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB,
                  kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
                  kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT,
                  kStrideGradOutVD, kStrideGradOutQB,
                  kStrideGradOutQHTransposed, kStrideGradOutQTTransposed,
                  kStrideGradOutQD, kStrideGradOutKB,
                  kStrideGradOutKHTransposed, kStrideGradOutKTTransposed,
                  kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                  kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<float>(), inits.data_ptr<float>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<float>(),
                  out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                  grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(),
                  grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(),
                  grad_inits.data_ptr<float>(), grad_logit_bias.data_ptr<float>(),
                  grad_logit_accum.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); })
                AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                 {
              ltv_fused_concat_head_major_register_cast_cub_backward_kernel_const<
                  at::BFloat16, float, at::BFloat16, kScanThreads, kRThreads,
                  kRItems, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                  kStrideCombinedT, kStrideCombinedE, kStrideCombinedB,
                  kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                  kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
                  kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT,
                  kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                  kStrideGradOutVT, kStrideGradOutVD, kStrideGradOutQB,
                  kStrideGradOutQHTransposed, kStrideGradOutQTTransposed,
                  kStrideGradOutQD, kStrideGradOutKB,
                  kStrideGradOutKHTransposed, kStrideGradOutKTTransposed,
                  kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                  kStrideGradOutVT, kStrideGradOutVD>
                  <<<grid, block, smem_bytes, stream>>>(
                  combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(),
                  logit_bias.data_ptr<float>(), out_q.data_ptr<at::BFloat16>(),
                  out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                  grad_out_q.data_ptr<at::BFloat16>(),
                  grad_out_k.data_ptr<at::BFloat16>(),
                  grad_out_v.data_ptr<at::BFloat16>(),
                  grad_combined.data_ptr<at::BFloat16>(),
                  grad_inits.data_ptr<at::BFloat16>(),
                  grad_logit_bias.data_ptr<float>(),
                  grad_logit_accum.data_ptr<float>());
              C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
        copy_grad_logits();
        return;
      }
      warn_specialized_backward_stride_mismatch(
          sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
          s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d,
          s_hv_b, s_hv_h, s_hv_t, s_hv_d, s_gq_b, s_gq_h, s_gq_t, s_gq_d,
          s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d,
          kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
          kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
          kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
          kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
          kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
          kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
          kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD);
    }

    TORCH_WARN_ONCE("ltv_fused_concat_cub_backward: using generic kernel ...");
    AT_DISPATCH_SWITCH(
        dtype, "ltv_fused_concat_cub_backward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
          ltv_fused_concat_head_major_register_cast_cub_backward_kernel<
              float, float, float, kScanThreads, kRThreads, kRItems>
              <<<grid, block, smem_bytes, stream>>>(
              combined.data_ptr<float>(), inits.data_ptr<float>(),
              logit_bias.data_ptr<float>(), out_q.data_ptr<float>(),
              out_k.data_ptr<float>(), out_v.data_ptr<float>(),
              grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(),
              grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(),
              grad_inits.data_ptr<float>(), grad_logit_bias.data_ptr<float>(),
              grad_logit_accum.data_ptr<float>(), sc_in_b, sc_in_t, sc_in_e,
              sc_out_b, sc_out_t, sc_out_e, s_hq_b, s_hq_h, s_hq_t, s_hq_d,
              s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
              s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d,
              s_gv_b, s_gv_h, s_gv_t, s_gv_d, T, NH, NKVH, Da, r, use_sigmoid);
          C10_CUDA_KERNEL_LAUNCH_CHECK(); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                             {
          ltv_fused_concat_head_major_register_cast_cub_backward_kernel<
              at::BFloat16, float, at::BFloat16, kScanThreads, kRThreads, kRItems>
              <<<grid, block, smem_bytes, stream>>>(
              combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(),
              logit_bias.data_ptr<float>(), out_q.data_ptr<at::BFloat16>(),
              out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
              grad_out_q.data_ptr<at::BFloat16>(),
              grad_out_k.data_ptr<at::BFloat16>(),
              grad_out_v.data_ptr<at::BFloat16>(),
              grad_combined.data_ptr<at::BFloat16>(),
              grad_inits.data_ptr<at::BFloat16>(),
              grad_logit_bias.data_ptr<float>(),
              grad_logit_accum.data_ptr<float>(), sc_in_b, sc_in_t, sc_in_e,
              sc_out_b, sc_out_t, sc_out_e, s_hq_b, s_hq_h, s_hq_t, s_hq_d,
              s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
              s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d,
              s_gv_b, s_gv_h, s_gv_t, s_gv_d, T, NH, NKVH, Da, r, use_sigmoid);
          C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
    copy_grad_logits();
  }
  else
  {
    TORCH_CHECK(false, "Invalid dispatch parameters (shared memory overflow). ",
                "scan_threads=", kScanThreads, ", r_threads=", kRThreads,
                ", r_items=", kRItems);
  }
}

// -------------------------------------------------------------------------
// Outer dispatch (unchanged)
// -------------------------------------------------------------------------
template <typename T_compute, int kScanThreads>
void backward_dispatch_outer(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_combined,
    torch::Tensor grad_inits, torch::Tensor grad_logit_bias, int B, int T,
    int NH, int NKVH, int Da, int r, int r_threads, int r_items,
    bool use_sigmoid, bool backward_has_expected_grad_output_strides,
    bool backward_has_transposed_qk_grad_output_strides,
    bool backward_has_specialized_strides, dim3 grid, cudaStream_t stream,
    at::ScalarType dtype, int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e, int64_t s_hq_b,
    int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d, int64_t s_hk_b,
    int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d, int64_t s_hv_b,
    int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d, int64_t s_gq_b,
    int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d, int64_t s_gk_b,
    int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d, int64_t s_gv_b,
    int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  DISPATCH_R_THREADS(r_threads, {
    DISPATCH_R_ITEMS(r_items, {
      backward_dispatch_inner<T_compute, kScanThreads, kRThreads, kRItems>(
          combined, inits, logit_bias, out_q, out_k, out_v, grad_out_q,
          grad_out_k, grad_out_v, grad_combined, grad_inits, grad_logit_bias, B,
          T, NH, NKVH, Da, r, use_sigmoid,
          backward_has_expected_grad_output_strides,
          backward_has_transposed_qk_grad_output_strides,
          backward_has_specialized_strides, grid, stream, dtype, sc_in_b,
          sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e, s_hq_b, s_hq_h,
          s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h,
          s_hv_t, s_hv_d, s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h,
          s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d);
    });
  });
}

// -------------------------------------------------------------------------
// Launcher (unchanged)
// -------------------------------------------------------------------------
void ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_combined,
    torch::Tensor grad_inits, torch::Tensor grad_logit_bias, int B, int T,
    int NH, int NKVH, int Da, int r, int scan_threads, int r_threads,
    int r_items, bool use_sigmoid)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t sc_in_b = combined.stride(0);
  int64_t sc_in_t = combined.stride(1);
  int64_t sc_in_e = combined.stride(2);
  int64_t sc_out_b = grad_combined.stride(0);
  int64_t sc_out_t = grad_combined.stride(1);
  int64_t sc_out_e = grad_combined.stride(2);

  int64_t s_hq_b = out_q.stride(0);
  int64_t s_hq_h = out_q.stride(1);
  int64_t s_hq_t = out_q.stride(2);
  int64_t s_hq_d = out_q.stride(3);
  int64_t s_hk_b = out_k.stride(0);
  int64_t s_hk_h = out_k.stride(1);
  int64_t s_hk_t = out_k.stride(2);
  int64_t s_hk_d = out_k.stride(3);
  int64_t s_hv_b = out_v.stride(0);
  int64_t s_hv_h = out_v.stride(1);
  int64_t s_hv_t = out_v.stride(2);
  int64_t s_hv_d = out_v.stride(3);

  int64_t s_gq_b = grad_out_q.stride(0);
  int64_t s_gq_h = grad_out_q.stride(1);
  int64_t s_gq_t = grad_out_q.stride(2);
  int64_t s_gq_d = grad_out_q.stride(3);
  int64_t s_gk_b = grad_out_k.stride(0);
  int64_t s_gk_h = grad_out_k.stride(1);
  int64_t s_gk_t = grad_out_k.stride(2);
  int64_t s_gk_d = grad_out_k.stride(3);
  int64_t s_gv_b = grad_out_v.stride(0);
  int64_t s_gv_h = grad_out_v.stride(1);
  int64_t s_gv_t = grad_out_v.stride(2);
  int64_t s_gv_d = grad_out_v.stride(3);

  int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
  dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks);
  auto dtype = combined.scalar_type();

  bool backward_has_expected_saved_output_strides =
      s_hq_b == kStrideGradOutQB && s_hq_h == kStrideGradOutQH &&
      s_hq_t == kStrideGradOutQT && s_hq_d == kStrideGradOutQD &&
      s_hk_b == kStrideGradOutKB && s_hk_h == kStrideGradOutKH &&
      s_hk_t == kStrideGradOutKT && s_hk_d == kStrideGradOutKD &&
      s_hv_b == kStrideGradOutVB && s_hv_h == kStrideGradOutVH &&
      s_hv_t == kStrideGradOutVT && s_hv_d == kStrideGradOutVD;

  bool backward_has_expected_grad_output_strides =
      s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQH &&
      s_gq_t == kStrideGradOutQT && s_gq_d == kStrideGradOutQD &&
      s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKH &&
      s_gk_t == kStrideGradOutKT && s_gk_d == kStrideGradOutKD &&
      s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH &&
      s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD;

  bool backward_has_transposed_qk_grad_output_strides =
      s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQHTransposed &&
      s_gq_t == kStrideGradOutQTTransposed && s_gq_d == kStrideGradOutQD &&
      s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKHTransposed &&
      s_gk_t == kStrideGradOutKTTransposed && s_gk_d == kStrideGradOutKD &&
      s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH &&
      s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD;

  constexpr int64_t kStrideCombinedB = kStrideCombinedB_f(true);

  bool backward_has_specialized_strides =
      sc_in_b == kStrideCombinedB && sc_in_t == kStrideCombinedT &&
      sc_in_e == kStrideCombinedE && sc_out_b == kStrideCombinedB &&
      sc_out_t == kStrideCombinedT && sc_out_e == kStrideCombinedE &&
      backward_has_expected_saved_output_strides &&
      (backward_has_expected_grad_output_strides ||
       backward_has_transposed_qk_grad_output_strides);

  DISPATCH_SCAN_THREADS(scan_threads, {
    backward_dispatch_outer<float, kScanThreads>(
        combined, inits, logit_bias, out_q, out_k, out_v, grad_out_q,
        grad_out_k, grad_out_v, grad_combined, grad_inits, grad_logit_bias, B,
        T, NH, NKVH, Da, r, r_threads, r_items, use_sigmoid,
        backward_has_expected_grad_output_strides,
        backward_has_transposed_qk_grad_output_strides,
        backward_has_specialized_strides, grid, stream, dtype, sc_in_b, sc_in_t,
        sc_in_e, sc_out_b, sc_out_t, sc_out_e, s_hq_b, s_hq_h, s_hq_t, s_hq_d,
        s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d, s_gq_b,
        s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h,
        s_gv_t, s_gv_d);
  });
}
