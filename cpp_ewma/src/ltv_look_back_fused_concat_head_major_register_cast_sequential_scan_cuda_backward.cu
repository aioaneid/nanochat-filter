#include <cub/cub.cuh>

#include "ltv_fused_concat_head_major_register_cast_sequential_scan_cuda_common.cuh"

#include "ltv_look_back_fused_concat_head_major_register_cast_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Macro that expands to the full sequential‑scan backward device function body.
// Parameters:
//   T_in, T_compute, T_out, kRThreads, kRItems,
//   UseSigmoid, P, Q, K,
//   combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
//     grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum,
//   sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
//   str_hq_b, str_hq_h, str_hq_t, str_hq_d, ... all stride arguments,
//   T_seq, NH, NKVH, Da, r
// -------------------------------------------------------------------------
#define LTV_LOOK_BACK_FUSED_CONCAT_SEQ_BACKWARD_DEVICE_BODY(                                                            \
    T_in, T_compute, T_out, kRThreads, kRItems,                                                                         \
    UseSigmoid, P, Q, K,                                                                                                \
    combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,                                                 \
    grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum,                                                 \
    sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,                                                            \
    str_hq_b, str_hq_h, str_hq_t, str_hq_d,                                                                             \
    str_hk_b, str_hk_h, str_hk_t, str_hk_d,                                                                             \
    str_hv_b, str_hv_h, str_hv_t, str_hv_d,                                                                             \
    str_gq_b, str_gq_h, str_gq_t, str_gq_d,                                                                             \
    str_gk_b, str_gk_h, str_gk_t, str_gk_d,                                                                             \
    str_gv_b, str_gv_h, str_gv_t, str_gv_d,                                                                             \
    T_seq, NH, NKVH, Da, r)                                                                                             \
  do                                                                                                                    \
  {                                                                                                                     \
    const int tid = threadIdx.x;                                                                                        \
    const int r_stride = kRThreads * kRItems;                                                                           \
                                                                                                                        \
    const int b_idx = blockIdx.x;                                                                                       \
    const int h_glob = blockIdx.y;                                                                                      \
                                                                                                                        \
    int M_dyn = (T_seq > P) ? (T_seq - P + Q - 1) / Q : 0;                                                              \
    const int num_t_chunks = 1 + M_dyn;                                                                                 \
    const int num_r_chunks = (r + r_stride - 1) / r_stride;                                                             \
                                                                                                                        \
    const int z = blockIdx.z;                                                                                           \
    const int t_chunk = z % num_t_chunks;                                                                               \
    const int rz = z / num_t_chunks;                                                                                    \
    const int da_idx = rz % Da;                                                                                         \
    const int r_chunk = rz / Da;                                                                                        \
                                                                                                                        \
    if (r_chunk >= num_r_chunks)                                                                                        \
      return;                                                                                                           \
                                                                                                                        \
    const unsigned int D = Da * r;                                                                                      \
    const unsigned int total_h = NH + 2 * NKVH;                                                                         \
    const unsigned int head_stride = D + Da;                                                                            \
                                                                                                                        \
    const T_out *h_ptr;                                                                                                 \
    const T_out *grad_h_ptr;                                                                                            \
    unsigned int h_loc;                                                                                                 \
    int64_t str_h_b, str_h_h, str_h_t, str_h_d;                                                                         \
    int64_t str_gh_b, str_gh_h, str_gh_t, str_gh_d;                                                                     \
                                                                                                                        \
    if (h_glob < NH)                                                                                                    \
    {                                                                                                                   \
      h_ptr = hq;                                                                                                       \
      grad_h_ptr = grad_hq;                                                                                             \
      h_loc = h_glob;                                                                                                   \
      str_h_b = str_hq_b;                                                                                               \
      str_h_h = str_hq_h;                                                                                               \
      str_h_t = str_hq_t;                                                                                               \
      str_h_d = str_hq_d;                                                                                               \
      str_gh_b = str_gq_b;                                                                                              \
      str_gh_h = str_gq_h;                                                                                              \
      str_gh_t = str_gq_t;                                                                                              \
      str_gh_d = str_gq_d;                                                                                              \
    }                                                                                                                   \
    else if (h_glob < NH + NKVH)                                                                                        \
    {                                                                                                                   \
      h_ptr = hk;                                                                                                       \
      grad_h_ptr = grad_hk;                                                                                             \
      h_loc = h_glob - NH;                                                                                              \
      str_h_b = str_hk_b;                                                                                               \
      str_h_h = str_hk_h;                                                                                               \
      str_h_t = str_hk_t;                                                                                               \
      str_h_d = str_hk_d;                                                                                               \
      str_gh_b = str_gk_b;                                                                                              \
      str_gh_h = str_gk_h;                                                                                              \
      str_gh_t = str_gk_t;                                                                                              \
      str_gh_d = str_gk_d;                                                                                              \
    }                                                                                                                   \
    else                                                                                                                \
    {                                                                                                                   \
      h_ptr = hv;                                                                                                       \
      grad_h_ptr = grad_hv;                                                                                             \
      h_loc = h_glob - (NH + NKVH);                                                                                     \
      str_h_b = str_hv_b;                                                                                               \
      str_h_h = str_hv_h;                                                                                               \
      str_h_t = str_hv_t;                                                                                               \
      str_h_d = str_hv_d;                                                                                               \
      str_gh_b = str_gv_b;                                                                                              \
      str_gh_h = str_gv_h;                                                                                              \
      str_gh_t = str_gv_t;                                                                                              \
      str_gh_d = str_gv_d;                                                                                              \
    }                                                                                                                   \
                                                                                                                        \
    const T_compute bias = logit_bias[h_glob * Da + da_idx];                                                            \
                                                                                                                        \
    __shared__ typename cub::BlockReduce<T_compute, kRThreads>::TempStorage temp_storage;                               \
    __shared__ T_compute shm_alpha;                                                                                     \
    __shared__ T_compute shm_d_l_multiplier;                                                                            \
    T_compute bias_sum_local = T_compute(0);                                                                            \
                                                                                                                        \
    int t_len = (t_chunk == 0) ? P : Q;                                                                                 \
    int t_start = (t_chunk == 0) ? 0 : (P + (t_chunk - 1) * Q);                                                         \
    if (t_chunk > 0 && t_start >= T_seq)                                                                                \
      return;                                                                                                           \
    int t_end = min(t_start + t_len, T_seq);                                                                            \
    int t_suf_end = min(t_end, T_seq - K) + K;                                                                          \
    int t_pref_start = max(0, t_start - (K - 1));                                                                       \
                                                                                                                        \
    const int64_t b_offset_in = (int64_t)b_idx * sc_in_b;                                                               \
    const int64_t b_offset_out = (int64_t)b_idx * sc_out_b;                                                             \
                                                                                                                        \
    const int64_t da_r = da_idx * r;                                                                                    \
    const int64_t h_base = (int64_t)b_idx * str_h_b + (int64_t)h_loc * str_h_h + da_r * str_h_d;                        \
    const int64_t gh_base = (int64_t)b_idx * str_gh_b + (int64_t)h_loc * str_gh_h + da_r * str_gh_d;                    \
    const int64_t init_base = (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + da_r;                              \
    const int64_t val_in_base = b_offset_in + (int64_t)(h_glob * head_stride + da_idx * r) * sc_in_e;                   \
    const int64_t val_out_base = b_offset_out + (int64_t)(h_glob * head_stride + da_idx * r) * sc_out_e;                \
    const int64_t logit_in_base = b_offset_in + (int64_t)(h_glob * head_stride + D + da_idx) * sc_in_e;                 \
                                                                                                                        \
    int r_idx_global[kRItems];                                                                                          \
    bool valid_r[kRItems];                                                                                              \
    _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                                 \
    {                                                                                                                   \
      r_idx_global[i] = r_chunk * r_stride + i * kRThreads + tid;                                                       \
      valid_r[i] = (r_idx_global[i] < r);                                                                               \
    }                                                                                                                   \
                                                                                                                        \
    T_compute running_grad[kRItems];                                                                                    \
    _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                                 \
        running_grad[i] = T_compute(0);                                                                                 \
                                                                                                                        \
    /* Phase 1: Suffix */                                                                                               \
    for (int t = t_suf_end - 1; t >= t_end; --t)                                                                        \
    {                                                                                                                   \
      T_compute alpha;                                                                                                  \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        T_compute logit = (T_compute)combined[logit_in_base + (int64_t)t * sc_in_t] + bias;                             \
        alpha = UseSigmoid ? sigmoid_f32(logit) : logit;                                                                \
        shm_alpha = alpha;                                                                                              \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
      if (tid != 0)                                                                                                     \
      {                                                                                                                 \
        alpha = shm_alpha;                                                                                              \
      }                                                                                                                 \
      T_compute one_minus_alpha = T_compute(1) - alpha;                                                                 \
      _Pragma("unroll") for (int i = 0; i < kRItems; ++i) if (valid_r[i]) running_grad[i] *= one_minus_alpha;           \
      __syncthreads();                                                                                                  \
    }                                                                                                                   \
                                                                                                                        \
    /* Phase 2: Main Body */                                                                                            \
    int main_end = (t_chunk == 0) ? 1 : t_start;                                                                        \
    for (int t = t_end - 1; t >= main_end; --t)                                                                         \
    {                                                                                                                   \
      T_compute alpha, d_l_multiplier;                                                                                  \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        T_compute logit = (T_compute)combined[logit_in_base + (int64_t)t * sc_in_t] + bias;                             \
        if constexpr (UseSigmoid)                                                                                       \
        {                                                                                                               \
          T_compute sig = sigmoid_f32(logit);                                                                           \
          alpha = sig;                                                                                                  \
          d_l_multiplier = sig * (T_compute(1) - sig);                                                                  \
        }                                                                                                               \
        else                                                                                                            \
        {                                                                                                               \
          alpha = logit;                                                                                                \
          d_l_multiplier = T_compute(1);                                                                                \
        }                                                                                                               \
        shm_alpha = alpha;                                                                                              \
        shm_d_l_multiplier = d_l_multiplier;                                                                            \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
      if (tid != 0)                                                                                                     \
      {                                                                                                                 \
        alpha = shm_alpha;                                                                                              \
        d_l_multiplier = shm_d_l_multiplier;                                                                            \
      }                                                                                                                 \
                                                                                                                        \
      T_compute one_minus_alpha = T_compute(1) - alpha;                                                                 \
      T_compute thread_logit_grad = T_compute(0);                                                                       \
                                                                                                                        \
      _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                               \
      {                                                                                                                 \
        if (!valid_r[i])                                                                                                \
          continue;                                                                                                     \
        int r_g = r_idx_global[i];                                                                                      \
        int64_t gh_idx = gh_base + (int64_t)r_g * str_gh_d + (int64_t)t * str_gh_t;                                     \
        T_compute g_cur = (T_compute)grad_h_ptr[gh_idx] + running_grad[i];                                              \
        T_compute d_x = alpha * g_cur;                                                                                  \
        int64_t out_val_idx = val_out_base + (int64_t)r_g * sc_out_e + (int64_t)t * sc_out_t;                           \
        grad_combined[out_val_idx] = (T_out)d_x;                                                                        \
        int64_t in_val_idx = val_in_base + (int64_t)r_g * sc_in_e + (int64_t)t * sc_in_t;                               \
        int64_t h_prev_idx = h_base + (int64_t)r_g * str_h_d + (int64_t)(t - 1) * str_h_t;                              \
        T_compute x_val = (T_compute)combined[in_val_idx];                                                              \
        T_compute h_prev = (T_compute)h_ptr[h_prev_idx];                                                                \
        T_compute d_alpha = (x_val - h_prev) * g_cur;                                                                   \
        thread_logit_grad += d_alpha * d_l_multiplier;                                                                  \
        running_grad[i] = one_minus_alpha * g_cur;                                                                      \
      }                                                                                                                 \
                                                                                                                        \
      T_compute total_logit_grad = cub::BlockReduce<T_compute, kRThreads>(temp_storage).Sum(thread_logit_grad);         \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        int64_t accum_idx = (((int64_t)b_idx * total_h + (int64_t)h_glob) * Da + (int64_t)da_idx) * T_seq + (int64_t)t; \
        atomicAdd(&grad_logit_accum[accum_idx], total_logit_grad);                                                      \
        bias_sum_local += total_logit_grad;                                                                             \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
    }                                                                                                                   \
                                                                                                                        \
    /* Phase 3: First Step (t=0) */                                                                                     \
    if (t_chunk == 0 && t_end > 0)                                                                                      \
    {                                                                                                                   \
      int t = 0;                                                                                                        \
      T_compute alpha, d_l_multiplier;                                                                                  \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        T_compute logit = (T_compute)combined[logit_in_base + (int64_t)t * sc_in_t] + bias;                             \
        if constexpr (UseSigmoid)                                                                                       \
        {                                                                                                               \
          T_compute sig = sigmoid_f32(logit);                                                                           \
          alpha = sig;                                                                                                  \
          d_l_multiplier = sig * (T_compute(1) - sig);                                                                  \
        }                                                                                                               \
        else                                                                                                            \
        {                                                                                                               \
          alpha = logit;                                                                                                \
          d_l_multiplier = T_compute(1);                                                                                \
        }                                                                                                               \
        shm_alpha = alpha;                                                                                              \
        shm_d_l_multiplier = d_l_multiplier;                                                                            \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
      if (tid != 0)                                                                                                     \
      {                                                                                                                 \
        alpha = shm_alpha;                                                                                              \
        d_l_multiplier = shm_d_l_multiplier;                                                                            \
      }                                                                                                                 \
                                                                                                                        \
      T_compute one_minus_alpha = T_compute(1) - alpha;                                                                 \
      T_compute thread_logit_grad = T_compute(0);                                                                       \
                                                                                                                        \
      _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                               \
      {                                                                                                                 \
        if (!valid_r[i])                                                                                                \
          continue;                                                                                                     \
        int r_g = r_idx_global[i];                                                                                      \
        int64_t gh_idx = gh_base + (int64_t)r_g * str_gh_d + (int64_t)t * str_gh_t;                                     \
        T_compute g_cur = (T_compute)grad_h_ptr[gh_idx] + running_grad[i];                                              \
        T_compute d_x = alpha * g_cur;                                                                                  \
        int64_t out_val_idx = val_out_base + (int64_t)r_g * sc_out_e + (int64_t)t * sc_out_t;                           \
        grad_combined[out_val_idx] = (T_out)d_x;                                                                        \
        int64_t in_val_idx = val_in_base + (int64_t)r_g * sc_in_e + (int64_t)t * sc_in_t;                               \
        T_compute x_val = (T_compute)combined[in_val_idx];                                                              \
        T_compute h_prev = (T_compute)inits[init_base + r_g];                                                           \
        T_compute d_alpha = (x_val - h_prev) * g_cur;                                                                   \
        thread_logit_grad += d_alpha * d_l_multiplier;                                                                  \
        running_grad[i] = one_minus_alpha * g_cur;                                                                      \
      }                                                                                                                 \
                                                                                                                        \
      T_compute total_logit_grad = cub::BlockReduce<T_compute, kRThreads>(temp_storage).Sum(thread_logit_grad);         \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        int64_t accum_idx = (((int64_t)b_idx * total_h + (int64_t)h_glob) * Da + (int64_t)da_idx) * T_seq + (int64_t)t; \
        atomicAdd(&grad_logit_accum[accum_idx], total_logit_grad);                                                      \
        bias_sum_local += total_logit_grad;                                                                             \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
    }                                                                                                                   \
                                                                                                                        \
    /* Phase 4: Prefix */                                                                                               \
    for (int t = t_start - 1; t >= t_pref_start; --t)                                                                   \
    {                                                                                                                   \
      T_compute alpha;                                                                                                  \
      if (tid == 0)                                                                                                     \
      {                                                                                                                 \
        T_compute logit = (T_compute)combined[logit_in_base + (int64_t)t * sc_in_t] + bias;                             \
        alpha = UseSigmoid ? sigmoid_f32(logit) : logit;                                                                \
        shm_alpha = alpha;                                                                                              \
      }                                                                                                                 \
      __syncthreads();                                                                                                  \
      if (tid != 0)                                                                                                     \
      {                                                                                                                 \
        alpha = shm_alpha;                                                                                              \
      }                                                                                                                 \
      T_compute one_minus_alpha = T_compute(1) - alpha;                                                                 \
      _Pragma("unroll") for (int i = 0; i < kRItems; ++i) if (valid_r[i]) running_grad[i] *= one_minus_alpha;           \
      __syncthreads();                                                                                                  \
    }                                                                                                                   \
                                                                                                                        \
    /* Inits Accumulation */                                                                                            \
    _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                                 \
    {                                                                                                                   \
      if (valid_r[i])                                                                                                   \
      {                                                                                                                 \
        int64_t init_idx = init_base + r_idx_global[i];                                                                 \
        atomicAdd(&grad_inits_accum[init_idx], running_grad[i]);                                                        \
      }                                                                                                                 \
    }                                                                                                                   \
                                                                                                                        \
    /* Bias gradient write */                                                                                           \
    if (tid == 0)                                                                                                       \
    {                                                                                                                   \
      int bias_idx = h_glob * Da + da_idx;                                                                              \
      atomicAdd(&grad_logit_bias[bias_idx], bias_sum_local);                                                            \
    }                                                                                                                   \
  } while (0)

// -------------------------------------------------------------------------
// Generic backward device function (runtime strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__device__ __forceinline__ void
ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_backward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_compute *__restrict__ grad_inits_accum,
    T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e, int64_t sc_out_b,
    int64_t sc_out_t, int64_t sc_out_e, int64_t str_hq_b, int64_t str_hq_h,
    int64_t str_hq_t, int64_t str_hq_d, int64_t str_hk_b, int64_t str_hk_h,
    int64_t str_hk_t, int64_t str_hk_d, int64_t str_hv_b, int64_t str_hv_h,
    int64_t str_hv_t, int64_t str_hv_d, int64_t str_gq_b, int64_t str_gq_h,
    int64_t str_gq_t, int64_t str_gq_d, int64_t str_gk_b, int64_t str_gk_h,
    int64_t str_gk_t, int64_t str_gk_d, int64_t str_gv_b, int64_t str_gv_h,
    int64_t str_gv_t, int64_t str_gv_d, int T_seq, int NH, int NKVH, int Da, int r)
{
  LTV_LOOK_BACK_FUSED_CONCAT_SEQ_BACKWARD_DEVICE_BODY(
      T_in, T_compute, T_out, kRThreads, kRItems,
      UseSigmoid, P, Q, K,
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum,
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
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
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
ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_backward_device_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_compute *__restrict__ grad_inits_accum,
    T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum)
{
  LTV_LOOK_BACK_FUSED_CONCAT_SEQ_BACKWARD_DEVICE_BODY(
      T_in, T_compute, T_out, kRThreads, kRItems,
      UseSigmoid, P, Q, K,
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum,
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
// Backward kernels (generic and constant‑strides)
// -------------------------------------------------------------------------

// 1) Generic kernel (unchanged, calls generic device function)
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__global__ void ltv_look_back_fused_concat_sequential_scan_backward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_compute *__restrict__ grad_inits_accum,
    T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e, int64_t sc_out_b,
    int64_t sc_out_t, int64_t sc_out_e, int64_t s_hq_b, int64_t s_hq_h,
    int64_t s_hq_t, int64_t s_hq_d, int64_t s_hk_b, int64_t s_hk_h,
    int64_t s_hk_t, int64_t s_hk_d, int64_t s_hv_b, int64_t s_hv_h,
    int64_t s_hv_t, int64_t s_hv_d, int64_t s_gq_b, int64_t s_gq_h,
    int64_t s_gq_t, int64_t s_gq_d, int64_t s_gk_b, int64_t s_gk_h,
    int64_t s_gk_t, int64_t s_gk_d, int64_t s_gv_b, int64_t s_gv_h,
    int64_t s_gv_t, int64_t s_gv_d, int T_seq, int NH, int NKVH, int Da, int r)
{
  ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_backward_device<
      T_in, T_compute, T_out, kRThreads, kRItems,
      UseSigmoid, P, Q, K>(
      combined, inits, logit_bias, hq, hk, hv, grad_hq, grad_hk, grad_hv,
      grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum,
      sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
      s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d,
      s_hv_b, s_hv_h, s_hv_t, s_hv_d, s_gq_b, s_gq_h, s_gq_t, s_gq_d,
      s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d,
      T_seq, NH, NKVH, Da, r);
}

// 2) Constant‑strides kernel (now calls const device function)
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
          int T_const, int NH_const, int NKVH_const,
          int Da_const, int r_const,
          int64_t sc_in_b_const, int64_t sc_in_t_const, int64_t sc_in_e_const,
          int64_t sc_out_b_const, int64_t sc_out_t_const, int64_t sc_out_e_const,
          int64_t s_hq_b_const, int64_t s_hq_h_const, int64_t s_hq_t_const,
          int64_t s_hq_d_const,
          int64_t s_hk_b_const, int64_t s_hk_h_const, int64_t s_hk_t_const,
          int64_t s_hk_d_const,
          int64_t s_hv_b_const, int64_t s_hv_h_const, int64_t s_hv_t_const,
          int64_t s_hv_d_const,
          int64_t s_gq_b_const, int64_t s_gq_h_const, int64_t s_gq_t_const,
          int64_t s_gq_d_const,
          int64_t s_gk_b_const, int64_t s_gk_h_const, int64_t s_gk_t_const,
          int64_t s_gk_d_const,
          int64_t s_gv_b_const, int64_t s_gv_h_const, int64_t s_gv_t_const,
          int64_t s_gv_d_const>
__global__ void ltv_look_back_fused_concat_sequential_scan_backward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, const T_out *__restrict__ hq,
    const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk,
    const T_out *__restrict__ grad_hv, T_out *__restrict__ grad_combined,
    T_compute *__restrict__ grad_inits_accum,
    T_compute *__restrict__ grad_logit_bias,
    T_compute *__restrict__ grad_logit_accum)
{
  ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_backward_device_const<
      T_in, T_compute, T_out, kRThreads, kRItems,
      UseSigmoid, P, Q, K,
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
      grad_combined, grad_inits_accum, grad_logit_bias, grad_logit_accum);
}

// -------------------------------------------------------------------------
// Shared memory size and validity helpers
// -------------------------------------------------------------------------
template <typename T_compute, int kRThreads>
constexpr size_t backward_shared_memory_bytes()
{
  return sizeof(typename cub::BlockReduce<T_compute, kRThreads>::TempStorage) + 2 * sizeof(T_compute);
}

template <int kRThreads>
constexpr bool is_dispatch_valid_backward()
{
  return kRThreads <= 1024;
}

// -------------------------------------------------------------------------
// Dispatch inner helper (unchanged, calls const kernel with constants)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int cP, int cQ, int cK, bool cSig,
          int kRThreads, int kRItems>
void ltv_look_back_backward_dispatch_inner(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, const torch::Tensor &out_q,
    const torch::Tensor &out_k, const torch::Tensor &out_v,
    const torch::Tensor &grad_out_q, const torch::Tensor &grad_out_k,
    const torch::Tensor &grad_out_v, torch::Tensor &grad_combined,
    torch::Tensor &grad_inits, torch::Tensor &grad_logit_bias,
    int B, int T, int NH, int NKVH, int Da, int r,
    bool backward_has_expected_grad_output_strides,
    bool backward_has_transposed_qk_grad_output_strides,
    bool backward_has_transposed_qkv_grad_output_strides,
    bool backward_has_specialized_strides,
    dim3 grid, cudaStream_t stream,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t s_hq_b, int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d,
    int64_t s_hk_b, int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d,
    int64_t s_hv_b, int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  constexpr bool kValid = is_dispatch_valid_backward<kRThreads>();
  if constexpr (kValid)
  {
    int smem_bytes = static_cast<int>(backward_shared_memory_bytes<T_compute, kRThreads>());
    dim3 block(kRThreads);
    int total_h = NH + 2 * NKVH;
    int D = Da * r;

    auto options = torch::TensorOptions()
                       .dtype(c10::CppTypeToScalarType<T_compute>::value)
                       .device(combined.device());

    torch::Tensor grad_logit_accum = torch::zeros({B, total_h, Da, T}, options);
    torch::Tensor grad_inits_accum = torch::zeros({B, total_h, D}, options);

    auto copy_grad_logits = [&]()
    {
      auto grad_logit_accum_t = grad_logit_accum.permute({0, 3, 1, 2});
      auto grad_logits_view = grad_combined.view({B, T, total_h, D + Da}).slice(3, D, D + Da);
      grad_logits_view.copy_(grad_logit_accum_t);
      grad_inits.copy_(grad_inits_accum);
    };

    constexpr int64_t kStrideCombinedB = kSeqStrideCombinedB;
    constexpr int64_t kStrideCombinedT = kSeqStrideCombinedT;
    constexpr int64_t kStrideCombinedE = kSeqStrideCombinedE;

    if (backward_has_specialized_strides &&
        (backward_has_expected_grad_output_strides ||
         backward_has_transposed_qk_grad_output_strides ||
         backward_has_transposed_qkv_grad_output_strides))
    {
      if (backward_has_expected_grad_output_strides)
      {
        ltv_look_back_fused_concat_sequential_scan_backward_kernel_const<
            T_in, T_compute, T_out, kRThreads, kRItems, cSig,
            cP, cQ, cK, kT, kNH, kNKVH, kDa, kR,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
            kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
            <<<grid, block, smem_bytes, stream>>>(
                combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
                logit_bias.data_ptr<T_compute>(),
                out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(), out_v.data_ptr<T_out>(),
                grad_out_q.data_ptr<T_out>(), grad_out_k.data_ptr<T_out>(), grad_out_v.data_ptr<T_out>(),
                grad_combined.data_ptr<T_out>(), grad_inits_accum.data_ptr<T_compute>(),
                grad_logit_bias.data_ptr<T_compute>(), grad_logit_accum.data_ptr<T_compute>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      else if (backward_has_transposed_qk_grad_output_strides)
      {
        ltv_look_back_fused_concat_sequential_scan_backward_kernel_const<
            T_in, T_compute, T_out, kRThreads, kRItems, cSig,
            cP, cQ, cK, kT, kNH, kNKVH, kDa, kR,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
            kStrideGradOutQB, kStrideGradOutQHTransposed, kStrideGradOutQTTransposed, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKHTransposed, kStrideGradOutKTTransposed, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
            <<<grid, block, smem_bytes, stream>>>(
                combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
                logit_bias.data_ptr<T_compute>(),
                out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(), out_v.data_ptr<T_out>(),
                grad_out_q.data_ptr<T_out>(), grad_out_k.data_ptr<T_out>(), grad_out_v.data_ptr<T_out>(),
                grad_combined.data_ptr<T_out>(), grad_inits_accum.data_ptr<T_compute>(),
                grad_logit_bias.data_ptr<T_compute>(), grad_logit_accum.data_ptr<T_compute>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      else if (backward_has_transposed_qkv_grad_output_strides)
      {
        ltv_look_back_fused_concat_sequential_scan_backward_kernel_const<
            T_in, T_compute, T_out, kRThreads, kRItems, cSig,
            cP, cQ, cK, kT, kNH, kNKVH, kDa, kR,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
            kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
            kStrideGradOutQB, kStrideGradOutQHTransposed, kStrideGradOutQTTransposed, kStrideGradOutQD,
            kStrideGradOutKB, kStrideGradOutKHTransposed, kStrideGradOutKTTransposed, kStrideGradOutKD,
            kStrideGradOutVB, kStrideGradOutVHTransposed, kStrideGradOutVTTransposed, kStrideGradOutVD>
            <<<grid, block, smem_bytes, stream>>>(
                combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
                logit_bias.data_ptr<T_compute>(),
                out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(), out_v.data_ptr<T_out>(),
                grad_out_q.data_ptr<T_out>(), grad_out_k.data_ptr<T_out>(), grad_out_v.data_ptr<T_out>(),
                grad_combined.data_ptr<T_out>(), grad_inits_accum.data_ptr<T_compute>(),
                grad_logit_bias.data_ptr<T_compute>(), grad_logit_accum.data_ptr<T_compute>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      copy_grad_logits();
      return;
    }
    else
    {
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

    TORCH_WARN_ONCE("ltv_look_back_backward: using generic kernel ...");
    ltv_look_back_fused_concat_sequential_scan_backward_kernel<
        T_in, T_compute, T_out, kRThreads, kRItems, cSig,
        cP, cQ, cK>
        <<<grid, block, smem_bytes, stream>>>(
            combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
            logit_bias.data_ptr<T_compute>(),
            out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(), out_v.data_ptr<T_out>(),
            grad_out_q.data_ptr<T_out>(), grad_out_k.data_ptr<T_out>(), grad_out_v.data_ptr<T_out>(),
            grad_combined.data_ptr<T_out>(), grad_inits_accum.data_ptr<T_compute>(),
            grad_logit_bias.data_ptr<T_compute>(), grad_logit_accum.data_ptr<T_compute>(),
            sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
            s_hq_b, s_hq_h, s_hq_t, s_hq_d,
            s_hk_b, s_hk_h, s_hk_t, s_hk_d,
            s_hv_b, s_hv_h, s_hv_t, s_hv_d,
            s_gq_b, s_gq_h, s_gq_t, s_gq_d,
            s_gk_b, s_gk_h, s_gk_t, s_gk_d,
            s_gv_b, s_gv_h, s_gv_t, s_gv_d,
            T, NH, NKVH, Da, r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    copy_grad_logits();
  }
  else
  {
    TORCH_CHECK(false, "Invalid backward dispatch parameters. ",
                "r_threads=", kRThreads, ", r_items=", kRItems);
  }
}

// -------------------------------------------------------------------------
// Outer dispatch (unchanged)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out>
void ltv_look_back_backward_dispatch(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, const torch::Tensor &out_q,
    const torch::Tensor &out_k, const torch::Tensor &out_v,
    const torch::Tensor &grad_out_q, const torch::Tensor &grad_out_k,
    const torch::Tensor &grad_out_v, torch::Tensor &grad_combined,
    torch::Tensor &grad_inits, torch::Tensor &grad_logit_bias,
    int B, int T, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int r_threads, int r_items, bool use_sigmoid,
    bool backward_has_expected_grad_output_strides,
    bool backward_has_transposed_qk_grad_output_strides,
    bool backward_has_transposed_qkv_grad_output_strides,
    bool backward_has_specialized_strides,
    dim3 grid, cudaStream_t stream,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t s_hq_b, int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d,
    int64_t s_hk_b, int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d,
    int64_t s_hv_b, int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
  if (P == 2147483647)
  {
    Q = 1;
    K = 1;
  }

#define LAUNCH_IF(Pv, Qv, Kv, Sigv)                                 \
  if (P == (Pv) && Q == (Qv) && K == (Kv) && use_sigmoid == (Sigv)) \
  {                                                                 \
    constexpr int cP = (Pv), cQ = (Qv), cK = (Kv);                  \
    constexpr bool cSig = (Sigv);                                   \
    DISPATCH_R_THREADS(r_threads, {                                 \
      DISPATCH_R_ITEMS(r_items, {                                   \
        ltv_look_back_backward_dispatch_inner<                      \
            T_in, T_compute, T_out,                                 \
            cP, cQ, cK, cSig,                                       \
            kRThreads, kRItems>(                                    \
            combined, inits, logit_bias, out_q, out_k, out_v,       \
            grad_out_q, grad_out_k, grad_out_v,                     \
            grad_combined, grad_inits, grad_logit_bias,             \
            B, T, NH, NKVH, Da, r,                                  \
            backward_has_expected_grad_output_strides,              \
            backward_has_transposed_qk_grad_output_strides,         \
            backward_has_transposed_qkv_grad_output_strides,        \
            backward_has_specialized_strides,                       \
            grid, stream, sc_in_b, sc_in_t, sc_in_e,                \
            sc_out_b, sc_out_t, sc_out_e,                           \
            s_hq_b, s_hq_h, s_hq_t, s_hq_d,                         \
            s_hk_b, s_hk_h, s_hk_t, s_hk_d,                         \
            s_hv_b, s_hv_h, s_hv_t, s_hv_d,                         \
            s_gq_b, s_gq_h, s_gq_t, s_gq_d,                         \
            s_gk_b, s_gk_h, s_gk_t, s_gk_d,                         \
            s_gv_b, s_gv_h, s_gv_t, s_gv_d);                        \
      });                                                           \
    });                                                             \
    return;                                                         \
  }

  LTV_LOOK_BACK_LAUNCH_EACH_PQK();

  TORCH_CHECK(false, "ltv_look_back_backward: unsupported dispatch configuration. "
                     " P=",
              P, " Q=", Q, " K=", K,
              " r_th=", r_threads, " r_it=", r_items, " sig=", use_sigmoid);

#undef LAUNCH_IF
}

// -------------------------------------------------------------------------
// Launcher (unchanged)
// -------------------------------------------------------------------------
void ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_combined,
    torch::Tensor grad_inits, torch::Tensor grad_logit_bias,
    int B, int T, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int r_threads, int r_items, bool use_sigmoid)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t sc_in_b = combined.stride(0);
  int64_t sc_in_t = combined.stride(1);
  int64_t sc_in_e = combined.stride(2);
  int64_t sc_out_b = grad_combined.stride(0);
  int64_t sc_out_t = grad_combined.stride(1);
  int64_t sc_out_e = grad_combined.stride(2);

  int64_t s_hq_b = out_q.stride(0), s_hq_h = out_q.stride(1), s_hq_t = out_q.stride(2), s_hq_d = out_q.stride(3);
  int64_t s_hk_b = out_k.stride(0), s_hk_h = out_k.stride(1), s_hk_t = out_k.stride(2), s_hk_d = out_k.stride(3);
  int64_t s_hv_b = out_v.stride(0), s_hv_h = out_v.stride(1), s_hv_t = out_v.stride(2), s_hv_d = out_v.stride(3);

  int64_t s_gq_b = grad_out_q.stride(0), s_gq_h = grad_out_q.stride(1), s_gq_t = grad_out_q.stride(2), s_gq_d = grad_out_q.stride(3);
  int64_t s_gk_b = grad_out_k.stride(0), s_gk_h = grad_out_k.stride(1), s_gk_t = grad_out_k.stride(2), s_gk_d = grad_out_k.stride(3);
  int64_t s_gv_b = grad_out_v.stride(0), s_gv_h = grad_out_v.stride(1), s_gv_t = grad_out_v.stride(2), s_gv_d = grad_out_v.stride(3);

  int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
  int M_dyn = (T > P) ? (T - P + Q - 1) / Q : 0;
  int num_t_chunks = 1 + M_dyn;
  dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks * num_t_chunks);

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

  bool qk_grad_transposed = s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQHTransposed &&
                            s_gq_t == kStrideGradOutQTTransposed && s_gq_d == kStrideGradOutQD &&
                            s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKHTransposed &&
                            s_gk_t == kStrideGradOutKTTransposed && s_gk_d == kStrideGradOutKD &&
                            s_gv_b == kStrideGradOutVB && s_gv_d == kStrideGradOutVD;

  bool backward_has_transposed_qk_grad_output_strides =
      qk_grad_transposed && s_gv_h == kStrideGradOutVH && s_gv_t == kStrideGradOutVT;

  bool backward_has_transposed_qkv_grad_output_strides =
      qk_grad_transposed && s_gv_h == kStrideGradOutVHTransposed && s_gv_t == kStrideGradOutVTTransposed;

  constexpr int64_t kStrideCombinedB = kSeqStrideCombinedB;
  constexpr int64_t kStrideCombinedT = kSeqStrideCombinedT;
  constexpr int64_t kStrideCombinedE = kSeqStrideCombinedE;

  bool backward_has_specialized_strides =
      sc_in_b == kStrideCombinedB && sc_in_t == kStrideCombinedT &&
      sc_in_e == kStrideCombinedE && sc_out_b == kStrideCombinedB &&
      sc_out_t == kStrideCombinedT && sc_out_e == kStrideCombinedE &&
      backward_has_expected_saved_output_strides &&
      (backward_has_expected_grad_output_strides ||
       backward_has_transposed_qk_grad_output_strides ||
       backward_has_transposed_qkv_grad_output_strides);

  AT_DISPATCH_SWITCH(
      dtype, "ltv_look_back_backward",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                       {
            using T_in = float;
            using T_compute = float;
            using T_out = float;
            ltv_look_back_backward_dispatch<T_in, T_compute, T_out>(
                combined, inits, logit_bias, out_q, out_k, out_v,
                grad_out_q, grad_out_k, grad_out_v,
                grad_combined, grad_inits, grad_logit_bias,
                B, T, NH, NKVH, Da, r,
                P, Q, K,
                r_threads, r_items, use_sigmoid,
                backward_has_expected_grad_output_strides,
                backward_has_transposed_qk_grad_output_strides,
                backward_has_transposed_qkv_grad_output_strides,
                backward_has_specialized_strides,
                grid, stream, sc_in_b, sc_in_t, sc_in_e,
                sc_out_b, sc_out_t, sc_out_e,
                s_hq_b, s_hq_h, s_hq_t, s_hq_d,
                s_hk_b, s_hk_h, s_hk_t, s_hk_d,
                s_hv_b, s_hv_h, s_hv_t, s_hv_d,
                s_gq_b, s_gq_h, s_gq_t, s_gq_d,
                s_gk_b, s_gk_h, s_gk_t, s_gk_d,
                s_gv_b, s_gv_h, s_gv_t, s_gv_d); })
          AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                           {
            using T_in = at::BFloat16;
            using T_compute = float;
            using T_out = at::BFloat16;
            ltv_look_back_backward_dispatch<T_in, T_compute, T_out>(
                combined, inits, logit_bias, out_q, out_k, out_v,
                grad_out_q, grad_out_k, grad_out_v,
                grad_combined, grad_inits, grad_logit_bias,
                B, T, NH, NKVH, Da, r,
                P, Q, K,
                r_threads, r_items, use_sigmoid,
                backward_has_expected_grad_output_strides,
                backward_has_transposed_qk_grad_output_strides,
                backward_has_transposed_qkv_grad_output_strides,
                backward_has_specialized_strides,
                grid, stream, sc_in_b, sc_in_t, sc_in_e,
                sc_out_b, sc_out_t, sc_out_e,
                s_hq_b, s_hq_h, s_hq_t, s_hq_d,
                s_hk_b, s_hk_h, s_hk_t, s_hk_d,
                s_hv_b, s_hv_h, s_hv_t, s_hv_d,
                s_gq_b, s_gq_h, s_gq_t, s_gq_d,
                s_gk_b, s_gk_h, s_gk_t, s_gk_d,
                s_gv_b, s_gv_h, s_gv_t, s_gv_d); }));
}
