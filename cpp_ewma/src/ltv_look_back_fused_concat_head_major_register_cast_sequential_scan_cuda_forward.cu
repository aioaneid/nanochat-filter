#include "ltv_fused_concat_head_major_register_cast_sequential_scan_cuda_common.cuh"
#include "ltv_look_back_fused_concat_head_major_register_cast_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Macro that expands to the full sequential‑scan forward device function body.
// Parameters:
//   T_in, T_compute, T_out, kRThreads, kRItems,
//   UseSigmoid, P, Q, K,
//   combined, inits, logit_bias, out_q, out_k, out_v,
//   sc_b, sc_t, sc_e,
//   T_seq, NH, NKVH, Da, r
// -------------------------------------------------------------------------
#define LTV_LOOK_BACK_FUSED_CONCAT_SEQ_FORWARD_DEVICE_BODY(                                                              \
    T_in, T_compute, T_out, kRThreads, kRItems,                                                                          \
    UseSigmoid, P, Q, K,                                                                                                 \
    combined, inits, logit_bias, out_q, out_k, out_v,                                                                    \
    sc_b, sc_t, sc_e,                                                                                                    \
    T_seq, NH, NKVH, Da, r)                                                                                              \
    do                                                                                                                   \
    {                                                                                                                    \
        const int tid = threadIdx.x;                                                                                     \
        const int r_stride = kRThreads * kRItems;                                                                        \
                                                                                                                         \
        const int b_idx = blockIdx.x;                                                                                    \
        const int h_glob = blockIdx.y;                                                                                   \
                                                                                                                         \
        int M_dyn = (T_seq > P) ? (T_seq - P + Q - 1) / Q : 0;                                                           \
        const int num_t_chunks = 1 + M_dyn;                                                                              \
        const int num_r_chunks = (r + r_stride - 1) / r_stride;                                                          \
                                                                                                                         \
        const int z = blockIdx.z;                                                                                        \
        const int t_chunk = z % num_t_chunks;                                                                            \
        const int rz = z / num_t_chunks;                                                                                 \
        const int da_idx = rz % Da;                                                                                      \
        const int r_chunk = rz / Da;                                                                                     \
                                                                                                                         \
        if (r_chunk >= num_r_chunks)                                                                                     \
            return;                                                                                                      \
                                                                                                                         \
        const unsigned int D = Da * r;                                                                                   \
        const unsigned int total_h = NH + 2 * NKVH;                                                                      \
        const unsigned int head_stride = D + Da;                                                                         \
                                                                                                                         \
        T_out *x_out;                                                                                                    \
        unsigned int h_loc, h_comp_total;                                                                                \
        if (h_glob < NH)                                                                                                 \
        {                                                                                                                \
            x_out = out_q;                                                                                               \
            h_loc = h_glob;                                                                                              \
            h_comp_total = NH;                                                                                           \
        }                                                                                                                \
        else if (h_glob < NH + NKVH)                                                                                     \
        {                                                                                                                \
            x_out = out_k;                                                                                               \
            h_loc = h_glob - NH;                                                                                         \
            h_comp_total = NKVH;                                                                                         \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            x_out = out_v;                                                                                               \
            h_loc = h_glob - (NH + NKVH);                                                                                \
            h_comp_total = NKVH;                                                                                         \
        }                                                                                                                \
                                                                                                                         \
        const T_compute bias_val = logit_bias[h_glob * Da + da_idx];                                                     \
                                                                                                                         \
        __shared__ T_compute shm_alpha;                                                                                  \
                                                                                                                         \
        const int r_start = tid * kRItems + r_chunk * r_stride;                                                          \
        bool valid_r[kRItems];                                                                                           \
        _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                              \
            valid_r[i] = (r_start + i < r);                                                                              \
                                                                                                                         \
        int t_len = (t_chunk == 0) ? P : Q;                                                                              \
        int t_start = (t_chunk == 0) ? 0 : (P + (t_chunk - 1) * Q);                                                      \
        if (t_chunk > 0 && t_start >= T_seq)                                                                             \
            return;                                                                                                      \
        int t_end = min(t_start + t_len, T_seq);                                                                         \
        int t_pref_start = max(0, t_start - (K - 1));                                                                    \
                                                                                                                         \
        const int64_t init_base = (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + (da_idx * r);                   \
        T_compute running_x[kRItems];                                                                                    \
        _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                              \
        {                                                                                                                \
            running_x[i] = valid_r[i] ? (T_compute)inits[init_base + r_start + i] : T_compute(0);                        \
        }                                                                                                                \
                                                                                                                         \
        const int64_t b_offset_combined = (int64_t)b_idx * sc_b;                                                         \
        const int64_t out_stride_b = (int64_t)h_comp_total * T_seq * D;                                                  \
        const int64_t out_stride_h = (int64_t)T_seq * D;                                                                 \
        const int64_t out_stride_t = (int64_t)D;                                                                         \
        const int64_t out_base = (int64_t)b_idx * out_stride_b + (int64_t)h_loc * out_stride_h + (da_idx * r) + r_start; \
                                                                                                                         \
        const int64_t logit_base_idx = b_offset_combined + (h_glob * head_stride + D + da_idx) * sc_e;                   \
                                                                                                                         \
        int64_t val_base_idx[kRItems];                                                                                   \
        _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                              \
        {                                                                                                                \
            if (valid_r[i])                                                                                              \
                val_base_idx[i] = b_offset_combined + (h_glob * head_stride + da_idx * r + r_start + i) * sc_e;          \
        }                                                                                                                \
                                                                                                                         \
        int64_t current_t_offset = (int64_t)t_pref_start * sc_t;                                                         \
                                                                                                                         \
        /* Phase 1: Preface loop (t < t_start) */                                                                        \
        for (int t = t_pref_start; t < t_start; ++t)                                                                     \
        {                                                                                                                \
            T_compute alpha;                                                                                             \
            if (tid == 0)                                                                                                \
            {                                                                                                            \
                T_compute l_raw = (T_compute)combined[logit_base_idx + current_t_offset] + bias_val;                     \
                alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                                                         \
                shm_alpha = alpha;                                                                                       \
            }                                                                                                            \
            __syncthreads();                                                                                             \
            if (tid != 0)                                                                                                \
            {                                                                                                            \
                alpha = shm_alpha;                                                                                       \
            }                                                                                                            \
                                                                                                                         \
            T_compute one_minus_alpha = T_compute(1) - alpha;                                                            \
                                                                                                                         \
            _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                          \
            {                                                                                                            \
                if (!valid_r[i])                                                                                         \
                    continue;                                                                                            \
                T_compute val = (T_compute)combined[val_base_idx[i] + current_t_offset];                                 \
                running_x[i] = alpha * val + one_minus_alpha * running_x[i];                                             \
            }                                                                                                            \
            current_t_offset += sc_t;                                                                                    \
            __syncthreads();                                                                                             \
        }                                                                                                                \
                                                                                                                         \
        /* Phase 2: Main loop (t >= t_start) */                                                                          \
        int64_t current_out_offset = out_base;                                                                           \
        for (int t = t_start; t < t_end; ++t)                                                                            \
        {                                                                                                                \
            T_compute alpha;                                                                                             \
            if (tid == 0)                                                                                                \
            {                                                                                                            \
                T_compute l_raw = (T_compute)combined[logit_base_idx + current_t_offset] + bias_val;                     \
                alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                                                         \
                shm_alpha = alpha;                                                                                       \
            }                                                                                                            \
            __syncthreads();                                                                                             \
            if (tid != 0)                                                                                                \
            {                                                                                                            \
                alpha = shm_alpha;                                                                                       \
            }                                                                                                            \
                                                                                                                         \
            T_compute one_minus_alpha = T_compute(1) - alpha;                                                            \
                                                                                                                         \
            _Pragma("unroll") for (int i = 0; i < kRItems; ++i)                                                          \
            {                                                                                                            \
                if (!valid_r[i])                                                                                         \
                    continue;                                                                                            \
                T_compute val = (T_compute)combined[val_base_idx[i] + current_t_offset];                                 \
                T_compute h_new = alpha * val + one_minus_alpha * running_x[i];                                          \
                running_x[i] = h_new;                                                                                    \
                x_out[current_out_offset + i] = (T_out)h_new;                                                            \
            }                                                                                                            \
            current_t_offset += sc_t;                                                                                    \
            current_out_offset += out_stride_t;                                                                          \
            __syncthreads();                                                                                             \
        }                                                                                                                \
    } while (0)

// -------------------------------------------------------------------------
// Generic forward device function (runtime strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__device__ __forceinline__ void
ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_forward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    LTV_LOOK_BACK_FUSED_CONCAT_SEQ_FORWARD_DEVICE_BODY(
        T_in, T_compute, T_out, kRThreads, kRItems,
        UseSigmoid, P, Q, K,
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e,
        T_seq, NH, NKVH, Da, r);
}

// -------------------------------------------------------------------------
// Const forward device function (compile‑time strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const>
__device__ __forceinline__ void
ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_forward_device_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
    LTV_LOOK_BACK_FUSED_CONCAT_SEQ_FORWARD_DEVICE_BODY(
        T_in, T_compute, T_out, kRThreads, kRItems,
        UseSigmoid, P, Q, K,
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b_const, sc_t_const, sc_e_const,
        T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Forward kernels
// -------------------------------------------------------------------------

// 1) Generic kernel
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__global__ void ltv_look_back_fused_concat_sequential_scan_forward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_forward_device<
        T_in, T_compute, T_out, kRThreads, kRItems,
        UseSigmoid, P, Q, K>(
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
}

// 2) Constant‑strides kernel (now calls const device function)
template <typename T_in, typename T_compute, typename T_out,
          int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
          int T_const, int NH_const, int NKVH_const,
          int Da_const, int r_const,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const>
__global__ void
ltv_look_back_fused_concat_sequential_scan_forward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
    ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_forward_device_const<
        T_in, T_compute, T_out, kRThreads, kRItems,
        UseSigmoid, P, Q, K,
        sc_b_const, sc_t_const, sc_e_const,
        T_const, NH_const, NKVH_const, Da_const, r_const>(
        combined, inits, logit_bias, out_q, out_k, out_v);
}

// -------------------------------------------------------------------------
// Dispatch inner helper (unchanged, calls const kernel with constants)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int cP, int cQ, int cK, bool cSig,
          int kRThreads, int kRItems>
void ltv_look_back_forward_dispatch_inner(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, torch::Tensor &out_q,
    torch::Tensor &out_k, torch::Tensor &out_v,
    int T_seq, int NH, int NKVH, int Da, int r,
    dim3 grid, cudaStream_t stream,
    int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
    dim3 block(kRThreads);

    constexpr int64_t kSeqStrideCombinedB = kStrideCombinedB_f(cSig);
    constexpr int64_t kSeqStrideCombinedT_val = kSeqStrideCombinedT;
    constexpr int64_t kSeqStrideCombinedE_val = kSeqStrideCombinedE;
    if (T_seq == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR &&
        sc_b == kSeqStrideCombinedB && sc_t == kSeqStrideCombinedT_val &&
        sc_e == kSeqStrideCombinedE_val)
    {
        ltv_look_back_fused_concat_sequential_scan_forward_kernel_const<
            T_in, T_compute, T_out, kRThreads, kRItems,
            cSig, cP, cQ, cK,
            kT, kNH, kNKVH, kDa, kR,
            kSeqStrideCombinedB, kSeqStrideCombinedT_val, kSeqStrideCombinedE_val>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
                logit_bias.data_ptr<T_compute>(),
                out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(),
                out_v.data_ptr<T_out>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    TORCH_WARN_ONCE(
        "ltv_look_back_forward: strides mismatch or unsupported shape for const kernel. "
        "Falling back to generic kernel. "
        "combined.stride=(",
        sc_b, ", ", sc_t, ", ", sc_e, "), "
                                      "expected=(",
        kSeqStrideCombinedB, ", ", kSeqStrideCombinedT_val, ", ",
        kSeqStrideCombinedE_val, ")");

    ltv_look_back_fused_concat_sequential_scan_forward_kernel<
        T_in, T_compute, T_out, kRThreads, kRItems,
        cSig, cP, cQ, cK>
        <<<grid, block, 0, stream>>>(
            combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
            logit_bias.data_ptr<T_compute>(),
            out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(),
            out_v.data_ptr<T_out>(),
            sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -------------------------------------------------------------------------
// Dispatch
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out>
void ltv_look_back_forward_dispatch(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, torch::Tensor &out_q,
    torch::Tensor &out_k, torch::Tensor &out_v,
    int T_seq, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int r_threads, int r_items, bool use_sigmoid,
    dim3 grid, cudaStream_t stream,
    int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
    if (P == 2147483647)
    {
        K = 1;
        Q = 1;
    }

#define LAUNCH_IF(Pv, Qv, Kv, Sigv)                                   \
    if (P == (Pv) && Q == (Qv) && K == (Kv) && use_sigmoid == (Sigv)) \
    {                                                                 \
        constexpr int cP = (Pv), cQ = (Qv), cK = (Kv);                \
        constexpr bool cSig = (Sigv);                                 \
        DISPATCH_R_THREADS(r_threads, {                               \
            DISPATCH_R_ITEMS(r_items, {                               \
                ltv_look_back_forward_dispatch_inner<                 \
                    T_in, T_compute, T_out,                           \
                    cP, cQ, cK, cSig,                                 \
                    kRThreads, kRItems>(                              \
                    combined, inits, logit_bias, out_q, out_k, out_v, \
                    T_seq, NH, NKVH, Da, r, grid, stream,             \
                    sc_b, sc_t, sc_e);                                \
            });                                                       \
        });                                                           \
        return;                                                       \
    }

    LTV_LOOK_BACK_LAUNCH_EACH_PQK();

    TORCH_CHECK(false, "ltv_look_back_forward: unsupported dispatch configuration. "
                       "P=",
                P, " Q=", Q, " K=", K,
                " r_th=", r_threads, " r_it=", r_items, " sig=", use_sigmoid);

#undef LAUNCH_IF
}

// -------------------------------------------------------------------------
// Launcher (called from Python)
// -------------------------------------------------------------------------
void ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    int B, int T, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int r_threads, int r_items, bool use_sigmoid)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t sc_b = combined.stride(0);
    int64_t sc_t = combined.stride(1);
    int64_t sc_e = combined.stride(2);

    int num_r_chunks = (r + r_threads * r_items - 1) / (r_threads * r_items);
    int M_dyn = (T > P) ? (T - P + Q - 1) / Q : 0;
    int num_t_chunks = 1 + M_dyn;
    dim3 grid(B, NH + 2 * NKVH, Da * num_r_chunks * num_t_chunks);

    auto dtype = combined.scalar_type();

    AT_DISPATCH_SWITCH(dtype, "ltv_look_back_forward",
                       AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                        {
            using T_in = float;
            using T_compute = float;
            using T_out = float;
            ltv_look_back_forward_dispatch<T_in, T_compute, T_out>(
                combined, inits, logit_bias, out_q, out_k, out_v,
                T, NH, NKVH, Da, r,
                P, Q, K,
                r_threads, r_items, use_sigmoid,
                grid, stream, sc_b, sc_t, sc_e); })
                           AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                            {
            using T_in = at::BFloat16;
            using T_compute = float;
            using T_out = at::BFloat16;
            ltv_look_back_forward_dispatch<T_in, T_compute, T_out>(
                combined, inits, logit_bias, out_q, out_k, out_v,
                T, NH, NKVH, Da, r,
                P, Q, K,
                r_threads, r_items, use_sigmoid,
                grid, stream, sc_b, sc_t, sc_e); }));
}
