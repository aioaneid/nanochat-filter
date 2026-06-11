#include "ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_common.cuh"

// -------------------------------------------------------------------------
// Macro that expands to the full look‑back forward device function body.
// Parameters:
//   T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
//   UseSigmoid, P, Q, K,
//   combined, inits, logit_bias, out_q, out_k, out_v,
//   sc_b, sc_t, sc_e,
//   T_seq, NH, NKVH, Da, r
// -------------------------------------------------------------------------
#define LTV_LOOK_BACK_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(                                                      \
    T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,                                                   \
    UseSigmoid, P, Q, K,                                                                                        \
    combined, inits, logit_bias, out_q, out_k, out_v,                                                           \
    sc_b, sc_t, sc_e,                                                                                           \
    T_seq, NH, NKVH, Da, r)                                                                                     \
    do                                                                                                          \
    {                                                                                                           \
        const int tid = threadIdx.x;                                                                            \
        const int lane_x = tid % kScanThreads;                                                                  \
        const int lane_y = tid / kScanThreads;                                                                  \
                                                                                                                \
        using ScanTemp = typename cub::WarpScan<AffineState<T_compute, kRItems>,                                \
                                                kScanThreads>::TempStorage;                                     \
        __shared__ alignas(16) ScanTemp scan_temp[kRThreads];                                                   \
                                                                                                                \
        const unsigned int b_idx = blockIdx.x;                                                                  \
        const unsigned int h_glob = blockIdx.y;                                                                 \
                                                                                                                \
        const int num_r_chunks = (r + kRThreads * kRItems - 1) / (kRThreads * kRItems);                         \
        int M_dyn = (T_seq > P) ? (T_seq - P + Q - 1) / Q : 0;                                                  \
        const int num_t_chunks = 1 + M_dyn;                                                                     \
                                                                                                                \
        const int z = blockIdx.z;                                                                               \
        const int t_chunk = z % num_t_chunks;                                                                   \
        const int rz = z / num_t_chunks;                                                                        \
        const int da_idx = rz % Da;                                                                             \
        const int r_chunk = rz / Da;                                                                            \
                                                                                                                \
        const unsigned int D = Da * r;                                                                          \
        const unsigned int total_h = NH + 2 * NKVH;                                                             \
        const unsigned int head_stride = D + Da;                                                                \
                                                                                                                \
        T_out *x_out;                                                                                           \
        unsigned int h_loc, h_comp_total;                                                                       \
        if (h_glob < NH)                                                                                        \
        {                                                                                                       \
            x_out = out_q;                                                                                      \
            h_loc = h_glob;                                                                                     \
            h_comp_total = NH;                                                                                  \
        }                                                                                                       \
        else if (h_glob < NH + NKVH)                                                                            \
        {                                                                                                       \
            x_out = out_k;                                                                                      \
            h_loc = h_glob - NH;                                                                                \
            h_comp_total = NKVH;                                                                                \
        }                                                                                                       \
        else                                                                                                    \
        {                                                                                                       \
            x_out = out_v;                                                                                      \
            h_loc = h_glob - (NH + NKVH);                                                                       \
            h_comp_total = NKVH;                                                                                \
        }                                                                                                       \
                                                                                                                \
        if (r_chunk >= num_r_chunks)                                                                            \
            return;                                                                                             \
                                                                                                                \
        const int64_t b_offset_combined = (int64_t)b_idx * sc_b;                                                \
        const unsigned int val_base = h_glob * head_stride + da_idx * r;                                        \
        const unsigned int logit_base = h_glob * head_stride + D + da_idx;                                      \
                                                                                                                \
        const int r_stride = kRThreads * kRItems;                                                               \
        const T_compute bias_val = logit_bias[h_glob * Da + da_idx];                                            \
                                                                                                                \
        const int r_start = lane_y * kRItems + r_chunk * r_stride;                                              \
                                                                                                                \
        const int64_t init_base = (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + (da_idx * r);          \
        int64_t init_idx_item = init_base + r_start;                                                            \
                                                                                                                \
        const int64_t logit_base_idx = b_offset_combined + (int64_t)logit_base * sc_e + (int64_t)lane_x * sc_t; \
        const int64_t logit_t_stride = (int64_t)kScanThreads * sc_t;                                            \
                                                                                                                \
        const int64_t val_r_stride = (int64_t)r_stride * sc_e;                                                  \
        const int64_t val_base_idx = b_offset_combined + (int64_t)val_base * sc_e +                             \
                                     (int64_t)lane_x * sc_t + (int64_t)(lane_y * kRItems) * sc_e;               \
        int64_t val_idx_t = val_base_idx + r_chunk * val_r_stride;                                              \
        const int64_t val_t_stride = (int64_t)kScanThreads * sc_t;                                              \
                                                                                                                \
        const int64_t out_base_idx = (int64_t)b_idx * (h_comp_total * T_seq * D) +                              \
                                     (int64_t)h_loc * (T_seq * D) + (da_idx * r) +                              \
                                     (int64_t)lane_x * D + (lane_y * kRItems);                                  \
        int64_t out_idx_t = out_base_idx + r_chunk * r_stride;                                                  \
        const int64_t out_t_stride = (int64_t)kScanThreads * D;                                                 \
                                                                                                                \
        bool valid_r[kRItems];                                                                                  \
        _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                            \
            valid_r[item] = (r_start + item < r);                                                               \
                                                                                                                \
        int t_len = t_chunk ? Q : P;                                                                            \
        int t_start = P + (t_chunk - 1) * t_len;                                                                \
        if (t_chunk && t_start >= T_seq)                                                                        \
            return;                                                                                             \
                                                                                                                \
        int t_end = min(t_start + t_len, T_seq);                                                                \
        int t_pref_start = max(0, t_start - (K - 1));                                                           \
                                                                                                                \
        T_compute running_x[kRItems];                                                                           \
        _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                            \
        {                                                                                                       \
            running_x[item] = valid_r[item] ? (T_compute)inits[init_idx_item] : T_compute(0);                   \
            init_idx_item += 1;                                                                                 \
        }                                                                                                       \
                                                                                                                \
        int t_global = t_pref_start + lane_x;                                                                   \
        int64_t logit_idx_t = logit_base_idx + (int64_t)t_pref_start * sc_t;                                    \
                                                                                                                \
        val_idx_t += (int64_t)t_pref_start * sc_t;                                                              \
        out_idx_t += (int64_t)t_pref_start * D;                                                                 \
                                                                                                                \
        int full_t_chunks = (t_end - t_pref_start) / kScanThreads;                                              \
        bool has_tail = ((t_end - t_pref_start) % kScanThreads) != 0;                                           \
                                                                                                                \
        /* ==================== FAST PATH ==================== */                                               \
        for (int tc = 0; tc < full_t_chunks; ++tc)                                                              \
        {                                                                                                       \
            bool valid_t = (t_global >= t_start) && (t_global < t_end);                                         \
                                                                                                                \
            T_compute l_raw = (T_compute)combined[logit_idx_t];                                                 \
            l_raw += bias_val;                                                                                  \
            T_compute alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                                          \
                                                                                                                \
            AffineState<T_compute, kRItems> thread_input;                                                       \
            thread_input.a = T_compute(1) - alpha;                                                              \
                                                                                                                \
            int64_t val_idx_item = val_idx_t;                                                                   \
            _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                        \
            {                                                                                                   \
                thread_input.x[item] = (valid_r[item] && t_global < t_end)                                      \
                                           ? alpha * (T_compute)combined[val_idx_item]                          \
                                           : T_compute(0);                                                      \
                val_idx_item += sc_e;                                                                           \
            }                                                                                                   \
                                                                                                                \
            AffineState<T_compute, kRItems> inclusive_out, block_aggregate;                                     \
            cub::WarpScan<AffineState<T_compute, kRItems>, kScanThreads>(                                       \
                scan_temp[lane_y])                                                                              \
                .InclusiveScan(thread_input, inclusive_out,                                                     \
                               AffineScanOp<T_compute, kRItems>(), block_aggregate);                            \
                                                                                                                \
            int64_t out_idx_item = out_idx_t;                                                                   \
            T_compute inc_a = inclusive_out.a;                                                                  \
            T_compute blk_a = block_aggregate.a;                                                                \
                                                                                                                \
            _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                        \
            {                                                                                                   \
                if (valid_r[item])                                                                              \
                {                                                                                               \
                    if (valid_t)                                                                                \
                        x_out[out_idx_item] = (T_out)(inc_a * running_x[item] + inclusive_out.x[item]);         \
                    running_x[item] = blk_a * running_x[item] + block_aggregate.x[item];                        \
                }                                                                                               \
                out_idx_item += 1;                                                                              \
            }                                                                                                   \
                                                                                                                \
            t_global += kScanThreads;                                                                           \
            logit_idx_t += logit_t_stride;                                                                      \
            val_idx_t += val_t_stride;                                                                          \
            out_idx_t += out_t_stride;                                                                          \
        }                                                                                                       \
                                                                                                                \
        /* ==================== SLOW PATH (FIXED) ==================== */                                       \
        if (has_tail)                                                                                           \
        {                                                                                                       \
            bool valid_read = (t_global < t_end);                                                               \
            bool valid_write = (t_global >= t_start) && (t_global < t_end);                                     \
                                                                                                                \
            AffineState<T_compute, kRItems> thread_input;                                                       \
            T_compute alpha = T_compute(0);                                                                     \
                                                                                                                \
            if (valid_read)                                                                                     \
            {                                                                                                   \
                T_compute l_raw = (T_compute)combined[logit_idx_t];                                             \
                l_raw += bias_val;                                                                              \
                alpha = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;                                                \
                thread_input.a = T_compute(1) - alpha;                                                          \
            }                                                                                                   \
            else                                                                                                \
            {                                                                                                   \
                thread_input.a = T_compute(1);                                                                  \
            }                                                                                                   \
                                                                                                                \
            int64_t val_idx_item = val_idx_t;                                                                   \
            _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                        \
            {                                                                                                   \
                thread_input.x[item] = (valid_read && valid_r[item])                                            \
                                           ? alpha * (T_compute)combined[val_idx_item]                          \
                                           : T_compute(0);                                                      \
                val_idx_item += sc_e;                                                                           \
            }                                                                                                   \
                                                                                                                \
            AffineState<T_compute, kRItems> inclusive_out, block_aggregate;                                     \
            cub::WarpScan<AffineState<T_compute, kRItems>, kScanThreads>(                                       \
                scan_temp[lane_y])                                                                              \
                .InclusiveScan(thread_input, inclusive_out,                                                     \
                               AffineScanOp<T_compute, kRItems>(), block_aggregate);                            \
                                                                                                                \
            if (valid_write)                                                                                    \
            {                                                                                                   \
                int64_t out_idx_item = out_idx_t;                                                               \
                T_compute inc_a = inclusive_out.a;                                                              \
                T_compute blk_a = block_aggregate.a;                                                            \
                                                                                                                \
                _Pragma("unroll") for (int item = 0; item < kRItems; ++item)                                    \
                {                                                                                               \
                    if (valid_r[item])                                                                          \
                    {                                                                                           \
                        x_out[out_idx_item] = (T_out)(inc_a * running_x[item] + inclusive_out.x[item]);         \
                        running_x[item] = blk_a * running_x[item] + block_aggregate.x[item];                    \
                    }                                                                                           \
                    out_idx_item += 1;                                                                          \
                }                                                                                               \
            }                                                                                                   \
        }                                                                                                       \
    } while (0)

// -------------------------------------------------------------------------
// Generic forward device function (runtime strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__device__ __forceinline__ void
ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    LTV_LOOK_BACK_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
        UseSigmoid, P, Q, K,
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e,
        T_seq, NH, NKVH, Da, r);
}

// -------------------------------------------------------------------------
// Const forward device function (compile‑time strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const>
__device__ __forceinline__ void
ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
    LTV_LOOK_BACK_FUSED_CONCAT_HM_FORWARD_DEVICE_BODY(
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
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
          int kScanThreads, int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K>
__global__ void ltv_look_back_fused_concat_cub_forward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
        UseSigmoid, P, Q, K>(
        combined, inits, logit_bias, out_q, out_k, out_v,
        sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
}

// 2) Constant-strides kernel (now calls const device function)
template <typename T_in, typename T_compute, typename T_out,
          int kScanThreads, int kRThreads, int kRItems,
          bool UseSigmoid, int P, int Q, int K,
          int T_const, int NH_const, int NKVH_const,
          int Da_const, int r_const,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const>
__global__ void
ltv_look_back_fused_concat_cub_forward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_compute *__restrict__ logit_bias, T_out *__restrict__ out_q,
    T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
    ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device_const<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
        UseSigmoid, P, Q, K,
        sc_b_const, sc_t_const, sc_e_const,
        T_const, NH_const, NKVH_const, Da_const, r_const>(
        combined, inits, logit_bias, out_q, out_k, out_v);
}

// -------------------------------------------------------------------------
// Dispatch (unchanged, but now const kernel uses compile‑time strides)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out,
          int cP, int cQ, int cK, bool cSig,
          int kScanThreads, int kRThreads, int kRItems>
void ltv_look_back_forward_dispatch_inner(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, torch::Tensor &out_q,
    torch::Tensor &out_k, torch::Tensor &out_v,
    int T_seq, int NH, int NKVH, int Da, int r,
    dim3 grid, cudaStream_t stream,
    int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
    dim3 block(kScanThreads * kRThreads);

    constexpr int64_t kStrideCombinedB = kStrideCombinedB_f(cSig);
    constexpr int64_t kStrideCombinedT_val = ::kStrideCombinedT;
    constexpr int64_t kStrideCombinedE_val = ::kStrideCombinedE;
    if (T_seq == kT && NH == kNH && NKVH == kNKVH && Da == kDa &&
        r == kR &&
        sc_b == kStrideCombinedB && sc_t == kStrideCombinedT_val &&
        sc_e == kStrideCombinedE_val)
    {
        ltv_look_back_fused_concat_cub_forward_kernel_const<
            T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
            cSig, cP, cQ, cK, kT, kNH, kNKVH, kDa, kR,
            kStrideCombinedB, kStrideCombinedT_val, kStrideCombinedE_val>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<T_in>(), inits.data_ptr<T_in>(),
                logit_bias.data_ptr<T_compute>(),
                out_q.data_ptr<T_out>(), out_k.data_ptr<T_out>(),
                out_v.data_ptr<T_out>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    else
    {
        TORCH_WARN_ONCE(
            "ltv_look_back_forward: matched specialized shape "
            "but not the specialized strides. Falling back to generic kernel. "
            "combined.stride=(",
            sc_b, ", ", sc_t, ", ", sc_e, "), "
                                          "expected=(",
            kStrideCombinedB, ", ", kStrideCombinedT_val, ", ",
            kStrideCombinedE_val, ")");
    }

    // Generic kernel launch
    ltv_look_back_fused_concat_cub_forward_kernel<
        T_in, T_compute, T_out, kScanThreads, kRThreads, kRItems,
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
// Outer dispatch (unchanged)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out>
void ltv_look_back_forward_dispatch(
    const torch::Tensor &combined, const torch::Tensor &inits,
    const torch::Tensor &logit_bias, torch::Tensor &out_q,
    torch::Tensor &out_k, torch::Tensor &out_v,
    int T_seq, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int scan_threads, int r_threads, int r_items, bool use_sigmoid,
    dim3 grid, cudaStream_t stream,
    int64_t sc_b, int64_t sc_t, int64_t sc_e)
{
    if (P == 2147483647)
    {
        K = 1;
        Q = 1;
    }

#define LAUNCH_IF(Pv, Qv, Kv, Sigv)                                                                                                 \
    if (P == (Pv) && Q == (Qv) && K == (Kv) && use_sigmoid == (Sigv))                                                               \
    {                                                                                                                               \
        constexpr int cP = (Pv), cQ = (Qv), cK = (Kv);                                                                              \
        constexpr bool cSig = (Sigv);                                                                                               \
        DISPATCH_SCAN_THREADS(scan_threads,                                                                                         \
                              {                                                                                                     \
                                  DISPATCH_R_THREADS(r_threads,                                                                     \
                                                     {                                                                              \
                                                         DISPATCH_R_ITEMS(r_items,                                                  \
                                                                          {                                                         \
                                                                              ltv_look_back_forward_dispatch_inner<                 \
                                                                                  T_in, T_compute, T_out,                           \
                                                                                  cP, cQ, cK, cSig,                                 \
                                                                                  kScanThreads, kRThreads, kRItems>(                \
                                                                                  combined, inits, logit_bias, out_q, out_k, out_v, \
                                                                                  T_seq, NH, NKVH, Da, r, grid, stream,             \
                                                                                  sc_b, sc_t, sc_e);                                \
                                                                          });                                                       \
                                                     });                                                                            \
                              });                                                                                                   \
        return;                                                                                                                     \
    }

    LAUNCH_IF(2147483647, 1, 1, true);
    LAUNCH_IF(16, 16, 9, true);

    TORCH_CHECK(false, "ltv_look_back_forward: unsupported dispatch configuration. "
                       "P=",
                P, " Q=", Q, " K=", K,
                " scan=", scan_threads, " r_th=", r_threads,
                " r_it=", r_items, " sig=", use_sigmoid);

#undef LAUNCH_IF
}

// -------------------------------------------------------------------------
// Launcher (unchanged)
// -------------------------------------------------------------------------
void ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor logit_bias,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v,
    int B, int T, int NH, int NKVH, int Da, int r,
    int P, int Q, int K,
    int scan_threads, int r_threads, int r_items, bool use_sigmoid)
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

    AT_DISPATCH_SWITCH(
        dtype, "ltv_look_back_forward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using T_in = float;
        using T_compute = float;
        using T_out = float;
        ltv_look_back_forward_dispatch<T_in, T_compute, T_out>(
            combined, inits, logit_bias, out_q, out_k, out_v,
            T, NH, NKVH, Da, r,
            P, Q, K,
            scan_threads, r_threads, r_items, use_sigmoid,
            grid, stream, sc_b, sc_t, sc_e); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                               {
        using T_in = at::BFloat16;
        using T_compute = float;
        using T_out = at::BFloat16;
        ltv_look_back_forward_dispatch<T_in, T_compute, T_out>(
            combined, inits, logit_bias, out_q, out_k, out_v,
            T, NH, NKVH, Da, r,
            P, Q, K,
            scan_threads, r_threads, r_items, use_sigmoid,
            grid, stream, sc_b, sc_t, sc_e); }));
}
