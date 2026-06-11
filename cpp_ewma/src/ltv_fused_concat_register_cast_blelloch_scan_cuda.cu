#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "ltv_scan_common.cuh"

// -------------------------------------------------------------------------
// Constants for specialization
// -------------------------------------------------------------------------
constexpr int kT = 1024;
constexpr int kNH = 8;
constexpr int kNKVH = 8;
constexpr int kDa = 1;
constexpr int kR = 128;

constexpr int64_t kTotalH = static_cast<int64_t>(kNH + 2 * kNKVH);
constexpr int64_t kD = static_cast<int64_t>(kDa) * kR;
constexpr int64_t kTotalE = kTotalH * kD + kTotalH * kDa;

constexpr int64_t kStrideCombinedB = static_cast<int64_t>(kT) * kTotalE;
constexpr int64_t kStrideCombinedT = kTotalE;
constexpr int64_t kStrideCombinedE = 1;

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

constexpr int64_t kStrideGradOutQHTransposed = kD;
constexpr int64_t kStrideGradOutQTTransposed = static_cast<int64_t>(kNH) * kD;
constexpr int64_t kStrideGradOutKHTransposed = kD;
constexpr int64_t kStrideGradOutKTTransposed = static_cast<int64_t>(kNKVH) * kD;

// -------------------------------------------------------------------------
// Dispatch Macros for CUB Compilation
// 32, 64, 96, 128, 160, 192, 224, 256 are common values.
// -------------------------------------------------------------------------
#define DISPATCH_BLOCK_THREADS(BLOCK_THREADS_VAL, ...)                                                                            \
    if (BLOCK_THREADS_VAL == 32)                                                                                                  \
    {                                                                                                                             \
        constexpr int kBlockThreads = 32;                                                                                         \
        __VA_ARGS__;                                                                                                              \
    }                                                                                                                             \
    else                                                                                                                          \
    {                                                                                                                             \
        TORCH_CHECK(false, "Unsupported block_threads: ", BLOCK_THREADS_VAL, ". Supported: 32, 64, 96, 128, 160, 192, 224, 256"); \
    }

    // 1, 2, 4, 8 are common values.
#define DISPATCH_ITEMS_PER_THREAD(ITEMS_PER_THREAD_VAL, ...)                                                   \
    if (ITEMS_PER_THREAD_VAL == 1)                                                                             \
    {                                                                                                          \
        constexpr int kItemsPerThread = 1;                                                                     \
        __VA_ARGS__;                                                                                           \
    }                                                                                                          \
    else                                                                                                       \
    {                                                                                                          \
        TORCH_CHECK(false, "Unsupported items_per_thread: ", ITEMS_PER_THREAD_VAL, ". Supported: 1, 2, 4, 8"); \
    }

// -------------------------------------------------------------------------
// Core Types and Helpers
// -------------------------------------------------------------------------
template <typename T>
struct AffineState
{
    T a;
    T x;
};

template <typename T>
struct AffineScanOp
{
    __device__ __forceinline__ AffineState<T> operator()(const AffineState<T> &left, const AffineState<T> &right) const
    {
        return {right.a * left.a, right.a * left.x + right.x};
    }
};

static inline void warn_specialized_stride_mismatch(
    const char *op_name, int64_t stride_combined_b, int64_t stride_combined_t, int64_t stride_combined_e)
{
    TORCH_WARN(
        op_name,
        " matched the specialized shape but not the specialized strides. Falling back to the generic kernel. ",
        "combined.stride=(", stride_combined_b, ", ", stride_combined_t, ", ", stride_combined_e, "), ",
        "expected=(", kStrideCombinedB, ", ", kStrideCombinedT, ", ", kStrideCombinedE, ")");
}

static inline void warn_specialized_backward_stride_mismatch(
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e,
    int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t s_hq_b, int64_t s_hq_h, int64_t s_hq_t, int64_t s_hq_d,
    int64_t s_hk_b, int64_t s_hk_h, int64_t s_hk_t, int64_t s_hk_d,
    int64_t s_hv_b, int64_t s_hv_h, int64_t s_hv_t, int64_t s_hv_d,
    int64_t s_gq_b, int64_t s_gq_h, int64_t s_gq_t, int64_t s_gq_d,
    int64_t s_gk_b, int64_t s_gk_h, int64_t s_gk_t, int64_t s_gk_d,
    int64_t s_gv_b, int64_t s_gv_h, int64_t s_gv_t, int64_t s_gv_d)
{
    TORCH_WARN(
        "ltv_fused_concat_register_cast_blelloch_scan_cuda_backward matched the specialized shape but not the specialized strides. ",
        "Falling back to the generic kernel. ",
        "combined.stride=(", sc_in_b, ", ", sc_in_t, ", ", sc_in_e, "), ",
        "expected=(", kStrideCombinedB, ", ", kStrideCombinedT, ", ", kStrideCombinedE, "); ",
        "grad_combined.stride=(", sc_out_b, ", ", sc_out_t, ", ", sc_out_e, "), ",
        "expected=(", kStrideCombinedB, ", ", kStrideCombinedT, ", ", kStrideCombinedE, "); ",
        "out_q.stride=(", s_hq_b, ", ", s_hq_h, ", ", s_hq_t, ", ", s_hq_d, "), ",
        "expected=(", kStrideGradOutQB, ", ", kStrideGradOutQH, ", ", kStrideGradOutQT, ", ", kStrideGradOutQD, "); ",
        "out_k.stride=(", s_hk_b, ", ", s_hk_h, ", ", s_hk_t, ", ", s_hk_d, "), ",
        "expected=(", kStrideGradOutKB, ", ", kStrideGradOutKH, ", ", kStrideGradOutKT, ", ", kStrideGradOutKD, "); ",
        "out_v.stride=(", s_hv_b, ", ", s_hv_h, ", ", s_hv_t, ", ", s_hv_d, "), ",
        "expected=(", kStrideGradOutVB, ", ", kStrideGradOutVH, ", ", kStrideGradOutVT, ", ", kStrideGradOutVD, "); ",
        "grad_out_q.stride=(", s_gq_b, ", ", s_gq_h, ", ", s_gq_t, ", ", s_gq_d, "), ",
        "expected=(", kStrideGradOutQB, ", ", kStrideGradOutQH, ", ", kStrideGradOutQT, ", ", kStrideGradOutQD, "); ",
        "grad_out_k.stride=(", s_gk_b, ", ", s_gk_h, ", ", s_gk_t, ", ", s_gk_d, "), ",
        "expected=(", kStrideGradOutKB, ", ", kStrideGradOutKH, ", ", kStrideGradOutKT, ", ", kStrideGradOutKD, "); ",
        "grad_out_v.stride=(", s_gv_b, ", ", s_gv_h, ", ", s_gv_t, ", ", s_gv_d, "), ",
        "expected=(", kStrideGradOutVB, ", ", kStrideGradOutVH, ", ", kStrideGradOutVT, ", ", kStrideGradOutVD, ")");
}

// -------------------------------------------------------------------------
// Forward Device Function (Delegation Pattern)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread, bool UseSigmoid>
__device__ __forceinline__ void ltv_fused_concat_register_cast_cub_forward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    T_out *__restrict__ out_q, T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    constexpr int ChunkSize = BlockThreads * ItemsPerThread;
    const unsigned int b_idx = blockIdx.x;
    const unsigned int h_glob = blockIdx.y;
    const unsigned int d_idx = blockIdx.z;

    typedef cub::BlockExchange<AffineState<T_compute>, BlockThreads, ItemsPerThread> BlockExchange;
    typedef cub::BlockScan<AffineState<T_compute>, BlockThreads> BlockScan;

    __shared__ union
    {
        typename BlockExchange::TempStorage exchange;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    const unsigned int D = Da * r;
    const unsigned int total_h = NH + 2 * NKVH;
    const unsigned int da_idx = d_idx / r;

    T_out *x_out;
    unsigned int h_loc, h_comp_total, offset_in_row;

    if (h_glob < NH)
    {
        x_out = out_q;
        h_loc = h_glob;
        h_comp_total = NH;
        offset_in_row = 0;
    }
    else if (h_glob < NH + NKVH)
    {
        x_out = out_k;
        h_loc = h_glob - NH;
        h_comp_total = NKVH;
        offset_in_row = NH * D;
    }
    else
    {
        x_out = out_v;
        h_loc = h_glob - (NH + NKVH);
        h_comp_total = NKVH;
        offset_in_row = (NH + NKVH) * D;
    }

    const int64_t b_offset_combined = (int64_t)b_idx * sc_b;
    const unsigned int offset_l = total_h * D;

    AffineState<T_compute> running_state = {1.0f, (T_compute)inits[(int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + d_idx]};

    for (int t_base = 0; t_base < T_seq; t_base += ChunkSize)
    {
        AffineState<T_compute> thread_data[ItemsPerThread];

// 1. Striped Load
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i)
        {
            int local_t = i * BlockThreads + threadIdx.x;
            int global_t = t_base + local_t;

            if (global_t < T_seq)
            {
                int64_t val_idx = b_offset_combined + (int64_t)(offset_in_row + h_loc * D + d_idx) * sc_e + (int64_t)global_t * sc_t;
                int64_t logit_idx = b_offset_combined + (int64_t)(offset_l + h_glob * Da + da_idx) * sc_e + (int64_t)global_t * sc_t;

                T_compute x_val = (T_compute)combined[val_idx];
                T_compute l_raw = (T_compute)combined[logit_idx];

                T_compute alpha_gate = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;
                thread_data[i].a = 1.0f - alpha_gate;
                thread_data[i].x = alpha_gate * x_val;
            }
            else
            {
                thread_data[i].a = 1.0f;
                thread_data[i].x = 0.0f;
            }
        }

        // 2. Exchange to Blocked
        __syncthreads();
        AffineState<T_compute> scan_data[ItemsPerThread];
        BlockExchange(temp_storage.exchange).StripedToBlocked(thread_data, scan_data);

        // 3. Exclusive Scan
        __syncthreads();
        AffineState<T_compute> block_aggregate;
        BlockScan(temp_storage.scan).ExclusiveScan(scan_data, scan_data, {1.0f, 0.0f}, AffineScanOp<T_compute>(), block_aggregate);

        // 4. Exchange back to Striped
        __syncthreads();
        AffineState<T_compute> prefix_data[ItemsPerThread];
        BlockExchange(temp_storage.exchange).BlockedToStriped(scan_data, prefix_data);

// 5. Compute Final Value and Store
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i)
        {
            int local_t = i * BlockThreads + threadIdx.x;
            int global_t = t_base + local_t;

            if (global_t < T_seq)
            {
                // L_total(running) = a_i * (pref_a * running_h + pref_x) + x_i
                T_compute h_i = thread_data[i].a * (prefix_data[i].a * running_state.x + prefix_data[i].x) + thread_data[i].x;
                int64_t out_idx = (int64_t)b_idx * (h_comp_total * T_seq * D) + (int64_t)h_loc * (T_seq * D) + (int64_t)d_idx + (int64_t)global_t * D;
                x_out[out_idx] = (T_out)h_i;
            }
        }

        running_state.x = block_aggregate.a * running_state.x + block_aggregate.x;
        __syncthreads();
    }
}

template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread>
__global__ void ltv_fused_concat_register_cast_cub_forward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    T_out *__restrict__ out_q, T_out *__restrict__ out_k, T_out *__restrict__ out_v,
    int64_t sc_b, int64_t sc_t, int64_t sc_e, int T_seq, int NH, int NKVH, int Da, int r, bool use_sigmoid)
{
    if (use_sigmoid)
    {
        ltv_fused_concat_register_cast_cub_forward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, true>(
            combined, inits, out_q, out_k, out_v, sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
    }
    else
    {
        ltv_fused_concat_register_cast_cub_forward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, false>(
            combined, inits, out_q, out_k, out_v, sc_b, sc_t, sc_e, T_seq, NH, NKVH, Da, r);
    }
}

template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const,
          int64_t sc_b_const, int64_t sc_t_const, int64_t sc_e_const>
__global__ void ltv_fused_concat_register_cast_cub_forward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    T_out *__restrict__ out_q, T_out *__restrict__ out_k, T_out *__restrict__ out_v)
{
    // Only True is supported as a fully constant kernel path
    ltv_fused_concat_register_cast_cub_forward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, true>(
        combined, inits, out_q, out_k, out_v, sc_b_const, sc_t_const, sc_e_const, T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Backward Device Function (Delegation Pattern)
// -------------------------------------------------------------------------
template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread, bool UseSigmoid>
__device__ __forceinline__ void ltv_fused_concat_register_cast_cub_backward_device(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_out *__restrict__ hq, const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk, const T_out *__restrict__ grad_hv,
    T_out *__restrict__ grad_combined, T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logits,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e, int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
    int64_t str_hq_b, int64_t str_hq_h, int64_t str_hq_t, int64_t str_hq_d,
    int64_t str_hk_b, int64_t str_hk_h, int64_t str_hk_t, int64_t str_hk_d,
    int64_t str_hv_b, int64_t str_hv_h, int64_t str_hv_t, int64_t str_hv_d,
    int64_t str_gq_b, int64_t str_gq_h, int64_t str_gq_t, int64_t str_gq_d,
    int64_t str_gk_b, int64_t str_gk_h, int64_t str_gk_t, int64_t str_gk_d,
    int64_t str_gv_b, int64_t str_gv_h, int64_t str_gv_t, int64_t str_gv_d,
    int T_seq, int NH, int NKVH, int Da, int r)
{
    constexpr int ChunkSize = BlockThreads * ItemsPerThread;
    const unsigned int b_idx = blockIdx.x;
    const unsigned int h_glob = blockIdx.y;
    const unsigned int d_idx = blockIdx.z;

    typedef cub::BlockExchange<AffineState<T_compute>, BlockThreads, ItemsPerThread> BlockExchange;
    typedef cub::BlockScan<AffineState<T_compute>, BlockThreads> BlockScan;

    __shared__ union
    {
        typename BlockExchange::TempStorage exchange;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    const unsigned int D = Da * r;
    const unsigned int total_h = NH + 2 * NKVH;
    const unsigned int da_idx = d_idx / r;

    const T_out *h_ptr;
    const T_out *grad_h_ptr;
    unsigned int h_loc, h_comp_total, offset_in_row;
    int64_t str_h_b, str_h_h, str_h_t, str_h_d;
    int64_t str_gh_b, str_gh_h, str_gh_t, str_gh_d;

    if (h_glob < NH)
    {
        h_ptr = hq;
        grad_h_ptr = grad_hq;
        h_loc = h_glob;
        h_comp_total = NH;
        offset_in_row = 0;
        str_h_b = str_hq_b;
        str_h_h = str_hq_h;
        str_h_t = str_hq_t;
        str_h_d = str_hq_d;
        str_gh_b = str_gq_b;
        str_gh_h = str_gq_h;
        str_gh_t = str_gq_t;
        str_gh_d = str_gq_d;
    }
    else if (h_glob < NH + NKVH)
    {
        h_ptr = hk;
        grad_h_ptr = grad_hk;
        h_loc = h_glob - NH;
        h_comp_total = NKVH;
        offset_in_row = NH * D;
        str_h_b = str_hk_b;
        str_h_h = str_hk_h;
        str_h_t = str_hk_t;
        str_h_d = str_hk_d;
        str_gh_b = str_gk_b;
        str_gh_h = str_gk_h;
        str_gh_t = str_gk_t;
        str_gh_d = str_gk_d;
    }
    else
    {
        h_ptr = hv;
        grad_h_ptr = grad_hv;
        h_loc = h_glob - (NH + NKVH);
        h_comp_total = NKVH;
        offset_in_row = (NH + NKVH) * D;
        str_h_b = str_hv_b;
        str_h_h = str_hv_h;
        str_h_t = str_hv_t;
        str_h_d = str_hv_d;
        str_gh_b = str_gv_b;
        str_gh_h = str_gv_h;
        str_gh_t = str_gv_t;
        str_gh_d = str_gv_d;
    }

    const unsigned int offset_l = total_h * D;
    const int64_t b_offset_in = (int64_t)b_idx * sc_in_b;
    const int64_t b_offset_out = (int64_t)b_idx * sc_out_b;

    const T_in *q_logit = combined + b_offset_in + (int64_t)(offset_l + h_glob * Da + da_idx) * sc_in_e;
    const T_in *q_val = combined + b_offset_in + (int64_t)(offset_in_row + h_loc * D + d_idx) * sc_in_e;

    T_compute running_g = 0.0f;
    int chunks = (T_seq + ChunkSize - 1) / ChunkSize;

    for (int chunk_idx = chunks - 1; chunk_idx >= 0; --chunk_idx)
    {
        int t_base = chunk_idx * ChunkSize;
        AffineState<T_compute> thread_data[ItemsPerThread];

// 1. Striped Load Backwards
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i)
        {
            int local_idx_striped = i * BlockThreads + threadIdx.x;
            int t = t_base + ChunkSize - 1 - local_idx_striped;

            if (t >= 0 && t < T_seq)
            {
                T_compute a_next = 1.0f;
                if (t + 1 < T_seq)
                {
                    T_compute l_raw_next = (T_compute)q_logit[(int64_t)(t + 1) * sc_in_t];
                    T_compute alpha_next = UseSigmoid ? sigmoid_f32(l_raw_next) : l_raw_next;
                    a_next = 1.0f - alpha_next;
                }

                int64_t gh_idx = (int64_t)b_idx * str_gh_b + (int64_t)h_loc * str_gh_h + (int64_t)t * str_gh_t + (int64_t)d_idx * str_gh_d;
                thread_data[i].a = a_next;
                thread_data[i].x = (T_compute)grad_h_ptr[gh_idx];
            }
            else
            {
                thread_data[i].a = 1.0f;
                thread_data[i].x = 0.0f;
            }
        }

        // 2. Exchange to Blocked
        __syncthreads();
        AffineState<T_compute> scan_data[ItemsPerThread];
        BlockExchange(temp_storage.exchange).StripedToBlocked(thread_data, scan_data);

        // 3. Exclusive Scan
        __syncthreads();
        AffineState<T_compute> block_aggregate;
        BlockScan(temp_storage.scan).ExclusiveScan(scan_data, scan_data, {1.0f, 0.0f}, AffineScanOp<T_compute>(), block_aggregate);

        // 4. Exchange back to Striped
        __syncthreads();
        AffineState<T_compute> prefix_data[ItemsPerThread];
        BlockExchange(temp_storage.exchange).BlockedToStriped(scan_data, prefix_data);

// 5. Compute Gradients
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i)
        {
            int local_idx_striped = i * BlockThreads + threadIdx.x;
            int t = t_base + ChunkSize - 1 - local_idx_striped;

            if (t >= 0 && t < T_seq)
            {
                T_compute g_i = thread_data[i].a * (prefix_data[i].a * running_g + prefix_data[i].x) + thread_data[i].x;

                T_compute h_prev;
                if (t == 0)
                {
                    h_prev = (T_compute)inits[(int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + d_idx];
                }
                else
                {
                    int64_t h_prev_idx = (int64_t)b_idx * str_h_b + (int64_t)h_loc * str_h_h + (int64_t)(t - 1) * str_h_t + (int64_t)d_idx * str_h_d;
                    h_prev = (T_compute)h_ptr[h_prev_idx];
                }

                T_compute x_val = (T_compute)q_val[(int64_t)t * sc_in_t];
                T_compute l_raw = (T_compute)q_logit[(int64_t)t * sc_in_t];
                T_compute alpha_gate = UseSigmoid ? sigmoid_f32(l_raw) : l_raw;
                T_compute a_i = 1.0f - alpha_gate;

                T_compute d_alpha = (x_val - h_prev) * g_i;
                T_compute d_x = alpha_gate * g_i;
                T_compute d_l = UseSigmoid ? d_alpha * alpha_gate * (1.0f - alpha_gate) : d_alpha;

                int64_t val_out_idx = b_offset_out + (int64_t)(offset_in_row + h_loc * D + d_idx) * sc_out_e + (int64_t)t * sc_out_t;
                int64_t logit_out_idx = b_offset_out + (int64_t)(offset_l + h_glob * Da + da_idx) * sc_out_e + (int64_t)t * sc_out_t;

                grad_combined[val_out_idx] = (T_out)d_x;

                if (r == 1)
                {
                    grad_combined[logit_out_idx] = (T_out)d_l;
                }
                else
                {
                    int64_t logit_lin_idx = (int64_t)b_idx * (T_seq * total_h * Da) + (int64_t)t * (total_h * Da) + (int64_t)h_glob * Da + (int64_t)da_idx;
                    atomicAdd(&grad_logits[logit_lin_idx], d_l);
                }

                if (t == 0)
                {
                    int64_t init_idx = (int64_t)b_idx * (total_h * D) + (int64_t)h_glob * D + (int64_t)d_idx;
                    grad_inits[init_idx] = (T_out)(g_i * a_i);
                }
            }
        }

        running_g = block_aggregate.a * running_g + block_aggregate.x;
        __syncthreads();
    }
}

template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread>
__global__ void ltv_fused_concat_register_cast_cub_backward_kernel(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_out *__restrict__ hq, const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk, const T_out *__restrict__ grad_hv,
    T_out *__restrict__ grad_combined, T_out *__restrict__ grad_inits, T_compute *__restrict__ grad_logits,
    int64_t sc_in_b, int64_t sc_in_t, int64_t sc_in_e, int64_t sc_out_b, int64_t sc_out_t, int64_t sc_out_e,
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
        ltv_fused_concat_register_cast_cub_backward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, true>(
            combined, inits, hq, hk, hv, grad_hq, grad_hk, grad_hv, grad_combined, grad_inits, grad_logits,
            sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
            s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
            s_gq_b, s_gq_h, s_gq_t, s_gq_d,
            s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d, T_seq, NH, NKVH, Da, r);
    }
    else
    {
        ltv_fused_concat_register_cast_cub_backward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, false>(
            combined, inits, hq, hk, hv, grad_hq, grad_hk, grad_hv, grad_combined, grad_inits, grad_logits,
            sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
            s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
            s_gq_b, s_gq_h, s_gq_t, s_gq_d,
            s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d, T_seq, NH, NKVH, Da, r);
    }
}

template <typename T_in, typename T_compute, typename T_out, int BlockThreads, int ItemsPerThread,
          int T_const, int NH_const, int NKVH_const, int Da_const, int r_const,
          int64_t sc_in_b_const, int64_t sc_in_t_const, int64_t sc_in_e_const,
          int64_t sc_out_b_const, int64_t sc_out_t_const, int64_t sc_out_e_const,
          int64_t s_hq_b_const, int64_t s_hq_h_const, int64_t s_hq_t_const, int64_t s_hq_d_const,
          int64_t s_hk_b_const, int64_t s_hk_h_const, int64_t s_hk_t_const, int64_t s_hk_d_const,
          int64_t s_hv_b_const, int64_t s_hv_h_const, int64_t s_hv_t_const, int64_t s_hv_d_const,
          int64_t s_gq_b_const, int64_t s_gq_h_const, int64_t s_gq_t_const, int64_t s_gq_d_const,
          int64_t s_gk_b_const, int64_t s_gk_h_const, int64_t s_gk_t_const, int64_t s_gk_d_const,
          int64_t s_gv_b_const, int64_t s_gv_h_const, int64_t s_gv_t_const, int64_t s_gv_d_const>
__global__ void ltv_fused_concat_register_cast_cub_backward_kernel_const(
    const T_in *__restrict__ combined, const T_in *__restrict__ inits,
    const T_out *__restrict__ hq, const T_out *__restrict__ hk, const T_out *__restrict__ hv,
    const T_out *__restrict__ grad_hq, const T_out *__restrict__ grad_hk, const T_out *__restrict__ grad_hv,
    T_out *__restrict__ grad_combined, T_out *__restrict__ grad_inits, float *__restrict__ grad_logits)
{
    // Only True is supported as a fully constant kernel path
    ltv_fused_concat_register_cast_cub_backward_device<T_in, T_compute, T_out, BlockThreads, ItemsPerThread, true>(
        combined, inits, hq, hk, hv, grad_hq, grad_hk, grad_hv, grad_combined, grad_inits, grad_logits,
        sc_in_b_const, sc_in_t_const, sc_in_e_const, sc_out_b_const, sc_out_t_const, sc_out_e_const,
        s_hq_b_const, s_hq_h_const, s_hq_t_const, s_hq_d_const,
        s_hk_b_const, s_hk_h_const, s_hk_t_const, s_hk_d_const,
        s_hv_b_const, s_hv_h_const, s_hv_t_const, s_hv_d_const,
        s_gq_b_const, s_gq_h_const, s_gq_t_const, s_gq_d_const, s_gk_b_const, s_gk_h_const, s_gk_t_const, s_gk_d_const,
        s_gv_b_const, s_gv_h_const, s_gv_t_const, s_gv_d_const, T_const, NH_const, NKVH_const, Da_const, r_const);
}

// -------------------------------------------------------------------------
// Launchers
// -------------------------------------------------------------------------
void ltv_fused_concat_register_cast_blelloch_scan_cuda_forward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r, int block_threads, int items_per_thread, bool use_sigmoid)
{

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t sc_b = combined.stride(0);
    int64_t sc_t = combined.stride(1);
    int64_t sc_e = combined.stride(2);

    dim3 grid(B, NH + 2 * NKVH, Da * r);
    auto dtype = combined.scalar_type();

    DISPATCH_BLOCK_THREADS(block_threads, {
        DISPATCH_ITEMS_PER_THREAD(items_per_thread, {
            constexpr bool kIsValid = (kBlockThreads * kItemsPerThread >= 128) &&
                                      (kBlockThreads * kItemsPerThread <= 2048);
            if constexpr (kIsValid)
            {
                dim3 block(kBlockThreads);

                // Only trigger specialized const kernels if use_sigmoid is true (per user constraints)
                if (use_sigmoid && T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
                {
                    if (sc_b == kStrideCombinedB && sc_t == kStrideCombinedT && sc_e == kStrideCombinedE)
                    {
                        AT_DISPATCH_SWITCH(dtype, "ltv_fused_concat_cub_forward_const",
                                           AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                                            { ltv_fused_concat_register_cast_cub_forward_kernel_const<float, float, float, kBlockThreads, kItemsPerThread,
                                                                                                                      kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE>
                                                                  <<<grid, block, 0, stream>>>(combined.data_ptr<float>(), inits.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_v.data_ptr<float>()); })
                                               AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                                { ltv_fused_concat_register_cast_cub_forward_kernel_const<at::BFloat16, float, at::BFloat16, kBlockThreads, kItemsPerThread,
                                                                                                                          kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE>
                                                                      <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(), out_q.data_ptr<at::BFloat16>(), out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>()); }));
                        return;
                    }
                    warn_specialized_stride_mismatch("ltv_fused_concat_register_cast_cub_forward", sc_b, sc_t, sc_e);
                }

                AT_DISPATCH_SWITCH(dtype, "ltv_fused_concat_cub_forward",
                                   AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                                    { ltv_fused_concat_register_cast_cub_forward_kernel<float, float, float, kBlockThreads, kItemsPerThread>
                                                          <<<grid, block, 0, stream>>>(combined.data_ptr<float>(), inits.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                                                                                       sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, use_sigmoid); })
                                       AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                        { ltv_fused_concat_register_cast_cub_forward_kernel<at::BFloat16, float, at::BFloat16, kBlockThreads, kItemsPerThread>
                                                              <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(), out_q.data_ptr<at::BFloat16>(), out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                                                                                           sc_b, sc_t, sc_e, T, NH, NKVH, Da, r, use_sigmoid); }));
            }
            else
            {
                TORCH_CHECK(false, "Unsupported hardware limits or suboptimal combination of tuning parameters. "
                                   "Valid combinations must satisfy: "
                                   "(block_threads * items_per_thread >= 128) && "
                                   "(block_threads * items_per_thread <= 2048). "
                                   "Got block_threads=", kBlockThreads,
                                   ", items_per_thread=", kItemsPerThread);
            }
        });
    });
}

void ltv_fused_concat_register_cast_blelloch_scan_cuda_backward_impl(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits,
    torch::Tensor grad_logits, int B, int T, int NH, int NKVH, int Da, int r,
    int block_threads, int items_per_thread, bool use_sigmoid)
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

    dim3 grid(B, NH + 2 * NKVH, Da * r);
    auto dtype = combined.scalar_type();

    bool backward_has_expected_saved_output_strides =
        s_hq_b == kStrideGradOutQB && s_hq_h == kStrideGradOutQH && s_hq_t == kStrideGradOutQT && s_hq_d == kStrideGradOutQD &&
        s_hk_b == kStrideGradOutKB && s_hk_h == kStrideGradOutKH && s_hk_t == kStrideGradOutKT && s_hk_d == kStrideGradOutKD &&
        s_hv_b == kStrideGradOutVB && s_hv_h == kStrideGradOutVH && s_hv_t == kStrideGradOutVT && s_hv_d == kStrideGradOutVD;

    bool backward_has_expected_grad_output_strides =
        s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQH && s_gq_t == kStrideGradOutQT && s_gq_d == kStrideGradOutQD &&
        s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKH && s_gk_t == kStrideGradOutKT && s_gk_d == kStrideGradOutKD &&
        s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH && s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD;

    bool backward_has_transposed_qk_grad_output_strides =
        s_gq_b == kStrideGradOutQB && s_gq_h == kStrideGradOutQHTransposed && s_gq_t == kStrideGradOutQTTransposed && s_gq_d == kStrideGradOutQD &&
        s_gk_b == kStrideGradOutKB && s_gk_h == kStrideGradOutKHTransposed && s_gk_t == kStrideGradOutKTTransposed && s_gk_d == kStrideGradOutKD &&
        s_gv_b == kStrideGradOutVB && s_gv_h == kStrideGradOutVH && s_gv_t == kStrideGradOutVT && s_gv_d == kStrideGradOutVD;

    bool backward_has_specialized_strides =
        sc_in_b == kStrideCombinedB && sc_in_t == kStrideCombinedT && sc_in_e == kStrideCombinedE &&
        sc_out_b == kStrideCombinedB && sc_out_t == kStrideCombinedT && sc_out_e == kStrideCombinedE &&
        backward_has_expected_saved_output_strides &&
        (backward_has_expected_grad_output_strides || backward_has_transposed_qk_grad_output_strides);

    DISPATCH_BLOCK_THREADS(block_threads, {
        DISPATCH_ITEMS_PER_THREAD(items_per_thread, {
            constexpr bool kIsValid = (kBlockThreads * kItemsPerThread >= 128) &&
                                      (kBlockThreads * kItemsPerThread <= 2048);
            if constexpr (kIsValid)
            {
                dim3 block(kBlockThreads);

                // Only trigger specialized const kernels if use_sigmoid is true
                if (use_sigmoid && T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
                {
                    if (backward_has_expected_grad_output_strides && backward_has_specialized_strides)
                    {
                        AT_DISPATCH_SWITCH(dtype, "ltv_fused_concat_cub_backward_const",
                                           AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                                            { ltv_fused_concat_register_cast_cub_backward_kernel_const<float, float, float, kBlockThreads, kItemsPerThread,
                                                                                                                       kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
                                                                                                                       kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
                                                                                                                       kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                                                                  <<<grid, block, 0, stream>>>(combined.data_ptr<float>(), inits.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                                                                                               grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(), grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(), grad_inits.data_ptr<float>(), grad_logits.data_ptr<float>()); })
                                               AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                                { ltv_fused_concat_register_cast_cub_backward_kernel_const<at::BFloat16, float, at::BFloat16, kBlockThreads, kItemsPerThread,
                                                                                                                           kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
                                                                                                                           kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
                                                                                                                           kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                                                                      <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(), out_q.data_ptr<at::BFloat16>(), out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                                                                                                   grad_out_q.data_ptr<at::BFloat16>(), grad_out_k.data_ptr<at::BFloat16>(), grad_out_v.data_ptr<at::BFloat16>(), grad_combined.data_ptr<at::BFloat16>(), grad_inits.data_ptr<at::BFloat16>(), grad_logits.data_ptr<float>()); }));
                        return;
                    }
                    if (backward_has_transposed_qk_grad_output_strides && backward_has_specialized_strides)
                    {
                        AT_DISPATCH_SWITCH(dtype, "ltv_fused_concat_cub_backward_const_transposed_qk_grad",
                                           AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                                            { ltv_fused_concat_register_cast_cub_backward_kernel_const<float, float, float, kBlockThreads, kItemsPerThread,
                                                                                                                       kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
                                                                                                                       kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
                                                                                                                       kStrideGradOutQB, kStrideGradOutQHTransposed, kStrideGradOutQTTransposed, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKHTransposed, kStrideGradOutKTTransposed, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                                                                  <<<grid, block, 0, stream>>>(combined.data_ptr<float>(), inits.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                                                                                               grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(), grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(), grad_inits.data_ptr<float>(), grad_logits.data_ptr<float>()); })
                                               AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                                { ltv_fused_concat_register_cast_cub_backward_kernel_const<at::BFloat16, float, at::BFloat16, kBlockThreads, kItemsPerThread,
                                                                                                                           kT, kNH, kNKVH, kDa, kR, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE, kStrideCombinedB, kStrideCombinedT, kStrideCombinedE,
                                                                                                                           kStrideGradOutQB, kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD,
                                                                                                                           kStrideGradOutQB, kStrideGradOutQHTransposed, kStrideGradOutQTTransposed, kStrideGradOutQD, kStrideGradOutKB, kStrideGradOutKHTransposed, kStrideGradOutKTTransposed, kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH, kStrideGradOutVT, kStrideGradOutVD>
                                                                      <<<grid, block, 0, stream>>>(combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(), out_q.data_ptr<at::BFloat16>(), out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                                                                                                   grad_out_q.data_ptr<at::BFloat16>(), grad_out_k.data_ptr<at::BFloat16>(), grad_out_v.data_ptr<at::BFloat16>(), grad_combined.data_ptr<at::BFloat16>(), grad_inits.data_ptr<at::BFloat16>(), grad_logits.data_ptr<float>()); }));
                        return;
                    }
                    warn_specialized_backward_stride_mismatch(
                        sc_in_b, sc_in_t, sc_in_e,
                        sc_out_b, sc_out_t, sc_out_e,
                        s_hq_b, s_hq_h, s_hq_t, s_hq_d,
                        s_hk_b, s_hk_h, s_hk_t, s_hk_d,
                        s_hv_b, s_hv_h, s_hv_t, s_hv_d,
                        s_gq_b, s_gq_h, s_gq_t, s_gq_d,
                        s_gk_b, s_gk_h, s_gk_t, s_gk_d,
                        s_gv_b, s_gv_h, s_gv_t, s_gv_d);
                }

                AT_DISPATCH_SWITCH(dtype, "ltv_fused_concat_cub_backward",
                                   AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                                    { ltv_fused_concat_register_cast_cub_backward_kernel<float, float, float, kBlockThreads, kItemsPerThread><<<grid, block, 0, stream>>>(
                                                          combined.data_ptr<float>(), inits.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_v.data_ptr<float>(),
                                                          grad_out_q.data_ptr<float>(), grad_out_k.data_ptr<float>(), grad_out_v.data_ptr<float>(), grad_combined.data_ptr<float>(), grad_inits.data_ptr<float>(), grad_logits.data_ptr<float>(),
                                                          sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
                                                          s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
                                                          s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d, T, NH, NKVH, Da, r, use_sigmoid); })
                                       AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                        { ltv_fused_concat_register_cast_cub_backward_kernel<at::BFloat16, float, at::BFloat16, kBlockThreads, kItemsPerThread><<<grid, block, 0, stream>>>(
                                                              combined.data_ptr<at::BFloat16>(), inits.data_ptr<at::BFloat16>(), out_q.data_ptr<at::BFloat16>(), out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
                                                              grad_out_q.data_ptr<at::BFloat16>(), grad_out_k.data_ptr<at::BFloat16>(), grad_out_v.data_ptr<at::BFloat16>(), grad_combined.data_ptr<at::BFloat16>(), grad_inits.data_ptr<at::BFloat16>(), grad_logits.data_ptr<float>(),
                                                              sc_in_b, sc_in_t, sc_in_e, sc_out_b, sc_out_t, sc_out_e,
                                                              s_hq_b, s_hq_h, s_hq_t, s_hq_d, s_hk_b, s_hk_h, s_hk_t, s_hk_d, s_hv_b, s_hv_h, s_hv_t, s_hv_d,
                                                              s_gq_b, s_gq_h, s_gq_t, s_gq_d, s_gk_b, s_gk_h, s_gk_t, s_gk_d, s_gv_b, s_gv_h, s_gv_t, s_gv_d, T, NH, NKVH, Da, r, use_sigmoid); }));
            }
            else
            {
                TORCH_CHECK(false, "Unsupported hardware limits or suboptimal combination of tuning parameters. "
                                   "Valid combinations must satisfy: "
                                   "(block_threads * items_per_thread >= 128) && "
                                   "(block_threads * items_per_thread <= 2048). "
                                   "Got block_threads=", kBlockThreads,
                                   ", items_per_thread=", kItemsPerThread);
            }
        });
    });
}
