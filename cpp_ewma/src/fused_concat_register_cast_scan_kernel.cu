#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ltv_scan_common.cuh"

// ----------------------------------------------------------------------------
// Forward Kernel
// ----------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ void ltv_fused_concat_register_cast_scan_cuda_forward_device(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v, int T, int NH, int NKVH, int Da, int r,
    int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_combined_e)
{
    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int d_idx = static_cast<int>(threadIdx.x);

    int D = Da * r;
    int total_h = NH + 2 * NKVH;
    bool is_q = h_idx < NH;
    bool is_k = h_idx >= NH && h_idx < NH + NKVH;

    scalar_t *out_ptr;
    int h_local;
    int64_t value_offset;
    if (is_q)
    {
        h_local = h_idx;
        value_offset = static_cast<int64_t>(h_idx) * D + static_cast<int64_t>(d_idx);
        int64_t out_base = static_cast<int64_t>(b_idx) * NH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_q + out_base;
    }
    else if (is_k)
    {
        h_local = h_idx - NH;
        value_offset = static_cast<int64_t>(NH) * D +
                       static_cast<int64_t>(h_local) * D +
                       static_cast<int64_t>(d_idx);
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_k + out_base;
    }
    else
    {
        h_local = h_idx - NH - NKVH;
        value_offset = static_cast<int64_t>(NH + NKVH) * D +
                       static_cast<int64_t>(h_local) * D +
                       static_cast<int64_t>(d_idx);
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_v + out_base;
    }

    int da_idx = d_idx / r;
    int64_t logit_offset = static_cast<int64_t>(total_h) * D +
                           static_cast<int64_t>(h_idx) * Da +
                           static_cast<int64_t>(da_idx);

    const scalar_t *combined_base =
        combined + static_cast<int64_t>(b_idx) * stride_combined_b;
    const scalar_t *value_ptr =
        combined_base + value_offset * stride_combined_e;
    const scalar_t *logit_ptr =
        combined_base + logit_offset * stride_combined_e;

    int64_t inits_base = static_cast<int64_t>(b_idx) * total_h * D +
                         static_cast<int64_t>(h_idx) * D;
    float h_state =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    // Initialize pointer chasers to remove multiplications inside the loop
    const scalar_t *cur_value_ptr = value_ptr;
    const scalar_t *cur_logit_ptr = logit_ptr;
    scalar_t *cur_out_ptr = out_ptr;

    for (int t = 0; t < T; ++t)
    {
        float u_t = static_cast<float>(*cur_value_ptr);
        float l_t = static_cast<float>(*cur_logit_ptr);
        float alpha_t = sigmoid_f32(l_t);
        h_state = (1.0f - alpha_t) * h_state + alpha_t * u_t;
        *cur_out_ptr = static_cast<scalar_t>(h_state);

        cur_value_ptr += stride_combined_t;
        cur_logit_ptr += stride_combined_t;
        cur_out_ptr += D;
    }
}

template <typename scalar_t>
__global__ void ltv_fused_concat_register_cast_scan_cuda_forward_kernel(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v, int64_t stride_combined_b,
    int64_t stride_combined_t, int64_t stride_combined_e, int T, int NH,
    int NKVH, int Da, int r)
{
    ltv_fused_concat_register_cast_scan_cuda_forward_device(
        combined, inits, out_q, out_k, out_v, T, NH, NKVH, Da, r,
        stride_combined_b, stride_combined_t, stride_combined_e);
}

template <
    typename scalar_t, int T, int NH, int NKVH, int Da, int r,
    int64_t StrideCombinedB, int64_t StrideCombinedT, int64_t StrideCombinedE>
__global__ void ltv_fused_concat_register_cast_scan_cuda_forward_kernel_const(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v)
{
    ltv_fused_concat_register_cast_scan_cuda_forward_device(
        combined, inits, out_q, out_k, out_v, T, NH, NKVH, Da, r,
        StrideCombinedB, StrideCombinedT, StrideCombinedE);
}

// ----------------------------------------------------------------------------
// Backward Kernel
// ----------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ void ltv_fused_concat_register_cast_scan_cuda_backward_device(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits,
    int T, int NH, int NKVH, int Da, int r, int lcm_r_32, int64_t stride_combined_b,
    int64_t stride_combined_t, int64_t stride_combined_e,
    int64_t stride_grad_out_q_b, int64_t stride_grad_out_q_h,
    int64_t stride_grad_out_q_t, int64_t stride_grad_out_q_d,
    int64_t stride_grad_out_k_b, int64_t stride_grad_out_k_h,
    int64_t stride_grad_out_k_t, int64_t stride_grad_out_k_d,
    int64_t stride_grad_out_v_b, int64_t stride_grad_out_v_h,
    int64_t stride_grad_out_v_t, int64_t stride_grad_out_v_d,
    float *smem_stage)
{
    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int d_idx = static_cast<int>(threadIdx.x);

    int D = Da * r;
    int total_h = NH + 2 * NKVH;
    int lane_idx = d_idx % 32;
    int d_idx_over_32 = d_idx / 32;
    int warp_base_d = d_idx_over_32 * 32;
    int da_idx = d_idx / r;
    int da_idx_start = da_idx * r;
    bool is_q = h_idx < NH;
    bool is_k = (h_idx >= NH) && (h_idx < NH + NKVH);
    int h_local = is_q ? h_idx : (is_k ? h_idx - NH : h_idx - NH - NKVH);

    bool is_group_leader = lane_idx == 0 || d_idx == da_idx_start;
    int leader_index = d_idx_over_32 + da_idx - d_idx / lcm_r_32;
    int start_leader = compute_leader_index(da_idx_start, r, 32, lcm_r_32);
    int end_leader = compute_leader_index(da_idx_start + r - 1, r, 32, lcm_r_32) + 1;

    int64_t value_offset;
    if (is_q)
    {
        value_offset =
            static_cast<int64_t>(h_idx) * D + static_cast<int64_t>(d_idx);
    }
    else if (is_k)
    {
        value_offset = static_cast<int64_t>(NH) * D +
                       static_cast<int64_t>(h_local) * D +
                       static_cast<int64_t>(d_idx);
    }
    else
    {
        value_offset = static_cast<int64_t>(NH + NKVH) * D +
                       static_cast<int64_t>(h_local) * D +
                       static_cast<int64_t>(d_idx);
    }

    int64_t logit_offset = static_cast<int64_t>(total_h) * D +
                           static_cast<int64_t>(h_idx) * Da +
                           static_cast<int64_t>(da_idx);

    const scalar_t *combined_base =
        combined + static_cast<int64_t>(b_idx) * stride_combined_b;
    const scalar_t *value_ptr =
        combined_base + value_offset * stride_combined_e;
    const scalar_t *logit_ptr =
        combined_base + logit_offset * stride_combined_e;

    int64_t total_e =
        static_cast<int64_t>(total_h) * D + static_cast<int64_t>(total_h) * Da;
    int64_t stride_grad_combined_b = static_cast<int64_t>(T) * total_e;
    int64_t stride_grad_combined_t = total_e;

    scalar_t *grad_combined_base =
        grad_combined + static_cast<int64_t>(b_idx) * stride_grad_combined_b;
    scalar_t *grad_value_ptr = grad_combined_base + value_offset;
    scalar_t *grad_logit_ptr = grad_combined_base + logit_offset;

    int64_t h_fwd_offset = is_q ? (static_cast<int64_t>(b_idx) * NH * T * D +
                                   static_cast<int64_t>(h_local) * T * D +
                                   static_cast<int64_t>(d_idx))
                                : (static_cast<int64_t>(b_idx) * NKVH * T * D +
                                   static_cast<int64_t>(h_local) * T * D +
                                   static_cast<int64_t>(d_idx));
    const scalar_t *h_fwd_ptr =
        (is_q ? out_q : (is_k ? out_k : out_v)) + h_fwd_offset;

    const scalar_t *grad_out_ptr;
    int64_t stride_grad_out_b;
    int64_t stride_grad_out_h;
    int64_t stride_grad_out_t;
    int64_t stride_grad_out_d;
    if (is_q)
    {
        grad_out_ptr = grad_out_q;
        stride_grad_out_b = stride_grad_out_q_b;
        stride_grad_out_h = stride_grad_out_q_h;
        stride_grad_out_t = stride_grad_out_q_t;
        stride_grad_out_d = stride_grad_out_q_d;
    }
    else if (is_k)
    {
        grad_out_ptr = grad_out_k;
        stride_grad_out_b = stride_grad_out_k_b;
        stride_grad_out_h = stride_grad_out_k_h;
        stride_grad_out_t = stride_grad_out_k_t;
        stride_grad_out_d = stride_grad_out_k_d;
    }
    else
    {
        grad_out_ptr = grad_out_v;
        stride_grad_out_b = stride_grad_out_v_b;
        stride_grad_out_h = stride_grad_out_v_h;
        stride_grad_out_t = stride_grad_out_v_t;
        stride_grad_out_d = stride_grad_out_v_d;
    }

    int64_t inits_base = static_cast<int64_t>(b_idx) * total_h * D +
                         static_cast<int64_t>(h_idx) * D;
    float dh_next = 0.0f;
    float init_val =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    int active_in_warp = min(32, D - warp_base_d);
    unsigned int active_mask =
        (active_in_warp == 32) ? 0xFFFFFFFFU : (1U << active_in_warp) - 1U;

    int next_group_start_d = (da_idx + 1) * r;
    int max_valid_lane = min(active_in_warp, next_group_start_d - warp_base_d);

    // Initialize pointer chasers starting from the end (T - 1)
    const scalar_t *cur_value_ptr = value_ptr + static_cast<int64_t>(T - 1) * stride_combined_t;
    const scalar_t *cur_logit_ptr = logit_ptr + static_cast<int64_t>(T - 1) * stride_combined_t;
    const scalar_t *cur_h_fwd_ptr = h_fwd_ptr + static_cast<int64_t>(T - 2) * D;

    int64_t grad_out_base_offset = static_cast<int64_t>(b_idx) * stride_grad_out_b +
                                   static_cast<int64_t>(h_local) * stride_grad_out_h +
                                   static_cast<int64_t>(d_idx) * stride_grad_out_d;
    const scalar_t *cur_grad_out_ptr = grad_out_ptr + grad_out_base_offset + static_cast<int64_t>(T - 1) * stride_grad_out_t;

    scalar_t *cur_grad_value_ptr = grad_value_ptr + static_cast<int64_t>(T - 1) * stride_grad_combined_t;
    scalar_t *cur_grad_logit_ptr = grad_logit_ptr + static_cast<int64_t>(T - 1) * stride_grad_combined_t;

    for (int t = T - 1; t >= 0; --t)
    {
        float l_t = static_cast<float>(*cur_logit_ptr);
        float alpha_t = sigmoid_f32(l_t);
        float h_prev = (t > 0) ? static_cast<float>(*cur_h_fwd_ptr) : init_val;

        float dh_t = static_cast<float>(*cur_grad_out_ptr) + dh_next;
        float u_t = static_cast<float>(*cur_value_ptr);

        *cur_grad_value_ptr = static_cast<scalar_t>(dh_t * alpha_t);
        float grad_logit = dh_t * (u_t - h_prev) * alpha_t * (1.0f - alpha_t);
        dh_next = dh_t * (1.0f - alpha_t);

        float grad_logit_sum = reduce_logit_grad_group(
            grad_logit, active_mask, lane_idx, max_valid_lane);

        if (is_group_leader)
        {
            smem_stage[leader_index] = grad_logit_sum;
        }

        __syncthreads();

        // O(r / 32)
        if (d_idx == da_idx_start)
        {
            float group_sum = 0.0f;
            for (int i = start_leader; i < end_leader; ++i)
            {
                group_sum += smem_stage[i];
            }
            *cur_grad_logit_ptr = static_cast<scalar_t>(group_sum);
        }

        __syncthreads();

        // Decrement pointers for the next iteration step
        cur_value_ptr -= stride_combined_t;
        cur_logit_ptr -= stride_combined_t;
        cur_h_fwd_ptr -= D;
        cur_grad_out_ptr -= stride_grad_out_t;
        cur_grad_value_ptr -= stride_grad_combined_t;
        cur_grad_logit_ptr -= stride_grad_combined_t;
    }

    grad_inits[inits_base + static_cast<int64_t>(d_idx)] =
        static_cast<scalar_t>(dh_next);
}

template <typename scalar_t>
__global__ void ltv_fused_concat_register_cast_scan_cuda_backward_kernel(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits,
    int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_combined_e, int64_t stride_grad_out_q_b,
    int64_t stride_grad_out_q_h, int64_t stride_grad_out_q_t,
    int64_t stride_grad_out_q_d, int64_t stride_grad_out_k_b,
    int64_t stride_grad_out_k_h, int64_t stride_grad_out_k_t,
    int64_t stride_grad_out_k_d, int64_t stride_grad_out_v_b,
    int64_t stride_grad_out_v_h, int64_t stride_grad_out_v_t,
    int64_t stride_grad_out_v_d, int T, int NH, int NKVH, int Da, int r,
    int lcm_r_32)
{
    extern __shared__ float smem_stage[];
    ltv_fused_concat_register_cast_scan_cuda_backward_device(
        combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k, grad_out_v,
        grad_combined, grad_inits, T, NH, NKVH, Da, r, lcm_r_32, stride_combined_b,
        stride_combined_t, stride_combined_e, stride_grad_out_q_b,
        stride_grad_out_q_h, stride_grad_out_q_t, stride_grad_out_q_d,
        stride_grad_out_k_b, stride_grad_out_k_h, stride_grad_out_k_t,
        stride_grad_out_k_d, stride_grad_out_v_b, stride_grad_out_v_h,
        stride_grad_out_v_t, stride_grad_out_v_d, smem_stage);
}

template <
    typename scalar_t, int T, int NH, int NKVH, int Da, int r,
    int LcmR32,
    int64_t StrideCombinedB, int64_t StrideCombinedT, int64_t StrideCombinedE,
    int64_t StrideGradOutQB, int64_t StrideGradOutQH,
    int64_t StrideGradOutQT, int64_t StrideGradOutQD,
    int64_t StrideGradOutKB, int64_t StrideGradOutKH,
    int64_t StrideGradOutKT, int64_t StrideGradOutKD,
    int64_t StrideGradOutVB, int64_t StrideGradOutVH,
    int64_t StrideGradOutVT, int64_t StrideGradOutVD>
__global__ void ltv_fused_concat_register_cast_scan_cuda_backward_kernel_const(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits)
{
    __shared__ float smem_stage[compute_leader_index(Da * r - 1, r, 32, LcmR32) + 1];
    ltv_fused_concat_register_cast_scan_cuda_backward_device(
        combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k, grad_out_v,
        grad_combined, grad_inits, T, NH, NKVH, Da, r, LcmR32, StrideCombinedB,
        StrideCombinedT, StrideCombinedE, StrideGradOutQB, StrideGradOutQH,
        StrideGradOutQT, StrideGradOutQD, StrideGradOutKB, StrideGradOutKH,
        StrideGradOutKT, StrideGradOutKD, StrideGradOutVB, StrideGradOutVH,
        StrideGradOutVT, StrideGradOutVD,
        smem_stage);
}

// ----------------------------------------------------------------------------
// Launcher Logic
// ----------------------------------------------------------------------------

constexpr int kT = 1024;
constexpr int kNH = 8;
constexpr int kNKVH = 8;
constexpr int kDa = 1;
constexpr int kR = 128;
constexpr int kLcmR32 = std::lcm(kR, 32);
constexpr int64_t kTotalH = static_cast<int64_t>(kNH + 2 * kNKVH);
constexpr int64_t kD = static_cast<int64_t>(kDa) * kR;
constexpr int64_t kTotalE = kTotalH * kD + kTotalH * kDa;
constexpr int64_t kStrideCombinedB = static_cast<int64_t>(kT) * kTotalE;
constexpr int64_t kStrideCombinedT = kTotalE;
constexpr int64_t kStrideCombinedE = 1;
constexpr int64_t kStrideGradOutQB =
    static_cast<int64_t>(kNH) * kT * kD;
constexpr int64_t kStrideGradOutQH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutQT = kD;
constexpr int64_t kStrideGradOutQD = 1;
constexpr int64_t kStrideGradOutKB =
    static_cast<int64_t>(kNKVH) * kT * kD;
constexpr int64_t kStrideGradOutKH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutKT = kD;
constexpr int64_t kStrideGradOutKD = 1;
constexpr int64_t kStrideGradOutVB =
    static_cast<int64_t>(kNKVH) * kT * kD;
constexpr int64_t kStrideGradOutVH = static_cast<int64_t>(kT) * kD;
constexpr int64_t kStrideGradOutVT = kD;
constexpr int64_t kStrideGradOutVD = 1;
constexpr int64_t kStrideGradOutQHTransposed = kD;
constexpr int64_t kStrideGradOutQTTransposed = static_cast<int64_t>(kNH) * kD;
constexpr int64_t kStrideGradOutKHTransposed = kD;
constexpr int64_t kStrideGradOutKTTransposed = static_cast<int64_t>(kNKVH) * kD;

static inline void check_valid_tensor(const torch::Tensor &t, at::ScalarType dtype,
                                      const char *name, bool check_contig = true)
{
    if (check_contig)
    {
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    }
    TORCH_CHECK(t.scalar_type() == dtype, name, " must match dtype");
}

static inline void warn_specialized_stride_mismatch(
    const char *op_name, int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_combined_e)
{
    TORCH_WARN(
        op_name,
        " matched the specialized shape but not the specialized strides; expected (",
        kStrideCombinedB, ", ", kStrideCombinedT, ", ", kStrideCombinedE,
        "), got (", stride_combined_b, ", ", stride_combined_t, ", ",
        stride_combined_e, "). Falling back to the generic kernel.");
}

static inline void warn_specialized_backward_stride_mismatch(
    int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_combined_e, int64_t stride_grad_out_q_b,
    int64_t stride_grad_out_q_h, int64_t stride_grad_out_q_t,
    int64_t stride_grad_out_q_d, int64_t stride_grad_out_k_b,
    int64_t stride_grad_out_k_h, int64_t stride_grad_out_k_t,
    int64_t stride_grad_out_k_d, int64_t stride_grad_out_v_b,
    int64_t stride_grad_out_v_h, int64_t stride_grad_out_v_t,
    int64_t stride_grad_out_v_d)
{
    TORCH_WARN(
        "ltv_fused_concat_register_cast_scan_cuda_backward matched the specialized "
        "shape but not the specialized strides; combined expected (",
        kStrideCombinedB, ", ", kStrideCombinedT, ", ", kStrideCombinedE,
        "), got (", stride_combined_b, ", ", stride_combined_t, ", ",
        stride_combined_e, "); grad_out_q expected (", kStrideGradOutQB, ", ",
        kStrideGradOutQH, ", ", kStrideGradOutQT, ", ", kStrideGradOutQD,
        "), got (", stride_grad_out_q_b, ", ", stride_grad_out_q_h, ", ",
        stride_grad_out_q_t, ", ", stride_grad_out_q_d, "); grad_out_k expected (",
        kStrideGradOutKB, ", ", kStrideGradOutKH, ", ", kStrideGradOutKT, ", ",
        kStrideGradOutKD, "), got (", stride_grad_out_k_b, ", ",
        stride_grad_out_k_h, ", ", stride_grad_out_k_t, ", ",
        stride_grad_out_k_d, "); grad_out_v expected (", kStrideGradOutVB, ", ",
        kStrideGradOutVH, ", ", kStrideGradOutVT, ", ", kStrideGradOutVD,
        "), got (", stride_grad_out_v_b, ", ", stride_grad_out_v_h, ", ",
        stride_grad_out_v_t, ", ", stride_grad_out_v_d,
        "). Falling back to the generic kernel.");
}

void ltv_fused_concat_register_cast_scan_cuda_forward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, int B, int T, int NH, int NKVH,
    int Da, int r)
{
    auto dtype = combined.scalar_type();
    check_valid_tensor(combined, dtype, "combined", false);
    check_valid_tensor(inits, dtype, "inits");
    check_valid_tensor(out_q, dtype, "out_q");
    check_valid_tensor(out_k, dtype, "out_k");
    check_valid_tensor(out_v, dtype, "out_v");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t stride_combined_b = combined.stride(0);
    int64_t stride_combined_t = combined.stride(1);
    int64_t stride_combined_e = combined.stride(2);
    bool combined_has_expected_strides =
        stride_combined_b == kStrideCombinedB &&
        stride_combined_t == kStrideCombinedT &&
        stride_combined_e == kStrideCombinedE;

    if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
        dim3 grid(kNH + 2 * kNKVH, B);
        dim3 block(kDa * kR);
        if (combined_has_expected_strides)
        {
            AT_DISPATCH_SWITCH(
                dtype, "ltv_register_cast_forward_const",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_register_cast_scan_cuda_forward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE><<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>()); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                       {
            using scalar_t = at::BFloat16;
            ltv_fused_concat_register_cast_scan_cuda_forward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE><<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>()); }));
            return;
        }
        warn_specialized_stride_mismatch(
            "ltv_fused_concat_register_cast_scan_cuda_forward", stride_combined_b,
            stride_combined_t, stride_combined_e);
    }

    dim3 grid(NH + 2 * NKVH, B);
    dim3 block(Da * r);
    AT_DISPATCH_SWITCH(
        dtype, "ltv_register_cast_forward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_concat_register_cast_scan_cuda_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), stride_combined_b, stride_combined_t,
                stride_combined_e, T, NH, NKVH, Da, r); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                               {
        using scalar_t = at::BFloat16;
        ltv_fused_concat_register_cast_scan_cuda_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), stride_combined_b, stride_combined_t,
                stride_combined_e, T, NH, NKVH, Da, r); }));
}

void ltv_fused_concat_register_cast_scan_cuda_backward(
    torch::Tensor combined, torch::Tensor inits, torch::Tensor out_q,
    torch::Tensor out_k, torch::Tensor out_v, torch::Tensor grad_out_q,
    torch::Tensor grad_out_k, torch::Tensor grad_out_v,
    torch::Tensor grad_combined, torch::Tensor grad_inits, int B, int T, int NH,
    int NKVH, int Da, int r)
{
    auto dtype = combined.scalar_type();
    check_valid_tensor(combined, dtype, "combined", false);
    check_valid_tensor(grad_combined, dtype, "grad_combined", true);
    check_valid_tensor(grad_inits, dtype, "grad_inits", true);
    check_valid_tensor(out_q, dtype, "out_q", true);
    check_valid_tensor(out_k, dtype, "out_k", true);
    check_valid_tensor(out_v, dtype, "out_v", true);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t stride_combined_b = combined.stride(0);
    int64_t stride_combined_t = combined.stride(1);
    int64_t stride_combined_e = combined.stride(2);

    int64_t stride_grad_out_q_b = grad_out_q.stride(0);
    int64_t stride_grad_out_q_h = grad_out_q.stride(1);
    int64_t stride_grad_out_q_t = grad_out_q.stride(2);
    int64_t stride_grad_out_q_d = grad_out_q.stride(3);
    int64_t stride_grad_out_k_b = grad_out_k.stride(0);
    int64_t stride_grad_out_k_h = grad_out_k.stride(1);
    int64_t stride_grad_out_k_t = grad_out_k.stride(2);
    int64_t stride_grad_out_k_d = grad_out_k.stride(3);
    int64_t stride_grad_out_v_b = grad_out_v.stride(0);
    int64_t stride_grad_out_v_h = grad_out_v.stride(1);
    int64_t stride_grad_out_v_t = grad_out_v.stride(2);
    int64_t stride_grad_out_v_d = grad_out_v.stride(3);
    bool combined_has_expected_strides =
        stride_combined_b == kStrideCombinedB &&
        stride_combined_t == kStrideCombinedT &&
        stride_combined_e == kStrideCombinedE;
    bool grad_out_q_has_expected_strides =
        stride_grad_out_q_b == kStrideGradOutQB &&
        stride_grad_out_q_h == kStrideGradOutQH &&
        stride_grad_out_q_t == kStrideGradOutQT &&
        stride_grad_out_q_d == kStrideGradOutQD;
    bool grad_out_q_has_transposed_strides =
        stride_grad_out_q_b == kStrideGradOutQB &&
        stride_grad_out_q_h == kStrideGradOutQHTransposed &&
        stride_grad_out_q_t == kStrideGradOutQTTransposed &&
        stride_grad_out_q_d == kStrideGradOutQD;
    bool grad_out_k_has_expected_strides =
        stride_grad_out_k_b == kStrideGradOutKB &&
        stride_grad_out_k_h == kStrideGradOutKH &&
        stride_grad_out_k_t == kStrideGradOutKT &&
        stride_grad_out_k_d == kStrideGradOutKD;
    bool grad_out_k_has_transposed_strides =
        stride_grad_out_k_b == kStrideGradOutKB &&
        stride_grad_out_k_h == kStrideGradOutKHTransposed &&
        stride_grad_out_k_t == kStrideGradOutKTTransposed &&
        stride_grad_out_k_d == kStrideGradOutKD;
    bool grad_out_v_has_expected_strides =
        stride_grad_out_v_b == kStrideGradOutVB &&
        stride_grad_out_v_h == kStrideGradOutVH &&
        stride_grad_out_v_t == kStrideGradOutVT &&
        stride_grad_out_v_d == kStrideGradOutVD;
    bool backward_has_expected_strides =
        combined_has_expected_strides && grad_out_q_has_expected_strides &&
        grad_out_k_has_expected_strides && grad_out_v_has_expected_strides;
    bool backward_has_transposed_qk_grad_strides =
        combined_has_expected_strides && grad_out_q_has_transposed_strides &&
        grad_out_k_has_transposed_strides && grad_out_v_has_expected_strides;

    if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
        dim3 grid(kNH + 2 * kNKVH, B);
        dim3 block(kDa * kR);
        if (backward_has_expected_strides)
        {
            AT_DISPATCH_SWITCH(
                dtype, "ltv_register_cast_backward_const",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_register_cast_scan_cuda_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kLcmR32, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
                kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT,
                kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                kStrideGradOutVT, kStrideGradOutVD>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                    out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                    out_v.data_ptr<scalar_t>(),
                    grad_out_q.data_ptr<scalar_t>(),
                    grad_out_k.data_ptr<scalar_t>(),
                    grad_out_v.data_ptr<scalar_t>(),
                    grad_combined.data_ptr<scalar_t>(),
                    grad_inits.data_ptr<scalar_t>()); })
                    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                     {
            using scalar_t = at::BFloat16;
            ltv_fused_concat_register_cast_scan_cuda_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kLcmR32, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                kStrideGradOutQH, kStrideGradOutQT, kStrideGradOutQD,
                kStrideGradOutKB, kStrideGradOutKH, kStrideGradOutKT,
                kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                kStrideGradOutVT, kStrideGradOutVD>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                    out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                    out_v.data_ptr<scalar_t>(),
                    grad_out_q.data_ptr<scalar_t>(),
                    grad_out_k.data_ptr<scalar_t>(),
                    grad_out_v.data_ptr<scalar_t>(),
                    grad_combined.data_ptr<scalar_t>(),
                    grad_inits.data_ptr<scalar_t>()); }));
            return;
        }
        if (backward_has_transposed_qk_grad_strides)
        {
            AT_DISPATCH_SWITCH(
                dtype, "ltv_register_cast_backward_const_transposed_qk_grad",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_register_cast_scan_cuda_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kLcmR32, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                kStrideGradOutQHTransposed, kStrideGradOutQTTransposed,
                kStrideGradOutQD, kStrideGradOutKB,
                kStrideGradOutKHTransposed, kStrideGradOutKTTransposed,
                kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                kStrideGradOutVT, kStrideGradOutVD>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                    out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                    out_v.data_ptr<scalar_t>(),
                    grad_out_q.data_ptr<scalar_t>(),
                    grad_out_k.data_ptr<scalar_t>(),
                    grad_out_v.data_ptr<scalar_t>(),
                    grad_combined.data_ptr<scalar_t>(),
                    grad_inits.data_ptr<scalar_t>()); })
                    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                     {
            using scalar_t = at::BFloat16;
            ltv_fused_concat_register_cast_scan_cuda_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kLcmR32, kStrideCombinedB,
                kStrideCombinedT, kStrideCombinedE, kStrideGradOutQB,
                kStrideGradOutQHTransposed, kStrideGradOutQTTransposed,
                kStrideGradOutQD, kStrideGradOutKB,
                kStrideGradOutKHTransposed, kStrideGradOutKTTransposed,
                kStrideGradOutKD, kStrideGradOutVB, kStrideGradOutVH,
                kStrideGradOutVT, kStrideGradOutVD>
                <<<grid, block, 0, stream>>>(
                    combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                    out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                    out_v.data_ptr<scalar_t>(),
                    grad_out_q.data_ptr<scalar_t>(),
                    grad_out_k.data_ptr<scalar_t>(),
                    grad_out_v.data_ptr<scalar_t>(),
                    grad_combined.data_ptr<scalar_t>(),
                    grad_inits.data_ptr<scalar_t>()); }));
            return;
        }
        warn_specialized_backward_stride_mismatch(
            stride_combined_b, stride_combined_t, stride_combined_e,
            stride_grad_out_q_b, stride_grad_out_q_h, stride_grad_out_q_t,
            stride_grad_out_q_d, stride_grad_out_k_b, stride_grad_out_k_h,
            stride_grad_out_k_t, stride_grad_out_k_d, stride_grad_out_v_b,
            stride_grad_out_v_h, stride_grad_out_v_t, stride_grad_out_v_d);
    }

    int lcm_r_32 = std::lcm(r, 32);
    int num_leaders = compute_leader_index(Da * r - 1, r, 32, lcm_r_32) + 1;
    int smem_size = static_cast<int>(num_leaders * sizeof(float));

    dim3 grid(NH + 2 * NKVH, B);
    dim3 block(Da * r);
    AT_DISPATCH_SWITCH(
        dtype, "ltv_register_cast_backward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_concat_register_cast_scan_cuda_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(),
                grad_combined.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), stride_combined_b,
                stride_combined_t, stride_combined_e, stride_grad_out_q_b,
                stride_grad_out_q_h, stride_grad_out_q_t,
                stride_grad_out_q_d, stride_grad_out_k_b,
                stride_grad_out_k_h, stride_grad_out_k_t,
                stride_grad_out_k_d, stride_grad_out_v_b,
                stride_grad_out_v_h, stride_grad_out_v_t,
                stride_grad_out_v_d, T, NH, NKVH, Da, r, lcm_r_32); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                             {
        using scalar_t = at::BFloat16;
        ltv_fused_concat_register_cast_scan_cuda_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(),
                grad_combined.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), stride_combined_b,
                stride_combined_t, stride_combined_e, stride_grad_out_q_b,
                stride_grad_out_q_h, stride_grad_out_q_t,
                stride_grad_out_q_d, stride_grad_out_k_b,
                stride_grad_out_k_h, stride_grad_out_k_t,
                stride_grad_out_k_d, stride_grad_out_v_b,
                stride_grad_out_v_h, stride_grad_out_v_t,
                stride_grad_out_v_d, T, NH, NKVH, Da, r, lcm_r_32); }));
}
