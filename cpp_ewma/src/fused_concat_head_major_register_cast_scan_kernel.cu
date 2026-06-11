#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ltv_scan_common.cuh"

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void ltv_fused_concat_head_major_forward_device(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v, int T, int NH, int NKVH, int Da, int r,
    int64_t stride_combined_b, int64_t stride_combined_t)
{
    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int da_idx = static_cast<int>(blockIdx.z);
    int r_idx = static_cast<int>(threadIdx.x);
    int d_idx = da_idx * r + r_idx;

    int D = Da * r;
    int total_h = NH + 2 * NKVH;
    int stride_head = D + Da;

    bool is_q = h_idx < NH;
    bool is_k = h_idx >= NH && h_idx < (NH + NKVH);
    int h_local = is_q ? h_idx : (is_k ? h_idx - NH : h_idx - NH - NKVH);

    scalar_t *out_ptr;
    if (is_q)
    {
        int64_t out_base = static_cast<int64_t>(b_idx) * NH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_q + out_base;
    }
    else if (is_k)
    {
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_k + out_base;
    }
    else
    {
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        out_ptr = out_v + out_base;
    }

    const scalar_t *combined_base =
        combined + static_cast<int64_t>(b_idx) * stride_combined_b;

    int64_t inits_base = static_cast<int64_t>(b_idx) * total_h * D +
                         static_cast<int64_t>(h_idx) * D;
    float h_state =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    // Start pointers for the first time step (t = 0)
    const scalar_t *head_data = combined_base +
                                static_cast<int64_t>(h_idx) * stride_head;
    scalar_t *out_ptr_t = out_ptr; // points to out_ptr[0 * D]

    for (int t = 0; t < T; ++t)
    {
        float u_t = static_cast<float>(head_data[d_idx]);
        float l_t = static_cast<float>(head_data[D + da_idx]);

        float alpha_t = sigmoid_f32(l_t);
        h_state = (1.0f - alpha_t) * h_state + alpha_t * u_t;
        *out_ptr_t = static_cast<scalar_t>(h_state);

        // Advance by stride for the next time step
        head_data += stride_combined_t;
        out_ptr_t += D;
    }
}

template <typename scalar_t>
__global__ void ltv_fused_concat_head_major_forward_kernel(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v, int64_t stride_combined_b,
    int64_t stride_combined_t, int T, int NH, int NKVH, int Da, int r)
{
    ltv_fused_concat_head_major_forward_device(
        combined, inits, out_q, out_k, out_v, T, NH, NKVH, Da, r,
        stride_combined_b, stride_combined_t);
}

template <
    typename scalar_t, int T, int NH, int NKVH, int Da, int r,
    int64_t StrideCombinedB, int64_t StrideCombinedT>
__global__ void ltv_fused_concat_head_major_forward_kernel_const(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_k,
    scalar_t *__restrict__ out_v)
{
    ltv_fused_concat_head_major_forward_device(
        combined, inits, out_q, out_k, out_v, T, NH, NKVH, Da, r,
        StrideCombinedB, StrideCombinedT);
}

// ============================================================================
// Backward Kernel
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void ltv_fused_concat_head_major_backward_device(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits,
    int T, int NH, int NKVH, int Da, int r, int lcm_r_32,
    int64_t stride_combined_b, int64_t stride_combined_t,
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
    int da_idx = static_cast<int>(blockIdx.z);
    int r_idx = static_cast<int>(threadIdx.x);
    int d_idx = da_idx * r + r_idx;

    int D = Da * r;
    int total_h = NH + 2 * NKVH;
    int stride_head = D + Da;

    int lane_idx = r_idx % 32;
    int d_idx_over_32 = d_idx / 32;
    int warp_base_r = (r_idx / 32) * 32;
    int da_idx_start = da_idx * r;

    bool is_q = h_idx < NH;
    bool is_k = (h_idx >= NH) && (h_idx < NH + NKVH);
    int h_local = is_q ? h_idx : (is_k ? h_idx - NH : h_idx - NH - NKVH);

    bool is_group_leader = lane_idx == 0 || d_idx == da_idx_start;
    int leader_index = d_idx_over_32 + da_idx - d_idx / lcm_r_32;
    int start_leader = compute_leader_index(da_idx_start, r, 32, lcm_r_32);
    int end_leader = compute_leader_index(da_idx_start + r - 1, r, 32, lcm_r_32) + 1;

    const scalar_t *combined_base =
        combined + static_cast<int64_t>(b_idx) * stride_combined_b;

    int64_t total_e = static_cast<int64_t>(total_h) * stride_head;
    int64_t stride_grad_combined_b = static_cast<int64_t>(T) * total_e;
    int64_t stride_grad_combined_t = total_e;

    scalar_t *grad_combined_base =
        grad_combined + static_cast<int64_t>(b_idx) * stride_grad_combined_b;

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

    int active_in_warp = min(32, r - warp_base_r);
    unsigned int active_mask =
        (active_in_warp == 32) ? 0xFFFFFFFFU : (1U << active_in_warp) - 1U;

    int max_valid_lane = active_in_warp;

    // Starting offsets for t = T - 1 (first iteration)
    const scalar_t *head_data = combined_base +
                                static_cast<int64_t>(T - 1) * stride_combined_t +
                                static_cast<int64_t>(h_idx) * stride_head;
    int64_t grad_out_base = static_cast<int64_t>(b_idx) * stride_grad_out_b +
                            static_cast<int64_t>(h_local) * stride_grad_out_h +
                            static_cast<int64_t>(d_idx) * stride_grad_out_d;
    int64_t grad_out_offset = grad_out_base +
                              static_cast<int64_t>(T - 1) * stride_grad_out_t;
    scalar_t *grad_head_data = grad_combined_base +
                               static_cast<int64_t>(T - 1) * stride_grad_combined_t +
                               static_cast<int64_t>(h_idx) * stride_head;
    // Pointer to h_fwd_ptr[(t-1)*D] for t == T-1 => (T-2)*D
    const scalar_t *h_prev_ptr = (T > 1) ? h_fwd_ptr + static_cast<int64_t>(T - 2) * D
                                         : nullptr;

    for (int t = T - 1; t >= 0; --t)
    {
        float l_t = static_cast<float>(head_data[D + da_idx]);
        float alpha_t = sigmoid_f32(l_t);
        float h_prev =
            (t > 0) ? static_cast<float>(*h_prev_ptr)
                    : init_val;

        float dh_t = static_cast<float>(grad_out_ptr[grad_out_offset]) + dh_next;
        float u_t = static_cast<float>(head_data[d_idx]);

        grad_head_data[d_idx] = static_cast<scalar_t>(dh_t * alpha_t);

        float grad_logit =
            dh_t * (u_t - h_prev) * alpha_t * (1.0f - alpha_t);
        dh_next = dh_t * (1.0f - alpha_t);

        float grad_logit_sum = reduce_logit_grad_group(
            grad_logit, active_mask, lane_idx, max_valid_lane);

        if (is_group_leader)
        {
            smem_stage[leader_index] = grad_logit_sum;
        }

        __syncthreads();

        if (d_idx == da_idx_start)
        {
            float group_sum = 0.0f;
            for (int i = start_leader; i < end_leader; ++i)
            {
                group_sum += smem_stage[i];
            }

            grad_head_data[D + da_idx] = static_cast<scalar_t>(group_sum);
        }

        __syncthreads();

        // Move to previous time step
        head_data -= stride_combined_t;
        grad_out_offset -= stride_grad_out_t;
        grad_head_data -= stride_grad_combined_t;
        if (t > 0) {
            h_prev_ptr -= D;
        }
    }

    grad_inits[inits_base + static_cast<int64_t>(d_idx)] =
        static_cast<scalar_t>(dh_next);
}

template <typename scalar_t>
__global__ void ltv_fused_concat_head_major_backward_kernel(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits,
    int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_grad_out_q_b, int64_t stride_grad_out_q_h,
    int64_t stride_grad_out_q_t, int64_t stride_grad_out_q_d,
    int64_t stride_grad_out_k_b, int64_t stride_grad_out_k_h,
    int64_t stride_grad_out_k_t, int64_t stride_grad_out_k_d,
    int64_t stride_grad_out_v_b, int64_t stride_grad_out_v_h,
    int64_t stride_grad_out_v_t, int64_t stride_grad_out_v_d, int T, int NH,
    int NKVH, int Da, int r, int lcm_r_32)
{
    extern __shared__ float smem_stage[];
    ltv_fused_concat_head_major_backward_device(
        combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k,
        grad_out_v, grad_combined, grad_inits, T, NH, NKVH, Da, r, lcm_r_32,
        stride_combined_b, stride_combined_t, stride_grad_out_q_b,
        stride_grad_out_q_h, stride_grad_out_q_t, stride_grad_out_q_d,
        stride_grad_out_k_b, stride_grad_out_k_h, stride_grad_out_k_t,
        stride_grad_out_k_d, stride_grad_out_v_b, stride_grad_out_v_h,
        stride_grad_out_v_t, stride_grad_out_v_d, smem_stage);
}

template <
    typename scalar_t, int T, int NH, int NKVH, int Da, int r,
    int lcm_r_32,
    int64_t StrideCombinedB, int64_t StrideCombinedT,
    int64_t StrideGradOutQB, int64_t StrideGradOutQH,
    int64_t StrideGradOutQT, int64_t StrideGradOutQD,
    int64_t StrideGradOutKB, int64_t StrideGradOutKH,
    int64_t StrideGradOutKT, int64_t StrideGradOutKD,
    int64_t StrideGradOutVB, int64_t StrideGradOutVH,
    int64_t StrideGradOutVT, int64_t StrideGradOutVD>
__global__ void ltv_fused_concat_head_major_backward_kernel_const(
    const scalar_t *__restrict__ combined, const scalar_t *__restrict__ inits,
    const scalar_t *__restrict__ out_q, const scalar_t *__restrict__ out_k,
    const scalar_t *__restrict__ out_v, const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v,
    scalar_t *__restrict__ grad_combined, scalar_t *__restrict__ grad_inits)
{
    __shared__ float smem_stage[compute_leader_index(Da * r - 1, r, 32, lcm_r_32) + 1];
    ltv_fused_concat_head_major_backward_device(
        combined, inits, out_q, out_k, out_v, grad_out_q, grad_out_k,
        grad_out_v, grad_combined, grad_inits, T, NH, NKVH, Da, r, lcm_r_32,
        StrideCombinedB, StrideCombinedT, StrideGradOutQB, StrideGradOutQH,
        StrideGradOutQT, StrideGradOutQD, StrideGradOutKB, StrideGradOutKH,
        StrideGradOutKT, StrideGradOutKD, StrideGradOutVB, StrideGradOutVH,
        StrideGradOutVT, StrideGradOutVD, smem_stage);
}

// ============================================================================
// Launcher Logic
// ============================================================================

constexpr int kT = 1024;
constexpr int kNH = 8;
constexpr int kNKVH = 8;
constexpr int kDa = 1;
constexpr int kR = 128;
constexpr int64_t kTotalH = static_cast<int64_t>(kNH + 2 * kNKVH);
constexpr int64_t kD = static_cast<int64_t>(kDa) * kR;
constexpr int64_t kStrideHead = kD + kDa;
constexpr int64_t kTotalE = kTotalH * kStrideHead;
constexpr int64_t kStrideCombinedB = static_cast<int64_t>(kT) * kTotalE;
constexpr int64_t kStrideCombinedT = kTotalE;
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
    const char *op_name, int64_t stride_combined_b, int64_t stride_combined_t)
{
    TORCH_WARN(
        op_name,
        " matched the specialized shape but not the specialized strides; expected (",
        kStrideCombinedB, ", ", kStrideCombinedT,
        "), got (", stride_combined_b, ", ", stride_combined_t,
        "). Falling back to the generic kernel.");
}

static inline void warn_specialized_backward_stride_mismatch(
    int64_t stride_combined_b, int64_t stride_combined_t,
    int64_t stride_grad_out_q_b, int64_t stride_grad_out_q_h,
    int64_t stride_grad_out_q_t, int64_t stride_grad_out_q_d,
    int64_t stride_grad_out_k_b, int64_t stride_grad_out_k_h,
    int64_t stride_grad_out_k_t, int64_t stride_grad_out_k_d,
    int64_t stride_grad_out_v_b, int64_t stride_grad_out_v_h,
    int64_t stride_grad_out_v_t, int64_t stride_grad_out_v_d)
{
    TORCH_WARN(
        "ltv_fused_concat_head_major_register_cast_scan_cuda_backward matched the specialized "
        "shape but not the specialized strides; combined expected (",
        kStrideCombinedB, ", ", kStrideCombinedT,
        "), got (", stride_combined_b, ", ", stride_combined_t,
        "); grad_out_q expected (", kStrideGradOutQB, ", ",
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

void ltv_fused_concat_head_major_register_cast_scan_cuda_forward(
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
    bool combined_has_expected_strides =
        stride_combined_b == kStrideCombinedB &&
        stride_combined_t == kStrideCombinedT;

    if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
        dim3 grid(kNH + 2 * kNKVH, B, kDa);
        dim3 block(kR);
        if (combined_has_expected_strides)
        {
            AT_DISPATCH_SWITCH(
                dtype, "ltv_head_major_forward_const",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_head_major_forward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                kStrideCombinedT><<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>()); })
                    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                     {
            using scalar_t = at::BFloat16;
            ltv_fused_concat_head_major_forward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                kStrideCombinedT><<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>()); })
                        AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                         {
            using scalar_t = at::Half;
            ltv_fused_concat_head_major_forward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, kStrideCombinedB,
                kStrideCombinedT><<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>()); }));
            return;
        }
        warn_specialized_stride_mismatch(
            "ltv_fused_concat_head_major_register_cast_scan_cuda_forward",
            stride_combined_b, stride_combined_t);
    }

    dim3 grid(NH + 2 * NKVH, B, Da);
    dim3 block(r);
    AT_DISPATCH_SWITCH(
        dtype, "ltv_head_major_forward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_concat_head_major_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), stride_combined_b, stride_combined_t,
                T, NH, NKVH, Da, r); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                             {
        using scalar_t = at::BFloat16;
        ltv_fused_concat_head_major_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), stride_combined_b, stride_combined_t,
                T, NH, NKVH, Da, r); })
                AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                 {
        using scalar_t = at::Half;
        ltv_fused_concat_head_major_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), stride_combined_b, stride_combined_t,
                T, NH, NKVH, Da, r); }));
}

void ltv_fused_concat_head_major_register_cast_scan_cuda_backward(
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
        stride_combined_t == kStrideCombinedT;
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
        constexpr int lcm_r_32 = std::lcm(kR, 32);

        dim3 grid(kNH + 2 * kNKVH, B, kDa);
        dim3 block(kR);
        if (backward_has_expected_strides)
        {
            AT_DISPATCH_SWITCH(
                dtype, "ltv_head_major_backward_const",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
                        AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                         {
            using scalar_t = at::Half;
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
                dtype, "ltv_head_major_backward_const_transposed_qk_grad",
                AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                 {
            using scalar_t = float;
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
                        AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                         {
            using scalar_t = at::Half;
            ltv_fused_concat_head_major_backward_kernel_const<
                scalar_t, kT, kNH, kNKVH, kDa, kR, lcm_r_32, kStrideCombinedB,
                kStrideCombinedT, kStrideGradOutQB,
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
            stride_combined_b, stride_combined_t,
            stride_grad_out_q_b, stride_grad_out_q_h, stride_grad_out_q_t,
            stride_grad_out_q_d, stride_grad_out_k_b, stride_grad_out_k_h,
            stride_grad_out_k_t, stride_grad_out_k_d, stride_grad_out_v_b,
            stride_grad_out_v_h, stride_grad_out_v_t, stride_grad_out_v_d);
    }

    int lcm_r_32 = std::lcm(r, 32);
    int num_leaders = compute_leader_index(Da * r - 1, r, 32, lcm_r_32) + 1;
    int smem_size = static_cast<int>(num_leaders * sizeof(float));

    dim3 grid(NH + 2 * NKVH, B, Da);
    dim3 block(r);
    AT_DISPATCH_SWITCH(
        dtype, "ltv_head_major_backward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_concat_head_major_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(),
                grad_combined.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), stride_combined_b,
                stride_combined_t, stride_grad_out_q_b,
                stride_grad_out_q_h, stride_grad_out_q_t,
                stride_grad_out_q_d, stride_grad_out_k_b,
                stride_grad_out_k_h, stride_grad_out_k_t,
                stride_grad_out_k_d, stride_grad_out_v_b,
                stride_grad_out_v_h, stride_grad_out_v_t,
                stride_grad_out_v_d, T, NH, NKVH, Da, r, lcm_r_32); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                             {
        using scalar_t = at::BFloat16;
        ltv_fused_concat_head_major_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(),
                grad_combined.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), stride_combined_b,
                stride_combined_t, stride_grad_out_q_b,
                stride_grad_out_q_h, stride_grad_out_q_t,
                stride_grad_out_q_d, stride_grad_out_k_b,
                stride_grad_out_k_h, stride_grad_out_k_t,
                stride_grad_out_k_d, stride_grad_out_v_b,
                stride_grad_out_v_h, stride_grad_out_v_t,
                stride_grad_out_v_d, T, NH, NKVH, Da, r, lcm_r_32); })
                AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                 {
        using scalar_t = at::Half;
        ltv_fused_concat_head_major_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                combined.data_ptr<scalar_t>(), inits.data_ptr<scalar_t>(),
                out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(),
                out_v.data_ptr<scalar_t>(), grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(),
                grad_combined.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), stride_combined_b,
                stride_combined_t, stride_grad_out_q_b,
                stride_grad_out_q_h, stride_grad_out_q_t,
                stride_grad_out_q_d, stride_grad_out_k_b,
                stride_grad_out_k_h, stride_grad_out_k_t,
                stride_grad_out_k_d, stride_grad_out_v_b,
                stride_grad_out_v_h, stride_grad_out_v_t,
                stride_grad_out_v_d, T, NH, NKVH, Da, r, lcm_r_32); }));
}
