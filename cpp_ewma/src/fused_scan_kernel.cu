#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <numeric>
#include <torch/extension.h>

#include "ltv_scan_common.cuh"

// ----------------------------------------------------------------------------
// Forward Kernel
// ----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void ltv_fused_scan_forward_kernel(
    const scalar_t *__restrict__ q, const scalar_t *__restrict__ k,
    const scalar_t *__restrict__ v, const scalar_t *__restrict__ logits,
    const scalar_t *__restrict__ inits, scalar_t *__restrict__ out_q,
    scalar_t *__restrict__ out_k, scalar_t *__restrict__ out_v, int T, int NH,
    int NKVH, int Da, int r)
{
    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int d_idx = static_cast<int>(threadIdx.x);

    int D = Da * r;

    int TOTAL_H = NH + 2 * NKVH;
    bool is_q = h_idx < NH;
    bool is_k = h_idx >= NH && h_idx < NH + NKVH;

    const scalar_t *in_ptr;
    scalar_t *out_ptr;
    int h_local;
    int h_comp_total;

    if (is_q)
    {
        h_local = h_idx;
        h_comp_total = NH;
        int64_t in_base = static_cast<int64_t>(b_idx) * T * NH * D +
                          static_cast<int64_t>(h_local) * D;
        int64_t out_base = static_cast<int64_t>(b_idx) * NH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        in_ptr = q + in_base + static_cast<int64_t>(d_idx);
        out_ptr = out_q + out_base;
    }
    else if (is_k)
    {
        h_local = h_idx - NH;
        h_comp_total = NKVH;
        int64_t in_base = static_cast<int64_t>(b_idx) * T * NKVH * D +
                          static_cast<int64_t>(h_local) * D;
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        in_ptr = k + in_base + static_cast<int64_t>(d_idx);
        out_ptr = out_k + out_base;
    }
    else
    {
        h_local = h_idx - NH - NKVH;
        h_comp_total = NKVH;
        int64_t in_base = static_cast<int64_t>(b_idx) * T * NKVH * D +
                          static_cast<int64_t>(h_local) * D;
        int64_t out_base = static_cast<int64_t>(b_idx) * NKVH * T * D +
                           static_cast<int64_t>(h_local) * T * D +
                           static_cast<int64_t>(d_idx);
        in_ptr = v + in_base + static_cast<int64_t>(d_idx);
        out_ptr = out_v + out_base;
    }

    int g_idx = d_idx / r;
    int64_t logits_base = static_cast<int64_t>(b_idx) * T * TOTAL_H * Da +
                          static_cast<int64_t>(h_idx) * Da +
                          static_cast<int64_t>(g_idx);
    int64_t logits_stride_t = static_cast<int64_t>(TOTAL_H) * Da;
    int64_t inits_base = static_cast<int64_t>(b_idx) * TOTAL_H * D +
                         static_cast<int64_t>(h_idx) * D;

    float h_state =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    for (int t = 0; t < T; ++t)
    {
        float u_t =
            static_cast<float>(in_ptr[static_cast<int64_t>(t) * h_comp_total * D]);
        float l_t = static_cast<float>(
            logits[logits_base + static_cast<int64_t>(t) * logits_stride_t]);

        float alpha_t = sigmoid_f32(l_t);
        h_state = (1.0f - alpha_t) * h_state + alpha_t * u_t;
        out_ptr[static_cast<int64_t>(t) * D] = static_cast<scalar_t>(h_state);
    }
}

// ----------------------------------------------------------------------------
// Backward Kernel
// ----------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ void ltv_fused_scan_backward_device(
    const scalar_t *__restrict__ q, const scalar_t *__restrict__ k,
    const scalar_t *__restrict__ v, const scalar_t *__restrict__ logits,
    const scalar_t *__restrict__ inits, const scalar_t *__restrict__ out_q,
    const scalar_t *__restrict__ out_k, const scalar_t *__restrict__ out_v,
    const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v, scalar_t *__restrict__ grad_q,
    scalar_t *__restrict__ grad_k, scalar_t *__restrict__ grad_v,
    scalar_t *__restrict__ grad_logits, scalar_t *__restrict__ grad_inits,
    int T, int NH, int NKVH, int Da, int r, int lcm_r_32,
    float *smem_stage)
{
    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int d_idx = static_cast<int>(threadIdx.x);
    int D = Da * r;

    int lane_idx = d_idx % 32;
    int d_idx_over_32 = d_idx / 32;
    int warp_base_d = d_idx_over_32 * 32;
    int da_idx = d_idx / r;
    int da_idx_start = da_idx * r;
    bool is_group_leader = lane_idx == 0 || d_idx == da_idx_start;
    int leader_index = d_idx_over_32 + da_idx - d_idx / lcm_r_32;
    int start_leader = compute_leader_index(da_idx_start, r, 32, lcm_r_32);
    int end_leader =
        compute_leader_index(da_idx_start + r - 1, r, 32, lcm_r_32) + 1;

    int TOTAL_H = NH + 2 * NKVH;
    bool is_q = h_idx < NH;
    bool is_k = (h_idx >= NH) && (h_idx < NH + NKVH);

    int h_local = is_q ? h_idx : (is_k ? h_idx - NH : h_idx - NH - NKVH);
    int h_comp_total = is_q ? NH : NKVH;

    int64_t u_base = static_cast<int64_t>(b_idx) * T * h_comp_total * D +
                     static_cast<int64_t>(h_local) * D +
                     static_cast<int64_t>(d_idx);
    int64_t h_base = is_q ? (static_cast<int64_t>(b_idx) * NH * T * D +
                             static_cast<int64_t>(h_local) * T * D +
                             static_cast<int64_t>(d_idx))
                          : (static_cast<int64_t>(b_idx) * NKVH * T * D +
                             static_cast<int64_t>(h_local) * T * D +
                             static_cast<int64_t>(d_idx));

    const scalar_t *u_ptr = (is_q ? q : (is_k ? k : v)) + u_base;
    const scalar_t *fwd_ptr = (is_q ? out_q : (is_k ? out_k : out_v)) + h_base;
    const scalar_t *dout_ptr =
        (is_q ? grad_out_q : (is_k ? grad_out_k : grad_out_v)) + h_base;
    scalar_t *du_ptr = (is_q ? grad_q : (is_k ? grad_k : grad_v)) + u_base;

    int64_t logits_base = static_cast<int64_t>(b_idx) * T * TOTAL_H * Da +
                          static_cast<int64_t>(h_idx) * Da +
                          static_cast<int64_t>(da_idx);
    int64_t inits_base = static_cast<int64_t>(b_idx) * TOTAL_H * D +
                         static_cast<int64_t>(h_idx) * D;
    int64_t u_stride = static_cast<int64_t>(h_comp_total) * D;
    int64_t fwd_stride = static_cast<int64_t>(D);
    int64_t logits_stride_t = static_cast<int64_t>(TOTAL_H) * Da;

    float dh_next = 0.0f;
    float init_val =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    int active_in_warp = min(32, D - warp_base_d);
    unsigned int active_mask =
        (active_in_warp == 32) ? 0xFFFFFFFFU : (1U << active_in_warp) - 1U;
    int next_group_start_d = (da_idx + 1) * r;
    int max_valid_lane = min(active_in_warp, next_group_start_d - warp_base_d);

    for (int t = T - 1; t >= 0; --t)
    {
        float l_t = static_cast<float>(
            logits[logits_base + static_cast<int64_t>(t) * logits_stride_t]);
        float alpha_t = sigmoid_f32(l_t);
        float h_prev =
            (t > 0) ? static_cast<float>(
                          fwd_ptr[static_cast<int64_t>(t - 1) * fwd_stride])
                    : init_val;

        float dh_t =
            static_cast<float>(dout_ptr[static_cast<int64_t>(t) * fwd_stride]) +
            dh_next;
        float u_t =
            static_cast<float>(u_ptr[static_cast<int64_t>(t) * u_stride]);

        du_ptr[static_cast<int64_t>(t) * u_stride] =
            static_cast<scalar_t>(dh_t * alpha_t);
        float d_logit = dh_t * (u_t - h_prev) * alpha_t * (1.0f - alpha_t);
        dh_next = dh_t * (1.0f - alpha_t);

        float d_logit_sum = reduce_logit_grad_group(
            d_logit, active_mask, lane_idx, max_valid_lane);

        if (is_group_leader)
        {
            smem_stage[leader_index] = d_logit_sum;
        }
        __syncthreads();

        if (d_idx == da_idx_start)
        {
            float group_sum = 0.0f;
            for (int i = start_leader; i < end_leader; ++i)
            {
                group_sum += smem_stage[i];
            }

            scalar_t *out_ptr_logits =
                grad_logits + logits_base + static_cast<int64_t>(t) * logits_stride_t;
            *out_ptr_logits = static_cast<scalar_t>(group_sum);
        }
    }

    grad_inits[inits_base + static_cast<int64_t>(d_idx)] =
        static_cast<scalar_t>(dh_next);
}

template <typename scalar_t>
__global__ void ltv_fused_scan_backward_kernel(
    const scalar_t *__restrict__ q, const scalar_t *__restrict__ k,
    const scalar_t *__restrict__ v, const scalar_t *__restrict__ logits,
    const scalar_t *__restrict__ inits, const scalar_t *__restrict__ out_q,
    const scalar_t *__restrict__ out_k, const scalar_t *__restrict__ out_v,
    const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v, scalar_t *__restrict__ grad_q,
    scalar_t *__restrict__ grad_k, scalar_t *__restrict__ grad_v,
    scalar_t *__restrict__ grad_logits, scalar_t *__restrict__ grad_inits,
    int T, int NH, int NKVH, int Da, int r, int lcm_r_32)
{
    extern __shared__ float smem_stage[];
    ltv_fused_scan_backward_device(
        q, k, v, logits, inits, out_q, out_k, out_v, grad_out_q, grad_out_k,
        grad_out_v, grad_q, grad_k, grad_v, grad_logits, grad_inits, T, NH, NKVH,
        Da, r, lcm_r_32, smem_stage);
}

// ----------------------------------------------------------------------------
// Specialized Kernels (compile-time constants)
// ----------------------------------------------------------------------------

template <typename scalar_t, int T, int NH, int NKVH, int Da, int r>
__global__ void ltv_fused_scan_forward_kernel_const(
    const scalar_t *__restrict__ q, const scalar_t *__restrict__ k,
    const scalar_t *__restrict__ v, const scalar_t *__restrict__ logits,
    const scalar_t *__restrict__ inits, scalar_t *__restrict__ out_q,
    scalar_t *__restrict__ out_k, scalar_t *__restrict__ out_v)
{
    constexpr int D = Da * r;
    constexpr int TOTAL_H = NH + 2 * NKVH;

    int h_idx = static_cast<int>(blockIdx.x);
    int b_idx = static_cast<int>(blockIdx.y);
    int d_idx = static_cast<int>(threadIdx.x);

    bool is_q = h_idx < NH;
    bool is_k = h_idx >= NH && h_idx < NH + NKVH;

    int h_local = is_q ? h_idx : (is_k ? h_idx - NH : h_idx - NH - NKVH);
    int h_comp_total = is_q ? NH : NKVH;

    int64_t in_base = static_cast<int64_t>(b_idx) * T * h_comp_total * D +
                      static_cast<int64_t>(h_local) * D +
                      static_cast<int64_t>(d_idx);
    int64_t out_base = is_q ? (static_cast<int64_t>(b_idx) * NH * T * D +
                               static_cast<int64_t>(h_local) * T * D +
                               static_cast<int64_t>(d_idx))
                            : (static_cast<int64_t>(b_idx) * NKVH * T * D +
                               static_cast<int64_t>(h_local) * T * D +
                               static_cast<int64_t>(d_idx));

    const scalar_t *curr_in = (is_q ? q : (is_k ? k : v)) + in_base;
    scalar_t *curr_out = (is_q ? out_q : (is_k ? out_k : out_v)) + out_base;

    int g_idx = d_idx / r;
    int64_t logits_base = static_cast<int64_t>(b_idx) * T * TOTAL_H * Da +
                          static_cast<int64_t>(h_idx) * Da +
                          static_cast<int64_t>(g_idx);
    int64_t logits_stride_t = static_cast<int64_t>(TOTAL_H) * Da;
    const scalar_t *curr_logits = logits + logits_base;

    int64_t inits_base = static_cast<int64_t>(b_idx) * TOTAL_H * D +
                         static_cast<int64_t>(h_idx) * D;
    float h_state =
        static_cast<float>(inits[inits_base + static_cast<int64_t>(d_idx)]);

    int64_t in_stride = static_cast<int64_t>(h_comp_total) * D;
    int64_t out_stride = static_cast<int64_t>(D);

    for (int t = 0; t < T; ++t)
    {
        float u_t = static_cast<float>(*curr_in);
        float l_t = static_cast<float>(*curr_logits);

        float alpha_t = sigmoid_f32(l_t);
        h_state = (1.0f - alpha_t) * h_state + alpha_t * u_t;

        *curr_out = static_cast<scalar_t>(h_state);

        curr_in += in_stride;
        curr_out += out_stride;
        curr_logits += logits_stride_t;
    }
}

template <typename scalar_t, int T, int NH, int NKVH, int Da, int r,
          int lcm_r_32>
__global__ void ltv_fused_scan_backward_kernel_const(
    const scalar_t *__restrict__ q, const scalar_t *__restrict__ k,
    const scalar_t *__restrict__ v, const scalar_t *__restrict__ logits,
    const scalar_t *__restrict__ inits, const scalar_t *__restrict__ out_q,
    const scalar_t *__restrict__ out_k, const scalar_t *__restrict__ out_v,
    const scalar_t *__restrict__ grad_out_q,
    const scalar_t *__restrict__ grad_out_k,
    const scalar_t *__restrict__ grad_out_v, scalar_t *__restrict__ grad_q,
    scalar_t *__restrict__ grad_k, scalar_t *__restrict__ grad_v,
    scalar_t *__restrict__ grad_logits, scalar_t *__restrict__ grad_inits)
{
    __shared__ float
        smem_stage[compute_leader_index(Da * r - 1, r, 32, lcm_r_32) + 1];
    ltv_fused_scan_backward_device(
        q, k, v, logits, inits, out_q, out_k, out_v, grad_out_q, grad_out_k,
        grad_out_v, grad_q, grad_k, grad_v, grad_logits, grad_inits, T, NH, NKVH,
        Da, r, lcm_r_32, smem_stage);
}

// ----------------------------------------------------------------------------
// Launcher Logic
// ----------------------------------------------------------------------------

constexpr int kT = 1024;
constexpr int kNH = 8;
constexpr int kNKVH = 8;
constexpr int kDa = 1;
constexpr int kR = 128;

static inline void check_valid_tensor(const torch::Tensor &t, at::ScalarType dtype,
                                      const char *name)
{
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name,
                " must match dtype of q (expected ", toString(dtype), ", got ",
                toString(t.scalar_type()), ")");
}

void ltv_fused_scan_cuda_forward(torch::Tensor q, torch::Tensor k,
                                 torch::Tensor v, torch::Tensor logits,
                                 torch::Tensor inits, torch::Tensor out_q,
                                 torch::Tensor out_k, torch::Tensor out_v,
                                 int B, int T, int NH, int NKVH, int Da,
                                 int r)
{
    auto dtype = q.scalar_type();

    check_valid_tensor(q, dtype, "q");
    check_valid_tensor(k, dtype, "k");
    check_valid_tensor(v, dtype, "v");
    check_valid_tensor(logits, dtype, "logits");
    check_valid_tensor(inits, dtype, "inits");

    check_valid_tensor(out_q, dtype, "out_q");
    check_valid_tensor(out_k, dtype, "out_k");
    check_valid_tensor(out_v, dtype, "out_v");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
        constexpr int kD = kDa * kR;
        constexpr int kTOTAL_H = kNH + 2 * kNKVH;

        dim3 grid(kTOTAL_H, B);
        dim3 block(kD);

        AT_DISPATCH_SWITCH(
            q.scalar_type(), "ltv_forward_const",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
          using scalar_t = float;
          ltv_fused_scan_forward_kernel_const<scalar_t, kT, kNH, kNKVH, kDa, kR>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                  inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                  out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>()); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                                   {
          using scalar_t = at::BFloat16;
          ltv_fused_scan_forward_kernel_const<scalar_t, kT, kNH, kNKVH, kDa, kR>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                  inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                  out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>()); }));
        return;
    }

    TORCH_WARN("ltv_fused_scan_cuda_forward",
               "Expected: T=", kT, ", NH=", kNH, ", NKVH=", kNKVH, ", Da=", kDa, ", r=", kR,
               "Got: T=", T, ", NH=", NH, ", NKVH=", NKVH, ", Da=", Da, ", r=", r,
               ". Falling back to generic kernel.");

    int TOTAL_H = NH + 2 * NKVH;
    dim3 grid(TOTAL_H, B);

    int threads = Da * r;
    TORCH_CHECK(threads <= 1024,
                "Da * r exceeds maximum block size of 1024 threads.");
    dim3 block(threads);

    AT_DISPATCH_SWITCH(
        q.scalar_type(), "ltv_forward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_scan_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
            inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
            out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(), T, NH, NKVH,
            Da, r); }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                               {
        using scalar_t = at::BFloat16;
        ltv_fused_scan_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
            inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
            out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(), T, NH, NKVH,
            Da, r); }));
}

void ltv_fused_scan_cuda_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor logits,
    torch::Tensor inits, torch::Tensor out_q, torch::Tensor out_k,
    torch::Tensor out_v, torch::Tensor grad_out_q, torch::Tensor grad_out_k,
    torch::Tensor grad_out_v, torch::Tensor grad_q, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_logits, torch::Tensor grad_inits,
    int B, int T, int NH, int NKVH, int Da, int r)
{

    auto dtype = q.scalar_type();

    check_valid_tensor(q, dtype, "q");
    check_valid_tensor(k, dtype, "k");
    check_valid_tensor(v, dtype, "v");
    check_valid_tensor(logits, dtype, "logits");
    check_valid_tensor(inits, dtype, "inits");

    check_valid_tensor(out_q, dtype, "out_q");
    check_valid_tensor(out_k, dtype, "out_k");
    check_valid_tensor(out_v, dtype, "out_v");

    check_valid_tensor(grad_out_q, dtype, "grad_out_q");
    check_valid_tensor(grad_out_k, dtype, "grad_out_k");
    check_valid_tensor(grad_out_v, dtype, "grad_out_v");

    check_valid_tensor(grad_q, dtype, "grad_q");
    check_valid_tensor(grad_k, dtype, "grad_k");
    check_valid_tensor(grad_v, dtype, "grad_v");

    check_valid_tensor(grad_logits, dtype, "grad_logits");
    check_valid_tensor(grad_inits, dtype, "grad_inits");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (T == kT && NH == kNH && NKVH == kNKVH && Da == kDa && r == kR)
    {
        constexpr int kD = kDa * kR;
        constexpr int kTOTAL_H = kNH + 2 * kNKVH;
        constexpr int kLcmR32 = std::lcm(kR, 32);

        dim3 grid(kTOTAL_H, B);
        dim3 block(kD);

        AT_DISPATCH_SWITCH(
            q.scalar_type(), "ltv_backward_const",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                             {
          using scalar_t = float;
          ltv_fused_scan_backward_kernel_const<scalar_t, kT, kNH, kNKVH, kDa,
                                               kR, kLcmR32>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                  inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                  out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                  grad_out_q.data_ptr<scalar_t>(),
                  grad_out_k.data_ptr<scalar_t>(),
                  grad_out_v.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
                  grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
                  grad_logits.data_ptr<scalar_t>(),
                  grad_inits.data_ptr<scalar_t>()); })
                AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                 {
          using scalar_t = at::BFloat16;
          ltv_fused_scan_backward_kernel_const<scalar_t, kT, kNH, kNKVH, kDa,
                                               kR, kLcmR32>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                  inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                  out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                  grad_out_q.data_ptr<scalar_t>(),
                  grad_out_k.data_ptr<scalar_t>(),
                  grad_out_v.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
                  grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
                  grad_logits.data_ptr<scalar_t>(),
                  grad_inits.data_ptr<scalar_t>()); }));
        return;
    }

    TORCH_WARN("ltv_fused_scan_cuda_backward",
               "Expected: T=", kT, ", NH=", kNH, ", NKVH=", kNKVH, ", Da=", kDa, ", r=", kR,
               "Got: T=", T, ", NH=", NH, ", NKVH=", NKVH, ", Da=", Da, ", r=", r,
               ". Falling back to generic kernel.");

    int TOTAL_H = NH + 2 * NKVH;
    dim3 grid(TOTAL_H, B);

    int threads = Da * r;
    TORCH_CHECK(threads <= 1024,
                "Da * r exceeds maximum block size of 1024 threads.");
    dim3 block(threads);

    int lcm_r_32 = std::lcm(r, 32);
    int num_leaders = compute_leader_index(Da * r - 1, r, 32, lcm_r_32) + 1;
    int smem_size = static_cast<int>(num_leaders * sizeof(float));

    AT_DISPATCH_SWITCH(
        q.scalar_type(), "ltv_backward",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                         {
        using scalar_t = float;
        ltv_fused_scan_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
                grad_logits.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), T, NH, NKVH, Da, r,
                lcm_r_32); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                             {
        using scalar_t = at::BFloat16;
        ltv_fused_scan_backward_kernel<scalar_t>
            <<<grid, block, smem_size, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
                inits.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                grad_out_q.data_ptr<scalar_t>(),
                grad_out_k.data_ptr<scalar_t>(),
                grad_out_v.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
                grad_logits.data_ptr<scalar_t>(),
                grad_inits.data_ptr<scalar_t>(), T, NH, NKVH, Da, r,
                lcm_r_32); }));
}
