#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Helper to keep math in float32
__device__ __forceinline__ float sigmoid_f32(float x)
{
    // Under use_fast_math expf will be replaced with __expf.
    return 1.0f / (1.0f + expf(-x));
}

// Warp reduction (unchanged but shared)
template <typename T_compute>
__device__ __forceinline__ float reduce_logit_grad_group(
    T_compute grad_logit,
    unsigned int active_mask,
    int lane_idx,
    int max_valid_lane)
{
    T_compute sum = grad_logit;

#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        T_compute other = __shfl_down_sync(active_mask, sum, offset);
        if (lane_idx + offset < max_valid_lane)
        {
            sum += other;
        }
    }
    return sum;
}

__host__ __device__ __forceinline__ constexpr int compute_leader_index(
    int d_idx, int a, int b, int lcm_a_b)
{
    return d_idx / a + d_idx / b - d_idx / lcm_a_b;
}
