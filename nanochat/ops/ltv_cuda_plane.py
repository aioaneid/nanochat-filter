import torch


try:
    import cpp_ewma_cuda
except ImportError:
    cpp_ewma_cuda = None


@torch.library.impl("nanochat_cuda::ltv_plane_cuda_kernel", "CUDA")
def ltv_plane_cuda_kernel_impl(
    alpha,
    x,
    out_alpha,
    out_x,
    t_seq: int,
    parallel_a: int,
    r: int,
    schedule: str,
):
    if cpp_ewma_cuda is None:
        raise RuntimeError("cpp_ewma_cuda extension is not available")

    # 1. Cast inputs to Float32
    alpha_f32 = alpha.float().contiguous()
    x_f32 = x.float().contiguous()
    
    # 2. Create Float32 temporary buffers for outputs 
    # (Since we cannot guarantee out_alpha/out_x are Float32)
    tmp_out_alpha = torch.empty_like(alpha_f32)
    tmp_out_x = torch.empty_like(x_f32)

    if schedule == "max":
        schedule_kind = 0
    elif schedule == "log":
        schedule_kind = 1
    else:
        raise RuntimeError(f"Unknown schedule: {schedule}")

    # 3. Call the kernel with consistent Float32 tensors
    cpp_ewma_cuda.ltv_plane(
        alpha_f32,
        x_f32,
        tmp_out_alpha,
        tmp_out_x,
        int(t_seq),
        int(parallel_a),
        int(r),
        int(schedule_kind),
    )

    # 4. Copy the results back into the original output buffers (handles dtype conversion)
    out_alpha.copy_(tmp_out_alpha)
    out_x.copy_(tmp_out_x)

    return None


def _ltv_plane_cuda_setup(ctx, inputs, output):
    h0, alpha, x, schedule = inputs
    ctx.save_for_backward(h0, alpha, output)
    ctx.dtype_x = x.dtype
    ctx.schedule = schedule


def ltv_plane_cuda_backward(ctx, grad_h_x):
    from nanochat.ltv.ltv_metal_core import core_scan_backward_spatial

    h0, alpha, h_out_user = ctx.saved_tensors

    dtype_h0 = h0.dtype
    dtype_alpha = alpha.dtype
    dtype_x = ctx.dtype_x

    def wrapped_kernel(a, x, oa, ox, ts, pa, r):
        return torch.ops.nanochat_cuda.ltv_plane_cuda_kernel(
            a, x, oa, ox, ts, pa, r, ctx.schedule
        )

    with torch.amp.autocast(device_type=grad_h_x.device.type, enabled=False):
        grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_spatial(
            h0.float(),
            alpha.float().transpose(1, 2),
            h_out_user.float().transpose(1, 2),
            grad_h_x.float(),
            wrapped_kernel,
        )

    return (
        grad_init_mps.to(dtype=dtype_h0),
        grad_alpha_mps_t.to(dtype=dtype_alpha),
        grad_x_mps_t.to(dtype=dtype_x),
        None,
    )


torch.library.register_autograd(
    "nanochat_cuda::ltv_plane_cuda_op",
    ltv_plane_cuda_backward,
    setup_context=_ltv_plane_cuda_setup,
)


@torch.library.impl("nanochat_cuda::ltv_plane_cuda_op", "CUDA")
def ltv_plane_cuda_op_impl(h0, alpha, x, schedule: str):
    from nanochat.ltv.ltv_metal_core import core_metal_forward_spatial

    dtype_x = x.dtype

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        out = core_metal_forward_spatial(
            h0.float(),
            alpha.float(),
            x.float(),
            lambda alpha_mps, x_mps, h_out_alpha_mps, h_out_x_mps, T, parallel_dim_a, r: torch.ops.nanochat_cuda.ltv_plane_cuda_kernel(
                alpha_mps,
                x_mps,
                h_out_alpha_mps,
                h_out_x_mps,
                T,
                parallel_dim_a,
                r,
                schedule,
            ),
        )[0]

    return out.to(dtype=dtype_x)
