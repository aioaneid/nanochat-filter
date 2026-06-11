import torch
try:
    import rust_ewma
except ImportError:
    rust_ewma = None
import logging
from torch.autograd.function import once_differentiable

from nanochat.ltv.ltv_metal_core import (
    core_metal_forward_time,
    core_metal_forward_spatial,
    core_scan_backward_time,
    core_scan_backward_spatial,
)

# Needed to avoid AttributeError
import nanochat.ops as _nanochat_ops

_ = _nanochat_ops

logger = logging.getLogger(__name__)


def scan_forward_time(ctx, initial_value, alpha, x, rust_fn):
    h_out_user_t, tensors_for_backward = core_metal_forward_time(
        initial_value, alpha, x, rust_fn
    )
    ctx.dtypes = (initial_value.dtype, alpha.dtype, x.dtype)
    ctx.save_for_backward(*tensors_for_backward)
    return h_out_user_t


def scan_backward_time(ctx, grad_h_x, rust_fn):
    h_init_mps, alpha_mps, h_out_user = ctx.saved_tensors
    grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_time(
        h_init_mps, alpha_mps, h_out_user, grad_h_x, rust_fn
    )
    dtype_init, dtype_alpha, dtype_x = ctx.dtypes
    return (
        grad_init_mps.to(dtype=dtype_init),
        grad_alpha_mps_t.to(dtype=dtype_alpha),
        grad_x_mps_t.to(dtype=dtype_x),
    )


def scan_forward_spatial(ctx, initial_value, alpha, x, rust_fn):
    h_out_user_t, tensors_for_backward = core_metal_forward_spatial(
        initial_value, alpha, x, rust_fn
    )
    ctx.dtypes = (initial_value.dtype, alpha.dtype, x.dtype)
    ctx.save_for_backward(*tensors_for_backward)
    return h_out_user_t


def scan_backward_spatial(ctx, grad_h_x, rust_fn):
    h_init_mps, alpha_mps, h_out_user = ctx.saved_tensors
    grad_init_mps, grad_alpha_mps_t, grad_x_mps_t = core_scan_backward_spatial(
        h_init_mps, alpha_mps, h_out_user, grad_h_x, rust_fn
    )
    dtype_init, dtype_alpha, dtype_x = ctx.dtypes
    return (
        grad_init_mps.to(dtype=dtype_init),
        grad_alpha_mps_t.to(dtype=dtype_alpha),
        grad_x_mps_t.to(dtype=dtype_x),
    )


def blelloch_forward_spatial(ctx, initial_value, alpha, x, schedule, rust_fn):
    ret = scan_forward_spatial(
        ctx, initial_value, alpha, x, lambda *args: rust_fn(*args, schedule)
    )
    ctx.schedule = schedule
    return ret


def blelloch_backward_spatial(ctx, grad_h_x, rust_fn):
    schedule = ctx.schedule
    return scan_backward_spatial(
        ctx, grad_h_x, lambda *args: rust_fn(*args, schedule)
    ) + (None,)


class LtvScanMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x) -> torch.Tensor:
        return scan_forward_time(ctx, initial_value, alpha, x, rust_ewma.ltv_scan_metal)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h_x):
        return scan_backward_time(ctx, grad_h_x, rust_ewma.ltv_scan_metal)


class LtvNoopMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x) -> torch.Tensor:
        return scan_forward_time(ctx, initial_value, alpha, x, rust_ewma.ltv_noop_metal)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h_x):
        return scan_backward_time(ctx, grad_h_x, rust_ewma.ltv_noop_metal)


class LtvSharedMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x, schedule) -> torch.Tensor:
        return blelloch_forward_spatial(
            ctx, initial_value, alpha, x, schedule, rust_ewma.ltv_shared_metal
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h_x):
        return blelloch_backward_spatial(ctx, grad_h_x, rust_ewma.ltv_shared_metal)


class LtvPlaneMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x, schedule) -> torch.Tensor:
        return blelloch_forward_spatial(
            ctx, initial_value, alpha, x, schedule, rust_ewma.ltv_plane_metal
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h_x):
        return blelloch_backward_spatial(ctx, grad_h_x, rust_ewma.ltv_plane_metal)


# Wrappers


def ltv_noop_metal(initial_value, alpha, x):
    # return NoopMetal.apply(initial_value, alpha, x)
    return torch.ops.nanochat.ltv_noop_metal_op(initial_value, alpha, x)


def ltv_shared_max_metal(initial_value, alpha, x):
    # return LtvSharedMetal.apply(initial_value, alpha, x, "max")
    return torch.ops.nanochat.ltv_shared_metal_op(initial_value, alpha, x, "max")


def ltv_shared_pmax_metal(initial_value, alpha, x):
    # return LtvSharedMetal.apply(initial_value, alpha, x, "pmax")
    return torch.ops.nanochat.ltv_shared_metal_op(initial_value, alpha, x, "pmax")


def ltv_shared_log_metal(initial_value, alpha, x):
    # return LtvSharedMetal.apply(initial_value, alpha, x, "log")
    return torch.ops.nanochat.ltv_shared_metal_op(initial_value, alpha, x, "log")


def ltv_plane_max_metal(initial_value, alpha, x):
    # return LtvPlaneMetal.apply(initial_value, alpha, x, "max")
    return torch.ops.nanochat.ltv_plane_metal_op(initial_value, alpha, x, "max")


def ltv_plane_log_metal(initial_value, alpha, x):
    # return LtvPlaneMetal.apply(initial_value, alpha, x, "log")
    return torch.ops.nanochat.ltv_plane_metal_op(initial_value, alpha, x, "log")

def ltv_scan_metal(initial_value, alpha, x):
    # return LtvScanMetal.apply(initial_value, alpha, x)
    return torch.ops.nanochat.ltv_scan_metal_op(initial_value, alpha, x)
