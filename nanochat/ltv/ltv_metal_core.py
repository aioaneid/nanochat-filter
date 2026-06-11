import torch
import logging

from nanochat.ltv.ltv_utils import dimensions

logger = logging.getLogger(__name__)


# --- Time-Major Helpers (Scan / Noop) ---


def _metal_preprocessing_time(initial_value, alpha, x):
    """
    Converts to float32, transposes to (T, B, D), and ensures contiguity.
    """
    h_init = initial_value.detach()
    alpha = alpha.detach()
    x = x.detach()

    # (B, T, D) -> (T, B, D)
    alpha = alpha.transpose(0, 1).contiguous()
    x = x.transpose(0, 1).contiguous()

    return h_init, alpha, x


def _metal_postprocessing_time(h_init_mps, h_out_alpha_mps, h_out_x_mps, r):
    # h_init: (B, D)
    # h_out_alpha: (T, B, Da)
    # h_out_x: (T, B, Dx)
    return h_out_x_mps + (
        h_init_mps.unsqueeze(0)
        * h_out_alpha_mps.unsqueeze(-1).expand(-1, -1, -1, r).flatten(start_dim=2)
    )


def _metal_backward_preprocessing_time(grad_h, alpha_mps, h_out_user, h_init_mps):
    # h_out_user: (T, B, D)
    h_prev_mps = torch.empty_like(h_out_user, memory_format=torch.contiguous_format)
    h_prev_mps[0, :, :] = h_init_mps
    h_prev_mps[1:, :, :] = h_out_user[:-1, :, :]

    # alpha_mps: (T, B, Da)
    alpha_rev_mps = torch.empty_like(alpha_mps, memory_format=torch.contiguous_format)
    alpha_rev_mps[0, :].fill_(0.0)
    alpha_rev_mps[1:, :] = alpha_mps.flip(0)[:-1, :]

    # grad_h: (B, T, D) -> (T, B, D)
    grad_h_rev_mps = grad_h.detach().flip(1).transpose(0, 1).contiguous()

    return alpha_rev_mps, grad_h_rev_mps, h_prev_mps


def _metal_backward_postprocessing_time(g_out_rev_mps, h_prev_mps, alpha_mps, dims):
    B, T, Da, r = dims
    D = Da * r

    # g_out_rev_mps: (T, B, D)
    g_mps = g_out_rev_mps.flip(0)
    grad_alpha_expanded_mps = g_mps * h_prev_mps
    grad_alpha_mps = grad_alpha_expanded_mps.view(T, B, Da, r).sum(dim=-1)

    alpha_init_exp_mps = alpha_mps[0, :, :].unsqueeze(-1).expand(B, Da, r).reshape(B, D)
    grad_init_mps = g_mps[0, :, :] * alpha_init_exp_mps
    return grad_init_mps, grad_alpha_mps, g_mps


# --- Spatial-Major Helpers (Plane / Shared) ---


def _metal_preprocessing_spatial(initial_value, alpha, x):
    h_init = initial_value.detach()
    alpha = alpha.detach()
    x = x.detach()

    # (B, T, D) -> (B, D, T)
    alpha = alpha.transpose(1, 2).contiguous()
    x = x.transpose(1, 2).contiguous()

    return h_init, alpha, x


def _metal_postprocessing_spatial(h_init_mps, h_out_alpha_mps, h_out_x_mps, r):
    # h_init: (B, D)
    # h_out_alpha: (B, Da, T)
    # h_out_x: (B, Dx, T)

    # Expand h_init for broadcasting: (B, D, 1)
    # Expand alpha: (B, Da, 1, T) -> expand r -> flatten -> (B, D, T)

    alpha_expanded = h_out_alpha_mps.unsqueeze(2).expand(-1, -1, r, -1).flatten(1, 2)

    return h_out_x_mps + h_init_mps.unsqueeze(-1) * alpha_expanded


def _metal_backward_preprocessing_spatial(grad_h, alpha_mps, h_out_user, h_init_mps):
    # h_out_user: (B, D, T)
    # alpha_mps: (B, Da, T)
    # grad_h: (B, T, D)

    h_prev_mps = torch.empty_like(h_out_user, memory_format=torch.contiguous_format)
    # Time dim is 2
    h_prev_mps[:, :, 0] = h_init_mps
    h_prev_mps[:, :, 1:] = h_out_user[:, :, :-1]

    alpha_rev_mps = torch.empty_like(alpha_mps, memory_format=torch.contiguous_format)
    alpha_rev_mps[:, :, 0].fill_(0.0)
    alpha_rev_mps[:, :, 1:] = alpha_mps.flip(2)[:, :, :-1]

    # grad_h: (B, T, D) -> (B, D, T)
    grad_h_rev_mps = grad_h.detach().flip(1).transpose(1, 2).contiguous()

    return alpha_rev_mps, grad_h_rev_mps, h_prev_mps


def _metal_backward_postprocessing_spatial(g_out_rev_mps, h_prev_mps, alpha_mps, dims):
    B, T, Da, r = dims
    D = Da * r

    # g_out_rev_mps: (B, D, T)
    g_mps = g_out_rev_mps.flip(2)
    grad_alpha_expanded_mps = g_mps * h_prev_mps

    # Sum over r: view (B, Da, r, T) -> sum(2) -> (B, Da, T)
    grad_alpha_mps = grad_alpha_expanded_mps.view(B, Da, r, T).sum(dim=2)

    # Gradient for init: Time=0
    # alpha_mps[:, :, 0] -> (B, Da). Expand to (B, D).
    alpha_init_exp_mps = alpha_mps[:, :, 0].unsqueeze(2).expand(B, Da, r).reshape(B, D)
    grad_init_mps = g_mps[:, :, 0] * alpha_init_exp_mps

    # logger.info(f"[backward post] grad_init_mps shape={grad_init_mps.shape} strides={grad_init_mps.stride()}")
    # logger.info(f"[backward post] grad_alpha_mps shape={grad_alpha_mps.shape} strides={grad_alpha_mps.stride()}")
    # logger.info(f"[backward post] g_mps shape={g_mps.shape} strides={g_mps.stride()}")

    return grad_init_mps, grad_alpha_mps, g_mps


# --- Core Forward/Backward Entry Points ---


def core_metal_forward_time(initial_value, alpha, x, rust_fn):
    """Used by Scan and Noop"""
    B, T, D_a, r = dimensions(alpha, initial_value, x)

    h_init_mps, alpha_mps, x_mps = _metal_preprocessing_time(initial_value, alpha, x)

    h_out_alpha_mps = torch.empty_like(alpha_mps, memory_format=torch.contiguous_format)
    h_out_x_mps = torch.empty_like(x_mps, memory_format=torch.contiguous_format)

    parallel_dim_a = B * D_a
    rust_fn(
        alpha_mps,
        x_mps,
        h_out_alpha_mps,
        h_out_x_mps,
        T,
        parallel_dim_a,
        r,
    )

    h_out_user = _metal_postprocessing_time(h_init_mps, h_out_alpha_mps, h_out_x_mps, r)

    # Return (B, T, D), ctx_tensors
    return h_out_user.transpose(0, 1), (h_init_mps, alpha_mps, h_out_user)


def core_metal_forward_spatial(initial_value, alpha, x, rust_fn):
    """Used by Plane and Shared"""
    B, T, D_a, r = dimensions(alpha, initial_value, x)

    h_init_mps, alpha_mps, x_mps = _metal_preprocessing_spatial(initial_value, alpha, x)

    h_out_alpha_mps = torch.empty_like(alpha_mps, memory_format=torch.contiguous_format)
    h_out_x_mps = torch.empty_like(x_mps, memory_format=torch.contiguous_format)

    parallel_dim_a = B * D_a

    rust_fn(
        alpha_mps,
        x_mps,
        h_out_alpha_mps,
        h_out_x_mps,
        T,
        parallel_dim_a,
        r,
    )

    h_out_user = _metal_postprocessing_spatial(
        h_init_mps, h_out_alpha_mps, h_out_x_mps, r
    )

    # logger.info(
    #     f"[forward spatial] h_out_user shape={h_out_user.shape} strides={h_out_user.stride()}"
    # )

    h_out_user_t = h_out_user.transpose(1, 2)

    # logger.info(
    #     f"[forward spatial] h_out_user_t shape={h_out_user_t.shape} strides={h_out_user_t.stride()}"
    # )

    # Return (B, T, D), ctx_tensors
    # h_out_user is (B, D, T). Transpose to (B, T, D).
    return h_out_user_t, (h_init_mps, alpha_mps, h_out_user)


def core_scan_backward_time(h_init_mps, alpha_mps, h_out_user, grad_h_x, rust_fn):
    """Used by Scan and Noop"""
    T, B, D_a = alpha_mps.shape
    r = h_out_user.shape[-1] // D_a
    grad_h_x_float = grad_h_x

    (
        alpha_rev_mps,
        grad_h_rev_mps,
        h_prev_mps,
    ) = _metal_backward_preprocessing_time(
        grad_h_x_float, alpha_mps, h_out_user, h_init_mps
    )

    g_out_alpha_rev_mps = torch.empty_like(
        alpha_rev_mps, memory_format=torch.contiguous_format
    )
    g_out_x_rev_mps = torch.empty_like(
        grad_h_rev_mps, memory_format=torch.contiguous_format
    )
    parallel_a = B * D_a

    rust_fn(
        alpha_rev_mps,
        grad_h_rev_mps,
        g_out_alpha_rev_mps,
        g_out_x_rev_mps,
        T,
        parallel_a,
        r,
    )

    grad_init_mps, grad_alpha_mps, grad_x_mps = _metal_backward_postprocessing_time(
        g_out_x_rev_mps, h_prev_mps, alpha_mps, dims=(B, T, D_a, r)
    )

    # Return (B, T, D) shaped gradients
    return (
        grad_init_mps,
        grad_alpha_mps.transpose(0, 1),
        grad_x_mps.transpose(0, 1),
    )


def core_scan_backward_spatial(h_init_mps, alpha_mps, h_out_user, grad_h_x, rust_fn):
    """Used by Plane and Shared"""
    # alpha_mps: (B, Da, T)
    B, D_a, T = alpha_mps.shape
    r = h_out_user.shape[1] // D_a
    grad_h_x_float = grad_h_x

    (
        alpha_rev_mps,
        grad_h_rev_mps,
        h_prev_mps,
    ) = _metal_backward_preprocessing_spatial(
        grad_h_x_float, alpha_mps, h_out_user, h_init_mps
    )

    g_out_alpha_rev_mps = torch.empty_like(
        alpha_rev_mps, memory_format=torch.contiguous_format
    )
    g_out_x_rev_mps = torch.empty_like(
        grad_h_rev_mps, memory_format=torch.contiguous_format
    )
    parallel_a = B * D_a

    rust_fn(
        alpha_rev_mps,
        grad_h_rev_mps,
        g_out_alpha_rev_mps,
        g_out_x_rev_mps,
        T,
        parallel_a,
        r,
    )

    grad_init_mps, grad_alpha_mps, grad_x_mps = _metal_backward_postprocessing_spatial(
        g_out_x_rev_mps, h_prev_mps, alpha_mps, dims=(B, T, D_a, r)
    )

    # Return (B, T, D) shaped gradients
    return (
        grad_init_mps,
        grad_alpha_mps.transpose(1, 2),
        grad_x_mps.transpose(1, 2),
    )
