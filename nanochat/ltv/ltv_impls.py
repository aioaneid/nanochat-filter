import torch
try:
    import rust_ewma
except ImportError:
    rust_ewma = None
import logging
from torch.autograd.function import once_differentiable

from nanochat.ltv.ltv_utils import dimensions

logger = logging.getLogger(__name__)


def _preprocessing_torch(initial_value, alpha, x):
    """Common pre-processing: Optimized for BF16 PCIe transfer."""
    # OPTIMIZED: Move to CPU in BF16/FP16, then cast to float
    h_init = initial_value.detach().cpu()
    alpha = alpha.detach().transpose(0, 1).contiguous().cpu()
    x = x.detach().transpose(0, 1).contiguous().cpu()

    return h_init, alpha, x, initial_value.device


def _postprocessing_torch(h_out, device):
    """Common post-processing: Cast on CPU to save PCIe."""
    h_out_c = h_out.to(device=device)
    return h_out_c.transpose(0, 1)


def _backward_preprocessing_torch(grad_h, alpha, h_out, initial_value):
    """Common backward pre-processing: Optimized for BF16 PCIe transfer."""

    h_prev = torch.cat([initial_value.unsqueeze(1), h_out[:, :-1, :]], dim=1)

    # Reverse Scan preparation
    grad_h_rev = grad_h.flip(1)
    alpha_rev = torch.zeros_like(alpha)
    alpha_rev[:, 1:] = alpha.flip(1)[:, :-1]
    h_init_rev = torch.zeros_like(grad_h[:, 0])

    # OPTIMIZED: Move to CPU before float cast
    h_init_np = h_init_rev.detach().cpu()
    alpha_np = alpha_rev.detach().transpose(0, 1).contiguous().cpu()
    x_np = grad_h_rev.detach().transpose(0, 1).contiguous().cpu()

    return h_init_np, alpha_np, x_np, h_prev, alpha.device


def _backward_postprocessing_torch(g_rev_torch, device, h_prev, alpha):
    """Common backward post-processing."""
    # OPTIMIZED: Cast on CPU
    g_rev_c = g_rev_torch.to(device=device)
    g = g_rev_c.transpose(0, 1).flip(1)

    # grad_alpha reduction
    grad_alpha_expanded = g * h_prev
    B, T, HD = grad_alpha_expanded.shape
    Ha = alpha.shape[2]
    r = HD // Ha if Ha > 0 else 1

    grad_alpha = grad_alpha_expanded.view(B, T, Ha, r).sum(dim=-1)

    # grad_init
    alpha_init_exp = alpha[:, 0, :].unsqueeze(-1).expand(-1, -1, r).reshape(B, HD)
    grad_init = g[:, 0, :] * alpha_init_exp

    return grad_init, grad_alpha, g


@torch.compiler.disable
def ltv_pytorch_loop(
    initial_value: torch.Tensor, alpha: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Linear Time-Varying (LTV) filter output using a plain PyTorch loop.

    h_t = alpha_t * h_{t-1} + x_t

    Args:
        initial_value: (B, D) Initial state h_{-1}
        alpha: (B, T, D_a) Decay/Transition term.
        x: (B, T, D) Input term.

    Returns:
        h: (B, T, D) The sequence of states h_0 ... h_{T-1}
    """
    B, T, D_a, r = dimensions(alpha, initial_value, x)

    # alpha: (B, T, H) -> (B, T, H, 1) -> (B, T, H, r) -> (B, T, H*r)
    alpha = alpha.unsqueeze(-1).expand(-1, -1, -1, r).reshape(B, T, D_a * r)

    h_prev = initial_value
    h_all = []

    for t in range(T):
        # Slice at t
        alpha_t = alpha[:, t]
        x_t = x[:, t]

        h_t = alpha_t * h_prev + x_t
        h_all.append(h_t)
        h_prev = h_t

    return torch.stack(h_all, dim=1)


def py_identity(
    initial_value: torch.Tensor, alpha: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    return x


class LtvRustCpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x):
        h_init_cpu, alpha_cpu, x_cpu, device = _preprocessing_torch(
            initial_value, alpha, x
        )
        h_out = torch.from_numpy(
            rust_ewma.ltv_scan_cpu(
                h_init_cpu.cpu().float().numpy(),
                alpha_cpu.cpu().float().numpy(),
                x_cpu.cpu().float().numpy(),
            )
        ).to(dtype=x.dtype)
        h_out = _postprocessing_torch(h_out, device)
        ctx.save_for_backward(initial_value, alpha, h_out)
        return h_out

    @staticmethod
    def backward(ctx, grad_h):
        initial_value, alpha, h_out = ctx.saved_tensors
        h_init_cpu, alpha_cpu, x_cpu, h_prev, device = (
            _backward_preprocessing_torch(grad_h, alpha, h_out, initial_value)
        )
        g_rev = torch.from_numpy(
            rust_ewma.ltv_scan_cpu(
                h_init_cpu.cpu().float().numpy(),
                alpha_cpu.cpu().float().numpy(),
                x_cpu.cpu().float().numpy(),
            )
        ).to(dtype=h_out.dtype)
        return _backward_postprocessing_torch(g_rev, device, h_prev, alpha)


class Noop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x):
        ctx.save_for_backward(initial_value, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_h):
        initial_value, alpha = ctx.saved_tensors
        grad_initial_value = torch.zeros_like(initial_value)
        grad_alpha = torch.zeros_like(alpha)
        grad_x = grad_h
        return grad_initial_value, grad_alpha, grad_x


class AutogradIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_value, alpha, x):
        return x

    @once_differentiable
    @staticmethod
    def backward(ctx, grad_x):
        return None, None, grad_x
