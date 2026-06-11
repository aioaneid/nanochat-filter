import torch

def ltv_plane_max_cuda(initial_value, alpha, x):
    """
    Apply ltv scan using the CUDA plane scan implementation.
    """
    return torch.ops.nanochat_cuda.ltv_plane_cuda_op(initial_value, alpha, x, "max")


def ltv_plane_log_cuda(initial_value, alpha, x):
    """
    Apply ltv scan using the CUDA plane scan implementation.
    """
    return torch.ops.nanochat_cuda.ltv_plane_cuda_op(initial_value, alpha, x, "log")
