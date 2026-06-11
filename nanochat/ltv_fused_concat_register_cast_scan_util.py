import torch

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.train_utils import env_int


CUDA_SUPPORTS_NATIVE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(
    including_emulation=False
)


def should_run_cuda_ltv_in_fp32(tensor: torch.Tensor) -> bool:
    return (
        not CUDA_SUPPORTS_NATIVE_BF16
        and tensor.dtype == torch.bfloat16
        and tensor.is_cuda
    )


def should_run_ltv_in_fp32(device_type: str) -> bool:
    """
    Returns True only when the LTV op should be run in float32.
    Respects autocast; only downgrades on CUDA when bfloat16 is requested
    but native support is missing.
    """
    if torch.is_autocast_enabled(device_type):
        autocast_dtype = torch.get_autocast_dtype(device_type)
        if device_type == "cuda":
            if autocast_dtype == torch.float32:
                return True
            if autocast_dtype == torch.bfloat16 and not CUDA_SUPPORTS_NATIVE_BF16:
                return True
        # For MPS/CPU or other device types, never force float32
        return False
    # No autocast – never force float32
    return False


def resolve_inits_original_dtype(B, init_param, previous_values: LtvInitValues):
    if previous_values.q is not None:
        return torch.cat(
            [previous_values.q, previous_values.k, previous_values.v], dim=1
        )
    return init_param.unsqueeze(0).expand(B, -1, -1)


# Used to be 4 before shared memory alpha cache.
_SCAN_THREADS_FORWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_THREADS_FORWARD", 8
)
# Used to be 16 before shared memory alpha cache.
_R_THREADS_FORWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_THREADS_FORWARD", 8
)
# Used to be 2 before shared memory alpha cache.
_R_ITEMS_FORWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_ITEMS_FORWARD", 4
)

_SCAN_THREADS_BACKWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_THREADS_BACKWARD", 32
)
# Used to be 2 before shared memory alpha cache.
_R_THREADS_BACKWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_THREADS_BACKWARD", 2
)
# Used to be 4 before shared memory alpha cache.
_R_ITEMS_BACKWARD = env_int(
    "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_ITEMS_BACKWARD", 8
)
