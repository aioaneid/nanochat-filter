from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- BYPASS CUDA VERSION CHECK ---
import torch.utils.cpp_extension

torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

import os
import subprocess
import sys

# Kernel to source mapping
KERNEL_SOURCES = {
    "BUILD_LTV_FUSED_SCAN": [
        "src/fused_scan.cpp",
        "src/fused_scan_kernel.cu",
    ],
    "BUILD_LTV_FUSED_CONCAT_SCAN": [
        "src/fused_concat_scan.cpp",
        "src/fused_concat_register_cast_scan_kernel.cu",
    ],
    "BUILD_LTV_FUSED_CONCAT_HM_SCAN": [
        "src/fused_concat_head_major_register_cast_scan_kernel.cu",
    ],
    "BUILD_LTV_FUSED_CONCAT_BLELLOCH": [
        "src/ltv_fused_concat_register_cast_blelloch_scan_cuda_bindings.cpp",
        "src/ltv_fused_concat_register_cast_blelloch_scan_cuda.cu",
    ],
    "BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH": [
        "src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_bindings.cpp",
        "src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_bindings.cpp",
        "src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu",
        "src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu",
        "src/fused_linear_transpose_forward.cu",
        "src/fused_linear_transpose_forward_bindings.cpp",
        "src/fused_linear_transpose_backward.cu",
        "src/fused_linear_transpose_backward_bindings.cpp",
    ],
    "BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL": [
        "src/cuda_fused_concat_head_major_original_cuda_forward.cu",
        "src/cuda_fused_concat_head_major_original_cuda_forward_bindings.cpp",
        "src/cuda_fused_concat_head_major_original_cuda_backward.cu",
        "src/cuda_fused_concat_head_major_original_cuda_backward_bindings.cpp",
        "src/fused_linear_transpose_forward.cu",
        "src/fused_linear_transpose_forward_bindings.cpp",
        "src/fused_linear_transpose_backward.cu",
        "src/fused_linear_transpose_backward_bindings.cpp",
    ],
    "BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH": [
        "src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward_bindings.cpp",
        "src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward_bindings.cpp",
        "src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu",
        "src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu",
        "src/fused_linear_transpose_forward.cu",
        "src/fused_linear_transpose_forward_bindings.cpp",
        "src/fused_linear_transpose_backward.cu",
        "src/fused_linear_transpose_backward_bindings.cpp",
    ],
    "BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_SEQUENTIAL": [
        "src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward_bindings.cpp",
        "src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward_bindings.cpp",
        "src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward.cu",
        "src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward.cu",
    ],
    "BUILD_LTV_PLANE": [
        "src/plane_cuda.cpp",
        "src/plane_cuda_kernel.cu",
    ],
}

# Determine which kernels to build.
# If no specific kernels are requested via env vars, build nothing.
enabled_kernels = [k for k in KERNEL_SOURCES if os.environ.get(k, "0") == "1"]

sources = ["src/bindings.cpp"]
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        # "-Xptxas=-v",
    ],
}


def resolve_hm_blelloch_max_smem_bytes():
    value = os.environ.get("LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES")
    if value is None:
        script = os.path.join(os.path.dirname(__file__), "cuda_arch_smem.py")
        value = subprocess.check_output([sys.executable, script], text=True).strip()
    bytes_value = int(value)
    if bytes_value <= 0:
        raise ValueError(
            "LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES must be a positive integer"
        )
    return bytes_value


hm_blelloch_max_smem_bytes = resolve_hm_blelloch_max_smem_bytes()
extra_compile_args["cxx"].append(
    f"-DLTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES={hm_blelloch_max_smem_bytes}"
)
extra_compile_args["nvcc"].append(
    f"-DLTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES={hm_blelloch_max_smem_bytes}"
)

for kernel in enabled_kernels:
    sources.extend(KERNEL_SOURCES[kernel])
    extra_compile_args["cxx"].append(f"-D{kernel}")
    extra_compile_args["nvcc"].append(f"-D{kernel}")

setup(
    name="cpp_ewma",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="cpp_ewma_cuda",
            sources=list(
                dict.fromkeys(sources)
            ),  # Remove duplicates while preserving order
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
        )
    },
)
