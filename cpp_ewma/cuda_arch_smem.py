#!/usr/bin/env python3
import os
import re
import sys


def clip_upper_bound(d, upper_bound):
    return {k: min(v, upper_bound) for k, v in d.items()}


# Conservative per-block shared-memory limits by CUDA compute capability.
# Values are bytes. The build uses the minimum across requested architectures
# so a single extension can be compiled for all entries in TORCH_CUDA_ARCH_LIST.
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-table-memory-information-per-compute-capability
SMEM_BYTES_BY_ARCH_MAXIMUM = clip_upper_bound(
    {
        # Maximum dynamic shared memory per block, with opt-in where applicable.
        "7.5": 64 * 1024,
        "8.0": 163 * 1024,
        "8.6": 99 * 1024,
        "8.7": 163 * 1024,
        "8.9": 99 * 1024,
        "9.0": 227 * 1024,
        "10.0": 227 * 1024,
        "11.0": 227 * 1024,
        "12.0": 99 * 1024,
    },
    # Kernels relying on shared memory allocations over 48 KB per block are
    # architecture-specific. As such they must use dynamic shared memory
    # rather than statically-sized arrays and require an explicit opt-in using
    # cudaFuncSetAttribute.
    48 * 1024,
)


def _normalize_arch(raw):
    arch = raw.strip()
    arch = arch.removesuffix("+PTX").removesuffix("+ptx")
    arch = arch.removeprefix("sm_").removeprefix("compute_")
    if re.fullmatch(r"\d{2,3}", arch):
        return f"{arch[:-1]}.{arch[-1]}"
    return arch


def _archs_from_torch_cuda_arch_list(value):
    archs = []
    for part in re.split(r"[;,\s]+", value.strip()):
        if not part:
            continue
        archs.append(_normalize_arch(part))
    return archs


def min_smem_for_archs(archs):
    unknown = sorted(set(archs) - set(SMEM_BYTES_BY_ARCH_MAXIMUM))
    if unknown:
        known = ", ".join(sorted(SMEM_BYTES_BY_ARCH_MAXIMUM))
        raise SystemExit(
            f"Unknown CUDA arch shared-memory limit for {unknown}; known: {known}"
        )
    return min(SMEM_BYTES_BY_ARCH_MAXIMUM[arch] for arch in archs)


def main():
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    archs = _archs_from_torch_cuda_arch_list(arch_list) if arch_list else []
    if not archs:
        raise SystemExit(
            "Cannot compute shared-memory limit without TORCH_CUDA_ARCH_LIST. "
            "Set TORCH_CUDA_ARCH_LIST or LTV_FUSED_CONCAT_HM_BLELLOCH_MAX_SMEM_BYTES."
        )
    print(min_smem_for_archs(archs))


if __name__ == "__main__":
    sys.exit(main())
