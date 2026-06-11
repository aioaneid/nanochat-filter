import dataclasses
import itertools
import random
import time
from typing import Optional

import modal
import os
import subprocess
import sys
from dataclasses import dataclass

GPU_NAME = "L40S"

TORCH_CUDA_ARCH_LIST = [
    {
        "T4": "7.5",
        "L40S": "8.9",
        "rtx-pro-6000": "12.0",
        "h100": "9.0",
        "b200": "10.0",
    }[GPU_NAME]
]

# ---------------------------------------------------------------------------
# Volume to store profiling outputs (PyTorch traces + SASS logs)
# ---------------------------------------------------------------------------
profiling_vol = modal.Volume.from_name("profiling-reports", create_if_missing=True)

# 1. Define the build environment
# Mirrors: remote_build.sh (cpp_ewma build) + run_bench_gpt_ltv.sh (env setup)
# Excludes: Rust (rust_ewma)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.2-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "clang", "curl", "build-essential", "ninja-build")
    .pip_install("uv")
    # Install torch + build tools (equivalent to run_bench_gpt_ltv.sh's torch install)
    .pip_install(
        "torch", "setuptools", "wheel", "numpy", "ninja", "pytest", "pytest-benchmark"
    )
    .env(
        {
            # Kernel build flags - toggle these for faster compilation
            "BUILD_LTV_FUSED_SCAN": "0",
            "BUILD_LTV_FUSED_CONCAT_SCAN": "0",
            "BUILD_LTV_FUSED_CONCAT_HM_SCAN": "0",
            "BUILD_LTV_FUSED_CONCAT_BLELLOCH": "0",
            "BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH": "1",
            "BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL": "0",
            "BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH": "0",
            "BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_SEQUENTIAL": "0",
            "BUILD_LTV_PLANE": "0",
            "TORCH_CUDA_ARCH_LIST": ";".join(TORCH_CUDA_ARCH_LIST),
        }
    )
    # Copy local code into the image for building
    .add_local_dir(
        "./cpp_ewma",
        remote_path="/root/cpp_ewma",
        ignore=[".venv", "__pycache__", "build", "dist", "target"],
        copy=True,
    )
    .run_commands(
        "cd /root/cpp_ewma && python setup.py bdist_wheel",
        "uv pip install --system /root/cpp_ewma/dist/*.whl",
    )
    .add_local_dir(
        "./nanochat", remote_path="/root/nanochat", ignore=["__pycache__"], copy=True
    )
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    # 2. Build cpp_ewma wheel (equivalent to remote_build.sh: uv build --wheel)
    # 3. Install nanochat editable (equivalent to run_bench_gpt_ltv.sh, minus rust_ewma)
    .run_commands(
        "cd /root && sed -i '/rust_ewma/d' pyproject.toml && uv pip install --system --no-deps -e .",
    )
    .pip_install(
        "arrgh>=1.0.0",
        "datasets>=4.0.0",
        "fastapi>=0.117.1",
        "ipykernel>=7.1.0",
        "matplotlib>=3.10.8",
        "ndjson>=0.3.1",
        "pandas>=2.3.2",
        "psutil>=7.1.0",
        "python-dotenv>=1.2.1",
        "regex>=2025.9.1",
        "rustbpe>=0.1.0",
        "tensorboard>=2.20.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "transformers>=4.57.3",
        "uvicorn>=0.36.0",
        "wandb>=0.21.3",
        "enum-actions>=0.1.2",
        "strenum>=0.4.15",
    )
    .add_local_dir(
        "./tests", remote_path="/root/tests", ignore=["__pycache__"], copy=True
    )
    .add_local_dir(
        "./scripts", remote_path="/root/scripts", ignore=["__pycache__"], copy=True
    )
)

app = modal.App("gpt-ltv-bench", image=image)


@app.function(
    gpu=GPU_NAME,
    timeout=7200,
)
def run_benchmark(mode: str, sigmoid: bool):
    """Run the benchmark (requires GPU)."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Python executable: {sys.executable}")

    if mode == "test-standard":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=8",
            "tests/test_ltv_cuda_fused_concat_register_cast_blelloch_scan.py",
            "--benchmark=false",
            f"--sigmoid={sigmoid}",
            "--check_against_expected=true",
        ]
    elif mode == "test-head-major-blelloch":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=4",
            # "-k",
            # "test_backward_exact[16-8-2-1-8-2-4]   ",
            "tests/test_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan.py",
            "--benchmark=false",
            f"--sigmoid={sigmoid}",
            "--check_against_expected=true",
        ]
    elif mode == "test-head-major-original":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=8",
            "tests/test_cuda_fused_concat_head_major_original.py",
            "--benchmark=false",
            f"--sigmoid={sigmoid}",
            "--check_against_expected=true",
        ]
    elif mode == "test-look-back-blelloch":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=1",
            "tests/test_ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan.py",
            "--benchmark=false",
            f"--sigmoid={sigmoid}",
            "--check_against_expected=true",
        ]
    elif mode == "test-look-back-sequential":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=1",
            "tests/test_ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan.py",
            "--benchmark=false",
            f"--sigmoid={sigmoid}",
            "--check_against_expected=true",
        ]
    elif mode == "test-fused-linear-transpose":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--capture=no",
            "--log-cli-level=INFO",
            "--maxfail=1",
            "tests/test_fused_linear_transpose.py",
        ]
    elif mode == "bench":
        cmd = [
            sys.executable,
            "-m",
            "nanochat.bench.bench_gpt_ltv",
            "--start_t",
            "1024",
            "--sequence_len",
            "1024",
            "--h",
            "8",
            "--b",
            "1",
            "--ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan",
            "--verify",
        ]
    else:
        return f"Unknown mode: {mode}"

    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


@dataclass(frozen=True)
class BaseTrainSpec:
    h: int
    head_dim: int
    device_batch_size: int
    max_seq_len: int
    base_train_iterations: int
    combined_optimizers: str
    depth: int
    model_type: str
    ltv_r: int
    ltv_query: str
    ltv_key: str
    ltv_value: str
    layer_specs: list[str]
    alpha_mlp_bias_init: float
    rand_filter_bias: str
    rand_filter_init: str
    scan_threads_forward: int
    r_threads_forward: int
    r_items_forward: int
    scan_threads_backward: int
    r_threads_backward: int
    r_items_backward: int

    def __post_init__(self):
        assert not self.ltv_r or self.head_dim % self.ltv_r == 0, (
            "If present, ltv_r must divide head_dim."
        )

    def suffix(self):
        s = f"d_{self.depth}-co_{self.combined_optimizers}-mt_{self.model_type}-r_{self.ltv_r}-q_{self.ltv_query}-k_{self.ltv_key}-v_{self.ltv_value}-ambi_{self.alpha_mlp_bias_init}-rfb_{self.rand_filter_bias}-rfi_{self.rand_filter_init}"
        m = f"-layer_specs_{'_'.join(self.layer_specs)}" if self.layer_specs else ""
        d = f"-scan_threads_forward_{self.scan_threads_forward}-r_threads_forward_{self.r_threads_forward}-r_items_forward_{self.r_items_forward}-scan_threads_backward_{self.scan_threads_backward}-r_threads_backward_{self.r_threads_backward}-r_items_backward_{self.r_items_backward}"
        return s + m + d

    def ltv_head_major_blelloch_env(self):
        return {
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_THREADS_FORWARD": str(
                self.scan_threads_forward
            ),
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_THREADS_FORWARD": str(
                self.r_threads_forward
            ),
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_ITEMS_FORWARD": str(
                self.r_items_forward
            ),
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_THREADS_BACKWARD": str(
                self.scan_threads_backward
            ),
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_THREADS_BACKWARD": str(
                self.r_threads_backward
            ),
            "LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_R_ITEMS_BACKWARD": str(
                self.r_items_backward
            ),
        }


def is_dispatch_valid_forward(
    scan_threads: int, r_threads: int, r_items: int, max_shared_mem: int
) -> bool:
    # Approximate size of WarpScan::TempStorage for AffineState<T, kRItems>
    base_scan_size = scan_temp_size(scan_threads)
    scan_size = base_scan_size * (1 + r_items) // 2

    # scan_temp array: scan_size * r_threads
    # alpha_cache: kScanThreads floats
    static_smem = scan_size * r_threads + scan_threads * 4
    # +4 for safety (e.g. potential alignment padding – not strictly needed but harmless)
    return (static_smem + 4) <= max_shared_mem


def is_dispatch_valid_backward(
    scan_threads: int, r_threads: int, r_items: int, max_shared_mem: int
) -> bool:
    # Approximate size of WarpScan::TempStorage (same heuristic)
    base_scan_size = scan_temp_size(scan_threads)
    scan_size = base_scan_size * (1 + r_items) // 2

    # Union of scan_temp (scan_size * r_threads) and partial_flat (r_threads * scan_threads * 4)
    union_bytes = max(scan_size * r_threads, r_threads * scan_threads * 4)
    # running_x: r_threads * r_items floats
    running_x_bytes = r_threads * r_items * 4
    # shm_alpha_gate: scan_threads floats; carry_alpha_cache: 1 float
    gate_bytes = scan_threads * 4 + 4

    static_smem = union_bytes + running_x_bytes + gate_bytes
    return static_smem <= max_shared_mem


def scan_temp_size(scan_threads: int) -> int:
    return 8 if (scan_threads & (scan_threads - 1)) == 0 else 1024


def ncycles(iterable, n):
    """Returns the sequence elements n times."""
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def get_kernel_sass(kernel_mangled_substring):
    """
    Extract the SASS of a single kernel from the cpp_ewma_cuda .so file.
    Prints the result to stdout (Modal logs) and also returns it as a string.
    """
    import torch  # ensures libc10.so etc. are on the loader path
    import cpp_ewma_cuda

    so_path = cpp_ewma_cuda.__file__
    result = subprocess.run(
        ["cuobjdump", "-sass", so_path], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    in_kernel = False
    sass_lines = []
    for line in lines:
        if in_kernel:
            if line.startswith(".text.") or line.startswith("Fatbin"):
                break
            sass_lines.append(line)
        elif kernel_mangled_substring in line:
            in_kernel = True
            sass_lines.append(line)

    if not sass_lines:
        print(f"Warning: kernel '{kernel_mangled_substring}' not found.")
        return ""

    output = "\n".join(sass_lines)
    print(output)
    return output


@app.function(
    gpu=GPU_NAME,
    timeout=12 * 3600,
    volumes={
        "/root/nanochat-cache": modal.Volume.from_name(
            "nanochat-cache", create_if_missing=False, version=2
        ).read_only(),
        "/root/profiling": profiling_vol,
    },
)
def run_perf_base_train(
    max_forward_specs: int, max_backward_specs: int, profile: bool = False
):
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = "/root/nanochat-cache/nanochat.base.240"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/root/torch_cache/torchinductor-{GPU_NAME}"
    env["TRITON_CACHE_DIR"] = f"/root/torch_cache/triton-{GPU_NAME}"

    concat_original_depth_11 = BaseTrainSpec(
        h=8,
        head_dim=128,
        device_batch_size=4,
        max_seq_len=1024,
        base_train_iterations=2 if GPU_NAME == "T4" else 4,
        combined_optimizers="true",
        depth=11,
        model_type="concat_original",
        ltv_r=0,
        ltv_query="none",
        ltv_key="none",
        ltv_value="none",
        layer_specs=[],
        alpha_mlp_bias_init=0,
        rand_filter_bias="false",
        rand_filter_init="false",
        scan_threads_forward=0,
        r_threads_forward=0,
        r_items_forward=0,
        scan_threads_backward=0,
        r_threads_backward=0,
        r_items_backward=0,
    )
    concat_original_depth_12 = dataclasses.replace(concat_original_depth_11, depth=12)
    ltv_head_major_blelloch_depth_11_base = dataclasses.replace(
        concat_original_depth_11,
        model_type="ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan",
        ltv_r=concat_original_depth_11.head_dim,
        ltv_query="active",
        ltv_key="active",
        ltv_value="active",
        rand_filter_init="true",
    )

    # New: r_threads: 1, 4; r_items: 16, 32, 64, 128
    sequential_specs = [
        (r_threads, r_items)
        for r_threads in [1, 2, 4, 8, 16, 32, 64]
        for r_items in [1, 2, 4, 8, 16, 32, 64]
        if r_threads * r_items <= ltv_head_major_blelloch_depth_11_base.ltv_r
    ]

    blelloch_specs = [
        (scan_threads, r_threads, r_items)
        for scan_threads in [4, 8, 16, 32]
        for (r_threads, r_items) in sequential_specs
        if scan_threads * r_threads <= 1024
    ]

    PQKs = [
        # (2_147_483_647, 1, 1),
        # (16, 16, 9),
    ]

    base_train_specs = (
        [
            concat_original_depth_11,
            concat_original_depth_12,
        ]
        + [
            dataclasses.replace(
                ltv_head_major_blelloch_depth_11_base,
                model_type="ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan",
                scan_threads_forward=scan_threads_forward,
                r_threads_forward=r_threads_forward,
                r_items_forward=r_items_forward,
                scan_threads_backward=scan_threads_backward,
                r_threads_backward=r_threads_backward,
                r_items_backward=r_items_backward,
                layer_specs=["{},{},{}".format(p, q, k)] * 11,
            )
            for (p, q, k) in PQKs
            for (
                scan_threads_forward,
                r_threads_forward,
                r_items_forward,
            ) in blelloch_specs
            if True
            and scan_threads_forward <= 1 << (max(p, q + k - 1) - 1).bit_length()
            and is_dispatch_valid_forward(
                scan_threads_forward,
                r_threads_forward,
                r_items_forward,
                max_shared_mem=48 * 1024,
            )
            and (p, q, k, scan_threads_forward, r_threads_forward, r_items_forward)
            in {
                (2147483647, 1, 1, 4, 16, 2),
                # (64, 56, 9, 32, 16, 8),
            }
            for (
                scan_threads_backward,
                r_threads_backward,
                r_items_backward,
            ) in blelloch_specs
            if True
            and scan_threads_backward <= 1 << (max(p, q + k - 1) - 1).bit_length()
            and is_dispatch_valid_backward(
                scan_threads_backward,
                r_threads_backward,
                r_items_backward,
                max_shared_mem=48 * 1024,
            )
            and (p, q, k, scan_threads_backward, r_threads_backward, r_items_backward)
            in {
                (2147483647, 1, 1, 32, 2, 4),
                # (64, 56, 9, 32, 16, 8),
            }
        ]
        + sum(
            [
                [
                    full_ltv := dataclasses.replace(
                        ltv_head_major_blelloch_depth_11_base,
                        scan_threads_forward=scan_threads_forward,
                        r_threads_forward=r_threads_forward,
                        r_items_forward=r_items_forward,
                        scan_threads_backward=scan_threads_backward,
                        r_threads_backward=r_threads_backward,
                        r_items_backward=r_items_backward,
                        layer_specs=["{},{},{}".format(p, q, k)] * 11,
                    ),
                    dataclasses.replace(
                        full_ltv,
                        layer_specs=["{},{},{}".format(p, q, k)] * 1,
                    ),
                ]
                for (p, q, k) in [(2147483647, 1, 1)]
                for (
                    scan_threads_forward,
                    r_threads_forward,
                    r_items_forward,
                ) in blelloch_specs
                if True
                and scan_threads_forward <= 1 << (max(p, q + k - 1) - 1).bit_length()
                and is_dispatch_valid_forward(
                    scan_threads_forward,
                    r_threads_forward,
                    r_items_forward,
                    max_shared_mem=48 * 1024,
                )
                and (scan_threads_forward, r_threads_forward, r_items_forward)
                in {
                    (8, 8, 4),
                }
                for (
                    scan_threads_backward,
                    r_threads_backward,
                    r_items_backward,
                ) in blelloch_specs
                if True
                and scan_threads_backward <= 1 << (max(p, q + k - 1) - 1).bit_length()
                and is_dispatch_valid_backward(
                    scan_threads_backward,
                    r_threads_backward,
                    r_items_backward,
                    max_shared_mem=48 * 1024,
                )
                # and scan_threads_backward * r_threads_backward >= 32
                # and (
                #     False
                #     or (
                #         True
                #         and scan_threads_backward >= 16
                #         and (r_threads_backward in {1, 4} or r_items_backward >= 16)
                #     )
                #     or (scan_threads_backward, r_threads_backward)
                #     in {
                #         (32, 2),
                #     }
                # )
                and (
                    (scan_threads_backward, r_threads_backward, r_items_backward)
                    in [
                        # (32, 2, 4),
                        (32, 2, 8),
                        # (32, 4, 8),
                    ]
                    # + [(32, t, 16) for t in [1, 2, 4, 8, 16, 32]]
                )
            ],
            start=[],
        )
        + [
            dataclasses.replace(
                ltv_head_major_blelloch_depth_11_base,
                model_type="ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan",
                r_threads_forward=r_threads_forward,
                r_items_forward=r_items_forward,
                r_threads_backward=r_threads_backward,
                r_items_backward=r_items_backward,
                layer_specs=["{},{},{}".format(p, q, k)] * 11,
            )
            for (p, q, k) in PQKs
            for (
                r_threads_forward,
                r_items_forward,
            ) in sequential_specs[:max_forward_specs]
            if True
            for (
                r_threads_backward,
                r_items_backward,
            ) in sequential_specs[:max_backward_specs]
            if True
        ]
        + []
    )
    print("len(base_train_specs):", len(base_train_specs))

    cmds = [
        (
            [
                sys.executable,
                "-m",
                "scripts.base_train",
                f"--model_tag={spec.suffix()}",
                "--report_name=",
                f"--depth={spec.depth}",
                f"--max_seq_len={spec.max_seq_len}",
                "--run=dummy",
                f"--device_batch_size={spec.device_batch_size}",
                "--eval_every=-1",
                "--eval_tokens=4096",
                "--core_metric_every=-1",
                "--sample_every=-1",
                "--save_every=-1",
                "--save_checkpoint_at_last_step=false",
                "--exit_after_save=false",
                f"--model_dim={spec.h * spec.head_dim}",
                f"--combined_optimizers={spec.combined_optimizers}",
                f"--model_type={spec.model_type}",
                f"--ltv_r={spec.ltv_r}",
                f"--ltv_query={spec.ltv_query}",
                f"--ltv_key={spec.ltv_key}",
                f"--ltv_value={spec.ltv_value}",
                "--layer_specs={}".format(
                    "_".join(layer_spec for layer_spec in spec.layer_specs)
                ),
                f"--alpha_mlp_bias_init={spec.alpha_mlp_bias_init}",
                f"--rand_filter_bias={spec.rand_filter_bias}",
                f"--rand_filter_init={spec.rand_filter_init}",
                "--resume_from_step=-1",
                f"--num_iterations={spec.base_train_iterations}",
            ],
            {**env, **spec.ltv_head_major_blelloch_env()},
        )
        for spec in base_train_specs
    ]

    rng = random.Random(13)
    rng.shuffle(cmds)

    for i, (cmd, env_dict) in enumerate(ncycles(cmds, 8)):
        ts = time.strftime("%Y%m%d_%H%M%S")

        if profile and i == 0:
            train_args = cmd[3:]
            wrapper = f"""
import sys
sys.path.insert(0, '/root')
import torch
import torch.profiler as prof
import runpy

sys.argv = ['scripts.base_train'] + {train_args!r}

trace_dir = "/root/profiling/trace_{ts}"
print(f"Saving lean profiler trace to {{trace_dir}}")

with prof.profile(activities=[prof.ProfilerActivity.CUDA]) as profiler:
    runpy.run_module('scripts.base_train', run_name='__main__')

profiler.export_chrome_trace(trace_dir + ".json")
print("Trace exported.")
"""
            wrapper_path = f"/tmp/train_wrapper_{ts}.py"
            with open(wrapper_path, "w") as f:
                f.write(wrapper)
            cmd = [sys.executable, wrapper_path]
            print(
                f"🔍 Profiling first training run → trace will be in /root/profiling/trace_{ts}"
            )
        else:
            print(f"🚀 Running: {' '.join(cmd)}")

        subprocess.run(cmd, env=env_dict, check=True)

    # No SASS extraction here – use dump_sass_only() separately.
    if profile:
        print("\n📊 Profile trace saved in /root/profiling/")
        print("Download with:  modal volume get profiling-reports /root/profiling/ .")


@app.function(
    gpu=GPU_NAME,
    timeout=300,
    volumes={"/root/profiling": profiling_vol},
)
def dump_sass_only(output_dir: str = None):
    import torch
    import cpp_ewma_cuda
    import subprocess
    import os
    import shutil
    import time

    if not output_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/root/profiling/{ts}"
    os.makedirs(output_dir, exist_ok=True)
    so_path = cpp_ewma_cuda.__file__

    # ---- full SASS ----
    sass_file = os.path.join(output_dir, "all_sass.txt")
    print(f"Dumping full SASS to {sass_file} ...")
    with open(sass_file, "w") as f:
        subprocess.run(["cuobjdump", "-sass", so_path], stdout=f, check=True)
    print(f"Full SASS saved ({os.path.getsize(sass_file)} bytes).")

    # ---- resource usage (registers / smem) ----
    res_file = os.path.join(output_dir, "all_resource_usage.txt")
    print(f"Dumping resource usage to {res_file} ...")
    with open(res_file, "w") as f:
        subprocess.run(["cuobjdump", "-res-usage", so_path], stdout=f, check=True)
    print(f"Resource usage saved ({os.path.getsize(res_file)} bytes).")

    # ---- copy the .so file instead of a non-existent cubin dump ----
    so_copy = os.path.join(output_dir, os.path.basename(so_path))
    print(f"Copying .so to {so_copy} ...")
    shutil.copy2(so_path, so_copy)
    print(f".so copied ({os.path.getsize(so_copy)} bytes).")


@app.local_entrypoint()
def main(
    mode: Optional[str] = None,
    sigmoid: bool = False,
    perf_base_train: bool = False,
    max_forward_specs: int = 1000,
    max_backward_specs: int = 1000,
    profile: bool = False,
):
    """Entrypoint: run benchmark (GPU required) or upload local cache (no GPU)."""
    if mode:
        run_benchmark.remote(mode=mode, sigmoid=sigmoid)
    if perf_base_train:
        run_perf_base_train.remote(
            max_forward_specs, max_backward_specs, profile=profile
        )
