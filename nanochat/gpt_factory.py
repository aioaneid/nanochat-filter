from nanochat.ltv_combined_qkv_computer_module import LtvCombinedQkvComputerModule
from nanochat.ltv_combined_qkv_filter_base import LtvCombinedQkvFilterBase
from nanochat.ltv_cuda_fused_concat_register_cast_blelloch_scan_qkv_filter import (
    LtvCudaFusedConcatRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter import (
    LtvCudaFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter,
)
from nanochat.cuda_fused_concat_head_major_original_qkv_filter import (
    cuda_fused_concat_head_major_original_qkv_computer,
)
from nanochat.ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_qkv_filter import (
    LookBackOp,
    LtvLookBackFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter,
    TransposeOp,
)
from nanochat.ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_qkv_filter import (
    LtvLookBackFusedConcatHeadMajorRegisterCastSequentialScanQkvFilter,
)
from nanochat.ltv_look_back_qkv_computer_module import LtvLookBackQkvComputerModule
from nanochat.ltv_metal_fused_concat_register_cast_blelloch_scan_qkv_filter import (
    LtvMetalFusedConcatRegisterCastBlellochScanQkvFilter,
)
from nanochat.ltv_concat_fused_scan_cuda_qkv_filter import (
    LtvConcatFusedScanCudaQkvFilter,
)
from nanochat.ltv_concat_fused_scan_triton_qkv_filter import (
    LtvConcatFusedScanTritonQkvFilter,
)
from nanochat.ltv_concat_register_cast_scan_cuda_qkv_filter import (
    LtvConcatRegisterCastScanCudaQkvFilter,
)
from enum import auto
from nanochat.concat_qkv_computer import (
    ConcatQkvComputerModule,
    default_qkv_computer,
)
from nanochat.ltv.ltv_filter import QkvSplitLtv
from strenum import StrEnum
from typing import Callable
import functools

import nanochat.ops.fused_linear_transpose  # Ensure op is registered
import nanochat.ops.ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan  # Ensure op is registered
import nanochat.ops.ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan  # Ensure op is registered


try:
    import rust_ewma
except ImportError:
    rust_ewma = None

import torch

from nanochat.gpt import (
    GPT,
    CausalSelfAttention,
    make_gpt_with_ltv_fn,
    make_ltv_fused_scan_filter,
    make_ltv_fused_linear_scan_filter,
    make_split_ltv_filter,
)
from nanochat.gpt_config import GPTConfig
from nanochat.ltv.ltv_metal_impls import (
    ltv_plane_max_metal,
    ltv_scan_metal,
    ltv_shared_max_metal,
    ltv_shared_pmax_metal,
)
from nanochat.ltv.ltv_cuda_impls import (
    ltv_plane_max_cuda,
)

from nanochat.ltv_combined_qkv_filter import LtvCombinedQkvFilter
from nanochat.ltv_concat_fused_scan_qkv_filter import (
    LtvConcatFusedScanQkvFilter,
)
from nanochat.ltv_fused_concat_register_cast_scan_cuda_qkv_filter import (
    LtvFusedConcatRegisterCastScanCudaQkvFilter,
)
from nanochat.ltv_fused_concat_head_major_register_cast_scan_cuda_qkv_filter import (
    LtvFusedConcatHeadMajorRegisterCastScanCudaQkvFilter,
)
from nanochat.ltv_fused_concat_register_cast_scan_metal_qkv_filter import (
    LtvFusedConcatRegisterCastScanMetalQkvFilter,
)
from nanochat.ltv_fused_concat_scan_qkv_filter import LtvFusedConcatScanQkvFilter
from nanochat.split_qkv_computer import SplitQkvComputerModule


class ModelType(StrEnum):
    ORIGINAL = auto()
    LTV_SCAN = auto()
    LTV_PLANE_MAX = auto()
    LTV_SHARED_MAX = auto()
    LTV_SHARED_PMAX = auto()
    LTV_FUSED_SCAN = auto()
    LTV_FUSED_LINEAR_SCAN = auto()
    SPLIT = auto()
    CONCAT_ORIGINAL = auto()
    CONCAT_DUPLICATE_ORIGINAL = auto()
    LTV_CONCAT_FUSED_SCAN = auto()
    LTV_CONCAT_DUPLICATE_FUSED_SCAN = auto()
    LTV_CONCAT_DUPLICATE2_FUSED_SCAN = auto()
    LTV_FUSED_CONCAT_SCAN = auto()
    LTV_CONCAT_FUSED_SCAN_CUDA = auto()
    LTV_CONCAT_FUSED_SCAN_TRITON = auto()
    LTV_CONCAT_REGISTER_CAST_SCAN_CUDA = auto()
    LTV_FUSED_CONCAT_REGISTER_CAST_SCAN_CUDA = auto()
    LTV_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_CUDA = auto()
    LTV_FUSED_CONCAT_REGISTER_CAST_SCAN_METAL = auto()
    LTV_FUSED_CONCAT_DUPLICATE_REGISTER_CAST_SCAN_METAL = auto()
    LTV_FUSED_CONCAT_DUPLICATE2_REGISTER_CAST_SCAN_METAL = auto()
    LTV_PLANE_CUDA_MAX = auto()
    LTV_METAL_FUSED_CONCAT_REGISTER_CAST_BLELLOCH_SCAN = auto()
    LTV_CUDA_FUSED_CONCAT_REGISTER_CAST_BLELLOCH_SCAN = auto()
    LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_BLELLOCH_SCAN = auto()
    LTV_CUDA_LOOK_BACK_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_BLELLOCH_SCAN = auto()
    LTV_CUDA_LOOK_BACK_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SEQUENTIAL_SCAN = auto()
    CUDA_FUSED_CONCAT_HEAD_MAJOR_ORIGINAL = auto()


class LtvSplitMode(StrEnum):
    NONE = auto()
    ACTIVE = auto()
    SPLIT_SCAN = auto()
    SPLIT_PLANE_MAX = auto()
    SPLIT_SHARED_MAX = auto()
    SPLIT_SHARED_PMAX = auto()


def always_throw(
    initial_value: torch.Tensor, alpha: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    raise ValueError("Not supposed to be ever called")


def make_split_original_model(config: GPTConfig):
    assert not config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config, layer_idx, SplitQkvComputerModule(config, layer_idx, None)
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_concat_original_model(config: GPTConfig):
    assert not config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                ConcatQkvComputerModule(config, layer_idx, None, default_qkv_computer),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_combined_model_with_filter_factory(
    config: GPTConfig, filter_factory: Callable[[], LtvCombinedQkvFilterBase]
):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvCombinedQkvComputerModule(
                    layer_idx,
                    filter_factory(),
                )
                if layer_idx < config.num_ltv_layers()
                else ConcatQkvComputerModule(
                    config.without_ltv(), layer_idx, None, default_qkv_computer
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_concat_fused_scan_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvConcatFusedScanQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_concat_fused_scan_cuda_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvConcatFusedScanCudaQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_concat_fused_scan_triton_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvConcatFusedScanTritonQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_concat_register_cast_scan_cuda_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvConcatRegisterCastScanCudaQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_fused_concat_register_cast_scan_cuda_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvFusedConcatRegisterCastScanCudaQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_fused_concat_head_major_register_cast_scan_cuda_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvFusedConcatHeadMajorRegisterCastScanCudaQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_fused_concat_register_cast_scan_metal_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvFusedConcatRegisterCastScanMetalQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_metal_fused_concat_register_cast_blelloch_scan_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvMetalFusedConcatRegisterCastBlellochScanQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_cuda_fused_concat_register_cast_blelloch_scan_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvCudaFusedConcatRegisterCastBlellochScanQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_model(
    config: GPTConfig,
):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvCudaFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_model(
    config: GPTConfig,
    transpose_op: TransposeOp,
    look_back_op: LookBackOp,
):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvLookBackQkvComputerModule(
                    layer_idx,
                    LtvLookBackFusedConcatHeadMajorRegisterCastBlellochScanQkvFilter(
                        config.n_head,
                        config.n_kv_head,
                        config.head_dim(),
                        config.ltv_r,
                        config.layer_specs[layer_idx],
                        transpose_op=transpose_op,
                        look_back_op=look_back_op,
                    ),
                )
                if layer_idx < config.num_ltv_layers()
                else ConcatQkvComputerModule(
                    config.without_ltv(), layer_idx, None, default_qkv_computer
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan_model(
    config: GPTConfig,
):
    from nanochat.ops.ltv_cuda_ops import (
        fused_linear_transpose,
        ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan,
    )

    return make_ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_model(
        config,
        transpose_op=fused_linear_transpose,
        look_back_op=ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan,
    )


def make_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_model(
    config: GPTConfig,
    look_back_op: LookBackOp,
):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                LtvLookBackQkvComputerModule(
                    layer_idx,
                    LtvLookBackFusedConcatHeadMajorRegisterCastSequentialScanQkvFilter(
                        config.n_head,
                        config.n_kv_head,
                        config.head_dim(),
                        config.ltv_r,
                        config.layer_specs[layer_idx],
                        sequential_look_back_op=look_back_op,
                    ),
                )
                if layer_idx < config.num_ltv_layers()
                else ConcatQkvComputerModule(
                    config.without_ltv(), layer_idx, None, default_qkv_computer
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan_model(
    config: GPTConfig,
):
    from nanochat.ops.ltv_cuda_ops import (
        ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan,
    )

    return make_ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_model(
        config,
        look_back_op=ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan,
    )


def make_cuda_fused_concat_head_major_original_model(config: GPTConfig):
    # The ltv_r parameter is only used for kernel launch topology, not for LTV filtering.
    # But if filtering is enabled, then for now it is forced to have the same "r".
    da = (head_dim := config.head_dim()) // config.ltv_r
    assert da * config.ltv_r == head_dim
    qkv_computer_fn = functools.partial(
        cuda_fused_concat_head_major_original_qkv_computer,
        da=da,
        r=config.ltv_r,
    )
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                ConcatQkvComputerModule(config, layer_idx, None, qkv_computer_fn),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_fused_concat_scan_model(config: GPTConfig):
    return make_ltv_combined_model_with_filter_factory(
        config,
        lambda: LtvFusedConcatScanQkvFilter(
            config.n_head, config.n_kv_head, config.head_dim(), config.ltv_r
        ),
    )


def make_ltv_scan_metal_model(config: GPTConfig):
    return make_gpt_with_ltv_fn(config, ltv_scan_metal)


def make_ltv_plane_max_metal_model(config: GPTConfig):
    return make_gpt_with_ltv_fn(config, ltv_plane_max_metal)


def make_ltv_plane_max_cuda_model(config: GPTConfig):
    return make_gpt_with_ltv_fn(config, ltv_plane_max_cuda)


def make_ltv_shared_max_metal_model(config: GPTConfig):
    return make_gpt_with_ltv_fn(config, ltv_shared_max_metal)


def make_ltv_shared_pmax_metal_model(config: GPTConfig):
    return make_gpt_with_ltv_fn(config, ltv_shared_pmax_metal)


def ltv_fn_from_ltv_split_mode(
    ltv_split_mode: LtvSplitMode,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    match ltv_split_mode:
        case LtvSplitMode.NONE:
            return always_throw
        case LtvSplitMode.ACTIVE:
            raise ValueError("Unspecified function")
        case LtvSplitMode.SPLIT_SCAN:
            return ltv_scan_metal
        case LtvSplitMode.SPLIT_PLANE_MAX:
            return ltv_plane_max_metal
        case LtvSplitMode.SPLIT_SHARED_MAX:
            return ltv_shared_max_metal
        case LtvSplitMode.SPLIT_SHARED_PMAX:
            return ltv_shared_pmax_metal
    raise ValueError("Unknown ltv_split_mode: {}".format(ltv_split_mode))


def make_ltv_fused_scan_model(config: GPTConfig):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                SplitQkvComputerModule(
                    config, layer_idx, make_ltv_fused_scan_filter(config)
                )
                if layer_idx < config.num_ltv_layers()
                else ConcatQkvComputerModule(
                    config.without_ltv(), layer_idx, None, default_qkv_computer
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_split_model(
    config: GPTConfig, query: LtvSplitMode, key: LtvSplitMode, value: LtvSplitMode
):
    assert config.ltv_r
    return GPT(
        config,
        attns=[
            CausalSelfAttention(
                config,
                layer_idx,
                SplitQkvComputerModule(
                    config,
                    layer_idx,
                    make_split_ltv_filter(
                        config,
                        QkvSplitLtv(
                            query=ltv_fn_from_ltv_split_mode(query),
                            key=ltv_fn_from_ltv_split_mode(key),
                            value=ltv_fn_from_ltv_split_mode(value),
                        ),
                    ),
                )
                if layer_idx < config.num_ltv_layers()
                else ConcatQkvComputerModule(
                    config.without_ltv(), layer_idx, None, default_qkv_computer
                ),
            )
            for layer_idx in range(config.n_layer)
        ],
    )


def make_ltv_split_model_factory(
    query: LtvSplitMode, key: LtvSplitMode, value: LtvSplitMode
):
    return lambda config: make_ltv_split_model(config, query, key, value)


def one_time_model_factory(
    model_type: ModelType, query: LtvSplitMode, key: LtvSplitMode, value: LtvSplitMode
) -> Callable[[GPTConfig], GPT]:
    s = {query, key, value}
    match model_type:
        case ModelType.ORIGINAL:
            assert query == key == value == LtvSplitMode.NONE
            return make_split_original_model
        case ModelType.CONCAT_ORIGINAL | ModelType.CONCAT_DUPLICATE_ORIGINAL:
            assert query == key == value == LtvSplitMode.NONE
            return make_concat_original_model
        case ModelType.LTV_SCAN:
            assert s.issubset({LtvSplitMode.NONE, LtvSplitMode.ACTIVE})
            assert LtvSplitMode.ACTIVE in s
            rust_ewma.init_ltv_scan_metal_context_for_default_device()
            return make_ltv_scan_metal_model
        case ModelType.LTV_FUSED_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
            return make_ltv_fused_scan_model
        case ModelType.LTV_FUSED_LINEAR_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_linear_scan_metal_context_for_default_device()
            return make_ltv_fused_linear_scan_filter
        case ModelType.LTV_PLANE_MAX:
            assert s.issubset({LtvSplitMode.NONE, LtvSplitMode.ACTIVE})
            assert LtvSplitMode.ACTIVE in s
            rust_ewma.init_ltv_plane_metal_context_for_default_device()
            return make_ltv_plane_max_metal_model
        case ModelType.LTV_PLANE_CUDA_MAX:
            assert s.issubset({LtvSplitMode.NONE, LtvSplitMode.ACTIVE})
            assert LtvSplitMode.ACTIVE in s
            return make_ltv_plane_max_cuda_model
        case ModelType.LTV_SHARED_MAX:
            assert s.issubset({LtvSplitMode.NONE, LtvSplitMode.ACTIVE})
            assert LtvSplitMode.ACTIVE in s
            rust_ewma.init_ltv_shared_metal_context_for_default_device()
            return make_ltv_shared_max_metal_model
        case ModelType.LTV_SHARED_PMAX:
            assert s.issubset({LtvSplitMode.NONE, LtvSplitMode.ACTIVE})
            assert LtvSplitMode.ACTIVE in s
            rust_ewma.init_ltv_shared_metal_context_for_default_device()
            return make_ltv_shared_pmax_metal_model
        case ModelType.SPLIT:
            assert LtvSplitMode.ACTIVE not in s
            # All None should be okay here I think.
            # assert s != {LtvSplitMode.SPLIT_NONE}
            if LtvSplitMode.SPLIT_SCAN in s:
                rust_ewma.init_ltv_scan_metal_context_for_default_device()
            if LtvSplitMode.SPLIT_PLANE_MAX in s:
                rust_ewma.init_ltv_plane_metal_context_for_default_device()
            if (
                LtvSplitMode.SPLIT_SHARED_MAX in s
                or LtvSplitMode.SPLIT_SHARED_PMAX in s
            ):
                rust_ewma.init_ltv_shared_metal_context_for_default_device()
            return make_ltv_split_model_factory(query, key, value)
        case ModelType.LTV_CONCAT_FUSED_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
            return make_ltv_concat_fused_scan_model
        case (
            ModelType.LTV_CONCAT_DUPLICATE_FUSED_SCAN
            | ModelType.LTV_CONCAT_DUPLICATE2_FUSED_SCAN
        ):
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_scan_metal_context_for_default_device()
            return make_ltv_concat_fused_scan_model
        case ModelType.LTV_CONCAT_FUSED_SCAN_CUDA:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_concat_fused_scan_cuda_model
        case ModelType.LTV_CONCAT_FUSED_SCAN_TRITON:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_concat_fused_scan_triton_model
        case ModelType.LTV_CONCAT_REGISTER_CAST_SCAN_CUDA:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_concat_register_cast_scan_cuda_model
        case ModelType.LTV_FUSED_CONCAT_REGISTER_CAST_SCAN_CUDA:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_fused_concat_register_cast_scan_cuda_model
        case ModelType.LTV_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SCAN_CUDA:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_fused_concat_head_major_register_cast_scan_cuda_model
        case (
            ModelType.LTV_FUSED_CONCAT_REGISTER_CAST_SCAN_METAL
            | ModelType.LTV_FUSED_CONCAT_DUPLICATE_REGISTER_CAST_SCAN_METAL
            | ModelType.LTV_FUSED_CONCAT_DUPLICATE2_REGISTER_CAST_SCAN_METAL
        ):
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device()
            return make_ltv_fused_concat_register_cast_scan_metal_model
        case ModelType.LTV_METAL_FUSED_CONCAT_REGISTER_CAST_BLELLOCH_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device(
                0, 0, 0, 1_000_000
            )
            return make_ltv_metal_fused_concat_register_cast_blelloch_scan_model
        case ModelType.LTV_CUDA_FUSED_CONCAT_REGISTER_CAST_BLELLOCH_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_cuda_fused_concat_register_cast_blelloch_scan_model
        case ModelType.LTV_CUDA_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_BLELLOCH_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            return (
                make_ltv_cuda_fused_concat_head_major_register_cast_blelloch_scan_model
            )
        case ModelType.LTV_CUDA_LOOK_BACK_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_BLELLOCH_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_cuda_look_back_fused_concat_head_major_register_cast_blelloch_scan_model
        case ModelType.LTV_CUDA_LOOK_BACK_FUSED_CONCAT_HEAD_MAJOR_REGISTER_CAST_SEQUENTIAL_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            return make_ltv_cuda_look_back_fused_concat_head_major_register_cast_sequential_scan_model
        case ModelType.CUDA_FUSED_CONCAT_HEAD_MAJOR_ORIGINAL:
            return make_cuda_fused_concat_head_major_original_model
        case ModelType.LTV_FUSED_CONCAT_SCAN:
            assert s == {LtvSplitMode.ACTIVE}
            rust_ewma.init_ltv_fused_concat_scan_metal_context_for_default_device()
            return make_ltv_fused_concat_scan_model

    raise ValueError("Unknown model type: {}".format(model_type))
