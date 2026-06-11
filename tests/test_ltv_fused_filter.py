import pytest
import torch
import logging

from nanochat.ltv.ltv_filter import LtvInitValues
from nanochat.ltv.ltv_fused_scan_filter import LtvFusedScanFilter, LtvFusedPyTorchFilter
from nanochat.ltv.ltv_fused_linear_scan_filter import LtvFusedLinearScanFilter

try:
    import rust_ewma
except ImportError:
    rust_ewma = None

logger = logging.getLogger(__name__)


# Determine device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def create_inputs(B, T, n_head, n_kv_head, head_dim, r):
    torch.manual_seed(42)
    n_embd = n_head * head_dim
    x = torch.randn(B, T, n_embd, device=DEVICE)
    q = torch.randn(B, T, n_head, head_dim, device=DEVICE)
    k = torch.randn(B, T, n_kv_head, head_dim, device=DEVICE)
    v = torch.randn(B, T, n_kv_head, head_dim, device=DEVICE)
    return {
        "x": x,
        "q": q,
        "k": k,
        "v": v,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "head_dim": head_dim,
        "r": r,
    }


def run_model(
    filter_class,
    input_data,
    *,
    requires_grad: bool,
    previous_values: LtvInitValues,
):
    torch.manual_seed(12345)

    # Both FusedScan and FusedLinearScan now inherit from Root
    # and use the same constructor signature (n_head, n_kv_head, head_dim, r)
    model = filter_class(
        n_head=input_data["n_head"],
        n_kv_head=input_data["n_kv_head"],
        head_dim=input_data["head_dim"],
        r=input_data["r"],
    ).to(DEVICE)

    with torch.no_grad():
        model.init_weights(alpha_mlp_bias_init=2, weight_s=3, bias_s=4, init_s=5)

    x = input_data["x"].clone()
    q = input_data["q"].clone()
    k = input_data["k"].clone()
    v = input_data["v"].clone()

    if requires_grad:
        x.requires_grad_(True)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

    out = model(x, q, k, v, previous_values=previous_values)
    return out, (model, x, q, k, v)


TEST_PARAMS = [
    (1, 1, 1, 1, 1, 1),
    (2, 1, 1, 1, 1, 1),
    (1, 2, 1, 1, 1, 1),
    (1, 1, 2, 1, 1, 1),
    (1, 1, 2, 2, 1, 1),
    (1, 1, 1, 1, 2, 1),
    (1, 1, 1, 1, 2, 2),
    (1, 1, 1, 1, 4, 2),
    (1, 1, 1, 1, 3, 3),
    (1, 1, 1, 1, 6, 3),
    (3, 128, 8, 1, 32, 1),
    (3, 128, 8, 1, 32, 2),
    (3, 128, 35, 5, 14, 1),
    (3, 128, 35, 5, 14, 7),
]

# The list of active implementations to test against the PyTorch Baseline
IMPLEMENTATIONS = [
    (
        LtvFusedScanFilter,
        rust_ewma.release_ltv_fused_scan_metal_context,
        rust_ewma.init_ltv_fused_scan_metal_context_for_default_device,
    ),
    (
        LtvFusedLinearScanFilter,
        rust_ewma.release_ltv_fused_linear_scan_metal_context,
        rust_ewma.init_ltv_fused_linear_scan_metal_context_for_default_device,
    ),
]


def release_init_fn(release_ltv_fn, init_ltv_fn):
    if rust_ewma:
        release_ltv_fn()
        try:
            init_ltv_fn()
        except RuntimeError as e:
            if not any("already initialized" in str(arg) for arg in e.args):
                raise
    else:
        pytest.skip("rust_ewma not available")


@pytest.mark.parametrize("filter_class, release_ltv_fn, init_ltv_fn", IMPLEMENTATIONS)
@pytest.mark.parametrize("B, T, n_head, n_kv_head, head_dim, r", TEST_PARAMS)
def test_ltv_fused_filter_forward(
    filter_class, release_ltv_fn, init_ltv_fn, B, T, n_head, n_kv_head, head_dim, r
):
    input_data = create_inputs(B, T, n_head, n_kv_head, head_dim, r)

    release_init_fn(release_ltv_fn, init_ltv_fn)

    with torch.no_grad():
        # Baseline
        out_loop, _ = run_model(
            LtvFusedPyTorchFilter,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

        # Target (FusedScan or FusedLinearScan)
        out_target, _ = run_model(
            filter_class,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    for i, (o_loop, o_target) in enumerate(zip(out_loop, out_target)):
        torch.testing.assert_close(
            o_target,
            o_loop,
            rtol=1e-4,
            atol=1.3e-5,
            msg=lambda s: f"{filter_class.__name__} forward mismatch: {s}",
        )


def force_stride_layout(t, dims):
    # Permute to the target physical layout, clone to fix memory, then permute back
    return (
        t.permute(*dims)
        .clone(memory_format=torch.contiguous_format)
        .permute(*(torch.argsort(torch.tensor(dims)).tolist()))
    )


@pytest.mark.parametrize("filter_class, release_ltv_fn, init_ltv_fn", IMPLEMENTATIONS)
@pytest.mark.parametrize("B, T, n_head, n_kv_head, head_dim, r", TEST_PARAMS)
def test_ltv_filter_backward(
    filter_class, release_ltv_fn, init_ltv_fn, B, T, n_head, n_kv_head, head_dim, r
):
    input_data = create_inputs(B, T, n_head, n_kv_head, head_dim, r)

    release_init_fn(release_ltv_fn, init_ltv_fn)

    # 1. Baseline
    out_loop_tuple, (model_loop, x_loop, q_loop, k_loop, v_loop) = run_model(
        LtvFusedPyTorchFilter,
        input_data,
        requires_grad=True,
        previous_values=LtvInitValues(None, None, None),
    )

    # MANUALLY VARY STRIDES TO MATCH PRODUCTION LOGS
    # Q and V usually come in as (B, H, T, D) -> (0, 2, 1, 3)
    # K often comes in as (B, H, D, T) -> (0, 2, 3, 1)
    # We apply multipliers (o * 5) and then permute to simulate Inductor behavior

    # Q: (B, T, NH, D) logically, but (B, NH, T, D) physically
    go_loop_q = force_stride_layout(out_loop_tuple[0] * 5.0, (0, 2, 1, 3))
    # K: (B, T, NKVH, D) logically, but (B, NKVH, D, T) physically
    go_loop_k = force_stride_layout(out_loop_tuple[1] * 5.0, (0, 2, 3, 1))
    # V: (B, T, NKVH, D) logically, but (B, NKVH, T, D) physically
    go_loop_v = force_stride_layout(out_loop_tuple[2] * 5.0, (0, 2, 1, 3))

    # Trigger backward with these specifically-strided tensors
    torch.autograd.backward(
        [out_loop_tuple[0], out_loop_tuple[1], out_loop_tuple[2]],
        [go_loop_q, go_loop_k, go_loop_v],
    )

    # 2. Target
    out_target_tuple, (model_target, x_target, q_target, k_target, v_target) = (
        run_model(
            filter_class,
            input_data,
            requires_grad=True,
            previous_values=LtvInitValues(None, None, None),
        )
    )

    # MANUALLY VARY STRIDES TO MATCH PRODUCTION LOGS
    # Q and V usually come in as (B, H, T, D) -> (0, 2, 1, 3)
    # K often comes in as (B, H, D, T) -> (0, 2, 3, 1)
    # We apply multipliers (o * 5) and then permute to simulate Inductor behavior

    # Q: (B, T, NH, D) logically, but (B, NH, T, D) physically
    go_target_q = force_stride_layout(out_target_tuple[0] * 5.0, (0, 2, 1, 3))
    # K: (B, T, NKVH, D) logically, but (B, NKVH, D, T) physically
    go_target_k = force_stride_layout(out_target_tuple[1] * 5.0, (0, 2, 3, 1))
    # V: (B, T, NKVH, D) logically, but (B, NKVH, T, D) physically
    go_target_v = force_stride_layout(out_target_tuple[2] * 5.0, (0, 2, 1, 3))

    # Trigger backward with these specifically-strided tensors
    torch.autograd.backward(
        [out_target_tuple[0], out_target_tuple[1], out_target_tuple[2]],
        [go_target_q, go_target_k, go_target_v],
    )

    atol = 1e-3
    rtol = 4e-3

    # Check model params (init_param and alpha_mlp)
    for (n_loop, p_loop), (n_target, p_target) in zip(
        model_loop.named_parameters(), model_target.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            p_target.grad,
            p_loop.grad,
            rtol=rtol,
            atol=atol,
            msg=lambda s: (
                f"{filter_class} backward {n_target} {p_target.grad} grad mismatch against {n_loop} {p_loop.grad}: {s}"
            ),
        )
        logger.info(f"Parameter gradient match for {n_target} vs {n_loop}")

    # Check input gradients
    for name, g_target, g_loop in [
        ("x", x_target.grad, x_loop.grad),
        ("q", q_target.grad, q_loop.grad),
        ("k", k_target.grad, k_loop.grad),
        ("v", v_target.grad, v_loop.grad),
    ]:
        torch.testing.assert_close(
            g_target,
            g_loop,
            rtol=rtol,
            atol=atol,
            msg=lambda s: (
                f"{filter_class} backward {name} {g_target} grad mismatch against {g_loop}: {s}"
            ),
        )
        logger.info(f"Input gradient match for {name}")


@pytest.mark.parametrize("filter_class, release_ltv_fn, init_ltv_fn", IMPLEMENTATIONS)
@pytest.mark.parametrize("B, T, n_head, n_kv_head, head_dim, r", TEST_PARAMS)
def test_ltv_filter_state_continuity(
    filter_class, release_ltv_fn, init_ltv_fn, B, T, n_head, n_kv_head, head_dim, r
):
    if T < 2:
        pytest.skip("T < 2")
    input_data = create_inputs(B, T, n_head, n_kv_head, head_dim, r)

    release_init_fn(release_ltv_fn, init_ltv_fn)

    # 1. Full Run
    with torch.no_grad():
        out_full, _ = run_model(
            filter_class,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    # 2. Segmented Run
    T_split = T // 2

    def slice_it(d, start, end):
        return {
            k: (
                v[:, start:end]
                .contiguous()
                .clone(memory_format=torch.contiguous_format)
                if isinstance(v, torch.Tensor) and v.ndim >= 2
                else v
            )
            for k, v in d.items()
        }

    with torch.no_grad():
        out_p1, _ = run_model(
            filter_class,
            slice_it(input_data, 0, T_split),
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

        # Pass state
        prev_vals = LtvInitValues(
            q=out_p1[0][:, -1], k=out_p1[1][:, -1], v=out_p1[2][:, -1]
        )

        out_p2, _ = run_model(
            filter_class,
            slice_it(input_data, T_split, T),
            requires_grad=False,
            previous_values=prev_vals,
        )

    # 3. Compare
    for full, p1, p2 in zip(out_full, out_p1, out_p2, strict=True):
        reconstructed = torch.cat([p1, p2], dim=1)
        torch.testing.assert_close(reconstructed, full, rtol=1e-4, atol=1e-5)
