from typing import Callable
import itertools
import pytest
import torch
from nanochat.ltv.ltv_filter import LtvFilter, QkvLtv, LtvMode, LtvInitValues
from nanochat.ltv.ltv_impls import (
    ltv_pytorch_loop,
    LtvRustCpu,
)

try:
    import rust_ewma
except ImportError:
    rust_ewma = None


# Determine device
def get_device():
    # Force MPS if available as requested by user
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


@pytest.fixture
def input_data():
    torch.manual_seed(42)
    B, T = 2, 8
    n_head, n_kv_head, head_dim = 4, 2, 8
    n_embd = n_head * head_dim

    # Inputs required by LtvFilter.forward
    x = torch.randn(B, T, n_embd, device=DEVICE)

    # Sequence values
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
        "r": 2,
    }


# Generate all 8 combinations of LtvMode
LTV_MODES = list(itertools.product([LtvMode.NONE, LtvMode.ACTIVE], repeat=3))
LTV_MODE_NAMES = [f"q={m[0].name},k={m[1].name},v={m[2].name}" for m in LTV_MODES]

IMPLEMENTATIONS = [
    ("pytorch_loop", ltv_pytorch_loop),
    ("rust_cpu", LtvRustCpu.apply),
]
IMPL_IDS = [name for name, _ in IMPLEMENTATIONS]


def run_model(
    ltv_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    input_data,
    modes,
    *,
    requires_grad: bool,
    previous_values: LtvInitValues,
):
    # Ensure determinstic weights
    if requires_grad:
        torch.manual_seed(12345)
    else:
        # If not training, we can just use same seed but maybe better to control it per call
        torch.manual_seed(12345)

    q_mode, k_mode, v_mode = modes
    qkv_ltv = QkvLtv(query=q_mode, key=k_mode, value=v_mode)

    model = LtvFilter(
        n_head=input_data["n_head"],
        n_kv_head=input_data["n_kv_head"],
        head_dim=input_data["head_dim"],
        r=input_data["r"],
        qkv_ltv=qkv_ltv,
        ltv_fn=ltv_fn,
    ).to(DEVICE)

    # Prepare inputs
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


@pytest.mark.parametrize("ltv_fn_name, ltv_fn", IMPLEMENTATIONS, ids=IMPL_IDS)
@pytest.mark.parametrize("modes", LTV_MODES, ids=LTV_MODE_NAMES)
def test_ltv_filter_forward(input_data, ltv_fn_name, ltv_fn, modes):
    print(f"\nTesting Forward: {ltv_fn_name} | {modes}")

    # Baseline (PyTorch Loop)
    with torch.no_grad():
        out_loop, _ = run_model(
            ltv_pytorch_loop,
            input_data,
            modes,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    # Target
    with torch.no_grad():
        out_target, _ = run_model(
            ltv_fn,
            input_data,
            modes,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    assert len(out_loop) == len(out_target) == 3

    for i, (o_loop, o_target) in enumerate(zip(out_loop, out_target)):
        # Check NaNs
        assert not torch.isnan(o_target).any(), f"Output {i} contains NaNs"

        rtol = 1e-4 if "gpu" in ltv_fn_name else 1e-5
        atol = 1e-5

        torch.testing.assert_close(
            o_target, o_loop, rtol=rtol, atol=atol, msg=f"Mismatch in output {i}"
        )


@pytest.mark.parametrize("ltv_fn_name, ltv_fn", IMPLEMENTATIONS, ids=IMPL_IDS)
@pytest.mark.parametrize("modes", LTV_MODES, ids=LTV_MODE_NAMES)
def test_ltv_filter_backward(input_data, ltv_fn_name, ltv_fn, modes):
    print(f"\nTesting Backward: {ltv_fn_name} | {modes}")

    # We compare gradients against PyTorch Loop baseline

    # 1. Baseline
    out_loop_tuple, (model_loop, x_loop, q_loop, k_loop, v_loop) = run_model(
        ltv_pytorch_loop,
        input_data,
        modes,
        requires_grad=True,
        previous_values=LtvInitValues(None, None, None),
    )
    # Sum all outputs for loss. Since out_loop is (q,k,v), we sum all elements.
    loss_loop = sum(o.sum() for o in out_loop_tuple)
    loss_loop.backward()

    # 2. Target
    out_target_tuple, (model_target, x_target, q_target, k_target, v_target) = (
        run_model(
            ltv_fn,
            input_data,
            modes,
            requires_grad=True,
            previous_values=LtvInitValues(None, None, None),
        )
    )
    loss_target = sum(o.sum() for o in out_target_tuple)
    loss_target.backward()

    # Compare Gradients

    def check_grad(name, g_target, g_loop):
        if g_loop is None:
            assert g_target is None, f"{name} grad should be None"
        else:
            assert g_target is not None, f"{name} grad should not be None"
            # Slightly loose tolerance for backward pass accumulation
            rtol = 1e-3 if "gpu" in ltv_fn_name else 1e-4
            atol = 1e-4
            torch.testing.assert_close(
                g_target, g_loop, rtol=rtol, atol=atol, msg=f"{name} grad mismatch"
            )

    check_grad("x", x_target.grad, x_loop.grad)
    check_grad("q", q_target.grad, q_loop.grad)
    check_grad("k", k_target.grad, k_loop.grad)
    check_grad("v", v_target.grad, v_loop.grad)

    # Check model parameter gradients
    # Since we seed and init same model, params should align.
    for (n_loop, p_loop), (n_target, p_target) in zip(
        model_loop.named_parameters(), model_target.named_parameters(), strict=True
    ):
        assert n_loop == n_target
        # Check grad if it exists (some params might not be used depending on mode)
        check_grad(f"param {n_loop}", p_target.grad, p_loop.grad)


@pytest.mark.parametrize("ltv_fn_name, ltv_fn", IMPLEMENTATIONS, ids=IMPL_IDS)
@pytest.mark.parametrize("modes", LTV_MODES, ids=LTV_MODE_NAMES)
def test_ltv_filter_state_continuity(input_data, ltv_fn_name, ltv_fn, modes):
    """
    Test that running the filter in two chunks (passing state from 1st to 2nd)
    yields the same result as running it on the full sequence.
    """
    print(f"\nTesting Continuity: {ltv_fn_name} | {modes}")

    # 1. Run Full Sequence
    with torch.no_grad():
        out_full, _ = run_model(
            ltv_fn,
            input_data,
            modes,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    # 2. Split data in half
    T_split = input_data["q"].shape[1] // 2

    data_p1 = {
        k: v[:, :T_split] if v.ndim >= 2 else v
        for k, v in input_data.items()
        if isinstance(v, torch.Tensor)
    }
    # Update scalar config params manually since simple slicing won't work on the dict
    for k in ["n_head", "n_kv_head", "head_dim", "r"]:
        data_p1[k] = input_data[k]

    data_p2 = {
        k: v[:, T_split:] if v.ndim >= 2 else v
        for k, v in input_data.items()
        if isinstance(v, torch.Tensor)
    }
    for k in ["n_head", "n_kv_head", "head_dim", "r"]:
        data_p2[k] = input_data[k]

    with torch.no_grad():
        # Run Part 1
        out_p1, _ = run_model(
            ltv_fn,
            data_p1,
            modes,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

        # Extract state from Part 1
        q_p1, k_p1, v_p1 = out_p1
        # State is the last timestep: (B, T, H, D) -> (B, H, D)
        prev_vals = LtvInitValues(
            q=q_p1[:, -1, :, :].clone(),
            k=k_p1[:, -1, :, :].clone(),
            v=v_p1[:, -1, :, :].clone(),
        )

        # Run Part 2 with previous values
        out_p2, _ = run_model(
            ltv_fn, data_p2, modes, requires_grad=False, previous_values=prev_vals
        )

    # 3. Reconstruct and Compare
    for i, (full, p1, p2) in enumerate(zip(out_full, out_p1, out_p2)):
        # Concatenate parts
        reconstructed = torch.cat([p1, p2], dim=1)

        rtol = 1e-4 if "gpu" in ltv_fn_name else 1e-5
        atol = 1e-5

        torch.testing.assert_close(
            reconstructed,
            full,
            rtol=rtol,
            atol=atol,
            msg=f"Continuity mismatch in output {i}",
        )
