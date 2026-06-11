import pytest
import torch

from nanochat.ltv.ltv_filter import (
    LtvFilter,
    LtvSplitFilter,
    QkvLtv,
    QkvSplitLtv,
    LtvInitValues,
)
from nanochat.ltv.ltv_mode import LtvMode
from nanochat.ltv.ltv_impls import ltv_pytorch_loop


def get_device():
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
        "r": 2,
    }


def build_models(input_data):
    qkv_ltv = QkvLtv(query=LtvMode.ACTIVE, key=LtvMode.ACTIVE, value=LtvMode.ACTIVE)
    split_ltv = QkvSplitLtv(
        query=ltv_pytorch_loop, key=ltv_pytorch_loop, value=ltv_pytorch_loop
    )

    base = LtvFilter(
        n_head=input_data["n_head"],
        n_kv_head=input_data["n_kv_head"],
        head_dim=input_data["head_dim"],
        r=input_data["r"],
        qkv_ltv=qkv_ltv,
        ltv_fn=ltv_pytorch_loop,
    ).to(DEVICE)

    split = LtvSplitFilter(
        n_head=input_data["n_head"],
        n_kv_head=input_data["n_kv_head"],
        head_dim=input_data["head_dim"],
        r=input_data["r"],
        qkv_split_ltv=split_ltv,
    ).to(DEVICE)

    split.load_state_dict(base.state_dict(), strict=True)
    return base, split


def run_model(
    model,
    input_data,
    *,
    requires_grad: bool,
    previous_values: LtvInitValues,
):
    if requires_grad:
        torch.manual_seed(12345)

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


def check_grad(name, g_target, g_base):
    if g_base is None:
        assert g_target is None, f"{name} grad should be None"
    else:
        assert g_target is not None, f"{name} grad should not be None"
        torch.testing.assert_close(
            g_target, g_base, rtol=1e-4, atol=1e-4, msg=f"{name} grad mismatch"
        )


def test_ltv_split_filter_forward_matches_base(input_data):
    print("\nTesting LtvSplitFilter forward matches LtvFilter")

    base, split = build_models(input_data)

    with torch.no_grad():
        out_base, _ = run_model(
            base,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )
        out_split, _ = run_model(
            split,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    assert len(out_base) == len(out_split) == 3
    for i, (base_out, split_out) in enumerate(zip(out_base, out_split)):
        assert not torch.isnan(split_out).any(), f"Output {i} contains NaNs"
        torch.testing.assert_close(
            split_out, base_out, rtol=1e-5, atol=1e-5, msg=f"Output {i} mismatch"
        )


def test_ltv_split_filter_backward_matches_base(input_data):
    print("\nTesting LtvSplitFilter backward matches LtvFilter")

    base, split = build_models(input_data)

    out_base, (base_model, x_base, q_base, k_base, v_base) = run_model(
        base,
        input_data,
        requires_grad=True,
        previous_values=LtvInitValues(None, None, None),
    )
    loss_base = sum(o.sum() for o in out_base)
    loss_base.backward()

    out_split, (split_model, x_split, q_split, k_split, v_split) = run_model(
        split,
        input_data,
        requires_grad=True,
        previous_values=LtvInitValues(None, None, None),
    )
    loss_split = sum(o.sum() for o in out_split)
    loss_split.backward()

    check_grad("x", x_split.grad, x_base.grad)
    check_grad("q", q_split.grad, q_base.grad)
    check_grad("k", k_split.grad, k_base.grad)
    check_grad("v", v_split.grad, v_base.grad)

    for (n_base, p_base), (n_split, p_split) in zip(
        base_model.named_parameters(), split_model.named_parameters(), strict=True
    ):
        assert n_base == n_split
        check_grad(f"param {n_base}", p_split.grad, p_base.grad)


def test_ltv_split_filter_state_continuity(input_data):
    print("\nTesting LtvSplitFilter state continuity")

    _, split = build_models(input_data)

    with torch.no_grad():
        out_full, _ = run_model(
            split,
            input_data,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

    T_split = input_data["q"].shape[1] // 2
    data_p1 = {
        k: v[:, :T_split] if v.ndim >= 2 else v
        for k, v in input_data.items()
        if isinstance(v, torch.Tensor)
    }
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
        out_p1, _ = run_model(
            split,
            data_p1,
            requires_grad=False,
            previous_values=LtvInitValues(None, None, None),
        )

        q_p1, k_p1, v_p1 = out_p1
        prev_vals = LtvInitValues(
            q=q_p1[:, -1, :, :].clone(),
            k=k_p1[:, -1, :, :].clone(),
            v=v_p1[:, -1, :, :].clone(),
        )

        out_p2, _ = run_model(
            split, data_p2, requires_grad=False, previous_values=prev_vals
        )

    for i, (full, p1, p2) in enumerate(zip(out_full, out_p1, out_p2)):
        reconstructed = torch.cat([p1, p2], dim=1)
        torch.testing.assert_close(
            reconstructed, full, rtol=1e-5, atol=1e-5, msg=f"Output {i} mismatch"
        )
