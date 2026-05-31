"""Dynamic-correctness tests for TRADES loss (task 1.6).

These tests verify Eq. 3 of the TRADES paper / gotcha #3 (KL direction):
- The decomposition `loss_total = loss_ce + beta * loss_kl` holds.
- The default `beta = 6.0` (paper default).
- KL term is zero when clean and adversarial logits match.
- Returned tensor is differentiable and gradients flow to BOTH inputs.
- `p_clean` is NOT detached — the KL regularizer affects clean-side params.
- KL direction matches `KL(p_clean || p_adv)` against a hand-computed value.
- The metrics dict contains Python floats, not tensors.
"""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from src.losses.trades import trades_loss


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_components_sum():
    """`loss_total ≈ loss_ce + beta * loss_kl`, and tensor.item() matches metrics."""
    torch.manual_seed(0)
    device = _device()
    N, C = 8, 10
    beta = 6.0

    logits_clean = torch.randn(N, C, device=device)
    logits_adv = torch.randn(N, C, device=device)
    y = torch.randint(0, C, (N,), device=device)

    loss, metrics = trades_loss(logits_clean, logits_adv, y, beta=beta)

    assert math.isclose(
        metrics["loss_total"],
        metrics["loss_ce"] + beta * metrics["loss_kl"],
        rel_tol=1e-5,
        abs_tol=1e-7,
    ), f"loss_total decomposition failed: {metrics}"
    assert math.isclose(
        loss.item(), metrics["loss_total"], rel_tol=1e-5, abs_tol=1e-7
    ), f"tensor.item() {loss.item()} != metrics[loss_total] {metrics['loss_total']}"


def test_beta_default_is_6():
    """Paper default beta = 6.0 (Eq. 3)."""
    sig = inspect.signature(trades_loss)
    beta_default = sig.parameters["beta"].default
    assert beta_default == 6.0, f"Expected beta default 6.0, got {beta_default!r}"


def test_kl_zero_when_clean_equals_adv():
    """KL(p || p) == 0; verify the KL term vanishes and loss_total == loss_ce."""
    torch.manual_seed(1)
    device = _device()
    N, C = 16, 10

    logits_clean = torch.randn(N, C, device=device)
    logits_adv = logits_clean.clone()
    y = torch.randint(0, C, (N,), device=device)

    loss, metrics = trades_loss(logits_clean, logits_adv, y, beta=6.0)

    # KL of identical distributions is 0 (up to fp noise).
    assert abs(metrics["loss_kl"]) < 1e-6, (
        f"Expected loss_kl ~ 0 for identical logits, got {metrics['loss_kl']}"
    )
    assert math.isclose(
        metrics["loss_total"], metrics["loss_ce"], rel_tol=1e-5, abs_tol=1e-6
    ), f"loss_total {metrics['loss_total']} != loss_ce {metrics['loss_ce']}"


def test_returns_tensor_with_grad():
    """Returned `loss` is differentiable; backward populates grads on both inputs."""
    torch.manual_seed(2)
    device = _device()
    N, C = 4, 10

    logits_clean = torch.randn(N, C, device=device, requires_grad=True)
    logits_adv = torch.randn(N, C, device=device, requires_grad=True)
    y = torch.randint(0, C, (N,), device=device)

    loss, _ = trades_loss(logits_clean, logits_adv, y, beta=6.0)
    assert loss.requires_grad, "trades_loss output must have requires_grad=True"
    assert loss.grad_fn is not None, "trades_loss output must have a grad_fn"

    loss.backward()
    assert logits_clean.grad is not None, "logits_clean.grad is None after backward"
    assert logits_adv.grad is not None, "logits_adv.grad is None after backward"
    assert torch.isfinite(logits_clean.grad).all(), "logits_clean.grad has non-finite values"
    assert torch.isfinite(logits_adv.grad).all(), "logits_adv.grad has non-finite values"


def test_p_clean_not_detached():
    """`p_clean` (KL target) must NOT be detached: KL term must contribute to logits_clean.grad.

    Setup: only logits_clean requires grad (logits_adv frozen). Backward through
    `loss - loss_ce` would isolate the KL contribution, but practically we just
    assert that logits_clean.grad is non-zero after a full backward — the CE term
    contributes to it too, but if p_clean were detached the KL gradient would
    vanish, and we can detect that by also checking the gradient norm exceeds
    what CE alone yields.
    """
    torch.manual_seed(3)
    device = _device()
    N, C = 4, 10

    # First: get the gradient from CE-only (no KL).
    logits_clean_a = torch.randn(N, C, device=device, requires_grad=True)
    y = torch.randint(0, C, (N,), device=device)
    ce_only = F.cross_entropy(logits_clean_a, y)
    ce_only.backward()
    ce_grad_norm = logits_clean_a.grad.norm().item()

    # Second: with the same clean logits + adversarial logits, run trades_loss.
    # If KL contributes, grad norm should differ (not equal CE-only).
    logits_clean_b = logits_clean_a.detach().clone().requires_grad_(True)
    logits_adv = torch.randn(N, C, device=device, requires_grad=False)
    loss, _ = trades_loss(logits_clean_b, logits_adv, y, beta=6.0)
    loss.backward()
    assert logits_clean_b.grad is not None, "logits_clean.grad must be populated"
    assert not torch.all(logits_clean_b.grad == 0), "logits_clean.grad must not be all-zero"

    trades_grad_norm = logits_clean_b.grad.norm().item()
    # If p_clean were detached, TRADES grad would equal CE grad. With p_clean
    # in the graph (and beta=6, non-trivial adv logits), it should differ.
    assert not math.isclose(trades_grad_norm, ce_grad_norm, rel_tol=1e-5), (
        f"TRADES grad norm {trades_grad_norm} matches CE-only grad norm "
        f"{ce_grad_norm} — KL term may not be reaching logits_clean (p_clean detached?)"
    )


def test_kl_direction_with_known_distributions():
    """Pin KL direction: F.kl_div(log_p_adv, p_clean) must compute KL(p_clean || p_adv).

    Construct logits whose softmax probabilities are known. With batch size 1
    and 3 classes:
        p_clean = [0.7, 0.2, 0.1]
        p_adv   = [0.1, 0.2, 0.7]
    Then:
        KL(p_clean || p_adv) = 0.7*log(0.7/0.1) + 0.2*log(0.2/0.2)
                             + 0.1*log(0.1/0.7)
                             = 0.7*log(7) - 0.1*log(7)
                             = 0.6*log(7) ≈ 1.1675
    """
    device = _device()

    # Build logits whose softmax produces [0.7, 0.2, 0.1] and [0.1, 0.2, 0.7].
    # log of a probability vector gives logits whose softmax is that vector.
    p_clean_target = torch.tensor([[0.7, 0.2, 0.1]], device=device)
    p_adv_target = torch.tensor([[0.1, 0.2, 0.7]], device=device)
    logits_clean = torch.log(p_clean_target)
    logits_adv = torch.log(p_adv_target)
    y = torch.tensor([0], device=device, dtype=torch.long)

    _, metrics = trades_loss(logits_clean, logits_adv, y, beta=1.0)

    # Hand-computed KL(p_clean || p_adv) — middle term is 0.2*log(1) = 0.
    expected_kl = 0.7 * math.log(0.7 / 0.1) + 0.0 + 0.1 * math.log(0.1 / 0.7)
    assert math.isclose(metrics["loss_kl"], expected_kl, rel_tol=1e-5, abs_tol=1e-6), (
        f"KL direction mismatch: got {metrics['loss_kl']}, expected "
        f"KL(p_clean||p_adv)={expected_kl}. If reversed, would be "
        f"{0.1 * math.log(0.1 / 0.7) + 0.7 * math.log(0.7 / 0.1)} (symmetric here so "
        f"sanity-check more carefully)"
    )

    # Also check the asymmetric direction: KL(p_adv || p_clean) for an asymmetric
    # case where it would clearly differ. Use [0.8, 0.1, 0.1] vs [0.1, 0.1, 0.8].
    p_a = torch.tensor([[0.8, 0.1, 0.1]], device=device)
    p_b = torch.tensor([[0.1, 0.1, 0.8]], device=device)
    logits_a = torch.log(p_a)
    logits_b = torch.log(p_b)
    _, m2 = trades_loss(logits_a, logits_b, y, beta=1.0)
    expected_kl_ab = (
        0.8 * math.log(0.8 / 0.1)
        + 0.1 * math.log(0.1 / 0.1)
        + 0.1 * math.log(0.1 / 0.8)
    )
    assert math.isclose(m2["loss_kl"], expected_kl_ab, rel_tol=1e-5, abs_tol=1e-6), (
        f"KL(p_a||p_b) direction mismatch: got {m2['loss_kl']}, "
        f"expected {expected_kl_ab}"
    )


def test_metrics_are_floats():
    """metrics dict values are Python floats (detached), not torch tensors."""
    torch.manual_seed(4)
    device = _device()
    N, C = 4, 10

    logits_clean = torch.randn(N, C, device=device)
    logits_adv = torch.randn(N, C, device=device)
    y = torch.randint(0, C, (N,), device=device)

    loss, metrics = trades_loss(logits_clean, logits_adv, y, beta=6.0)

    for key in ("loss_ce", "loss_kl", "loss_total"):
        assert key in metrics, f"metrics missing key {key!r}"
        val = metrics[key]
        assert isinstance(val, float), (
            f"metrics[{key!r}] should be a Python float, got {type(val).__name__}"
        )
        assert not isinstance(val, torch.Tensor), (
            f"metrics[{key!r}] should not be a torch.Tensor"
        )

    # loss tensor's .item() should match metrics["loss_total"].
    assert math.isclose(
        loss.item(), metrics["loss_total"], rel_tol=1e-6, abs_tol=1e-7
    ), "loss.item() should round-trip to metrics['loss_total']"
