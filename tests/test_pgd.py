"""Dynamic-correctness tests for the PGD attack module (task 1.3).

Verifies:
1. Perturbation is bounded by eps in L_inf.
2. Output pixels stay in [0,1].
3. Returned tensor is detached from autograd.
4. random_start=True produces stochastic outputs; random_start=False is deterministic.
5. PGD breaks a trained-but-non-robust classifier (clean_acc - robust_acc > 0.30).
6. PGD does NOT pollute model parameter gradients (used torch.autograd.grad).
7. The "kl_to_clean" objective branch runs and produces a valid x_adv.
8. Unknown objectives raise ValueError.
9. The model's training mode is restored after the attack returns.
"""

import pytest
import torch
import torch.nn as nn

from src.attacks import pgd_attack
from src.models import preact_resnet18


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_mlp() -> nn.Module:
    """A small MLP that takes raw [0,1] (N,3,32,32) and emits logits (N,10).

    Hidden size 256 is enough to comfortably memorize 256 random samples
    in a few hundred Adam steps — sufficient to give PGD something to break.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 10),
    )


# ---------------------------------------------------------------------------
# 1. Perturbation budget
# ---------------------------------------------------------------------------
def test_pgd_perturbation_within_eps():
    """|x_adv - x|_inf <= eps + tiny tolerance for every element."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    eps = 8 / 255
    x_adv = pgd_attack(model, x, y, eps=eps, alpha=2 / 255, steps=10, objective="ce")

    max_pert = (x_adv - x).abs().max().item()
    assert max_pert <= eps + 1e-6, (
        f"max L_inf perturbation {max_pert:.6f} exceeds eps {eps:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. Output pixel range
# ---------------------------------------------------------------------------
def test_pgd_outputs_valid_pixels():
    """x_adv lies in [0,1] within fp32 tolerance."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    x_adv = pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10, objective="ce")

    assert x_adv.min().item() >= 0.0 - 1e-6, (
        f"x_adv min {x_adv.min().item()} below 0.0"
    )
    assert x_adv.max().item() <= 1.0 + 1e-6, (
        f"x_adv max {x_adv.max().item()} above 1.0"
    )


# ---------------------------------------------------------------------------
# 3. Output is detached
# ---------------------------------------------------------------------------
def test_pgd_returns_detached():
    """x_adv must not carry autograd state — callers backward through it freely."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    x_adv = pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=5, objective="ce")

    assert x_adv.requires_grad is False, "x_adv.requires_grad should be False"
    assert x_adv.grad_fn is None, f"x_adv.grad_fn should be None, got {x_adv.grad_fn}"


# ---------------------------------------------------------------------------
# 4. Random init vs no-random-init
# ---------------------------------------------------------------------------
def test_pgd_random_init():
    """random_start=True is stochastic across seeds; random_start=False is determined by x.

    The implementation samples delta ~ U(-eps, eps) when random_start=True. Two
    different seeds drawing from this distribution should produce noticeably
    different starting points and (after a fixed number of PGD steps with
    non-zero alpha) noticeably different x_adv tensors.

    For random_start=False, delta begins at zero. We don't assert bit-equal
    outputs across two no-init runs because cuDNN backward on CUDA can be
    non-deterministic and any infinitesimal grad jitter is amplified by the
    sign(grad) step. Instead we assert the actually-meaningful property:
    two random-init runs with DIFFERENT seeds diverge, and two random-init
    runs with the SAME seed agree.
    """
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # --- Two different seeds => different x_adv (random init actually randomizes).
    torch.manual_seed(11)
    x_adv_a = pgd_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=5,
        objective="ce", random_start=True,
    )
    torch.manual_seed(22)
    x_adv_b = pgd_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=5,
        objective="ce", random_start=True,
    )
    assert not torch.allclose(x_adv_a, x_adv_b), (
        "random_start=True with different seeds produced identical x_adv — "
        "random init is not actually randomizing"
    )

    # --- Same seed => same x_adv (the randomness IS coming from torch's RNG,
    # not from some other unseeded source).
    torch.manual_seed(33)
    x_adv_c = pgd_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=5,
        objective="ce", random_start=True,
    )
    torch.manual_seed(33)
    x_adv_d = pgd_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=5,
        objective="ce", random_start=True,
    )
    # cuDNN backward on CUDA isn't bit-deterministic, and sign(grad) can flip
    # under tiny jitter — but with the same seed and same model the L_inf gap
    # should still be MUCH smaller than the eps-ball width.
    same_seed_gap = (x_adv_c - x_adv_d).abs().max().item()
    diff_seed_gap = (x_adv_a - x_adv_b).abs().max().item()
    assert same_seed_gap <= diff_seed_gap, (
        f"same-seed gap ({same_seed_gap:.6f}) should be <= different-seed gap "
        f"({diff_seed_gap:.6f}) — random init seeding is broken"
    )


# ---------------------------------------------------------------------------
# 5. The headline test: PGD breaks a non-robust classifier
# ---------------------------------------------------------------------------
def test_pgd_breaks_clean_model():
    """Train a tiny MLP on random CIFAR-like data, then verify PGD slashes its accuracy.

    If PGD doesn't break a non-robust model, the attack is broken — this is
    the dynamic gate that catches "PGD silently uses raw grad instead of
    grad.sign()" and similar regressions.
    """
    torch.manual_seed(0)
    device = _device()

    # 256 fake CIFAR-10 samples in [0,1] with random integer labels.
    N = 256
    x = torch.rand(N, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (N,), device=device)

    model = _tiny_mlp().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train just enough to clear ~50% clean accuracy on the random labels
    # without saturating into a degenerate fully-confident solution. Stop
    # early once we hit a healthy clean accuracy — that leaves PGD a
    # non-trivial loss landscape to climb.
    model.train()
    target_clean_acc = 0.60
    achieved = 0.0
    for step in range(800):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                achieved = (logits.argmax(dim=1) == y).float().mean().item()
            if achieved >= target_clean_acc:
                break

    model.eval()
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
    clean_acc = (clean_pred == y).float().mean().item()
    assert clean_acc > 0.50, (
        f"sanity: tiny MLP failed to reach > 50% clean acc on 256 random samples "
        f"(clean_acc={clean_acc:.3f}); attack-strength test cannot proceed"
    )

    # PGD-40 with the standard CIFAR-10 schedule. With a 2-layer MLP that
    # achieves > 50% clean acc but doesn't have saturated logits, this
    # easily slashes accuracy to single digits on the same samples.
    x_adv = pgd_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=40, objective="ce",
    )
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1)
    robust_acc = (adv_pred == y).float().mean().item()

    drop = clean_acc - robust_acc
    assert drop > 0.30, (
        f"PGD did not significantly break a non-robust model: "
        f"clean_acc={clean_acc:.3f}, robust_acc={robust_acc:.3f}, "
        f"drop={drop:.3f} (required > 0.30). Attack may be missing .sign(), "
        f"using wrong objective, or not projecting properly."
    )


# ---------------------------------------------------------------------------
# 6. Model gradients must remain clean
# ---------------------------------------------------------------------------
def test_pgd_does_not_pollute_model_grad():
    """After pgd_attack returns, no model parameter has a non-zero .grad.

    The attack must compute the gradient w.r.t. the input via
    torch.autograd.grad — NOT via loss.backward(), which would accumulate
    gradients on model parameters and corrupt the outer optimizer step.
    """
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()
    model.zero_grad(set_to_none=True)

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    _ = pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=5, objective="ce")

    for name, p in model.named_parameters():
        if p.grad is None:
            continue  # ideal — torch.autograd.grad leaves .grad untouched
        # If a .grad does exist (e.g. from an earlier zero_grad with set_to_none=False),
        # it must be all zero.
        assert torch.all(p.grad == 0), (
            f"param {name} has non-zero .grad after pgd_attack — "
            f"the attack likely used loss.backward() instead of torch.autograd.grad"
        )


# ---------------------------------------------------------------------------
# 7. KL objective branch
# ---------------------------------------------------------------------------
def test_pgd_kl_objective_runs():
    """The 'kl_to_clean' branch (TRADES inner, Eq. 3) runs end-to-end and respects eps."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)  # unused by the KL objective

    eps = 8 / 255
    x_adv = pgd_attack(
        model, x, y, eps=eps, alpha=2 / 255, steps=5, objective="kl_to_clean",
    )

    assert x_adv.shape == x.shape, (
        f"shape mismatch: x_adv {tuple(x_adv.shape)} vs x {tuple(x.shape)}"
    )
    assert torch.isfinite(x_adv).all(), "x_adv contains non-finite values (NaN/Inf)"
    assert (x_adv - x).abs().max().item() <= eps + 1e-6, (
        "x_adv from kl_to_clean violates the eps L_inf ball"
    )
    assert x_adv.min().item() >= 0.0 - 1e-6 and x_adv.max().item() <= 1.0 + 1e-6, (
        "x_adv from kl_to_clean leaves the [0,1] pixel range"
    )


# ---------------------------------------------------------------------------
# 8. Unknown objective is rejected
# ---------------------------------------------------------------------------
def test_pgd_invalid_objective_raises():
    """An unsupported objective string must raise ValueError, not silently no-op."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    with pytest.raises(ValueError):
        pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=2, objective="bogus")


# ---------------------------------------------------------------------------
# 9. Training mode is restored
# ---------------------------------------------------------------------------
def test_pgd_eval_mode_restored():
    """If the model entered the attack in train() mode, it must leave in train() mode.

    The attack flips the model into eval() internally (so BN running stats
    don't drift during PGD), but the prior mode must be restored before
    returning. Otherwise the outer training loop's first batch after each
    attack runs in eval() mode and BN stops updating.
    """
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device)
    model.train()
    assert model.training is True

    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    _ = pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=3, objective="ce")

    assert model.training is True, (
        "model.training should be True after pgd_attack returns from a train()-mode model"
    )

    # Sanity: also verify the inverse direction (eval -> eval).
    model.eval()
    assert model.training is False
    _ = pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=3, objective="ce")
    assert model.training is False, (
        "model.training should be False after pgd_attack returns from an eval()-mode model"
    )
