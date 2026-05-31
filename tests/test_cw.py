"""Dynamic-correctness tests for the ℓ∞ CW-PGD attack (task 1.4).

These tests verify:
1. Perturbation lies within the ℓ∞ ε-ball (gotcha: projection per step).
2. Output pixels are in [0,1] (gotcha #10: raw-pixel space).
3. The returned tensor is detached (no graph leak).
4. The attack actually lowers the true-class margin on a tiny clean-trained MLP
   (functional sanity — if CW does nothing, the implementation is broken).
5. The attack does not pollute the model parameters' gradients.
6. With ``kappa=0`` and pre-flipped predictions, the loss saturates at 0 and
   the gradient vanishes, so the attack should not move ``x`` beyond random init.
7. (Minimal) sanity behavior on dtype mismatches.
8. ``model.training`` is restored to its pre-attack value (try/finally pattern).
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attacks.cw import cw_attack


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TinyMLP(nn.Module):
    """Small MLP over flattened CIFAR-sized inputs. Trains fast on CPU/GPU."""

    def __init__(self, in_features: int = 3 * 32 * 32, hidden: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _train_clean_mlp(steps: int = 100, device: torch.device | None = None) -> tuple[_TinyMLP, torch.Tensor, torch.Tensor]:
    """Train a tiny MLP for `steps` SGD steps on a fixed random "dataset".

    Returns ``(model, x_eval, y_eval)`` where ``x_eval`` and ``y_eval`` are a
    held-out evaluation batch (drawn from the same distribution as the training
    batches but with a fresh seed). Sufficient for the margin-reduction sanity
    test — not an accuracy benchmark.
    """
    device = device or _device()
    torch.manual_seed(0)
    model = _TinyMLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    g = torch.Generator(device="cpu").manual_seed(123)
    for _ in range(steps):
        x = torch.rand(32, 3, 32, 32, generator=g).to(device)
        y = torch.randint(0, 10, (32,), generator=g).to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    g2 = torch.Generator(device="cpu").manual_seed(7777)
    x_eval = torch.rand(32, 3, 32, 32, generator=g2).to(device)
    y_eval = torch.randint(0, 10, (32,), generator=g2).to(device)
    return model, x_eval, y_eval


def _margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CW margin: z_y - max_{i != y} z_i (untargeted, per-sample)."""
    idx = torch.arange(logits.size(0), device=logits.device)
    z_y = logits[idx, y]
    z_other = logits.clone()
    z_other[idx, y] = torch.finfo(logits.dtype).min
    z_max_other = z_other.max(dim=1).values
    return z_y - z_max_other


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cw_perturbation_within_eps():
    """``(x_adv - x).abs().max() <= eps + 1e-6`` (per-step ε-projection)."""
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    eps = 8 / 255

    x_adv = cw_attack(model, x, y, eps=eps, alpha=2 / 255, steps=10)

    linf = (x_adv - x).abs().max().item()
    assert linf <= eps + 1e-6, f"linf norm of perturbation {linf:.6f} exceeds eps {eps:.6f}"


def test_cw_outputs_valid_pixels():
    """``x_adv`` is in [0, 1] (per-step pixel-projection)."""
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    x_adv = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10)

    assert (x_adv >= 0.0).all(), f"x_adv has negative pixels (min={x_adv.min().item()})"
    assert (x_adv <= 1.0).all(), f"x_adv has pixels >1 (max={x_adv.max().item()})"


def test_cw_returns_detached():
    """``x_adv.requires_grad == False`` and ``x_adv.grad_fn is None``."""
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    x_adv = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=5)

    assert x_adv.requires_grad is False, "x_adv must not require grad"
    assert x_adv.grad_fn is None, "x_adv has a grad_fn — graph leaked"


def test_cw_lowers_true_class_logit_gap():
    """A clean-trained tiny model should have its mean true-class margin reduced by CW.

    Train the MLP for ~100 steps so initial margins are positive (model is
    confidently making *some* prediction on each sample, even if not the true
    label). Then CW should drive margin down on average.
    """
    device = _device()
    model, x, y = _train_clean_mlp(steps=100, device=device)
    model.eval()

    with torch.no_grad():
        margin_clean = _margin(model(x), y)

    x_adv = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=20)
    with torch.no_grad():
        margin_adv = _margin(model(x_adv), y)

    assert margin_adv.mean().item() < margin_clean.mean().item(), (
        f"CW attack did not reduce mean margin: "
        f"clean={margin_clean.mean().item():.4f}, adv={margin_adv.mean().item():.4f}"
    )


def test_cw_does_not_pollute_model_grad():
    """Running CW must not leave gradients on any model parameter.

    The attack uses ``torch.autograd.grad(loss, delta, only_inputs=True)`` so
    no model parameter ``.grad`` field should be populated. We zero gradients
    beforehand and assert each param's grad is either None or an all-zero tensor.
    """
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device).eval()

    for p in model.parameters():
        p.grad = None

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    _ = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=5)

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0), (
                f"param {name} accumulated a non-zero gradient through CW attack "
                f"(max abs={p.grad.abs().max().item():.6g})"
            )


def test_cw_kappa_zero_stops_on_flip():
    """With ``kappa=0`` and pre-flipped predictions, gradient vanishes -> attack stays at init.

    Construct a 2-class linear model with hard-coded weights such that the
    prediction is wrong (argmax != y) for every sample. Then ``margin < 0``,
    ``clamp(margin, min=-kappa=0)`` returns 0 for all samples, the mean loss is
    a constant 0, and the gradient w.r.t. ``delta`` is exactly 0. With
    ``random_start=False`` the attack should therefore return ``x`` byte-for-byte.
    """
    device = _device()
    torch.manual_seed(0)

    # 2-class linear model on flattened inputs. Weights/biases hard-coded so
    # that class-1 logit is always strictly greater than class-0 logit for any
    # input in [0,1]^d. We then label every sample as class 0 -> guaranteed
    # margin = z_0 - z_1 < 0, so kappa=0 saturates the loss to 0.
    in_features = 3 * 32 * 32
    linear = nn.Linear(in_features, 2, bias=True).to(device)
    with torch.no_grad():
        linear.weight.zero_()  # all zero — logit depends purely on bias
        linear.bias.copy_(torch.tensor([0.0, 10.0], device=device))  # class-1 always wins

    class _LinearModel(nn.Module):
        def __init__(self, head: nn.Linear) -> None:
            super().__init__()
            self.head = head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x.flatten(1))

    model = _LinearModel(linear).eval()

    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.zeros(2, dtype=torch.long, device=device)  # label every sample as class 0

    # Sanity: the model is indeed wrong on every sample.
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    assert (preds != y).all(), "test setup failure: predictions should all be flipped"

    x_adv = cw_attack(
        model, x, y, eps=8 / 255, alpha=2 / 255, steps=5, kappa=0.0, random_start=False
    )

    assert torch.equal(x_adv, x), (
        "with kappa=0 and pre-flipped predictions and random_start=False, "
        "the attack should not move x at all "
        f"(max abs diff = {(x_adv - x).abs().max().item():.6g})"
    )


def test_cw_dtype_y_long_required():
    """Minimal check that label tensor of dtype long works as expected."""
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device).eval()

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device, dtype=torch.long)

    # Should run without error and return same-shape tensor.
    x_adv = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=3)
    assert x_adv.shape == x.shape


def test_cw_eval_mode_restored():
    """``model.training`` is True after attack if it was True before (try/finally)."""
    torch.manual_seed(0)
    device = _device()
    model = _TinyMLP().to(device)
    model.train()
    assert model.training is True  # sanity

    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    _ = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=3)

    assert model.training is True, (
        "model.training was not restored to True after cw_attack — "
        "the try/finally restore block is broken"
    )

    # And the symmetric case: starting in eval(), the attack should also leave
    # the model in eval() mode.
    model.eval()
    _ = cw_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=3)
    assert model.training is False, "model.training was not restored to False"
