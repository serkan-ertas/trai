"""ℓ∞ PGD attack for adversarial training and evaluation.

Supports two attack objectives on raw [0,1] pixel inputs:

- ``"ce"``: maximize ``CrossEntropy(model(x'), y)``. Used for the AMS outer
  adversarial loss (paper Eq. 6) and for PGD-K evaluation (e.g. PGD-20).
- ``"kl_to_clean"``: maximize ``KL(softmax(model(x).detach()) || softmax(model(x')))``,
  the TRADES inner maximization (paper Eq. 3). The clean-side distribution is
  computed once with ``no_grad()`` before the loop and frozen.

Both objectives use ``sign(grad)`` PGD steps (paper Eq. 2) and project to the
ε-ball and the [0,1] pixel range at every iteration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    objective: str = "ce",
    random_start: bool = True,
) -> torch.Tensor:
    """Run an ℓ∞ PGD attack. Returns adversarial examples in [0,1].

    Args:
        model: Classifier that accepts raw [0,1] inputs and returns logits.
        x: Clean inputs in [0,1], shape ``(N, 3, H, W)``.
        y: Integer class labels, shape ``(N,)``.
        eps: ℓ∞ perturbation budget in pixel units.
        alpha: Per-step size in pixel units.
        steps: Number of PGD iterations.
        objective: ``"ce"`` (Eq. 6 / evaluation) or ``"kl_to_clean"`` (Eq. 3 inner).
        random_start: If True, init δ ~ U(-eps, eps); else δ = 0.

    Returns:
        Adversarial examples ``x + δ`` clamped to [0,1] and to the ε-ball, detached.
    """
    was_training = model.training
    model.eval()
    try:
        if objective == "kl_to_clean":
            with torch.no_grad():
                p_clean = F.softmax(model(x), dim=1)
        else:
            p_clean = None

        if random_start:
            delta = torch.empty_like(x).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(x)
        delta.requires_grad_(True)

        delta.data = torch.clamp(x + delta, 0.0, 1.0).sub_(x)
        delta.data = torch.clamp(delta, -eps, eps)

        for _ in range(steps):
            x_adv = x + delta
            logits = model(x_adv)
            if objective == "ce":
                loss = F.cross_entropy(logits, y)
            elif objective == "kl_to_clean":
                log_p_adv = F.log_softmax(logits, dim=1)
                # KL(p_clean || p_adv): F.kl_div(input=log_p_adv, target=p_clean)
                # computes sum target * (log target - input) = KL(target || exp(input)).
                loss = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
            else:
                raise ValueError(f"unknown objective: {objective}")

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            delta.data = delta.data + alpha * grad.sign()  # GOTCHA #1: sign, not raw grad
            delta.data = torch.clamp(delta, -eps, eps)
            delta.data = torch.clamp(x + delta, 0.0, 1.0).sub_(x)

        return (x + delta).detach()
    finally:
        model.train(was_training)
