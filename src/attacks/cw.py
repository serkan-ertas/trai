"""ℓ∞ CW-PGD attack for adversarial evaluation (a.k.a. "CW-20").

PGD outer loop with the Carlini-Wagner margin loss instead of cross-entropy
(Carlini & Wagner 2017, untargeted, κ-confidence). Same ε-ball + [0,1]
projection and ``sign(grad)`` step rule as paper Eq. 2.
"""

import torch
import torch.nn as nn


def cw_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 20,
    kappa: float = 0.0,
    random_start: bool = True,
) -> torch.Tensor:
    """Run an ℓ∞ CW-PGD attack (CW margin loss + PGD optimization).

    The loss is the Carlini-Wagner margin: max(z_y - max_{i != y} z_i, -kappa).
    Minimizing this margin (equivalently, maximizing its negation) is the
    standard adversarial-eval objective. With kappa=0 the attack stops 'pushing'
    once the prediction has flipped; higher kappa demands higher confidence.

    Returns adversarial examples in [0,1].
    """
    was_training = model.training
    model.eval()
    try:
        if random_start:
            delta = torch.empty_like(x).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(x)
        delta.requires_grad_(True)

        delta.data = torch.clamp(x + delta, 0.0, 1.0).sub_(x)
        delta.data = torch.clamp(delta, -eps, eps)

        idx = torch.arange(x.size(0), device=x.device)
        for _ in range(steps):
            logits = model(x + delta)
            z_y = logits[idx, y]
            z_other = logits.clone()
            z_other[idx, y] = torch.finfo(logits.dtype).min  # AMP-safe -inf
            z_max_other = z_other.max(dim=1).values
            margin = z_y - z_max_other  # >0 ⇒ currently correct
            # Maximize negation of CW loss ⇒ keep the standard "+ alpha * sign" PGD rule.
            loss = -torch.clamp(margin, min=-kappa).mean()

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            delta.data = delta.data + alpha * grad.sign()  # GOTCHA #1: sign, not raw grad
            delta.data = torch.clamp(delta, -eps, eps)
            delta.data = torch.clamp(x + delta, 0.0, 1.0).sub_(x)

        return (x + delta).detach()
    finally:
        model.train(was_training)
