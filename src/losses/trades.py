"""TRADES outer-loop loss (Zhang et al. 2019; paper Eq. 3).

Formula:
    L = CE(logits_clean, y) + beta * KL( softmax(logits_clean) || softmax(logits_adv) )

The KL regularizer matches the inner attack's objective (`pgd_attack`,
``objective="kl_to_clean"``), so the inner max and outer min agree on direction.
Gradient flows through both ``logits_clean`` (CE + KL target) and
``logits_adv`` (KL input) — neither side is detached.
"""

import torch
import torch.nn.functional as F


def trades_loss(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    y: torch.Tensor,
    beta: float = 6.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the TRADES outer-loop loss (paper Eq. 3).

    Args:
        logits_clean: ``model(x)`` logits, shape ``(N, C)``.
        logits_adv: ``model(x')`` logits, shape ``(N, C)``.
        y: Integer class labels, shape ``(N,)``.
        beta: KL regularizer weight (paper default 6).

    Returns:
        Tuple ``(loss, metrics)`` where ``loss`` is the differentiable scalar
        ``CE + beta * KL`` and ``metrics`` carries detached floats
        ``loss_ce``, ``loss_kl``, ``loss_total`` for logging.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    log_p_adv = F.log_softmax(logits_adv, dim=1)

    loss_ce = F.cross_entropy(logits_clean, y)
    # F.kl_div(input=log_p_adv, target=p_clean) computes KL(p_clean || p_adv) (Eq. 3).
    loss_kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")

    loss_total = loss_ce + beta * loss_kl

    metrics = {
        "loss_ce": loss_ce.detach().item(),
        "loss_kl": loss_kl.detach().item(),
        "loss_total": loss_total.detach().item(),
    }
    return loss_total, metrics
