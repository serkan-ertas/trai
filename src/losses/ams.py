"""AMS multi-teacher self-distillation loss with RLC (paper Eqs. 7 & 8).

Eq. 7  w_{i,j} = softmax(h_{theta_j}(x'_i))[y_i]
Eq. 8  L_ams = (lambda / sum_j w_{i,j}) * sum_j w_{i,j} * KL( h_{theta_j}(x'_i) || h_{theta_t}(x'_i) )

The training loop iterates teachers one at a time (CPU-resident state_dicts swapped
into a scratch GPU model — see DECISIONS.md), accumulating per-teacher adversarial
logits into a list. This function takes that list and produces the scalar loss.
"""

import torch
import torch.nn.functional as F


def ams_distill_loss(
    student_logits_adv: torch.Tensor,
    teacher_logits_adv: list[torch.Tensor],
    y: torch.Tensor,
    lam: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the AMS multi-teacher distillation loss with RLC reweighting.

    Args:
        student_logits_adv: ``h_{theta_t}(x')`` logits, shape ``(N, C)``. Gradient flows.
        teacher_logits_adv: list of ``h_{theta_j}(x')`` logits (each ``(N, C)``) for
            ``j = 1..s-1``. Detached defensively; teachers do not receive gradients.
        y: Integer class labels, shape ``(N,)``.
        lam: Regularizer weight lambda in Eq. 8 (paper default 0.5).
        eps: Numerical stabilizer added to the per-sample weight sum denominator.

    Returns:
        ``(loss, metrics)`` where ``loss`` is the differentiable scalar L_ams and
        ``metrics`` has keys ``loss_ams``, ``mean_total_weight``, ``num_teachers``.
        When ``teacher_logits_adv`` is empty (early stages with no snapshots yet),
        returns a zero tensor on the student's device with ``num_teachers=0``.
    """
    if len(teacher_logits_adv) == 0:
        zero = student_logits_adv.new_zeros(())
        return zero, {"loss_ams": 0.0, "mean_total_weight": 0.0, "num_teachers": 0}

    log_p_student = F.log_softmax(student_logits_adv, dim=1)  # (N, C), grad flows

    # Per-teacher: softmax probabilities and the RLC weight on the TRUE class
    # (gotcha #5: true label, on the ADVERSARIAL example — not max class, not clean).
    teacher_probs: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for tl in teacher_logits_adv:
        tl_det = tl.detach()  # defensive — teachers are frozen snapshots
        p_t = F.softmax(tl_det, dim=1)
        teacher_probs.append(p_t)
        weights.append(p_t.gather(1, y.unsqueeze(1)).squeeze(1))  # Eq. 7, (N,)

    weights_stack = torch.stack(weights, dim=1)   # (N, T)
    weight_sum = weights_stack.sum(dim=1)         # (N,)

    # Per-teacher KL term: KL(p_teacher || p_student) — Eq. 8 direction (gotcha #3).
    # F.kl_div(input=log_p_student, target=p_teacher) = sum p_t*(log p_t - log p_s).
    kl_per_teacher: list[torch.Tensor] = []
    for p_t in teacher_probs:
        kl_nc = F.kl_div(log_p_student, p_t, reduction="none").sum(dim=1)  # (N,)
        kl_per_teacher.append(kl_nc)
    kl_stack = torch.stack(kl_per_teacher, dim=1)  # (N, T)

    weighted_kl_sum = (weights_stack * kl_stack).sum(dim=1)             # (N,)
    per_sample_loss = lam * weighted_kl_sum / (weight_sum + eps)        # (N,)
    loss_ams = per_sample_loss.mean()

    metrics = {
        "loss_ams": float(loss_ams.detach().item()),
        "mean_total_weight": float(weight_sum.mean().detach().item()),
        "num_teachers": len(teacher_logits_adv),
    }
    return loss_ams, metrics
