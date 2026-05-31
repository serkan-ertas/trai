"""Dynamic-correctness tests for the AMS multi-teacher distillation loss (task 1.7).

Verifies:
1. Empty teacher list returns a zero scalar on the student's device and num_teachers=0.
2. RLC weight equals the teacher's softmax probability on the TRUE class (Eq. 7,
   gotcha #5).
3. A high-confidence teacher dominates over a low-confidence one (RLC re-weighting
   actually changes the loss in the expected direction).
4. No gradient flows backward into teacher parameters (gotcha #4).
5. Gradient flows to the student logits.
6. KL direction and RLC normalizer combined: a hand-computed scalar matches the
   loss to numerical tolerance (pins gotchas #3 and #5 simultaneously).
7. Default lambda is 0.5 (paper default).
8. metrics["num_teachers"] matches len(teacher_logits_adv).
"""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from src.losses.ams import ams_distill_loss
from src.models import preact_resnet18


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Empty teacher list -> zero scalar on student's device
# ---------------------------------------------------------------------------
def test_zero_teachers_returns_zero():
    device = _device()
    B, C = 4, 10
    student_logits = torch.randn(B, C, device=device)
    y = torch.randint(0, C, (B,), device=device)

    loss, metrics = ams_distill_loss(student_logits, [], y)

    assert loss.item() == 0.0
    assert metrics["num_teachers"] == 0
    assert loss.device == student_logits.device


# ---------------------------------------------------------------------------
# 2. mean_total_weight equals mean of teacher's softmax on the TRUE class (Eq. 7)
# ---------------------------------------------------------------------------
def test_rlc_weight_is_true_class_prob():
    device = _device()
    torch.manual_seed(0)
    B, C = 4, 10

    # Build a teacher whose softmax we control directly. Put a known logit on the
    # true-class column for each sample; the rest are zeros, so softmax is
    #   exp(v) / (exp(v) + (C-1) * exp(0))  for the true column
    #     1 / (exp(v) + (C-1))              for any non-true column.
    y = torch.tensor([0, 3, 7, 9], device=device)
    teacher_logits = torch.zeros(B, C, device=device)
    true_logit_values = torch.tensor([2.0, 1.0, 3.0, 0.5], device=device)
    for i in range(B):
        teacher_logits[i, y[i]] = true_logit_values[i]

    student_logits = torch.randn(B, C, device=device)

    # Manual ground truth for the per-sample true-class probability.
    expected_w = F.softmax(teacher_logits, dim=1).gather(
        1, y.unsqueeze(1)
    ).squeeze(1)
    expected_mean = expected_w.mean().item()

    _, metrics = ams_distill_loss(student_logits, [teacher_logits], y)

    assert metrics["mean_total_weight"] == pytest.approx(expected_mean, rel=1e-5)
    assert metrics["num_teachers"] == 1


# ---------------------------------------------------------------------------
# 3. A correct (high-RLC-weight) teacher dominates over an incorrect one
# ---------------------------------------------------------------------------
def test_correct_teacher_dominates():
    """With teacher A confident on the true class and teacher B basically uniform
    /wrong on the true class, the AMS loss with [A, B] should be very close to
    the loss with [A] alone: the per-sample normalizer 1 / (w_A + w_B) downweights
    B's KL contribution since w_B << w_A.
    """
    device = _device()
    torch.manual_seed(1)
    B, C = 8, 10
    y = torch.randint(0, C, (B,), device=device)
    student_logits = torch.randn(B, C, device=device)

    # Teacher A: large positive logit on the true class, ~0.9 prob.
    teacher_a = torch.zeros(B, C, device=device)
    for i in range(B):
        teacher_a[i, y[i]] = 4.0  # softmax ~ exp(4)/(exp(4)+9) ~ 0.858 — close to 0.9.

    # Teacher B: large negative logit on the true class so its true-class
    # softmax is tiny (~0.012), and positive elsewhere — definitely "wrong".
    teacher_b = torch.ones(B, C, device=device) * 2.0
    for i in range(B):
        teacher_b[i, y[i]] = -2.0

    # Sanity-check that the constructed weights have the intended magnitudes.
    w_a = F.softmax(teacher_a, dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
    w_b = F.softmax(teacher_b, dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
    assert (w_a > 0.7).all(), f"teacher A w too small: {w_a}"
    assert (w_b < 0.05).all(), f"teacher B w too large: {w_b}"

    loss_AB, _ = ams_distill_loss(student_logits, [teacher_a, teacher_b], y)
    loss_A, _ = ams_distill_loss(student_logits, [teacher_a], y)

    rel_diff = (loss_AB - loss_A).abs().item() / max(loss_A.abs().item(), 1e-8)
    assert rel_diff < 0.10, (
        f"Expected loss_AB ~= loss_A within 10% (B should be down-weighted), "
        f"got loss_A={loss_A.item():.4f}, loss_AB={loss_AB.item():.4f}, "
        f"rel_diff={rel_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. No gradient flows back into teacher parameters (gotcha #4)
# ---------------------------------------------------------------------------
def test_no_grad_through_teachers():
    device = _device()
    torch.manual_seed(2)

    # Real PreActResNet-18 acting as a teacher.
    teacher = preact_resnet18().to(device).eval()
    B, C = 2, 10
    x_adv = torch.rand(B, 3, 32, 32, device=device)
    y = torch.randint(0, C, (B,), device=device)

    # Mimic the training loop: teacher inference happens under no_grad.
    with torch.no_grad():
        teacher_logits = teacher(x_adv)

    # Student logits must have grad enabled so backward has something to do.
    student_logits = torch.randn(B, C, device=device, requires_grad=True)

    loss, _ = ams_distill_loss(student_logits, [teacher_logits], y)
    loss.backward()

    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter {name!r} unexpectedly has a grad"


# ---------------------------------------------------------------------------
# 5. Gradient does flow to the student
# ---------------------------------------------------------------------------
def test_gradient_flows_to_student():
    device = _device()
    torch.manual_seed(3)
    B, C = 4, 10
    student_logits = torch.randn(B, C, device=device, requires_grad=True)
    y = torch.randint(0, C, (B,), device=device)

    teacher_1 = torch.randn(B, C, device=device)
    teacher_2 = torch.randn(B, C, device=device)

    loss, _ = ams_distill_loss(student_logits, [teacher_1, teacher_2], y)
    loss.backward()

    assert student_logits.grad is not None
    assert torch.any(student_logits.grad != 0), (
        "Student gradient is identically zero — loss is not differentiable wrt student."
    )


# ---------------------------------------------------------------------------
# 6. Hand-computed KL value (pins KL direction + RLC weight semantics)
# ---------------------------------------------------------------------------
def test_kl_direction_against_handcomputed():
    """One teacher, one student, B=1, C=3. Distributions known, hand-compute KL.

    p_teacher = [0.7, 0.2, 0.1]; p_student = [0.1, 0.2, 0.7]; y = 0; lam = 1.0.

    KL(p_t || p_s) = 0.7 log(0.7/0.1) + 0.2 log(0.2/0.2) + 0.1 log(0.1/0.7)
                   = 0.6 log(7)
                   ~ 1.16754569

    Per-sample loss = lam * (w * KL) / (w + eps) where w = p_t[y=0] = 0.7.
    With eps=1e-8 the w cancellation is essentially exact: loss ~ 1.16754569.
    """
    device = _device()
    # Build logits whose softmax exactly equals the target probabilities by
    # using logits = log(p). softmax(log(p)) = p / sum(p) = p (since p sums to 1).
    p_teacher = torch.tensor([[0.7, 0.2, 0.1]], device=device)
    p_student = torch.tensor([[0.1, 0.2, 0.7]], device=device)
    teacher_logits = torch.log(p_teacher)
    student_logits = torch.log(p_student)
    y = torch.tensor([0], device=device)

    # Hand-computed reference.
    kl_ref = (
        0.7 * math.log(0.7 / 0.1)
        + 0.2 * math.log(0.2 / 0.2)
        + 0.1 * math.log(0.1 / 0.7)
    )
    # Per-sample loss in the function: lam * (w * KL) / (w + eps).
    # With lam=1, w=0.7, eps=1e-8 the multiplicative factor is 1/(1 + eps/0.7),
    # which is 1 - O(1e-8). So the loss matches KL to ~7 decimals.
    expected_loss = 1.0 * (0.7 * kl_ref) / (0.7 + 1e-8)

    loss, _ = ams_distill_loss(student_logits, [teacher_logits], y, lam=1.0)

    assert loss.item() == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
    # Also pin the bare KL value for clarity in failure output.
    assert loss.item() == pytest.approx(0.6 * math.log(7.0), rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. Default lambda is 0.5
# ---------------------------------------------------------------------------
def test_default_lam_is_0_5():
    sig = inspect.signature(ams_distill_loss)
    assert sig.parameters["lam"].default == 0.5


# ---------------------------------------------------------------------------
# 8. metrics["num_teachers"] reflects the input list length
# ---------------------------------------------------------------------------
def test_num_teachers_in_metrics():
    device = _device()
    B, C = 3, 10
    student_logits = torch.randn(B, C, device=device)
    y = torch.randint(0, C, (B,), device=device)
    teachers = [torch.randn(B, C, device=device) for _ in range(3)]

    _, metrics = ams_distill_loss(student_logits, teachers, y)

    assert metrics["num_teachers"] == 3
