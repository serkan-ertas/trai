"""Dynamic-correctness tests for the AutoAttack wrapper (task 1.5).

These tests verify the *plumbing* of `src.attacks.autoattack_wrap`:
- the public functions exist and are callable,
- the lazy-import pattern works,
- `autoattack_perturb` returns a tensor with the right shape/dtype/device,
  the perturbation respects the ε-ball, and pixels stay in [0,1],
- `model.training` is preserved across calls,
- `evaluate_autoattack` returns a float in [0,1],
- `max_batches` caps loader iteration.

NOTE on `version=`: AutoAttack's `version="standard"` runs the full
APGD-CE + APGD-DLR + FAB + Square ensemble, which costs 30+ seconds even
on tiny inputs. Since these tests check *plumbing*, not attack quality,
we use `version="rand"` (APGD-CE + APGD-DLR only, no FAB / Square)
which on this hardware takes ~15 s per 2-sample batch. The wrapper
threads the `version` argument straight through to `AutoAttack(...)`,
so anything that works for "rand" works identically for "standard"
— it just takes longer. Choice documented in TEST_REPORT_1.5_autoattack.md.
"""

import sys
import time

import pytest
import torch
import torch.nn as nn

from src.attacks import autoattack_perturb, evaluate_autoattack
from src.models import preact_resnet18


CUDA_AVAILABLE = torch.cuda.is_available()
EPS = 8.0 / 255.0


def _device() -> torch.device:
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


# ---------------------------------------------------------------------------
# 1. Plumbing: imports and lazy loading
# ---------------------------------------------------------------------------


def test_imports_and_lazy_loading():
    """Both public symbols are callables, and the wrapper module is importable
    without triggering an `autoattack` import.

    The wrapper performs `from autoattack import AutoAttack` inside each
    function body (autoattack_wrap.py lines 28, 63). We can't fully prove this
    without mocking, but we can sanity-check by importing the wrapper module
    cold (after removing any cached `autoattack` import) and confirming the
    `autoattack` module isn't pulled in transitively.
    """
    assert callable(autoattack_perturb), "autoattack_perturb is not callable"
    assert callable(evaluate_autoattack), "evaluate_autoattack is not callable"

    # Drop any cached references and re-import the wrapper. The wrapper itself
    # must not import `autoattack` at top level.
    for mod in [
        "autoattack",
        "src.attacks.autoattack_wrap",
        "src.attacks",  # the package __init__ pulls in the wrapper, but only its names
    ]:
        sys.modules.pop(mod, None)

    import src.attacks.autoattack_wrap as wrap_mod  # noqa: F401
    assert "autoattack" not in sys.modules, (
        "importing src.attacks.autoattack_wrap pulled in `autoattack` — "
        "lazy-import pattern is broken"
    )

    # Re-importing the package shouldn't pull autoattack in either (it just
    # re-exports the wrapper's public functions).
    import src.attacks  # noqa: F401
    assert "autoattack" not in sys.modules, (
        "importing src.attacks pulled in `autoattack` — lazy import broken"
    )


# ---------------------------------------------------------------------------
# 2. Shape, bounds, dtype, device on autoattack_perturb
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="AutoAttack default device is CUDA")
def test_perturb_shape_and_bounds():
    """autoattack_perturb returns the right shape/dtype/device, respects ε,
    and stays in [0,1].

    Uses `version="rand"` (APGD-CE + APGD-DLR) for speed; see module docstring.
    AutoAttack short-circuits on already-misclassified samples, so even when
    the random init model predicts something other than the supplied labels
    (in which case x_adv == x), all four asserted properties (shape, dtype,
    device, ε-ball, [0,1]) still hold.
    """
    device = _device()
    torch.manual_seed(0)
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.tensor([3, 7], device=device)

    t0 = time.time()
    x_adv = autoattack_perturb(
        model, x, y, eps=EPS, version="rand", verbose=False, seed=0
    )
    elapsed = time.time() - t0
    # Sanity guardrail: if this blows past 90 s on the test machine, the
    # "fast" version regressed and we should refactor.
    assert elapsed < 90.0, f"autoattack_perturb took {elapsed:.1f}s (>90s budget)"

    assert isinstance(x_adv, torch.Tensor), f"expected Tensor, got {type(x_adv)}"
    assert tuple(x_adv.shape) == (2, 3, 32, 32), f"shape mismatch: {tuple(x_adv.shape)}"
    assert x_adv.dtype == torch.float32, f"dtype was {x_adv.dtype}, expected float32"
    assert x_adv.device == x.device, (
        f"x_adv device {x_adv.device} != x device {x.device}"
    )

    assert x_adv.min().item() >= 0.0, f"x_adv min {x_adv.min().item()} < 0"
    assert x_adv.max().item() <= 1.0, f"x_adv max {x_adv.max().item()} > 1"

    linf = (x_adv - x).abs().max().item()
    assert linf <= EPS + 1e-6, (
        f"linf perturbation {linf} exceeds eps+tol ({EPS} + 1e-6)"
    )


# ---------------------------------------------------------------------------
# 3. Model state preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="AutoAttack default device is CUDA")
def test_perturb_model_state_preserved():
    """Calling autoattack_perturb on a model that was in train() mode leaves
    it in train() mode afterwards (try/finally pattern in lines 30, 47–48).
    """
    device = _device()
    torch.manual_seed(0)
    model = preact_resnet18(num_classes=10).to(device)
    model.train()
    assert model.training is True, "test precondition: model should start in train mode"

    x = torch.rand(2, 3, 32, 32, device=device)
    y = torch.tensor([3, 7], device=device)
    _ = autoattack_perturb(
        model, x, y, eps=EPS, version="rand", verbose=False, seed=0
    )

    assert model.training is True, (
        "autoattack_perturb did not restore model.train() — the try/finally "
        "restoration of `was_training` is broken"
    )


# ---------------------------------------------------------------------------
# 4. evaluate_autoattack returns a fraction in [0,1]
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="AutoAttack default device is CUDA")
def test_evaluate_returns_fraction():
    """evaluate_autoattack returns a Python float in [0.0, 1.0].

    Single 2-sample batch via a trivial generator. With `version="rand"` and
    a random-init model, AutoAttack short-circuits most samples; either way
    the function must return a float in the unit interval.
    """
    device = _device()
    torch.manual_seed(0)
    model = preact_resnet18(num_classes=10).to(device).eval()

    def loader():
        # CPU tensors — evaluate_autoattack should .to(device) them itself.
        yield (
            torch.rand(2, 3, 32, 32),
            torch.tensor([0, 1]),
        )

    t0 = time.time()
    acc = evaluate_autoattack(
        model, loader(),
        eps=EPS, device="cuda", version="rand",
        max_batches=1, seed=0, verbose=False,
    )
    elapsed = time.time() - t0
    assert elapsed < 90.0, f"evaluate_autoattack took {elapsed:.1f}s (>90s budget)"

    assert isinstance(acc, float), f"expected float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} not in [0.0, 1.0]"


# ---------------------------------------------------------------------------
# 5. max_batches caps loader consumption
# ---------------------------------------------------------------------------


class _CountingLoader:
    """A trivial iterable that yields up to N batches but counts how many
    batches were actually pulled. Lets the test assert that evaluate_autoattack
    honored `max_batches`.
    """

    def __init__(self, n_batches: int, batch_size: int = 2):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.pulled = 0

    def __iter__(self):
        for _ in range(self.n_batches):
            self.pulled += 1
            yield (
                torch.rand(self.batch_size, 3, 32, 32),
                torch.tensor([0] * self.batch_size),
            )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="AutoAttack default device is CUDA")
def test_max_batches_caps_iteration():
    """A loader carrying 3 batches but called with max_batches=1 must NOT
    consume all 3 — i.e. the cap actually limits iteration.

    Note on the exact count: the wrapper uses

        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches: break
            ... process batch i ...

    With `max_batches=1`, `enumerate` will pull batch 0 (processed), then pull
    batch 1 (at which point `i >= 1` triggers `break` before processing). So
    a Python generator typically yields 2 batches before the break fires, but
    only 1 is processed. The semantic invariants we assert are:

      (a) NOT all 3 batches were pulled — the cap is honored;
      (b) at least 1 batch was pulled — the function actually ran;
      (c) at most `max_batches + 1` were pulled — bounded by the standard
          enumerate/break idiom (no runaway).
    """
    device = _device()
    torch.manual_seed(0)
    model = preact_resnet18(num_classes=10).to(device).eval()

    n_batches = 3
    max_batches = 1
    loader = _CountingLoader(n_batches=n_batches, batch_size=2)

    t0 = time.time()
    _ = evaluate_autoattack(
        model, loader,
        eps=EPS, device="cuda", version="rand",
        max_batches=max_batches, seed=0, verbose=False,
    )
    elapsed = time.time() - t0
    assert elapsed < 90.0, f"evaluate_autoattack took {elapsed:.1f}s (>90s budget)"

    assert loader.pulled < n_batches, (
        f"evaluate_autoattack pulled {loader.pulled}/{n_batches} batches — "
        f"max_batches={max_batches} did not cap iteration"
    )
    assert loader.pulled >= 1, (
        f"evaluate_autoattack pulled 0 batches — never ran"
    )
    assert loader.pulled <= max_batches + 1, (
        f"evaluate_autoattack pulled {loader.pulled} batches with "
        f"max_batches={max_batches}; expected at most {max_batches + 1} "
        f"(one lookahead by enumerate before break)"
    )
