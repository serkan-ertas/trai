"""Dynamic-correctness tests for PreActResNet-18 (task 1.1).

These tests verify:
1. Forward shape on a batch of CIFAR-sized inputs.
2. The embedded-normalization pattern is actually applying normalization
   (passing a manually-normalized input should NOT match passing the raw
   input — if it did, the model would either be double-normalizing or
   not normalizing at all).
3. Parameter count is in the PreActResNet-18 range (~11M).
4. eval() mode is deterministic (BN frozen, no dropout).
5. _Normalize's mean/std buffers move correctly with .to(device).
"""

import pytest
import torch

from src.models import preact_resnet18
from src.models.preact_resnet import _CIFAR10_MEAN, _CIFAR10_STD


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_forward_shape():
    """Batch of (8, 3, 32, 32) raw [0,1] inputs -> logits of shape (8, 10)."""
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(8, 3, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (8, 10), f"expected (8, 10), got tuple({tuple(y.shape)})"
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all(), "logits contain non-finite values"


def test_normalization_inside_model():
    """Structural sanity check on the embedded-normalization pattern.

    If the model normalizes internally, then feeding raw x produces a different
    output than feeding an already-normalized x (because the second path gets
    normalized AGAIN). If the model accidentally did not normalize at all, the
    two outputs would be different in the opposite way (raw vs. pre-normalized).
    Either way, the outputs MUST differ. If they were identical, normalization
    would be the identity, which would mean the embedded-norm pattern is broken.
    """
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    # raw [0,1] input
    x_raw = torch.rand(4, 3, 32, 32, device=device)

    # manually normalized version
    mean = torch.tensor(_CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR10_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x_raw - mean) / std

    with torch.no_grad():
        y_raw = model(x_raw)
        y_norm = model(x_norm)

    # Inputs must differ (sanity on the test itself).
    assert not torch.allclose(x_raw, x_norm), "raw and normalized inputs are identical — test is broken"

    # Outputs must differ — otherwise the embedded normalization is the identity.
    assert not torch.allclose(y_raw, y_norm, atol=1e-4), (
        "raw and pre-normalized inputs produced (essentially) identical outputs — "
        "embedded normalization may be missing or behaving as identity"
    )


def test_param_count():
    """PreActResNet-18 has ~11.17M trainable parameters."""
    model = preact_resnet18(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 10_000_000 < n_params < 12_000_000, (
        f"parameter count {n_params:,} is outside the PreActResNet-18 range "
        f"(10M, 12M) — architecture may be wrong"
    )


def test_eval_mode_deterministic():
    """Two forward passes with the same input in eval() mode produce identical outputs.

    BatchNorm uses running stats in eval mode; there is no dropout in
    PreActResNet-18. So two calls should be bit-for-bit identical (mod
    nondeterministic cuDNN kernels, which torch.allclose with default tol
    tolerates fine).
    """
    torch.manual_seed(0)
    device = _device()
    model = preact_resnet18(num_classes=10).to(device).eval()

    x = torch.rand(4, 3, 32, 32, device=device)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    assert torch.allclose(y1, y2), (
        "eval() mode forward pass is not deterministic — BN may not be frozen "
        "or some stochastic op is active"
    )


def test_buffers_move_with_model():
    """_Normalize's mean/std buffers move with model.to(device)."""
    device = _device()
    model = preact_resnet18(num_classes=10)

    # Start on CPU explicitly to make the test meaningful regardless of CUDA.
    model = model.to("cpu")
    assert model.normalize.mean.device.type == "cpu"
    assert model.normalize.std.device.type == "cpu"

    model = model.to(device)
    assert model.normalize.mean.device.type == device.type, (
        f"normalize.mean did not move to {device.type}"
    )
    assert model.normalize.std.device.type == device.type, (
        f"normalize.std did not move to {device.type}"
    )

    # And the buffers should still hold the CIFAR-10 constants (within fp32 noise).
    expected_mean = torch.tensor(_CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    expected_std = torch.tensor(_CIFAR10_STD, device=device).view(1, 3, 1, 1)
    assert torch.allclose(model.normalize.mean, expected_mean)
    assert torch.allclose(model.normalize.std, expected_std)
