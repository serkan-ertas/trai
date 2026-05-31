"""Tests for src/data/cifar10.py — CIFAR-10 data loaders.

Notes
-----
- CIFAR-10 (~170 MB) is downloaded once into ``C:/Users/user/Desktop/trai/data``
  by a session-scoped fixture and reused across all tests.
- ``num_workers=0`` everywhere to keep Windows tests deterministic and fast.
- ``batch_size=32`` and ``val_size=64`` keep iteration cheap; they intentionally
  differ from the production defaults (the loader API does not care).
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import get_cifar10_loaders


DATA_ROOT = "C:/Users/user/Desktop/trai/data"
BATCH_SIZE = 32
VAL_SIZE = 64


@pytest.fixture(scope="session")
def cifar_cache() -> str:
    """Ensure CIFAR-10 is on disk under DATA_ROOT.

    First invocation downloads (~170 MB). Subsequent calls (including in
    later tests of the same session) hit the cache.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    # Trigger torchvision's download / cache check exactly once.
    get_cifar10_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        num_workers=0,
        seed=0,
        download=True,
    )
    return DATA_ROOT


@pytest.fixture()
def loaders(cifar_cache):
    """Fresh (train, val, test) triple, seed=0, small batch/val for speed."""
    return get_cifar10_loaders(
        data_root=cifar_cache,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        num_workers=0,
        seed=0,
        download=False,
    )


def test_loaders_returned_in_order(loaders):
    """Function must return exactly (train, val, test) in that order."""
    assert isinstance(loaders, tuple), f"expected tuple, got {type(loaders)}"
    assert len(loaders) == 3, f"expected 3 loaders, got {len(loaders)}"
    train, val, test = loaders
    assert isinstance(train, DataLoader)
    assert isinstance(val, DataLoader)
    assert isinstance(test, DataLoader)

    # Train is the largest, test is exactly 10000, val matches val_size.
    assert len(train.dataset) > len(val.dataset)
    assert len(train.dataset) > len(test.dataset)
    assert len(test.dataset) == 10000
    assert len(val.dataset) == VAL_SIZE


def test_split_sizes(loaders):
    """With val_size=64: train=49936, val=64, test=10000."""
    train, val, test = loaders
    assert len(train.dataset) == 50000 - VAL_SIZE == 49936
    assert len(val.dataset) == VAL_SIZE == 64
    assert len(test.dataset) == 10000


def test_train_emits_unnormalized(loaders):
    """Raw [0,1] tensors only — no per-channel Normalize is allowed."""
    train, _, _ = loaders
    x, y = next(iter(train))
    assert x.dtype == torch.float32, f"expected float32, got {x.dtype}"
    assert x.shape[1:] == (3, 32, 32), f"expected (B,3,32,32), got {tuple(x.shape)}"
    assert x.min().item() >= 0.0 - 1e-6, (
        f"train min {x.min().item():.6f} < 0; normalization may be active"
    )
    assert x.max().item() <= 1.0 + 1e-6, (
        f"train max {x.max().item():.6f} > 1; normalization may be active"
    )
    # Labels should be ints in [0, 9].
    assert y.dtype in (torch.int64, torch.long)
    assert int(y.min().item()) >= 0
    assert int(y.max().item()) <= 9


def test_val_test_same_bounds(loaders):
    """val and test loaders also emit raw [0,1] tensors."""
    _, val, test = loaders
    for name, loader in (("val", val), ("test", test)):
        x, _ = next(iter(loader))
        assert x.min().item() >= 0.0 - 1e-6, (
            f"{name} min {x.min().item():.6f} < 0"
        )
        assert x.max().item() <= 1.0 + 1e-6, (
            f"{name} max {x.max().item():.6f} > 1"
        )
        assert x.shape[1:] == (3, 32, 32)


def test_deterministic_split(cifar_cache):
    """Same seed -> same val indices -> identical labels of first val batch."""
    _, val_a, _ = get_cifar10_loaders(
        data_root=cifar_cache,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        num_workers=0,
        seed=0,
        download=False,
    )
    _, val_b, _ = get_cifar10_loaders(
        data_root=cifar_cache,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        num_workers=0,
        seed=0,
        download=False,
    )
    _, y_a = next(iter(val_a))
    _, y_b = next(iter(val_b))
    assert torch.equal(y_a, y_b), (
        f"val labels differ with same seed: {y_a.tolist()} vs {y_b.tolist()}"
    )


def test_train_augmentation_active(cifar_cache):
    """Train transform must be stochastic: same index, two epochs -> different tensors.

    We build a one-sample train loader with ``shuffle=False`` so the same
    underlying image is fetched on each pass; if RandomCrop/HFlip are active,
    the resulting tensors will differ across iterations with overwhelming
    probability.
    """
    train, _, _ = get_cifar10_loaders(
        data_root=cifar_cache,
        batch_size=1,
        val_size=VAL_SIZE,
        num_workers=0,
        seed=0,
        download=False,
    )
    # Replace with a deterministic-order, single-sample loader on the same
    # underlying training subset, so we hit the SAME image twice.
    single = DataLoader(
        train.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    it = iter(single)
    x1, y1 = next(it)

    it2 = iter(single)
    x2, y2 = next(it2)

    assert torch.equal(y1, y2), "first sample's label should be stable"
    # Same image went through stochastic train transform twice.
    assert not torch.equal(x1, x2), (
        "train tensors should differ across epochs (RandomCrop/HFlip not active)"
    )


def test_val_no_augmentation(loaders):
    """Val/test must be deterministic across iterations (no augmentation)."""
    _, val, _ = loaders
    it1 = iter(val)
    x1, y1 = next(it1)
    it2 = iter(val)
    x2, y2 = next(it2)
    assert torch.equal(y1, y2), "val labels not stable across iterations"
    assert torch.equal(x1, x2), (
        "val tensors changed across iterations — augmentation may be leaking in"
    )
