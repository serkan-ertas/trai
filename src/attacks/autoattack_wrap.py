"""Thin wrapper around the ``autoattack`` package for ℓ∞ evaluation.

Both functions operate on raw [0,1] pixel inputs (the model is responsible for
its own normalization — see the project spec gotcha #10).

The ``autoattack`` import is performed lazily inside each function so that
importing the rest of ``src.attacks`` (e.g. PGD, CW) does not pay the cost of
pulling in scipy and the rest of the AutoAttack dependency tree.
"""

from typing import Optional

import torch
import torch.nn as nn


def autoattack_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    version: str = "standard",
    norm: str = "Linf",
    verbose: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Return AutoAttack adversarial examples in [0,1]. Wraps the autoattack package."""
    from autoattack import AutoAttack  # lazy import (heavy: pulls in scipy etc.)

    was_training = model.training
    model.eval()
    try:
        # Older autoattack==0.1 builds may not accept a ``device`` kwarg; if absent
        # the library infers device from the model's parameters.
        try:
            attacker = AutoAttack(
                model, norm=norm, eps=eps, version=version,
                verbose=verbose, seed=seed, device=x.device,
            )
        except TypeError:
            attacker = AutoAttack(
                model, norm=norm, eps=eps, version=version,
                verbose=verbose, seed=seed,
            )
        x_adv = attacker.run_standard_evaluation(x, y, bs=x.shape[0])
        return x_adv.to(x.device)
    finally:
        model.train(was_training)


def evaluate_autoattack(
    model: nn.Module,
    loader,
    eps: float = 8 / 255,
    version: str = "standard",
    norm: str = "Linf",
    device: str = "cuda",
    verbose: bool = False,
    seed: int = 0,
    max_batches: Optional[int] = None,
) -> float:
    """Run AutoAttack over a loader; return robust accuracy (fraction in [0,1])."""
    from autoattack import AutoAttack  # lazy import (heavy: pulls in scipy etc.)

    model.to(device)
    was_training = model.training
    model.eval()
    try:
        try:
            attacker = AutoAttack(
                model, norm=norm, eps=eps, version=version,
                verbose=verbose, seed=seed, device=device,
            )
        except TypeError:
            attacker = AutoAttack(
                model, norm=norm, eps=eps, version=version,
                verbose=verbose, seed=seed,
            )

        correct = 0
        total = 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            x_adv = attacker.run_standard_evaluation(x, y, bs=x.shape[0])
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / total if total > 0 else 0.0
    finally:
        model.train(was_training)
