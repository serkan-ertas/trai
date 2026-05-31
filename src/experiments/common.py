"""Shared helpers for the training experiments.

Imported by both ``train_trades.py`` (TRADES baseline, task 2.1) and the
upcoming ``train_ams.py`` (TRADES+AMS, task 2.2). Keep this module generic;
nothing here is allowed to assume which loss the outer loop uses.

What lives here:

- :func:`load_config`        — YAML config loader.
- :func:`make_run_dir`       — idempotent ``runs/<run_id>/`` creator.
- :class:`JsonlLogger`       — append-mode JSONL writer with per-line flush.
- :func:`save_checkpoint`    — atomic checkpoint save (tmp + os.replace).
- :func:`load_checkpoint`    — counterpart loader, restores model/opt/sched/scaler.
- :func:`evaluate_pgd20`     — clean + PGD-20 robust accuracy on a loader.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config into a plain dict via ``yaml.safe_load``."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(output_dir: str, run_id: str) -> Path:
    """Create ``output_dir/run_id/`` if missing and return the path.

    Does NOT delete existing contents — resume relies on prior files.
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class JsonlLogger:
    """One JSON object per line, flushed on every write.

    Pass ``resume=True`` to append to an existing file (used by ``--resume``
    runs); otherwise the file is truncated to start fresh.
    """

    def __init__(self, path: Path, resume: bool = False) -> None:
        mode = "a" if resume else "w"
        # encoding kept explicit so Windows doesn't default to cp1252.
        self._fp = open(path, mode, encoding="utf-8")

    def log(self, record: dict) -> None:
        """Serialize ``record`` as one JSON line and flush to disk."""
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    best_metric: float,
    extras: Optional[dict] = None,
) -> None:
    """Atomically save a training checkpoint.

    The dict written contains ``epoch``, ``model``, ``optimizer``,
    ``scheduler``, ``best_metric``, and optionally ``scaler``. Any keys in
    ``extras`` are merged in; they must not collide with the standard keys.

    Atomicity: torch.save writes to ``<path>.tmp`` first, then ``os.replace``
    swaps it into place. ``os.replace`` is atomic on POSIX and on Windows
    (single-volume), so a Ctrl-C mid-save cannot corrupt the on-disk file.
    """
    state: dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extras is not None:
        overlap = set(extras.keys()) & set(state.keys())
        assert not overlap, f"extras overwrite standard keys: {overlap}"
        state.update(extras)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Any = "cuda",
) -> dict:
    """Load a checkpoint and restore the components provided.

    Returns the full state dict so the caller can read ``epoch``,
    ``best_metric``, and any ``extras`` it wrote.
    """
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state


def evaluate_pgd20(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float,
    alpha: float,
    steps: int,
    max_batches: Optional[int] = None,
) -> dict:
    """Clean accuracy and PGD-K robust accuracy on a loader.

    PGD here is the CE-objective attack (paper Eq. 6 / standard PGD-20 eval);
    it runs in FP32 outside ``autocast`` because the sign attack is fragile
    under FP16. Model train/eval state is restored on return.
    """
    from src.attacks import pgd_attack

    was_training = model.training
    model.eval()
    try:
        total = 0
        n_clean_correct = 0
        n_robust_correct = 0

        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                pred_clean = model(x).argmax(dim=1)
            n_clean_correct += (pred_clean == y).sum().item()

            x_adv = pgd_attack(
                model, x, y,
                eps=eps,
                alpha=alpha,
                steps=steps,
                objective="ce",
                random_start=True,
            )
            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1)
            n_robust_correct += (pred_adv == y).sum().item()

            total += x.size(0)

        if total == 0:
            return {"clean_acc": 0.0, "robust_acc": 0.0, "num_samples": 0}
        return {
            "clean_acc": n_clean_correct / total,
            "robust_acc": n_robust_correct / total,
            "num_samples": total,
        }
    finally:
        model.train(was_training)
