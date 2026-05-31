"""Teacher snapshot buffer for AMS adversarial training (Algorithm 1).

Implements the multi-teacher self-distillation buffer M from the AMS paper
(Wang & Ding, ICLR 2025), Algorithm 1, lines 3 (iterate teachers) and 5
(append a snapshot every m epochs). Designed for a 4 GB VRAM budget: only one
GPU model object ("scratch") exists at any time, and its weights are
overwritten via `load_state_dict` as we iterate over the stored snapshots.
See the project spec gotchas #4 and #9.
"""

import copy
from collections.abc import Iterator
from typing import Callable

import torch
import torch.nn as nn


class TeacherBuffer:
    """Stores past-snapshot model weights on CPU, runs them through one shared
    GPU scratch model. Used by the AMS training loop (Algorithm 1) to evaluate
    each teacher h_{theta_j} on the current batch's adversarial examples
    without holding multiple model copies in VRAM (only 4 GB on the target
    laptop GPU).

    Constraint enforced for VRAM safety:
        ``__iter__`` reuses ONE underlying ``nn.Module`` object. Each yield
        loads a different snapshot's state_dict into the same module. Callers
        MUST NOT cache the yielded module across iterations and MUST NOT call
        a previous yield's module from a later iteration body -- doing so will
        silently use the wrong weights. The intended idiom is::

            buffer = TeacherBuffer(lambda: preact_resnet18(num_classes=10),
                                   device="cuda")
            # ... training proceeds ...
            if epoch % m == 0:
                buffer.append(model)            # snapshot current weights
            # in the next training step:
            teacher_logits = []
            with torch.no_grad():
                for teacher in buffer:           # iterate yields scratch model
                    teacher_logits.append(teacher(x_adv))

    The buffer keeps the scratch model in ``.eval()`` mode with
    ``requires_grad_(False)`` on all parameters; the caller is responsible for
    wrapping forward calls in ``torch.no_grad()`` per the project spec gotcha #4.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        device: torch.device | str,
    ) -> None:
        self._factory = model_factory
        self._device = torch.device(device)
        self._snapshots: list[dict[str, torch.Tensor]] = []
        self._scratch = model_factory().to(self._device).eval()
        for p in self._scratch.parameters():
            p.requires_grad_(False)

    def append(self, model: nn.Module) -> None:
        """Snapshot ``model``'s current parameters and buffers onto CPU.

        ``to("cpu", copy=True)`` is used so that even if ``model`` is already
        on CPU we get a freshly allocated tensor -- otherwise an in-place
        update to ``model`` would silently corrupt the stored snapshot.
        """
        sd = model.state_dict()
        cpu_sd = {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}
        self._snapshots.append(cpu_sd)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __iter__(self) -> Iterator[nn.Module]:
        """Yield the scratch model loaded with each snapshot in turn.

        The same ``nn.Module`` object is yielded every iteration with
        different weights. ``load_state_dict`` copies into the existing GPU
        buffers via ``copy_`` -- no fresh allocation, so the GPU footprint is
        constant in the number of snapshots.
        """
        for sd in self._snapshots:
            self._scratch.load_state_dict(sd)
            self._scratch.eval()
            yield self._scratch

    def state_dict(self) -> dict:
        """Return a deep copy of the snapshot list for checkpointing."""
        return {"snapshots": [copy.deepcopy(sd) for sd in self._snapshots]}

    def load_state_dict(self, state: dict) -> None:
        """Restore snapshots from a previously saved ``state_dict()`` output."""
        self._snapshots = [
            {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}
            for sd in state["snapshots"]
        ]
