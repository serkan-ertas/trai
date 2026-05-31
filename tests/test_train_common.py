"""Unit tests for the shared training helpers in ``src/experiments/common.py`` (task 2.1).

Coverage:

- ``load_config`` parses the TRADES baseline YAML to the expected schema.
- ``JsonlLogger`` writes one JSON object per line, flushes per write, and
  appends correctly when ``resume=True``.
- ``save_checkpoint`` / ``load_checkpoint`` roundtrip model + optimizer +
  scheduler + (optional) GradScaler state.
- ``save_checkpoint`` is atomic (no ``.tmp`` left over after success).
- ``make_run_dir`` is idempotent and does not delete pre-existing files.
- ``evaluate_pgd20`` returns the expected dict and restores ``model.training``.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.common import (
    JsonlLogger,
    evaluate_pgd20,
    load_checkpoint,
    load_config,
    make_run_dir,
    save_checkpoint,
)
from src.models import preact_resnet18


PROJECT_ROOT = Path("C:/Users/user/Desktop/trai")
BASELINE_CFG = PROJECT_ROOT / "configs" / "trades_baseline.yaml"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# (a) load_config
# ---------------------------------------------------------------------------


def test_load_config_parses_baseline_yaml():
    """Verify the TRADES baseline YAML loads with the expected schema."""
    cfg = load_config(str(BASELINE_CFG))

    assert cfg["seed"] == 0
    assert cfg["train"]["epochs"] == 40
    assert cfg["loss"]["trades_beta"] == 6.0
    assert cfg["attack"]["train_inner"]["steps"] == 5
    assert cfg["attack"]["val_pgd20"]["steps"] == 20


# ---------------------------------------------------------------------------
# (b) JsonlLogger
# ---------------------------------------------------------------------------


def test_jsonl_logger_writes_and_flushes(tmp_path):
    """Two writes -> two lines; resume=True appends a third line."""
    path = tmp_path / "m.jsonl"

    # First session: fresh file, two records.
    logger = JsonlLogger(path, resume=False)
    logger.log({"epoch": 0, "loss": 1.234})
    logger.log({"epoch": 1, "loss": 0.567})
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert isinstance(rec0, dict) and isinstance(rec1, dict)
    assert rec0["epoch"] == 0 and rec0["loss"] == 1.234
    assert rec1["epoch"] == 1 and rec1["loss"] == 0.567

    # Second session: resume=True appends.
    logger2 = JsonlLogger(path, resume=True)
    logger2.log({"epoch": 2, "loss": 0.3})
    logger2.close()

    lines2 = path.read_text(encoding="utf-8").splitlines()
    assert len(lines2) == 3, f"Expected 3 lines after resume, got {len(lines2)}"
    rec2 = json.loads(lines2[2])
    assert rec2["epoch"] == 2 and rec2["loss"] == 0.3

    # The earlier records must still be intact (append, not truncate).
    assert json.loads(lines2[0])["epoch"] == 0
    assert json.loads(lines2[1])["epoch"] == 1


# ---------------------------------------------------------------------------
# (c) save_checkpoint / load_checkpoint roundtrip
# ---------------------------------------------------------------------------


def _build_tiny_training_state(use_cuda: bool):
    """A small Linear model + SGD + MultiStepLR + (CUDA) GradScaler."""
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    torch.manual_seed(0)
    model = nn.Linear(8, 4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 4], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda) if use_cuda else None
    return model, opt, sched, scaler, device


def _run_one_optim_step(model, opt, scaler, device):
    """Take one tiny forward/backward step so optimizer state is non-empty."""
    x = torch.randn(4, 8, device=device)
    y = torch.randint(0, 4, (4,), device=device)
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()


def test_save_load_checkpoint_roundtrip(tmp_path):
    """Save -> rebuild fresh state -> load -> all components match source."""
    use_cuda = torch.cuda.is_available()
    model, opt, sched, scaler, device = _build_tiny_training_state(use_cuda)

    # Populate optimizer + scheduler + scaler state with one real step.
    _run_one_optim_step(model, opt, scaler, device)
    sched.step()
    sched.step()  # last_epoch == 2 now

    # Save the state we want to recover later.
    src_model_state = copy.deepcopy(model.state_dict())
    src_opt_state = copy.deepcopy(opt.state_dict())
    src_sched_last_epoch = sched.last_epoch
    src_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

    ckpt_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        ckpt_path,
        epoch=5,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        best_metric=0.42,
    )
    assert ckpt_path.exists(), "checkpoint file was not created"

    # Build fresh state (different init), then load.
    torch.manual_seed(123)  # different seed -> different fresh params
    fresh_model, fresh_opt, fresh_sched, fresh_scaler, _ = _build_tiny_training_state(
        use_cuda
    )

    # Sanity check: fresh model parameters DIFFER from saved before loading.
    fresh_first_param = next(iter(fresh_model.state_dict().values()))
    src_first_param = next(iter(src_model_state.values()))
    assert not torch.equal(
        fresh_first_param.cpu(), src_first_param.cpu()
    ), "fresh model must differ from saved before load (test is degenerate otherwise)"

    map_loc = device if use_cuda else "cpu"
    returned = load_checkpoint(
        ckpt_path,
        model=fresh_model,
        optimizer=fresh_opt,
        scheduler=fresh_sched,
        scaler=fresh_scaler,
        map_location=map_loc,
    )

    # Returned metadata.
    assert returned["epoch"] == 5
    assert returned["best_metric"] == 0.42

    # Model parameters byte-equal after load.
    loaded_state = fresh_model.state_dict()
    for k, v in src_model_state.items():
        assert torch.equal(loaded_state[k].cpu(), v.cpu()), (
            f"model param {k!r} mismatch after load"
        )

    # Optimizer state matches.
    assert fresh_opt.state_dict()["state"].keys() == src_opt_state["state"].keys()
    for pid in src_opt_state["state"]:
        src_entry = src_opt_state["state"][pid]
        new_entry = fresh_opt.state_dict()["state"][pid]
        for k, v in src_entry.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(new_entry[k].cpu(), v.cpu()), (
                    f"optimizer state[{pid}][{k!r}] mismatch"
                )
            else:
                assert new_entry[k] == v, (
                    f"optimizer state[{pid}][{k!r}] mismatch ({new_entry[k]} != {v})"
                )

    # Scheduler last_epoch matches.
    assert fresh_sched.last_epoch == src_sched_last_epoch, (
        f"scheduler last_epoch {fresh_sched.last_epoch} != {src_sched_last_epoch}"
    )

    # GradScaler state matches (CUDA only).
    if scaler is not None:
        loaded_scaler_state = fresh_scaler.state_dict()
        # Scaler dict has scalar fields like "scale", "growth_factor", etc.
        for k, v in src_scaler_state.items():
            assert loaded_scaler_state[k] == v, (
                f"scaler state[{k!r}] mismatch ({loaded_scaler_state[k]} != {v})"
            )


# ---------------------------------------------------------------------------
# (d) save_checkpoint atomic
# ---------------------------------------------------------------------------


def test_save_checkpoint_is_atomic(tmp_path):
    """After a successful save, no .tmp sibling remains; final file exists."""
    use_cuda = torch.cuda.is_available()
    model, opt, sched, scaler, _ = _build_tiny_training_state(use_cuda)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        epoch=0,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        best_metric=0.0,
    )

    assert ckpt_path.exists(), "final checkpoint file missing after save"
    tmp_sibling = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    assert not tmp_sibling.exists(), (
        f"left-over .tmp file: {tmp_sibling} — os.replace did not run"
    )

    # No other .tmp files in the directory either.
    leftover_tmps = list(tmp_path.glob("*.tmp"))
    assert leftover_tmps == [], f"unexpected .tmp files: {leftover_tmps}"


# ---------------------------------------------------------------------------
# (e) make_run_dir idempotent
# ---------------------------------------------------------------------------


def test_make_run_dir_idempotent(tmp_path):
    """Calling make_run_dir twice does not error and does not delete contents."""
    out_dir = tmp_path
    run_id = "myrun"

    # First call: creates.
    p1 = make_run_dir(str(out_dir), run_id)
    assert p1.is_dir()
    assert (out_dir / run_id).is_dir()

    # Drop a file in there — it must survive the second call.
    sentinel = p1 / "sentinel.txt"
    sentinel.write_text("important", encoding="utf-8")
    assert sentinel.exists()

    # Second call: finds existing, returns the path, does NOT delete the file.
    p2 = make_run_dir(str(out_dir), run_id)
    assert p2 == p1
    assert p2.is_dir()
    assert sentinel.exists(), (
        "make_run_dir deleted a pre-existing file inside the run dir — resume would be broken"
    )
    assert sentinel.read_text(encoding="utf-8") == "important"


# ---------------------------------------------------------------------------
# (f) evaluate_pgd20 returns expected shape
# ---------------------------------------------------------------------------


def _make_tiny_loader(device: torch.device, num_batches: int = 2, bs: int = 4):
    """2 batches of 4 raw-[0,1] tensors with random labels."""
    torch.manual_seed(7)
    xs = torch.rand(num_batches * bs, 3, 32, 32)
    ys = torch.randint(0, 10, (num_batches * bs,), dtype=torch.long)
    ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def test_evaluate_pgd20_on_tiny_model():
    """Returns the expected dict; accuracies are floats in [0,1]; num_samples is correct."""
    device = _device()
    model = preact_resnet18(num_classes=10).to(device)
    loader = _make_tiny_loader(device)

    result = evaluate_pgd20(
        model, loader, device, eps=8 / 255, alpha=2 / 255, steps=2
    )

    assert isinstance(result, dict)
    for k in ("clean_acc", "robust_acc", "num_samples"):
        assert k in result, f"missing key {k!r} in evaluate_pgd20 output"

    assert isinstance(result["clean_acc"], float)
    assert isinstance(result["robust_acc"], float)
    assert isinstance(result["num_samples"], int)

    assert 0.0 <= result["clean_acc"] <= 1.0
    assert 0.0 <= result["robust_acc"] <= 1.0
    assert result["num_samples"] == 8, (
        f"expected num_samples == 8 (2 batches * 4 samples), got {result['num_samples']}"
    )


# ---------------------------------------------------------------------------
# (g) evaluate_pgd20 restores train mode on return
# ---------------------------------------------------------------------------


def test_evaluate_pgd20_restores_train_mode():
    """Model is in train() before; after evaluate_pgd20 it is still in train()."""
    device = _device()
    model = preact_resnet18(num_classes=10).to(device)
    model.train()
    assert model.training is True

    loader = _make_tiny_loader(device)
    _ = evaluate_pgd20(model, loader, device, eps=8 / 255, alpha=2 / 255, steps=2)

    assert model.training is True, (
        "evaluate_pgd20 did not restore train() mode after eval (try/finally broken?)"
    )
