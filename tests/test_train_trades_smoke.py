"""End-to-end smoke test for the TRADES baseline training script (task 2.1).

Launches ``python -m src.experiments.train_trades --config <smoke.yaml>`` as a
subprocess with a 1-epoch / 32-batch / tiny-attack config built in ``tmp_path``
and verifies the expected output artifacts are produced.

This is the project's highest-value test: it exercises every layer
(data -> attack -> AMP outer step -> scheduler -> validation -> JSONL
logging -> atomic checkpoint save -> final checkpoint), and would catch
most bugs an undergrad spends 6 hours debugging.

Budget: ~3 minutes wall-clock on RTX 3050. Test timeout is 5 minutes.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path("C:/Users/user/Desktop/trai")
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
BASELINE_CFG = PROJECT_ROOT / "configs" / "trades_baseline.yaml"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke run requires CUDA (AMP path); training on CPU is too slow.",
)
def test_one_epoch_smoke_run(tmp_path):
    """Run train_trades for 1 epoch on a tiny subset; check all expected artifacts."""

    # ---- Build a smoke config based on the real baseline YAML ----
    with open(BASELINE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["run_id"] = "smoke_trades"
    cfg["output_dir"] = str((tmp_path / "runs").as_posix())
    cfg["seed"] = 0

    cfg["data"]["batch_size"] = 32
    cfg["data"]["val_size"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["data"]["data_root"] = "C:/Users/user/Desktop/trai/data"
    # download=False so the test fails fast if the cache is missing instead of
    # silently hitting the network mid-CI; the file lives there per the task spec.
    cfg["data"]["download"] = False

    cfg["train"]["epochs"] = 1
    # Milestone past epoch 0 — no actual LR decay should fire during this run.
    cfg["train"]["lr_schedule"]["milestones"] = [10]
    cfg["train"]["amp"] = True

    cfg["attack"]["train_inner"]["steps"] = 2
    cfg["attack"]["val_pgd20"]["steps"] = 5

    smoke_yaml = tmp_path / "smoke.yaml"
    with open(smoke_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- Launch the training script as a subprocess ----
    cmd = [
        str(PYTHON_EXE),
        "-m", "src.experiments.train_trades",
        "--config", str(smoke_yaml),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # check manually so we can include stdout/stderr in the failure
    )
    if proc.returncode != 0:
        pytest.fail(
            f"train_trades subprocess exited with code {proc.returncode}\n"
            f"---- stdout ----\n{proc.stdout}\n"
            f"---- stderr ----\n{proc.stderr}\n"
        )

    # ---- Verify the run directory layout ----
    run_dir = tmp_path / "runs" / "smoke_trades"
    assert run_dir.is_dir(), f"run directory not created at {run_dir}"

    # config snapshot
    cfg_snap = run_dir / "config.yaml"
    assert cfg_snap.exists(), f"config snapshot missing: {cfg_snap}"

    # metrics.jsonl with exactly one line of expected shape
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl missing: {metrics_path}"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, (
        f"expected exactly 1 metrics line for 1-epoch run, got {len(lines)}: {lines}"
    )
    rec = json.loads(lines[0])
    expected_keys = {
        "epoch",
        "lr",
        "train_loss_ce",
        "train_loss_kl",
        "train_acc_clean",
        "val_acc_clean",
        "val_robust_acc",
        "train_time_sec",
    }
    missing = expected_keys - set(rec.keys())
    assert not missing, f"metrics record missing keys: {missing} (got keys: {set(rec.keys())})"
    assert rec["epoch"] == 0, f"epoch should be 0 for 1-epoch run, got {rec['epoch']!r}"

    # All three checkpoint files exist
    last_ckpt = run_dir / "checkpoint_last.pt"
    best_ckpt = run_dir / "checkpoint_best.pt"
    final_ckpt = run_dir / "checkpoint_final.pt"
    assert last_ckpt.exists(), f"checkpoint_last.pt missing at {last_ckpt}"
    assert best_ckpt.exists(), (
        f"checkpoint_best.pt missing at {best_ckpt} — first epoch should always "
        f"write best since initial best_metric=-1.0"
    )
    assert final_ckpt.exists(), f"checkpoint_final.pt missing at {final_ckpt}"

    # No left-over .tmp files (atomic save cleaned them up)
    leftover_tmps = list(run_dir.glob("*.tmp"))
    assert leftover_tmps == [], f"unexpected .tmp leftovers: {leftover_tmps}"

    # ---- Inspect checkpoint_final.pt schema ----
    state = torch.load(final_ckpt, map_location="cpu")
    assert isinstance(state, dict), f"checkpoint state must be a dict, got {type(state)}"
    for key in ("epoch", "model", "optimizer", "scheduler", "scaler", "best_metric"):
        assert key in state, f"checkpoint_final.pt missing key {key!r}"
    assert state["epoch"] == 0, f"final checkpoint epoch should be 0, got {state['epoch']}"
    # best_metric was set on the only epoch run (val_robust_acc), should equal it.
    assert state["best_metric"] == rec["val_robust_acc"], (
        f"best_metric in final ckpt ({state['best_metric']}) should equal "
        f"val_robust_acc from the only epoch ({rec['val_robust_acc']})"
    )
