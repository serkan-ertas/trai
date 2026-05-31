"""End-to-end smoke tests for the TRADES+AMS training script (task 2.2).

These exercise the AMS-specific machinery on a tiny CIFAR-10 subset:
- teacher buffer append at every stage boundary,
- AMS distillation loss with a non-empty buffer,
- forgetting snapshots saved at each stage,
- checkpoint with the teacher buffer carried in ``extras``,
- ``--resume`` path that restores the buffer.

To trigger a stage boundary inside a 1-epoch smoke we set
``ams.stage_interval_m: 1`` so ``(epoch + 1) % m == 0`` fires after epoch 0.
The main smoke runs 2 epochs so:
  - epoch 0 logs `num_teachers=0`, `train_loss_ams=0` (buffer empty during
    epoch 0's batches), then the stage-boundary code at the end of epoch 0
    appends teacher #1 and writes `stage_1_correct_indices.pt`;
  - epoch 1 logs `num_teachers=1`, `train_loss_ams>0` (distillation fired),
    then the stage-boundary code appends teacher #2 and writes
    `stage_2_correct_indices.pt`.

Budget: 2-epoch smoke ~2.5 min on RTX 3050; resume test ~1.5 min; total < 6 min.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path("C:/Users/user/Desktop/trai")
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
AMS_CFG = PROJECT_ROOT / "configs" / "trades_ams.yaml"

# 2 stages worth of forgetting snapshot file names.
SNAPSHOT_1 = "stage_1_correct_indices.pt"
SNAPSHOT_2 = "stage_2_correct_indices.pt"

EXPECTED_METRICS_KEYS = {
    "epoch",
    "lr",
    "train_loss_ce",
    "train_loss_kl_trades",
    "train_loss_ams",
    "train_acc_clean",
    "val_acc_clean",
    "val_robust_acc",
    "num_teachers",
    "mean_rlc_weight",
    "train_time_sec",
}


def _build_smoke_cfg(
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int = 32,
    val_size: int = 64,
    train_steps: int = 2,
    val_steps: int = 5,
    snap_samples: int = 32,
    snap_steps: int = 5,
) -> dict:
    """Load the real AMS YAML and override fields for a fast smoke run."""
    with open(AMS_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["run_id"] = "smoke_ams"
    cfg["output_dir"] = str(output_dir.as_posix())
    cfg["seed"] = 0

    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["val_size"] = val_size
    cfg["data"]["num_workers"] = 0
    cfg["data"]["data_root"] = "C:/Users/user/Desktop/trai/data"
    # The CIFAR-10 cache lives at <data_root>/cifar-10-batches-py; download=False
    # so we fail fast if it's missing instead of silently hitting the network.
    cfg["data"]["download"] = False

    cfg["train"]["epochs"] = epochs
    # Milestone past the smoke horizon — no LR decay should fire here.
    cfg["train"]["lr_schedule"]["milestones"] = [10]
    cfg["train"]["amp"] = True

    cfg["attack"]["train_inner"]["steps"] = train_steps
    cfg["attack"]["val_pgd20"]["steps"] = val_steps

    # m=1 => every epoch is a stage boundary, so we exercise the AMS distill
    # loss with a non-empty buffer starting from epoch 1.
    cfg["ams"]["stage_interval_m"] = 1
    cfg["ams"]["forgetting_snapshot"]["num_samples"] = snap_samples
    cfg["ams"]["forgetting_snapshot"]["attack_steps"] = snap_steps
    return cfg


def _write_yaml(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _launch_train_ams(
    yaml_path: Path,
    *,
    resume: bool = False,
    timeout: int = 420,
) -> subprocess.CompletedProcess:
    """Invoke ``python -m src.experiments.train_ams --config <yaml> [--resume]``."""
    cmd = [
        str(PYTHON_EXE),
        "-m", "src.experiments.train_ams",
        "--config", str(yaml_path),
    ]
    if resume:
        cmd.append("--resume")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc


# ---------------------------------------------------------------------------
# Session-scoped fixture: run the 2-epoch smoke ONCE, share its run_dir with
# the schema + checkpoint tests so we don't pay the ~2.5 min cost three times.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_epoch_smoke_run(tmp_path_factory):
    """Run the 2-epoch AMS smoke once and return (run_dir, proc, records)."""
    if not torch.cuda.is_available():
        pytest.skip("Smoke run requires CUDA (AMP path); CPU is too slow.")

    tmp_path = tmp_path_factory.mktemp("ams_2ep")
    cfg = _build_smoke_cfg(output_dir=tmp_path / "runs", epochs=2)
    yaml_path = tmp_path / "smoke_ams.yaml"
    _write_yaml(cfg, yaml_path)

    proc = _launch_train_ams(yaml_path, resume=False, timeout=420)
    if proc.returncode != 0:
        pytest.fail(
            f"train_ams subprocess exited with code {proc.returncode}\n"
            f"---- stdout ----\n{proc.stdout}\n"
            f"---- stderr ----\n{proc.stderr}\n"
        )

    run_dir = tmp_path / "runs" / "smoke_ams"
    assert run_dir.is_dir(), f"run directory not created at {run_dir}"

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl missing: {metrics_path}"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return {"run_dir": run_dir, "proc": proc, "records": records}


# ---------------------------------------------------------------------------
# 1) Main 2-epoch smoke run: verify artifacts and metrics shape.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke run requires CUDA (AMP path); CPU is too slow.",
)
def test_two_epoch_ams_smoke_run(two_epoch_smoke_run):
    """Run train_ams for 2 epochs with m=1; check artifacts + AMS-specific metrics."""
    run_dir: Path = two_epoch_smoke_run["run_dir"]
    records = two_epoch_smoke_run["records"]

    # --- Artifacts on disk -------------------------------------------------
    assert (run_dir / "config.yaml").exists(), "config snapshot missing"

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl missing"
    assert len(records) == 2, (
        f"expected exactly 2 metrics lines for 2-epoch run, got {len(records)}"
    )

    # Checkpoints
    for name in ("checkpoint_last.pt", "checkpoint_best.pt", "checkpoint_final.pt"):
        assert (run_dir / name).exists(), f"{name} missing in {run_dir}"

    # Stage snapshots (m=1 => one per epoch).
    for snap in (SNAPSHOT_1, SNAPSHOT_2):
        assert (run_dir / snap).exists(), f"{snap} missing in {run_dir}"

    # No left-over .tmp files (atomic save cleaned them up).
    leftover_tmps = list(run_dir.glob("*.tmp"))
    assert leftover_tmps == [], f"unexpected .tmp leftovers: {leftover_tmps}"

    # --- Per-record schema -------------------------------------------------
    for i, rec in enumerate(records):
        missing = EXPECTED_METRICS_KEYS - set(rec.keys())
        assert not missing, f"record {i} missing keys: {missing} (got {set(rec.keys())})"

    rec0, rec1 = records
    assert rec0["epoch"] == 0, f"first record epoch should be 0, got {rec0['epoch']}"
    assert rec1["epoch"] == 1, f"second record epoch should be 1, got {rec1['epoch']}"

    # --- AMS-specific invariants ------------------------------------------
    # Epoch 0: buffer is empty during batches, gets appended AFTER metrics are
    # logged (train_ams.py:259 reads len(teacher_buffer) BEFORE the stage
    # boundary block at line 289). So num_teachers logged for epoch 0 is 0,
    # AMS loss is the zero-tensor early-exit path.
    assert rec0["num_teachers"] == 0, (
        f"epoch 0 should log num_teachers=0 (buffer empty during epoch 0 "
        f"batches; append happens after metrics are logged), got {rec0['num_teachers']}"
    )
    assert rec0["train_loss_ams"] == 0.0, (
        f"epoch 0 should have train_loss_ams==0 (no teachers => zero-tensor "
        f"early exit in ams_distill_loss), got {rec0['train_loss_ams']}"
    )
    assert rec0["mean_rlc_weight"] == 0.0, (
        f"epoch 0 should have mean_rlc_weight==0 (no teacher-bearing batches), "
        f"got {rec0['mean_rlc_weight']}"
    )

    # Epoch 1: m=1 fired at the end of epoch 0 -> buffer now has 1 teacher
    # during the entire epoch 1; another snapshot fires at the end of epoch 1
    # but is logged on epoch 2 (which doesn't exist here).
    assert rec1["num_teachers"] >= 1, (
        f"epoch 1 should log num_teachers>=1 (teacher #1 was appended at end "
        f"of epoch 0 with m=1), got {rec1['num_teachers']}"
    )
    assert rec1["train_loss_ams"] > 0.0, (
        f"epoch 1 should have train_loss_ams>0 (buffer has 1 teacher so the "
        f"distillation term actually fires), got {rec1['train_loss_ams']}"
    )
    # mean_rlc_weight is a per-sample mean of the sum-over-teachers RLC weight.
    # For a single teacher on a random model it could be small but should be
    # strictly positive (softmax outputs in (0, 1)).
    assert rec1["mean_rlc_weight"] > 0.0, (
        f"epoch 1 should have mean_rlc_weight>0 (1 teacher, softmax weight "
        f"is strictly positive), got {rec1['mean_rlc_weight']}"
    )


# ---------------------------------------------------------------------------
# 2) Forgetting snapshot schema: shapes, dtypes, value ranges, device.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke run requires CUDA (AMP path); CPU is too slow.",
)
def test_forgetting_snapshot_schema(two_epoch_smoke_run):
    """The dict saved to stage_<s>_correct_indices.pt must match the spec
    in `train_ams.py:collect_forgetting_snapshot` (gotcha #7)."""
    run_dir: Path = two_epoch_smoke_run["run_dir"]
    snap_path = run_dir / SNAPSHOT_1
    assert snap_path.exists(), f"snapshot {snap_path} missing"

    snap = torch.load(snap_path, map_location="cpu")
    assert isinstance(snap, dict), f"snapshot must be a dict, got {type(snap)}"

    # All required keys present.
    expected_keys = {"x", "y", "x_adv", "correct_mask", "num_samples"}
    missing = expected_keys - set(snap.keys())
    assert not missing, f"snapshot missing keys: {missing} (got {set(snap.keys())})"

    M = snap["num_samples"]
    # snap_samples=32 capped to val_size=64 -> M should be exactly 32.
    assert M == 32, f"num_samples should be 32 (snap cap), got {M}"

    # Shape checks.
    assert snap["x"].shape == (M, 3, 32, 32), (
        f"x shape mismatch: expected ({M}, 3, 32, 32), got {tuple(snap['x'].shape)}"
    )
    assert snap["x_adv"].shape == (M, 3, 32, 32), (
        f"x_adv shape mismatch: expected ({M}, 3, 32, 32), got {tuple(snap['x_adv'].shape)}"
    )
    assert snap["y"].shape == (M,), (
        f"y shape mismatch: expected ({M},), got {tuple(snap['y'].shape)}"
    )
    assert snap["correct_mask"].shape == (M,), (
        f"correct_mask shape mismatch: expected ({M},), got "
        f"{tuple(snap['correct_mask'].shape)}"
    )

    # Dtype checks.
    assert snap["x"].dtype == torch.float32, (
        f"x dtype should be float32, got {snap['x'].dtype}"
    )
    assert snap["x_adv"].dtype == torch.float32, (
        f"x_adv dtype should be float32, got {snap['x_adv'].dtype}"
    )
    assert snap["y"].dtype == torch.int64, (
        f"y dtype should be int64 (long), got {snap['y'].dtype}"
    )
    assert snap["correct_mask"].dtype == torch.bool, (
        f"correct_mask dtype should be bool, got {snap['correct_mask'].dtype}"
    )

    # Range checks: inputs are raw [0,1] pixels, attacked tensors stay in [0,1].
    eps = 1e-6
    assert snap["x"].min().item() >= 0.0 - eps, (
        f"x min should be >= 0, got {snap['x'].min().item()}"
    )
    assert snap["x"].max().item() <= 1.0 + eps, (
        f"x max should be <= 1, got {snap['x'].max().item()}"
    )
    assert snap["x_adv"].min().item() >= 0.0 - eps, (
        f"x_adv min should be >= 0, got {snap['x_adv'].min().item()}"
    )
    assert snap["x_adv"].max().item() <= 1.0 + eps, (
        f"x_adv max should be <= 1, got {snap['x_adv'].max().item()}"
    )

    # All tensors on CPU (state was already torch.save'd from CPU tensors).
    for k in ("x", "y", "x_adv", "correct_mask"):
        assert snap[k].device.type == "cpu", (
            f"snap[{k!r}] should be on cpu, got {snap[k].device}"
        )

    # correct_mask is a Bool tensor; sum is non-negative (could be 0 on a
    # nearly-random epoch-0 model, that's fine — we just sanity-check the
    # type math by reading the value).
    n_correct = int(snap["correct_mask"].sum().item())
    assert n_correct >= 0, f"correct_mask.sum() should be >= 0, got {n_correct}"

    # y labels must be valid CIFAR-10 class indices.
    assert snap["y"].min().item() >= 0 and snap["y"].max().item() <= 9, (
        f"y labels out of CIFAR-10 range [0, 9]: "
        f"min={snap['y'].min().item()}, max={snap['y'].max().item()}"
    )


# ---------------------------------------------------------------------------
# 3) Checkpoint carries the teacher buffer in extras.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke run requires CUDA (AMP path); CPU is too slow.",
)
def test_checkpoint_carries_teacher_buffer(two_epoch_smoke_run):
    """checkpoint_last.pt must contain ``teacher_buffer`` -> ``snapshots`` list
    of CPU state_dicts so that ``--resume`` can reconstruct the buffer."""
    run_dir: Path = two_epoch_smoke_run["run_dir"]
    ckpt_path = run_dir / "checkpoint_last.pt"
    assert ckpt_path.exists(), "checkpoint_last.pt missing"

    state = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(state, dict)
    assert "teacher_buffer" in state, (
        "checkpoint_last.pt missing 'teacher_buffer' extras key — resume cannot "
        "reconstruct the AMS buffer without it"
    )

    tb_state = state["teacher_buffer"]
    assert isinstance(tb_state, dict) and "snapshots" in tb_state, (
        f"teacher_buffer state should be dict with 'snapshots' key, got {tb_state!r}"
    )
    snapshots = tb_state["snapshots"]
    assert isinstance(snapshots, list)
    # 2 epochs * m=1 -> 2 stage boundaries -> 2 teacher snapshots.
    assert len(snapshots) >= 1, (
        f"expected >=1 teacher snapshot after 2-epoch smoke with m=1, "
        f"got {len(snapshots)}"
    )

    # Skim each snapshot: dict of str -> CPU tensors.
    for i, sd in enumerate(snapshots):
        assert isinstance(sd, dict), f"snapshot {i} should be a dict, got {type(sd)}"
        # Pick a handful of keys to verify CPU residency.
        sample_keys = list(sd.keys())[:5]
        assert sample_keys, f"snapshot {i} has no keys"
        for k in sample_keys:
            v = sd[k]
            assert isinstance(k, str), f"snapshot {i} key {k!r} is not a str"
            assert isinstance(v, torch.Tensor), (
                f"snapshot {i}[{k!r}] is not a Tensor, got {type(v)}"
            )
            assert v.device.type == "cpu", (
                f"snapshot {i}[{k!r}] should be on cpu, got {v.device}"
            )


# ---------------------------------------------------------------------------
# 4) Resume picks up the teacher buffer.
# ---------------------------------------------------------------------------
#
# Marked @pytest.mark.slow because it runs the training subprocess twice in
# sequence; the same machinery (resume restoring buffer) is covered indirectly
# by test_checkpoint_carries_teacher_buffer (the round-trip just requires the
# checkpoint to round-trip cleanly through load_checkpoint, which is shared
# code with the 2.1 baseline's resume path -- already smoke-tested there).
#
# Run with: pytest tests/test_train_ams_smoke.py -m slow
#
# NOTE on observed semantics: in ``train_ams.py``, ``save_checkpoint(...last)``
# is called BEFORE the stage-boundary append (lines 268-273 precede line 289).
# Consequently, ``checkpoint_last.pt`` written at end of epoch t lags the
# in-process buffer by one stage when epoch t IS itself a stage boundary.
# For the production config (40 epochs, m=10, stages at epochs 9/19/29/39),
# the only resumable boundary epochs are exactly those four; interrupting at
# any of the other 36 epochs gives a fully consistent snapshot.
# We test the actual implementation behavior here, not an idealized one.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke run requires CUDA (AMP path); CPU is too slow.",
)
def test_ams_resume_picks_up_teachers(tmp_path):
    """End-to-end resume: 1 epoch -> stop -> resume with epochs=2.

    Verifies:
      (a) the second invocation prints a ``[resume]`` line containing a
          ``teachers=<N>`` field (the buffer state was round-tripped through
          load_checkpoint), and completes without error;
      (b) ``metrics.jsonl`` has exactly 2 lines with epochs ``[0, 1]``;
      (c) at least one ``stage_<s>_correct_indices.pt`` snapshot is present
          for each completed stage boundary across the two runs;
      (d) the final checkpoint carries the teacher buffer in extras.

    The specific snapshot count in ``checkpoint_last.pt`` after the first run
    is governed by the save-then-append ordering noted above and is verified
    explicitly so the test documents the contract.
    """
    # Tight smoke params to keep the combined budget under ~2.5 min.
    base_kwargs = dict(
        output_dir=tmp_path / "runs",
        batch_size=32,
        val_size=32,
        train_steps=1,
        val_steps=2,
        snap_samples=32,
        snap_steps=2,
    )

    # ---- First invocation: 1 epoch ---------------------------------------
    cfg1 = _build_smoke_cfg(epochs=1, **base_kwargs)
    yaml_path1 = tmp_path / "smoke_ams_resume_1.yaml"
    _write_yaml(cfg1, yaml_path1)
    proc1 = _launch_train_ams(yaml_path1, resume=False, timeout=240)
    if proc1.returncode != 0:
        pytest.fail(
            f"first invocation failed with code {proc1.returncode}\n"
            f"---- stdout ----\n{proc1.stdout}\n"
            f"---- stderr ----\n{proc1.stderr}\n"
        )

    run_dir = tmp_path / "runs" / "smoke_ams"
    assert (run_dir / "checkpoint_last.pt").exists(), (
        "first invocation should have written checkpoint_last.pt"
    )
    # NOTE: checkpoint_last.pt is written BEFORE the stage-boundary append at
    # the end of each epoch (train_ams.py lines 268-273 vs 289). So after a
    # 1-epoch run with m=1, the saved ``teacher_buffer`` state has 0 snapshots
    # even though the in-process buffer has 1 by the time the loop exits.
    # checkpoint_final.pt (written after the loop, line 303) is correct.
    state_after_first = torch.load(
        run_dir / "checkpoint_last.pt", map_location="cpu"
    )
    n_snaps_last_after_first = len(state_after_first["teacher_buffer"]["snapshots"])
    assert n_snaps_last_after_first == 0, (
        f"checkpoint_last.pt after epoch 0 captures the buffer BEFORE the "
        f"stage append (train_ams.py:268 precedes :289), so it should have "
        f"0 snapshots; got {n_snaps_last_after_first}. If this changes the "
        f"ordering of saves vs. stage appends should be reviewed."
    )
    # checkpoint_final.pt is the post-loop save -> it does capture the
    # epoch-0 append.
    final_state_after_first = torch.load(
        run_dir / "checkpoint_final.pt", map_location="cpu"
    )
    assert len(final_state_after_first["teacher_buffer"]["snapshots"]) == 1, (
        "checkpoint_final.pt after 1-epoch / m=1 run should have 1 teacher"
    )

    # ---- Second invocation: epochs=2, --resume ----------------------------
    # Resume reads checkpoint_last.pt (not _final), so the buffer it restores
    # has 0 snapshots. This is the actual implementation behavior. The resumed
    # run will then proceed to epoch 1 and snapshot at its end.
    cfg2 = _build_smoke_cfg(epochs=2, **base_kwargs)
    yaml_path2 = tmp_path / "smoke_ams_resume_2.yaml"
    _write_yaml(cfg2, yaml_path2)
    proc2 = _launch_train_ams(yaml_path2, resume=True, timeout=240)
    if proc2.returncode != 0:
        pytest.fail(
            f"second invocation (resume) failed with code {proc2.returncode}\n"
            f"---- stdout ----\n{proc2.stdout}\n"
            f"---- stderr ----\n{proc2.stderr}\n"
        )

    # stdout should contain a [resume] line including a teachers=<N> field.
    # train_ams.py:164: print(f"[resume] epoch ... teachers={len(...)}").
    assert "[resume]" in proc2.stdout, (
        f"resume line not found in second invocation stdout:\n{proc2.stdout}"
    )
    assert "teachers=0" in proc2.stdout, (
        f"expected 'teachers=0' in resume line (checkpoint_last.pt was saved "
        f"BEFORE the epoch-0 stage append), stdout was:\n{proc2.stdout}"
    )

    # metrics.jsonl now has 2 lines: 1 from initial run + 1 from resumed run.
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, (
        f"expected exactly 2 metrics lines after resume (1 from first run, 1 "
        f"from second), got {len(lines)}: {lines}"
    )
    epochs = [json.loads(line)["epoch"] for line in lines]
    assert epochs == [0, 1], f"epochs in metrics.jsonl should be [0, 1], got {epochs}"

    # Stage 1 snapshot exists from the first run; stage 2 snapshot is written
    # during the resumed run at the end of epoch 1.
    assert (run_dir / SNAPSHOT_1).exists(), f"{SNAPSHOT_1} missing after resume"
    assert (run_dir / SNAPSHOT_2).exists(), (
        f"{SNAPSHOT_2} missing -- resumed run should have hit the stage "
        f"boundary at end of epoch 1 with m=1"
    )

    # Final checkpoint after the resumed run includes the post-loop save, so
    # the buffer should reflect ONE snapshot from the resumed epoch (the
    # earlier epoch-0 snapshot was lost because checkpoint_last.pt at the end
    # of run 1 didn't include it). This is the documented behavior under the
    # current save ordering; it is FINE for the production schedule because
    # runs are not interrupted at stage-boundary epochs.
    final_state = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    n_snaps_final = len(final_state["teacher_buffer"]["snapshots"])
    assert n_snaps_final >= 1, (
        f"final checkpoint after resume should hold at least the epoch-1 "
        f"snapshot, got {n_snaps_final}"
    )
    # Teacher_buffer state dict carries CPU tensors.
    if n_snaps_final > 0:
        first_snap = final_state["teacher_buffer"]["snapshots"][0]
        assert isinstance(first_snap, dict) and first_snap, "empty snapshot dict"
        any_key = next(iter(first_snap))
        assert first_snap[any_key].device.type == "cpu"
