"""TRADES+AMS training script for CIFAR-10 / PreActResNet-18 (task 2.2).

End-to-end entrypoint:
    python -m src.experiments.train_ams --config configs/trades_ams.yaml [--resume]

Implements Algorithm 1 of Wang & Ding (ICLR 2025): the TRADES outer objective
(paper Eq. 3) plus the AMS multi-teacher self-distillation term (Eqs. 7, 8)
with RLC reweighting. Differs from ``train_trades.py`` in three places:

  (a) builds a :class:`TeacherBuffer` and runs each snapshot's forward on the
      current batch's adversarial examples (under ``no_grad``, OUTSIDE AMP);
  (b) per-batch loss is ``trades_loss(...) + ams_distill_loss(...)``;
  (c) at every stage boundary (epochs t where ``(t + 1) % m == 0``) the script
      appends the current model to the buffer and writes a forgetting
      verification snapshot to ``stage_<s>_correct_indices.pt`` for the
      Phase 4 FAA / FF analysis (the project spec gotcha #7).

AMP wraps the OUTER forward/loss only; the inner PGD and teacher forwards run
in FP32 with ``torch.no_grad()`` (the project spec gotchas #4 and #8). The "best"
checkpoint tracks PGD-20 robust val accuracy (gotcha #6).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.attacks import pgd_attack
from src.data import get_cifar10_loaders
from src.experiments.common import (
    JsonlLogger, evaluate_pgd20, load_checkpoint,
    load_config, make_run_dir, save_checkpoint,
)
from src.losses import ams_distill_loss, trades_loss
from src.models import preact_resnet18
from src.training import TeacherBuffer
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRADES+AMS training (task 2.2).")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from runs/<run_id>/checkpoint_last.pt if present.")
    return p.parse_args()


def collect_forgetting_snapshot(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float,
    alpha: float,
    steps: int,
    num_samples: int,
    save_path: Path,
) -> None:
    """Save a fixed PGD adversarial subset + the model's correctness mask at
    this stage. Used in Phase 4 (analyst, task 4.4) to compute the FAA and FF
    forgetting metrics (paper Table 1 / Table 12).

    The dict written via ``torch.save`` has keys:
        ``x``            (M, 3, 32, 32) float32 — clean tensors in [0,1]
        ``y``            (M,) long       — true labels
        ``x_adv``        (M, 3, 32, 32) float32 — adversarial tensors in [0,1]
        ``correct_mask`` (M,) bool       — was the sample correctly classified
                                            ON x_adv by the model AT THIS STAGE
        ``num_samples``  int             — M = min(num_samples, len(val))

    Gotcha #7: correctness must be evaluated by the model that existed at this
    stage boundary, not reconstructed from final checkpoints.
    """
    was_training = model.training
    model.eval()
    try:
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        collected = 0
        for x, y in val_loader:
            need = num_samples - collected
            if need <= 0:
                break
            xs.append(x[:need])
            ys.append(y[:need])
            collected += min(need, x.size(0))
        x_clean = torch.cat(xs, dim=0).to(device)
        y_all = torch.cat(ys, dim=0).to(device)

        # PGD-K, CE objective on raw [0,1] inputs (Eq. 6 / standard PGD-20 eval).
        x_adv = pgd_attack(model, x_clean, y_all, eps=eps, alpha=alpha,
                           steps=steps, objective="ce", random_start=True)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        correct_mask = preds == y_all

        torch.save({
            "x": x_clean.detach().cpu(),
            "y": y_all.detach().cpu(),
            "x_adv": x_adv.detach().cpu(),
            "correct_mask": correct_mask.detach().cpu(),
            "num_samples": int(x_clean.size(0)),
        }, save_path)
    finally:
        model.train(was_training)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(cfg["output_dir"], cfg["run_id"])
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader, _test_loader = get_cifar10_loaders(
        data_root=cfg["data"]["data_root"],
        batch_size=cfg["data"]["batch_size"],
        val_size=cfg["data"]["val_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
        download=cfg["data"]["download"],
    )

    model = preact_resnet18(num_classes=cfg["model"]["num_classes"]).to(device)

    opt_cfg = cfg["train"]["optimizer"]
    opt = torch.optim.SGD(
        model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg["momentum"],
        nesterov=opt_cfg["nesterov"], weight_decay=opt_cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=cfg["train"]["lr_schedule"]["milestones"],
        gamma=cfg["train"]["lr_schedule"]["gamma"],
    )

    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Teacher buffer: CPU-resident snapshots, one shared GPU scratch model (gotcha #9).
    factory = lambda: preact_resnet18(num_classes=cfg["model"]["num_classes"])
    teacher_buffer = TeacherBuffer(model_factory=factory, device=device)

    start_epoch = 0
    best_robust = -1.0
    if args.resume and (run_dir / "checkpoint_last.pt").exists():
        state = load_checkpoint(
            run_dir / "checkpoint_last.pt",
            model=model, optimizer=opt, scheduler=sched, scaler=scaler,
            map_location=device,
        )
        start_epoch = state["epoch"] + 1
        best_robust = state.get("best_metric", -1.0)
        if "teacher_buffer" in state:
            teacher_buffer.load_state_dict(state["teacher_buffer"])
        print(f"[resume] epoch {start_epoch} best_robust={best_robust:.4f} "
              f"teachers={len(teacher_buffer)}")

    logger = JsonlLogger(run_dir / "metrics.jsonl", resume=args.resume)

    inner_cfg = cfg["attack"]["train_inner"]
    val_cfg = cfg["attack"]["val_pgd20"]
    ams_cfg = cfg["ams"]
    snap_cfg = ams_cfg["forgetting_snapshot"]
    m_stage = int(ams_cfg["stage_interval_m"])
    beta = float(cfg["loss"]["trades_beta"])
    lam = float(ams_cfg["lam"])
    rlc_eps = float(ams_cfg["rlc_eps"])

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        running = {"loss_ce": 0.0, "loss_kl_trades": 0.0, "loss_ams": 0.0,
                   "correct": 0, "total": 0,
                   "weight_sum": 0.0, "samples_with_teachers": 0}

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Inner attack OUTSIDE autocast (FP32) — sign attack is fragile under FP16.
            x_adv = pgd_attack(
                model, x, y,
                eps=inner_cfg["eps"], alpha=inner_cfg["alpha"],
                steps=inner_cfg["steps"], objective=inner_cfg["objective"],
                random_start=inner_cfg["random_start"],
            )

            # Teacher logits: no_grad + OUTSIDE autocast (gotcha #4). The buffer
            # iterates by swapping each snapshot's weights into the shared
            # scratch model, so we ``.clone()`` each output before the next
            # load_state_dict mutates the underlying tensors. Teachers have
            # requires_grad_(False) and we are inside no_grad(), so no autograd
            # graph is attached — the clone is cheap and just copies storage.
            teacher_logits: list[torch.Tensor] = []
            if len(teacher_buffer) > 0:
                with torch.no_grad():
                    for teacher in teacher_buffer:
                        teacher_logits.append(teacher(x_adv).clone())

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_clean = model(x)
                logits_adv = model(x_adv)
                trades_l, trades_parts = trades_loss(
                    logits_clean, logits_adv, y, beta=beta)
                ams_l, ams_parts = ams_distill_loss(
                    student_logits_adv=logits_adv,
                    teacher_logits_adv=teacher_logits,
                    y=y, lam=lam, eps=rlc_eps,
                )
                loss = trades_l + ams_l

            # Standard AMP recipe (gotcha #8): scale -> backward -> step -> update.
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                bs = x.size(0)
                running["loss_ce"] += trades_parts["loss_ce"] * bs
                running["loss_kl_trades"] += trades_parts["loss_kl"] * bs
                running["loss_ams"] += ams_parts["loss_ams"] * bs
                running["correct"] += (logits_clean.argmax(dim=1) == y).sum().item()
                running["total"] += bs
                if ams_parts["num_teachers"] > 0:
                    running["weight_sum"] += ams_parts["mean_total_weight"] * bs
                    running["samples_with_teachers"] += bs

            pbar.set_postfix(loss=f"{loss.item():.3f}", n_t=len(teacher_buffer))

        sched.step()
        train_time = time.time() - t0

        val_metrics = evaluate_pgd20(
            model, val_loader, device,
            eps=val_cfg["eps"], alpha=val_cfg["alpha"], steps=val_cfg["steps"],
        )

        total = running["total"]
        record = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_loss_ce": running["loss_ce"] / total,
            "train_loss_kl_trades": running["loss_kl_trades"] / total,
            "train_loss_ams": running["loss_ams"] / total,
            "train_acc_clean": running["correct"] / total,
            "val_acc_clean": val_metrics["clean_acc"],
            "val_robust_acc": val_metrics["robust_acc"],
            "num_teachers": len(teacher_buffer),
            "mean_rlc_weight": (running["weight_sum"] / running["samples_with_teachers"]
                                if running["samples_with_teachers"] > 0 else 0.0),
            "train_time_sec": train_time,
        }
        logger.log(record)
        print(json.dumps(record))

        # Atomic "last" checkpoint (used by --resume). Teacher buffer in extras.
        save_checkpoint(
            run_dir / "checkpoint_last.pt",
            epoch=epoch, model=model, optimizer=opt, scheduler=sched,
            scaler=scaler, best_metric=best_robust,
            extras={"teacher_buffer": teacher_buffer.state_dict()},
        )

        # Best by PGD-20 robust val accuracy (gotcha #6 / paper Section 5).
        if val_metrics["robust_acc"] > best_robust:
            best_robust = val_metrics["robust_acc"]
            save_checkpoint(
                run_dir / "checkpoint_best.pt",
                epoch=epoch, model=model, optimizer=opt, scheduler=sched,
                scaler=scaler, best_metric=best_robust,
                extras={"teacher_buffer": teacher_buffer.state_dict()},
            )

        # Stage boundary — Algorithm 1, lines 5-6. Paper uses 1-indexed epochs:
        # snapshot at end of epoch t where t % m == 0. Our 0-indexed loop var
        # epoch in {0..T-1}, so snapshot AFTER epochs where (epoch + 1) % m == 0.
        # 40 epochs / m=10 -> snapshots after epochs 9, 19, 29, 39 (4 stages).
        if (epoch + 1) % m_stage == 0:
            teacher_buffer.append(model)
            stage_idx = (epoch + 1) // m_stage  # 1, 2, 3, ...
            snap_path = run_dir / f"stage_{stage_idx}_correct_indices.pt"
            collect_forgetting_snapshot(
                model=model, val_loader=val_loader, device=device,
                eps=val_cfg["eps"], alpha=val_cfg["alpha"],
                steps=snap_cfg["attack_steps"],
                num_samples=snap_cfg["num_samples"],
                save_path=snap_path,
            )
            print(f"[stage {stage_idx}] appended teacher (now {len(teacher_buffer)}); "
                  f"saved forgetting snapshot -> {snap_path.name}")

    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        epoch=cfg["train"]["epochs"] - 1, model=model, optimizer=opt,
        scheduler=sched, scaler=scaler, best_metric=best_robust,
        extras={"teacher_buffer": teacher_buffer.state_dict()},
    )

    logger.close()
    print(f"[done] best robust val acc = {best_robust:.4f}; "
          f"teachers = {len(teacher_buffer)}")


if __name__ == "__main__":
    main()
