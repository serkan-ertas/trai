"""TRADES baseline training script for CIFAR-10 / PreActResNet-18 (task 2.1).

End-to-end entrypoint:
    python -m src.experiments.train_trades --config configs/trades_baseline.yaml [--resume]

Implements the TRADES outer objective (paper Eq. 3) with a PGD-5 inner attack
(reduced from the paper's PGD-10 — see the project spec reduction table). Mixed
precision (AMP) is enabled around the OUTER forward/loss only — the inner
PGD runs in FP32 because the sign attack is fragile under FP16.

Per-epoch JSONL metrics are appended to ``runs/<run_id>/metrics.jsonl``.
Two checkpoints are maintained:
    * ``checkpoint_best.pt``  — highest PGD-20 robust validation accuracy
      (the project spec gotcha #6 / paper Section 5).
    * ``checkpoint_last.pt``  — every epoch, atomic, used by ``--resume``.
A ``checkpoint_final.pt`` is written at the end of training.
"""

import argparse
import json
import time

import torch
import yaml
from tqdm import tqdm

from src.attacks import pgd_attack
from src.data import get_cifar10_loaders
from src.experiments.common import (
    JsonlLogger,
    evaluate_pgd20,
    load_checkpoint,
    load_config,
    make_run_dir,
    save_checkpoint,
)
from src.losses import trades_loss
from src.models import preact_resnet18
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRADES baseline training (task 2.1).")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from runs/<run_id>/checkpoint_last.pt if present.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(cfg["output_dir"], cfg["run_id"])
    # Snapshot the resolved config into the run dir for reproducibility.
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
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        nesterov=opt_cfg["nesterov"],
        weight_decay=opt_cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=cfg["train"]["lr_schedule"]["milestones"],
        gamma=cfg["train"]["lr_schedule"]["gamma"],
    )

    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_robust = -1.0

    if args.resume and (run_dir / "checkpoint_last.pt").exists():
        state = load_checkpoint(
            run_dir / "checkpoint_last.pt",
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = state["epoch"] + 1
        best_robust = state.get("best_metric", -1.0)
        print(f"[resume] starting from epoch {start_epoch}, best_robust={best_robust:.4f}")

    logger = JsonlLogger(run_dir / "metrics.jsonl", resume=args.resume)

    inner_cfg = cfg["attack"]["train_inner"]
    val_cfg = cfg["attack"]["val_pgd20"]
    beta = float(cfg["loss"]["trades_beta"])

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()

        running_loss_ce = 0.0
        running_loss_kl = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Inner attack runs OUTSIDE autocast (FP32) — sign attack is fragile
            # under FP16. pgd_attack handles model.eval()/restore internally.
            x_adv = pgd_attack(
                model, x, y,
                eps=inner_cfg["eps"],
                alpha=inner_cfg["alpha"],
                steps=inner_cfg["steps"],
                objective=inner_cfg["objective"],
                random_start=inner_cfg["random_start"],
            )

            # Outer step under AMP autocast.
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_clean = model(x)
                logits_adv = model(x_adv)
                loss, parts = trades_loss(logits_clean, logits_adv, y, beta=beta)

            # Standard AMP recipe (gotcha #8): scale -> backward -> step -> update.
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                bs = x.size(0)
                running_loss_ce += parts["loss_ce"] * bs
                running_loss_kl += parts["loss_kl"] * bs
                running_correct += (logits_clean.argmax(dim=1) == y).sum().item()
                running_total += bs

            pbar.set_postfix(loss=f"{loss.item():.3f}")

        sched.step()
        train_time = time.time() - t0

        # Validation — PGD-20 robust acc + clean acc on the 1k val split.
        val_metrics = evaluate_pgd20(
            model, val_loader, device,
            eps=val_cfg["eps"],
            alpha=val_cfg["alpha"],
            steps=val_cfg["steps"],
        )

        record = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_loss_ce": running_loss_ce / running_total,
            "train_loss_kl": running_loss_kl / running_total,
            "train_acc_clean": running_correct / running_total,
            "val_acc_clean": val_metrics["clean_acc"],
            "val_robust_acc": val_metrics["robust_acc"],
            "train_time_sec": train_time,
        }
        logger.log(record)
        print(json.dumps(record))

        # Atomic "last" checkpoint (used by --resume).
        save_checkpoint(
            run_dir / "checkpoint_last.pt",
            epoch=epoch,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            best_metric=best_robust,
        )

        # Best by PGD-20 robust val accuracy (gotcha #6 / paper Section 5).
        if val_metrics["robust_acc"] > best_robust:
            best_robust = val_metrics["robust_acc"]
            save_checkpoint(
                run_dir / "checkpoint_best.pt",
                epoch=epoch,
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                best_metric=best_robust,
            )

    # Final checkpoint (separate file, never overwritten by best).
    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        epoch=cfg["train"]["epochs"] - 1,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        best_metric=best_robust,
    )

    logger.close()
    print(f"[done] best robust val acc = {best_robust:.4f}")


if __name__ == "__main__":
    main()
