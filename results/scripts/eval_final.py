"""Final evaluation: Clean / PGD-20 / CW-20 / AutoAttack on a checkpoint_best.pt.

Usage:
    python results/scripts/eval_final.py --run trades_baseline --aa_subset 0
    python results/scripts/eval_final.py --run trades_ams      --aa_subset 1000

`--aa_subset 0` runs AutoAttack on the full test set (10000 samples).
`--aa_subset N>0` runs AutoAttack on the first N test samples (deterministic).
PGD-20 and CW-20 always run on the full test set.

Writes a single-checkpoint summary JSON to results/scripts/tmp/eval_<run>.json.
The aggregation into a markdown table happens in build_eval_table.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import cw_attack, evaluate_autoattack, pgd_attack
from src.data import get_cifar10_loaders
from src.models import preact_resnet18
from src.utils.seed import set_seed


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total


def pgd20_accuracy(model, loader, device, eps=8 / 255, alpha=2 / 255, steps=20):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = pgd_attack(
            model, x, y,
            eps=eps, alpha=alpha, steps=steps,
            objective="ce", random_start=True,
        )
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def cw20_accuracy(model, loader, device, eps=8 / 255, alpha=2 / 255, steps=20):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = cw_attack(
            model, x, y,
            eps=eps, alpha=alpha, steps=steps,
            kappa=0.0, random_start=True,
        )
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=["trades_baseline", "trades_ams"])
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--aa_subset", type=int, default=0,
                        help="0 = full test set; N>0 = first N samples (deterministic)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", default="",
                        help="comma-separated subset of {clean,pgd20,cw20,aa}")
    args = parser.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    set_seed(0, deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_dir = PROJECT_ROOT / "runs" / args.run
    ckpt_path = run_dir / args.checkpoint

    # Build model and load weights.
    model = preact_resnet18(num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # CIFAR-10 test loader.
    _, _, test_loader = get_cifar10_loaders(
        data_root=str(PROJECT_ROOT / "data"),
        batch_size=args.batch_size,
        val_size=1000,
        num_workers=2,
        seed=0,
        download=False,
    )

    print(f"[{args.run}] loaded {ckpt_path}; epoch={state.get('epoch')} best_metric={state.get('best_metric')}")
    print(f"[{args.run}] full test set size = {len(test_loader.dataset)}")

    results = {
        "run": args.run,
        "checkpoint": str(ckpt_path),
        "epoch": int(state.get("epoch", -1)),
        "best_metric": float(state.get("best_metric", -1.0)),
        "test_set_size": len(test_loader.dataset),
    }

    if "clean" not in skip:
        t0 = time.time()
        acc = clean_accuracy(model, test_loader, device)
        dt = time.time() - t0
        results["clean_acc"] = acc
        results["clean_time_sec"] = dt
        print(f"[{args.run}] clean acc = {acc:.4f}  ({dt:.1f}s)")

    if "pgd20" not in skip:
        t0 = time.time()
        acc = pgd20_accuracy(model, test_loader, device)
        dt = time.time() - t0
        results["pgd20_acc"] = acc
        results["pgd20_time_sec"] = dt
        print(f"[{args.run}] pgd20 acc = {acc:.4f}  ({dt:.1f}s)")

    if "cw20" not in skip:
        t0 = time.time()
        acc = cw20_accuracy(model, test_loader, device)
        dt = time.time() - t0
        results["cw20_acc"] = acc
        results["cw20_time_sec"] = dt
        print(f"[{args.run}] cw20 acc = {acc:.4f}  ({dt:.1f}s)")

    if "aa" not in skip:
        # Optionally restrict to a fixed subset for AutoAttack to control wall time.
        if args.aa_subset > 0:
            n = args.aa_subset
            indices = list(range(n))
            subset = Subset(test_loader.dataset, indices)
            aa_loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            results["aa_subset_size"] = n
            print(f"[{args.run}] AutoAttack on first {n} samples (subset)")
        else:
            aa_loader = test_loader
            results["aa_subset_size"] = len(test_loader.dataset)
            print(f"[{args.run}] AutoAttack on full test set ({len(test_loader.dataset)} samples)")
        t0 = time.time()
        acc = evaluate_autoattack(
            model,
            aa_loader,
            eps=8 / 255,
            version="standard",
            norm="Linf",
            device=str(device),
            verbose=False,
            seed=0,
        )
        dt = time.time() - t0
        results["aa_acc"] = acc
        results["aa_time_sec"] = dt
        print(f"[{args.run}] AutoAttack acc = {acc:.4f}  ({dt:.1f}s)")

    out_path = PROJECT_ROOT / "results" / "scripts" / "tmp" / f"eval_{args.run}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing file if present (so partial reruns can extend).
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            prev.update(results)
            results = prev
        except Exception:
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.run}] wrote {out_path}")


if __name__ == "__main__":
    main()
