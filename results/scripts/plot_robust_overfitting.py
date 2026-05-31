"""Plot PGD-20 val acc vs epoch for both runs and emit the Best-vs-Final table.

Saves:
- results/plots/robust_overfitting.png
- results/tables/best_vs_final.md
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_metrics(run_dir: Path):
    """Return list of dicts, one per epoch."""
    rows = []
    with open(run_dir / "metrics.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["epoch"])
    return rows


def best_and_final(rows):
    """Return (best_epoch, best_robust, final_robust)."""
    best_epoch, best_robust = max(
        ((r["epoch"], r["val_robust_acc"]) for r in rows), key=lambda t: t[1]
    )
    final_robust = rows[-1]["val_robust_acc"]
    return best_epoch, best_robust, final_robust


def main():
    baseline = load_metrics(PROJECT_ROOT / "runs" / "trades_baseline")
    ams = load_metrics(PROJECT_ROOT / "runs" / "trades_ams")

    bl_epochs = [r["epoch"] for r in baseline]
    bl_robust = [r["val_robust_acc"] for r in baseline]
    am_epochs = [r["epoch"] for r in ams]
    am_robust = [r["val_robust_acc"] for r in ams]

    bl_best_ep, bl_best, bl_final = best_and_final(baseline)
    am_best_ep, am_best, am_final = best_and_final(ams)

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        bl_epochs, bl_robust,
        color="gray", linewidth=1.8, marker="o", markersize=3.5,
        label="TRADES baseline",
    )
    ax.plot(
        am_epochs, am_robust,
        color="#ff7f0e", linewidth=1.8, marker="s", markersize=3.5,
        label="TRADES+AMS (ours)",
    )
    # Best-epoch verticals.
    ax.axvline(bl_best_ep, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(am_best_ep, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.7)
    # LR-decay verticals at 30 and 36.
    for ep in (30, 36):
        ax.axvline(ep, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Robust val acc (PGD-20, 1000 samples)")
    ax.set_title(
        "PGD-20 Robust Validation Accuracy vs Epoch\n"
        "(CIFAR-10, PreActResNet-18, $\\epsilon$=8/255)"
    )
    ax.set_xlim(-0.5, max(max(bl_epochs), max(am_epochs)) + 0.5)
    ax.set_ylim(0.2, 0.55)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    # Annotate best
    ax.annotate(
        f"baseline best={bl_best:.3f} @ ep{bl_best_ep}",
        xy=(bl_best_ep, bl_best), xytext=(2, 0.52),
        fontsize=8, color="gray",
    )
    ax.annotate(
        f"AMS best={am_best:.3f} @ ep{am_best_ep}",
        xy=(am_best_ep, am_best), xytext=(2, 0.50),
        fontsize=8, color="#ff7f0e",
    )
    out_png = PROJECT_ROOT / "results" / "plots" / "robust_overfitting.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    # ----- Best vs Final table -----
    table = (
        "# Best vs Final PGD-20 Validation Accuracy\n"
        "\n"
        "Robust validation accuracy at the best-checkpoint epoch versus the\n"
        "final-epoch checkpoint, both measured by PGD-20 (ε=8/255, α=2/255, 20\n"
        "steps, random start) on 1000 held-out validation samples. Smaller Diff\n"
        "indicates less robust overfitting (paper Table 6 / Section 5.3).\n"
        "\n"
        "| Method | Best epoch | Best PGD-20 (val) | Final PGD-20 (val) | Diff (best − final) |\n"
        "|---|---|---|---|---|\n"
        f"| TRADES (baseline) | {bl_best_ep} | {bl_best*100:.2f} | {bl_final*100:.2f} | {(bl_best - bl_final)*100:+.2f} |\n"
        f"| TRADES+AMS (ours) | {am_best_ep} | {am_best*100:.2f} | {am_final*100:.2f} | {(am_best - am_final)*100:+.2f} |\n"
        "\n"
        "*At 40 epochs both gaps are essentially zero — robust overfitting is\n"
        "barely present at this reduced training horizon. The paper's claim\n"
        "(AMS reduces Diff from ~2-3% to ~0.5% on PreActResNet-18 at 200\n"
        "epochs) cannot be directionally tested at our scale because the\n"
        "baseline shows almost no overfitting either. Both numbers are\n"
        "consistent with the experimentalist's RUN_LOG headline.*\n"
    )
    out_table = PROJECT_ROOT / "results" / "tables" / "best_vs_final.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(table, encoding="utf-8")
    print(f"wrote {out_table}")


if __name__ == "__main__":
    main()
