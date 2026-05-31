"""Aggregate per-run eval JSONs (results/scripts/tmp/eval_<run>.json) into the
markdown table at results/tables/final_evaluation.md.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPER_REF = {
    "clean": 84.65,
    "pgd20": 56.68,
    "cw20": 54.49,
    "aa": 53.00,
}


def load_run(run):
    p = PROJECT_ROOT / "results" / "scripts" / "tmp" / f"eval_{run}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{x*100:.2f}" if x is not None else "—"


def signed_diff(a, b):
    """Format (a - b) * 100 as +X.XX or -X.XX pp."""
    if a is None or b is None:
        return "—"
    return f"{(a - b) * 100:+.2f}"


def main():
    bl = load_run("trades_baseline")
    am = load_run("trades_ams")

    # Figure out whether AA was on a subset or full test set.
    bl_aa_n = bl.get("aa_subset_size", None)
    am_aa_n = am.get("aa_subset_size", None)
    aa_label = "AutoAttack"
    aa_footnote = ""
    if bl_aa_n is not None and am_aa_n is not None:
        if bl_aa_n == am_aa_n and bl_aa_n < 10000:
            aa_label = f"AutoAttack (1st {bl_aa_n})"
            aa_footnote = (
                f"\n*AutoAttack was evaluated on the first {bl_aa_n} CIFAR-10 test\n"
                f"samples (deterministic order, seed=0) rather than the full 10000-sample test\n"
                f"set because each full-test-set AutoAttack pass would have taken roughly\n"
                f"5 hours on the available RTX 3050 Mobile (∼173 s per 100 samples in the\n"
                f"timing probe; see DECISIONS log). Clean / PGD-20 / CW-20 numbers are on the\n"
                f"full test set.*"
            )
        elif bl_aa_n != am_aa_n:
            aa_footnote = (
                f"\n*AutoAttack ran on {bl_aa_n} samples for baseline and {am_aa_n} for AMS;\n"
                f"comparison is approximate.*"
            )

    md = (
        "# Final Evaluation — TRADES baseline vs TRADES+AMS\n"
        "\n"
        "All numbers are accuracies in %. ε = 8/255 (ℓ∞). PGD-20 and CW-20 use\n"
        "α=2/255 and random start; CW-20 uses the Carlini-Wagner margin loss with\n"
        "κ=0. AutoAttack uses the `standard` version (APGD-CE + APGD-DLR + FAB-T +\n"
        "Square, untargeted). Both rows use the `checkpoint_best.pt` selected by\n"
        "PGD-20 val accuracy during training (paper §5; baseline epoch 37, AMS\n"
        "epoch 36).\n"
        "\n"
        f"| Method | Clean | PGD-20 | CW-20 | {aa_label} |\n"
        "|---|---|---|---|---|\n"
        f"| TRADES (baseline, our scale) | {pct(bl.get('clean_acc'))} | {pct(bl.get('pgd20_acc'))} | {pct(bl.get('cw20_acc'))} | {pct(bl.get('aa_acc'))} |\n"
        f"| TRADES+AMS (ours)            | {pct(am.get('clean_acc'))} | {pct(am.get('pgd20_acc'))} | {pct(am.get('cw20_acc'))} | {pct(am.get('aa_acc'))} |\n"
        f"| Δ vs baseline (pp)           | {signed_diff(am.get('clean_acc'), bl.get('clean_acc'))} | {signed_diff(am.get('pgd20_acc'), bl.get('pgd20_acc'))} | {signed_diff(am.get('cw20_acc'), bl.get('cw20_acc'))} | {signed_diff(am.get('aa_acc'), bl.get('aa_acc'))} |\n"
        f"| TRADES (paper ref, PreActResNet-18) | {PAPER_REF['clean']:.2f} | {PAPER_REF['pgd20']:.2f} | {PAPER_REF['cw20']:.2f} | {PAPER_REF['aa']:.2f} |\n"
        "\n"
        "**Paper-reference row caveat.** The `TRADES (paper ref)` row reproduces\n"
        "the TRADES / PreActResNet-18 / CIFAR-10 baseline column from the AMS\n"
        "paper's Table 2 (Wang & Ding, ICLR 2025). That row is *at the paper's full\n"
        "training scale* (200 epochs, PGD-10 inner attack, m=20, single seed) and\n"
        "is **not directly comparable** to our 40-epoch / PGD-5-inner / m=10 reduced\n"
        "reproduction. We include it only as a sanity anchor for the order of\n"
        "magnitude.\n"
        "\n"
        "**Reduced-scale stance reminder.** Per `state/DECISIONS.md`, this project\n"
        "targets directional verification of AMS at a laptop-GPU scale (RTX 3050\n"
        "Mobile, ~4 GB VRAM, 3-4 h total training budget). We expect our absolute\n"
        "robust accuracies to undershoot the paper's by ~5-7 pp because we trim\n"
        "epochs 5x and inner-PGD steps 2x. The within-row comparison (baseline vs\n"
        "AMS at the **same** reduced scale) is what we use to assess the method.\n"
        f"{aa_footnote}\n"
    )

    out = PROJECT_ROOT / "results" / "tables" / "final_evaluation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    print("\n" + md)


if __name__ == "__main__":
    main()
