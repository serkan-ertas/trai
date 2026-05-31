# AMS Adversarial Training (Reduced-Scale Reproduction)

A course-project reproduction of Wang & Ding, *"Exploring the Forgetting in Adversarial Training: A Novel Method for Enhancing Robustness"* (ICLR 2025). The method, **Adaptive Multi-teacher Self-distillation (AMS)**, snapshots the model at fixed intervals during adversarial training and distills from those past snapshots to combat the forgetting of robust features. This repository runs a reduced-scale version (40 epochs, CIFAR-10, PreActResNet-18, ℓ∞ ε=8/255) on a laptop GPU and is **not** trying to match the paper's absolute numbers - the goal is to show the directional improvement of AMS over a TRADES baseline.

## Data & artifacts (Google Drive)

This repository contains **only the code, configs, and scripts** needed to
reproduce the project. The large/post-run artifacts - the dataset, trained model
checkpoints, generated results, plots, and the final report - live on Google
Drive:

**📁 Google Drive:** `https://drive.google.com/drive/folders/13GZHKE4oaYH3RJSr4o0ZmptHr2pX3oD8?usp=sharing`

| Drive folder | Contents | Role |
|---|---|---|
| `data/` | CIFAR-10 dataset | *Optional* - auto-downloads on the first training run; included so evaluation can run on the checkpoints without retraining. |
| `runs/` | TRADES baseline and TRADES+AMS checkpoints, `stage_*_correct_indices.pt`, `metrics.jsonl`, configs, logs | Source of all reported numbers and the forgetting analysis. |
| `results/` | Generated tables, figures, intermediate eval JSONs | Tables/plots are reproduced by the commands in [Reproducing the results](#reproducing-the-results) |

Two ways to verify the results:

1. **Reproduce from scratch** - follow [Reproducing the results](#reproducing-the-results) below using only this repo (no Drive download needed; the dataset auto-downloads).
2. **Verify against the provided run** - download `data/` and `runs/` from Drive into the repo root, then run only the evaluation and table/plot commands (steps 3–4) to regenerate `results/` from the trained checkpoints.

## Hardware requirements

- Target: NVIDIA RTX 3050 Mobile class, ~4 GB VRAM (the original development machine).
- Will also run on any larger CUDA GPU with no changes; you can raise the batch size if you have headroom.
- CPU-only is **not supported**. Adversarial training requires many forward/backward passes per step and would take days without a GPU.

## Installation

1. Create and activate a Python 3.10+ virtual environment.

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows PowerShell
   ```

2. Install dependencies, **including the CUDA wheel index**.

   ```
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```


3. Sanity-check that PyTorch sees the GPU.

   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   Expected output: `True`. If it prints `False`, your CUDA driver, your PyTorch wheel, or both are wrong - fix this before continuing.

## Repository layout

```
trai/
├── README.md            # this file
├── requirements.txt
├── src/                 # source code (models, attacks, losses, training, experiments)
├── configs/             # YAML configs for runs
├── tests/               # pytest sanity tests
├── runs/                # one subdirectory per training run
└── results/             # final tables, plots, writeup
```

`src/`, `configs/`, `tests/`, `runs/`, and `results/` are populated by later phases of the project and may not exist on a fresh clone.

## Reproducing the results

Run the following commands in order from the repository root, with the virtual
environment activated. This is the exact sequence used to produce the tables and
plots in `results/`. All training hyperparameters live in the YAML configs; the
scripts take only the arguments shown.

**1. (Optional) Smoke tests** - fast 1–2 epoch runs that verify the full
pipeline works before committing to the multi-hour training runs.

```
python -m src.experiments.train_trades --config configs/trades_baseline_smoke.yaml
python -m src.experiments.train_ams    --config configs/trades_ams_smoke.yaml
```

**2. Train both models** (40 epochs each; CIFAR-10 is downloaded automatically
on the first run). On an RTX 3050 Mobile this is the bulk of the wall-clock time.

```
python -m src.experiments.train_trades --config configs/trades_baseline.yaml
python -m src.experiments.train_ams    --config configs/trades_ams.yaml
```

**3. Evaluate** each best checkpoint against Clean / PGD-20 / CW-20 / AutoAttack.
`--aa_subset 1000` runs AutoAttack on the first 1000 test samples (~26 min each);
use `--aa_subset 0` for the full 10k test set (~4.5 h each).

```
python results/scripts/eval_final.py --run trades_baseline --aa_subset 1000
python results/scripts/eval_final.py --run trades_ams      --aa_subset 1000
```

**4. Build the tables and plots** from the trained runs and eval JSONs. These
scripts take no arguments - they read `runs/` and `results/scripts/tmp/` directly.

```
python results/scripts/build_eval_table.py          # -> results/tables/final_evaluation.md
python results/scripts/compute_faa_ff.py            # -> forgetting table + stage plot
python results/scripts/plot_robust_overfitting.py   # -> robust-overfitting plot + best-vs-final table
```

Approximate wall-clock on the reference RTX 3050 Mobile: ~1.5 h per training run,
~30 min per evaluation (with `--aa_subset 1000`), and seconds for each table/plot
script.

> **Note - `results/scripts/aa_probe.py` is optional.** It is a one-off AutoAttack
> timing probe (runs AutoAttack on 100 samples and prints an extrapolated
> full-test-set wall-time). It is not part of the reproduction pipeline, writes no
> files, and nothing depends on it - it was only used to decide the `--aa_subset`
> value. You can ignore it.

## Outputs

Each training run writes to its own directory under `runs/<run_id>/`:

- `config.yaml` - frozen copy of the config used.
- `metrics.jsonl` - one JSON line per epoch with `epoch`, `lr`, `train_loss_ce`, `train_loss_kl`, `train_acc_clean`, `val_acc_clean`, `val_robust_acc`, `train_time_sec`.
- `checkpoint_best.pt` - model state with the highest PGD-20 validation robust accuracy seen so far.
- `checkpoint_final.pt` - model state at the last epoch.

AMS runs additionally drop `stage_<s>_correct_indices.pt` files used for the forgetting verification experiment.

Final tables, plots, and the writeup land under `results/`.

## Reproduction stance

This is an undergraduate course project running on a 4 GB laptop GPU, with a total training budget of 3–4 hours wall-clock across all runs. We therefore deviate from the paper as follows:

| Setting | Paper | Here |
|---|---|---|
| Epochs | 200 | 40 |
| Inner PGD steps (training) | 10 | 5 |
| Stage interval m | 20 | 10 |
| Batch size | 128 | 128 (falls back to 64 if VRAM is tight) |
| Precision | FP32 | AMP (mixed precision) |
| ε (ℓ∞) | 8/255 | 8/255 |
| λ (AMS regularizer) | 0.5 | 0.5 |

Absolute robust accuracies will be lower than the numbers reported in the paper. The success criterion for this reproduction is the *directional* result: AMS should outperform vanilla TRADES under the same reduced-scale conditions.


