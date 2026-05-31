# AMS Adversarial Training (Reduced-Scale Reproduction)

A course-project reproduction of Wang & Ding, *"Exploring the Forgetting in Adversarial Training: A Novel Method for Enhancing Robustness"* (ICLR 2025). The method, **Adaptive Multi-teacher Self-distillation (AMS)**, snapshots the model at fixed intervals during adversarial training and distills from those past snapshots to combat the forgetting of robust features. This repository runs a reduced-scale version (40 epochs, CIFAR-10, PreActResNet-18, ℓ∞ ε=8/255) on a laptop GPU and is **not** trying to match the paper's absolute numbers - the goal is to show the directional improvement of AMS over a TRADES baseline.

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

## How to run


- Train the TRADES baseline:

  ```
  python -m src.experiments.train_trades --config configs/trades_baseline.yaml
  ```

- Train TRADES + AMS (Algorithm 1 from the paper):

  ```
  python -m src.experiments.train_ams --config configs/trades_ams.yaml
  ```

- Evaluate a completed run against PGD-20, CW-20, and AutoAttack:

  ```
  python -m src.experiments.evaluate --run runs/<run_id>
  ```

Each script accepts only `--config` and run-identifier arguments; all hyperparameters live in the YAML.

## Outputs

Each training run writes to its own directory under `runs/<run_id>/`:

- `config.yaml` - frozen copy of the config used.
- `metrics.jsonl` - one JSON line per epoch with `train_loss`, `train_robust_acc`, `val_clean_acc`, `val_pgd20_acc`, `lr`, `wall_time_s`.
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


