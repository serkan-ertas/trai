"""Sanity probe: inspect a checkpoint dict layout and a stage_*_correct_indices.pt file."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    cb = torch.load(
        "runs/trades_ams/checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    cf = torch.load(
        "runs/trades_ams/checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    print("=== checkpoint_best.pt top-level keys ===")
    print(sorted(cb.keys()))
    print(f"best epoch: {cb.get('epoch')}  best_metric: {cb.get('best_metric')}")
    print()
    print("=== checkpoint_final.pt top-level keys ===")
    print(sorted(cf.keys()))
    if "teacher_buffer" in cf:
        tb = cf["teacher_buffer"]
        print(f"teacher_buffer top keys: {sorted(tb.keys()) if isinstance(tb, dict) else type(tb)}")
        if isinstance(tb, dict) and "snapshots" in tb:
            snaps = tb["snapshots"]
            print(f"num snapshots: {len(snaps)}")
            print(f"snapshot 0 keys count: {len(snaps[0])}; sample key: {list(snaps[0].keys())[0]}")
    print()
    for s in (1, 2, 3, 4):
        d = torch.load(
            f"runs/trades_ams/stage_{s}_correct_indices.pt",
            map_location="cpu",
            weights_only=False,
        )
        print(f"=== stage {s} ===")
        print(f"  keys: {sorted(d.keys())}")
        print(f"  x shape: {d['x'].shape}  dtype: {d['x'].dtype}  min/max: {d['x'].min().item():.4f}/{d['x'].max().item():.4f}")
        print(f"  x_adv shape: {d['x_adv'].shape}  min/max: {d['x_adv'].min().item():.4f}/{d['x_adv'].max().item():.4f}")
        print(f"  y shape: {d['y'].shape}  dtype: {d['y'].dtype}")
        print(f"  correct_mask shape: {d['correct_mask'].shape}  dtype: {d['correct_mask'].dtype}  sum={d['correct_mask'].sum().item()}/{d['correct_mask'].numel()}")
        print(f"  num_samples: {d['num_samples']}")
    print()
    # Inspect baseline checkpoint
    cbase = torch.load(
        "runs/trades_baseline/checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    print("=== baseline checkpoint_best.pt top-level keys ===")
    print(sorted(cbase.keys()))
    print(f"best epoch: {cbase.get('epoch')}  best_metric: {cbase.get('best_metric')}")


if __name__ == "__main__":
    main()
