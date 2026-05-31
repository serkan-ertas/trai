"""Quick AutoAttack timing probe on 100 samples to estimate full-test-set wall time."""
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import evaluate_autoattack
from src.data import get_cifar10_loaders
from src.models import preact_resnet18

device = torch.device("cuda")
model = preact_resnet18(num_classes=10).to(device)
state = torch.load(
    PROJECT_ROOT / "runs" / "trades_baseline" / "checkpoint_best.pt",
    map_location=device, weights_only=False,
)
model.load_state_dict(state["model"])
model.eval()

_, _, test_loader = get_cifar10_loaders(
    data_root=str(PROJECT_ROOT / "data"),
    batch_size=128,
    val_size=1000,
    num_workers=0,
    seed=0,
    download=False,
)
n = 100
subset = Subset(test_loader.dataset, list(range(n)))
loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
t0 = time.time()
acc = evaluate_autoattack(model, loader, eps=8/255, version="standard",
                          norm="Linf", device="cuda", verbose=True, seed=0)
dt = time.time() - t0
print(f"\nProbe: AA on {n} samples = {acc:.4f}  ({dt:.1f}s)")
print(f"Extrapolated full test set (10000): {dt * 10000 / n / 60:.1f} min")
print(f"Extrapolated 1000-sample subset: {dt * 1000 / n / 60:.1f} min")
