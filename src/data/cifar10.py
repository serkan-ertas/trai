"""CIFAR-10 data loaders for adversarial training.

Builds three loaders: train (~49000 samples, augmented), val (1000 samples
carved deterministically off the end of the train set, no augmentation), and
test (10000 samples, no augmentation).

The split uses a fixed RNG seeded with the `seed` argument so the same val
indices are produced on every run. To keep train and val transforms
independent, the underlying CIFAR-10 train set is instantiated twice (once
with train augmentation, once with eval-only transforms) and `Subset` is used
to pick disjoint index sets.

NO per-channel normalization is applied here. The model
(`src/models/preact_resnet.py`) embeds normalization as its first layer so
adversarial attacks operate on raw [0,1] pixel tensors and the ε-ball stays
in pixel space (see the project spec gotcha #10).
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    val_size: int = 1000,
    num_workers: int = 2,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.ToTensor()

    cifar_train_full = CIFAR10(
        root=data_root, train=True, transform=train_transform, download=download
    )
    cifar_val_full = CIFAR10(
        root=data_root, train=True, transform=eval_transform, download=download
    )
    cifar_test = CIFAR10(
        root=data_root, train=False, transform=eval_transform, download=download
    )

    n_total = len(cifar_train_full)
    n_train = n_total - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(n_total), [n_train, val_size], generator=gen
    )
    train_idx = list(train_subset)
    val_idx = list(val_subset)

    train_set = Subset(cifar_train_full, train_idx)
    val_set = Subset(cifar_val_full, val_idx)

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        cifar_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader
