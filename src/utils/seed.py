import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility.

    Args:
        seed: Integer seed used for all RNGs.
        deterministic: If True, also force deterministic cuDNN and CUBLAS at a
            speed cost. Use False (default) for training, True for eval or for
            debugging numerical issues. Note: with AMP some ops fall back to
            non-deterministic kernels and will emit a warning rather than raise.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS workspace required by torch.use_deterministic_algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
