"""
utils/seed.py
-------------
Sets all random seeds for reproducibility across random, numpy, and torch.
Call set_seed() at the start of every training run.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for random, numpy, and torch for reproducible results.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # MPS does not support deterministic mode — skip on Apple Silicon
    if not torch.backends.mps.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False