"""Utility functions for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Ensures deterministic behavior across runs as described in the paper
    (Section 4.1): "all random operations in Python, NumPy, and PyTorch
    are controlled by a fixed seed."

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
