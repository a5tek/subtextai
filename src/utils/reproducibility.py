"""
Reproducibility Utilities
Ensures deterministic behavior across Python random, NumPy, PyTorch CPU/CUDA, and CUDNN.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets deterministic random seed across random, numpy, os, and torch (if available).
    
    Args:
        seed: Integer seed value.
        deterministic: If True, sets cudnn flags for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
