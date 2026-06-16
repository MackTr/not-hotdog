"""Small shared helpers used across the notebooks.

Kept deliberately tiny — everything here is something you'll understand by Phase 3.
"""

import torch


def get_device() -> torch.device:
    """Pick the best available device.

    On your Mac, that's "mps" (the Apple GPU). On a machine with an NVIDIA card
    it'd be "cuda". Otherwise we fall back to the CPU, which still works fine for
    a dataset this small.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
