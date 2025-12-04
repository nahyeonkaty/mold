from __future__ import annotations

import random
import sys
from typing import TextIO

import numpy as np
import torch


def unnormalize(
    tens: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Unnormalize a tensor of shape NxCxHxW."""
    return (
        tens * torch.Tensor(std)[None, :, None, None]
        + torch.Tensor(mean)[None, :, None, None]
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across different components.

    Args:
        seed: The desired seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    # Note: Some operations may still be non-deterministic,
    # so for complete reproducibility, you might also need to
    # set torch.backends.cudnn.benchmark = False
    # and torch.backends.cudnn.deterministic = True
    # but this can impact performance.


class Logger:
    """Log stdout messages."""

    def __init__(self, outfile: str) -> None:
        self.terminal: TextIO = sys.stdout
        self.log: TextIO = open(outfile, "a")
        sys.stdout = self

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
