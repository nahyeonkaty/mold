from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset


def get_bal_sampler(dataset: RealFakeDataset) -> WeightedRandomSampler:
    """Create a balanced sampler for the dataset."""
    targets: list[int] = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(
    opt: Namespace, preprocess: Any | None = None
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Create a dataloader for training or validation.

    Args:
        opt: Options namespace with dataset configuration.
        preprocess: Optional preprocessing transform.

    Returns:
        A DataLoader for the dataset.
    """
    shuffle = not opt.serial_batches if (opt.is_train and not opt.class_bal) else False
    dataset = RealFakeDataset(opt)
    if "2b" in opt.arch:
        dataset.transform = preprocess
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
    )
    return data_loader
