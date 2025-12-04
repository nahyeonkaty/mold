from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32

model_dict: dict[str, Callable[..., nn.Module]] = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
}


CHANNELS: dict[str, int] = {
    "resnet50": 2048,
    "vit_b_16": 768,
}


class ImagenetModel(nn.Module):
    """ImageNet pretrained model wrapper."""

    def __init__(self, name: str, num_classes: int = 1) -> None:
        super().__init__()

        self.model = model_dict[name](pretrained=True)
        self.fc = nn.Linear(
            CHANNELS[name], num_classes
        )  # manually define a fc layer here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.model(x)["penultimate"]
        return self.fc(feature)
