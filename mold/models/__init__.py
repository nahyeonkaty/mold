from __future__ import annotations

import torch.nn as nn

from .dino_models import DINOv2Model
from .imagenet_models import ImagenetModel

VALID_NAMES: list[str] = [
    "Imagenet:vgg11",
    "Imagenet:vgg19",
    "Imagenet:swin-b",
    "Imagenet:swin-s",
    "Imagenet:swin-t",
    "Imagenet:vit_b_16",
    "Imagenet:vit_b_32",
    "Imagenet:vit_l_16",
    "Imagenet:vit_l_32",
    "CLIP:RN50",
    "CLIP:RN101",
    "CLIP:RN50x4",
    "CLIP:RN50x16",
    "CLIP:RN50x64",
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",  # paper
    "CLIP:ViT-L/14@336px",
    "DINOv2:LARGE",
    "DINOv2:BASE",
]


def get_model(name: str, truncate_layer: int | None = None) -> nn.Module:
    """Get a model by name.

    Args:
        name: Model name in format 'Backend:model_name'.
        truncate_layer: Layer index to truncate at (required for some models).

    Returns:
        The initialized model.
    """
    assert name in VALID_NAMES
    assert truncate_layer is not None
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:])
    elif name.startswith("DINOv2"):
        return DINOv2Model(name, truncate_layer=truncate_layer)
    else:
        raise ValueError(f"Unknown model: {name}")
