from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel

from .models.imagenet_models import ImagenetModel


def build_vit(name: str) -> nn.Module:
    """Build a Vision Transformer backbone from the given name."""
    model, config = name.split(":")
    model = model.lower()
    if model == "clip":
        vit = CLIPVisionModel.from_pretrained(config)
    elif "dino" in model:
        vit = AutoModel.from_pretrained(config)
    elif model == "imagenet":
        vit = ImagenetModel(config)
    else:
        raise ValueError(f"Unknown backbone model: {name}")
    return vit


class Detector(nn.Module):
    """MoLD detector that combines features from multiple ViT layers."""

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.vit = build_vit(backbone)
        self.vit.eval().requires_grad_(False)

        self.hidden_size: int = self.vit.config.hidden_size
        self.num_layers: int = self.vit.config.num_hidden_layers

        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(self.num_layers, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.vit(
            pixel_values=x,
            output_attentions=False,
            output_hidden_states=True,
            **kwargs,
        )

        features = []
        for hs in outputs.hidden_states[1:]:
            features.append(hs[:, 0, :])
        features = torch.stack(features, dim=0)  # num_layers, batch_size, hidden_size

        features = F.gelu(self.fc1(features))
        features = self.fc2(features).squeeze().permute(1, 0)
        outputs = self.fc3(features)
        return outputs
