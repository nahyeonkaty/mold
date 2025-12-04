from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Model(nn.Module):
    """DINOv2 model wrapper for feature extraction."""

    def __init__(
        self, name: str, num_classes: int = 1, truncate_layer: int | None = None
    ) -> None:
        super().__init__()

        if "LARGE" in name:
            self.model = AutoModel.from_pretrained("facebook/dinov2-large")
            self.out_channels: int = 1024
            self.num_layers: int = 24
        elif "BASE" in name:
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.out_channels = 768
            self.num_layers = 12
        else:
            raise NotImplementedError(f"Unknown DINOv2 variant: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS features from all layers.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            CLS features from all layers, shape (num_layers, N, hidden_size).
        """
        N, C, H, W = x.shape
        x = self.model.embeddings(x)  # N L D

        cls_features: list[torch.Tensor] = []
        for i, layer_module in enumerate(self.model.encoder.layer):
            x = layer_module(x)[0]
            cls_features.append(x[:, 0, :])
        return torch.stack(cls_features, dim=0)  # num_layers, N, hidden_size
