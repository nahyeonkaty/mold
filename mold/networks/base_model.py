from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base model class with common functionality for training."""

    model: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, opt: Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.total_steps: int = 0
        self.save_dir: str = os.path.join(opt.checkpoints_dir, opt.expname)
        os.makedirs(self.save_dir, exist_ok=True)
        self.device: torch.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )

    def save_networks(self, save_filename: str) -> None:
        """Save model and optimizer state to a file."""
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def test(self) -> None:
        """Run forward pass without gradient computation."""
        with torch.no_grad():
            self.forward()

    def forward(self) -> None:
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """Initialize network weights.

    Args:
        net: The network to initialize.
        init_type: Type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal').
        gain: Scaling factor for the initialization.
    """

    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.zeros_(m.bias.data)

    print("initialize network with %s" % init_type)
    net.apply(init_func)
